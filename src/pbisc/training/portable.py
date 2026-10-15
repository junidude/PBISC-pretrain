"""Relocate sealed training profiles without changing their numeric contracts."""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

PROFILE_SCHEMA = "pbisc.release_training_profile.v1"
MATRIX_PROFILE_SCHEMA = "pbisc.pretrain.profile.v2"
_PATH_KEYS = frozenset({
    "paths.output_dir", "paths.conditioner_cache", "paths.model7_root", "data.gene_positions_path",
    "data.gene_selection_manifest", "data.coverage_audit_summary",
    "data.topup_manifest", "data.gene_selection_implementation_path",
})


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _within(root: Path, relative: str) -> Path:
    if Path(relative).is_absolute():
        raise ValueError("release asset must use a relative path")
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError("release asset escapes dataset root")
    return path


def resolve_artifact(original: str, config: Mapping[str, Any]) -> Path:
    """Resolve a preserved source-manifest path via the explicit release map."""
    release = config.get("_release", {})
    profile_mapping = release.get("profile_asset_paths", {})
    if original in profile_mapping:
        return _within(Path(release["profile_asset_root"]), str(profile_mapping[original]))
    if release.get("profile_asset_root") and Path(original).is_absolute():
        candidate = Path(original).expanduser().resolve()
        if candidate.is_relative_to(Path(release["profile_asset_root"]).resolve()):
            return candidate
    mapping = release.get("asset_paths", {})
    if original in mapping:
        return _within(Path(release["dataset_root"]), str(mapping[original]))
    if release.get("dataset_root"):
        candidate = Path(original).expanduser().resolve()
        if candidate.is_relative_to(Path(release["dataset_root"]).resolve()):
            return candidate
        raise ValueError(f"dataset relocation map does not bind artifact: {original}")
    return Path(original).expanduser().resolve()


def _profile_file(root: Path, value: object, *, relative_to: Path | None = None) -> Path:
    """Resolve a profile reference without permitting an escape from its assets."""
    reference = Path(str(value))
    if reference.is_absolute():
        raise ValueError("profile assets must use a relative path")
    candidate = ((relative_to or root) / reference).resolve()
    if not candidate.is_relative_to(root):
        raise ValueError("profile asset escapes trusted asset root")
    return candidate


def _matrix_config(path: Path, profile: Mapping[str, Any]) -> tuple[dict[str, Any], Path, dict[str, Any]]:
    """Bind repository-owned panel files independently of restored corpus files."""
    from .panel_contracts import COLLECTION_PANEL_MODE, COLLECTION_PANEL_SCHEMA

    root_reference = Path(str(profile.get("asset_root", "../..")))
    if root_reference.is_absolute():
        raise ValueError("profile asset_root must use a relative path")
    asset_root = (path.parent / root_reference).resolve()
    # Profiles live inside the asset tree in both a checkout and wheel share/.
    if not path.is_relative_to(asset_root) or not (asset_root / "configs").is_dir():
        raise ValueError("profile asset_root must contain the profile and configs")
    base = _profile_file(asset_root, profile["base_config"], relative_to=path.parent)
    if _sha(base) != profile["base_config_sha256"]:
        raise ValueError("base training config hash differs from profile")
    config = json.loads(base.read_text())
    if config.get("_release"):
        raise ValueError("new profile base config must not contain release provenance")
    for section in ("paths", "e2_data"):
        for key, value in config.get(section, {}).items():
            if isinstance(value, str) and Path(value).is_absolute():
                raise ValueError(f"new profile base {section}.{key} must be relative")
    for key in _PATH_KEYS:
        section, field = key.split(".", 1)
        value = config.get(section, {}).get(field)
        if isinstance(value, str) and Path(value).is_absolute():
            raise ValueError(f"new profile base {key} must be relative")
    overrides = profile.get("overrides", {})
    if set(overrides) - {"experiment_id", "data", "model"}:
        raise ValueError("unsupported matrix profile override")
    if "experiment_id" in overrides:
        config["experiment_id"] = overrides["experiment_id"]
    for section, allowed in (("data", {"n_genes"}), ("model", {"hidden_dims", "context_dim"})):
        values = overrides.get(section, {})
        if not isinstance(values, dict) or set(values) - allowed:
            raise ValueError(f"unsupported matrix profile {section} override")
        config[section].update(values)
    manifest_path = _profile_file(asset_root, profile["panel_manifest"])
    if _sha(manifest_path) != profile["panel_manifest_sha256"]:
        raise ValueError("collection panel manifest hash differs from profile")
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("schema") != COLLECTION_PANEL_SCHEMA:
        raise ValueError("unsupported collection panel schema")
    panel_relative = Path(str(manifest["panel"]))
    if panel_relative.is_absolute():
        raise ValueError("collection panel must use a relative path")
    # Tier manifests explicitly name their collection-relative panel directory.
    collection_root = manifest_path
    for _ in range(len(panel_relative.parts) + 1):
        collection_root = collection_root.parent
    if (_profile_file(collection_root, panel_relative / "selection_manifest.json") != manifest_path
            or not collection_root.is_relative_to(asset_root)):
        raise ValueError("collection panel manifest escapes trusted asset root")
    raw_contract = collection_root / "inputs/raw_axis_contract.json"
    if _sha(raw_contract) != profile["raw_axis_contract_sha256"]:
        raise ValueError("collection raw-axis contract hash differs from profile")
    axis = json.loads(raw_contract.read_text())
    if axis.get("schema") != "pbisc.gene_context_collection.raw_axis.v1":
        raise ValueError("unsupported collection raw-axis contract")
    if int(config["data"]["n_genes"]) != int(manifest["n_genes"]):
        raise ValueError("profile gene count differs from collection panel")
    if manifest.get("corpus_fingerprint") != axis.get("corpus_fingerprint"):
        raise ValueError("collection panel and raw-axis corpus fingerprints differ")
    asset_paths: dict[str, str] = {}
    records = [*manifest["files"], *manifest["input_bindings"], axis["ranked_gene_positions"],
               {"path": str(panel_relative / "selection_manifest.json")},
               {"path": "inputs/raw_axis_contract.json"}, {"path": "build_collection.py"},
               {"path": "MANIFEST.json"}]
    # Register every collection file so cross-tier retention never falls back to
    # supplemental remote copies. Hash validation is done by the strict loader.
    collection_index_path = collection_root / "MANIFEST.json"
    collection_index_hash = profile.get("collection_manifest_sha256")
    if (not isinstance(collection_index_hash, str) or len(collection_index_hash) != 64
            or _sha(collection_index_path) != collection_index_hash):
        raise ValueError("collection index hash differs from profile")
    collection_index = json.loads(collection_index_path.read_text())
    records.extend(collection_index["files"])
    for record in records:
        relative = str(record["path"])
        artifact = _profile_file(collection_root, relative)
        asset_paths[relative] = str(artifact.relative_to(asset_root))
    positions_relative = str(panel_relative / "gene_positions.npy")
    files = {item["path"]: item for item in manifest["files"]}
    config["data"].update(
        gene_selection=COLLECTION_PANEL_MODE,
        gene_positions_path=positions_relative,
        gene_positions_sha256=files[positions_relative]["sha256"],
        gene_selection_manifest=str(panel_relative / "selection_manifest.json"),
        gene_selection_manifest_sha256=profile["panel_manifest_sha256"],
        gene_context_collection_contract={
            "corpus_fingerprint": axis["corpus_fingerprint"],
            "canonical_panel_sha256": axis["canonical_panel_sha256"],
            "raw_var_sha256": axis["raw_var_sha256"],
            "raw_axis_contract_sha256": profile["raw_axis_contract_sha256"],
            "collection_manifest_sha256": collection_index_hash,
        },
    )
    config["data"].pop("expanded_panel_contract", None)
    config["data"].pop("gene_selection_implementation_path", None)
    return config, base, {
        "profile_schema": MATRIX_PROFILE_SCHEMA,
        "profile_asset_root": str(asset_root), "profile_asset_paths": asset_paths,
    }


def canonical_resume_config(config: Mapping[str, Any]) -> dict[str, Any]:
    """Restore path-only provenance for the original trainer replay checks.

    All numeric/data semantics remain the currently requested values. The
    adapter records only path substitutions it actually performed; optimizer,
    recipe, seed, masks, worker count and model checks remain unchanged.
    """
    result = copy.deepcopy(dict(config))
    release = result.pop("_release", {})
    for dotted, original in release.get("original_path_values", {}).items():
        if dotted not in _PATH_KEYS:
            raise ValueError(f"release provenance may restore only path fields: {dotted}")
        section, key = dotted.split(".", 1)
        if original is None:
            result[section].pop(key, None)
        else:
            result[section][key] = original
    return result


def prepare_config(
    config_path: Path | str,
    *,
    dataset_root: Path | str | None = None,
    output_dir: Path | str | None = None,
    conditioner_cache: Path | str | None = None,
) -> tuple[dict[str, Any], str]:
    """Bind an original run config or release profile to a restored dataset."""
    path = Path(config_path).expanduser().resolve()
    profile = json.loads(path.read_text())
    trainer = str(profile.get("trainer", "stable"))
    matrix_release: dict[str, Any] = {}
    is_matrix = profile.get("schema_version") == MATRIX_PROFILE_SCHEMA
    if is_matrix:
        if output_dir is None or dataset_root is None:
            raise ValueError("new training profiles require explicit output_dir and dataset_root")
        config, original, matrix_release = _matrix_config(path, profile)
    elif profile.get("schema_version") == PROFILE_SCHEMA:
        if output_dir is None:
            raise ValueError("release training profiles require an explicit output_dir; original run paths are provenance only")
        if dataset_root is None:
            raise ValueError("release training profiles require an explicit dataset_root")
        original = (path.parent / profile["original_config"]).resolve()
        if _sha(original) != profile["original_config_sha256"]:
            raise ValueError("original training config hash differs from release profile")
        config = json.loads(original.read_text())
    else:
        original = path
        config = copy.deepcopy(profile)
        if "topup_manifest" in config.get("data", {}):
            trainer = "allcell"
        elif "gradient_accumulation_steps" in config.get("training", {}):
            trainer = "ladder"
    previous_release = config.get("_release", {})
    if not set(previous_release.get("original_path_values", {})).issubset(_PATH_KEYS):
        raise ValueError("release provenance may restore only path fields")
    release: dict[str, Any] = {
        **previous_release,
        **matrix_release,
        "schema_version": "pbisc.training_relocation.v1",
        "original_config_sha256": previous_release.get("original_config_sha256", _sha(original)),
        "original_path_values": dict(previous_release.get("original_path_values", {})),
    }
    config["_release"] = release

    def substitute(section: str, key: str, value: str) -> None:
        release["original_path_values"].setdefault(f"{section}.{key}", config[section].get(key))
        config[section][key] = value

    if dataset_root is not None:
        root = Path(dataset_root).expanduser().resolve()
        relocation_path = root / "relocation.json"
        relocation = json.loads(relocation_path.read_text())
        if relocation.get("schema_version") != "pbisc.dataset_relocation.v1":
            raise ValueError("unsupported dataset relocation schema")
        mapping = dict(relocation.get("asset_paths", {}))
        extension_path = root / "relocation.pretrain.json"
        if not is_matrix and extension_path.is_file():
            extension = json.loads(extension_path.read_text())
            if extension.get("schema_version") != "pbisc.pretrain_dataset_relocation.v1":
                raise ValueError("unsupported pretrain dataset relocation schema")
            if extension.get("original_relocation_sha256") != _sha(relocation_path):
                raise ValueError("pretrain relocation does not bind the original relocation")
            for key, value in relocation.get("asset_paths", {}).items():
                if extension.get("asset_paths", {}).get(key) != value:
                    raise ValueError("pretrain relocation changed an original artifact mapping")
            mapping.update(extension.get("asset_paths", {}))
            release["pretrain_relocation_sha256"] = _sha(extension_path)
        if not is_matrix:
            mapping.update(profile.get("asset_paths", {}))
        release.update(dataset_root=str(root), asset_paths=mapping,
                       relocation_sha256=_sha(relocation_path))
        relative_roots = relocation["relative_roots"]
        config["e2_data"] = {**config.get("e2_data", {}), **{
            key: str(_within(root, str(relative_roots[key])))
            for key in ("source_root", "base_source_root", "index_root", "gene_panel_path")
        }}
        config["e2_data"]["relocation_manifest"] = str(relocation_path)
        if is_matrix and "model7_root" in config["paths"]:
            substitute("paths", "model7_root", str(_within(root, str(config["paths"]["model7_root"]))))
        for key in ("gene_positions_path", "gene_selection_manifest", "coverage_audit_summary", "topup_manifest"):
            if key in config["data"]:
                value = str(config["data"][key])
                resolved = (_within(root, value) if is_matrix and key in {"coverage_audit_summary", "topup_manifest"}
                            else resolve_artifact(value, config))
                substitute("data", key, str(resolved))
        if conditioner_cache is None:
            cache = str(config["paths"]["conditioner_cache"])
            substitute("paths", "conditioner_cache", str(_within(root, cache) if is_matrix
                                                          else resolve_artifact(cache, config)))
        selection = config["data"].get("gene_selection_manifest")
        if selection:
            manifest = json.loads(Path(selection).read_text())
            if manifest.get("schema_version") == "pbisc_sdd.hvg_selection.v1":
                implementation = resolve_artifact(manifest["implementation"], config)
                if _sha(implementation) != manifest["implementation_sha256"]:
                    raise ValueError("preserved gene selection implementation hash differs")
                substitute("data", "gene_selection_implementation_path", str(implementation))
            elif manifest.get("schema") not in {"pbisc.rare_gene_expansion.candidate.v1", "pbisc.gene_context_collection.tier.v1"}:
                raise ValueError("unsupported gene-selection manifest schema")
    if output_dir is not None:
        substitute("paths", "output_dir", str(Path(output_dir).expanduser().resolve()))
    if conditioner_cache is not None:
        substitute("paths", "conditioner_cache", str(Path(conditioner_cache).expanduser().resolve()))
    if dataset_root is None and "e2_data" not in config:
        model7_root = Path(config["paths"].get("model7_root", "")).expanduser().resolve()
        if not str(config["paths"].get("model7_root", "")):
            raise ValueError("provide --dataset-root or explicit e2_data roots")
        source = model7_root / "data/e2_full_source"
        manifest = json.loads((source / "dataset_manifest.json").read_text())
        config["e2_data"] = {
            "source_root": str(source), "base_source_root": manifest["base_source_root"],
            "index_root": str(model7_root / "data/e2_full_index"),
            "gene_panel_path": manifest["gene_panel"]["path"],
        }
    return config, trainer
