"""Relocate sealed training profiles without changing their numeric contracts."""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

PROFILE_SCHEMA = "pbisc.release_training_profile.v1"
_PATH_KEYS = frozenset({
    "paths.output_dir", "paths.conditioner_cache", "data.gene_positions_path",
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
    mapping = release.get("asset_paths", {})
    if original in mapping:
        return _within(Path(release["dataset_root"]), str(mapping[original]))
    if release.get("dataset_root"):
        candidate = Path(original).expanduser().resolve()
        if candidate.is_relative_to(Path(release["dataset_root"]).resolve()):
            return candidate
        raise ValueError(f"dataset relocation map does not bind artifact: {original}")
    return Path(original).expanduser().resolve()


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
    if profile.get("schema_version") == PROFILE_SCHEMA:
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
        if extension_path.is_file():
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
        mapping.update(profile.get("asset_paths", {}))
        release.update(dataset_root=str(root), asset_paths=mapping,
                       relocation_sha256=_sha(relocation_path))
        relative_roots = relocation["relative_roots"]
        config["e2_data"] = {**config.get("e2_data", {}), **{
            key: str(_within(root, str(relative_roots[key])))
            for key in ("source_root", "base_source_root", "index_root", "gene_panel_path")
        }}
        config["e2_data"]["relocation_manifest"] = str(relocation_path)
        for key in ("gene_positions_path", "gene_selection_manifest", "coverage_audit_summary", "topup_manifest"):
            if key in config["data"]:
                substitute("data", key, str(resolve_artifact(config["data"][key], config)))
        if conditioner_cache is None:
            substitute("paths", "conditioner_cache", str(resolve_artifact(config["paths"]["conditioner_cache"], config)))
        selection = config["data"].get("gene_selection_manifest")
        if selection:
            manifest = json.loads(Path(selection).read_text())
            if manifest.get("schema_version") == "pbisc_sdd.hvg_selection.v1":
                implementation = resolve_artifact(manifest["implementation"], config)
                if _sha(implementation) != manifest["implementation_sha256"]:
                    raise ValueError("preserved gene selection implementation hash differs")
                substitute("data", "gene_selection_implementation_path", str(implementation))
            elif manifest.get("schema") != "pbisc.rare_gene_expansion.candidate.v1":
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
