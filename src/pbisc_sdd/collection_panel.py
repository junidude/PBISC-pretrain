"""Strict runtime binding for the unchanged gene-context collection schema.

The collection's provenance manifests stay byte exact. This loader explicitly
accepts their separate schema; they are never passed off as legacy HVG panels.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from pbisc.training.panel_contracts import COLLECTION_PANEL_MODE as MODE
from pbisc.training.panel_contracts import COLLECTION_PANEL_SCHEMA as SCHEMA


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _check(path: Path, expected: object, label: str, size: object = None) -> Path:
    if (not isinstance(expected, str) or len(expected) != 64 or not path.is_file()
            or _sha(path) != expected or (size is not None and path.stat().st_size != int(size))):
        raise ValueError(f"collection panel {label} SHA-256 or size differs")
    return path


def validate_collection_panel(
    config: Mapping[str, Any], *, gene_panel: Any = None,
    corpus_fingerprint: str | None = None,
) -> None:
    """Verify raw coordinates, full source axis, frozen copies and recipe replay."""
    from pbisc.training.portable import resolve_artifact

    data = config["data"]
    if data.get("gene_selection") != MODE:
        raise ValueError("collection panel requires its explicit selection mode")
    contract = data.get("gene_context_collection_contract", {})
    manifest_path = _check(Path(data["gene_selection_manifest"]),
                           data["gene_selection_manifest_sha256"], "manifest")
    manifest = json.loads(manifest_path.read_text())
    n = int(data["n_genes"])
    if (manifest.get("schema") != SCHEMA or manifest.get("n_genes") != n
            or n not in {7000, 8000, 10000} or manifest.get("panel") != f"panels/{n // 1000}k"):
        raise ValueError("unsupported collection panel schema or selected-gene count")
    if (manifest.get("endpoint_informed") is not True or not manifest.get("gene_specificity_limit")
            or manifest.get("trained_checkpoint_included") is not False):
        raise ValueError("collection panel lacks preserved source-exposure and weight caveats")

    def artifact(relative: str) -> Path:
        # resolve_artifact requires an explicit repository map, independent of
        # the corpus map; neither a working-directory lookup nor old paths work.
        if relative not in config.get("_release", {}).get("profile_asset_paths", {}):
            raise ValueError(f"collection asset is not bound by its profile: {relative}")
        return resolve_artifact(relative, config)

    raw_contract_path = _check(artifact("inputs/raw_axis_contract.json"),
                               contract.get("raw_axis_contract_sha256"), "raw-axis contract")
    axis = json.loads(raw_contract_path.read_text())
    if axis.get("schema") != "pbisc.gene_context_collection.raw_axis.v1" or axis.get("n_genes") != 20097:
        raise ValueError("unsupported collection raw-axis contract")
    for key in ("corpus_fingerprint", "raw_var_sha256", "canonical_panel_sha256"):
        if contract.get(key) != axis.get(key) or not axis.get(key):
            raise ValueError(f"collection panel {key} contract differs")
    if manifest.get("corpus_fingerprint") != axis["corpus_fingerprint"]:
        raise ValueError("collection panel corpus fingerprint differs")
    if corpus_fingerprint is not None and corpus_fingerprint != axis["corpus_fingerprint"]:
        raise ValueError("collection panel runtime corpus fingerprint differs")
    full_axis = _check(Path(config["e2_data"]["gene_panel_path"]), axis["raw_var_sha256"], "full raw gene axis")
    canonical = pd.read_parquet(full_axis)
    if len(canonical) != 20097 or canonical["gene_id"].isna().any() or not canonical["gene_id"].is_unique:
        raise ValueError("collection panel requires the unique full 20,097-gene canonical axis")

    records: dict[str, Mapping[str, Any]] = {}
    index_path = _check(artifact("MANIFEST.json"), contract.get("collection_manifest_sha256"), "collection index")
    index = json.loads(index_path.read_text())
    if index.get("schema") != "pbisc.gene_context_collection.v1":
        raise ValueError("unsupported gene-context collection index")
    for record in index["files"]:
        relative = str(record["path"])
        if relative in records:
            raise ValueError("collection index has duplicate artifact paths")
        records[relative] = record
    for record in [*manifest["files"], *manifest["input_bindings"]]:
        relative = str(record["path"])
        records[relative] = record
        _check(artifact(relative), record["sha256"], relative, record.get("bytes"))
    _check(artifact("build_collection.py"), manifest.get("builder_sha256"), "source builder")

    def checked(relative: str) -> Path:
        if relative not in records:
            raise ValueError(f"collection manifest does not bind {relative}")
        item = records[relative]
        return _check(artifact(relative), item["sha256"], relative, item.get("bytes"))

    source = json.loads(checked("inputs/source_bindings.json").read_text())
    if source.get("corpus_fingerprint") != axis["corpus_fingerprint"]:
        raise ValueError("collection source corpus fingerprint differs")
    for record in source["source_records"]:
        if record["role"] == "frozen_panel_copy":
            _check(artifact(record["collection_copy"]), record["sha256"], "frozen source copy", record.get("bytes"))
    universe = pd.read_csv(checked("inputs/gene_universe.csv"))
    required = {"gene_id", "gene_symbol", "panel_position", "universal_rank", "eligible", "measured_cells", "detected_cells"}
    if (not required.issubset(universe.columns) or len(universe) != 20097
            or universe["gene_id"].isna().any() or not universe["gene_id"].is_unique
            or not np.array_equal(universe["panel_position"].to_numpy(), np.arange(20097))
            or not np.array_equal(universe["gene_id"].astype(str).to_numpy(), canonical["gene_id"].astype(str).to_numpy())
            or not np.array_equal(np.sort(universe["universal_rank"].to_numpy()), np.arange(1, 20098))):
        raise ValueError("collection gene universe differs from canonical raw coordinates")
    rank_record = axis["ranked_gene_positions"]
    ranks = np.load(_check(artifact(rank_record["path"]), rank_record["sha256"], "source train-HVG ranking", rank_record.get("bytes")), allow_pickle=False)
    expected_ranks = universe.sort_values("universal_rank")["panel_position"].to_numpy()
    if ranks.dtype.kind not in "iu" or not np.array_equal(ranks, expected_ranks):
        raise ValueError("collection train-HVG rank order differs from frozen rank source")
    priority = pd.read_csv(checked("inputs/literature_addition_order.csv"))
    if (len(priority) != 2091 or not priority["gene_id"].is_unique
            or not np.array_equal(priority["addition_priority_rank"].to_numpy(), np.arange(1, 2092))):
        raise ValueError("collection literature curation order differs")

    panels: dict[int, pd.DataFrame] = {}
    for size in (6000, 7000, 8000, 10000):
        prefix = f"panels/{size // 1000}k"
        table = pd.read_csv(checked(f"{prefix}/selected_genes.csv"))
        positions = np.load(checked(f"{prefix}/gene_positions.npy"), allow_pickle=False)
        if (positions.ndim != 1 or positions.dtype.kind not in "iu" or len(positions) != size
                or np.any(positions < 0) or np.any(positions >= 20097) or np.any(np.diff(positions) <= 0)):
            raise ValueError("collection positions must be unique ascending integer raw coordinates")
        if (len(table) != size or not {"panel_position", "gene_id", "gene_symbol"}.issubset(table.columns)
                or table["gene_id"].isna().any() or not table["gene_id"].is_unique
                or not np.array_equal(table["panel_position"].to_numpy(), positions)
                or not np.array_equal(table["gene_id"].astype(str).to_numpy(), universe.iloc[positions]["gene_id"].astype(str).to_numpy())
                or not np.array_equal(table["gene_symbol"].astype(str).to_numpy(), universe.iloc[positions]["gene_symbol"].astype(str).to_numpy())
                or ("tensor_index" in table and not np.array_equal(table["tensor_index"].to_numpy(), np.arange(size)))):
            raise ValueError("collection selected IDs, symbols or tensor coordinates differ")
        for filename, column in (("gene_ids.txt", "gene_id"), ("gene_symbols.txt", "gene_symbol")):
            if checked(f"{prefix}/{filename}").read_text().splitlines() != table[column].astype(str).tolist():
                raise ValueError("collection selected CSV and TXT tensor order differs")
        panels[size] = table
    positions_path = _check(Path(data["gene_positions_path"]), data["gene_positions_sha256"], "configured positions")
    if positions_path.resolve() != artifact(f"panels/{n // 1000}k/gene_positions.npy").resolve():
        raise ValueError("collection positions must use the repository-owned panel")
    selected = panels[n]
    if gene_panel is not None and not np.array_equal(gene_panel["gene_id"].astype(str).to_numpy(), selected["gene_id"].astype(str).to_numpy()):
        raise ValueError("collection runtime Ensembl tensor order differs")
    eligible = universe[universe["eligible"] & universe["measured_cells"].gt(0)].sort_values(["universal_rank", "gene_id"])
    base = set(panels[6000]["gene_id"])
    literature = priority["gene_id"].tolist()
    if (len(eligible) != 15114 or set(eligible.head(6000)["gene_id"]) != base
            or base.intersection(literature) or not set(literature).issubset(set(universe["gene_id"]))):
        raise ValueError("collection failed original6k retention or frozen literature binding")
    for size in (7000, 8000):
        if set(panels[size]["gene_id"]) != base | set(literature[:size - 6000]):
            raise ValueError("collection frozen literature addition order replay differs")
    literature_set = set(literature)
    backfill = [gene for gene in eligible["gene_id"] if gene not in base and gene not in literature_set][:1909]
    expected_ten = base | literature_set | set(backfill)
    ten = panels[10000]
    expected_reasons = {"preserved_final6k": 6000, "frozen_primary_literature_curated": 2091,
                        "train_only_HVG_ranked_backfill_not_literature_curated": 1909}
    if (len(backfill) != 1909 or set(ten["gene_id"]) != expected_ten
            or ten["selection_reason"].value_counts().to_dict() != expected_reasons):
        raise ValueError("collection 10k 6000+2091+1909 selection recipe replay differs")
    reasons = ["preserved_final6k" if gene in base else "frozen_primary_literature_curated"
               if gene in literature_set else "train_only_HVG_ranked_backfill_not_literature_curated"
               for gene in ten["gene_id"]]
    addition_order = {gene: rank + 1 for rank, gene in enumerate(literature + backfill)}
    if (not np.array_equal(ten["selection_reason"].to_numpy(), reasons)
            or not np.array_equal(ten["addition_priority_rank"].fillna(0).to_numpy(),
                                  ten["gene_id"].map(addition_order).fillna(0).to_numpy())):
        raise ValueError("collection 10k per-gene curation and train-HVG order differs")
    additions = pd.read_csv(checked("panels/10k/additions.csv"))
    expected_additions = ten[~ten["gene_id"].isin(base)].reset_index(drop=True)
    if (not additions.equals(expected_additions)
            or not additions["measured_cells"].gt(0).all() or not additions["detected_cells"].gt(0).all()):
        raise ValueError("collection 10k addition evidence differs")
    mapping = pd.read_csv(checked("panels/10k/old6k_to_10k.csv"))
    required_map = {"gene_id", "raw_panel_position", "old_tensor_index", "new_tensor_index"}
    if len(mapping) != 6000 or not required_map.issubset(mapping.columns):
        raise ValueError("collection original6k preservation map dimensions differ")
    new_indices = mapping["new_tensor_index"].to_numpy()
    if (new_indices.dtype.kind not in "iu" or np.any(new_indices < 0) or np.any(new_indices >= 10000)
            or not np.array_equal(mapping["old_tensor_index"].to_numpy(), np.arange(6000))
            or not np.array_equal(mapping["gene_id"].to_numpy(), panels[6000]["gene_id"].to_numpy())
            or not np.array_equal(mapping["raw_panel_position"].to_numpy(), panels[6000]["panel_position"].to_numpy())
            or not np.array_equal(ten.iloc[new_indices]["gene_id"].to_numpy(), panels[6000]["gene_id"].to_numpy())):
        raise ValueError("collection failed exact original6k preservation")
