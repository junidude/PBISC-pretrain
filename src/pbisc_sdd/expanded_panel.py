"""Strict loader for the separately reviewed literature-expanded panel schema.

This selector remains distinct from train-only HVG selection. Its immutable
manifest retains the benchmark-endpoint and external-reference exposure caveats.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

SCHEMA = "pbisc.rare_gene_expansion.candidate.v1"
MODE = "explicit_rare_gene_expansion_manifest"


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _checked(path: Path, expected: object, label: str) -> Path:
    expected = str(expected)
    if len(expected) != 64 or not path.is_file() or _sha(path) != expected:
        raise ValueError(f"expanded panel {label} SHA-256 differs")
    return path


def validate_expanded_panel(
    config: Mapping[str, Any], *, gene_panel: Any = None,
    corpus_fingerprint: str | None = None,
) -> None:
    """Bind tensor order, exact Ensembl IDs, raw coordinates and old6k members."""
    from pbisc.training.portable import resolve_artifact

    data = config["data"]
    if data.get("gene_selection") != MODE:
        raise ValueError("expanded panel requires its explicit selection mode")
    manifest_path = _checked(Path(data["gene_selection_manifest"]),
                             data["gene_selection_manifest_sha256"], "manifest")
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("schema") != SCHEMA or manifest.get("legacy_loader_compatible_manifest") is not False:
        raise ValueError("unsupported expanded panel schema or legacy compatibility claim")
    if not manifest.get("literature_scope") or not manifest.get("known_limitations"):
        raise ValueError("expanded panel lacks source exposure caveats")
    n = int(data["n_genes"])
    if manifest.get("n_selected_genes") != n:
        raise ValueError("expanded panel selected-gene count differs")
    positions_path = _checked(Path(data["gene_positions_path"]),
                              manifest["gene_positions_sha256"], "positions")
    if _sha(positions_path) != data["gene_positions_sha256"]:
        raise ValueError("expanded panel config positions SHA-256 differs")
    positions = np.load(positions_path, allow_pickle=False)
    if (positions.ndim != 1 or positions.dtype.kind not in "iu" or len(positions) != n
            or n == 0 or np.any(positions < 0) or np.any(np.diff(positions) <= 0)):
        raise ValueError("expanded panel positions must be unique ascending integer coordinates")
    selected_path = _checked(resolve_artifact(manifest["selected_genes_path"], config),
                             manifest["selected_genes_sha256"], "selected IDs")
    selected = pd.read_csv(selected_path)
    required = {"panel_position", "tensor_index", "gene_id"}
    if not required.issubset(selected.columns) or len(selected) != n:
        raise ValueError("expanded panel selected table dimensions differ")
    if (selected["gene_id"].isna().any() or selected["gene_id"].duplicated().any()
            or not np.array_equal(selected["panel_position"].to_numpy(), positions)
            or not np.array_equal(selected["tensor_index"].to_numpy(), np.arange(n))):
        raise ValueError("expanded panel selected IDs or tensor coordinates differ")
    bindings = manifest.get("source_bindings", [])

    def source(suffix: str) -> Mapping[str, Any]:
        matches = [item for item in bindings if str(item["path"]).endswith(suffix)]
        if len(matches) != 1:
            raise ValueError(f"expanded panel source binding is ambiguous: {suffix}")
        return matches[0]

    axis = source("/metadata/var.parquet")
    canonical_path = Path(config["e2_data"]["gene_panel_path"])
    _checked(canonical_path, axis["sha256"], "canonical gene axis")
    canonical = pd.read_parquet(canonical_path)
    if int(positions[-1]) >= len(canonical) or not np.array_equal(
        canonical.iloc[positions]["gene_id"].astype(str).to_numpy(), selected["gene_id"].astype(str).to_numpy()
    ):
        raise ValueError("expanded panel Ensembl IDs differ from canonical raw coordinates")
    contract = data.get("expanded_panel_contract", {})
    if (contract.get("canonical_panel_sha256") != axis["sha256"]
            or not contract.get("corpus_fingerprint")):
        raise ValueError("expanded panel lacks an explicit corpus and canonical-axis contract")
    if corpus_fingerprint is not None and corpus_fingerprint != contract["corpus_fingerprint"]:
        raise ValueError("expanded panel source corpus fingerprint differs")
    if gene_panel is not None and not np.array_equal(
        gene_panel["gene_id"].astype(str).to_numpy(), selected["gene_id"].astype(str).to_numpy()
    ):
        raise ValueError("expanded panel runtime Ensembl IDs differ")
    old_binding = source("/tiers/6k/selected_genes.csv")
    old_path = _checked(resolve_artifact(old_binding["path"], config), old_binding["sha256"], "original6k IDs")
    old = pd.read_csv(old_path)
    old_positions_binding = source("/tiers/6k/gene_positions.npy")
    old_positions_path = _checked(resolve_artifact(old_positions_binding["path"], config),
                                  old_positions_binding["sha256"], "original6k positions")
    old_positions = np.load(old_positions_path, allow_pickle=False)
    if (len(old) != int(manifest["old_n_genes"])
            or not np.array_equal(old["panel_position"].to_numpy(), old_positions)):
        raise ValueError("expanded panel original6k coordinates differ")
    mapping = manifest["old_to_new"]
    remap = pd.read_csv(_checked(resolve_artifact(mapping["path"], config), mapping["sha256"], "old-to-new map"))
    if (len(remap) != len(old) or not {"gene_id", "raw_panel_position", "old_tensor_index", "new_tensor_index"}.issubset(remap.columns)):
        raise ValueError("expanded panel preservation map dimensions differ")
    new_indices = remap["new_tensor_index"].to_numpy()
    if (new_indices.dtype.kind not in "iu" or np.any(new_indices < 0) or np.any(new_indices >= n)
            or not np.array_equal(remap["old_tensor_index"].to_numpy(), np.arange(len(old)))
            or not np.array_equal(remap["gene_id"].astype(str), old["gene_id"].astype(str))
            or not np.array_equal(remap["raw_panel_position"].to_numpy(), old_positions)
            or not np.array_equal(selected.iloc[new_indices]["gene_id"].astype(str).to_numpy(), old["gene_id"].astype(str).to_numpy())
            or not np.array_equal(positions[new_indices], old_positions)):
        raise ValueError("expanded panel failed exact original6k preservation")
