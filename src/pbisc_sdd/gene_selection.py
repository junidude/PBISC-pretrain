"""Train-only, measurement-aware highly variable gene selection for PBISC-SDD."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .data import DEFAULT_MODEL7_ROOT


@dataclass(frozen=True)
class HVGSelection:
    positions: np.ndarray
    mean_log1p: np.ndarray
    variance_log1p: np.ndarray
    normalized_dispersion: np.ndarray
    measured_cells: np.ndarray
    detected_cells: np.ndarray
    sampled_cells: int
    sampled_recipes: int
    corpus_fingerprint: str


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def rank_hvg_statistics(
    *,
    value_sum: np.ndarray,
    value_square_sum: np.ndarray,
    measured_cells: np.ndarray,
    detected_cells: np.ndarray,
    marker_mask: np.ndarray,
    n_genes: int,
    min_measured_cells: int,
    min_detected_cells: int,
    mean_bins: int = 20,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Rank genes by within-mean-bin log dispersion with marker retention."""

    arrays = [value_sum, value_square_sum, measured_cells, detected_cells, marker_mask]
    shape = np.asarray(value_sum).shape
    if len(shape) != 1 or any(np.asarray(value).shape != shape for value in arrays):
        raise ValueError("all HVG statistic arrays must be aligned 1-D vectors")
    if n_genes <= 0 or n_genes > shape[0]:
        raise ValueError("n_genes falls outside the statistic vectors")
    measured = np.asarray(measured_cells, dtype=np.float64)
    detected = np.asarray(detected_cells, dtype=np.float64)
    denominator = np.maximum(measured, 1.0)
    mean = np.asarray(value_sum, dtype=np.float64) / denominator
    variance = np.maximum(
        np.asarray(value_square_sum, dtype=np.float64) / denominator - mean**2,
        0.0,
    )
    eligible = (
        (measured >= int(min_measured_cells))
        & (detected >= int(min_detected_cells))
        & np.isfinite(mean)
        & np.isfinite(variance)
    )
    if int(eligible.sum()) < n_genes:
        raise ValueError(
            f"only {int(eligible.sum())} genes pass HVG eligibility, fewer than {n_genes}"
        )
    log_dispersion = np.log1p(variance / np.maximum(mean, 1.0e-8))
    score = np.full(shape, -np.inf, dtype=np.float64)
    eligible_indices = np.flatnonzero(eligible)
    eligible_mean = mean[eligible]
    quantiles = np.linspace(0.0, 1.0, max(int(mean_bins), 2) + 1)
    edges = np.unique(np.quantile(eligible_mean, quantiles))
    if edges.size < 2:
        score[eligible] = log_dispersion[eligible]
    else:
        bins = np.clip(np.searchsorted(edges, eligible_mean, side="right") - 1, 0, edges.size - 2)
        for bin_index in range(edges.size - 1):
            local = eligible_indices[bins == bin_index]
            if local.size == 0:
                continue
            values = log_dispersion[local]
            center = float(values.mean())
            # Standardized dispersion within expression bins mirrors common
            # HVG selection and avoids tiny-MAD explosions in sparse bins.
            scale = max(float(values.std()), 1.0e-3)
            score[local] = (values - center) / scale
    marker = np.asarray(marker_mask, dtype=bool) & eligible
    order = np.lexsort(
        (
            np.arange(shape[0], dtype=np.int64),
            -score,
            -marker.astype(np.int8),
        )
    )
    positions = np.sort(order[:n_genes].astype(np.int64, copy=False))
    return positions, mean.astype(np.float32), variance.astype(np.float32), score.astype(np.float32)


def select_train_hvgs(
    *,
    model7_root: Path = DEFAULT_MODEL7_ROOT,
    n_genes: int = 1000,
    recipes: int = 64,
    cells_per_recipe: int = 64,
    seed: int = 20260627,
    mean_bins: int = 20,
    target_sum: float = 10_000.0,
    progress: bool = True,
) -> tuple[HVGSelection, Any]:
    """Estimate HVGs from train-split cells without reading held-out expression."""

    del model7_root  # Legacy compatibility; use the installed E2 reader.
    from model7_no_annotation.data_e2 import E2FullDataConfig, build_e2_full_data_module

    if n_genes <= 0 or recipes <= 0 or cells_per_recipe <= 0:
        raise ValueError("n_genes, recipes, and cells_per_recipe must be positive")
    if not np.isfinite(target_sum) or target_sum <= 0:
        raise ValueError("target_sum must be finite and positive")
    base = build_e2_full_data_module(E2FullDataConfig(progress=False))
    train_recipes = base.split_manifest("train", max_blocks=recipes, seed=seed)
    if train_recipes.empty:
        raise RuntimeError("train split emitted no recipes for HVG selection")
    base.prime_pools(train_recipes)
    panel_size = int(len(base.gene_panel))
    value_sum = np.zeros(panel_size, dtype=np.float64)
    square_sum = np.zeros(panel_size, dtype=np.float64)
    measured_cells = np.zeros(panel_size, dtype=np.int64)
    detected_cells = np.zeros(panel_size, dtype=np.int64)
    sampled_cells = 0
    rows: Any = train_recipes.iterrows()
    if progress:
        from tqdm import tqdm

        rows = tqdm(rows, total=len(train_recipes), desc="select train-only PBISC HVGs")
    rng = np.random.default_rng(seed + 97_531)
    for _, recipe in rows:
        refs, pools = base._recipe_refs_and_pools(recipe)
        take = min(int(cells_per_recipe), len(refs))
        selected_indices = np.asarray(
            rng.choice(len(refs), size=take, replace=False), dtype=np.int64
        )
        selected_refs = [refs[int(index)] for index in selected_indices]
        matrix = base._load_sparse_refs(selected_refs, pools)[:, base.gene_indices]
        matrix = matrix.tocsr().astype(np.float64)
        library_size = np.asarray(matrix.sum(axis=1), dtype=np.float64).reshape(-1)
        row_scale = np.divide(
            float(target_sum),
            library_size,
            out=np.zeros_like(library_size),
            where=library_size > 0,
        )
        normalized = matrix.multiply(row_scale[:, None]).tocsr()
        logged = normalized.copy()
        logged.data = np.log1p(logged.data)
        value_sum += np.asarray(logged.sum(axis=0)).reshape(-1)
        squared = logged.copy()
        squared.data **= 2
        square_sum += np.asarray(squared.sum(axis=0)).reshape(-1)
        detected_cells += np.asarray((matrix > 0).sum(axis=0)).reshape(-1).astype(
            np.int64
        )
        segment_counts = np.bincount(
            np.asarray([segment for segment, _ in selected_refs], dtype=np.int64),
            minlength=len(pools),
        )
        for segment, count in enumerate(segment_counts):
            if int(count) <= 0:
                continue
            group = base._zarr_group(str(pools[segment]["count_zarr"]))
            mask = np.asarray(group["measurement_mask"][:], dtype=np.uint8)
            if mask.shape != (int(base.source_n_genes),) or not bool(
                np.isin(mask, (0, 1)).all()
            ):
                raise RuntimeError("invalid source measurement_mask during HVG selection")
            measured_cells += int(count) * mask[np.asarray(base.gene_indices)]
        sampled_cells += take
    gene_panel = base.gene_panel
    marker = np.zeros(panel_size, dtype=bool)
    for column in ("is_immune_marker", "is_interferon_gene"):
        if column in gene_panel.columns:
            marker |= np.asarray(gene_panel[column], dtype=bool)
    positions, mean, variance, score = rank_hvg_statistics(
        value_sum=value_sum,
        value_square_sum=square_sum,
        measured_cells=measured_cells,
        detected_cells=detected_cells,
        marker_mask=marker,
        n_genes=int(n_genes),
        min_measured_cells=max(16, int(0.1 * sampled_cells)),
        min_detected_cells=max(20, int(0.01 * sampled_cells)),
        mean_bins=int(mean_bins),
    )
    return (
        HVGSelection(
            positions=positions,
            mean_log1p=mean,
            variance_log1p=variance,
            normalized_dispersion=score,
            measured_cells=measured_cells,
            detected_cells=detected_cells,
            sampled_cells=sampled_cells,
            sampled_recipes=int(len(train_recipes)),
            corpus_fingerprint=str(base.corpus_fingerprint),
        ),
        base.gene_panel,
    )


def save_hvg_selection(
    output_dir: Path | str,
    selection: HVGSelection,
    gene_panel: Any,
    *,
    parameters: dict[str, object],
) -> dict[str, object]:
    output = Path(output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    positions_path = output / "gene_positions.npy"
    np.save(positions_path, selection.positions, allow_pickle=False)
    table = gene_panel.iloc[selection.positions][["gene_id", "gene_symbol"]].copy()
    table.insert(0, "panel_position", selection.positions)
    table["mean_log1p"] = selection.mean_log1p[selection.positions]
    table["variance_log1p"] = selection.variance_log1p[selection.positions]
    table["normalized_dispersion"] = selection.normalized_dispersion[
        selection.positions
    ]
    table["measured_cells"] = selection.measured_cells[selection.positions]
    table["detected_cells"] = selection.detected_cells[selection.positions]
    selected_genes_path = output / "selected_genes.csv"
    table.to_csv(selected_genes_path, index=False)
    implementation_path = Path(__file__).resolve()
    manifest: dict[str, object] = {
        "schema_version": "pbisc_sdd.hvg_selection.v1",
        "selection_scope": "train_split_only",
        "corpus_fingerprint": selection.corpus_fingerprint,
        "sampled_cells": int(selection.sampled_cells),
        "sampled_recipes": int(selection.sampled_recipes),
        "n_selected_genes": int(len(selection.positions)),
        "positions_path": str(positions_path),
        "positions_sha256": _sha256_file(positions_path),
        "selected_genes_path": str(selected_genes_path),
        "selected_genes_sha256": _sha256_file(selected_genes_path),
        "implementation": str(implementation_path),
        "implementation_sha256": _sha256_file(implementation_path),
        "algorithm": (
            "per-cell library normalization followed by log1p; measurement-aware "
            "mean/variance; log dispersion z-score within configured mean-expression "
            "bins; eligible immune/interferon markers prioritized"
        ),
        "parameters": parameters,
    }
    manifest_path = output / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


__all__ = [
    "HVGSelection",
    "rank_hvg_statistics",
    "save_hvg_selection",
    "select_train_hvgs",
]
