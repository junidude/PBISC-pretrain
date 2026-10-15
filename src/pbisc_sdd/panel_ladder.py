"""Versioned, train-only PBISC gene-panel ladder publication.

The legacy :mod:`pbisc_sdd.gene_selection` module is intentionally left
unchanged because published v1 artifacts bind its file hash.  This module
reuses that ranking kernel, computes one total order for the canonical gene
panel, and derives every finite tier from a prefix of that order.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import inspect
import io
import json
from pathlib import Path
import re
import shutil
import tempfile
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from .data import DEFAULT_MODEL7_ROOT
from .gene_selection import HVGSelection, rank_hvg_statistics


PANEL_LADDER_SCHEMA = "pbisc_sdd.panel_ladder.v2"
LEAKAGE_AUDIT_SCHEMA = "pbisc_sdd.panel_ladder.leakage_audit.v1"
REPLAY_PROOF_SCHEMA = "pbisc_sdd.panel_ladder.replay_proof.v1"
LEGACY_SELECTION_SCHEMA = "pbisc_sdd.hvg_selection.v1"
SUFFICIENT_STATISTICS_SCHEMA = "pbisc_sdd.panel_ladder.sufficient_statistics.v1"
MODEL7_IMPLEMENTATION_SCHEMA = "pbisc_sdd.panel_ladder.model7_implementation.v1"
DEFAULT_TIER_SIZES: Mapping[str, int] = {"2k": 2_000, "3k": 3_000, "6k": 6_000}
REPLAY_PARAMETER_KEYS = (
    "recipes",
    "cells_per_recipe",
    "mean_bins",
    "target_sum",
    "seed",
)

ORDERED_RECIPE_COLUMNS = (
    "recipe_order",
    "stream_recipe_index",
    "recipe_id",
    "block_id",
    "recipe_index",
    "schedule_slot",
    "split",
    "platform",
    "validation_scope",
    "heldout",
    "sample_key",
    "dataset_id",
    "donor_id",
    "sample_id",
    "k",
    "rng_seed",
    "sampling_policy",
)
SAMPLED_CELL_REFERENCE_COLUMNS = (
    "sample_order",
    "recipe_order",
    "chosen_order_within_recipe",
    "recipe_id",
    "block_id",
    "sample_key",
    "ref_position_in_recipe",
    "selection_occurrence_id",
    "segment_index",
    "shard_id",
    "count_zarr",
    "source_coordinate_id",
    "strict_row_index",
    "row_index",
)


class ReplayMismatchError(ValueError):
    """Raised before publication when a requested legacy replay is not exact."""


@dataclass(frozen=True)
class FullGeneRanking:
    """Tier-independent ranking and statistics for every canonical gene."""

    ranked_positions: np.ndarray
    universal_rank: np.ndarray
    eligible_rank: np.ndarray
    eligible: np.ndarray
    eligibility_reason: tuple[str, ...]
    mean_log1p: np.ndarray
    variance_log1p: np.ndarray
    log_dispersion: np.ndarray
    normalized_dispersion: np.ndarray
    measured_cells: np.ndarray
    detected_cells: np.ndarray
    marker_mask: np.ndarray

    @property
    def n_genes(self) -> int:
        return int(self.ranked_positions.size)

    @property
    def n_eligible(self) -> int:
        return int(self.eligible.sum())


@dataclass(frozen=True)
class PanelLadderSelection:
    """One full ranking plus canonical-order tensor subsets for every tier."""

    full_ranking: FullGeneRanking
    tiers: Mapping[str, np.ndarray]
    sampled_cells: int
    sampled_recipes: int
    corpus_fingerprint: str
    gene_panel_sha256: str
    min_measured_cells: int
    min_detected_cells: int
    leakage_audit: Mapping[str, object]
    provenance: TrainSelectionProvenance | None = None
    sufficient_statistics: BoundSufficientStatistics | None = None
    model7_implementation: Mapping[str, object] | None = None


@dataclass(frozen=True)
class TrainSelectionProvenance:
    """Exact ordered recipes and cell references used to compute train statistics."""

    ordered_train_recipes: pd.DataFrame
    sampled_cell_references: pd.DataFrame


@dataclass(frozen=True)
class BoundSufficientStatistics:
    """Raw statistic arrays retained so publication can revalidate and re-rank."""

    value_sum: np.ndarray
    value_square_sum: np.ndarray
    measured_cells: np.ndarray
    detected_cells: np.ndarray
    binding: Mapping[str, object]


@dataclass(frozen=True)
class TrainPanelStatistics:
    """Sufficient statistics and exact provenance from one train-only pass."""

    value_sum: np.ndarray
    value_square_sum: np.ndarray
    measured_cells: np.ndarray
    detected_cells: np.ndarray
    sampled_cells: int
    sampled_recipes: int
    corpus_fingerprint: str
    gene_panel: Any
    leakage_audit: Mapping[str, object]
    provenance: TrainSelectionProvenance
    min_measured_cells: int
    min_detected_cells: int
    mean_bins: int
    sufficient_statistics_binding: Mapping[str, object]
    model7_implementation: Mapping[str, object]


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path | str) -> str:
    """Return a streaming SHA-256 digest for a local file."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_ready(value: object) -> object:
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        value = float(value)
    if isinstance(value, float):
        if not np.isfinite(value):
            raise ValueError("manifest values must not contain NaN or Inf")
        return value
    if value is None or isinstance(value, (str, int, bool)):
        return value
    raise TypeError(f"unsupported manifest value: {type(value).__name__}")


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        _json_ready(value),
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def canonical_sha256(value: object) -> str:
    """Hash a JSON-compatible value using a stable canonical encoding."""

    return _sha256_bytes(_canonical_json_bytes(value))


def _array_content_binding(
    value: np.ndarray, *, dtype: np.dtype[Any], name: str
) -> tuple[np.ndarray, dict[str, object]]:
    array = np.asarray(value)
    expected_dtype = np.dtype(dtype)
    if array.dtype != expected_dtype:
        raise ValueError(f"{name} must have dtype {expected_dtype.name}")
    if array.ndim != 1:
        raise ValueError(f"{name} must be a 1-D vector")
    canonical = np.ascontiguousarray(array.astype(expected_dtype.newbyteorder("<"), copy=False))
    if np.issubdtype(expected_dtype, np.floating) and not np.isfinite(canonical).all():
        raise ValueError(f"{name} contains NaN or Inf")
    if np.issubdtype(expected_dtype, np.integer) and bool((canonical < 0).any()):
        raise ValueError(f"{name} contains negative counts")
    return canonical, {
        "dtype": expected_dtype.name,
        "shape": [int(canonical.size)],
        "content_sha256": _sha256_bytes(canonical.tobytes(order="C")),
    }


def build_sufficient_statistics_binding(
    *,
    value_sum: np.ndarray,
    value_square_sum: np.ndarray,
    measured_cells: np.ndarray,
    detected_cells: np.ndarray,
    sampled_cells: int,
    sampled_recipes: int,
    corpus_fingerprint: str,
    min_measured_cells: int,
    min_detected_cells: int,
    mean_bins: int,
    collection_parameters: Mapping[str, object],
) -> dict[str, object]:
    """Build a deterministic digest over raw selection statistics and their context."""

    arrays: dict[str, object] = {}
    shapes: set[int] = set()
    for name, value, dtype in (
        ("value_sum", value_sum, np.dtype(np.float64)),
        ("value_square_sum", value_square_sum, np.dtype(np.float64)),
        ("measured_cells", measured_cells, np.dtype(np.int64)),
        ("detected_cells", detected_cells, np.dtype(np.int64)),
    ):
        canonical, descriptor = _array_content_binding(value, dtype=dtype, name=name)
        shapes.add(int(canonical.size))
        arrays[name] = descriptor
    if len(shapes) != 1 or next(iter(shapes)) <= 0:
        raise ValueError("sufficient-statistic arrays must be non-empty and aligned")
    core: dict[str, object] = {
        "schema_version": SUFFICIENT_STATISTICS_SCHEMA,
        "arrays": arrays,
        "n_genes": next(iter(shapes)),
        "sampled_cells": int(sampled_cells),
        "sampled_recipes": int(sampled_recipes),
        "corpus_fingerprint": str(corpus_fingerprint),
        "min_measured_cells": int(min_measured_cells),
        "min_detected_cells": int(min_detected_cells),
        "mean_bins": int(mean_bins),
        "collection_parameters": dict(collection_parameters),
    }
    if int(sampled_cells) <= 0 or int(sampled_recipes) <= 0:
        raise ValueError("sampled cell and recipe counts must be positive")
    if int(min_measured_cells) <= 0 or int(min_detected_cells) <= 0:
        raise ValueError("selection thresholds must be positive")
    if int(mean_bins) <= 0:
        raise ValueError("mean_bins must be positive")
    return {**core, "binding_sha256": canonical_sha256(core)}


def _validate_sufficient_statistics_binding(
    statistics: BoundSufficientStatistics,
    *,
    sampled_cells: int,
    sampled_recipes: int,
    corpus_fingerprint: str,
    min_measured_cells: int,
    min_detected_cells: int,
    mean_bins: int,
) -> str:
    binding = statistics.binding
    collection_parameters = binding.get("collection_parameters")
    if not isinstance(collection_parameters, Mapping):
        raise ValueError("sufficient-statistics binding lacks collection parameters")
    expected = build_sufficient_statistics_binding(
        value_sum=statistics.value_sum,
        value_square_sum=statistics.value_square_sum,
        measured_cells=statistics.measured_cells,
        detected_cells=statistics.detected_cells,
        sampled_cells=sampled_cells,
        sampled_recipes=sampled_recipes,
        corpus_fingerprint=corpus_fingerprint,
        min_measured_cells=min_measured_cells,
        min_detected_cells=min_detected_cells,
        mean_bins=mean_bins,
        collection_parameters=collection_parameters,
    )
    if dict(binding) != expected:
        raise ValueError("sufficient-statistics binding or arrays differ")
    return str(expected["binding_sha256"])


def bind_model7_implementation(data_module: Any) -> dict[str, object]:
    """Bind the concrete Model7 class and methods used for selection I/O."""

    method_names = (
        "split_manifest",
        "prime_pools",
        "_recipe_refs_and_pools",
        "_load_sparse_refs",
        "_zarr_group",
    )
    data_class = type(data_module)
    methods: dict[str, object] = {}
    for name in method_names:
        method = getattr(data_class, name, None)
        if method is None:
            raise ValueError(f"Model7 data module lacks required method: {name}")
        source_name = inspect.getsourcefile(method) or inspect.getfile(method)
        source_path = Path(source_name).resolve()
        if not source_path.is_file():
            raise FileNotFoundError(f"Model7 implementation source not found: {source_path}")
        methods[name] = {
            "qualified_name": f"{data_class.__module__}.{data_class.__qualname__}.{name}",
            "path": str(source_path),
            "sha256": sha256_file(source_path),
        }
    core: dict[str, object] = {
        "schema_version": MODEL7_IMPLEMENTATION_SCHEMA,
        "data_module_class": f"{data_class.__module__}.{data_class.__qualname__}",
        "methods": methods,
    }
    return {**core, "bundle_sha256": canonical_sha256(core)}


def _validate_model7_implementation(binding: Mapping[str, object]) -> str:
    if binding.get("schema_version") != MODEL7_IMPLEMENTATION_SCHEMA:
        raise ValueError("unsupported Model7 implementation binding schema")
    methods = binding.get("methods")
    if not isinstance(methods, Mapping):
        raise ValueError("Model7 implementation binding lacks methods")
    required = {
        "split_manifest",
        "prime_pools",
        "_recipe_refs_and_pools",
        "_load_sparse_refs",
        "_zarr_group",
    }
    if set(str(name) for name in methods) != required:
        raise ValueError("Model7 implementation binding method set differs")
    for name, raw_entry in methods.items():
        if not isinstance(raw_entry, Mapping):
            raise ValueError(f"Model7 method binding is invalid: {name}")
        path = Path(str(raw_entry.get("path", ""))).resolve()
        if not path.is_file() or sha256_file(path) != raw_entry.get("sha256"):
            raise ValueError(f"Model7 implementation changed: {name}")
    core = {key: value for key, value in binding.items() if key != "bundle_sha256"}
    expected = canonical_sha256(core)
    if binding.get("bundle_sha256") != expected:
        raise ValueError("Model7 implementation bundle hash differs")
    return expected


def _write_json(path: Path, value: object) -> None:
    path.write_bytes(_json_file_bytes(value))


def _json_file_bytes(value: object) -> bytes:
    return (
        json.dumps(
            _json_ready(value),
            allow_nan=False,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")


def gene_panel_identity_sha256(gene_panel: Any) -> str:
    """Hash canonical gene identity and tensor order, independent of file format."""

    columns = set(getattr(gene_panel, "columns", ()))
    required = {"gene_id", "gene_symbol"}
    if not required.issubset(columns):
        missing = sorted(required.difference(columns))
        raise ValueError(f"gene panel lacks required columns: {missing}")
    panel = gene_panel.reset_index(drop=True)
    if panel.empty:
        raise ValueError("gene panel is empty")
    identity_columns = ["gene_id", "gene_symbol"]
    if "panel_index" in columns:
        identity_columns.append("panel_index")
    rows: list[dict[str, object]] = []
    for position, row in panel[identity_columns].iterrows():
        item: dict[str, object] = {
            "canonical_tensor_position": int(position),
            "gene_id": str(row["gene_id"]),
            "gene_symbol": str(row["gene_symbol"]),
        }
        if "panel_index" in identity_columns:
            item["panel_index"] = int(row["panel_index"])
        rows.append(item)
    return canonical_sha256(
        {"schema_version": "pbisc_sdd.gene_panel_identity.v1", "rows": rows}
    )


def train_only_leakage_audit(
    expression_splits_accessed: Sequence[str] = ("train",),
    *,
    evidence: Mapping[str, object] | None = None,
    ordered_recipe_ids_sha256: str | None = None,
    sampled_cell_refs_sha256: str | None = None,
    sufficient_statistics_sha256: str | None = None,
    model7_implementation_sha256: str | None = None,
) -> dict[str, object]:
    """Validate and record the expression-data split boundary used for ranking."""

    accessed = tuple(dict.fromkeys(str(split_name) for split_name in expression_splits_accessed))
    if accessed != ("train",):
        raise ValueError(
            "panel ranking may access expression statistics from the train split only"
        )
    hex_digest = re.compile(r"[0-9a-f]{64}")
    digests = {
        "ordered_recipe_ids_sha256": ordered_recipe_ids_sha256,
        "sampled_cell_refs_sha256": sampled_cell_refs_sha256,
        "sufficient_statistics_sha256": sufficient_statistics_sha256,
        "model7_implementation_sha256": model7_implementation_sha256,
    }
    for name, value in digests.items():
        if value is not None and not hex_digest.fullmatch(value):
            raise ValueError(f"{name} must be a lowercase SHA-256")
    exact_recipe_ids = ordered_recipe_ids_sha256 is not None
    exact_sample_refs = sampled_cell_refs_sha256 is not None
    exact_statistics = sufficient_statistics_sha256 is not None
    exact_implementation = model7_implementation_sha256 is not None
    production_ready = bool(
        exact_recipe_ids and exact_sample_refs and exact_statistics and exact_implementation
    )
    combined_binding = (
        canonical_sha256(digests) if production_ready else None
    )
    return {
        "schema_version": LEAKAGE_AUDIT_SCHEMA,
        "status": "pass",
        "selection_scope": "train_split_only",
        "allowed_expression_splits": ["train"],
        "expression_splits_accessed": ["train"],
        "held_out_expression_accessed": False,
        "held_out_splits": ["val", "test", "diagnostic", "evaluation"],
        "ordered_recipe_ids_sha256": ordered_recipe_ids_sha256,
        "exact_ordered_recipe_ids_persisted": exact_recipe_ids,
        "sampled_cell_refs_sha256": sampled_cell_refs_sha256,
        "exact_sample_refs_persisted": exact_sample_refs,
        "sufficient_statistics_sha256": sufficient_statistics_sha256,
        "exact_sufficient_statistics_bound": exact_statistics,
        "model7_implementation_sha256": model7_implementation_sha256,
        "exact_model7_implementation_bound": exact_implementation,
        "provenance_statistics_binding_sha256": combined_binding,
        "production_ready": production_ready,
        "provenance_limitation": (
            None
            if production_ready
            else (
                "recipe IDs, sampled refs, sufficient statistics, and Model7 "
                "implementation are not all cryptographically bound"
            )
        ),
        "evidence": {} if evidence is None else dict(evidence),
    }


def _validate_leakage_audit(audit: Mapping[str, object]) -> None:
    if audit.get("schema_version") != LEAKAGE_AUDIT_SCHEMA:
        raise ValueError("unsupported train-only leakage audit schema")
    if audit.get("status") != "pass":
        raise ValueError("train-only leakage audit did not pass")
    if audit.get("selection_scope") != "train_split_only":
        raise ValueError("panel selection scope is not train-only")
    if audit.get("expression_splits_accessed") != ["train"]:
        raise ValueError("leakage audit includes non-train expression access")
    if audit.get("held_out_expression_accessed") is not False:
        raise ValueError("leakage audit reports held-out expression access")
    exact_recipe_ids = audit.get("exact_ordered_recipe_ids_persisted") is True
    exact_sample_refs = audit.get("exact_sample_refs_persisted") is True
    exact_statistics = audit.get("exact_sufficient_statistics_bound") is True
    exact_implementation = audit.get("exact_model7_implementation_bound") is True
    if audit.get("production_ready") is True and not (
        exact_recipe_ids and exact_sample_refs and exact_statistics and exact_implementation
    ):
        raise ValueError("production-ready selection lacks a required exact binding")
    if audit.get("production_ready") is True:
        keys = (
            "ordered_recipe_ids_sha256",
            "sampled_cell_refs_sha256",
            "sufficient_statistics_sha256",
            "model7_implementation_sha256",
        )
        for key in keys:
            value = audit.get(key)
            if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
                raise ValueError(f"production-ready provenance lacks valid {key}")
        expected_combined = canonical_sha256(
            {key: audit.get(key) for key in keys}
        )
        if audit.get("provenance_statistics_binding_sha256") != expected_combined:
            raise ValueError("combined provenance/statistics binding differs")


def _marker_mask(gene_panel: Any) -> np.ndarray:
    marker = np.zeros(len(gene_panel), dtype=bool)
    for column in ("is_immune_marker", "is_interferon_gene"):
        if column in gene_panel.columns:
            marker |= np.asarray(gene_panel[column], dtype=bool)
    return marker


def _eligibility_reasons(
    *,
    measured_cells: np.ndarray,
    detected_cells: np.ndarray,
    mean_log1p: np.ndarray,
    variance_log1p: np.ndarray,
    normalized_dispersion: np.ndarray,
    min_measured_cells: int,
    min_detected_cells: int,
) -> tuple[np.ndarray, tuple[str, ...]]:
    reasons: list[str] = []
    eligible_values: list[bool] = []
    for measured, detected, mean, variance, score in zip(
        measured_cells,
        detected_cells,
        mean_log1p,
        variance_log1p,
        normalized_dispersion,
        strict=True,
    ):
        local: list[str] = []
        if float(measured) < int(min_measured_cells):
            local.append("below_min_measured_cells")
        if float(detected) < int(min_detected_cells):
            local.append("below_min_detected_cells")
        if not np.isfinite(mean):
            local.append("nonfinite_mean_log1p")
        if not np.isfinite(variance):
            local.append("nonfinite_variance_log1p")
        if not local and not np.isfinite(score):
            local.append("nonfinite_normalized_dispersion")
        eligible_values.append(not local)
        reasons.append("eligible" if not local else "|".join(local))
    return np.asarray(eligible_values, dtype=bool), tuple(reasons)


def _full_ranking(
    *,
    mean_log1p: np.ndarray,
    variance_log1p: np.ndarray,
    normalized_dispersion: np.ndarray,
    measured_cells: np.ndarray,
    detected_cells: np.ndarray,
    marker_mask: np.ndarray,
    min_measured_cells: int,
    min_detected_cells: int,
) -> FullGeneRanking:
    arrays = [
        np.asarray(mean_log1p),
        np.asarray(variance_log1p),
        np.asarray(normalized_dispersion),
        np.asarray(measured_cells),
        np.asarray(detected_cells),
        np.asarray(marker_mask),
    ]
    shape = arrays[0].shape
    if len(shape) != 1 or any(array.shape != shape for array in arrays):
        raise ValueError("all full-ranking arrays must be aligned 1-D vectors")
    if shape[0] == 0:
        raise ValueError("full ranking requires at least one gene")
    mean = np.asarray(mean_log1p, dtype=np.float32)
    variance = np.asarray(variance_log1p, dtype=np.float32)
    score = np.asarray(normalized_dispersion, dtype=np.float32)
    measured = np.asarray(measured_cells, dtype=np.int64)
    detected = np.asarray(detected_cells, dtype=np.int64)
    marker = np.asarray(marker_mask, dtype=bool)
    eligible, reasons = _eligibility_reasons(
        measured_cells=measured,
        detected_cells=detected,
        mean_log1p=mean,
        variance_log1p=variance,
        normalized_dispersion=score,
        min_measured_cells=min_measured_cells,
        min_detected_cells=min_detected_cells,
    )
    if not bool(eligible.any()):
        raise ValueError("no genes pass panel-ladder eligibility")
    canonical = np.arange(shape[0], dtype=np.int64)
    ranked = np.lexsort((canonical, -score, -(marker & eligible).astype(np.int8))).astype(
        np.int64, copy=False
    )
    if not bool(eligible[ranked[: int(eligible.sum())]].all()):
        raise RuntimeError("eligible genes are not a contiguous ranking prefix")
    universal_rank = np.empty(shape[0], dtype=np.int64)
    universal_rank[ranked] = np.arange(1, shape[0] + 1, dtype=np.int64)
    eligible_rank = np.zeros(shape[0], dtype=np.int64)
    eligible_rank[ranked[: int(eligible.sum())]] = np.arange(
        1, int(eligible.sum()) + 1, dtype=np.int64
    )
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        log_dispersion = np.log1p(
            variance.astype(np.float64) / np.maximum(mean.astype(np.float64), 1.0e-8)
        ).astype(np.float32)
    return FullGeneRanking(
        ranked_positions=ranked,
        universal_rank=universal_rank,
        eligible_rank=eligible_rank,
        eligible=eligible,
        eligibility_reason=reasons,
        mean_log1p=mean,
        variance_log1p=variance,
        log_dispersion=log_dispersion,
        normalized_dispersion=score,
        measured_cells=measured,
        detected_cells=detected,
        marker_mask=marker,
    )


def rank_hvg_full_statistics(
    *,
    value_sum: np.ndarray,
    value_square_sum: np.ndarray,
    measured_cells: np.ndarray,
    detected_cells: np.ndarray,
    marker_mask: np.ndarray,
    min_measured_cells: int,
    min_detected_cells: int,
    mean_bins: int = 20,
) -> FullGeneRanking:
    """Run the sealed v1 scoring kernel once and return a total gene ranking."""

    _, mean, variance, score = rank_hvg_statistics(
        value_sum=value_sum,
        value_square_sum=value_square_sum,
        measured_cells=measured_cells,
        detected_cells=detected_cells,
        marker_mask=marker_mask,
        n_genes=1,
        min_measured_cells=min_measured_cells,
        min_detected_cells=min_detected_cells,
        mean_bins=mean_bins,
    )
    return _full_ranking(
        mean_log1p=mean,
        variance_log1p=variance,
        normalized_dispersion=score,
        measured_cells=measured_cells,
        detected_cells=detected_cells,
        marker_mask=marker_mask,
        min_measured_cells=min_measured_cells,
        min_detected_cells=min_detected_cells,
    )


def _normalise_tier_sizes(
    tier_sizes: Mapping[str, int] | Sequence[int],
) -> tuple[tuple[str, int], ...]:
    if isinstance(tier_sizes, Mapping):
        items = [(str(name).lower(), int(size)) for name, size in tier_sizes.items()]
    else:
        items = []
        for raw_size in tier_sizes:
            size = int(raw_size)
            name = f"{size // 1000}k" if size >= 1000 and size % 1000 == 0 else str(size)
            items.append((name, size))
    if not items:
        raise ValueError("at least one finite panel tier is required")
    names = [name for name, _ in items]
    sizes = [size for _, size in items]
    if "broad" in names:
        raise ValueError("the broad tier is created automatically")
    if len(set(names)) != len(names) or len(set(sizes)) != len(sizes):
        raise ValueError("finite panel tier names and sizes must be unique")
    if any(not re.fullmatch(r"[a-z0-9][a-z0-9_-]*", name) for name in names):
        raise ValueError("panel tier names must be safe lowercase path components")
    if any(size <= 0 for size in sizes):
        raise ValueError("panel tier sizes must be positive")
    return tuple(sorted(items, key=lambda item: item[1]))


def _provenance_hashes(
    provenance: TrainSelectionProvenance,
) -> tuple[str, str]:
    recipes = provenance.ordered_train_recipes
    references = provenance.sampled_cell_references
    if tuple(recipes.columns) != ORDERED_RECIPE_COLUMNS:
        raise ValueError("ordered train-recipe provenance columns differ")
    if tuple(references.columns) != SAMPLED_CELL_REFERENCE_COLUMNS:
        raise ValueError("sampled-cell provenance columns differ")
    return (
        _sha256_bytes(_csv_bytes(recipes)),
        _sha256_bytes(_csv_bytes(references)),
    )


def _validate_provenance(
    provenance: TrainSelectionProvenance,
    *,
    sampled_recipes: int,
    sampled_cells: int,
    corpus_fingerprint: str,
    leakage_audit: Mapping[str, object],
) -> tuple[str, str]:
    recipes = provenance.ordered_train_recipes
    references = provenance.sampled_cell_references
    if len(recipes) != int(sampled_recipes):
        raise ValueError("ordered recipe provenance row count differs")
    if len(references) != int(sampled_cells):
        raise ValueError("sampled-cell provenance row count differs")
    if recipes["recipe_order"].tolist() != list(range(len(recipes))):
        raise ValueError("ordered recipe provenance is not in stable selection order")
    if recipes["stream_recipe_index"].tolist() != list(range(len(recipes))):
        raise ValueError("Model7 stream recipe order differs from persisted recipe order")
    if references["sample_order"].tolist() != list(range(len(references))):
        raise ValueError("sampled-cell provenance is not in stable chosen order")
    if not recipes["split"].astype(str).eq("train").all():
        raise ValueError("ordered recipe provenance contains a held-out split")
    if recipes["heldout"].astype(bool).any():
        raise ValueError("ordered recipe provenance contains held-out recipes")
    recipe_identity = {
        int(row.recipe_order): (
            str(row.recipe_id),
            str(row.block_id),
            str(row.sample_key),
        )
        for row in recipes.itertuples(index=False)
    }
    for row in references.itertuples(index=False):
        expected = recipe_identity.get(int(row.recipe_order))
        actual = (str(row.recipe_id), str(row.block_id), str(row.sample_key))
        if expected is None or actual != expected:
            raise ValueError("sampled-cell reference does not bind its recipe identity")
        if int(row.chosen_order_within_recipe) < 0 or int(row.ref_position_in_recipe) < 0:
            raise ValueError("sampled-cell reference contains a negative order/index")
        if int(row.segment_index) < 0 or int(row.row_index) < 0:
            raise ValueError("sampled-cell reference contains a negative source coordinate")
        if int(row.strict_row_index) != int(row.row_index):
            raise ValueError("strict and compatibility row indices differ")
        if not str(row.shard_id) or not str(row.count_zarr):
            raise ValueError("sampled-cell reference lacks source-shard identity")
        expected_occurrence = canonical_sha256(
            {
                "recipe_id": str(row.recipe_id),
                "ref_position_in_recipe": int(row.ref_position_in_recipe),
            }
        )
        if str(row.selection_occurrence_id) != expected_occurrence:
            raise ValueError("sampled-cell occurrence identity differs")
        expected_source = canonical_sha256(
            {
                "corpus_fingerprint": str(corpus_fingerprint),
                "shard_id": str(row.shard_id),
                "strict_row_index": int(row.strict_row_index),
            }
        )
        if str(row.source_coordinate_id) != expected_source:
            raise ValueError("sampled-cell source-coordinate identity differs")
    for recipe_order, group in references.groupby("recipe_order", sort=False):
        chosen = group["chosen_order_within_recipe"].astype(int).tolist()
        if chosen != list(range(len(group))):
            raise ValueError(
                f"sampled-cell chosen order is not contiguous for recipe {recipe_order}"
            )
        if group["ref_position_in_recipe"].astype(int).duplicated().any():
            raise ValueError(f"sampled-cell references repeat within recipe {recipe_order}")
    recipe_sha256, refs_sha256 = _provenance_hashes(provenance)
    if leakage_audit.get("ordered_recipe_ids_sha256") != recipe_sha256:
        raise ValueError("ordered train-recipe provenance hash differs from audit")
    if leakage_audit.get("sampled_cell_refs_sha256") != refs_sha256:
        raise ValueError("sampled-cell provenance hash differs from audit")
    if leakage_audit.get("production_ready") is not True:
        raise ValueError("exact provenance is present but audit is not production-ready")
    return recipe_sha256, refs_sha256


def _assemble_ladder(
    *,
    full_ranking: FullGeneRanking,
    gene_panel: Any,
    tier_sizes: Mapping[str, int] | Sequence[int],
    sampled_cells: int,
    sampled_recipes: int,
    corpus_fingerprint: str,
    min_measured_cells: int,
    min_detected_cells: int,
    leakage_audit: Mapping[str, object],
    provenance: TrainSelectionProvenance | None = None,
    sufficient_statistics: BoundSufficientStatistics | None = None,
    model7_implementation: Mapping[str, object] | None = None,
) -> PanelLadderSelection:
    if len(gene_panel) != full_ranking.n_genes:
        raise ValueError("gene panel and full ranking have different lengths")
    finite_tiers = _normalise_tier_sizes(tier_sizes)
    largest = finite_tiers[-1][1]
    if largest > full_ranking.n_eligible:
        raise ValueError(
            f"only {full_ranking.n_eligible} genes are eligible, fewer than tier {largest}"
        )
    _validate_leakage_audit(leakage_audit)
    if provenance is None:
        if leakage_audit.get("production_ready") is True:
            raise ValueError("production-ready selection lacks exact provenance tables")
        if sufficient_statistics is not None or model7_implementation is not None:
            raise ValueError("production bindings require exact provenance tables")
    else:
        if sufficient_statistics is None or model7_implementation is None:
            raise ValueError(
                "exact provenance requires sufficient-statistics and Model7 bindings"
            )
        _validate_provenance(
            provenance,
            sampled_recipes=sampled_recipes,
            sampled_cells=sampled_cells,
            corpus_fingerprint=corpus_fingerprint,
            leakage_audit=leakage_audit,
        )
        statistics_sha256 = _validate_sufficient_statistics_binding(
            sufficient_statistics,
            sampled_cells=sampled_cells,
            sampled_recipes=sampled_recipes,
            corpus_fingerprint=corpus_fingerprint,
            min_measured_cells=min_measured_cells,
            min_detected_cells=min_detected_cells,
            mean_bins=int(sufficient_statistics.binding.get("mean_bins", -1)),
        )
        if leakage_audit.get("sufficient_statistics_sha256") != statistics_sha256:
            raise ValueError("audit does not bind the supplied sufficient statistics")
        model7_sha256 = _validate_model7_implementation(model7_implementation)
        if leakage_audit.get("model7_implementation_sha256") != model7_sha256:
            raise ValueError("audit does not bind the Model7 implementation")
    eligible_order = full_ranking.ranked_positions[: full_ranking.n_eligible]
    tiers: dict[str, np.ndarray] = {
        name: np.sort(eligible_order[:size]).astype(np.int64, copy=False)
        for name, size in finite_tiers
    }
    tiers["broad"] = np.sort(eligible_order).astype(np.int64, copy=False)
    previous: set[int] = set()
    for name, _ in finite_tiers:
        current = set(int(value) for value in tiers[name])
        if not previous.issubset(current):
            raise RuntimeError("finite panel tiers are not nested")
        previous = current
    if not previous.issubset(set(int(value) for value in tiers["broad"])):
        raise RuntimeError("broad panel does not contain every finite tier")
    return PanelLadderSelection(
        full_ranking=full_ranking,
        tiers=tiers,
        sampled_cells=int(sampled_cells),
        sampled_recipes=int(sampled_recipes),
        corpus_fingerprint=str(corpus_fingerprint),
        gene_panel_sha256=gene_panel_identity_sha256(gene_panel),
        min_measured_cells=int(min_measured_cells),
        min_detected_cells=int(min_detected_cells),
        leakage_audit=dict(leakage_audit),
        provenance=provenance,
        sufficient_statistics=sufficient_statistics,
        model7_implementation=(
            None if model7_implementation is None else dict(model7_implementation)
        ),
    )


def build_panel_ladder(
    *,
    value_sum: np.ndarray,
    value_square_sum: np.ndarray,
    measured_cells: np.ndarray,
    detected_cells: np.ndarray,
    gene_panel: Any,
    sampled_cells: int,
    sampled_recipes: int,
    corpus_fingerprint: str,
    min_measured_cells: int,
    min_detected_cells: int,
    mean_bins: int = 20,
    tier_sizes: Mapping[str, int] | Sequence[int] = DEFAULT_TIER_SIZES,
    leakage_audit: Mapping[str, object] | None = None,
    provenance: TrainSelectionProvenance | None = None,
    sufficient_statistics_binding: Mapping[str, object] | None = None,
    model7_implementation: Mapping[str, object] | None = None,
) -> PanelLadderSelection:
    """Build a fixture-friendly ladder directly from sufficient statistics."""

    sufficient_statistics: BoundSufficientStatistics | None = None
    if provenance is not None:
        if sufficient_statistics_binding is None or model7_implementation is None:
            raise ValueError(
                "exact provenance requires matching statistics and Model7 bindings"
            )
        sufficient_statistics = BoundSufficientStatistics(
            value_sum=np.asarray(value_sum, dtype=np.float64).copy(),
            value_square_sum=np.asarray(value_square_sum, dtype=np.float64).copy(),
            measured_cells=np.asarray(measured_cells, dtype=np.int64).copy(),
            detected_cells=np.asarray(detected_cells, dtype=np.int64).copy(),
            binding=dict(sufficient_statistics_binding),
        )
        _validate_sufficient_statistics_binding(
            sufficient_statistics,
            sampled_cells=sampled_cells,
            sampled_recipes=sampled_recipes,
            corpus_fingerprint=corpus_fingerprint,
            min_measured_cells=min_measured_cells,
            min_detected_cells=min_detected_cells,
            mean_bins=mean_bins,
        )
    elif sufficient_statistics_binding is not None or model7_implementation is not None:
        raise ValueError("statistics/implementation bindings require exact provenance")
    ranking = rank_hvg_full_statistics(
        value_sum=value_sum,
        value_square_sum=value_square_sum,
        measured_cells=measured_cells,
        detected_cells=detected_cells,
        marker_mask=_marker_mask(gene_panel),
        min_measured_cells=min_measured_cells,
        min_detected_cells=min_detected_cells,
        mean_bins=mean_bins,
    )
    audit = leakage_audit or train_only_leakage_audit(
        evidence={"statistics_origin": "caller_attested_train_only_fixture"}
    )
    return _assemble_ladder(
        full_ranking=ranking,
        gene_panel=gene_panel,
        tier_sizes=tier_sizes,
        sampled_cells=sampled_cells,
        sampled_recipes=sampled_recipes,
        corpus_fingerprint=corpus_fingerprint,
        min_measured_cells=min_measured_cells,
        min_detected_cells=min_detected_cells,
        leakage_audit=audit,
        provenance=provenance,
        sufficient_statistics=sufficient_statistics,
        model7_implementation=model7_implementation,
    )


def panel_ladder_from_hvg_selection(
    selection: HVGSelection,
    gene_panel: Any,
    *,
    tier_sizes: Mapping[str, int] | Sequence[int] = DEFAULT_TIER_SIZES,
    min_measured_cells: int | None = None,
    min_detected_cells: int | None = None,
    leakage_audit: Mapping[str, object] | None = None,
    provenance: TrainSelectionProvenance | None = None,
) -> PanelLadderSelection:
    """Derive all tiers from one v1 train-only statistics pass."""

    measured_min = (
        max(16, int(0.1 * selection.sampled_cells))
        if min_measured_cells is None
        else int(min_measured_cells)
    )
    detected_min = (
        max(20, int(0.01 * selection.sampled_cells))
        if min_detected_cells is None
        else int(min_detected_cells)
    )
    ranking = _full_ranking(
        mean_log1p=selection.mean_log1p,
        variance_log1p=selection.variance_log1p,
        normalized_dispersion=selection.normalized_dispersion,
        measured_cells=selection.measured_cells,
        detected_cells=selection.detected_cells,
        marker_mask=_marker_mask(gene_panel),
        min_measured_cells=measured_min,
        min_detected_cells=detected_min,
    )
    audit = leakage_audit or train_only_leakage_audit(
        evidence={
            "selector": "pbisc_sdd.gene_selection.select_train_hvgs",
            "selector_split_argument": "train",
        }
    )
    return _assemble_ladder(
        full_ranking=ranking,
        gene_panel=gene_panel,
        tier_sizes=tier_sizes,
        sampled_cells=selection.sampled_cells,
        sampled_recipes=selection.sampled_recipes,
        corpus_fingerprint=selection.corpus_fingerprint,
        min_measured_cells=measured_min,
        min_detected_cells=detected_min,
        leakage_audit=audit,
        provenance=provenance,
        sufficient_statistics=None,
        model7_implementation=None,
    )


def _build_model7_data_module(model7_root: Path, data_config: Mapping[str, object] | None = None) -> Any:
    del model7_root  # Legacy compatibility; use the installed E2 reader.
    from model7_no_annotation.data_e2 import (  # type: ignore[import-not-found]
        E2FullDataConfig,
        build_e2_full_data_module,
    )

    return build_e2_full_data_module(E2FullDataConfig(**{**dict(data_config or {}), "progress": False}))


def collect_train_panel_statistics(
    data_module: Any,
    *,
    recipes: int = 256,
    cells_per_recipe: int = 128,
    seed: int = 20260627,
    target_sum: float = 10_000.0,
    mean_bins: int = 20,
    progress: bool = True,
) -> TrainPanelStatistics:
    """Collect exact train statistics and ordered source references in one pass."""

    if recipes <= 0 or cells_per_recipe <= 0:
        raise ValueError("recipes and cells_per_recipe must be positive")
    if not np.isfinite(target_sum) or target_sum <= 0:
        raise ValueError("target_sum must be finite and positive")
    if int(mean_bins) <= 0:
        raise ValueError("mean_bins must be positive")
    model7_implementation = bind_model7_implementation(data_module)
    model7_implementation_sha256 = _validate_model7_implementation(
        model7_implementation
    )
    train_recipes = data_module.split_manifest(
        "train", max_blocks=int(recipes), seed=int(seed)
    )
    if train_recipes.empty:
        raise RuntimeError("train split emitted no recipes for panel selection")
    required_recipe_columns = set(ORDERED_RECIPE_COLUMNS).difference({"recipe_order"})
    missing = sorted(required_recipe_columns.difference(train_recipes.columns))
    if missing:
        raise RuntimeError(f"train recipe manifest lacks provenance columns: {missing}")
    if not train_recipes["split"].astype(str).eq("train").all():
        raise RuntimeError("train split manifest returned held-out recipe rows")
    if train_recipes["heldout"].astype(bool).any():
        raise RuntimeError("train split manifest returned held-out recipes")
    if train_recipes["recipe_id"].astype(str).duplicated().any():
        raise RuntimeError("train split manifest returned duplicate recipe IDs")
    data_module.prime_pools(train_recipes)

    panel_size = int(len(data_module.gene_panel))
    value_sum = np.zeros(panel_size, dtype=np.float64)
    square_sum = np.zeros(panel_size, dtype=np.float64)
    measured_cells = np.zeros(panel_size, dtype=np.int64)
    detected_cells = np.zeros(panel_size, dtype=np.int64)
    recipe_records: list[dict[str, object]] = []
    reference_records: list[dict[str, object]] = []
    sampled_cells = 0
    rows: Any = train_recipes.iterrows()
    if progress:
        from tqdm import tqdm

        rows = tqdm(rows, total=len(train_recipes), desc="select train-only panel ladder")
    rng = np.random.default_rng(int(seed) + 97_531)
    gene_indices = np.asarray(data_module.gene_indices, dtype=np.int64)
    corpus_fingerprint = str(data_module.corpus_fingerprint)

    for recipe_order, (_, recipe) in enumerate(rows):
        recipe_record = {
            "recipe_order": int(recipe_order),
            **{
                column: (
                    bool(recipe[column])
                    if column == "heldout"
                    else int(recipe[column])
                    if column
                    in {
                        "stream_recipe_index",
                        "recipe_index",
                        "schedule_slot",
                        "k",
                        "rng_seed",
                    }
                    else str(recipe[column])
                )
                for column in ORDERED_RECIPE_COLUMNS
                if column != "recipe_order"
            },
        }
        recipe_records.append(recipe_record)
        refs, pools = data_module._recipe_refs_and_pools(recipe)
        take = min(int(cells_per_recipe), len(refs))
        selected_indices = np.asarray(
            rng.choice(len(refs), size=take, replace=False), dtype=np.int64
        )
        selected_refs = [refs[int(index)] for index in selected_indices]
        for chosen_order, (ref_position, selected_ref) in enumerate(
            zip(selected_indices, selected_refs, strict=True)
        ):
            segment, row_index = (int(selected_ref[0]), int(selected_ref[1]))
            if segment < 0 or segment >= len(pools):
                raise RuntimeError("recipe reference contains an invalid pool segment")
            pool = pools[segment]
            shard_id = str(pool.get("shard_id", ""))
            count_zarr = str(pool.get("count_zarr", ""))
            if not shard_id or not count_zarr or row_index < 0:
                raise RuntimeError("recipe reference lacks exact source identity")
            reference_records.append(
                {
                    "sample_order": int(sampled_cells + chosen_order),
                    "recipe_order": int(recipe_order),
                    "chosen_order_within_recipe": int(chosen_order),
                    "recipe_id": str(recipe["recipe_id"]),
                    "block_id": str(recipe["block_id"]),
                    "sample_key": str(recipe["sample_key"]),
                    "ref_position_in_recipe": int(ref_position),
                    "selection_occurrence_id": canonical_sha256(
                        {
                            "recipe_id": str(recipe["recipe_id"]),
                            "ref_position_in_recipe": int(ref_position),
                        }
                    ),
                    "segment_index": segment,
                    "shard_id": shard_id,
                    "count_zarr": count_zarr,
                    "source_coordinate_id": canonical_sha256(
                        {
                            "corpus_fingerprint": corpus_fingerprint,
                            "shard_id": shard_id,
                            "strict_row_index": row_index,
                        }
                    ),
                    "strict_row_index": row_index,
                    "row_index": row_index,
                }
            )

        matrix = data_module._load_sparse_refs(selected_refs, pools)[:, gene_indices]
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
            group = data_module._zarr_group(str(pools[segment]["count_zarr"]))
            mask = np.asarray(group["measurement_mask"][:], dtype=np.uint8)
            if mask.shape != (int(data_module.source_n_genes),) or not bool(
                np.isin(mask, (0, 1)).all()
            ):
                raise RuntimeError("invalid source measurement_mask during panel selection")
            measured_cells += int(count) * mask[gene_indices]
        sampled_cells += take

    ordered_recipes = pd.DataFrame.from_records(
        recipe_records, columns=ORDERED_RECIPE_COLUMNS
    )
    sampled_references = pd.DataFrame.from_records(
        reference_records, columns=SAMPLED_CELL_REFERENCE_COLUMNS
    )
    provenance = TrainSelectionProvenance(
        ordered_train_recipes=ordered_recipes,
        sampled_cell_references=sampled_references,
    )
    recipe_sha256, reference_sha256 = _provenance_hashes(provenance)
    min_measured_cells = max(16, int(0.1 * sampled_cells))
    min_detected_cells = max(20, int(0.01 * sampled_cells))
    statistics_binding = build_sufficient_statistics_binding(
        value_sum=value_sum,
        value_square_sum=square_sum,
        measured_cells=measured_cells,
        detected_cells=detected_cells,
        sampled_cells=sampled_cells,
        sampled_recipes=len(train_recipes),
        corpus_fingerprint=corpus_fingerprint,
        min_measured_cells=min_measured_cells,
        min_detected_cells=min_detected_cells,
        mean_bins=int(mean_bins),
        collection_parameters={
            "recipes": int(recipes),
            "cells_per_recipe": int(cells_per_recipe),
            "seed": int(seed),
            "target_sum": float(target_sum),
            "mean_bins": int(mean_bins),
        },
    )
    audit = train_only_leakage_audit(
        evidence={
            "collector": "pbisc_sdd.panel_ladder.collect_train_panel_statistics",
            "split_manifest_requests": ["train"],
            "heldout_recipe_rows": 0,
            "sampling_passes": 1,
            "stable_sample_order": "recipe order then RNG-chosen order within recipe",
        },
        ordered_recipe_ids_sha256=recipe_sha256,
        sampled_cell_refs_sha256=reference_sha256,
        sufficient_statistics_sha256=str(statistics_binding["binding_sha256"]),
        model7_implementation_sha256=model7_implementation_sha256,
    )
    _validate_provenance(
        provenance,
        sampled_recipes=len(train_recipes),
        sampled_cells=sampled_cells,
        corpus_fingerprint=corpus_fingerprint,
        leakage_audit=audit,
    )
    return TrainPanelStatistics(
        value_sum=value_sum,
        value_square_sum=square_sum,
        measured_cells=measured_cells,
        detected_cells=detected_cells,
        sampled_cells=sampled_cells,
        sampled_recipes=int(len(train_recipes)),
        corpus_fingerprint=corpus_fingerprint,
        gene_panel=data_module.gene_panel,
        leakage_audit=audit,
        provenance=provenance,
        min_measured_cells=min_measured_cells,
        min_detected_cells=min_detected_cells,
        mean_bins=int(mean_bins),
        sufficient_statistics_binding=statistics_binding,
        model7_implementation=model7_implementation,
    )


def build_production_panel_ladder(
    statistics: TrainPanelStatistics,
    *,
    tier_sizes: Mapping[str, int] | Sequence[int] = DEFAULT_TIER_SIZES,
) -> PanelLadderSelection:
    """Assemble a ladder only from one collector-owned, fully bound statistics pass."""

    return build_panel_ladder(
        value_sum=statistics.value_sum,
        value_square_sum=statistics.value_square_sum,
        measured_cells=statistics.measured_cells,
        detected_cells=statistics.detected_cells,
        gene_panel=statistics.gene_panel,
        sampled_cells=statistics.sampled_cells,
        sampled_recipes=statistics.sampled_recipes,
        corpus_fingerprint=statistics.corpus_fingerprint,
        min_measured_cells=statistics.min_measured_cells,
        min_detected_cells=statistics.min_detected_cells,
        mean_bins=statistics.mean_bins,
        tier_sizes=tier_sizes,
        leakage_audit=statistics.leakage_audit,
        provenance=statistics.provenance,
        sufficient_statistics_binding=statistics.sufficient_statistics_binding,
        model7_implementation=statistics.model7_implementation,
    )


def select_train_panel_ladder(
    *,
    model7_root: Path = DEFAULT_MODEL7_ROOT,
    recipes: int = 256,
    cells_per_recipe: int = 128,
    seed: int = 20260627,
    mean_bins: int = 20,
    target_sum: float = 10_000.0,
    tier_sizes: Mapping[str, int] | Sequence[int] = DEFAULT_TIER_SIZES,
    progress: bool = True,
    data_config: Mapping[str, object] | None = None,
) -> tuple[PanelLadderSelection, Any]:
    """Collect a production-auditable train pass, then derive every panel width."""

    data_module = _build_model7_data_module(Path(model7_root), data_config)
    statistics = collect_train_panel_statistics(
        data_module,
        recipes=recipes,
        cells_per_recipe=cells_per_recipe,
        seed=seed,
        target_sum=target_sum,
        mean_bins=mean_bins,
        progress=progress,
    )
    ladder = build_production_panel_ladder(statistics, tier_sizes=tier_sizes)
    return ladder, statistics.gene_panel


def full_gene_statistics_table(
    ladder: PanelLadderSelection, gene_panel: Any
) -> pd.DataFrame:
    """Return one canonical-order row for every gene, including rejections."""

    ranking = ladder.full_ranking
    if len(gene_panel) != ranking.n_genes:
        raise ValueError("gene panel and ladder have different lengths")
    table = gene_panel.reset_index(drop=True)[["gene_id", "gene_symbol"]].copy()
    canonical = np.arange(ranking.n_genes, dtype=np.int64)
    table.insert(0, "canonical_tensor_position", canonical)
    table.insert(0, "panel_position", canonical)
    table["universal_rank"] = ranking.universal_rank
    table["eligible_rank"] = pd.array(
        [value if value > 0 else pd.NA for value in ranking.eligible_rank],
        dtype="Int64",
    )
    table["eligible"] = ranking.eligible
    table["eligibility_reason"] = ranking.eligibility_reason
    table["is_priority_marker"] = ranking.marker_mask
    table["mean_log1p"] = ranking.mean_log1p
    table["variance_log1p"] = ranking.variance_log1p
    table["log_dispersion"] = ranking.log_dispersion
    table["normalized_dispersion"] = np.where(
        ranking.eligible, ranking.normalized_dispersion, np.nan
    )
    table["measured_cells"] = ranking.measured_cells
    table["detected_cells"] = ranking.detected_cells
    for name, positions in ladder.tiers.items():
        membership = np.zeros(ranking.n_genes, dtype=bool)
        membership[np.asarray(positions, dtype=np.int64)] = True
        table[f"in_tier_{name}"] = membership
    return table


def _selected_genes_table(
    ladder: PanelLadderSelection, gene_panel: Any, positions: np.ndarray
) -> pd.DataFrame:
    ranking = ladder.full_ranking
    table = gene_panel.iloc[positions][["gene_id", "gene_symbol"]].copy()
    table.insert(0, "panel_position", positions)
    table["mean_log1p"] = ranking.mean_log1p[positions]
    table["variance_log1p"] = ranking.variance_log1p[positions]
    table["normalized_dispersion"] = ranking.normalized_dispersion[positions]
    table["measured_cells"] = ranking.measured_cells[positions]
    table["detected_cells"] = ranking.detected_cells[positions]
    return table


def _npy_bytes(array: np.ndarray) -> bytes:
    buffer = io.BytesIO()
    np.save(buffer, np.asarray(array, dtype=np.int64), allow_pickle=False)
    return buffer.getvalue()


def _csv_bytes(table: pd.DataFrame) -> bytes:
    buffer = io.StringIO(newline="")
    table.to_csv(buffer, index=False)
    return buffer.getvalue().encode("utf-8")


def _legacy_paths(path: Path) -> tuple[Path, Path, Path]:
    if not path.is_dir():
        raise ValueError("exact legacy replay requires an artifact directory")
    positions = path / "gene_positions.npy"
    selected = path / "selected_genes.csv"
    manifest = path / "manifest.json"
    missing = [candidate.name for candidate in (positions, selected, manifest) if not candidate.is_file()]
    if missing:
        raise FileNotFoundError(
            f"legacy replay directory is incomplete ({', '.join(missing)}): {path}"
        )
    return positions, selected, manifest


def _selection_parameter_view(parameters: Mapping[str, object]) -> Mapping[str, object]:
    nested = parameters.get("selection")
    return nested if isinstance(nested, Mapping) else parameters


def _parameters_equal(key: str, left: object, right: object) -> bool:
    if key in {"recipes", "cells_per_recipe", "mean_bins", "seed"}:
        return int(left) == int(right)
    if key == "target_sum":
        return float(left) == float(right)
    return left == right


def _validate_selection_parameter_subset(
    actual_parameters: Mapping[str, object],
    expected_parameters: Mapping[str, object],
    *,
    error_prefix: str,
) -> dict[str, object]:
    actual = _selection_parameter_view(actual_parameters)
    expected = _selection_parameter_view(expected_parameters)
    validated: dict[str, object] = {}
    for key in REPLAY_PARAMETER_KEYS:
        if key not in actual or key not in expected:
            raise ValueError(f"{error_prefix} parameter is missing: {key}")
        if not _parameters_equal(key, actual[key], expected[key]):
            raise ValueError(f"{error_prefix} parameter differs: {key}")
        validated[key] = actual[key]
    return validated


def _validate_replay_parameters(
    *,
    legacy_parameters: Mapping[str, object],
    expected_parameters: Mapping[str, object],
    n_genes: int,
) -> dict[str, object]:
    try:
        validated = _validate_selection_parameter_subset(
            legacy_parameters,
            expected_parameters,
            error_prefix="legacy replay",
        )
    except (TypeError, ValueError) as error:
        raise ReplayMismatchError(str(error)) from error
    if int(legacy_parameters.get("n_genes", -1)) != int(n_genes):
        raise ReplayMismatchError("legacy replay n_genes parameter differs")
    validated["n_genes"] = int(n_genes)
    return validated


def build_exact_replay_proof(
    ladder: PanelLadderSelection,
    legacy_artifacts: Mapping[str, Path | str],
    *,
    gene_panel: Any | None = None,
    expected_parameters: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Prove requested tier files exactly reproduce legacy 2K/3K artifacts."""

    proof_tiers: dict[str, object] = {}
    if legacy_artifacts and expected_parameters is None:
        raise ValueError("exact legacy replay requires expected selection parameters")
    if legacy_artifacts and gene_panel is None:
        raise ValueError("exact legacy replay requires the canonical gene panel")
    for raw_name, raw_path in legacy_artifacts.items():
        name = str(raw_name).lower()
        if name not in ladder.tiers or name == "broad":
            raise ValueError(f"legacy replay requested for unknown finite tier: {name}")
        source = Path(raw_path).expanduser().resolve()
        positions_path, selected_path, manifest_path = _legacy_paths(source)
        expected = np.asarray(ladder.tiers[name], dtype=np.int64)
        actual = np.load(positions_path, allow_pickle=False)
        array_exact = actual.dtype == np.int64 and np.array_equal(actual, expected)
        file_exact = sha256_file(positions_path) == _sha256_bytes(_npy_bytes(expected))
        entry: dict[str, object] = {
            "legacy_positions_path": str(positions_path),
            "legacy_positions_sha256": sha256_file(positions_path),
            "generated_positions_sha256": _sha256_bytes(_npy_bytes(expected)),
            "positions_array_exact": bool(array_exact),
            "positions_file_sha256_exact": bool(file_exact),
        }
        assert gene_panel is not None
        selected_exact = selected_path.read_bytes() == _csv_bytes(
            _selected_genes_table(ladder, gene_panel, expected)
        )
        legacy_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if legacy_manifest.get("schema_version") != LEGACY_SELECTION_SCHEMA:
            raise ReplayMismatchError(f"legacy {name} manifest schema differs")
        if legacy_manifest.get("selection_scope") != "train_split_only":
            raise ReplayMismatchError(f"legacy {name} selection scope differs")
        if Path(str(legacy_manifest.get("positions_path", ""))).resolve() != positions_path:
            raise ReplayMismatchError(f"legacy {name} manifest positions path differs")
        if Path(str(legacy_manifest.get("selected_genes_path", ""))).resolve() != selected_path:
            raise ReplayMismatchError(f"legacy {name} manifest selected table path differs")
        if legacy_manifest.get("positions_sha256") != sha256_file(positions_path):
            raise ReplayMismatchError(
                f"legacy {name} manifest does not bind its positions file"
            )
        if legacy_manifest.get("selected_genes_sha256") != sha256_file(selected_path):
            raise ReplayMismatchError(
                f"legacy {name} manifest does not bind its selected-gene table"
            )
        if legacy_manifest.get("corpus_fingerprint") != ladder.corpus_fingerprint:
            raise ReplayMismatchError(f"legacy {name} corpus fingerprint differs")
        if int(legacy_manifest.get("n_selected_genes", -1)) != expected.size:
            raise ReplayMismatchError(f"legacy {name} selected-gene count differs")
        if int(legacy_manifest.get("sampled_cells", -1)) != ladder.sampled_cells:
            raise ReplayMismatchError(f"legacy {name} sampled-cell count differs")
        if int(legacy_manifest.get("sampled_recipes", -1)) != ladder.sampled_recipes:
            raise ReplayMismatchError(f"legacy {name} sampled-recipe count differs")
        legacy_parameters = legacy_manifest.get("parameters")
        if not isinstance(legacy_parameters, Mapping):
            raise ReplayMismatchError(f"legacy {name} selection parameters are missing")
        assert expected_parameters is not None
        validated_parameters = _validate_replay_parameters(
            legacy_parameters=legacy_parameters,
            expected_parameters=expected_parameters,
            n_genes=expected.size,
        )
        implementation_path = Path(str(legacy_manifest.get("implementation", ""))).resolve()
        if (
            not implementation_path.is_file()
            or legacy_manifest.get("implementation_sha256")
            != sha256_file(implementation_path)
        ):
            raise ReplayMismatchError(f"legacy {name} implementation binding differs")
        entry.update(
            {
                "legacy_selected_genes_path": str(selected_path),
                "legacy_selected_genes_sha256": sha256_file(selected_path),
                "selected_genes_file_exact": bool(selected_exact),
                "legacy_manifest_path": str(manifest_path),
                "legacy_manifest_sha256": sha256_file(manifest_path),
                "manifest_schema_verified": True,
                "manifest_hashes_verified": True,
                "selection_parameters_verified": validated_parameters,
                "legacy_implementation_verified": True,
            }
        )
        exact = array_exact and file_exact and selected_exact
        entry["exact"] = bool(exact)
        if not exact:
            raise ReplayMismatchError(f"legacy {name} replay is not byte-for-byte exact")
        proof_tiers[name] = entry
    requested = bool(legacy_artifacts)
    return {
        "schema_version": REPLAY_PROOF_SCHEMA,
        "requested": requested,
        "status": "pass" if requested else "not_requested",
        "all_exact": True if requested else None,
        "tiers": proof_tiers,
    }


def _selection_code_bindings(
    model7_implementation: Mapping[str, object] | None = None,
) -> dict[str, object]:
    implementation = Path(__file__).resolve()
    legacy_kernel = Path(rank_hvg_statistics.__code__.co_filename).resolve()
    files = {
        "panel_ladder": {
            "path": str(implementation),
            "sha256": sha256_file(implementation),
        },
        "legacy_ranking_kernel": {
            "path": str(legacy_kernel),
            "sha256": sha256_file(legacy_kernel),
        },
    }
    payload: dict[str, object] = {"files": files}
    if model7_implementation is not None:
        _validate_model7_implementation(model7_implementation)
        payload["model7_data_e2"] = dict(model7_implementation)
    return {**payload, "bundle_sha256": canonical_sha256(payload)}


def _legacy_parameters(parameters: Mapping[str, object], n_genes: int) -> dict[str, object]:
    source: Mapping[str, object] = parameters
    nested = parameters.get("selection")
    if isinstance(nested, Mapping):
        source = nested
    result = {str(key): value for key, value in source.items() if key != "n_genes"}
    result["n_genes"] = int(n_genes)
    return result


def _validate_production_ladder(
    ladder: PanelLadderSelection, gene_panel: Any
) -> tuple[str, str, str, str] | None:
    if ladder.provenance is None:
        if (
            ladder.sufficient_statistics is not None
            or ladder.model7_implementation is not None
            or ladder.leakage_audit.get("production_ready") is True
        ):
            raise ValueError("incomplete production binding on panel ladder")
        return None
    if ladder.sufficient_statistics is None or ladder.model7_implementation is None:
        raise ValueError("production ladder lacks statistics or Model7 binding")
    recipe_sha256, reference_sha256 = _validate_provenance(
        ladder.provenance,
        sampled_recipes=ladder.sampled_recipes,
        sampled_cells=ladder.sampled_cells,
        corpus_fingerprint=ladder.corpus_fingerprint,
        leakage_audit=ladder.leakage_audit,
    )
    mean_bins = int(ladder.sufficient_statistics.binding.get("mean_bins", -1))
    statistics_sha256 = _validate_sufficient_statistics_binding(
        ladder.sufficient_statistics,
        sampled_cells=ladder.sampled_cells,
        sampled_recipes=ladder.sampled_recipes,
        corpus_fingerprint=ladder.corpus_fingerprint,
        min_measured_cells=ladder.min_measured_cells,
        min_detected_cells=ladder.min_detected_cells,
        mean_bins=mean_bins,
    )
    if ladder.leakage_audit.get("sufficient_statistics_sha256") != statistics_sha256:
        raise ValueError("audit sufficient-statistics hash differs")
    model7_sha256 = _validate_model7_implementation(ladder.model7_implementation)
    if ladder.leakage_audit.get("model7_implementation_sha256") != model7_sha256:
        raise ValueError("audit Model7 implementation hash differs")
    recomputed = rank_hvg_full_statistics(
        value_sum=ladder.sufficient_statistics.value_sum,
        value_square_sum=ladder.sufficient_statistics.value_square_sum,
        measured_cells=ladder.sufficient_statistics.measured_cells,
        detected_cells=ladder.sufficient_statistics.detected_cells,
        marker_mask=_marker_mask(gene_panel),
        min_measured_cells=ladder.min_measured_cells,
        min_detected_cells=ladder.min_detected_cells,
        mean_bins=mean_bins,
    )
    for name in (
        "ranked_positions",
        "universal_rank",
        "eligible_rank",
        "eligible",
        "mean_log1p",
        "variance_log1p",
        "log_dispersion",
        "normalized_dispersion",
        "measured_cells",
        "detected_cells",
        "marker_mask",
    ):
        actual = np.asarray(getattr(ladder.full_ranking, name))
        expected = np.asarray(getattr(recomputed, name))
        if not np.array_equal(actual, expected, equal_nan=True):
            raise ValueError(f"published ranking differs from bound statistics: {name}")
    if ladder.full_ranking.eligibility_reason != recomputed.eligibility_reason:
        raise ValueError("published eligibility reasons differ from bound statistics")
    eligible_order = recomputed.ranked_positions[: recomputed.n_eligible]
    for name, positions_value in ladder.tiers.items():
        prefix = recomputed.n_eligible if name == "broad" else len(positions_value)
        expected = np.sort(eligible_order[:prefix]).astype(np.int64, copy=False)
        if not np.array_equal(np.asarray(positions_value), expected):
            raise ValueError(f"tier {name} differs from bound universal ranking")
    return recipe_sha256, reference_sha256, statistics_sha256, model7_sha256


def publish_panel_ladder(
    output_dir: Path | str,
    ladder: PanelLadderSelection,
    gene_panel: Any,
    *,
    parameters: Mapping[str, object],
    legacy_replay_artifacts: Mapping[str, Path | str] | None = None,
) -> dict[str, object]:
    """Publish a hash-bound ladder once; an existing target is never modified."""

    output = Path(output_dir).expanduser().resolve()
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"refusing to overwrite panel ladder: {output}")
    if gene_panel_identity_sha256(gene_panel) != ladder.gene_panel_sha256:
        raise ValueError("gene panel changed after the ladder was built")
    _validate_leakage_audit(ladder.leakage_audit)
    production_bindings = _validate_production_ladder(ladder, gene_panel)
    provenance_hashes = (
        None
        if production_bindings is None
        else (production_bindings[0], production_bindings[1])
    )
    statistics_sha256 = (
        None if production_bindings is None else production_bindings[2]
    )
    model7_sha256 = None if production_bindings is None else production_bindings[3]
    output.parent.mkdir(parents=True, exist_ok=True)

    parameter_payload = dict(parameters)
    if ladder.sufficient_statistics is not None:
        collected_parameters = ladder.sufficient_statistics.binding.get(
            "collection_parameters"
        )
        if not isinstance(collected_parameters, Mapping):
            raise ValueError("bound sufficient statistics lack collection parameters")
        _validate_selection_parameter_subset(
            parameter_payload,
            collected_parameters,
            error_prefix="published selection",
        )
    parameters_sha256 = canonical_sha256(parameter_payload)
    code = _selection_code_bindings(ladder.model7_implementation)
    code_files = code["files"]
    assert isinstance(code_files, Mapping)
    implementation = code_files["panel_ladder"]
    assert isinstance(implementation, Mapping)
    audit_file_sha256 = _sha256_bytes(_json_file_bytes(ladder.leakage_audit))
    provenance_binding: dict[str, object] = {
        "ordered_train_recipes_sha256": (
            None if provenance_hashes is None else provenance_hashes[0]
        ),
        "sampled_cell_references_sha256": (
            None if provenance_hashes is None else provenance_hashes[1]
        ),
        "leakage_audit_sha256": audit_file_sha256,
        "sufficient_statistics_sha256": statistics_sha256,
        "model7_implementation_sha256": model7_sha256,
    }
    binding_core = {
        "corpus_fingerprint": ladder.corpus_fingerprint,
        "gene_panel_sha256": ladder.gene_panel_sha256,
        "selection_code_bundle_sha256": code["bundle_sha256"],
        "parameters_sha256": parameters_sha256,
        "provenance": provenance_binding,
    }
    binding_sha256 = canonical_sha256(binding_core)

    stage = Path(tempfile.mkdtemp(prefix=f".{output.name}.staging-", dir=output.parent))
    try:
        ranked_path = stage / "ranked_gene_positions.npy"
        np.save(ranked_path, ladder.full_ranking.ranked_positions, allow_pickle=False)
        statistics_path = stage / "full_gene_statistics.csv"
        full_gene_statistics_table(ladder, gene_panel).to_csv(statistics_path, index=False)
        audit_path = stage / "leakage_audit.json"
        _write_json(audit_path, ladder.leakage_audit)
        if sha256_file(audit_path) != audit_file_sha256:
            raise RuntimeError("leakage audit serialization is not deterministic")

        provenance_manifest: dict[str, object]
        if ladder.provenance is None:
            provenance_manifest = {
                "status": "incomplete",
                "ordered_train_recipes_path": None,
                "ordered_train_recipes_sha256": None,
                "ordered_train_recipes_rows": 0,
                "sampled_cell_references_path": None,
                "sampled_cell_references_sha256": None,
                "sampled_cell_references_rows": 0,
            }
        else:
            assert provenance_hashes is not None
            recipes_path = stage / "ordered_train_recipes.csv"
            references_path = stage / "sampled_cell_references.csv"
            recipes_path.write_bytes(
                _csv_bytes(ladder.provenance.ordered_train_recipes)
            )
            references_path.write_bytes(
                _csv_bytes(ladder.provenance.sampled_cell_references)
            )
            if sha256_file(recipes_path) != provenance_hashes[0]:
                raise RuntimeError("ordered train-recipe serialization drifted")
            if sha256_file(references_path) != provenance_hashes[1]:
                raise RuntimeError("sampled-cell reference serialization drifted")
            provenance_manifest = {
                "status": "complete",
                "ordered_train_recipes_path": str(output / recipes_path.name),
                "ordered_train_recipes_sha256": provenance_hashes[0],
                "ordered_train_recipes_rows": len(
                    ladder.provenance.ordered_train_recipes
                ),
                "ordered_train_recipes_columns": list(ORDERED_RECIPE_COLUMNS),
                "sampled_cell_references_path": str(output / references_path.name),
                "sampled_cell_references_sha256": provenance_hashes[1],
                "sampled_cell_references_rows": len(
                    ladder.provenance.sampled_cell_references
                ),
                "sampled_cell_references_columns": list(
                    SAMPLED_CELL_REFERENCE_COLUMNS
                ),
                "source_coordinate_contract": (
                    "corpus_fingerprint + shard_id + strict_row_index"
                ),
                "selection_occurrence_contract": (
                    "recipe_id + ref_position_in_recipe; ordered by sample_order"
                ),
            }

        sufficient_statistics_manifest: dict[str, object]
        if ladder.sufficient_statistics is None:
            sufficient_statistics_manifest = {
                "status": "incomplete",
                "binding_path": None,
                "binding_file_sha256": None,
                "binding_sha256": None,
                "arrays": {},
            }
        else:
            assert statistics_sha256 is not None
            sufficient_dir = stage / "sufficient_statistics"
            sufficient_dir.mkdir(parents=True, exist_ok=False)
            binding_path = sufficient_dir / "binding.json"
            _write_json(binding_path, ladder.sufficient_statistics.binding)
            array_files: dict[str, object] = {}
            for name, array in (
                ("value_sum", ladder.sufficient_statistics.value_sum),
                ("value_square_sum", ladder.sufficient_statistics.value_square_sum),
                ("measured_cells", ladder.sufficient_statistics.measured_cells),
                ("detected_cells", ladder.sufficient_statistics.detected_cells),
            ):
                path = sufficient_dir / f"{name}.npy"
                np.save(path, array, allow_pickle=False)
                array_files[name] = {
                    "path": str(output / "sufficient_statistics" / path.name),
                    "sha256": sha256_file(path),
                    "dtype": np.asarray(array).dtype.name,
                    "shape": [int(np.asarray(array).size)],
                }
            sufficient_statistics_manifest = {
                "status": "complete",
                "binding_path": str(
                    output / "sufficient_statistics" / "binding.json"
                ),
                "binding_file_sha256": sha256_file(binding_path),
                "binding_sha256": statistics_sha256,
                "arrays": array_files,
            }

        replay_proof = build_exact_replay_proof(
            ladder,
            {} if legacy_replay_artifacts is None else legacy_replay_artifacts,
            gene_panel=gene_panel,
            expected_parameters=parameter_payload,
        )
        replay_path = stage / "replay_proof.json"
        _write_json(replay_path, replay_proof)

        tier_entries: dict[str, object] = {}
        finite_names = [name for name in ladder.tiers if name != "broad"]
        for name, positions_value in ladder.tiers.items():
            positions = np.asarray(positions_value, dtype=np.int64)
            tier_dir = stage / "tiers" / name
            tier_dir.mkdir(parents=True, exist_ok=False)
            positions_path = tier_dir / "gene_positions.npy"
            np.save(positions_path, positions, allow_pickle=False)
            selected_path = tier_dir / "selected_genes.csv"
            _selected_genes_table(ladder, gene_panel, positions).to_csv(
                selected_path, index=False
            )
            final_tier_dir = output / "tiers" / name
            entry: dict[str, object] = {
                "n_genes": int(positions.size),
                "rank_prefix_length": int(positions.size),
                "tensor_order": "canonical_gene_panel_order",
                "gene_positions_path": str(final_tier_dir / "gene_positions.npy"),
                "gene_positions_sha256": sha256_file(positions_path),
                "selected_genes_path": str(final_tier_dir / "selected_genes.csv"),
                "selected_genes_sha256": sha256_file(selected_path),
            }
            if name == "broad":
                entry["policy"] = "all_eligible_genes"
            else:
                legacy_manifest_path = tier_dir / "legacy_selection_manifest.v1.json"
                legacy_manifest = {
                    "schema_version": LEGACY_SELECTION_SCHEMA,
                    "selection_scope": "train_split_only",
                    "corpus_fingerprint": ladder.corpus_fingerprint,
                    "sampled_cells": ladder.sampled_cells,
                    "sampled_recipes": ladder.sampled_recipes,
                    "n_selected_genes": int(positions.size),
                    "positions_path": str(final_tier_dir / "gene_positions.npy"),
                    "positions_sha256": sha256_file(positions_path),
                    "selected_genes_path": str(final_tier_dir / "selected_genes.csv"),
                    "selected_genes_sha256": sha256_file(selected_path),
                    "implementation": str(implementation["path"]),
                    "implementation_sha256": str(implementation["sha256"]),
                    "algorithm": (
                        "universal train-only measurement-aware HVG ranking; "
                        "this tier is a rank prefix emitted in canonical tensor order"
                    ),
                    "parameters": _legacy_parameters(parameter_payload, positions.size),
                    "parent_panel_ladder": {
                        "schema_version": PANEL_LADDER_SCHEMA,
                        "manifest_path": str(output / "manifest.json"),
                        "binding_sha256": binding_sha256,
                        "tier": name,
                    },
                }
                _write_json(legacy_manifest_path, legacy_manifest)
                entry["legacy_selection_manifest_path"] = str(
                    final_tier_dir / "legacy_selection_manifest.v1.json"
                )
                entry["legacy_selection_manifest_sha256"] = sha256_file(
                    legacy_manifest_path
                )
            tier_entries[name] = entry

        manifest: dict[str, object] = {
            "schema_version": PANEL_LADDER_SCHEMA,
            "selection_scope": "train_split_only",
            "production_ready": bool(ladder.leakage_audit.get("production_ready")),
            "corpus_fingerprint": ladder.corpus_fingerprint,
            "gene_panel_sha256": ladder.gene_panel_sha256,
            "sampled_cells": ladder.sampled_cells,
            "sampled_recipes": ladder.sampled_recipes,
            "n_full_genes": ladder.full_ranking.n_genes,
            "n_eligible_genes": ladder.full_ranking.n_eligible,
            "n_rejected_genes": (
                ladder.full_ranking.n_genes - ladder.full_ranking.n_eligible
            ),
            "ranking_contract": {
                "tier_independent": True,
                "ranked_population": "all_canonical_genes_eligible_first",
                "rank_base": 1,
                "tensor_order": "canonical_gene_panel_order",
                "finite_tier_policy": "universal_eligible_rank_prefix",
                "broad_tier_policy": "all_eligible_genes",
                "min_measured_cells": ladder.min_measured_cells,
                "min_detected_cells": ladder.min_detected_cells,
            },
            "ranked_gene_positions_path": str(output / "ranked_gene_positions.npy"),
            "ranked_gene_positions_sha256": sha256_file(ranked_path),
            "full_gene_statistics_path": str(output / "full_gene_statistics.csv"),
            "full_gene_statistics_sha256": sha256_file(statistics_path),
            "leakage_audit_path": str(output / "leakage_audit.json"),
            "leakage_audit_sha256": sha256_file(audit_path),
            "provenance": provenance_manifest,
            "sufficient_statistics": sufficient_statistics_manifest,
            "replay_proof_path": str(output / "replay_proof.json"),
            "replay_proof_sha256": sha256_file(replay_path),
            "tiers_are_nested": True,
            "finite_tiers": finite_names,
            "tiers": tier_entries,
            "parameters": parameter_payload,
            "parameters_sha256": parameters_sha256,
            "selection_code": code,
            "bindings": {**binding_core, "binding_sha256": binding_sha256},
        }
        _write_json(stage / "manifest.json", manifest)
        if output.exists() or output.is_symlink():
            raise FileExistsError(f"refusing to overwrite panel ladder: {output}")
        stage.rename(output)
        return manifest
    except BaseException:
        if stage.exists():
            shutil.rmtree(stage)
        raise


# A concise alias for callers that used the v1 ``save_*`` naming convention.
save_panel_ladder = publish_panel_ladder


__all__ = [
    "BoundSufficientStatistics",
    "DEFAULT_TIER_SIZES",
    "FullGeneRanking",
    "LEAKAGE_AUDIT_SCHEMA",
    "MODEL7_IMPLEMENTATION_SCHEMA",
    "ORDERED_RECIPE_COLUMNS",
    "PANEL_LADDER_SCHEMA",
    "PanelLadderSelection",
    "REPLAY_PROOF_SCHEMA",
    "ReplayMismatchError",
    "SAMPLED_CELL_REFERENCE_COLUMNS",
    "SUFFICIENT_STATISTICS_SCHEMA",
    "TrainPanelStatistics",
    "TrainSelectionProvenance",
    "build_exact_replay_proof",
    "build_panel_ladder",
    "build_production_panel_ladder",
    "build_sufficient_statistics_binding",
    "bind_model7_implementation",
    "canonical_sha256",
    "collect_train_panel_statistics",
    "full_gene_statistics_table",
    "gene_panel_identity_sha256",
    "panel_ladder_from_hvg_selection",
    "publish_panel_ladder",
    "rank_hvg_full_statistics",
    "save_panel_ladder",
    "select_train_panel_ladder",
    "sha256_file",
    "train_only_leakage_audit",
]
