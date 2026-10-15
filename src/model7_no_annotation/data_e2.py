"""Strict, annotation-independent data path for Model 7 E2-full.

This module deliberately does not import the historical Method 1 recipe
loader.  E2 recipes point at row-aligned CSR Zarr shards and a five-column
provenance projection under a strict source root.  Only dataset, donor,
sample, and positional row identifiers are ever requested from source obs
Parquet files; cell annotations and their masks are neither required nor
consulted.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass, field
import hashlib
import json
import multiprocessing as mp
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import sparse
from torch.utils.data import IterableDataset, get_worker_info

from .data import (
    AnnotationFreeConditioner,
    AnnotationFreeRecipeBlock,
    audit_annotation_free_batch,
    audit_annotation_free_names,
)


MODEL7_ROOT = Path(__file__).resolve().parents[2]
IMMUN_ROOT = MODEL7_ROOT.parent
DEFAULT_SOURCE_ROOT = MODEL7_ROOT / "data" / "e2_full_source"
DEFAULT_BASE_SOURCE_ROOT = IMMUN_ROOT / "single-cell-data-block-blood"
DEFAULT_INDEX_ROOT = MODEL7_ROOT / "data" / "e2_full_index"
DEFAULT_GENE_PANEL_PATH = DEFAULT_SOURCE_ROOT / "metadata" / "var.parquet"

E2_INDEX_SCHEMA = "model7.e2_full.index.v1"
E2_SOURCE_SCHEMA = "model7.e2_strict_source.v1"
E2_SAMPLE_KEYS = frozenset(
    {"x_log", "condition", "block_index", "block_id", "recipe_k"}
)
E2_PLATFORMS = ("10x", "Parse", "ScaleBio", "Seq-Well")
E2_GLOBAL_VALIDATION_CONTRACT = {
    "split": "val",
    "scope": "heldout_dataset_donor",
    "platforms": ["10x", "Parse", "Seq-Well"],
    "excludes_diagnostic_platforms": ["ScaleBio"],
    "used_for_best_checkpoint": True,
}
E2_OBS_READ_COLUMNS = (
    "dataset_id",
    "donor_id",
    "sample_id",
    "row_index",
)
E2_SHARD_READ_COLUMNS = (
    "dataset_id",
    "shard_id",
    "n_cells",
    "n_genes",
    "count_zarr",
    "obs_part",
)
E2_RECIPE_COLUMNS = (
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
E2_SAMPLE_POOL_SCHEMA = {
    "sample_key": "string",
    "shard_id": "string",
    "row_indices": "list<element: int32>",
}
E2_CORPUS_CORE_KEYS = frozenset(
    {
        "source_fingerprint",
        "source_shards_sha256",
        "build_config_signature",
        "gene_panel_sha256",
        "recipe_ids_sha256",
        "sample_pools_sha256",
        "sample_index_sha256",
        "donor_splits_sha256",
        "index_source_shards_sha256",
        "recipe_parts_manifest_sha256",
        "recipe_parts_content_sha256",
    }
)

_MISSING_VALUES = frozenset(
    {"", "na", "n/a", "nan", "none", "null", "unknown", "unassigned"}
)
_MISSING_DONOR = "__missing_donor__"
_MISSING_SAMPLE_PREFIX = "__missing_sample__"


class E2IndexContractError(ValueError):
    """Raised when an E2 source/index violates the strict data contract."""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_digest(payload: object) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _verify_file_fingerprint(
    path: Path, expected: object, *, description: str
) -> str:
    text = str(expected).lower()
    if len(text) != 64 or any(char not in "0123456789abcdef" for char in text):
        raise E2IndexContractError(f"{description} fingerprint is missing or invalid")
    if not path.is_file():
        raise FileNotFoundError(f"missing {description}: {path}")
    if _sha256_file(path) != text:
        raise E2IndexContractError(f"{description} fingerprint differs")
    return text


def _stable_hash(parts: Iterable[object], seed: int = 0) -> int:
    text = "||".join(str(part) for part in parts) + f"||seed={int(seed)}"
    return int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:16], 16)


def _sample_key(dataset_id: object, donor_id: object, sample_id: object) -> str:
    text = "\x1f".join(str(value) for value in (dataset_id, donor_id, sample_id))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:24]


def platform_for_dataset(dataset_id: object) -> str:
    """Map the four known platform-specific dataset prefixes."""

    text = str(dataset_id)
    if text.startswith("304_"):
        return "Parse"
    if text.startswith("010_"):
        return "ScaleBio"
    if text.startswith("081_"):
        return "Seq-Well"
    return "10x"


def _normalize_identifier(value: object) -> str:
    text = str(value).strip()
    return "" if text.lower() in _MISSING_VALUES else text


def _normalize_obs_groups(frame: pd.DataFrame, *, shard_id: str) -> pd.DataFrame:
    required = set(E2_OBS_READ_COLUMNS)
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise E2IndexContractError(f"strict obs projection is missing columns: {missing}")
    out = frame.loc[:, list(E2_OBS_READ_COLUMNS)].copy()
    out["dataset_id"] = out["dataset_id"].map(_normalize_identifier)
    out["donor_id"] = out["donor_id"].map(_normalize_identifier)
    out["sample_id"] = out["sample_id"].map(_normalize_identifier)
    out.loc[out["donor_id"].eq(""), "donor_id"] = _MISSING_DONOR
    missing_sample = out["sample_id"].eq("")
    if bool(missing_sample.any()):
        out.loc[missing_sample, "sample_id"] = f"{_MISSING_SAMPLE_PREFIX}:{shard_id}"
    row_index = pd.to_numeric(out["row_index"], errors="raise").to_numpy(
        dtype=np.int64, copy=False
    )
    if not np.array_equal(row_index, np.arange(len(out), dtype=np.int64)):
        raise E2IndexContractError(
            f"strict obs row_index must equal CSR row position in {shard_id}"
        )
    return out


def _contained_path(root: Path, relative: Path, *, description: str) -> Path:
    if relative.is_absolute():
        raise E2IndexContractError(f"{description} must be root-relative: {relative}")
    lexical = (root / relative).absolute()
    if root.absolute() not in (lexical, *lexical.parents):
        raise E2IndexContractError(f"{description} escapes its pinned root: {relative}")
    resolved = (root / relative).resolve()
    try:
        resolved.relative_to(root.resolve())
    except ValueError as error:
        raise E2IndexContractError(
            f"{description} escapes strict source root: {relative}"
        ) from error
    return resolved


def _resolve_source_uri(
    raw_uri: object,
    *,
    strict_root: Path,
    base_root: Path,
    description: str,
) -> Path:
    text = str(raw_uri)
    for scheme, root in (("strict://", strict_root), ("base://", base_root)):
        if text.startswith(scheme):
            relative_text = text[len(scheme) :]
            if not relative_text:
                raise E2IndexContractError(f"empty {scheme} URI for {description}")
            return _contained_path(root, Path(relative_text), description=description)
    raise E2IndexContractError(
        f"{description} must use strict:// or base://, got {text!r}"
    )


def _read_json(path: Path, *, description: str) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"missing {description}: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise E2IndexContractError(f"{description} must be a JSON object: {path}")
    return payload


def _published_source_fingerprint(
    payload: Mapping[str, Any], *, description: str
) -> str:
    for key in (
        "source_fingerprint_sha256",
        "source_fingerprint",
        "corpus_fingerprint",
        "fingerprint",
    ):
        text = str(payload.get(key, "")).lower()
        if len(text) == 64 and all(char in "0123456789abcdef" for char in text):
            return text
    raise E2IndexContractError(f"{description} does not publish a SHA256 fingerprint")


def _validate_strict_source_metadata(
    manifest: Mapping[str, Any],
    complete: Mapping[str, Any],
    *,
    manifest_path: Path,
    source_root: Path,
    base_root: Path,
    gene_panel_path: Path,
    expected_datasets: int,
    expected_cells: int,
    provenance_roots: Mapping[str, str] | None = None,
) -> str:
    """Validate the immutable strict-source sentinel shared by builder/loader."""

    source_root = source_root.resolve()
    base_root = base_root.resolve()
    gene_panel_path = gene_panel_path.resolve()
    provenance = provenance_roots or {}
    pinned_source = Path(provenance.get("source_root", str(source_root))).resolve()
    pinned_base = Path(provenance.get("base_source_root", str(base_root))).resolve()
    pinned_panel = Path(provenance.get("gene_panel_path", str(gene_panel_path))).resolve()
    if manifest.get("schema_version") != E2_SOURCE_SCHEMA:
        raise E2IndexContractError("unsupported strict E2 source schema")
    if complete.get("schema_version") != E2_SOURCE_SCHEMA:
        raise E2IndexContractError("strict E2 completion schema differs")
    if manifest.get("status") != "complete" or complete.get("status") != "complete":
        raise E2IndexContractError("strict E2 source is not marked complete")
    if Path(str(manifest.get("strict_source_root", ""))).resolve() != pinned_source:
        raise E2IndexContractError("strict source root differs from its manifest")
    for payload, description in (
        (manifest, "strict source manifest"),
        (complete, "strict source completion marker"),
    ):
        if Path(str(payload.get("base_source_root", ""))).resolve() != pinned_base:
            raise E2IndexContractError(
                f"{description} pinned base_source_root differs from configuration"
            )
        if int(payload.get("n_datasets", -1)) != int(expected_datasets):
            raise E2IndexContractError(f"{description} dataset count differs")
        if int(payload.get("logical_cells", -1)) != int(expected_cells):
            raise E2IndexContractError(f"{description} logical cell count differs")
    if manifest.get("recipe_pool_annotation_free") is not True:
        raise E2IndexContractError(
            "strict source does not certify recipe_pool_annotation_free=true"
        )
    for key in (
        "annotation_columns_referenced_by_builder",
        "annotation_columns_used_for_selection",
        "annotation_columns_used_for_qc",
    ):
        if manifest.get(key) != []:
            raise E2IndexContractError(f"strict source requires {key}=[]")
    if manifest.get("tabula_ref_excluded") is not True:
        raise E2IndexContractError("strict source does not certify REF exclusion")
    panel_contract = manifest.get("gene_panel")
    if not isinstance(panel_contract, Mapping):
        raise E2IndexContractError("strict source lacks its gene-panel contract")
    if Path(str(panel_contract.get("path", ""))).resolve() != pinned_panel:
        raise E2IndexContractError("strict source gene-panel path differs")
    panel_sha256 = _sha256_file(gene_panel_path)
    for observed in (
        panel_contract.get("sha256"),
        manifest.get("gene_panel_sha256"),
        complete.get("gene_panel_sha256"),
    ):
        if str(observed) != panel_sha256:
            raise E2IndexContractError("strict source gene-panel fingerprint differs")
    if str(complete.get("dataset_manifest_sha256", "")) != _sha256_file(
        manifest_path
    ):
        raise E2IndexContractError("strict source manifest fingerprint differs")
    shard_path = source_root / "metadata" / "sc_shards.parquet"
    if str(manifest.get("shard_manifest_sha256", "")) != _sha256_file(shard_path):
        raise E2IndexContractError("strict source shard-manifest fingerprint differs")
    fingerprint = _published_source_fingerprint(
        manifest, description="strict source manifest"
    )
    if (
        _published_source_fingerprint(
            complete, description="strict source completion marker"
        )
        != fingerprint
    ):
        raise E2IndexContractError("strict source fingerprint/marker mismatch")
    return fingerprint


@dataclass(frozen=True)
class E2FullDataConfig:
    """Pinned E2-full source, recipe, split, and streaming settings."""

    project_root: Path = IMMUN_ROOT / "model" / "vae_bulk2sc"
    source_root: Path = DEFAULT_SOURCE_ROOT
    base_source_root: Path = DEFAULT_BASE_SOURCE_ROOT
    index_root: Path = DEFAULT_INDEX_ROOT
    gene_panel_path: Path = DEFAULT_GENE_PANEL_PATH
    train_split: str = "train"
    validation_split: str = "val"
    train_recipes_per_epoch: int = 8192
    validation_recipes: int = 192
    validation_recipes_per_platform: int = 64
    fit_blocks: int = 2048
    train_cells_per_block: int = 512
    validation_cells_per_block: int = 64
    seed: int = 20260627
    condition_mode: str = "bio_bulk_depth"
    sampling_policy: str = "random"
    platform_sampling_policy: str = "sqrt_cells"
    platforms: tuple[str, ...] = E2_PLATFORMS
    k_values: tuple[int, ...] = (250, 500, 1000, 2000, 4000)
    k_weights: tuple[int, ...] = (1, 2, 3, 4, 1)
    split_seed: int = 20260523
    split_unit: str = "dataset_donor"
    sample_fallback_for_missing_donor: bool = True
    split_fractions: tuple[float, float, float] = (0.8, 0.1, 0.1)
    excluded_dataset_ids: tuple[str, ...] = ("REF-001_Tabula_Sapiens",)
    train_recipe_pool_size: int = 131_072
    expected_datasets: int = 22
    expected_cells: int = 41_395_184
    progress: bool = True
    relocation_manifest: Path | None = None
    recipe_pool_annotation_free: bool = field(default=True, init=False)

    def __post_init__(self) -> None:
        for name in (
            "project_root",
            "source_root",
            "base_source_root",
            "index_root",
            "gene_panel_path",
        ):
            object.__setattr__(self, name, Path(getattr(self, name)).expanduser())
        object.__setattr__(self, "platforms", tuple(str(v) for v in self.platforms))
        object.__setattr__(self, "k_values", tuple(int(v) for v in self.k_values))
        object.__setattr__(self, "k_weights", tuple(int(v) for v in self.k_weights))
        object.__setattr__(
            self,
            "excluded_dataset_ids",
            tuple(str(value) for value in self.excluded_dataset_ids),
        )
        fractions: object = self.split_fractions
        if isinstance(fractions, Mapping):
            fractions = tuple(float(fractions[name]) for name in ("train", "val", "test"))
        object.__setattr__(
            self, "split_fractions", tuple(float(v) for v in fractions)  # type: ignore[arg-type]
        )
        if self.condition_mode != "bio_bulk_depth":
            raise ValueError("E2-full requires condition_mode='bio_bulk_depth'")
        if self.sampling_policy != "random":
            raise ValueError("E2-full requires sampling_policy='random'")
        if self.platform_sampling_policy != "sqrt_cells":
            raise ValueError("E2-full requires platform_sampling_policy='sqrt_cells'")
        if self.platforms != E2_PLATFORMS:
            raise ValueError(f"E2-full platforms must be exactly {E2_PLATFORMS}")
        if self.split_unit != "dataset_donor":
            raise ValueError("E2-full requires split_unit='dataset_donor'")
        if self.sample_fallback_for_missing_donor is not True:
            raise ValueError("E2-full requires sample fallback for missing donor IDs")
        if self.excluded_dataset_ids != ("REF-001_Tabula_Sapiens",):
            raise ValueError("E2-full must exclude exactly REF-001_Tabula_Sapiens")
        if not self.train_split or not self.validation_split:
            raise ValueError("train and validation split names must be non-empty")
        if self.train_split == self.validation_split:
            raise ValueError("train_split and validation_split must differ")
        positive_fields = (
            "train_recipes_per_epoch",
            "validation_recipes",
            "validation_recipes_per_platform",
            "fit_blocks",
            "train_cells_per_block",
            "validation_cells_per_block",
            "train_recipe_pool_size",
            "expected_datasets",
            "expected_cells",
        )
        for name in positive_fields:
            if int(getattr(self, name)) <= 0:
                raise ValueError(f"{name} must be positive")
        if len(self.k_values) != len(self.k_weights) or not self.k_values:
            raise ValueError("k_values and k_weights must be non-empty and aligned")
        if any(k <= 0 for k in self.k_values) or any(w <= 0 for w in self.k_weights):
            raise ValueError("k_values and k_weights must contain positive integers")
        if len(self.split_fractions) != 3 or any(v < 0 for v in self.split_fractions):
            raise ValueError("split_fractions must contain non-negative train/val/test values")
        if not np.isclose(sum(self.split_fractions), 1.0):
            raise ValueError("split_fractions must sum to one")
        if self.train_recipe_pool_size < 4 * self.train_recipes_per_epoch:
            raise ValueError(
                "train_recipe_pool_size must support four disjoint recipe epochs"
            )
        if self.train_recipe_pool_size % self.train_recipes_per_epoch:
            raise ValueError(
                "train_recipe_pool_size must be divisible by train_recipes_per_epoch"
            )

    def audit_metadata(self) -> dict[str, object]:
        return {
            "experiment_stage": "E2-full",
            "model_batch_annotation_free": True,
            "recipe_pool_annotation_free": True,
            "source_obs_columns_read": list(E2_OBS_READ_COLUMNS),
            "sampling_policy": self.sampling_policy,
            "platform_sampling_policy": self.platform_sampling_policy,
            "split_unit": "dataset_donor",
            "split_unit_detail": (
                "dataset_id,donor_id; sample_id fallback when donor_id is missing"
            ),
            "platforms": list(self.platforms),
            "k_values": list(self.k_values),
            "k_weights": list(self.k_weights),
            "split_seed": int(self.split_seed),
            "excluded_dataset_ids": list(self.excluded_dataset_ids),
            "recipes_per_epoch": int(self.train_recipes_per_epoch),
            "train_recipe_pool_size": int(self.train_recipe_pool_size),
            "expected_datasets": int(self.expected_datasets),
            "expected_cells": int(self.expected_cells),
            "base_source_root": str(self.base_source_root),
            "scale_bio_heldout_available": False,
        }


def _open_csr_zarr(path: Path) -> Any:
    try:
        import zarr
    except ImportError as error:  # pragma: no cover - environment contract
        raise RuntimeError("zarr is required for the E2-full sparse source") from error
    group = zarr.open_group(str(path), mode="r")
    missing = [name for name in ("data", "indices", "indptr") if name not in group]
    if missing:
        raise E2IndexContractError(f"missing CSR arrays in {path}: {missing}")
    return group


def _csr_rows(
    group: Any,
    rows: np.ndarray,
    n_cols: int,
    *,
    dtype: np.dtype[Any] = np.dtype(np.float32),
) -> sparse.csr_matrix:
    rows = np.asarray(rows, dtype=np.int64)
    if rows.ndim != 1:
        raise ValueError("CSR row indices must be one-dimensional")
    if rows.size == 0:
        return sparse.csr_matrix((0, int(n_cols)), dtype=dtype)
    if np.any(rows < 0):
        raise ValueError("CSR row indices must be non-negative")
    sort_order = np.argsort(rows, kind="mergesort")
    sorted_rows = rows[sort_order]
    indptr_array = group["indptr"]
    indices_array = group["indices"]
    data_array = group["data"]
    # Fetch indptr by contiguous row runs.  Zarr scalar reads are especially
    # expensive for K=4,000 recipes, whereas most sorted refs contain runs.
    breaks = np.flatnonzero(np.diff(sorted_rows) > 1) + 1
    runs = np.split(sorted_rows, breaks)
    starts_parts: list[np.ndarray] = []
    ends_parts: list[np.ndarray] = []
    for run in runs:
        first = int(run[0])
        last = int(run[-1])
        span = np.asarray(indptr_array[first : last + 2], dtype=np.int64)
        offsets = run - first
        starts_parts.append(span[offsets])
        ends_parts.append(span[offsets + 1])
    starts = np.concatenate(starts_parts)
    ends = np.concatenate(ends_parts)
    lengths = ends - starts
    out_indptr = np.concatenate(
        [np.zeros(1, dtype=np.int64), np.cumsum(lengths, dtype=np.int64)]
    )
    nnz = int(out_indptr[-1])
    data = np.empty(nnz, dtype=dtype)
    indices = np.empty(nnz, dtype=np.int32)
    chunk_size = int(getattr(data_array, "chunks", (0,))[0] or 0)
    if chunk_size <= 0:
        cursor = 0
        for start, end in zip(starts, ends, strict=True):
            length = int(end - start)
            if length:
                data[cursor : cursor + length] = np.asarray(
                    data_array[int(start) : int(end)], dtype=dtype
                )
                indices[cursor : cursor + length] = np.asarray(
                    indices_array[int(start) : int(end)], dtype=np.int32
                )
                cursor += length
    else:
        chunk_segments: dict[int, list[tuple[int, int, int]]] = defaultdict(list)
        for row_number, (start, end) in enumerate(zip(starts, ends, strict=True)):
            cursor = int(start)
            while cursor < int(end):
                chunk_id = cursor // chunk_size
                chunk_end = min(int(end), (chunk_id + 1) * chunk_size)
                chunk_segments[chunk_id].append((row_number, cursor, chunk_end))
                cursor = chunk_end
        write_offsets = out_indptr[:-1].copy()
        for chunk_id, segments in chunk_segments.items():
            chunk_start = int(chunk_id) * chunk_size
            chunk_stop = min(
                (int(chunk_id) + 1) * chunk_size, int(data_array.shape[0])
            )
            data_chunk = np.asarray(data_array[chunk_start:chunk_stop], dtype=dtype)
            index_chunk = np.asarray(
                indices_array[chunk_start:chunk_stop], dtype=np.int32
            )
            for row_number, start, end in segments:
                length = int(end) - int(start)
                output_start = int(write_offsets[row_number])
                output_end = output_start + length
                relative_start = int(start) - chunk_start
                relative_end = relative_start + length
                data[output_start:output_end] = data_chunk[
                    relative_start:relative_end
                ]
                indices[output_start:output_end] = index_chunk[
                    relative_start:relative_end
                ]
                write_offsets[row_number] = output_end
    result = sparse.csr_matrix(
        (data, indices, out_indptr), shape=(len(rows), int(n_cols)), dtype=dtype
    )
    if not np.array_equal(sort_order, np.arange(len(rows), dtype=np.int64)):
        result = result[np.argsort(sort_order, kind="mergesort")].tocsr()
    return result


def _validated_relocation(config: E2FullDataConfig) -> dict[str, str]:
    """Bind relocated files to unchanged, hash-verified legacy manifests.

    Path changes never disable the original content, split, recipe, or corpus
    checks. An omitted map retains the historical strict absolute-root checks.
    """
    if config.relocation_manifest is None:
        return {}
    path = Path(config.relocation_manifest).expanduser().resolve()
    payload = _read_json(path, description="dataset relocation map")
    if payload.get("schema_version") != "pbisc.dataset_relocation.v1":
        raise E2IndexContractError("unsupported dataset relocation schema")
    originals = payload.get("original_roots")
    relatives = payload.get("relative_roots")
    hashes = payload.get("original_sha256")
    if not all(isinstance(value, Mapping) for value in (originals, relatives, hashes)):
        raise E2IndexContractError("relocation map requires roots and original hashes")
    roots = {}
    for key in ("source_root", "base_source_root", "index_root", "gene_panel_path"):
        original = Path(str(originals.get(key, "")))
        if not original.is_absolute():
            raise E2IndexContractError(f"relocation original {key} must be absolute")
        mapped = _contained_path(path.parent, Path(str(relatives.get(key, ""))), description=key)
        if mapped != Path(getattr(config, key)).expanduser().resolve():
            raise E2IndexContractError(f"relocation mapped {key} differs from configuration")
        roots[key] = str(original)
    for key, artifact in (
        ("source_manifest", config.source_root / "dataset_manifest.json"),
        ("source_complete", config.source_root / "BUILD_COMPLETE.json"),
        ("index_manifest", config.index_root / "dataset_manifest.json"),
    ):
        _verify_file_fingerprint(artifact, hashes.get(key), description=f"original {key}")
    return roots


class E2FullDataModule:
    """Direct sparse loader for a built E2-full index."""

    def __init__(self, config: E2FullDataConfig) -> None:
        self.config = config
        self.source_root = config.source_root.expanduser().resolve()
        self.base_source_root = config.base_source_root.expanduser().resolve()
        self.index_root = config.index_root.expanduser().resolve()
        if not self.source_root.is_dir():
            raise FileNotFoundError(f"strict E2 source root not found: {self.source_root}")
        if not self.base_source_root.is_dir():
            raise FileNotFoundError(
                f"pinned E2 base source root not found: {self.base_source_root}"
            )
        if not (self.source_root / "BUILD_COMPLETE.json").is_file():
            raise FileNotFoundError(
                f"strict E2 source completion marker not found: "
                f"{self.source_root / 'BUILD_COMPLETE.json'}"
            )
        if not (self.index_root / "COMPLETE").is_file():
            raise FileNotFoundError(
                f"E2 index is not complete; run build_e2_full_index.py first: {self.index_root}"
            )
        provenance_roots = _validated_relocation(config)
        source_manifest_path = self.source_root / "dataset_manifest.json"
        self.source_metadata = _read_json(
            source_manifest_path,
            description="strict E2 source manifest",
        )
        complete_metadata = _read_json(
            self.source_root / "BUILD_COMPLETE.json",
            description="strict E2 source completion marker",
        )
        source_fingerprint = _validate_strict_source_metadata(
            self.source_metadata,
            complete_metadata,
            manifest_path=source_manifest_path,
            source_root=self.source_root,
            base_root=self.base_source_root,
            gene_panel_path=config.gene_panel_path.expanduser(),
            expected_datasets=config.expected_datasets,
            expected_cells=config.expected_cells,
            provenance_roots=provenance_roots,
        )

        self.build_metadata = _read_json(
            self.index_root / "dataset_manifest.json", description="E2 index manifest"
        )
        if self.build_metadata.get("schema_version") != E2_INDEX_SCHEMA:
            raise E2IndexContractError("unsupported E2 index schema")
        configured_source = provenance_roots.get("source_root", str(self.source_root))
        if str(Path(str(self.build_metadata.get("source_root", ""))).resolve()) != configured_source:
            raise E2IndexContractError("E2 index source root does not match configuration")
        configured_base = provenance_roots.get("base_source_root", str(self.base_source_root))
        index_base = str(
            Path(str(self.build_metadata.get("base_source_root", ""))).resolve()
        )
        if index_base != configured_base:
            raise E2IndexContractError("E2 index pinned base source root differs")
        if str(self.build_metadata.get("source_fingerprint", "")) != source_fingerprint:
            raise E2IndexContractError("E2 index strict-source fingerprint differs")
        self.corpus_fingerprint = str(self.build_metadata.get("corpus_fingerprint", ""))
        if len(self.corpus_fingerprint) != 64:
            raise E2IndexContractError("E2 index corpus fingerprint is missing or invalid")
        if (
            self.index_root.joinpath("COMPLETE").read_text(encoding="utf-8").strip()
            != self.corpus_fingerprint
        ):
            raise E2IndexContractError("E2 index COMPLETE fingerprint differs")

        strict_source_shards_path = self.source_root / "metadata" / "sc_shards.parquet"
        _verify_file_fingerprint(
            strict_source_shards_path,
            self.build_metadata.get("source_shards_sha256"),
            description="strict-source shard manifest",
        )
        index_source_shards_path = self.index_root / "metadata" / "source_shards.parquet"
        _verify_file_fingerprint(
            index_source_shards_path,
            self.build_metadata.get("index_source_shards_sha256"),
            description="indexed source-shard table",
        )
        sample_index_path = self.index_root / "metadata" / "sample_index.parquet"
        _verify_file_fingerprint(
            sample_index_path,
            self.build_metadata.get("sample_index_sha256"),
            description="E2 sample index",
        )
        donor_splits_path = self.index_root / "metadata" / "donor_splits.parquet"
        _verify_file_fingerprint(
            donor_splits_path,
            self.build_metadata.get("donor_splits_sha256"),
            description="E2 donor split table",
        )
        recipe_parts_manifest_path = (
            self.index_root / "metadata" / "recipe_parts.parquet"
        )
        _verify_file_fingerprint(
            recipe_parts_manifest_path,
            self.build_metadata.get("recipe_parts_manifest_sha256"),
            description="E2 recipe-part manifest",
        )

        self.gene_panel_path = config.gene_panel_path.expanduser().resolve()
        if not self.gene_panel_path.is_file():
            raise FileNotFoundError(f"E2 gene panel not found: {self.gene_panel_path}")
        expected_panel_hash = str(self.build_metadata.get("gene_panel_sha256", ""))
        if _sha256_file(self.gene_panel_path) != expected_panel_hash:
            raise E2IndexContractError("E2 gene panel changed after index construction")
        expected_build_signature = _canonical_digest(
            {
                "source_root": configured_source,
                "base_source_root": configured_base,
                "gene_panel_sha256": expected_panel_hash,
                "seed": int(config.seed),
                "split_seed": int(config.split_seed),
                "split_fractions": tuple(config.split_fractions),
                "platforms": tuple(config.platforms),
                "platform_sampling_policy": config.platform_sampling_policy,
                "sampling_policy": config.sampling_policy,
                "k_values": tuple(config.k_values),
                "k_weights": tuple(config.k_weights),
                "train_recipe_pool_size": int(config.train_recipe_pool_size),
                "train_recipes_per_epoch": int(config.train_recipes_per_epoch),
                "validation_recipes": int(config.validation_recipes),
                "validation_recipes_per_platform": int(
                    config.validation_recipes_per_platform
                ),
                "global_validation_contract": E2_GLOBAL_VALIDATION_CONTRACT,
                "expected_datasets": int(config.expected_datasets),
                "expected_cells": int(config.expected_cells),
            }
        )
        if (
            str(self.build_metadata.get("build_config_signature", ""))
            != expected_build_signature
        ):
            raise E2IndexContractError("E2 index build configuration signature differs")
        self.gene_panel = pd.read_parquet(self.gene_panel_path).reset_index(drop=True)
        if "panel_index" not in self.gene_panel.columns:
            raise E2IndexContractError("gene panel requires panel_index")
        self.gene_indices = self.gene_panel["panel_index"].to_numpy(dtype=np.int64)

        self.source_shards = pd.read_parquet(
            index_source_shards_path, columns=list(E2_SHARD_READ_COLUMNS)
        ).reset_index(drop=True)
        if self.source_shards.empty:
            raise E2IndexContractError("E2 index contains no source shards")
        if self.source_shards["dataset_id"].astype(str).str.startswith("REF-").any():
            raise E2IndexContractError("E2 index must exclude REF datasets")
        if int(self.source_shards["dataset_id"].nunique()) != config.expected_datasets:
            raise E2IndexContractError("E2 index dataset count differs from configuration")
        if int(self.source_shards["n_cells"].sum()) != config.expected_cells:
            raise E2IndexContractError("E2 index cell count differs from configuration")
        n_genes_values = set(self.source_shards["n_genes"].astype(int).tolist())
        if len(n_genes_values) != 1:
            raise E2IndexContractError("E2 source shards do not share one gene dimension")
        self.source_n_genes = int(next(iter(n_genes_values)))
        if self.gene_indices.size == 0 or int(self.gene_indices.min()) < 0:
            raise E2IndexContractError("gene panel indices must be non-empty and non-negative")
        if int(self.gene_indices.max()) >= self.source_n_genes:
            raise E2IndexContractError("gene panel index exceeds E2 source gene dimension")

        self._shards_by_id = {
            str(row.shard_id): {
                "count_zarr": str(row.count_zarr),
                "n_cells": int(row.n_cells),
            }
            for row in self.source_shards.itertuples(index=False)
        }
        self._sample_pool_path = self.index_root / "metadata" / "sample_pools.parquet"
        _verify_file_fingerprint(
            self._sample_pool_path,
            self.build_metadata.get("sample_pools_sha256"),
            description="E2 persisted sample-pool index",
        )
        if self.build_metadata.get("sample_pools_schema") != E2_SAMPLE_POOL_SCHEMA:
            raise E2IndexContractError("E2 persisted sample-pool schema differs")
        self._sample_pool_lookup = self._index_sample_pool_row_groups()

        part_manifest = pd.read_parquet(recipe_parts_manifest_path)
        required_part_columns = {
            "recipe_part",
            "split",
            "schedule_slot",
            "rows",
        }
        if not required_part_columns.issubset(part_manifest.columns):
            raise E2IndexContractError("recipe part manifest has an invalid schema")
        if part_manifest["recipe_part"].astype(str).duplicated().any():
            raise E2IndexContractError("recipe part manifest contains duplicate paths")
        part_manifest = part_manifest.sort_values(
            ["split", "schedule_slot", "recipe_part"], kind="mergesort"
        ).reset_index(drop=True)
        published_part_files = self.build_metadata.get("recipe_part_files")
        if not isinstance(published_part_files, list):
            raise E2IndexContractError("E2 index does not publish recipe-part hashes")
        if any(not isinstance(entry, Mapping) for entry in published_part_files):
            raise E2IndexContractError("E2 recipe-part hash entries must be objects")
        published_by_path = {
            str(entry.get("recipe_part", "")): entry
            for entry in published_part_files
        }
        if len(published_by_path) != len(published_part_files):
            raise E2IndexContractError("E2 recipe-part hashes contain duplicate paths")
        frames: list[pd.DataFrame] = []
        actual_part_files: list[dict[str, object]] = []
        for row in part_manifest.itertuples(index=False):
            part = str(row.recipe_part)
            path = _contained_path(
                self.index_root, Path(part), description="recipe part path"
            )
            published = published_by_path.get(part)
            expected_entry = {
                "recipe_part": part,
                "split": str(row.split),
                "schedule_slot": int(row.schedule_slot),
                "rows": int(row.rows),
                "sha256": "" if published is None else str(published.get("sha256", "")),
            }
            if published != expected_entry:
                raise E2IndexContractError(
                    f"E2 recipe-part manifest/hash metadata differs: {part}"
                )
            part_sha256 = _verify_file_fingerprint(
                path,
                expected_entry["sha256"],
                description=f"E2 recipe part {part}",
            )
            frame = pd.read_parquet(path, columns=list(E2_RECIPE_COLUMNS))
            if len(frame) != int(row.rows):
                raise E2IndexContractError(f"recipe part row count differs: {part}")
            actual_part_files.append({**expected_entry, "sha256": part_sha256})
            frames.append(frame)
        if actual_part_files != published_part_files:
            raise E2IndexContractError("E2 recipe-part file fingerprints differ")
        recipe_parts_content_sha256 = _canonical_digest(actual_part_files)
        if recipe_parts_content_sha256 != str(
            self.build_metadata.get("recipe_parts_content_sha256", "")
        ):
            raise E2IndexContractError("E2 recipe-part content fingerprint differs")
        self.recipes = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
        self._audit_recipes(self.recipes)
        recipe_id_digest = hashlib.sha256()
        ordered_recipes = self.recipes.sort_values("recipe_index", kind="mergesort")
        for recipe_id in ordered_recipes["recipe_id"].astype(str):
            recipe_id_digest.update(recipe_id.encode("ascii"))
            recipe_id_digest.update(b"\n")
        expected_core = {
            "source_fingerprint": source_fingerprint,
            "source_shards_sha256": str(
                self.build_metadata.get("source_shards_sha256", "")
            ),
            "build_config_signature": str(
                self.build_metadata.get("build_config_signature", "")
            ),
            "gene_panel_sha256": expected_panel_hash,
            "recipe_ids_sha256": recipe_id_digest.hexdigest(),
            "sample_pools_sha256": str(
                self.build_metadata.get("sample_pools_sha256", "")
            ),
            "sample_index_sha256": str(
                self.build_metadata.get("sample_index_sha256", "")
            ),
            "donor_splits_sha256": str(
                self.build_metadata.get("donor_splits_sha256", "")
            ),
            "index_source_shards_sha256": str(
                self.build_metadata.get("index_source_shards_sha256", "")
            ),
            "recipe_parts_manifest_sha256": str(
                self.build_metadata.get("recipe_parts_manifest_sha256", "")
            ),
            "recipe_parts_content_sha256": recipe_parts_content_sha256,
        }
        if frozenset(expected_core) != E2_CORPUS_CORE_KEYS:
            raise RuntimeError("internal E2 corpus-core schema drift")
        if self.build_metadata.get("corpus_core") != expected_core:
            raise E2IndexContractError("E2 corpus fingerprint core differs")
        if _canonical_digest(expected_core) != self.corpus_fingerprint:
            raise E2IndexContractError("E2 corpus fingerprint does not bind its artifacts")
        self._recipe_counts = self._compute_recipe_counts()
        self._pools_by_sample: dict[str, list[dict[str, Any]]] = {}
        self._zarr_cache: dict[str, Any] = {}

    def _index_sample_pool_row_groups(self) -> dict[str, list[tuple[int, int]]]:
        """Index small keys only; defer nested row-index reads until needed."""

        import pyarrow.parquet as pq

        parquet_file = pq.ParquetFile(self._sample_pool_path)
        expected_rows = int(self.build_metadata.get("sample_pools_rows", -1))
        if parquet_file.metadata.num_rows != expected_rows:
            raise E2IndexContractError("E2 persisted sample-pool row count differs")
        actual_schema = {
            field.name: str(field.type)
            for field in parquet_file.schema_arrow
            if field.name in E2_SAMPLE_POOL_SCHEMA
        }
        if actual_schema != E2_SAMPLE_POOL_SCHEMA:
            raise E2IndexContractError("E2 persisted sample-pool physical schema differs")
        lookup: dict[str, list[tuple[int, int]]] = defaultdict(list)
        for row_group in range(parquet_file.num_row_groups):
            keys = (
                parquet_file.read_row_group(row_group, columns=["sample_key"])
                .column("sample_key")
                .to_pylist()
            )
            for local_row, sample_key in enumerate(keys):
                lookup[str(sample_key)].append((int(row_group), int(local_row)))
        if not lookup:
            raise E2IndexContractError("E2 persisted sample-pool index is empty")
        return dict(lookup)

    def _audit_recipes(self, recipes: pd.DataFrame) -> None:
        missing = sorted(set(E2_RECIPE_COLUMNS).difference(recipes.columns))
        if missing:
            raise E2IndexContractError(f"E2 recipes are missing columns: {missing}")
        if recipes.empty:
            raise E2IndexContractError("E2 index contains no recipes")
        if not recipes["sampling_policy"].astype(str).eq("random").all():
            raise E2IndexContractError("E2 recipes must all use random sampling")
        if recipes["dataset_id"].astype(str).str.startswith("REF-").any():
            raise E2IndexContractError("E2 recipes contain an excluded REF dataset")
        if not set(recipes["platform"].astype(str)).issubset(self.config.platforms):
            raise E2IndexContractError("E2 recipes contain an unknown platform")
        if not set(recipes["k"].astype(int)).issubset(self.config.k_values):
            raise E2IndexContractError("E2 recipes contain unsupported K values")
        if recipes["recipe_id"].astype(str).duplicated().any():
            raise E2IndexContractError("E2 recipe_id values must be unique")
        train = recipes[recipes["split"].astype(str).eq(self.config.train_split)]
        if len(train) != self.config.train_recipe_pool_size:
            raise E2IndexContractError("E2 train recipe pool size differs from configuration")
        slots = train.groupby("schedule_slot", sort=True).size()
        if slots.empty or not slots.eq(self.config.train_recipes_per_epoch).all():
            raise E2IndexContractError("every E2 train schedule slot must contain one epoch")
        if len(slots) < 4:
            raise E2IndexContractError("E2 index does not contain four disjoint train epochs")

    def _compute_recipe_counts(self) -> dict[str, dict[str, int]]:
        counts: dict[str, dict[str, int]] = {
            split: {platform: 0 for platform in self.config.platforms}
            for split in ("train", "val", "test", "diagnostic", "evaluation")
        }
        grouped = self.recipes.groupby(["split", "platform"], sort=True).size()
        for (split, platform), count in grouped.items():
            counts.setdefault(str(split), {name: 0 for name in self.config.platforms})[
                str(platform)
            ] = int(count)
        for platform in self.config.platforms:
            counts["evaluation"][platform] = (
                0 if platform == "ScaleBio" else counts["val"][platform]
            )
        return counts

    def valid_recipe_counts(self) -> dict[str, dict[str, int]]:
        return {split: dict(values) for split, values in self._recipe_counts.items()}

    def _training_slot_for_seed(self, seed: int) -> int:
        slots = int(
            self.recipes.loc[
                self.recipes["split"].astype(str).eq(self.config.train_split),
                "schedule_slot",
            ].nunique()
        )
        delta = int(seed) - int(self.config.seed)
        if delta >= 0 and delta % 1_000_003 == 0:
            return int(delta // 1_000_003) % slots
        return _stable_hash(("train-slot", int(seed)), self.config.seed) % slots

    def split_manifest(
        self,
        split: str,
        max_blocks: int | None = None,
        seed: int = 0,
        platform: str | None = None,
    ) -> pd.DataFrame:
        split = str(split)
        if platform is not None and platform not in self.config.platforms:
            raise ValueError(f"unknown E2 platform: {platform}")
        if split == "evaluation":
            selected = self.recipes[self.recipes["split"].astype(str).eq("val")]
        else:
            selected = self.recipes[self.recipes["split"].astype(str).eq(split)]
        if split == self.config.train_split:
            slot = self._training_slot_for_seed(int(seed))
            selected = selected[selected["schedule_slot"].astype(int).eq(slot)]
        if platform is not None:
            selected = selected[selected["platform"].astype(str).eq(platform)]
        if selected.empty:
            return selected.copy().reset_index(drop=True)
        selected = selected.sample(frac=1.0, random_state=int(seed) % (2**32 - 1))
        if max_blocks is not None:
            selected = selected.head(max(0, int(max_blocks)))
        selected = selected.reset_index(drop=True)
        selected["stream_recipe_index"] = np.arange(len(selected), dtype=np.int64)
        return selected

    def prime_pools(self, recipes: pd.DataFrame) -> None:
        if recipes.empty:
            return
        needed = set(recipes["sample_key"].astype(str)) - set(self._pools_by_sample)
        if not needed:
            return
        missing_lookup = sorted(needed.difference(self._sample_pool_lookup))
        if missing_lookup:
            raise E2IndexContractError(
                f"E2 recipes reference unknown persisted sample pools: {missing_lookup[:5]}"
            )
        import pyarrow.parquet as pq

        references_by_group: dict[int, list[tuple[str, int]]] = defaultdict(list)
        for sample_key in sorted(needed):
            for row_group, local_row in self._sample_pool_lookup[sample_key]:
                references_by_group[row_group].append((sample_key, local_row))
        parquet_file = pq.ParquetFile(self._sample_pool_path)
        for row_group, references in references_by_group.items():
            table = parquet_file.read_row_group(
                row_group, columns=list(E2_SAMPLE_POOL_SCHEMA)
            )
            keys = table.column("sample_key")
            shard_ids = table.column("shard_id")
            row_indices = table.column("row_indices")
            for expected_key, local_row in references:
                actual_key = str(keys[local_row].as_py())
                if actual_key != expected_key:
                    raise E2IndexContractError("E2 sample-pool row-group lookup drifted")
                shard_id = str(shard_ids[local_row].as_py())
                shard = self._shards_by_id.get(shard_id)
                if shard is None:
                    raise E2IndexContractError(
                        f"E2 sample pool references unknown shard: {shard_id}"
                    )
                rows = np.asarray(row_indices[local_row].as_py(), dtype=np.int64)
                if (
                    rows.ndim != 1
                    or bool((rows < 0).any())
                    or bool((rows >= int(shard["n_cells"])).any())
                ):
                    raise E2IndexContractError(
                        f"E2 sample pool contains invalid rows for shard {shard_id}"
                    )
                self._pools_by_sample.setdefault(expected_key, []).append(
                    {
                        "shard_id": shard_id,
                        "count_zarr": str(shard["count_zarr"]),
                        "row_indices": rows,
                    }
                )
        missing = sorted(needed.difference(self._pools_by_sample))
        if missing:
            raise E2IndexContractError(
                f"E2 recipes could not resolve strict source sample pools: {missing[:5]}"
            )

    def _recipe_refs_and_pools(
        self, recipe: pd.Series
    ) -> tuple[list[tuple[int, int]], list[dict[str, Any]]]:
        sample_key = str(recipe["sample_key"])
        if sample_key not in self._pools_by_sample:
            self.prime_pools(pd.DataFrame([recipe.to_dict()]))
        pools = self._pools_by_sample.get(sample_key, [])
        lengths = np.asarray([len(pool["row_indices"]) for pool in pools], dtype=np.int64)
        total = int(lengths.sum())
        k = int(recipe["k"])
        if total < k:
            raise E2IndexContractError(
                f"sample {sample_key} contains {total} strict rows, fewer than K={k}"
            )
        rng = np.random.default_rng(int(recipe["rng_seed"]))
        flat = rng.choice(total, size=k, replace=False)
        cumulative = np.cumsum(lengths)
        segments = np.searchsorted(cumulative, flat, side="right")
        prior = np.concatenate([np.zeros(1, dtype=np.int64), cumulative[:-1]])
        local = flat - prior[segments]
        refs = [
            (int(segment), int(pools[int(segment)]["row_indices"][int(offset)]))
            for segment, offset in zip(segments, local, strict=True)
        ]
        rng.shuffle(refs)
        return refs, pools

    def _zarr_group(self, relative_path: str) -> Any:
        cached = self._zarr_cache.get(relative_path)
        if cached is not None:
            return cached
        if len(self._zarr_cache) >= 16:
            self._zarr_cache.pop(next(iter(self._zarr_cache)))
        path = _resolve_source_uri(
            relative_path,
            strict_root=self.source_root,
            base_root=self.base_source_root,
            description="strict count Zarr path",
        )
        group = _open_csr_zarr(path)
        self._zarr_cache[relative_path] = group
        return group

    def _load_sparse_refs(
        self, refs: list[tuple[int, int]], pools: list[dict[str, Any]]
    ) -> sparse.csr_matrix:
        by_segment: dict[int, list[tuple[int, int]]] = defaultdict(list)
        for position, (segment, row) in enumerate(refs):
            by_segment[int(segment)].append((int(position), int(row)))
        matrices: list[sparse.csr_matrix] = []
        original_positions: list[int] = []
        for segment, positioned_rows in by_segment.items():
            ordered = sorted(positioned_rows, key=lambda item: item[1])
            rows = np.asarray([row for _, row in ordered], dtype=np.int64)
            group = self._zarr_group(str(pools[segment]["count_zarr"]))
            matrices.append(_csr_rows(group, rows, self.source_n_genes))
            original_positions.extend(position for position, _ in ordered)
        if not matrices:
            return sparse.csr_matrix((0, self.source_n_genes), dtype=np.float32)
        combined = (
            matrices[0]
            if len(matrices) == 1
            else sparse.vstack(matrices, format="csr")
        )
        restore_order = np.argsort(
            np.asarray(original_positions, dtype=np.int64), kind="mergesort"
        )
        return combined[restore_order].tocsr()

    def load_training_block(
        self,
        recipe: pd.Series,
        cells_per_block: int,
        rng: np.random.Generator,
    ) -> AnnotationFreeRecipeBlock:
        refs, pools = self._recipe_refs_and_pools(recipe)
        k = int(recipe["k"])
        if len(refs) != k:
            raise RuntimeError("E2 recipe resolved a row count different from K")
        matrix = self._load_sparse_refs(refs, pools)[:, self.gene_indices].tocsr()
        bulk_raw = np.asarray(matrix.sum(axis=0), dtype=np.float64).reshape(-1)
        bulk = np.log1p(bulk_raw).astype(np.float32)
        take = min(max(int(cells_per_block), 0), k)
        if take:
            chosen = rng.choice(k, size=take, replace=False)
            x = matrix[chosen].toarray().astype(np.float32, copy=False)
        else:
            x = np.empty((0, len(self.gene_indices)), dtype=np.float32)
        return AnnotationFreeRecipeBlock(
            block_id=str(recipe["block_id"]), x=x, bulk=bulk, n_cells=k
        )


class E2AnnotationFreeCellDataset(IterableDataset[dict[str, object]]):
    """Stream one E2 split while keeping platform only in sampler state."""

    def __init__(
        self,
        data_module: E2FullDataModule,
        *,
        split: str,
        recipes_per_epoch: int,
        cells_per_block: int,
        conditioner: AnnotationFreeConditioner,
        seed: int,
        platform: str | None = None,
        progress: bool = True,
    ) -> None:
        super().__init__()
        if not conditioner.fitted_:
            raise ValueError("conditioner must be fit before building E2 datasets")
        if recipes_per_epoch <= 0 or cells_per_block <= 0:
            raise ValueError("recipes_per_epoch and cells_per_block must be positive")
        self.data_module = data_module
        self.split = str(split)
        self.recipes_per_epoch = int(recipes_per_epoch)
        self.cells_per_block = int(cells_per_block)
        self.conditioner = conditioner
        self.seed = int(seed)
        self.platform = platform
        self.progress = bool(progress)
        self._condition_feature_names = conditioner.feature_names
        audit_annotation_free_names(
            self._condition_feature_names, context="condition features"
        )
        self._shared_epoch = mp.Value("q", 0)

    def set_epoch(self, epoch: int) -> None:
        with self._shared_epoch.get_lock():
            self._shared_epoch.value = int(epoch)

    def __len__(self) -> int:
        with self._shared_epoch.get_lock():
            epoch = int(self._shared_epoch.value)
        epoch_seed = self.seed + 1_000_003 * epoch
        recipes = self.data_module.split_manifest(
            self.split,
            max_blocks=self.recipes_per_epoch,
            seed=epoch_seed,
            platform=self.platform,
        )
        return int(
            np.minimum(
                recipes["k"].to_numpy(dtype=np.int64, copy=False),
                self.cells_per_block,
            ).sum()
        )

    def __iter__(self) -> Iterator[dict[str, object]]:
        with self._shared_epoch.get_lock():
            epoch = int(self._shared_epoch.value)
        epoch_seed = self.seed + 1_000_003 * epoch
        recipes = self.data_module.split_manifest(
            self.split,
            max_blocks=self.recipes_per_epoch,
            seed=epoch_seed,
            platform=self.platform,
        )
        worker = get_worker_info()
        worker_id = 0
        if worker is not None:
            worker_id = int(worker.id)
            recipes = recipes.iloc[worker_id :: int(worker.num_workers)].reset_index(
                drop=True
            )
        self.data_module.prime_pools(recipes)
        rows: Iterable[tuple[object, pd.Series]] = recipes.iterrows()
        if self.progress:
            from tqdm import tqdm

            description = f"E2 {self.split} epoch {epoch}"
            if self.platform is not None:
                description += f" {self.platform}"
            if worker is not None:
                description += f" worker {worker_id}"
            rows = tqdm(rows, total=len(recipes), desc=description)
        rng = np.random.default_rng(epoch_seed + 7_919 + 104_729 * worker_id)
        audited = False
        for fallback_index, (_, recipe) in enumerate(rows):
            block = self.data_module.load_training_block(
                recipe, self.cells_per_block, rng
            )
            counts = np.asarray(block.x, dtype=np.float32)
            if counts.ndim != 2 or counts.shape[0] == 0:
                continue
            if not np.isfinite(counts).all() or float(counts.min(initial=0.0)) < 0:
                raise ValueError("E2 source block contains invalid count values")
            recipe_k = int(block.n_cells)
            condition = self.conditioner.transform(block.bulk, recipe_k)
            block_index = np.int64(
                int(recipe.get("stream_recipe_index", fallback_index))
            )
            for cell_counts in counts:
                sample: dict[str, object] = {
                    "x_log": np.log1p(cell_counts).astype(np.float32, copy=False),
                    "condition": condition,
                    "block_index": block_index,
                    "block_id": str(block.block_id),
                    "recipe_k": np.int64(recipe_k),
                }
                if frozenset(sample) != E2_SAMPLE_KEYS:
                    raise RuntimeError("E2 emitted an unexpected model-batch schema")
                if not audited:
                    audit_annotation_free_batch(
                        sample,
                        condition_feature_names=self._condition_feature_names,
                    )
                    audited = True
                yield sample


@dataclass(frozen=True)
class E2FullDatasets:
    train: E2AnnotationFreeCellDataset
    validation: E2AnnotationFreeCellDataset
    validation_by_platform: Mapping[str, E2AnnotationFreeCellDataset]


def build_e2_full_data_module(config: E2FullDataConfig) -> E2FullDataModule:
    return E2FullDataModule(config)


def iter_e2_full_fit_records(
    data_module: E2FullDataModule, config: E2FullDataConfig
) -> Iterator[tuple[np.ndarray, int]]:
    recipes = data_module.split_manifest(
        config.train_split, max_blocks=config.fit_blocks, seed=config.seed
    )
    data_module.prime_pools(recipes)
    rows: Iterable[tuple[object, pd.Series]] = recipes.iterrows()
    if config.progress:
        from tqdm import tqdm

        rows = tqdm(rows, total=len(recipes), desc="fit E2 annotation-free conditioner")
    rng = np.random.default_rng(config.seed + 31_337)
    for _, recipe in rows:
        block = data_module.load_training_block(recipe, 0, rng)
        yield np.asarray(block.bulk, dtype=np.float32), int(block.n_cells)


def fit_e2_full_conditioner(
    data_module: E2FullDataModule, config: E2FullDataConfig
) -> AnnotationFreeConditioner:
    conditioner = AnnotationFreeConditioner(n_genes=int(len(data_module.gene_panel)))
    return conditioner.fit(iter_e2_full_fit_records(data_module, config))


def build_e2_full_datasets(
    data_module: E2FullDataModule,
    conditioner: AnnotationFreeConditioner,
    config: E2FullDataConfig,
) -> E2FullDatasets:
    train = E2AnnotationFreeCellDataset(
        data_module,
        split=config.train_split,
        recipes_per_epoch=config.train_recipes_per_epoch,
        cells_per_block=config.train_cells_per_block,
        conditioner=conditioner,
        seed=config.seed,
        progress=config.progress,
    )
    validation = E2AnnotationFreeCellDataset(
        data_module,
        split="evaluation",
        recipes_per_epoch=config.validation_recipes,
        cells_per_block=config.validation_cells_per_block,
        conditioner=conditioner,
        seed=config.seed + 10_007,
        progress=config.progress,
    )
    by_platform: dict[str, E2AnnotationFreeCellDataset] = {}
    for offset, platform in enumerate(config.platforms):
        split = "diagnostic" if platform == "ScaleBio" else config.validation_split
        by_platform[platform] = E2AnnotationFreeCellDataset(
            data_module,
            split=split,
            recipes_per_epoch=config.validation_recipes_per_platform,
            cells_per_block=config.validation_cells_per_block,
            conditioner=conditioner,
            seed=config.seed + 20_011 + 997 * offset,
            platform=platform,
            progress=config.progress,
        )
    return E2FullDatasets(
        train=train, validation=validation, validation_by_platform=by_platform
    )


__all__ = [
    "E2FullDataConfig",
    "E2FullDataModule",
    "E2FullDatasets",
    "E2IndexContractError",
    "E2AnnotationFreeCellDataset",
    "build_e2_full_data_module",
    "build_e2_full_datasets",
    "fit_e2_full_conditioner",
    "iter_e2_full_fit_records",
    "platform_for_dataset",
]
