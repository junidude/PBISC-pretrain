#!/usr/bin/env python3
"""Build the deterministic, random-only E2-full sparse recipe index.

The builder reads an already completed strict E2 source.  Source obs reads use
an explicit four-column projection and selection depends only on dataset,
donor/sample provenance, row position, and cell counts.  No annotation value,
annotation mask, or label-derived QC field participates in the index.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


from model7_no_annotation.data_e2 import (  # noqa: E402
    E2_INDEX_SCHEMA,
    E2_GLOBAL_VALIDATION_CONTRACT,
    E2_OBS_READ_COLUMNS,
    E2_PLATFORMS,
    E2_RECIPE_COLUMNS,
    E2_SAMPLE_POOL_SCHEMA,
    E2_SHARD_READ_COLUMNS,
    E2FullDataConfig,
    E2IndexContractError,
    _normalize_obs_groups,
    _resolve_source_uri,
    _sample_key,
    _sha256_file,
    _stable_hash,
    _validate_strict_source_metadata,
    platform_for_dataset,
)


POOL_SCHEMA = pa.schema(
    [
        ("sample_key", pa.string()),
        ("shard_id", pa.string()),
        ("row_indices", pa.list_(pa.int32())),
    ]
)


def _json_default(value: object) -> object:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, tuple):
        return list(value)
    raise TypeError(f"cannot JSON-encode {type(value).__name__}")


def _canonical_json(payload: object) -> str:
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        default=_json_default,
    )


def _digest(payload: object) -> str:
    return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


def _load_config(path: Path) -> tuple[dict[str, Any], E2FullDataConfig]:
    config_path = path.expanduser().resolve()
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError("E2 configuration root must be a JSON object")
    paths = payload.get("paths")
    data = payload.get("data")
    training = payload.get("training")
    conditioner = payload.get("conditioner")
    if not all(isinstance(section, dict) for section in (paths, data, training, conditioner)):
        raise ValueError("E2 configuration requires paths/data/training/conditioner objects")
    fractions = data.get("split_fractions", {"train": 0.8, "val": 0.1, "test": 0.1})
    values: dict[str, object] = {
        "project_root": Path(str(paths["method1_root"])),
        "source_root": Path(str(paths["source_root"])),
        "base_source_root": Path(str(paths["base_source_root"])),
        "index_root": Path(str(paths["index_root"])),
        "gene_panel_path": Path(str(paths["gene_panel_path"])),
        "train_split": str(data.get("train_split", "train")),
        "validation_split": str(data.get("eval_split", "val")),
        "train_recipes_per_epoch": int(data.get("recipes_per_epoch", 8192)),
        "validation_recipes": int(data.get("max_eval_blocks", 192)),
        "validation_recipes_per_platform": int(
            data.get("validation_recipes_per_platform", 64)
        ),
        "fit_blocks": int(data.get("fit_blocks", 2048)),
        "train_cells_per_block": int(data.get("cells_per_block", 512)),
        "validation_cells_per_block": int(
            data.get("validation_cells_per_block", 64)
        ),
        "seed": int(training.get("seed", 20260627)),
        "condition_mode": str(conditioner.get("mode", "bio_bulk_depth")),
        "sampling_policy": str(data.get("sampling_policy", "random")),
        "platform_sampling_policy": str(
            data.get("platform_sampling_policy", "sqrt_cells")
        ),
        "platforms": tuple(data.get("platforms", E2_PLATFORMS)),
        "k_values": tuple(data.get("k_values", (250, 500, 1000, 2000, 4000))),
        "k_weights": tuple(data.get("k_weights", (1, 2, 3, 4, 1))),
        "split_seed": int(data.get("split_seed", 20260523)),
        "split_unit": str(data.get("split_unit", "dataset_donor")),
        "sample_fallback_for_missing_donor": bool(
            data.get("sample_fallback_for_missing_donor", True)
        ),
        "split_fractions": fractions,
        "excluded_dataset_ids": tuple(
            data.get("excluded_dataset_ids", ("REF-001_Tabula_Sapiens",))
        ),
        "train_recipe_pool_size": int(data.get("train_recipe_pool_size", 131_072)),
        "expected_datasets": int(data.get("expected_datasets", 22)),
        "expected_cells": int(data.get("expected_cells", 41_395_184)),
        "progress": True,
    }
    return payload, E2FullDataConfig(**values)


def _integer_quotas(total: int, labels: Iterable[object], weights: Iterable[float]) -> dict[Any, int]:
    labels_tuple = tuple(labels)
    weight_array = np.asarray(tuple(weights), dtype=np.float64)
    if len(labels_tuple) != len(weight_array) or not labels_tuple:
        raise ValueError("quota labels and weights must be non-empty and aligned")
    if not np.isfinite(weight_array).all() or bool((weight_array < 0).any()):
        raise ValueError("quota weights must be finite and non-negative")
    if float(weight_array.sum()) <= 0:
        raise ValueError("quota weights must have positive total mass")
    raw = weight_array / weight_array.sum() * int(total)
    counts = np.floor(raw).astype(np.int64)
    remainder = int(total) - int(counts.sum())
    if remainder:
        order = np.argsort(-(raw - counts), kind="mergesort")
        counts[order[:remainder]] += 1
    return {label: int(count) for label, count in zip(labels_tuple, counts, strict=True)}


def _joint_quotas(
    row_quotas: Mapping[str, int], column_quotas: Mapping[int, int]
) -> dict[tuple[str, int], int]:
    rows = tuple(row_quotas)
    columns = tuple(column_quotas)
    total = int(sum(row_quotas.values()))
    if total != int(sum(column_quotas.values())):
        raise ValueError("joint quota margins have different totals")
    raw = np.outer(
        np.asarray([row_quotas[row] for row in rows], dtype=np.float64),
        np.asarray([column_quotas[column] for column in columns], dtype=np.float64),
    ) / float(total)
    matrix = np.floor(raw).astype(np.int64)
    row_remaining = np.asarray(
        [row_quotas[row] for row in rows], dtype=np.int64
    ) - matrix.sum(axis=1)
    column_remaining = np.asarray(
        [column_quotas[column] for column in columns], dtype=np.int64
    ) - matrix.sum(axis=0)
    fractions = raw - matrix
    while int(row_remaining.sum()) > 0:
        choices = [
            (float(fractions[i, j]), -i, -j, i, j)
            for i in range(len(rows))
            for j in range(len(columns))
            if row_remaining[i] > 0 and column_remaining[j] > 0
        ]
        if not choices:
            raise RuntimeError("could not satisfy integer joint quotas")
        _, _, _, i, j = max(choices)
        matrix[i, j] += 1
        row_remaining[i] -= 1
        column_remaining[j] -= 1
    return {
        (row, column): int(matrix[i, j])
        for i, row in enumerate(rows)
        for j, column in enumerate(columns)
    }


def _split_counts(n_units: int, fractions: tuple[float, float, float]) -> dict[str, int]:
    if n_units <= 0:
        return {"train": 0, "val": 0, "test": 0}
    if n_units == 1:
        return {"train": 1, "val": 0, "test": 0}
    counts = _integer_quotas(n_units, ("train", "val", "test"), fractions)
    counts["train"] = max(1, counts["train"])
    if n_units >= 3:
        counts["val"] = max(1, counts["val"])
        counts["test"] = max(1, counts["test"])
    while sum(counts.values()) > n_units:
        candidates = [name for name in ("train", "val", "test") if counts[name] > 1]
        if not candidates:
            candidates = [name for name in ("test", "val") if counts[name] > 0]
        counts[max(candidates, key=lambda name: counts[name])] -= 1
    while sum(counts.values()) < n_units:
        counts["train"] += 1
    return counts


def _assign_donor_splits(
    samples: pd.DataFrame,
    *,
    fractions: tuple[float, float, float],
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    split_samples = samples.copy()
    split_samples["split_key_type"] = np.where(
        split_samples["donor_id"].eq("__missing_donor__"), "sample", "donor"
    )
    split_samples["split_key_value"] = np.where(
        split_samples["split_key_type"].eq("sample"),
        split_samples["sample_id"],
        split_samples["donor_id"],
    )
    units = (
        split_samples.groupby(
            [
                "platform",
                "dataset_id",
                "split_key_type",
                "split_key_value",
            ],
            sort=True,
            dropna=False,
        )
        .agg(
            n_cells=("n_cells", "sum"),
            n_samples=("sample_key", "size"),
            max_sample_cells=("n_cells", "max"),
        )
        .reset_index()
    )
    unit_frames: list[pd.DataFrame] = []
    for (platform, dataset_id), group in units.groupby(
        ["platform", "dataset_id"], sort=True
    ):
        group = group.copy().reset_index(drop=True)
        group["hash_rank"] = group["split_key_value"].map(
            lambda value: _stable_hash((dataset_id, value, "donor-split"), seed)
        )
        group = group.sort_values(
            ["max_sample_cells", "hash_rank", "split_key_value"],
            ascending=[False, True, True],
            kind="mergesort",
        ).reset_index(drop=True)
        counts = _split_counts(len(group), fractions)
        if platform == "ScaleBio":
            counts = {"train": len(group), "val": 0, "test": 0}
        assignment = ["train"] * len(group)
        cursor = 0
        # Preserve the highest-capacity group in train, then reserve held-out
        # capacity.  This keeps all K values available for E2 training and, when
        # possible, validation without looking at biological annotations.
        if counts["train"] > 0:
            assignment[0] = "train"
            cursor = 1
        if counts["val"] > 0 and cursor < len(group):
            assignment[cursor] = "val"
            cursor += 1
        if counts["test"] > 0 and cursor < len(group):
            assignment[cursor] = "test"
            cursor += 1
        remaining_indices = list(range(cursor, len(group)))
        remaining_indices.sort(key=lambda i: (int(group.at[i, "hash_rank"]), i))
        already = Counter(assignment[:cursor])
        desired: list[str] = []
        for split in ("val", "test", "train"):
            desired.extend([split] * max(0, counts[split] - already[split]))
        for index, split in zip(remaining_indices, desired, strict=True):
            assignment[index] = split
        group["split"] = assignment
        group["heldout_available"] = bool(platform != "ScaleBio" and len(group) > 1)
        unit_frames.append(group)
    donor_splits = pd.concat(unit_frames, ignore_index=True)
    split_samples = split_samples.merge(
        donor_splits[
            ["dataset_id", "split_key_type", "split_key_value", "split"]
        ],
        on=["dataset_id", "split_key_type", "split_key_value"],
        how="left",
        validate="many_to_one",
    )
    if split_samples["split"].isna().any():
        raise RuntimeError("failed to assign every E2 sample to a split")
    leakage = donor_splits.groupby(
        ["dataset_id", "split_key_type", "split_key_value"], sort=False
    )["split"].nunique()
    if bool((leakage > 1).any()):
        raise RuntimeError("E2 donor/sample split leakage detected")
    return split_samples, donor_splits


def _sampling_weights(candidates: pd.DataFrame) -> np.ndarray:
    dataset_cells = candidates.groupby("dataset_id")["n_cells"].sum()
    donor_counts = candidates.groupby("dataset_id")["split_key_value"].nunique()
    sample_sqrt = np.sqrt(candidates["n_cells"].to_numpy(dtype=np.float64))
    donor_sample_sqrt = candidates.assign(_sqrt=sample_sqrt).groupby(
        ["dataset_id", "split_key_value"]
    )["_sqrt"].transform("sum").to_numpy(dtype=np.float64)
    weights = np.asarray(
        [
            np.sqrt(float(dataset_cells[dataset_id]))
            / float(donor_counts[dataset_id])
            for dataset_id in candidates["dataset_id"]
        ],
        dtype=np.float64,
    )
    weights *= sample_sqrt / np.maximum(donor_sample_sqrt, 1.0e-12)
    return weights / weights.sum()


def _candidate_cache(samples: pd.DataFrame, k_values: tuple[int, ...]) -> dict[tuple[str, str, int], tuple[pd.DataFrame, np.ndarray]]:
    cache: dict[tuple[str, str, int], tuple[pd.DataFrame, np.ndarray]] = {}
    for split in ("train", "val", "test", "diagnostic"):
        actual_split = "train" if split == "diagnostic" else split
        for platform in E2_PLATFORMS:
            for k in k_values:
                candidates = samples[
                    samples["split"].astype(str).eq(actual_split)
                    & samples["platform"].astype(str).eq(platform)
                    & (samples["n_cells"].astype(int) >= int(k))
                ].reset_index(drop=True)
                if not candidates.empty:
                    cache[(split, platform, int(k))] = (
                        candidates,
                        _sampling_weights(candidates),
                    )
    return cache


def _effective_k_quotas(
    total: int,
    k_values: tuple[int, ...],
    k_weights: tuple[int, ...],
    available: Iterable[int],
) -> dict[int, int]:
    available_set = set(int(k) for k in available)
    labels = tuple(k for k in k_values if k in available_set)
    weights = tuple(weight for k, weight in zip(k_values, k_weights, strict=True) if k in available_set)
    if not labels:
        raise E2IndexContractError("no configured K value is available for a recipe stratum")
    quotas = {int(k): 0 for k in k_values}
    quotas.update(_integer_quotas(total, labels, weights))
    return quotas


def _recipe_row(
    sample: pd.Series,
    *,
    split: str,
    platform: str,
    k: int,
    schedule_slot: int,
    slot_position: int,
    global_index: int,
    seed: int,
    heldout: bool,
    validation_scope: str,
) -> dict[str, Any]:
    rng_seed = _stable_hash(
        (
            "E2-recipe",
            split,
            schedule_slot,
            slot_position,
            sample["sample_key"],
            int(k),
        ),
        seed,
    )
    identity = hashlib.sha256(
        f"{split}|{schedule_slot}|{slot_position}|{sample['sample_key']}|{k}|{rng_seed}".encode(
            "utf-8"
        )
    ).hexdigest()
    recipe_id = identity[:24]
    return {
        "recipe_id": recipe_id,
        "block_id": f"E2_{split}_s{schedule_slot:03d}_r{slot_position:06d}_{recipe_id}",
        "recipe_index": int(global_index),
        "schedule_slot": int(schedule_slot),
        "split": str(split),
        "platform": str(platform),
        "validation_scope": str(validation_scope),
        "heldout": bool(heldout),
        "sample_key": str(sample["sample_key"]),
        "dataset_id": str(sample["dataset_id"]),
        "donor_id": str(sample["donor_id"]),
        "sample_id": str(sample["sample_id"]),
        "k": int(k),
        "rng_seed": int(rng_seed),
        "sampling_policy": "random",
    }


def _draw_recipe_rows(
    *,
    split: str,
    schedule_slot: int,
    joint_quotas: Mapping[tuple[str, int], int],
    candidates: Mapping[tuple[str, str, int], tuple[pd.DataFrame, np.ndarray]],
    seed: int,
    global_start: int,
    heldout_by_platform: Mapping[str, bool],
    scope_by_platform: Mapping[str, str],
) -> list[dict[str, Any]]:
    strata = [
        (platform, int(k))
        for (platform, k), count in joint_quotas.items()
        for _ in range(int(count))
    ]
    rng = np.random.default_rng(
        _stable_hash(("E2-schedule", split, schedule_slot), seed)
    )
    rng.shuffle(strata)
    draw_counts = Counter(strata)
    sampled_positions: dict[tuple[str, int], np.ndarray] = {}
    sampled_cursors: dict[tuple[str, int], int] = defaultdict(int)
    for key, count in draw_counts.items():
        platform, k = key
        candidate_key = (split, platform, int(k))
        if candidate_key not in candidates:
            raise E2IndexContractError(
                f"no E2 source sample can satisfy split={split}, "
                f"platform={platform}, K={k}"
            )
        frame, probabilities = candidates[candidate_key]
        # One vectorized weighted draw builds the probability CDF once for the
        # whole stratum instead of once per one of 131,072 train recipes.
        sampled_positions[key] = np.asarray(
            rng.choice(len(frame), size=int(count), replace=True, p=probabilities),
            dtype=np.int64,
        )
    rows: list[dict[str, Any]] = []
    for position, (platform, k) in enumerate(strata):
        key = (split, platform, int(k))
        frame, _ = candidates[key]
        draw_key = (platform, int(k))
        cursor = sampled_cursors[draw_key]
        sample_position = int(sampled_positions[draw_key][cursor])
        sampled_cursors[draw_key] = cursor + 1
        rows.append(
            _recipe_row(
                frame.iloc[sample_position],
                split=split,
                platform=platform,
                k=k,
                schedule_slot=schedule_slot,
                slot_position=position,
                global_index=global_start + position,
                seed=seed,
                heldout=bool(heldout_by_platform[platform]),
                validation_scope=scope_by_platform[platform],
            )
        )
    return rows


def _strict_source_metadata(config: E2FullDataConfig) -> tuple[dict[str, Any], dict[str, Any], str]:
    source_root = config.source_root.expanduser().resolve()
    manifest_path = source_root / "dataset_manifest.json"
    complete_path = source_root / "BUILD_COMPLETE.json"
    if not manifest_path.is_file() or not complete_path.is_file():
        raise FileNotFoundError(
            f"strict E2 source is incomplete; expected {manifest_path} and {complete_path}"
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    complete = json.loads(complete_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict) or not isinstance(complete, dict):
        raise E2IndexContractError("strict E2 source manifests must be JSON objects")
    fingerprint = _validate_strict_source_metadata(
        manifest,
        complete,
        manifest_path=manifest_path,
        source_root=source_root,
        base_root=config.base_source_root.expanduser(),
        gene_panel_path=config.gene_panel_path.expanduser(),
        expected_datasets=config.expected_datasets,
        expected_cells=config.expected_cells,
    )
    return manifest, complete, fingerprint


def _scan_source(
    config: E2FullDataConfig,
    output_root: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    source_root = config.source_root.expanduser().resolve()
    base_root = config.base_source_root.expanduser().resolve()
    shard_path = source_root / "metadata" / "sc_shards.parquet"
    shards = pq.read_table(shard_path, columns=list(E2_SHARD_READ_COLUMNS)).to_pandas()
    shards = shards[~shards["dataset_id"].astype(str).str.startswith("REF-")].reset_index(
        drop=True
    )
    if len(shards) == 0:
        raise E2IndexContractError("strict E2 source contains no non-REF shards")
    if int(shards["dataset_id"].nunique()) != config.expected_datasets:
        raise E2IndexContractError("strict E2 source dataset count differs from configuration")
    if int(shards["n_cells"].sum()) != config.expected_cells:
        raise E2IndexContractError("strict E2 source cell count differs from configuration")
    if shards["shard_id"].astype(str).duplicated().any():
        raise E2IndexContractError("strict E2 source shard IDs must be unique")

    metadata_root = output_root / "metadata"
    pool_path = metadata_root / "sample_pools.parquet"
    writer = pq.ParquetWriter(pool_path, POOL_SCHEMA, compression="zstd")
    sample_counts: Counter[tuple[str, str, str]] = Counter()
    missing_donor_cells = 0
    pool_rows = 0
    try:
        for shard_number, shard in enumerate(shards.itertuples(index=False), start=1):
            obs_path = _resolve_source_uri(
                shard.obs_part,
                strict_root=source_root,
                base_root=base_root,
                description="strict projected obs",
            )
            obs = pq.read_table(obs_path, columns=list(E2_OBS_READ_COLUMNS)).to_pandas()
            obs = _normalize_obs_groups(obs, shard_id=str(shard.shard_id))
            if len(obs) != int(shard.n_cells):
                raise E2IndexContractError(
                    f"obs row count differs from sc_shards for {shard.shard_id}"
                )
            if not obs["dataset_id"].eq(str(shard.dataset_id)).all():
                raise E2IndexContractError(
                    f"obs dataset_id differs from sc_shards for {shard.shard_id}"
                )
            missing_donor_cells += int(obs["donor_id"].eq("__missing_donor__").sum())
            records: list[dict[str, object]] = []
            for key, group in obs.groupby(
                ["dataset_id", "donor_id", "sample_id"], sort=True, dropna=False
            ):
                dataset_id, donor_id, sample_id = (str(value) for value in key)
                sample_counts[(dataset_id, donor_id, sample_id)] += int(len(group))
                records.append(
                    {
                        "sample_key": _sample_key(dataset_id, donor_id, sample_id),
                        "shard_id": str(shard.shard_id),
                        "row_indices": group["row_index"].to_numpy(
                            dtype=np.int32, copy=False
                        ).tolist(),
                    }
                )
            if records:
                table = pa.Table.from_pylist(records, schema=POOL_SCHEMA)
                writer.write_table(table, row_group_size=64)
                pool_rows += len(records)
            if shard_number % 50 == 0:
                print(
                    f"E2 index source scan {shard_number:,}/{len(shards):,} shards; "
                    f"cells={sum(sample_counts.values()):,}",
                    flush=True,
                )
    finally:
        writer.close()

    sample_rows = [
        {
            "sample_key": _sample_key(dataset_id, donor_id, sample_id),
            "dataset_id": dataset_id,
            "donor_id": donor_id,
            "sample_id": sample_id,
            "platform": platform_for_dataset(dataset_id),
            "n_cells": int(count),
        }
        for (dataset_id, donor_id, sample_id), count in sample_counts.items()
    ]
    samples = pd.DataFrame(sample_rows).sort_values(
        ["platform", "dataset_id", "donor_id", "sample_id"], kind="mergesort"
    ).reset_index(drop=True)
    if int(samples["n_cells"].sum()) != int(shards["n_cells"].sum()):
        raise RuntimeError("E2 sample index does not conserve strict source cells")
    scan_audit = {
        "source_obs_columns_read": list(E2_OBS_READ_COLUMNS),
        "source_shard_columns_read": list(E2_SHARD_READ_COLUMNS),
        "annotation_columns_requested": [],
        "selection_uses_annotation_values": False,
        "n_pool_rows": int(pool_rows),
        "missing_donor_cells_using_sample_fallback": int(missing_donor_cells),
    }
    return shards, samples, scan_audit


def _write_recipe_parts(root: Path, recipes: pd.DataFrame) -> pd.DataFrame:
    part_rows: list[dict[str, object]] = []
    train = recipes[recipes["split"].astype(str).eq("train")]
    for slot, frame in train.groupby("schedule_slot", sort=True):
        relative = f"recipes/train-slot-{int(slot):03d}.parquet"
        frame.loc[:, list(E2_RECIPE_COLUMNS)].to_parquet(root / relative, index=False)
        part_rows.append(
            {"recipe_part": relative, "split": "train", "schedule_slot": int(slot), "rows": len(frame)}
        )
    other = recipes[~recipes["split"].astype(str).eq("train")]
    for split, frame in other.groupby("split", sort=True):
        relative = f"recipes/{split}.parquet"
        frame.loc[:, list(E2_RECIPE_COLUMNS)].to_parquet(root / relative, index=False)
        part_rows.append(
            {"recipe_part": relative, "split": str(split), "schedule_slot": -1, "rows": len(frame)}
        )
    manifest = pd.DataFrame(part_rows)
    manifest.to_parquet(root / "metadata" / "recipe_parts.parquet", index=False)
    return manifest


def _existing_index_matches(root: Path, source_fingerprint: str, signature: str) -> dict[str, Any] | None:
    complete = root / "COMPLETE"
    manifest_path = root / "dataset_manifest.json"
    if not root.exists():
        return None
    if not complete.is_file() or not manifest_path.is_file():
        raise FileExistsError(f"partial E2 index exists; inspect or archive it: {root}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        manifest.get("schema_version") == E2_INDEX_SCHEMA
        and manifest.get("source_fingerprint") == source_fingerprint
        and manifest.get("build_config_signature") == signature
        and complete.read_text(encoding="utf-8").strip()
        == str(manifest.get("corpus_fingerprint", ""))
    ):
        return manifest
    raise FileExistsError(
        f"existing E2 index was built from different source/config; archive it first: {root}"
    )


def build_index(config_path: Path) -> dict[str, Any]:
    raw_config, config = _load_config(config_path)
    source_manifest, _, source_fingerprint = _strict_source_metadata(config)
    gene_panel_path = config.gene_panel_path.expanduser().resolve()
    if not gene_panel_path.is_file():
        raise FileNotFoundError(f"configured E2 gene panel not found: {gene_panel_path}")
    gene_panel_sha256 = _sha256_file(gene_panel_path)
    relevant_config = {
        "source_root": str(config.source_root.expanduser().resolve()),
        "base_source_root": str(config.base_source_root.expanduser().resolve()),
        "gene_panel_sha256": gene_panel_sha256,
        "seed": config.seed,
        "split_seed": config.split_seed,
        "split_fractions": config.split_fractions,
        "platforms": config.platforms,
        "platform_sampling_policy": config.platform_sampling_policy,
        "sampling_policy": config.sampling_policy,
        "k_values": config.k_values,
        "k_weights": config.k_weights,
        "train_recipe_pool_size": config.train_recipe_pool_size,
        "train_recipes_per_epoch": config.train_recipes_per_epoch,
        "validation_recipes": config.validation_recipes,
        "validation_recipes_per_platform": config.validation_recipes_per_platform,
        "global_validation_contract": E2_GLOBAL_VALIDATION_CONTRACT,
        "expected_datasets": config.expected_datasets,
        "expected_cells": config.expected_cells,
    }
    signature = _digest(relevant_config)
    output_root = config.index_root.expanduser().resolve()
    existing = _existing_index_matches(output_root, source_fingerprint, signature)
    if existing is not None:
        print(f"E2 index already complete and compatible: {output_root}", flush=True)
        return existing

    temporary = output_root.with_name(f".{output_root.name}.tmp-{os.getpid()}")
    if temporary.exists():
        shutil.rmtree(temporary)
    (temporary / "metadata").mkdir(parents=True)
    (temporary / "recipes").mkdir(parents=True)
    try:
        shards, samples, scan_audit = _scan_source(config, temporary)
        samples, donor_splits = _assign_donor_splits(
            samples,
            fractions=config.split_fractions,
            seed=config.split_seed,
        )
        samples["eligible_k_values_json"] = samples["n_cells"].map(
            lambda count: _canonical_json([k for k in config.k_values if int(count) >= k])
        )
        eligible_train = samples[
            samples["split"].astype(str).eq("train")
            & samples["eligible_k_values_json"].ne("[]")
        ]
        missing_train_datasets = sorted(
            set(samples["dataset_id"]) - set(eligible_train["dataset_id"])
        )
        if missing_train_datasets:
            raise E2IndexContractError(
                f"datasets without a K-eligible train sample: {missing_train_datasets}"
            )

        corpus_platform_cells = {
            platform: int(count)
            for platform, count in samples.groupby("platform")["n_cells"].sum().items()
        }
        if set(corpus_platform_cells) != set(config.platforms):
            raise E2IndexContractError("strict source does not contain all four E2 platforms")
        platform_quotas = _integer_quotas(
            config.train_recipes_per_epoch,
            config.platforms,
            [np.sqrt(corpus_platform_cells[p]) for p in config.platforms],
        )
        k_quotas = _integer_quotas(
            config.train_recipes_per_epoch, config.k_values, config.k_weights
        )
        joint = _joint_quotas(platform_quotas, k_quotas)
        candidates = _candidate_cache(samples, config.k_values)
        for platform, k in joint:
            if joint[(platform, k)] > 0 and ("train", platform, k) not in candidates:
                raise E2IndexContractError(
                    f"train source lacks platform={platform}, K={k} required by quotas"
                )

        schedule_slots = config.train_recipe_pool_size // config.train_recipes_per_epoch
        if schedule_slots * config.train_recipes_per_epoch != config.train_recipe_pool_size:
            raise E2IndexContractError(
                "train_recipe_pool_size must be divisible by recipes_per_epoch"
            )
        train_rows: list[dict[str, Any]] = []
        heldout_false = {platform: False for platform in config.platforms}
        train_scope = {platform: "train_dataset_donor" for platform in config.platforms}
        for slot in range(schedule_slots):
            train_rows.extend(
                _draw_recipe_rows(
                    split="train",
                    schedule_slot=slot,
                    joint_quotas=joint,
                    candidates=candidates,
                    seed=config.seed,
                    global_start=len(train_rows),
                    heldout_by_platform=heldout_false,
                    scope_by_platform=train_scope,
                )
            )

        evaluation_rows: list[dict[str, Any]] = []
        global_index = len(train_rows)
        validation_contract: dict[str, dict[str, object]] = {}
        for split in ("val", "test"):
            for platform in config.platforms:
                if platform == "ScaleBio":
                    continue
                available = [
                    k for k in config.k_values if (split, platform, k) in candidates
                ]
                quotas = _effective_k_quotas(
                    config.validation_recipes_per_platform,
                    config.k_values,
                    config.k_weights,
                    available,
                )
                joint_one = {(platform, k): count for k, count in quotas.items()}
                rows = _draw_recipe_rows(
                    split=split,
                    schedule_slot=-1,
                    joint_quotas=joint_one,
                    candidates=candidates,
                    seed=config.seed,
                    global_start=global_index,
                    heldout_by_platform={platform: True},
                    scope_by_platform={platform: "heldout_dataset_donor"},
                )
                evaluation_rows.extend(rows)
                global_index += len(rows)

        scale = "ScaleBio"
        available_scale = [
            k for k in config.k_values if ("diagnostic", scale, k) in candidates
        ]
        scale_quotas = _effective_k_quotas(
            config.validation_recipes_per_platform,
            config.k_values,
            config.k_weights,
            available_scale,
        )
        scale_rows = _draw_recipe_rows(
            split="diagnostic",
            schedule_slot=-1,
            joint_quotas={(scale, k): count for k, count in scale_quotas.items()},
            candidates=candidates,
            seed=config.seed,
            global_start=global_index,
            heldout_by_platform={scale: False},
            scope_by_platform={scale: "in_corpus_diagnostic_singleton_donor"},
        )
        evaluation_rows.extend(scale_rows)

        recipes = pd.DataFrame(train_rows + evaluation_rows)
        recipes = recipes.loc[:, list(E2_RECIPE_COLUMNS)]
        if recipes["recipe_id"].duplicated().any():
            raise RuntimeError("generated E2 recipe IDs are not unique")
        train_coverage = set(
            recipes.loc[recipes["split"].eq("train"), "dataset_id"].astype(str)
        )
        if len(train_coverage) != config.expected_datasets:
            raise E2IndexContractError(
                f"train recipes cover {len(train_coverage)} datasets, expected {config.expected_datasets}"
            )

        index_source_shards_path = temporary / "metadata" / "source_shards.parquet"
        sample_index_path = temporary / "metadata" / "sample_index.parquet"
        donor_splits_path = temporary / "metadata" / "donor_splits.parquet"
        shards.to_parquet(index_source_shards_path, index=False)
        samples.to_parquet(sample_index_path, index=False)
        donor_splits.to_parquet(donor_splits_path, index=False)
        pd.DataFrame(
            [
                {"schedule_slot": slot, "platform": platform, "recipes": count}
                for slot in range(schedule_slots)
                for platform, count in platform_quotas.items()
            ]
        ).to_parquet(temporary / "metadata" / "platform_quotas.parquet", index=False)
        pd.DataFrame(
            [
                {"schedule_slot": slot, "k": k, "recipes": count}
                for slot in range(schedule_slots)
                for k, count in k_quotas.items()
            ]
        ).to_parquet(temporary / "metadata" / "k_quotas.parquet", index=False)
        pd.DataFrame(
            [
                {
                    "schedule_slot": slot,
                    "platform": platform,
                    "k": k,
                    "recipes": count,
                }
                for slot in range(schedule_slots)
                for (platform, k), count in joint.items()
            ]
        ).to_parquet(temporary / "metadata" / "joint_quotas.parquet", index=False)
        recipe_parts = _write_recipe_parts(temporary, recipes)

        sample_pool_path = temporary / "metadata" / "sample_pools.parquet"
        sample_pools_sha256 = _sha256_file(sample_pool_path)
        sample_index_sha256 = _sha256_file(sample_index_path)
        donor_splits_sha256 = _sha256_file(donor_splits_path)
        index_source_shards_sha256 = _sha256_file(index_source_shards_path)
        recipe_parts_manifest_path = temporary / "metadata" / "recipe_parts.parquet"
        recipe_parts_manifest_sha256 = _sha256_file(recipe_parts_manifest_path)
        recipe_part_files = [
            {
                "recipe_part": str(row.recipe_part),
                "split": str(row.split),
                "schedule_slot": int(row.schedule_slot),
                "rows": int(row.rows),
                "sha256": _sha256_file(temporary / str(row.recipe_part)),
            }
            for row in recipe_parts.sort_values(
                ["split", "schedule_slot", "recipe_part"], kind="mergesort"
            ).itertuples(index=False)
        ]
        recipe_parts_content_sha256 = _digest(recipe_part_files)

        recipe_counts: dict[str, dict[str, int]] = defaultdict(dict)
        for (split, platform), count in recipes.groupby(["split", "platform"]).size().items():
            recipe_counts[str(split)][str(platform)] = int(count)
        for platform in config.platforms:
            if platform == "ScaleBio":
                validation_contract[platform] = {
                    "split": "diagnostic",
                    "scope": "in_corpus_diagnostic_singleton_donor",
                    "heldout": False,
                    "heldout_available": False,
                }
            else:
                validation_contract[platform] = {
                    "split": "val",
                    "scope": "heldout_dataset_donor",
                    "heldout": True,
                    "heldout_available": True,
                }

        source_shard_hash = _sha256_file(
            config.source_root.expanduser().resolve() / "metadata" / "sc_shards.parquet"
        )
        recipe_id_digest = hashlib.sha256()
        for recipe_id in recipes["recipe_id"].astype(str):
            recipe_id_digest.update(recipe_id.encode("ascii"))
            recipe_id_digest.update(b"\n")
        corpus_core = {
            "source_fingerprint": source_fingerprint,
            "source_shards_sha256": source_shard_hash,
            "build_config_signature": signature,
            "gene_panel_sha256": gene_panel_sha256,
            "recipe_ids_sha256": recipe_id_digest.hexdigest(),
            "sample_pools_sha256": sample_pools_sha256,
            "sample_index_sha256": sample_index_sha256,
            "donor_splits_sha256": donor_splits_sha256,
            "index_source_shards_sha256": index_source_shards_sha256,
            "recipe_parts_manifest_sha256": recipe_parts_manifest_sha256,
            "recipe_parts_content_sha256": recipe_parts_content_sha256,
        }
        corpus_fingerprint = _digest(corpus_core)
        bulk_source_cells_per_train_epoch = int(
            sum(int(k) * int(count) for k, count in k_quotas.items())
        )
        model_target_cells_per_train_epoch = int(
            sum(
                min(int(k), int(config.train_cells_per_block)) * int(count)
                for k, count in k_quotas.items()
            )
        )
        manifest: dict[str, Any] = {
            "schema_version": E2_INDEX_SCHEMA,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "source_root": str(config.source_root.expanduser().resolve()),
            "base_source_root": str(config.base_source_root.expanduser().resolve()),
            "source_fingerprint": source_fingerprint,
            "source_shards_sha256": source_shard_hash,
            "index_source_shards_sha256": index_source_shards_sha256,
            "sample_index_sha256": sample_index_sha256,
            "donor_splits_sha256": donor_splits_sha256,
            "recipe_parts_manifest_sha256": recipe_parts_manifest_sha256,
            "recipe_parts_content_sha256": recipe_parts_content_sha256,
            "recipe_part_files": recipe_part_files,
            "source_manifest_schema": source_manifest.get("schema_version"),
            "build_config_signature": signature,
            "corpus_fingerprint": corpus_fingerprint,
            "gene_panel_sha256": gene_panel_sha256,
            "n_datasets": int(shards["dataset_id"].nunique()),
            "n_cells": int(shards["n_cells"].sum()),
            "n_shards": int(len(shards)),
            "recipe_pool_annotation_free": True,
            "sampling_policy": "random",
            "platform_sampling_policy": "sqrt_cells",
            "platform_quota_basis": "sqrt(all_strict_corpus_cells_by_platform)",
            "split_unit": "dataset_donor",
            "split_unit_detail": (
                "dataset_id,donor_id; sample_id fallback when donor_id is missing"
            ),
            "split_fractions": {
                key: value
                for key, value in zip(
                    ("train", "val", "test"), config.split_fractions, strict=True
                )
            },
            "train_recipe_pool_size": int(config.train_recipe_pool_size),
            "train_recipes_per_epoch": int(config.train_recipes_per_epoch),
            "schedule_slots": int(schedule_slots),
            "max_disjoint_epochs": int(schedule_slots),
            "k_values": list(config.k_values),
            "k_weights": list(config.k_weights),
            "bulk_source_cells_per_train_epoch": bulk_source_cells_per_train_epoch,
            "model_target_cells_per_train_epoch": model_target_cells_per_train_epoch,
            "model_target_cells_all_four_epochs": int(
                4 * model_target_cells_per_train_epoch
            ),
            "platform_cell_counts": corpus_platform_cells,
            "platform_quotas_per_epoch": platform_quotas,
            "k_quotas_per_epoch": {str(k): v for k, v in k_quotas.items()},
            "joint_quotas_per_epoch": {
                f"{platform}|{k}": count for (platform, k), count in joint.items()
            },
            "recipe_counts_by_split_platform": dict(recipe_counts),
            "platform_validation_contract": validation_contract,
            "global_validation_contract": E2_GLOBAL_VALIDATION_CONTRACT,
            "sample_pools_sha256": sample_pools_sha256,
            "sample_pools_rows": int(scan_audit["n_pool_rows"]),
            "sample_pools_schema": E2_SAMPLE_POOL_SCHEMA,
            "source_obs_columns_read": list(E2_OBS_READ_COLUMNS),
            "source_shard_columns_read": list(E2_SHARD_READ_COLUMNS),
            "selection_uses_annotation_values": False,
            "annotation_fields_never_requested_by_index_builder": True,
            "train_dataset_coverage": int(len(train_coverage)),
            "corpus_core": corpus_core,
        }
        audit = {
            **scan_audit,
            "status": "PASS",
            "corpus_fingerprint": corpus_fingerprint,
            "source_fingerprint": source_fingerprint,
            "donor_split_overlap_count": 0,
            "excluded_ref_datasets": True,
            "scale_bio_train_only": True,
            "scale_bio_heldout_available": False,
            "train_recipe_ids_unique": bool(
                ~recipes.loc[recipes["split"].eq("train"), "recipe_id"].duplicated().any()
            ),
            "max_disjoint_epochs": int(schedule_slots),
            "exact_platform_quota_each_slot": True,
            "exact_k_quota_each_slot": True,
        }
        (temporary / "dataset_manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True, default=_json_default) + "\n",
            encoding="utf-8",
        )
        (temporary / "audit.json").write_text(
            json.dumps(audit, indent=2, sort_keys=True, default=_json_default) + "\n",
            encoding="utf-8",
        )
        (temporary / "COMPLETE").write_text(corpus_fingerprint + "\n", encoding="utf-8")
        output_root.parent.mkdir(parents=True, exist_ok=True)
        os.replace(temporary, output_root)
        print(
            f"built E2 index: {output_root} datasets={manifest['n_datasets']} "
            f"cells={manifest['n_cells']:,} train_recipes={manifest['train_recipe_pool_size']:,}",
            flush=True,
        )
        return manifest
    except Exception:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    result = build_index(args.config)
    print(json.dumps(result, indent=2, sort_keys=True, default=_json_default))


if __name__ == "__main__":
    main()
