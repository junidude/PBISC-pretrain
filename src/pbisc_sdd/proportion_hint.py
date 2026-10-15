"""Optional cell-type proportion hint appended to the bulk condition.

The annotation-free conditioner feeds the denoiser ``G + 6`` bulk features.  When a run's
``data.proportion_hint`` block is present, a fixed-order vector of ``K`` cell-type fractions
(AIFI L1 by default) is appended to that base condition for every training cell, looked up
by the cell's recipe ``block_id`` -> ``sample_key`` -> per-sample proportion table.  At sampling
time the same ``K`` columns must be supplied per external bulk (``sample_sdd_standalone.py
--proportion-csv``); the checkpoint records the label order so the two cannot drift.

Config (``data.proportion_hint``)::

    {"table_parquet": ".../sample_proportions.parquet",   # sample_key + prop columns
     "column_prefix": "prop_l1::",                        # columns <prefix><label>
     "labels": ["T cell", ...],                           # fixed order (K)
     "recipe_sources": [".../e2_full_index/recipes", ".../topup_recipes.csv"],
     "fallback": "train_mean" | "uniform",                # for blocks whose sample is absent
     "max_unknown_fraction": 0.0}                         # fail-closed on unmapped block_ids

Everything here is CPU-only and side-effect free; nothing changes when the block is absent.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np

DEFAULT_LABELS = ("T cell", "B cell", "Monocyte", "NK cell", "DC", "Platelet", "Erythrocyte", "Progenitor cell", "ILC")
_SHA_BLOCK = 1 << 24


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(_SHA_BLOCK), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class ProportionHintConfig:
    table_parquet: Path
    labels: tuple[str, ...]
    recipe_sources: tuple[Path, ...]
    column_prefix: str = "prop_l1::"
    fallback: str = "train_mean"
    max_unknown_fraction: float = 0.0

    @classmethod
    def from_mapping(cls, raw: Mapping[str, object]) -> "ProportionHintConfig":
        if not isinstance(raw, Mapping):
            raise ValueError("data.proportion_hint must be a mapping")
        labels = tuple(str(x) for x in raw.get("labels", DEFAULT_LABELS))
        if len(labels) < 2 or len(set(labels)) != len(labels):
            raise ValueError("proportion_hint.labels must be >= 2 distinct labels")
        sources = raw.get("recipe_sources")
        if not isinstance(sources, (list, tuple)) or not sources:
            raise ValueError("proportion_hint.recipe_sources must be a non-empty list")
        fallback = str(raw.get("fallback", "train_mean"))
        if fallback not in {"train_mean", "uniform"}:
            raise ValueError("proportion_hint.fallback must be 'train_mean' or 'uniform'")
        frac = float(raw.get("max_unknown_fraction", 0.0))
        if not 0.0 <= frac < 1.0:
            raise ValueError("proportion_hint.max_unknown_fraction must be in [0, 1)")
        return cls(
            table_parquet=Path(str(raw["table_parquet"])).expanduser(),
            labels=labels,
            recipe_sources=tuple(Path(str(p)).expanduser() for p in sources),
            column_prefix=str(raw.get("column_prefix", "prop_l1::")),
            fallback=fallback,
            max_unknown_fraction=frac,
        )


class ProportionHint:
    """block_id -> K-dim proportion vector lookup with a fail-closed unknown policy."""

    def __init__(self, config: ProportionHintConfig) -> None:
        import pandas as pd

        self.config = config
        table = pd.read_parquet(config.table_parquet)
        cols = [f"{config.column_prefix}{label}" for label in config.labels]
        missing = [c for c in cols if c not in table.columns]
        if missing:
            raise ValueError(f"proportion table lacks columns {missing}")
        if "sample_key" not in table.columns:
            raise ValueError("proportion table must carry sample_key")
        values = table[cols].to_numpy(dtype=np.float64)
        if not np.isfinite(values).all() or (values < 0).any():
            raise ValueError("proportion table contains negative or non-finite values")
        row_sum = values.sum(axis=1)
        if (row_sum <= 0).any():
            raise ValueError("proportion table has rows summing to zero")
        values = values / row_sum[:, None]
        self._by_sample: dict[str, np.ndarray] = {
            str(key): vec.astype(np.float32) for key, vec in zip(table["sample_key"].astype(str), values)
        }
        if config.fallback == "train_mean":
            if "split" in table.columns and (table["split"].astype(str) == "train").any():
                fallback = values[(table["split"].astype(str) == "train").to_numpy()].mean(axis=0)
            else:
                fallback = values.mean(axis=0)
        else:
            fallback = np.full(len(config.labels), 1.0 / len(config.labels))
        self.fallback = (fallback / fallback.sum()).astype(np.float32)

        block_to_sample: dict[str, str] = {}
        source_records = []
        for source in config.recipe_sources:
            files = sorted(source.glob("*.parquet")) if source.is_dir() else [source]
            for path in files:
                frame = pd.read_parquet(path, columns=["block_id", "sample_key"]) if path.suffix == ".parquet" else pd.read_csv(path, usecols=["block_id", "sample_key"])
                for block, key in zip(frame["block_id"].astype(str), frame["sample_key"].astype(str)):
                    prev = block_to_sample.get(block)
                    if prev is not None and prev != key:
                        raise ValueError(f"block_id {block} maps to two samples ({prev}, {key})")
                    block_to_sample[block] = key
                source_records.append({"path": str(path), "rows": int(len(frame)), "sha256": _sha256_file(path)})
        self._block_to_sample = block_to_sample
        self._cache: dict[str, np.ndarray] = {}
        self.unknown_blocks: dict[str, int] = {}
        self.rows_seen = 0
        self.rows_unknown = 0
        self.rows_sample_missing = 0
        self._contract = {
            "schema_version": "pbisc_sdd.proportion_hint.v1",
            "labels": list(config.labels),
            "dim": len(config.labels),
            "column_prefix": config.column_prefix,
            "table_parquet": str(config.table_parquet),
            "table_sha256": _sha256_file(config.table_parquet),
            "n_samples": len(self._by_sample),
            "n_blocks_mapped": len(block_to_sample),
            "recipe_sources": source_records,
            "fallback": config.fallback,
            "fallback_vector": self.fallback.astype(float).tolist(),
            "max_unknown_fraction": config.max_unknown_fraction,
            "placement": "appended to the base bulk condition (after the G+6 conditioner features, before the 2G measurement context)",
        }

    # ------------------------------------------------------------------ properties
    @property
    def dim(self) -> int:
        return len(self.config.labels)

    @property
    def labels(self) -> tuple[str, ...]:
        return self.config.labels

    def contract(self) -> dict:
        return dict(self._contract)

    # ------------------------------------------------------------------ lookups
    def vector_for_sample(self, sample_key: str) -> np.ndarray | None:
        return self._by_sample.get(str(sample_key))

    def vectors_for(self, block_ids: Sequence[str]) -> np.ndarray:
        out = np.empty((len(block_ids), self.dim), dtype=np.float32)
        for i, block in enumerate(block_ids):
            block = str(block)
            vec = self._cache.get(block)
            if vec is None:
                key = self._block_to_sample.get(block)
                if key is None:
                    self.rows_unknown += 1
                    self.unknown_blocks[block] = self.unknown_blocks.get(block, 0) + 1
                    vec = self.fallback
                else:
                    vec = self._by_sample.get(key)
                    if vec is None:
                        self.rows_sample_missing += 1
                        vec = self.fallback
                if len(self._cache) < 200_000:
                    self._cache[block] = vec
            out[i] = vec
        self.rows_seen += len(block_ids)
        if self.rows_seen >= 4096:
            frac = (self.rows_unknown + self.rows_sample_missing) / max(self.rows_seen, 1)
            if frac > self.config.max_unknown_fraction:
                raise RuntimeError(
                    f"proportion hint: {frac:.4%} of rows fell back (unknown block_ids {self.rows_unknown}, "
                    f"samples without proportions {self.rows_sample_missing}); e.g. {list(self.unknown_blocks)[:3]}"
                )
        return out

    def verify_coverage(self, block_ids: Iterable[str], sample_keys: Iterable[str]) -> dict:
        """Fail fast before training: every recipe block and every sample must be resolvable."""
        blocks = [str(b) for b in block_ids]
        keys = [str(k) for k in sample_keys]
        unknown_blocks = [b for b in blocks if b not in self._block_to_sample]
        missing_samples = sorted({k for k in keys if k not in self._by_sample})
        report = {
            "blocks_checked": len(blocks),
            "unknown_blocks": len(unknown_blocks),
            "samples_checked": len(set(keys)),
            "samples_without_proportions": len(missing_samples),
            "examples_unknown_blocks": unknown_blocks[:5],
            "examples_missing_samples": missing_samples[:5],
        }
        frac = (len(unknown_blocks) + sum(1 for k in keys if k in set(missing_samples))) / max(len(blocks), 1)
        if frac > self.config.max_unknown_fraction:
            raise RuntimeError(f"proportion hint coverage check failed: {json.dumps(report)}")
        return report

    def stats(self) -> dict:
        return {
            "rows_seen": self.rows_seen,
            "rows_unknown_block": self.rows_unknown,
            "rows_sample_missing": self.rows_sample_missing,
            "distinct_unknown_blocks": len(self.unknown_blocks),
        }

    # ------------------------------------------------------------------ factory
    @classmethod
    def from_data_section(cls, data: Mapping[str, object]) -> "ProportionHint | None":
        raw = data.get("proportion_hint") if isinstance(data, Mapping) else None
        if raw is None:
            return None
        return cls(ProportionHintConfig.from_mapping(raw))


def load_hint_csv(path: Path | str, labels: Sequence[str], *, sample_ids: Sequence[str]) -> np.ndarray:
    """Read an inference-time hint table (``sample_id`` + one column per label) aligned to ``sample_ids``."""
    import pandas as pd

    frame = pd.read_csv(path)
    if "sample_id" not in frame.columns:
        raise ValueError("proportion CSV must carry a sample_id column")
    missing = [lab for lab in labels if lab not in frame.columns]
    if missing:
        raise ValueError(f"proportion CSV lacks label columns {missing}")
    extra_labels = [c for c in frame.columns if c != "sample_id" and c not in labels]
    if extra_labels:
        raise ValueError(f"proportion CSV has label columns outside the checkpoint contract: {extra_labels[:5]}")
    frame = frame.set_index(frame["sample_id"].astype(str))
    absent = [s for s in sample_ids if str(s) not in frame.index]
    if absent:
        raise ValueError(f"proportion CSV lacks rows for {len(absent)} bulks, e.g. {absent[:3]}")
    values = frame.loc[[str(s) for s in sample_ids], list(labels)].to_numpy(dtype=np.float64)
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("proportion CSV contains negative or non-finite values")
    sums = values.sum(axis=1)
    if (sums <= 0).any() or (np.abs(sums - 1.0) > 0.02).any():
        raise ValueError("proportion CSV rows must sum to 1 (+-0.02)")
    return (values / sums[:, None]).astype(np.float32)


__all__ = ["DEFAULT_LABELS", "ProportionHint", "ProportionHintConfig", "load_hint_csv"]
