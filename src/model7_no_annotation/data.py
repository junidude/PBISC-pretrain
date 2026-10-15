"""Annotation-free E1 data contract built on the historical B048 recipe pool.

E1 is a *model-side* annotation ablation.  It intentionally reuses B048's
10x recipe scope so that removing annotation inputs is the primary change.
The underlying recipe corpus was historically filtered with an L2 annotation
mask and some recipe policies used L1 labels.  Consequently, this module must
not be described as an annotation-free data-curation pipeline; that stricter
contract belongs to the later full-data experiment.

No public sample emitted by this module contains a cell label, cell-type
proportion, marker program, or annotation-derived condition feature.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
import multiprocessing as mp
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import torch
from torch.utils.data import IterableDataset, get_worker_info


PROJECT_ROOT = Path(__file__).resolve().parents[3]
METHOD1_ROOT = PROJECT_ROOT / "model" / "vae_bulk2sc"
DEFAULT_RECIPE_ROOT = METHOD1_ROOT / "data" / "blood-kmatrix-recipes-v1-10x"
DEFAULT_SOURCE_ROOT = PROJECT_ROOT / "single-cell-data-block-blood"
DEFAULT_GENE_PANEL_PATH = DEFAULT_RECIPE_ROOT / "metadata" / "var.parquet"

E1_RECIPE_SCOPE = "B048_10x_historical_L2_filtered"
E1_RECIPE_LIMITATION = (
    "E1 removes annotation at the model/data-batch boundary only. Its reused "
    "B048 10x recipe manifest was historically curated with "
    "celltype_l2_loss_mask, and the pinned Method 1 L1 lazy-loader pool applies "
    "celltype_l1_loss_mask. E1 excludes label-aware recipe policies but is not "
    "a fully annotation-free source-cell pool."
)

ANNOTATION_FREE_SAMPLE_KEYS = frozenset(
    {"x_log", "condition", "block_index", "block_id", "recipe_k"}
)
ANNOTATION_FREE_RECIPE_POLICY = "random"
RECIPE_AUDIT_COLUMNS = (
    "sampling_policy",
    "target_l1_labels_json",
    "target_l1_counts_json",
    "n_target_l1_labels",
)
FORBIDDEN_NAME_FRAGMENTS = (
    "annotation",
    "celltype",
    "cell_type",
    "proportion",
    "prop::",
    "prop_",
    "marker",
    "program",
)
FORBIDDEN_EXACT_NAMES = frozenset({"y", "label", "labels", "prop", "props"})


class AnnotationLeakageError(ValueError):
    """Raised when an annotation-derived field crosses the E1 batch boundary."""


class RecipeDataModuleProtocol(Protocol):
    """Subset of Method 1's lazy recipe module used by the E1 adapter."""

    gene_panel: Any
    recipe_root: Path
    recipe_parts: list[str]

    def split_manifest(
        self, split: str, max_blocks: int | None = None, seed: int = 0
    ) -> Any: ...

    def prime_pools(self, recipes: Any) -> None: ...

    gene_indices: np.ndarray

    def _recipe_refs_and_pools(
        self, recipe: Any
    ) -> tuple[list[tuple[int, int]], list[dict[str, Any]], dict[int, list[tuple[int, int]]]]: ...

    def _sum_panel_rows_from_refs(
        self,
        pools: list[dict[str, Any]],
        by_segment: dict[int, list[tuple[int, int]]],
    ) -> np.ndarray: ...

    def _load_sparse_refs(
        self,
        refs: list[tuple[int, int]],
        pools: list[dict[str, Any]],
    ) -> Any: ...


@dataclass(frozen=True)
class AnnotationFreeDataConfig:
    """Pinned E1 recipe and streaming settings.

    Defaults match the B048 10x exposure recipe.  ``recipe_layer='L1'`` is
    retained only to reproduce the already-curated B048 source pool; L1 is
    never yielded to the model.
    """

    project_root: Path = METHOD1_ROOT
    recipe_root: Path = DEFAULT_RECIPE_ROOT
    source_root: Path = DEFAULT_SOURCE_ROOT
    gene_panel_path: Path = DEFAULT_GENE_PANEL_PATH
    recipe_layer: str = "L1"
    train_split: str = "train"
    validation_split: str = "val"
    train_recipes_per_epoch: int = 8192
    validation_recipes: int = 256
    fit_blocks: int = 2048
    train_cells_per_block: int = 512
    validation_cells_per_block: int = 64
    seed: int = 20260627
    condition_mode: str = "bio_bulk_depth"
    sampling_policy: str = ANNOTATION_FREE_RECIPE_POLICY
    progress: bool = True
    recipe_scope: str = field(default=E1_RECIPE_SCOPE, init=False)
    recipe_pool_annotation_free: bool = field(default=False, init=False)
    historical_row_filter_layer: str = field(default="L2", init=False)
    runtime_lazy_pool_filter_layer: str = field(default="L1", init=False)

    def __post_init__(self) -> None:
        for field_name in ("project_root", "recipe_root", "source_root", "gene_panel_path"):
            object.__setattr__(self, field_name, Path(getattr(self, field_name)).expanduser())
        if self.recipe_layer.upper() != "L1":
            raise ValueError("E1 must keep recipe_layer='L1' to match B048's historical pool")
        if self.condition_mode != "bio_bulk_depth":
            raise ValueError("E1 only permits annotation-free condition_mode='bio_bulk_depth'")
        if self.sampling_policy != ANNOTATION_FREE_RECIPE_POLICY:
            raise ValueError("E1 only permits sampling_policy='random'")
        for field_name in (
            "train_recipes_per_epoch",
            "validation_recipes",
            "fit_blocks",
            "train_cells_per_block",
            "validation_cells_per_block",
        ):
            if int(getattr(self, field_name)) <= 0:
                raise ValueError(f"{field_name} must be positive")
        if not self.train_split or not self.validation_split:
            raise ValueError("train and validation split names must be non-empty")
        if self.train_split == self.validation_split:
            raise ValueError("train_split and validation_split must differ")

    def audit_metadata(self) -> dict[str, object]:
        """Machine-readable disclosure for run manifests and reports."""

        return {
            "experiment_stage": "E1",
            "recipe_scope": self.recipe_scope,
            "model_batch_annotation_free": True,
            "recipe_pool_annotation_free": self.recipe_pool_annotation_free,
            "historical_row_filter_layer": self.historical_row_filter_layer,
            "runtime_lazy_pool_filter_layer": self.runtime_lazy_pool_filter_layer,
            "historical_recipe_policy_uses_celltype_labels": True,
            "selected_sampling_policy": self.sampling_policy,
            "selected_recipe_targets_empty": True,
            "limitation": E1_RECIPE_LIMITATION,
        }


def audit_annotation_free_recipe_manifest(recipes: Any) -> None:
    """Reject recipe rows selected through a label-aware sampling policy."""

    missing = sorted(set(RECIPE_AUDIT_COLUMNS).difference(recipes.columns))
    if missing:
        raise AnnotationLeakageError(f"recipe manifest is missing audit columns: {missing}")
    if recipes.empty:
        return
    policy = recipes["sampling_policy"].astype(str)
    labels = recipes["target_l1_labels_json"].fillna("").astype(str)
    counts = recipes["target_l1_counts_json"].fillna("").astype(str)
    target_n = recipes["n_target_l1_labels"].fillna(-1).astype(int)
    valid = (
        policy.eq(ANNOTATION_FREE_RECIPE_POLICY)
        & labels.eq("[]")
        & counts.eq("[]")
        & target_n.eq(0)
    )
    if not bool(valid.all()):
        invalid = recipes.loc[
            ~valid,
            [
                column
                for column in (
                    "block_id",
                    "sampling_policy",
                    "target_l1_labels_json",
                    "target_l1_counts_json",
                    "n_target_l1_labels",
                )
                if column in recipes.columns
            ],
        ].head(5)
        raise AnnotationLeakageError(
            "E1 recipes must use random sampling with empty L1 targets; "
            f"invalid examples={invalid.to_dict(orient='records')}"
        )


class AnnotationFreeRecipeDataModule:
    """Policy-filtered view over Method 1's lazy recipe data module.

    Filtering occurs while reading every parquet part and *before* the final
    deterministic sample.  Calling Method 1 ``split_manifest(max_blocks=N)``
    first would be incorrect because it samples a mixture of label-aware and
    random policies before this adapter could inspect the rows.
    """

    def __init__(self, base: RecipeDataModuleProtocol) -> None:
        self.base = base
        self.gene_panel = base.gene_panel
        self.recipe_root = Path(base.recipe_root)
        self.recipe_parts = list(base.recipe_parts)
        self._valid_counts_by_split: dict[str, list[int]] | None = None

    def __getattr__(self, name: str) -> Any:
        return getattr(self.base, name)

    @staticmethod
    def _valid_recipe_mask(frame: Any, split: str) -> Any:
        return (
            frame["split"].astype(str).eq(str(split))
            & frame["sampling_policy"].astype(str).eq(
                ANNOTATION_FREE_RECIPE_POLICY
            )
            & frame["target_l1_labels_json"].fillna("").astype(str).eq("[]")
            & frame["target_l1_counts_json"].fillna("").astype(str).eq("[]")
            & frame["n_target_l1_labels"].fillna(-1).astype(int).eq(0)
        )

    def _load_valid_counts(self) -> dict[str, list[int]]:
        """Cache lightweight valid-policy counts before worker streaming starts."""

        if self._valid_counts_by_split is not None:
            return self._valid_counts_by_split
        import pandas as pd

        counts_per_part: list[dict[str, int]] = []
        observed_splits: set[str] = set()
        columns = ["split", *RECIPE_AUDIT_COLUMNS]
        for part in self.recipe_parts:
            frame = pd.read_parquet(self.recipe_root / part, columns=columns)
            part_counts: dict[str, int] = {}
            for split in frame["split"].astype(str).unique():
                count = int(self._valid_recipe_mask(frame, str(split)).sum())
                if count > 0:
                    part_counts[str(split)] = count
                    observed_splits.add(str(split))
            counts_per_part.append(part_counts)
        self._valid_counts_by_split = {
            split: [int(counts.get(split, 0)) for counts in counts_per_part]
            for split in sorted(observed_splits)
        }
        return self._valid_counts_by_split

    def valid_recipe_counts(self) -> dict[str, int]:
        """Return audited random/empty-target recipe counts by split.

        Calling this once in the parent trainer also warms the lightweight
        per-part index before persistent DataLoader workers are forked.
        """

        return {
            split: int(sum(counts))
            for split, counts in self._load_valid_counts().items()
        }

    def split_manifest(
        self, split: str, max_blocks: int | None = None, seed: int = 0
    ) -> Any:
        import pandas as pd

        counts = np.asarray(
            self._load_valid_counts().get(str(split), []), dtype=np.int64
        )
        available = int(counts.sum())
        if available <= 0:
            return pd.DataFrame()
        target = available if max_blocks is None else min(max(int(max_blocks), 0), available)
        if target == 0:
            return pd.DataFrame()
        weights = counts.astype(np.float64)
        raw_allocations = weights / weights.sum() * target
        allocations = np.floor(raw_allocations).astype(np.int64)
        remainder = int(target - allocations.sum())
        if remainder:
            order = np.argsort(-(raw_allocations - allocations), kind="mergesort")
            allocations[order[:remainder]] += 1

        rng = np.random.default_rng(int(seed))
        frames: list[Any] = []
        for part, needed in zip(self.recipe_parts, allocations, strict=True):
            needed = int(needed)
            if needed <= 0:
                continue
            frame = pd.read_parquet(self.recipe_root / part)
            selected = frame.loc[self._valid_recipe_mask(frame, str(split))]
            if len(selected) < needed:
                raise RuntimeError(
                    f"cached E1 recipe count exceeds observed rows in {part}: "
                    f"needed={needed}, observed={len(selected)}"
                )
            part_seed = int(rng.integers(0, 2**31 - 1))
            frames.append(
                selected.sample(n=needed, replace=False, random_state=part_seed)
            )
        if not frames:
            return pd.DataFrame()
        candidates = pd.concat(frames, ignore_index=True).sample(
            frac=1.0, random_state=int(seed) + 17
        ).reset_index(drop=True)
        candidates["stream_recipe_index"] = np.arange(
            len(candidates), dtype=np.int64
        )
        audit_annotation_free_recipe_manifest(candidates)
        return candidates

    def prime_pools(self, recipes: Any) -> None:
        audit_annotation_free_recipe_manifest(recipes)
        self.base.prime_pools(recipes)

    def load_training_block(
        self,
        row_or_block_id: Any,
        cells_per_block: int,
        rng: np.random.Generator,
    ) -> Any:
        try:
            import pandas as pd

            audit_annotation_free_recipe_manifest(
                pd.DataFrame([row_or_block_id.to_dict()])
            )
        except AttributeError as error:
            raise AnnotationLeakageError(
                "E1 load_training_block requires an audited recipe row"
            ) from error
        recipe = row_or_block_id
        required_methods = (
            "_recipe_refs_and_pools",
            "_sum_panel_rows_from_refs",
            "_load_sparse_refs",
        )
        missing = [name for name in required_methods if not callable(getattr(self.base, name, None))]
        if missing:
            raise RuntimeError(
                "Method 1 recipe API is incompatible with annotation-free thin loading; "
                f"missing methods={missing}"
            )
        # Deliberately bypass base.load_training_block: that upstream method
        # reads label-bearing obs, derives proportions, and filters cells by a
        # valid cell-type ID.  E1 samples uniformly from the already-curated
        # historical pool's row references without opening annotation columns.
        refs, pools, by_segment = self.base._recipe_refs_and_pools(recipe)
        bulk = np.log1p(
            self.base._sum_panel_rows_from_refs(pools, by_segment)
        ).astype(np.float32)
        take = min(max(int(cells_per_block), 0), len(refs))
        if take > 0:
            chosen = rng.choice(len(refs), size=take, replace=False)
            selected_refs = [refs[int(index)] for index in chosen]
            x = (
                self.base._load_sparse_refs(selected_refs, pools)[:, self.base.gene_indices]
                .toarray()
                .astype(np.float32, copy=False)
            )
        else:
            x = np.empty((0, int(len(self.base.gene_indices))), dtype=np.float32)
        return AnnotationFreeRecipeBlock(
            block_id=str(recipe["block_id"]),
            x=x,
            bulk=bulk,
            n_cells=int(recipe["k"]),
        )


@dataclass(frozen=True)
class AnnotationFreeRecipeBlock:
    """Internal thin block intentionally lacking labels and proportions."""

    block_id: str
    x: np.ndarray
    bulk: np.ndarray
    n_cells: int


def _is_forbidden_name(name: object) -> bool:
    normalized = str(name).strip().lower().replace("-", "_").replace(" ", "_")
    if normalized in FORBIDDEN_EXACT_NAMES:
        return True
    return any(fragment in normalized for fragment in FORBIDDEN_NAME_FRAGMENTS)


def audit_annotation_free_names(names: Iterable[object], *, context: str) -> None:
    """Reject condition/schema names that imply an annotation-derived signal."""

    leaked = sorted({str(name) for name in names if _is_forbidden_name(name)})
    if leaked:
        raise AnnotationLeakageError(f"{context} contains forbidden annotation fields: {leaked}")


def _assert_finite_array(value: object, *, name: str, allowed_ndim: tuple[int, ...]) -> None:
    if torch.is_tensor(value):
        if int(value.ndim) not in allowed_ndim:
            raise ValueError(f"{name} must have ndim in {allowed_ndim}, got {value.ndim}")
        if not bool(torch.isfinite(value).all()):
            raise ValueError(f"{name} contains NaN or Inf")
        return
    array = np.asarray(value)
    if int(array.ndim) not in allowed_ndim:
        raise ValueError(f"{name} must have ndim in {allowed_ndim}, got {array.ndim}")
    if not bool(np.isfinite(array).all()):
        raise ValueError(f"{name} contains NaN or Inf")


def audit_annotation_free_batch(
    batch: Mapping[str, object],
    *,
    condition_feature_names: Sequence[str] | None = None,
) -> None:
    """Validate the exact E1 sample/batch schema and screen for label leakage.

    The function accepts either one uncollated sample (1-D expression and
    condition) or a default-PyTorch-collated batch (2-D tensors).
    """

    audit_annotation_free_names(batch.keys(), context="batch")
    actual_keys = frozenset(str(key) for key in batch)
    if actual_keys != ANNOTATION_FREE_SAMPLE_KEYS:
        missing = sorted(ANNOTATION_FREE_SAMPLE_KEYS - actual_keys)
        extra = sorted(actual_keys - ANNOTATION_FREE_SAMPLE_KEYS)
        raise AnnotationLeakageError(
            f"annotation-free batch keys must be exactly {sorted(ANNOTATION_FREE_SAMPLE_KEYS)}; "
            f"missing={missing}, extra={extra}"
        )
    if condition_feature_names is not None:
        audit_annotation_free_names(condition_feature_names, context="condition features")

    _assert_finite_array(batch["x_log"], name="x_log", allowed_ndim=(1, 2))
    _assert_finite_array(batch["condition"], name="condition", allowed_ndim=(1, 2))
    x_shape = tuple(batch["x_log"].shape)  # type: ignore[union-attr]
    condition_shape = tuple(batch["condition"].shape)  # type: ignore[union-attr]
    if not x_shape or x_shape[-1] <= 0 or not condition_shape or condition_shape[-1] <= 0:
        raise ValueError("x_log and condition must have non-empty feature dimensions")
    if len(x_shape) != len(condition_shape) or (len(x_shape) == 2 and x_shape[0] != condition_shape[0]):
        raise ValueError("x_log and condition batch dimensions do not match")

    recipe_k = batch["recipe_k"]
    if torch.is_tensor(recipe_k):
        if not bool((recipe_k > 0).all()):
            raise ValueError("recipe_k must be positive")
    elif not bool((np.asarray(recipe_k) > 0).all()):
        raise ValueError("recipe_k must be positive")

    block_id = batch["block_id"]
    block_ids = [block_id] if isinstance(block_id, str) else list(block_id)  # type: ignore[arg-type]
    if not block_ids or any(not isinstance(value, str) or not value for value in block_ids):
        raise ValueError("block_id must contain non-empty strings")


@dataclass
class AnnotationFreeConditioner:
    """Train-fitted bulk/depth conditioner with no label-derived features."""

    n_genes: int
    eps: float = 1.0e-6
    clip_z: float = 8.0
    gene_mean_: np.ndarray | None = field(default=None, init=False, repr=False)
    gene_std_: np.ndarray | None = field(default=None, init=False, repr=False)
    scalar_mean_: np.ndarray | None = field(default=None, init=False, repr=False)
    scalar_std_: np.ndarray | None = field(default=None, init=False, repr=False)
    fitted_: bool = field(default=False, init=False)

    SCALAR_FEATURE_NAMES = (
        "bulk_total_log1p",
        "bulk_per_cell_log1p",
        "bulk_detected_fraction",
        "bulk_log_mean",
        "bulk_log_std",
        "bulk_top1pct_mass_fraction",
    )

    def __post_init__(self) -> None:
        if int(self.n_genes) <= 0:
            raise ValueError("n_genes must be positive")
        if not np.isfinite(self.eps) or self.eps <= 0:
            raise ValueError("eps must be finite and positive")
        if not np.isfinite(self.clip_z) or self.clip_z <= 0:
            raise ValueError("clip_z must be finite and positive")
        audit_annotation_free_names(self.feature_names, context="condition features")

    @property
    def dim(self) -> int:
        return int(self.n_genes) + len(self.SCALAR_FEATURE_NAMES)

    @property
    def feature_names(self) -> list[str]:
        return [f"bulk_gene::{index}" for index in range(int(self.n_genes))] + list(
            self.SCALAR_FEATURE_NAMES
        )

    def _raw_bulk(self, bulk_log1p: np.ndarray) -> np.ndarray:
        bulk = np.asarray(bulk_log1p, dtype=np.float64)
        if bulk.shape != (int(self.n_genes),):
            raise ValueError(
                f"bulk_log1p must have shape ({self.n_genes},), got {tuple(bulk.shape)}"
            )
        if not np.isfinite(bulk).all():
            raise ValueError("bulk_log1p contains NaN or Inf")
        raw = np.expm1(bulk)
        if float(raw.min(initial=0.0)) < -self.eps:
            raise ValueError("bulk_log1p decodes to negative counts")
        return np.maximum(raw, 0.0)

    def _feature_parts(
        self, bulk_log1p: np.ndarray, recipe_k: int
    ) -> tuple[np.ndarray, np.ndarray]:
        k = int(recipe_k)
        if k <= 0:
            raise ValueError("recipe_k must be positive")
        bulk = np.asarray(bulk_log1p, dtype=np.float64)
        raw = self._raw_bulk(bulk)
        total = max(float(raw.sum()), self.eps)
        size_factor = total / float(k)
        normalized_bulk = np.log1p(raw / max(size_factor, self.eps)).astype(np.float64)
        sorted_raw = np.sort(raw)
        top_n = max(1, int(np.ceil(0.01 * sorted_raw.size)))
        scalar = np.asarray(
            [
                np.log1p(total),
                np.log1p(total / float(k)),
                float((raw > 0).mean()),
                float(bulk.mean()),
                float(bulk.std()),
                float(sorted_raw[-top_n:].sum() / total),
            ],
            dtype=np.float64,
        )
        return normalized_bulk, scalar

    def fit(
        self, records: Iterable[tuple[np.ndarray, int]]
    ) -> "AnnotationFreeConditioner":
        """Fit normalization statistics from train-split bulk/K records only."""

        count = 0
        gene_mean = np.zeros(int(self.n_genes), dtype=np.float64)
        gene_m2 = np.zeros_like(gene_mean)
        scalar_mean = np.zeros(len(self.SCALAR_FEATURE_NAMES), dtype=np.float64)
        scalar_m2 = np.zeros_like(scalar_mean)
        for bulk_log1p, recipe_k in records:
            genes, scalars = self._feature_parts(bulk_log1p, recipe_k)
            count += 1
            gene_delta = genes - gene_mean
            gene_mean += gene_delta / count
            gene_m2 += gene_delta * (genes - gene_mean)
            scalar_delta = scalars - scalar_mean
            scalar_mean += scalar_delta / count
            scalar_m2 += scalar_delta * (scalars - scalar_mean)
        if count == 0:
            raise ValueError("cannot fit annotation-free conditioner with no train records")
        self.gene_mean_ = gene_mean.astype(np.float32)
        self.gene_std_ = (np.sqrt(gene_m2 / count) + self.eps).astype(np.float32)
        self.scalar_mean_ = scalar_mean.astype(np.float32)
        self.scalar_std_ = (np.sqrt(scalar_m2 / count) + self.eps).astype(np.float32)
        self.fitted_ = True
        return self

    def transform(self, bulk_log1p: np.ndarray, recipe_k: int) -> np.ndarray:
        if not self.fitted_:
            raise RuntimeError("annotation-free conditioner must be fit on train data first")
        if any(
            value is None
            for value in (self.gene_mean_, self.gene_std_, self.scalar_mean_, self.scalar_std_)
        ):
            raise RuntimeError("annotation-free conditioner normalization state is incomplete")
        genes, scalars = self._feature_parts(bulk_log1p, recipe_k)
        gene_z = (genes - self.gene_mean_) / np.maximum(self.gene_std_, self.eps)
        scalar_z = (scalars - self.scalar_mean_) / np.maximum(self.scalar_std_, self.eps)
        condition = np.concatenate(
            [
                np.clip(gene_z, -self.clip_z, self.clip_z),
                np.clip(scalar_z, -self.clip_z, self.clip_z),
            ]
        ).astype(np.float32, copy=False)
        if condition.shape != (self.dim,) or not np.isfinite(condition).all():
            raise RuntimeError("annotation-free condition has invalid shape or non-finite values")
        return condition

    def state_dict(self) -> dict[str, object]:
        if not self.fitted_:
            raise RuntimeError("cannot serialize an unfitted annotation-free conditioner")
        return {
            "schema_version": "model7.annotation_free_conditioner.v1",
            "condition_mode": "bio_bulk_depth",
            "n_genes": int(self.n_genes),
            "dim": int(self.dim),
            "eps": float(self.eps),
            "clip_z": float(self.clip_z),
            "feature_names": self.feature_names,
            "gene_mean": np.asarray(self.gene_mean_, dtype=np.float32),
            "gene_std": np.asarray(self.gene_std_, dtype=np.float32),
            "scalar_mean": np.asarray(self.scalar_mean_, dtype=np.float32),
            "scalar_std": np.asarray(self.scalar_std_, dtype=np.float32),
        }

    def load_state_dict(self, state: Mapping[str, object]) -> "AnnotationFreeConditioner":
        if state.get("schema_version") != "model7.annotation_free_conditioner.v1":
            raise ValueError("unsupported annotation-free conditioner state schema")
        if int(state.get("n_genes", -1)) != int(self.n_genes):
            raise ValueError("conditioner state n_genes does not match")
        if list(state.get("feature_names", [])) != self.feature_names:
            raise ValueError("conditioner state feature layout does not match")
        expected_shapes = {
            "gene_mean": (int(self.n_genes),),
            "gene_std": (int(self.n_genes),),
            "scalar_mean": (len(self.SCALAR_FEATURE_NAMES),),
            "scalar_std": (len(self.SCALAR_FEATURE_NAMES),),
        }
        values: dict[str, np.ndarray] = {}
        for name, shape in expected_shapes.items():
            value = np.asarray(state.get(name), dtype=np.float32)
            if value.shape != shape or not np.isfinite(value).all():
                raise ValueError(f"conditioner state {name} must have finite shape {shape}")
            values[name] = value
        if (values["gene_std"] <= 0).any() or (values["scalar_std"] <= 0).any():
            raise ValueError("conditioner state standard deviations must be positive")
        self.gene_mean_ = values["gene_mean"]
        self.gene_std_ = values["gene_std"]
        self.scalar_mean_ = values["scalar_mean"]
        self.scalar_std_ = values["scalar_std"]
        self.fitted_ = True
        return self


class AnnotationFreeCellDataset(IterableDataset[dict[str, object]]):
    """Stream E1 cells while discarding all upstream annotation outputs."""

    def __init__(
        self,
        data_module: RecipeDataModuleProtocol,
        *,
        split: str,
        recipes_per_epoch: int,
        cells_per_block: int,
        conditioner: AnnotationFreeConditioner,
        seed: int,
        progress: bool = True,
    ) -> None:
        super().__init__()
        if not conditioner.fitted_:
            raise ValueError("conditioner must be fit before building the cell stream")
        if int(recipes_per_epoch) <= 0 or int(cells_per_block) <= 0:
            raise ValueError("recipes_per_epoch and cells_per_block must be positive")
        self.data_module = data_module
        self.split = str(split)
        self.recipes_per_epoch = int(recipes_per_epoch)
        self.cells_per_block = int(cells_per_block)
        self.conditioner = conditioner
        self.seed = int(seed)
        self.progress = bool(progress)
        # Feature-name auditing is semantic and invariant for the lifetime of
        # the fitted conditioner.  Rebuilding and scanning 20,103 strings for
        # every yielded cell would dominate the real E1 input pipeline.
        self._condition_feature_names = conditioner.feature_names
        audit_annotation_free_names(
            self._condition_feature_names, context="condition features"
        )
        self._shared_epoch = mp.Value("q", 0)

    def set_epoch(self, epoch: int) -> None:
        with self._shared_epoch.get_lock():
            self._shared_epoch.value = int(epoch)

    def __len__(self) -> int:
        return self.recipes_per_epoch * self.cells_per_block

    def __iter__(self) -> Iterator[dict[str, object]]:
        with self._shared_epoch.get_lock():
            epoch = int(self._shared_epoch.value)
        epoch_seed = self.seed + 1_000_003 * epoch
        recipes = self.data_module.split_manifest(
            self.split, max_blocks=self.recipes_per_epoch, seed=epoch_seed
        )
        worker = get_worker_info()
        worker_id = 0
        if worker is not None:
            worker_id = int(worker.id)
            recipes = recipes.iloc[worker_id :: int(worker.num_workers)].reset_index(drop=True)
        self.data_module.prime_pools(recipes)
        rows: Iterable[tuple[object, Any]] = recipes.iterrows()
        if self.progress:
            from tqdm import tqdm

            description = f"annotation-free {self.split} epoch {epoch}"
            if worker is not None:
                description += f" worker {worker_id}"
            rows = tqdm(rows, total=len(recipes), desc=description)
        rng = np.random.default_rng(epoch_seed + 7_919 + 104_729 * worker_id)
        audited_first_sample = False
        for fallback_index, (_, recipe) in enumerate(rows):
            block = self.data_module.load_training_block(
                recipe, self.cells_per_block, rng
            )
            counts = np.asarray(block.x, dtype=np.float32)
            if counts.ndim != 2:
                raise ValueError("upstream recipe block expression must be a 2-D array")
            if counts.shape[0] == 0:
                continue
            if not np.isfinite(counts).all() or float(counts.min(initial=0.0)) < 0:
                raise ValueError("upstream recipe block contains invalid count values")
            recipe_k = int(getattr(block, "n_cells", recipe["k"]))
            block_id = str(getattr(block, "block_id", recipe["block_id"]))
            block_index = np.int64(
                int(recipe.get("stream_recipe_index", fallback_index))
            )
            condition = self.conditioner.transform(
                np.asarray(block.bulk, dtype=np.float32), recipe_k
            )
            for cell_counts in counts:
                sample: dict[str, object] = {
                    "x_log": np.log1p(cell_counts).astype(np.float32, copy=False),
                    "condition": condition,
                    "block_index": block_index,
                    "block_id": block_id,
                    "recipe_k": np.int64(recipe_k),
                }
                # Full semantic/schema audit once per worker/epoch.  The
                # trainer independently audits the first collated batch; all
                # per-block numeric arrays were validated above/transform().
                if not audited_first_sample:
                    audit_annotation_free_batch(
                        sample,
                        condition_feature_names=self._condition_feature_names,
                    )
                    audited_first_sample = True
                yield sample


@dataclass(frozen=True)
class AnnotationFreeDatasets:
    train: AnnotationFreeCellDataset
    validation: AnnotationFreeCellDataset


def build_e1_data_module(config: AnnotationFreeDataConfig) -> AnnotationFreeRecipeDataModule:
    """Construct the pinned Method 1 lazy loader for the historical E1 scope."""

    from method1.kmatrix_recipe_data import KMatrixRecipeDataModule

    base = KMatrixRecipeDataModule(
        project_root=config.project_root,
        recipe_root=config.recipe_root,
        source_root=config.source_root,
        gene_panel_path=config.gene_panel_path,
        layer=config.recipe_layer,
    )
    return AnnotationFreeRecipeDataModule(base)


def iter_annotation_free_fit_records(
    data_module: RecipeDataModuleProtocol,
    config: AnnotationFreeDataConfig,
) -> Iterator[tuple[np.ndarray, int]]:
    """Yield train-only bulk/K records for fitting, never labels/proportions."""

    recipes = data_module.split_manifest(
        config.train_split, max_blocks=config.fit_blocks, seed=config.seed
    )
    data_module.prime_pools(recipes)
    rows: Iterable[tuple[object, Any]] = recipes.iterrows()
    if config.progress:
        from tqdm import tqdm

        rows = tqdm(rows, total=len(recipes), desc="fit annotation-free conditioner")
    rng = np.random.default_rng(config.seed + 31_337)
    for _, recipe in rows:
        # cells_per_block=0 avoids dense cell loading.  The upstream B048 loader
        # still resolves its historical annotated recipe pool, but only bulk and
        # K cross this adapter boundary.
        block = data_module.load_training_block(recipe, 0, rng)
        recipe_k = int(getattr(block, "n_cells", recipe["k"]))
        yield np.asarray(block.bulk, dtype=np.float32), recipe_k


def fit_annotation_free_conditioner(
    data_module: RecipeDataModuleProtocol,
    config: AnnotationFreeDataConfig,
) -> AnnotationFreeConditioner:
    """Fit the E1 conditioner exclusively on the configured train split."""

    n_genes = int(len(data_module.gene_panel))
    conditioner = AnnotationFreeConditioner(n_genes=n_genes)
    conditioner.fit(iter_annotation_free_fit_records(data_module, config))
    return conditioner


def build_annotation_free_datasets(
    data_module: RecipeDataModuleProtocol,
    conditioner: AnnotationFreeConditioner,
    config: AnnotationFreeDataConfig,
) -> AnnotationFreeDatasets:
    """Build matched E1 train/validation streams with different split seeds."""

    train = AnnotationFreeCellDataset(
        data_module,
        split=config.train_split,
        recipes_per_epoch=config.train_recipes_per_epoch,
        cells_per_block=config.train_cells_per_block,
        conditioner=conditioner,
        seed=config.seed,
        progress=config.progress,
    )
    validation = AnnotationFreeCellDataset(
        data_module,
        split=config.validation_split,
        recipes_per_epoch=config.validation_recipes,
        cells_per_block=config.validation_cells_per_block,
        conditioner=conditioner,
        seed=config.seed + 10_007,
        progress=config.progress,
    )
    return AnnotationFreeDatasets(train=train, validation=validation)
