"""Measurement-aware PBISC input pipeline for sparse data diffusion.

This module deliberately reuses Model 7's sealed E2 recipe/index reader and
annotation-free bulk conditioner.  It adds one piece of information that the
Model 7 model batch omits: the per-source-shard gene measurement mask.

The three states are kept distinct throughout the pipeline:

* measured and detected (count > 0),
* measured zero (count == 0),
* unmeasured / unknown (availability == 0).

Only the first two states supervise the support head.  Unmeasured coordinates
are never silently converted into negative support labels.
"""

from __future__ import annotations

from dataclasses import dataclass
import multiprocessing as mp
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

import numpy as np
import torch
from torch.utils.data import IterableDataset, get_worker_info

from . import fast_rows


MODEL8_ROOT = Path(__file__).resolve().parents[2]
IMMUN_ROOT = MODEL8_ROOT.parent
DEFAULT_MODEL7_ROOT = IMMUN_ROOT / "model_7_no_annotation_sh_hwang"
DEFAULT_CONDITIONER_CACHE = (
    DEFAULT_MODEL7_ROOT
    / "runs"
    / "E2_full_no_annotation_4epoch_seed20260627"
    / "conditioner_cache.pt"
)


@dataclass(frozen=True)
class SparseValueScaler:
    """Map non-negative log1p counts to the unit interval and back.

    This scaler first maps PBISC values to ``[0, 1]``.  The diffusion config
    may then center that interval to the paper's ``[-1, 1]`` representation.
    A fixed log-count ceiling keeps the transform invertible below the ceiling
    and, unlike dataset-wide min/max fitting, does not leak validation data.
    """

    max_log_value: float = 8.0

    def __post_init__(self) -> None:
        if not np.isfinite(self.max_log_value) or self.max_log_value <= 0:
            raise ValueError("max_log_value must be finite and positive")

    def to_unit_tensor(self, x_log: torch.Tensor) -> torch.Tensor:
        if not torch.is_floating_point(x_log):
            x_log = x_log.float()
        if not bool(torch.isfinite(x_log).all()):
            raise ValueError("x_log contains NaN or Inf")
        if bool((x_log < 0).any()):
            raise ValueError("x_log must be non-negative")
        return (x_log / float(self.max_log_value)).clamp(0.0, 1.0)

    def from_unit_tensor(self, value_unit: torch.Tensor) -> torch.Tensor:
        if not bool(torch.isfinite(value_unit).all()):
            raise ValueError("value_unit contains NaN or Inf")
        return value_unit.clamp(0.0, 1.0) * float(self.max_log_value)

    def state_dict(self) -> dict[str, float | str]:
        return {
            "schema_version": "pbisc_sdd.value_scaler.v1",
            "max_log_value": float(self.max_log_value),
        }

    @classmethod
    def from_state_dict(cls, state: Mapping[str, object]) -> "SparseValueScaler":
        if state.get("schema_version") != "pbisc_sdd.value_scaler.v1":
            raise ValueError("unsupported value scaler state")
        return cls(max_log_value=float(state["max_log_value"]))


@dataclass(frozen=True)
class MeasurementAwareRecipeBlock:
    """A sampled recipe block with a measurement mask for every selected cell."""

    block_id: str
    counts: np.ndarray
    availability_mask: np.ndarray
    bulk_measurement_fraction: np.ndarray
    # Full 20,097-gene bulk.  The canonical Model 7 conditioner must compute
    # its size factor and scalar features in the same full-panel space in which
    # its train-only statistics were fitted.
    bulk_log1p: np.ndarray
    n_cells: int


def _load_model7_api(model7_root: Path) -> tuple[Any, Any, Any]:
    # Compatibility argument remains for legacy configs; imports are package-local.
    del model7_root
    from model7_no_annotation.data import AnnotationFreeConditioner
    from model7_no_annotation.data_e2 import E2FullDataConfig, build_e2_full_data_module

    return AnnotationFreeConditioner, E2FullDataConfig, build_e2_full_data_module


def _torch_load(path: Path) -> object:
    """Load a trusted, local PBISC artifact across supported torch versions."""

    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:  # pragma: no cover - compatibility with older torch
        return torch.load(path, map_location="cpu")


def resolve_gene_positions(
    gene_panel: Any,
    *,
    n_genes: int | None = None,
    positions_path: Path | None = None,
    positions: Sequence[int] | None = None,
    measurement_coverage: np.ndarray | None = None,
) -> np.ndarray:
    """Resolve a stable subset of panel-row positions.

    A user-provided ``.npy`` or JSON list is authoritative.  Otherwise a
    reduced smoke/pilot panel is chosen by measurement coverage, while forcing
    known immune/interferon markers into the candidate set.  The returned rows
    are sorted in canonical panel order so all tensors retain a stable gene
    vocabulary.
    """

    panel_size = int(len(gene_panel))
    if panel_size <= 0:
        raise ValueError("gene panel is empty")
    if positions_path is not None and positions is not None:
        raise ValueError("pass positions_path or positions, not both")
    if positions is not None:
        positions = np.asarray(positions, dtype=np.int64)
    elif positions_path is not None:
        path = Path(positions_path).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"gene positions file not found: {path}")
        if path.suffix == ".npy":
            positions = np.asarray(np.load(path, allow_pickle=False), dtype=np.int64)
        elif path.suffix == ".json":
            import json

            payload = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(payload, list):
                raise ValueError("gene positions JSON must contain a list")
            positions = np.asarray(payload, dtype=np.int64)
        else:
            raise ValueError("gene positions file must be .npy or .json")
    else:
        requested = panel_size if n_genes is None else int(n_genes)
        if requested <= 0 or requested > panel_size:
            raise ValueError(f"n_genes must be in [1, {panel_size}]")
        if requested == panel_size:
            positions = np.arange(panel_size, dtype=np.int64)
        else:
            coverage = np.asarray(
                measurement_coverage
                if measurement_coverage is not None
                else gene_panel.get(
                    "measurement_frequency_sc", np.zeros(panel_size)
                ),
                dtype=np.float64,
            )
            if coverage.shape != (panel_size,) or not np.isfinite(coverage).all():
                raise ValueError("measurement coverage must be finite and panel-aligned")
            if float(np.ptp(coverage)) <= 0.0:
                raise ValueError(
                    "measurement coverage is degenerate; provide explicit gene "
                    "positions or compute coverage from train-split source masks"
                )
            marker = np.zeros(panel_size, dtype=bool)
            for column in ("is_immune_marker", "is_interferon_gene"):
                if column in gene_panel.columns:
                    marker |= np.asarray(gene_panel[column], dtype=bool)
            # Markers win ties; coverage and canonical index provide a stable fill.
            order = np.lexsort(
                (
                    np.arange(panel_size, dtype=np.int64),
                    -coverage,
                    -marker.astype(np.int8),
                )
            )
            positions = np.sort(order[:requested].astype(np.int64, copy=False))
    if positions.ndim != 1 or positions.size == 0:
        raise ValueError("gene positions must be a non-empty 1-D array")
    if len(np.unique(positions)) != len(positions):
        raise ValueError("gene positions contain duplicates")
    if int(positions.min()) < 0 or int(positions.max()) >= panel_size:
        raise ValueError("gene positions fall outside the canonical panel")
    return np.sort(positions.astype(np.int64, copy=False))


class MeasurementAwareE2DataModule:
    """Adapter around Model 7 E2 that restores per-cell measurement masks."""

    def __init__(
        self,
        base: Any,
        panel_positions: Sequence[int],
        *,
        measurement_coverage: np.ndarray | None = None,
    ) -> None:
        self.base = base
        positions = np.asarray(panel_positions, dtype=np.int64)
        if positions.ndim != 1 or positions.size == 0:
            raise ValueError("panel_positions must be a non-empty 1-D sequence")
        if len(np.unique(positions)) != positions.size:
            raise ValueError("panel_positions contain duplicates")
        if int(positions.min()) < 0 or int(positions.max()) >= len(base.gene_panel):
            raise ValueError("panel_positions fall outside the Model 7 gene panel")
        self.panel_positions = positions
        self.source_gene_indices = np.asarray(base.gene_indices, dtype=np.int64)[positions]
        self.gene_panel = base.gene_panel.iloc[positions].reset_index(drop=True)
        if measurement_coverage is None:
            self.measurement_coverage = np.full(
                positions.size, np.nan, dtype=np.float32
            )
        else:
            full_coverage = np.asarray(measurement_coverage, dtype=np.float32)
            if full_coverage.shape != (len(base.gene_panel),) or not np.isfinite(
                full_coverage
            ).all():
                raise ValueError("measurement_coverage must align with the full panel")
            self.full_measurement_coverage = full_coverage.copy()
            self.measurement_coverage = full_coverage[positions]
        if measurement_coverage is None:
            self.full_measurement_coverage = np.full(
                len(base.gene_panel), np.nan, dtype=np.float32
            )
        self._measurement_cache: dict[str, np.ndarray] = {}

    @property
    def corpus_fingerprint(self) -> str:
        return str(self.base.corpus_fingerprint)

    def split_manifest(self, *args: object, **kwargs: object) -> Any:
        return self.base.split_manifest(*args, **kwargs)

    def prime_pools(self, recipes: Any) -> None:
        self.base.prime_pools(recipes)

    def _measurement_mask(self, pool: Mapping[str, object]) -> np.ndarray:
        relative_path = str(pool["count_zarr"])
        cached = self._measurement_cache.get(relative_path)
        if cached is not None:
            return cached
        group = self.base._zarr_group(relative_path)
        if "measurement_mask" not in group:
            raise RuntimeError(
                f"source shard lacks required measurement_mask: {relative_path}"
            )
        full = np.asarray(group["measurement_mask"][:], dtype=np.uint8)
        if full.shape != (int(self.base.source_n_genes),):
            raise RuntimeError(
                f"measurement_mask has shape {full.shape}, expected "
                f"({self.base.source_n_genes},): {relative_path}"
            )
        if not bool(np.isin(full, (0, 1)).all()):
            raise RuntimeError(f"measurement_mask is not binary: {relative_path}")
        selected = full[self.source_gene_indices].astype(np.float32, copy=False)
        self._measurement_cache[relative_path] = selected
        return selected

    def load_training_block(
        self,
        recipe: Any,
        cells_per_block: int,
        rng: np.random.Generator,
    ) -> MeasurementAwareRecipeBlock:
        refs, pools = self.base._recipe_refs_and_pools(recipe)
        k = int(recipe["k"])
        if len(refs) != k:
            raise RuntimeError("recipe resolved a row count different from K")
        threads = fast_rows.fast_rows_threads()
        full_matrix = (
            fast_rows.load_sparse_refs_fast(self.base, refs, pools, threads=threads)
            if threads else self.base._load_sparse_refs(refs, pools)
        ).tocsr()
        bulk_raw = np.asarray(full_matrix.sum(axis=0), dtype=np.float64).reshape(-1)
        bulk_log1p = np.log1p(bulk_raw).astype(np.float32)
        matrix = full_matrix[:, self.source_gene_indices].tocsr()
        segment_counts = np.bincount(
            np.asarray([int(segment) for segment, _ in refs], dtype=np.int64),
            minlength=len(pools),
        )
        bulk_measurement_fraction = np.zeros(
            len(self.source_gene_indices), dtype=np.float64
        )
        for segment, count in enumerate(segment_counts):
            if int(count) > 0:
                bulk_measurement_fraction += float(count) * self._measurement_mask(
                    pools[segment]
                )
        bulk_measurement_fraction = (
            bulk_measurement_fraction / max(k, 1)
        ).astype(np.float32)

        take = min(max(int(cells_per_block), 0), k)
        if take:
            chosen = np.asarray(rng.choice(k, size=take, replace=False), dtype=np.int64)
            counts = matrix[chosen].toarray().astype(np.float32, copy=False)
            masks = np.stack(
                [self._measurement_mask(pools[int(refs[int(i)][0])]) for i in chosen],
                axis=0,
            ).astype(np.float32, copy=False)
        else:
            counts = np.empty((0, len(self.source_gene_indices)), dtype=np.float32)
            masks = np.empty_like(counts)
        if counts.shape != masks.shape:
            raise RuntimeError("count and measurement-mask shapes differ")
        if counts.size and bool((counts[masks == 0] != 0).any()):
            raise RuntimeError(
                "source contains a non-zero count at an unmeasured gene; "
                "measurement contract is inconsistent"
            )
        return MeasurementAwareRecipeBlock(
            block_id=str(recipe["block_id"]),
            counts=counts,
            availability_mask=masks,
            bulk_measurement_fraction=bulk_measurement_fraction,
            bulk_log1p=bulk_log1p,
            n_cells=k,
        )


class PBISCSparseCellDataset(IterableDataset[dict[str, object]]):
    """Stream annotation-free cells, bulk conditions, and availability masks."""

    def __init__(
        self,
        data_module: MeasurementAwareE2DataModule,
        *,
        split: str,
        recipes_per_epoch: int,
        cells_per_block: int,
        conditioner: Any,
        seed: int,
        platform: str | None = None,
        progress: bool = True,
    ) -> None:
        super().__init__()
        if not bool(getattr(conditioner, "fitted_", False)):
            raise ValueError("conditioner must be fitted")
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
        self._shared_epoch = mp.Value("q", 0)

    def set_epoch(self, epoch: int) -> None:
        with self._shared_epoch.get_lock():
            self._shared_epoch.value = int(epoch)

    def __len__(self) -> int:
        with self._shared_epoch.get_lock():
            epoch = int(self._shared_epoch.value)
        recipes = self.data_module.split_manifest(
            self.split,
            max_blocks=self.recipes_per_epoch,
            seed=self.seed + 1_000_003 * epoch,
            platform=self.platform,
        )
        return int(
            np.minimum(
                recipes["k"].to_numpy(dtype=np.int64, copy=False),
                self.cells_per_block,
            ).sum()
        )

    def __iter__(self) -> Iterator[dict[str, object]]:
        worker = get_worker_info()
        worker_id = 0 if worker is None else int(worker.id)
        with self._shared_epoch.get_lock():
            epoch = int(self._shared_epoch.value)
        epoch_seed = self.seed + 1_000_003 * epoch
        recipes = self.data_module.split_manifest(
            self.split,
            max_blocks=self.recipes_per_epoch,
            seed=epoch_seed,
            platform=self.platform,
        )
        if worker is not None:
            recipes = recipes.iloc[worker_id :: int(worker.num_workers)].reset_index(
                drop=True
            )
        self.data_module.prime_pools(recipes)
        rows = recipes.iterrows()
        if self.progress:
            from tqdm import tqdm

            description = f"PBISC-SDD {self.split} epoch {epoch}"
            if self.platform:
                description += f" {self.platform}"
            if worker is not None:
                description += f" worker {worker_id}"
            rows = tqdm(rows, total=len(recipes), desc=description)
        rng = np.random.default_rng(epoch_seed + 7_919 + 104_729 * worker_id)
        for fallback_index, (_, recipe) in enumerate(rows):
            block = self.data_module.load_training_block(
                recipe, self.cells_per_block, rng
            )
            if block.counts.shape[0] == 0:
                continue
            condition = self.conditioner.transform(block.bulk_log1p, block.n_cells)
            block_index = np.int64(
                int(recipe.get("stream_recipe_index", fallback_index))
            )
            for counts, availability in zip(
                block.counts, block.availability_mask, strict=True
            ):
                yield {
                    "x_log": np.log1p(counts).astype(np.float32, copy=False),
                    "condition": np.asarray(condition, dtype=np.float32),
                    "availability_mask": availability.astype(np.float32, copy=False),
                    "bulk_measurement_fraction": block.bulk_measurement_fraction,
                    "block_index": block_index,
                    "block_id": block.block_id,
                    "recipe_k": np.int64(block.n_cells),
                }


@dataclass(frozen=True)
class PBISCE2Streams:
    data_module: MeasurementAwareE2DataModule
    conditioner: Any
    train: PBISCSparseCellDataset
    validation: PBISCSparseCellDataset
    test: PBISCSparseCellDataset


class SubsetAnnotationFreeConditioner:
    """Use canonical full-panel normalization, then select condition genes.

    Re-fitting or directly applying the conditioner on a reduced panel changes
    the library-size factor and all six scalar features.  This wrapper first
    performs the exact full-panel transform used by Model 7 and only then keeps
    the requested gene positions plus the unchanged scalar tail.
    """

    def __init__(self, full_conditioner: Any, panel_positions: np.ndarray) -> None:
        positions = np.asarray(panel_positions, dtype=np.int64)
        if positions.ndim != 1 or positions.size == 0:
            raise ValueError("panel_positions must be a non-empty 1-D array")
        self.full_conditioner = full_conditioner
        self.panel_positions = positions
        self.fitted_ = bool(getattr(full_conditioner, "fitted_", False))
        if not self.fitted_:
            raise ValueError("full conditioner must be fitted")

    @property
    def dim(self) -> int:
        return int(len(self.panel_positions)) + 6

    @property
    def feature_names(self) -> list[str]:
        full_names = list(self.full_conditioner.feature_names)
        gene_names = [full_names[int(index)] for index in self.panel_positions]
        return gene_names + full_names[-6:]

    def transform(self, bulk_log1p: np.ndarray, recipe_k: int) -> np.ndarray:
        full = np.asarray(
            self.full_conditioner.transform(bulk_log1p, recipe_k), dtype=np.float32
        )
        n_full = int(self.full_conditioner.n_genes)
        if full.shape != (n_full + 6,):
            raise RuntimeError("full conditioner emitted an unexpected shape")
        return np.concatenate(
            [full[self.panel_positions], full[n_full:]], axis=0
        ).astype(np.float32, copy=False)

    def state_dict(self) -> dict[str, object]:
        return {
            "schema_version": "pbisc_sdd.subset_conditioner.v1",
            "panel_positions": self.panel_positions.copy(),
            "full_conditioner_state": self.full_conditioner.state_dict(),
        }


def load_subset_conditioner(
    conditioner_class: Any,
    *,
    cache_path: Path,
    panel_positions: np.ndarray,
    expected_corpus_fingerprint: str,
) -> Any:
    """Load canonical train-only normalization with a stable output subset."""

    path = Path(cache_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"conditioner cache not found: {path}")
    payload = _torch_load(path)
    if not isinstance(payload, Mapping) or not isinstance(
        payload.get("conditioner_state"), Mapping
    ):
        raise ValueError("conditioner cache has an unsupported structure")
    if str(payload.get("corpus_fingerprint", "")) != str(
        expected_corpus_fingerprint
    ):
        raise ValueError("conditioner cache corpus fingerprint does not match E2 index")
    full = payload["conditioner_state"]
    n_full = int(full.get("n_genes", -1))
    if n_full <= int(panel_positions.max()):
        raise ValueError("conditioner cache is smaller than requested gene positions")
    full_conditioner = conditioner_class(n_genes=n_full).load_state_dict(full)
    return SubsetAnnotationFreeConditioner(full_conditioner, panel_positions)


def compute_train_measurement_coverage(
    base: Any,
) -> np.ndarray:
    """Compute panel coverage from train-split sample pools and real zarr masks.

    The persisted metadata column is currently degenerate in PBISC E2, so it
    must not drive automatic panel selection. This calculation reads only
    train-split sample pools and weights each shard mask by its train rows.
    """

    from collections import defaultdict

    import pandas as pd
    import pyarrow.compute as pc
    import pyarrow.parquet as pq

    sample_index_path = base.index_root / "metadata" / "sample_index.parquet"
    sample_index = pd.read_parquet(
        sample_index_path, columns=["sample_key", "split"]
    )
    train_keys = set(
        sample_index.loc[sample_index["split"].astype(str) == "train", "sample_key"]
        .astype(str)
        .tolist()
    )
    if not train_keys:
        raise RuntimeError("cannot compute gene coverage from an empty train split")
    rows_by_shard: defaultdict[str, int] = defaultdict(int)
    seen_train_keys: set[str] = set()
    pool_file = pq.ParquetFile(base._sample_pool_path)
    for row_group in range(pool_file.metadata.num_row_groups):
        table = pool_file.read_row_group(
            row_group, columns=["sample_key", "shard_id", "row_indices"]
        )
        keys = [str(value) for value in table.column("sample_key").to_pylist()]
        shards = [str(value) for value in table.column("shard_id").to_pylist()]
        lengths = np.asarray(
            pc.list_value_length(table.column("row_indices")).to_numpy(),
            dtype=np.int64,
        )
        for sample_key, shard_id, rows in zip(keys, shards, lengths, strict=True):
            if sample_key in train_keys:
                rows_by_shard[shard_id] += int(rows)
                seen_train_keys.add(sample_key)
    missing_keys = train_keys.difference(seen_train_keys)
    if missing_keys:
        raise RuntimeError(
            f"train sample keys missing from persisted pools: {sorted(missing_keys)[:5]}"
        )

    source_coverage = np.zeros(int(base.source_n_genes), dtype=np.float64)
    total_rows = 0
    for shard_id, rows in sorted(rows_by_shard.items()):
        shard = base._shards_by_id.get(shard_id)
        if shard is None:
            raise RuntimeError(f"train pool references unknown shard: {shard_id}")
        group = base._zarr_group(str(shard["count_zarr"]))
        if "measurement_mask" not in group:
            raise RuntimeError(
                f"source shard lacks measurement_mask: {shard['count_zarr']}"
            )
        mask = np.asarray(group["measurement_mask"][:], dtype=np.uint8)
        if mask.shape != (int(base.source_n_genes),) or not bool(
            np.isin(mask, (0, 1)).all()
        ):
            raise RuntimeError(f"invalid measurement_mask: {shard['count_zarr']}")
        source_coverage += float(rows) * mask
        total_rows += rows
    if total_rows <= 0:
        raise RuntimeError("train-split sample pools contain no source rows")
    panel_coverage = source_coverage[np.asarray(base.gene_indices, dtype=np.int64)]
    panel_coverage = (panel_coverage / float(total_rows)).astype(np.float32)
    if float(np.ptp(panel_coverage)) <= 0.0:
        raise RuntimeError("train-split measurement coverage is degenerate")
    return panel_coverage


def subset_conditioner_from_state(
    conditioner_class: Any,
    *,
    state: Mapping[str, object],
    panel_positions: np.ndarray,
) -> SubsetAnnotationFreeConditioner:
    """Reconstruct the exact full-panel conditioner stored in a checkpoint."""

    if state.get("schema_version") != "pbisc_sdd.subset_conditioner.v1":
        raise ValueError("unsupported checkpoint conditioner state")
    stored_positions = np.asarray(state.get("panel_positions"), dtype=np.int64)
    if not np.array_equal(stored_positions, np.asarray(panel_positions, dtype=np.int64)):
        raise ValueError("checkpoint conditioner gene positions differ")
    full_state = state.get("full_conditioner_state")
    if not isinstance(full_state, Mapping):
        raise ValueError("checkpoint conditioner lacks full_conditioner_state")
    n_full = int(full_state.get("n_genes", -1))
    if n_full <= 0:
        raise ValueError("checkpoint full conditioner has invalid n_genes")
    full_conditioner = conditioner_class(n_genes=n_full).load_state_dict(full_state)
    return SubsetAnnotationFreeConditioner(full_conditioner, stored_positions)


def build_pbisc_e2_streams(
    *,
    model7_root: Path = DEFAULT_MODEL7_ROOT,
    conditioner_cache: Path = DEFAULT_CONDITIONER_CACHE,
    n_genes: int | None = None,
    gene_positions_path: Path | None = None,
    gene_positions: Sequence[int] | None = None,
    conditioner_state: Mapping[str, object] | None = None,
    measurement_coverage: Sequence[float] | None = None,
    train_recipes: int = 8,
    validation_recipes: int = 2,
    test_recipes: int = 2,
    train_cells_per_block: int = 32,
    validation_cells_per_block: int = 32,
    test_cells_per_block: int = 32,
    seed: int = 20260627,
    progress: bool = True,
    data_config: Mapping[str, object] | None = None,
) -> PBISCE2Streams:
    """Build small or full native PBISC streams without rebuilding E2 data."""

    conditioner_class, data_config_class, build_data_module = _load_model7_api(
        Path(model7_root)
    )
    options = dict(data_config or {})
    options["progress"] = progress
    base = build_data_module(data_config_class(**options))
    if measurement_coverage is None:
        resolved_coverage = compute_train_measurement_coverage(base)
    else:
        resolved_coverage = np.asarray(measurement_coverage, dtype=np.float32)
        if resolved_coverage.shape != (len(base.gene_panel),) or not np.isfinite(
            resolved_coverage
        ).all():
            raise ValueError("checkpoint measurement coverage is not full-panel aligned")
        if float(np.ptp(resolved_coverage)) <= 0.0:
            raise ValueError("checkpoint measurement coverage is degenerate")
    positions = resolve_gene_positions(
        base.gene_panel,
        n_genes=n_genes,
        positions_path=gene_positions_path,
        positions=gene_positions,
        measurement_coverage=resolved_coverage,
    )
    data_module = MeasurementAwareE2DataModule(
        base, positions, measurement_coverage=resolved_coverage
    )
    if conditioner_state is None:
        conditioner = load_subset_conditioner(
            conditioner_class,
            cache_path=Path(conditioner_cache),
            panel_positions=positions,
            expected_corpus_fingerprint=data_module.corpus_fingerprint,
        )
    else:
        conditioner = subset_conditioner_from_state(
            conditioner_class,
            state=conditioner_state,
            panel_positions=positions,
        )
    train = PBISCSparseCellDataset(
        data_module,
        split="train",
        recipes_per_epoch=int(train_recipes),
        cells_per_block=int(train_cells_per_block),
        conditioner=conditioner,
        seed=int(seed),
        progress=progress,
    )
    validation = PBISCSparseCellDataset(
        data_module,
        split="evaluation",
        recipes_per_epoch=int(validation_recipes),
        cells_per_block=int(validation_cells_per_block),
        conditioner=conditioner,
        seed=int(seed) + 10_007,
        progress=progress,
    )
    test = PBISCSparseCellDataset(
        data_module,
        split="test",
        recipes_per_epoch=int(test_recipes),
        cells_per_block=int(test_cells_per_block),
        conditioner=conditioner,
        seed=int(seed) + 30_013,
        progress=progress,
    )
    return PBISCE2Streams(
        data_module=data_module,
        conditioner=conditioner,
        train=train,
        validation=validation,
        test=test,
    )


def condition_with_availability(
    condition: torch.Tensor, availability_mask: torch.Tensor
) -> torch.Tensor:
    """Append a fixed analog measurement mask to the bulk/depth condition."""

    if condition.ndim != 2 or availability_mask.ndim != 2:
        raise ValueError("condition and availability_mask must be rank-2 tensors")
    if condition.shape[0] != availability_mask.shape[0]:
        raise ValueError("condition and availability_mask batch sizes differ")
    if not bool(torch.isfinite(condition).all()):
        raise ValueError("condition contains NaN or Inf")
    if not bool(torch.isfinite(availability_mask).all()):
        raise ValueError("availability_mask contains NaN or Inf")
    if bool(((availability_mask < 0) | (availability_mask > 1)).any()):
        raise ValueError("availability_mask must lie in [0, 1]")
    analog_mask = availability_mask.to(dtype=condition.dtype) * 2.0 - 1.0
    return torch.cat([condition, analog_mask], dim=-1)


def condition_with_measurement_context(
    condition: torch.Tensor,
    bulk_measurement_fraction: torch.Tensor,
    output_availability_mask: torch.Tensor,
) -> torch.Tensor:
    """Append distinct input-bulk and output-assay measurement context.

    ``bulk_measurement_fraction`` may be fractional because a recipe can mix
    source shards with different gene panels. ``output_availability_mask`` is
    binary for each generated cell. Both are mapped to ``[-1, 1]``.
    """

    if condition.ndim != 2:
        raise ValueError("condition must be rank-2")
    expected = output_availability_mask.shape
    if bulk_measurement_fraction.shape != expected:
        raise ValueError("bulk and output measurement tensors must have equal shape")
    if condition.shape[0] != expected[0] or len(expected) != 2:
        raise ValueError("measurement context batch dimensions do not match condition")
    for name, value in (
        ("condition", condition),
        ("bulk_measurement_fraction", bulk_measurement_fraction),
        ("output_availability_mask", output_availability_mask),
    ):
        if not bool(torch.isfinite(value).all()):
            raise ValueError(f"{name} contains NaN or Inf")
    if bool(
        ((bulk_measurement_fraction < 0) | (bulk_measurement_fraction > 1)).any()
    ):
        raise ValueError("bulk_measurement_fraction must lie in [0, 1]")
    if bool(
        ((output_availability_mask != 0) & (output_availability_mask != 1)).any()
    ):
        raise ValueError("output_availability_mask must be binary")
    bulk_analog = bulk_measurement_fraction.to(dtype=condition.dtype) * 2.0 - 1.0
    output_analog = output_availability_mask.to(dtype=condition.dtype) * 2.0 - 1.0
    return torch.cat([condition, bulk_analog, output_analog], dim=-1)
