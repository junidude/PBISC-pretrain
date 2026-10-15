"""Training, checkpointing, and generation utilities for PBISC SDD experiments."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from contextlib import nullcontext
import hashlib
import json
import math
import os
from pathlib import Path
import random
import socket
import sys
import time
from typing import Any, Callable, ContextManager, Iterable, Mapping

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from .data import (
    DEFAULT_CONDITIONER_CACHE,
    DEFAULT_MODEL7_ROOT,
    PBISCE2Streams,
    SparseValueScaler,
    build_pbisc_e2_streams,
    condition_with_measurement_context,
)
from .diffusion import SparseDataDiffusion, SparseDiffusionConfig, SparseDiffusionLoss
from .ema import ExponentialMovingAverage
from .model import ConditionalSBDDenoiser, SparseDenoiserConfig


CONFIG_SCHEMA = "pbisc_sdd.train.v1"
CHECKPOINT_SCHEMA = "pbisc_sdd.checkpoint.v5"
LEGACY_CHECKPOINT_SCHEMAS = frozenset({"pbisc_sdd.checkpoint.v4"})
STATUS_SCHEMA = "pbisc_sdd.status.v1"
REFERENCE_REPOSITORY = "https://github.com/PhilSid/sparse-data-diffusion"
REFERENCE_COMMIT = "1d5423c13477f2a52269a6f79113b45c6af5510f"


def _require_mapping(parent: Mapping[str, object], name: str) -> dict[str, object]:
    value = parent.get(name)
    if not isinstance(value, Mapping):
        raise ValueError(f"configuration requires object section {name!r}")
    return dict(value)


def _positive_int(value: object, *, name: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a positive integer")
    try:
        numeric = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be a positive integer") from error
    if not math.isfinite(numeric) or numeric != math.floor(numeric) or numeric <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return int(numeric)


def _non_negative_int(value: object, *, name: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a non-negative integer")
    try:
        numeric = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be a non-negative integer") from error
    if not math.isfinite(numeric) or numeric != math.floor(numeric) or numeric < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return int(numeric)


def _finite_float(value: object, *, name: str, minimum: float | None = None) -> float:
    result = float(value)
    if not math.isfinite(result) or (minimum is not None and result < minimum):
        suffix = "finite" if minimum is None else f"finite and >= {minimum}"
        raise ValueError(f"{name} must be {suffix}")
    return result


def load_experiment_config(path: Path | str) -> dict[str, object]:
    config_path = Path(path).expanduser().resolve()
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("configuration root must be an object")
    if payload.get("schema_version") != CONFIG_SCHEMA:
        raise ValueError(f"schema_version must be {CONFIG_SCHEMA!r}")
    experiment_id = str(payload.get("experiment_id", "")).strip()
    if not experiment_id or any(char in experiment_id for char in "/\\"):
        raise ValueError("experiment_id must be a non-empty path-safe name")
    paths = _require_mapping(payload, "paths")
    data = _require_mapping(payload, "data")
    model = _require_mapping(payload, "model")
    diffusion = _require_mapping(payload, "diffusion")
    training = _require_mapping(payload, "training")
    sampling = _require_mapping(payload, "sampling")

    required_paths = {"conditioner_cache", "output_dir"}
    missing = sorted(required_paths.difference(paths))
    if missing:
        raise ValueError(f"paths section is missing {missing}")
    for name in required_paths:
        if not str(paths[name]).strip():
            raise ValueError(f"paths.{name} must be non-empty")

    _positive_int(data.get("n_genes", 0), name="data.n_genes")
    for name in (
        "train_recipes",
        "validation_recipes",
        "test_recipes",
        "train_cells_per_block",
        "validation_cells_per_block",
        "test_cells_per_block",
    ):
        _positive_int(data.get(name, 0), name=f"data.{name}")
    if data.get("include_availability_in_condition") is not True:
        raise ValueError(
            "data.include_availability_in_condition must be true for the "
            "measurement-aware Model 8 contract"
        )
    if data.get("include_bulk_measurement_in_condition") is not True:
        raise ValueError(
            "data.include_bulk_measurement_in_condition must be true for the "
            "measurement-aware Model 8 contract"
        )
    gene_positions_path = data.get("gene_positions_path")
    if gene_positions_path not in (None, "") and not Path(
        str(gene_positions_path)
    ).expanduser().is_file():
        raise FileNotFoundError(
            f"data.gene_positions_path not found: {gene_positions_path}"
        )
    selection_mode = str(data.get("gene_selection", ""))
    if gene_positions_path in (None, ""):
        if selection_mode != "coverage_with_immune_markers":
            raise ValueError(
                "automatic gene selection must be coverage_with_immune_markers"
            )
    else:
        if selection_mode not in {"explicit_train_only_hvg_manifest", "explicit_rare_gene_expansion_manifest"}:
            raise ValueError(
                "an explicit positions file requires gene_selection="
                "'explicit_train_only_hvg_manifest' or 'explicit_rare_gene_expansion_manifest'"
            )
        manifest_path = Path(str(data.get("gene_selection_manifest", ""))).expanduser()
        if not manifest_path.is_file():
            raise FileNotFoundError(
                f"data.gene_selection_manifest not found: {manifest_path}"
            )
        expected_manifest_hash = str(
            data.get("gene_selection_manifest_sha256", "")
        )
        if len(expected_manifest_hash) != 64 or _sha256_file(
            manifest_path.resolve()
        ) != expected_manifest_hash:
            raise ValueError("gene-selection manifest SHA-256 does not match")
        manifest_schema = json.loads(manifest_path.read_text())
        if (selection_mode == "explicit_train_only_hvg_manifest"
                and manifest_schema.get("schema_version") != "pbisc_sdd.hvg_selection.v1"):
            raise ValueError("unsupported gene-selection manifest schema for legacy HVG selection")
        expected_hash = str(data.get("gene_positions_sha256", ""))
        resolved_positions_path = Path(str(gene_positions_path)).expanduser().resolve()
        if len(expected_hash) != 64 or _sha256_file(resolved_positions_path) != expected_hash:
            raise ValueError("gene positions SHA-256 does not match configuration")
        if resolved_positions_path.suffix == ".npy":
            configured_positions = np.load(
                resolved_positions_path, allow_pickle=False
            )
        elif resolved_positions_path.suffix == ".json":
            configured_positions = np.asarray(
                json.loads(resolved_positions_path.read_text(encoding="utf-8"))
            )
        else:
            raise ValueError("gene positions file must be .npy or .json")
        if configured_positions.ndim != 1 or len(configured_positions) != int(
            data["n_genes"]
        ):
            raise ValueError("data.n_genes differs from gene positions length")
        if selection_mode == "explicit_rare_gene_expansion_manifest":
            from .expanded_panel import validate_expanded_panel
            validate_expanded_panel(payload)

    hidden_dims = model.get("hidden_dims")
    if not isinstance(hidden_dims, list) or not hidden_dims:
        raise ValueError("model.hidden_dims must be a non-empty integer list")
    for index, width in enumerate(hidden_dims):
        _positive_int(width, name=f"model.hidden_dims[{index}]")
    for name in ("time_embedding_dim", "context_dim"):
        _positive_int(model.get(name, 0), name=f"model.{name}")
    dropout = _finite_float(model.get("dropout", 0.0), name="model.dropout", minimum=0.0)
    if dropout >= 1.0:
        raise ValueError("model.dropout must be < 1")

    if diffusion.get("value_loss_mode") not in ("all_measured", "active_only"):
        raise ValueError(
            "diffusion.value_loss_mode must be all_measured or active_only"
        )
    if not isinstance(diffusion.get("center_values", True), bool):
        raise ValueError("diffusion.center_values must be boolean")
    if not isinstance(diffusion.get("clip_denoised", True), bool):
        raise ValueError("diffusion.clip_denoised must be boolean")
    for name in ("value_loss_weight", "support_loss_weight"):
        _finite_float(
            diffusion.get(name, 1.0), name=f"diffusion.{name}", minimum=0.0
        )
    self_condition = _finite_float(
        diffusion.get("self_condition_probability", 0.5),
        name="diffusion.self_condition_probability",
        minimum=0.0,
    )
    if self_condition > 1.0:
        raise ValueError("diffusion.self_condition_probability must be <= 1")
    _finite_float(
        diffusion.get("max_log1p_value", 8.0),
        name="diffusion.max_log1p_value",
        minimum=1.0e-12,
    )

    for name in ("epochs", "batch_size"):
        _positive_int(training.get(name, 0), name=f"training.{name}")
    for name in ("max_train_steps", "max_validation_steps", "num_workers"):
        _non_negative_int(training.get(name, 0), name=f"training.{name}")
    for name in ("lr", "grad_clip"):
        _finite_float(training.get(name, 0.0), name=f"training.{name}", minimum=1.0e-12)
    _finite_float(
        training.get("weight_decay", 0.0),
        name="training.weight_decay",
        minimum=0.0,
    )
    device = str(training.get("device", "auto"))
    if device not in {"auto", "cpu", "cuda"}:
        raise ValueError("training.device must be auto, cpu, or cuda")
    amp = str(training.get("amp", "none"))
    if amp not in {"none", "bf16"}:
        raise ValueError("training.amp must be none or bf16")
    _non_negative_int(
        training.get("heartbeat_every_steps", 50),
        name="training.heartbeat_every_steps",
    )
    ema_decay = _finite_float(
        training.get("ema_decay", 0.9999),
        name="training.ema_decay",
        minimum=0.0,
    )
    if ema_decay >= 1.0:
        raise ValueError("training.ema_decay must be < 1")
    _positive_int(
        training.get("ema_update_every", 1), name="training.ema_update_every"
    )
    cuda_limit = training.get("max_cuda_memory_gib")
    if cuda_limit is not None:
        _finite_float(
            cuda_limit,
            name="training.max_cuda_memory_gib",
            minimum=1.0e-6,
        )
    _finite_float(
        training.get("min_system_memory_available_gib", 0.0),
        name="training.min_system_memory_available_gib",
        minimum=0.0,
    )
    _finite_float(
        training.get("min_cuda_memory_available_gib", 0.0),
        name="training.min_cuda_memory_available_gib",
        minimum=0.0,
    )
    max_utilization = _finite_float(
        training.get("max_cuda_utilization_percent_at_start", 100.0),
        name="training.max_cuda_utilization_percent_at_start",
        minimum=0.0,
    )
    if max_utilization > 100.0:
        raise ValueError("max_cuda_utilization_percent_at_start must be <= 100")
    _finite_float(
        training.get("max_cuda_temperature_c_at_start", 1_000.0),
        name="training.max_cuda_temperature_c_at_start",
        minimum=0.0,
    )

    _positive_int(sampling.get("steps", 0), name="sampling.steps")
    _positive_int(sampling.get("preview_cells", 0), name="sampling.preview_cells")
    eta = _finite_float(sampling.get("eta", 0.0), name="sampling.eta", minimum=0.0)
    if eta > 1.0:
        raise ValueError("sampling.eta must be <= 1")
    _finite_float(
        sampling.get("active_logit_threshold", 0.0),
        name="sampling.active_logit_threshold",
    )
    return payload


def require_single_process() -> None:
    """Fail before device initialization when invoked by a distributed launcher."""
    for key in ("WORLD_SIZE", "LOCAL_WORLD_SIZE"):
        try:
            value = int(os.environ.get(key, "1"))
        except ValueError as error:
            raise ValueError(f"{key} must be an integer") from error
        if value != 1:
            raise RuntimeError("PBISC pretraining supports one process and one GPU; distributed launch is unsupported")
    if "LOCAL_RANK" in os.environ or "RANK" in os.environ:
        raise RuntimeError("PBISC pretraining does not support torchrun or distributed rank variables")


def resolve_device(requested: str) -> torch.device:
    require_single_process()
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return torch.device(requested)


def _autocast_context(
    device: torch.device, amp_mode: str
) -> ContextManager[object]:
    if amp_mode == "none":
        return nullcontext()
    if amp_mode != "bf16":
        raise ValueError(f"unsupported amp mode: {amp_mode}")
    if device.type != "cuda":
        raise RuntimeError("bf16 autocast requires a CUDA device")
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("the selected CUDA device does not support bfloat16")
    return torch.autocast(device_type="cuda", dtype=torch.bfloat16)


def _system_memory_available_bytes() -> int:
    with Path("/proc/meminfo").open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.startswith("MemAvailable:"):
                fields = line.split()
                if len(fields) >= 2:
                    return int(fields[1]) * 1024
    raise RuntimeError("cannot read MemAvailable from /proc/meminfo")


def _memory_snapshot(device: torch.device) -> dict[str, float | int]:
    gib = float(1024**3)
    snapshot: dict[str, float | int] = {
        "system_available_gib": _system_memory_available_bytes() / gib,
    }
    if device.type == "cuda":
        free_bytes, total_bytes = torch.cuda.mem_get_info(device)
        snapshot.update(
            {
                "cuda_free_gib": free_bytes / gib,
                "cuda_total_gib": total_bytes / gib,
                "cuda_allocated_gib": torch.cuda.memory_allocated(device) / gib,
                "cuda_reserved_gib": torch.cuda.memory_reserved(device) / gib,
                "cuda_peak_allocated_gib": (
                    torch.cuda.max_memory_allocated(device) / gib
                ),
            }
        )
    return snapshot


def configure_memory_safety(
    training: Mapping[str, object], device: torch.device
) -> dict[str, object]:
    """Apply a PyTorch allocator cap and enforce host-memory headroom."""

    gib = float(1024**3)
    minimum_system_gib = float(
        training.get("min_system_memory_available_gib", 0.0)
    )
    available_gib = _system_memory_available_bytes() / gib
    if available_gib < minimum_system_gib:
        raise MemoryError(
            f"system MemAvailable is {available_gib:.2f} GiB, below the "
            f"configured {minimum_system_gib:.2f} GiB safety floor"
        )
    policy: dict[str, object] = {
        "system_memory_floor_gib": minimum_system_gib,
        "system_available_at_start_gib": available_gib,
        "cuda_allocator_limit_gib": None,
        "cuda_allocator_fraction": None,
    }
    cuda_limit = training.get("max_cuda_memory_gib")
    minimum_cuda_gib = float(
        training.get("min_cuda_memory_available_gib", 0.0)
    )
    policy["cuda_memory_floor_gib"] = minimum_cuda_gib
    if device.type == "cuda":
        free_bytes, _ = torch.cuda.mem_get_info(device)
        free_gib = free_bytes / gib
        utilization = float(torch.cuda.utilization(device))
        temperature = float(torch.cuda.temperature(device))
        policy["cuda_available_at_start_gib"] = free_gib
        policy["cuda_utilization_at_start_percent"] = utilization
        policy["cuda_temperature_at_start_c"] = temperature
        if free_gib < minimum_cuda_gib:
            raise MemoryError(
                f"CUDA free memory is {free_gib:.2f} GiB, below the "
                f"configured {minimum_cuda_gib:.2f} GiB safety floor"
            )
        utilization_limit = float(
            training.get("max_cuda_utilization_percent_at_start", 100.0)
        )
        if utilization > utilization_limit:
            raise RuntimeError(
                f"CUDA utilization is {utilization:.1f}%, above the configured "
                f"{utilization_limit:.1f}% exclusive-start ceiling"
            )
        temperature_limit = float(
            training.get("max_cuda_temperature_c_at_start", 1_000.0)
        )
        if temperature > temperature_limit:
            raise RuntimeError(
                f"CUDA temperature is {temperature:.1f}C, above the configured "
                f"{temperature_limit:.1f}C start ceiling"
            )
    if device.type == "cuda" and cuda_limit is not None:
        # Some PyTorch releases reject an unindexed ``torch.device("cuda")``
        # in set_per_process_memory_fraction even though the other CUDA memory
        # APIs accept it. Resolve the current device explicitly so the safety
        # cap is fail-closed without requiring every config to spell cuda:0.
        cuda_device_index = (
            int(device.index)
            if device.index is not None
            else int(torch.cuda.current_device())
        )
        total_bytes = int(
            torch.cuda.get_device_properties(cuda_device_index).total_memory
        )
        limit_bytes = float(cuda_limit) * gib
        if limit_bytes >= total_bytes:
            raise ValueError(
                "max_cuda_memory_gib must be smaller than visible CUDA memory"
            )
        fraction = limit_bytes / float(total_bytes)
        torch.cuda.set_per_process_memory_fraction(
            fraction, device=cuda_device_index
        )
        policy["cuda_allocator_limit_gib"] = float(cuda_limit)
        policy["cuda_allocator_fraction"] = fraction
        policy["cuda_device_index"] = cuda_device_index
        policy["cuda_visible_total_gib"] = total_bytes / gib
    return policy


def enforce_memory_floor(
    training: Mapping[str, object], device: torch.device
) -> dict[str, float | int]:
    snapshot = _memory_snapshot(device)
    minimum = float(training.get("min_system_memory_available_gib", 0.0))
    if float(snapshot["system_available_gib"]) < minimum:
        raise MemoryError(
            "system memory safety floor crossed: "
            f"{float(snapshot['system_available_gib']):.2f} GiB < {minimum:.2f} GiB"
        )
    minimum_cuda = float(training.get("min_cuda_memory_available_gib", 0.0))
    if device.type == "cuda" and float(snapshot["cuda_free_gib"]) < minimum_cuda:
        raise MemoryError(
            "CUDA memory safety floor crossed: "
            f"{float(snapshot['cuda_free_gib']):.2f} GiB < {minimum_cuda:.2f} GiB"
        )
    return snapshot


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _path_or_none(value: object) -> Path | None:
    return None if value in (None, "") else Path(str(value)).expanduser()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_streams_from_config(
    config: Mapping[str, object],
    *,
    gene_positions: Iterable[int] | None = None,
    conditioner_state: Mapping[str, object] | None = None,
    measurement_coverage: Iterable[float] | None = None,
    progress: bool | None = None,
) -> PBISCE2Streams:
    paths = _require_mapping(config, "paths")
    data = _require_mapping(config, "data")
    training = _require_mapping(config, "training")
    if progress is None:
        progress = bool(data.get("progress", True))
    direct_positions = None if gene_positions is None else list(gene_positions)
    streams = build_pbisc_e2_streams(
        model7_root=Path(str(paths.get("model7_root", DEFAULT_MODEL7_ROOT))),
        data_config=config.get("e2_data"),
        conditioner_cache=Path(
            str(paths.get("conditioner_cache", DEFAULT_CONDITIONER_CACHE))
        ),
        n_genes=int(data["n_genes"]),
        gene_positions_path=(
            None
            if direct_positions is not None
            else _path_or_none(data.get("gene_positions_path"))
        ),
        gene_positions=direct_positions,
        conditioner_state=conditioner_state,
        measurement_coverage=(
            None if measurement_coverage is None else list(measurement_coverage)
        ),
        train_recipes=int(data["train_recipes"]),
        validation_recipes=int(data["validation_recipes"]),
        test_recipes=int(data["test_recipes"]),
        train_cells_per_block=int(data["train_cells_per_block"]),
        validation_cells_per_block=int(data["validation_cells_per_block"]),
        test_cells_per_block=int(data["test_cells_per_block"]),
        seed=int(training.get("seed", 20260627)),
        progress=progress,
    )
    manifest_path = data.get("gene_selection_manifest")
    if manifest_path not in (None, "") and data.get("gene_selection") == "explicit_rare_gene_expansion_manifest":
        from .expanded_panel import validate_expanded_panel
        validate_expanded_panel(config, gene_panel=streams.data_module.gene_panel,
                                corpus_fingerprint=streams.data_module.corpus_fingerprint)
        return streams
    if manifest_path not in (None, ""):
        manifest = json.loads(
            Path(str(manifest_path)).expanduser().resolve().read_text(encoding="utf-8")
        )
        if manifest.get("schema_version") != "pbisc_sdd.hvg_selection.v1":
            raise ValueError("unsupported gene-selection manifest schema")
        positions_path = Path(str(data["gene_positions_path"])).expanduser().resolve()
        if manifest.get("positions_sha256") != _sha256_file(positions_path):
            raise ValueError("gene-selection manifest does not bind positions file")
        implementation_path = Path(str(data.get("gene_selection_implementation_path", manifest.get("implementation", ""))))
        if not implementation_path.is_file() or manifest.get(
            "implementation_sha256"
        ) != _sha256_file(implementation_path):
            raise ValueError("gene-selection implementation no longer matches manifest")
        if manifest.get("corpus_fingerprint") != streams.data_module.corpus_fingerprint:
            raise ValueError("gene-selection manifest corpus fingerprint differs")
        if int(manifest.get("n_selected_genes", -1)) != len(
            streams.data_module.panel_positions
        ):
            raise ValueError("gene-selection manifest selected-gene count differs")
        if int(data["n_genes"]) != len(streams.data_module.panel_positions):
            raise ValueError("data.n_genes differs from explicit gene positions length")
    return streams


def build_model_from_config(
    config: Mapping[str, object],
    *,
    n_genes: int,
    base_condition_dim: int,
) -> tuple[ConditionalSBDDenoiser, SparseDataDiffusion, SparseValueScaler]:
    model_section = _require_mapping(config, "model")
    diffusion_section = _require_mapping(config, "diffusion")
    condition_dim = int(base_condition_dim) + 2 * int(n_genes)
    denoiser_config = SparseDenoiserConfig(
        n_genes=int(n_genes),
        condition_dim=condition_dim,
        hidden_dims=tuple(int(value) for value in model_section["hidden_dims"]),
        time_embedding_dim=int(model_section["time_embedding_dim"]),
        context_dim=int(model_section["context_dim"]),
        dropout=float(model_section.get("dropout", 0.0)),
    )
    denoiser = ConditionalSBDDenoiser(denoiser_config)
    diffusion_config = SparseDiffusionConfig(
        # SparseValueScaler performs the PBISC log-count transform.  The core
        # optionally centers the unit interval to the paper's [-1, 1] range.
        value_scale=1.0,
        center_values=bool(diffusion_section.get("center_values", True)),
        clip_denoised=bool(diffusion_section.get("clip_denoised", True)),
        support_scale=float(diffusion_section.get("support_scale", 1.0)),
        cosine_offset=float(diffusion_section.get("cosine_offset", 0.008)),
        log_snr_clip=float(diffusion_section.get("log_snr_clip", 20.0)),
        value_loss_mode=str(diffusion_section["value_loss_mode"]),
        value_loss_weight=float(diffusion_section.get("value_loss_weight", 1.0)),
        support_loss_weight=float(
            diffusion_section.get("support_loss_weight", 1.0)
        ),
        self_condition_probability=float(
            diffusion_section.get("self_condition_probability", 0.5)
        ),
    )
    diffusion = SparseDataDiffusion(denoiser, diffusion_config)
    scaler = SparseValueScaler(
        max_log_value=float(diffusion_section.get("max_log1p_value", 8.0))
    )
    return denoiser, diffusion, scaler


def _loader(dataset: object, training: Mapping[str, object]) -> DataLoader[Any]:
    workers = int(training.get("num_workers", 0))
    return DataLoader(
        dataset,
        batch_size=int(training["batch_size"]),
        num_workers=workers,
        pin_memory=bool(training.get("pin_memory", False)),
        persistent_workers=workers > 0,
        drop_last=False,
    )


# Optional cell-type proportion hint (see pbisc_sdd.proportion_hint).  ``None`` keeps the
# annotation-free contract unchanged; when set, ``_prepare_batch`` appends the K-dim vector of
# each row's sample to the base bulk condition before the measurement context.
ACTIVE_PROPORTION_HINT = None


def set_active_proportion_hint(hint) -> None:
    global ACTIVE_PROPORTION_HINT
    ACTIVE_PROPORTION_HINT = hint


def active_proportion_hint_contract() -> dict | None:
    return None if ACTIVE_PROPORTION_HINT is None else ACTIVE_PROPORTION_HINT.contract()


def _prepare_batch(
    batch: Mapping[str, object],
    *,
    device: torch.device,
    scaler: SparseValueScaler,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    for key in (
        "x_log",
        "condition",
        "availability_mask",
        "bulk_measurement_fraction",
    ):
        if key not in batch or not torch.is_tensor(batch[key]):
            raise ValueError(f"collated batch is missing tensor field {key!r}")
    non_blocking = device.type == "cuda"
    x_log = batch["x_log"].to(  # type: ignore[union-attr]
        device=device, dtype=torch.float32, non_blocking=non_blocking
    )
    base_condition = batch["condition"].to(  # type: ignore[union-attr]
        device=device, dtype=torch.float32, non_blocking=non_blocking
    )
    availability = batch["availability_mask"].to(  # type: ignore[union-attr]
        device=device, dtype=torch.bool, non_blocking=non_blocking
    )
    bulk_measurement = batch["bulk_measurement_fraction"].to(  # type: ignore[union-attr]
        device=device, dtype=torch.float32, non_blocking=non_blocking
    )
    if x_log.ndim != 2 or availability.shape != x_log.shape:
        raise ValueError("x_log and availability_mask must have equal [B,G] shapes")
    if bulk_measurement.shape != x_log.shape:
        raise ValueError("bulk_measurement_fraction must match x_log shape")
    values = scaler.to_unit_tensor(x_log)
    active = x_log > 0
    hint = ACTIVE_PROPORTION_HINT
    if hint is not None:
        block_ids = batch.get("block_id")
        if block_ids is None or len(block_ids) != x_log.shape[0]:
            raise ValueError("proportion hint is active but the batch lacks a per-row block_id")
        hint_values = torch.from_numpy(hint.vectors_for(list(block_ids))).to(
            device=device, dtype=torch.float32, non_blocking=non_blocking
        )
        base_condition = torch.cat((base_condition, hint_values), dim=1)
    condition = condition_with_measurement_context(
        base_condition, bulk_measurement, availability
    )
    return values, condition, availability, active


def _support_counts(
    logits: torch.Tensor, active: torch.Tensor, measured: torch.Tensor
) -> tuple[int, int, int, int]:
    predicted = logits > 0
    mask = measured.bool()
    tp = int((predicted & active & mask).sum().item())
    fp = int((predicted & ~active & mask).sum().item())
    fn = int((~predicted & active & mask).sum().item())
    tn = int((~predicted & ~active & mask).sum().item())
    return tp, fp, fn, tn


def _metric_summary(
    *,
    value_error_sum: float,
    value_elements: int,
    support_error_sum: float,
    support_elements: int,
    value_loss_weight: float,
    support_loss_weight: float,
    batches: int,
    counts: tuple[int, int, int, int],
) -> dict[str, float | int]:
    value_loss = value_error_sum / max(value_elements, 1)
    support_loss = support_error_sum / max(support_elements, 1)
    tp, fp, fn, tn = counts
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    return {
        "loss": value_loss_weight * value_loss + support_loss_weight * support_loss,
        "value_loss": value_loss,
        "support_loss": support_loss,
        "value_elements": int(value_elements),
        "support_elements": int(support_elements),
        "support_precision": precision,
        "support_recall": recall,
        "support_f1": 2.0 * precision * recall / max(precision + recall, 1.0e-12),
        "support_accuracy": (tp + tn) / max(tp + fp + fn + tn, 1),
        "batches": batches,
    }


def _accumulate_counts(
    left: tuple[int, int, int, int], right: tuple[int, int, int, int]
) -> tuple[int, int, int, int]:
    return tuple(a + b for a, b in zip(left, right, strict=True))  # type: ignore[return-value]


def train_one_epoch(
    diffusion: SparseDataDiffusion,
    loader: DataLoader[Any],
    optimizer: torch.optim.Optimizer,
    *,
    device: torch.device,
    scaler: SparseValueScaler,
    grad_clip: float,
    max_steps: int,
    generator: torch.Generator,
    amp_mode: str = "none",
    heartbeat_every_steps: int = 0,
    heartbeat_callback: Callable[[int, float, float], None] | None = None,
    ema: ExponentialMovingAverage[nn.Module] | None = None,
) -> dict[str, float | int]:
    diffusion.train()
    value_error_sum = support_error_sum = 0.0
    value_elements = support_elements = 0
    counts = (0, 0, 0, 0)
    batches = 0
    for batch in loader:
        values, condition, availability, active = _prepare_batch(
            batch, device=device, scaler=scaler
        )
        optimizer.zero_grad(set_to_none=True)
        with _autocast_context(device, amp_mode):
            result = diffusion.training_loss(
                values,
                condition,
                availability_mask=availability,
                active_target=active,
                generator=generator,
            )
        if not bool(torch.isfinite(result.loss)):
            raise FloatingPointError("training loss became non-finite")
        result.loss.backward()
        norm = nn.utils.clip_grad_norm_(diffusion.parameters(), float(grad_clip))
        if not bool(torch.isfinite(norm)):
            raise FloatingPointError("gradient norm became non-finite")
        optimizer.step()
        if ema is not None:
            ema.update(diffusion.denoiser)
        batches += 1
        value_mask = (
            availability
            if diffusion.config.value_loss_mode == "all_measured"
            else availability & active
        )
        batch_value_elements = int(value_mask.sum().item())
        batch_support_elements = int(availability.sum().item())
        value_error_sum += float(result.value_loss.detach()) * batch_value_elements
        support_error_sum += (
            float(result.support_loss.detach()) * batch_support_elements
        )
        value_elements += batch_value_elements
        support_elements += batch_support_elements
        counts = _accumulate_counts(
            counts,
            _support_counts(result.prediction.active_logits.detach(), active, availability),
        )
        if (
            heartbeat_callback is not None
            and heartbeat_every_steps > 0
            and batches % heartbeat_every_steps == 0
        ):
            heartbeat_callback(
                batches,
                float(result.loss.detach()),
                float(norm.detach()),
            )
        if max_steps and batches >= max_steps:
            break
    if batches == 0:
        raise RuntimeError("training stream emitted no batches")
    return _metric_summary(
        value_error_sum=value_error_sum,
        value_elements=value_elements,
        support_error_sum=support_error_sum,
        support_elements=support_elements,
        value_loss_weight=float(diffusion.config.value_loss_weight),
        support_loss_weight=float(diffusion.config.support_loss_weight),
        batches=batches,
        counts=counts,
    )


@torch.no_grad()
def evaluate(
    diffusion: SparseDataDiffusion,
    loader: DataLoader[Any],
    *,
    device: torch.device,
    scaler: SparseValueScaler,
    max_steps: int,
    seed: int,
    amp_mode: str = "none",
) -> dict[str, float | int]:
    diffusion.eval()
    generator = torch.Generator(device=device).manual_seed(int(seed))
    value_error_sum = support_error_sum = 0.0
    value_elements = support_elements = 0
    counts = (0, 0, 0, 0)
    batches = 0
    for batch in loader:
        values, condition, availability, active = _prepare_batch(
            batch, device=device, scaler=scaler
        )
        with _autocast_context(device, amp_mode):
            result = diffusion.training_loss(
                values,
                condition,
                availability_mask=availability,
                active_target=active,
                generator=generator,
            )
        for name, tensor in (
            ("validation loss", result.loss),
            ("validation value loss", result.value_loss),
            ("validation support loss", result.support_loss),
            ("validation value prediction", result.prediction.value_prediction),
            ("validation active logits", result.prediction.active_logits),
        ):
            if not bool(torch.isfinite(tensor).all()):
                raise FloatingPointError(f"{name} became non-finite")
        batches += 1
        value_mask = (
            availability
            if diffusion.config.value_loss_mode == "all_measured"
            else availability & active
        )
        batch_value_elements = int(value_mask.sum().item())
        batch_support_elements = int(availability.sum().item())
        value_error_sum += float(result.value_loss) * batch_value_elements
        support_error_sum += float(result.support_loss) * batch_support_elements
        value_elements += batch_value_elements
        support_elements += batch_support_elements
        counts = _accumulate_counts(
            counts,
            _support_counts(result.prediction.active_logits, active, availability),
        )
        if max_steps and batches >= max_steps:
            break
    if batches == 0:
        raise RuntimeError("validation stream emitted no batches")
    return _metric_summary(
        value_error_sum=value_error_sum,
        value_elements=value_elements,
        support_error_sum=support_error_sum,
        support_elements=support_elements,
        value_loss_weight=float(diffusion.config.value_loss_weight),
        support_loss_weight=float(diffusion.config.support_loss_weight),
        batches=batches,
        counts=counts,
    )


def _json_ready(value: object) -> object:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if torch.is_tensor(value):
        return value.detach().cpu().tolist()
    if isinstance(value, Mapping):
        return {str(key): _json_ready(child) for key, child in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(child) for child in value]
    return value


def atomic_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(_json_ready(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def atomic_torch_save(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(dict(payload), temporary)
    os.replace(temporary, path)


def _checkpoint_payload(
    *,
    config: Mapping[str, object],
    streams: PBISCE2Streams,
    denoiser: ConditionalSBDDenoiser,
    diffusion: SparseDataDiffusion,
    scaler: SparseValueScaler,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    global_step: int,
    best_validation_loss: float,
    history: list[dict[str, object]],
    training_generator: torch.Generator,
    ema: ExponentialMovingAverage[nn.Module],
) -> dict[str, object]:
    panel = streams.data_module.gene_panel
    return {
        "schema_version": CHECKPOINT_SCHEMA,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "reference_repository": REFERENCE_REPOSITORY,
        "reference_commit": REFERENCE_COMMIT,
        "experiment_config": dict(config),
        "model_config": asdict(denoiser.config),
        "diffusion_config": asdict(diffusion.config),
        "value_scaler": scaler.state_dict(),
        "model_state": denoiser.state_dict(),
        "ema_state": ema.state_dict(),
        "training_device_type": next(denoiser.parameters()).device.type,
        "optimizer_state": optimizer.state_dict(),
        "conditioner_state": streams.conditioner.state_dict(),
        "gene_positions": streams.data_module.panel_positions.copy(),
        "gene_ids": panel["gene_id"].astype(str).tolist(),
        "gene_symbols": panel["gene_symbol"].astype(str).tolist(),
        "gene_measurement_coverage": streams.data_module.measurement_coverage.copy(),
        "full_gene_measurement_coverage": (
            streams.data_module.full_measurement_coverage.copy()
        ),
        "corpus_fingerprint": streams.data_module.corpus_fingerprint,
        "epoch": int(epoch),
        "global_step": int(global_step),
        "best_validation_loss": float(best_validation_loss),
        "history": list(history),
        "torch_rng_state": torch.get_rng_state(),
        "cuda_rng_state_all": (
            torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []
        ),
        "training_generator_state": training_generator.get_state(),
        "python_random_state": random.getstate(),
        "numpy_random_state": np.random.get_state(),
    }


def load_checkpoint(path: Path | str, *, device: torch.device) -> dict[str, object]:
    checkpoint_path = Path(path).expanduser().resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"checkpoint not found: {checkpoint_path}")
    try:
        payload = torch.load(
            checkpoint_path, map_location=device, weights_only=False
        )
    except TypeError:  # pragma: no cover
        payload = torch.load(checkpoint_path, map_location=device)
    if not isinstance(payload, dict) or payload.get("schema_version") not in (
        {CHECKPOINT_SCHEMA} | LEGACY_CHECKPOINT_SCHEMAS
    ):
        raise ValueError("unsupported PBISC-SDD checkpoint")
    return payload


def _resume_contract(config: Mapping[str, object]) -> dict[str, object]:
    if "_release" in config:
        from pbisc.training.portable import canonical_resume_config
        config = canonical_resume_config(config)
    training = _require_mapping(config, "training")
    return {
        "schema_version": config.get("schema_version"),
        "experiment_id": config.get("experiment_id"),
        "paths": _require_mapping(config, "paths"),
        "data": _require_mapping(config, "data"),
        "model": _require_mapping(config, "model"),
        "diffusion": _require_mapping(config, "diffusion"),
        # Extending the epoch target is the sole allowed training-contract
        # change. Sampling settings do not affect optimizer replay.
        "training": {
            key: value for key, value in training.items() if key != "epochs"
        },
    }


def _validate_resume_config(
    current: Mapping[str, object], checkpoint_config: Mapping[str, object]
) -> None:
    current_contract = json.dumps(
        _json_ready(_resume_contract(current)), sort_keys=True, separators=(",", ":")
    )
    checkpoint_contract = json.dumps(
        _json_ready(_resume_contract(checkpoint_config)),
        sort_keys=True,
        separators=(",", ":"),
    )
    if current_contract != checkpoint_contract:
        raise ValueError(
            "resume configuration differs from the checkpoint; only "
            "training.epochs and sampling settings may change"
        )
    completed_epochs = int(checkpoint_config["training"]["epochs"])  # type: ignore[index]
    requested_epochs = int(current["training"]["epochs"])  # type: ignore[index]
    if requested_epochs < completed_epochs:
        raise ValueError("resume training.epochs cannot move backwards")


def _validate_checkpoint_binding(
    payload: Mapping[str, object], streams: PBISCE2Streams
) -> None:
    if payload.get("corpus_fingerprint") != streams.data_module.corpus_fingerprint:
        raise ValueError("checkpoint corpus fingerprint differs from live E2 data")
    positions = np.asarray(payload.get("gene_positions"), dtype=np.int64)
    if not np.array_equal(positions, streams.data_module.panel_positions):
        raise ValueError("checkpoint gene positions differ from live selection")
    panel = streams.data_module.gene_panel
    if list(payload.get("gene_ids", [])) != panel["gene_id"].astype(str).tolist():
        raise ValueError("checkpoint gene IDs differ from the live canonical panel")
    if list(payload.get("gene_symbols", [])) != panel["gene_symbol"].astype(str).tolist():
        raise ValueError("checkpoint gene symbols differ from the live canonical panel")
    stored_coverage = np.asarray(
        payload.get("gene_measurement_coverage"), dtype=np.float32
    )
    if not np.array_equal(stored_coverage, streams.data_module.measurement_coverage):
        raise ValueError("checkpoint train-split measurement coverage differs")
    stored_full_coverage = np.asarray(
        payload.get("full_gene_measurement_coverage"), dtype=np.float32
    )
    if not np.array_equal(
        stored_full_coverage, streams.data_module.full_measurement_coverage
    ):
        raise ValueError("checkpoint full-panel measurement coverage differs")


def _restore_rng_state(
    payload: Mapping[str, object], generator: torch.Generator
) -> None:
    required = (
        "torch_rng_state",
        "training_generator_state",
        "python_random_state",
        "numpy_random_state",
    )
    missing = [name for name in required if name not in payload]
    if missing:
        raise ValueError(f"checkpoint lacks RNG state fields: {missing}")
    torch_state = payload["torch_rng_state"]
    generator_state = payload["training_generator_state"]
    if not torch.is_tensor(torch_state) or not torch.is_tensor(generator_state):
        raise ValueError("checkpoint torch RNG states must be tensors")
    torch.set_rng_state(torch_state.cpu())
    cuda_states = payload.get("cuda_rng_state_all", [])
    if torch.cuda.is_available() and cuda_states:
        if not isinstance(cuda_states, list) or not all(
            torch.is_tensor(state) for state in cuda_states
        ):
            raise ValueError("checkpoint CUDA RNG states have an invalid type")
        torch.cuda.set_rng_state_all([state.cpu() for state in cuda_states])
    generator.set_state(generator_state.cpu())
    random.setstate(payload["python_random_state"])  # type: ignore[arg-type]
    numpy_state = payload["numpy_random_state"]
    if not isinstance(numpy_state, tuple):
        raise ValueError("checkpoint NumPy RNG state has an invalid type")
    np.random.set_state(numpy_state)


@torch.no_grad()
def generate_preview(
    diffusion: SparseDataDiffusion,
    loader: DataLoader[Any],
    *,
    device: torch.device,
    scaler: SparseValueScaler,
    cells: int,
    steps: int,
    eta: float,
    active_logit_threshold: float,
    seed: int,
) -> dict[str, np.ndarray | float | int]:
    batch = next(iter(loader))
    values, condition, availability, active = _prepare_batch(
        batch, device=device, scaler=scaler
    )
    take = min(int(cells), int(values.shape[0]))
    condition = condition[:take]
    availability = availability[:take]
    active = active[:take]
    x_log_real = batch["x_log"][:take].detach().cpu().numpy()  # type: ignore[index,union-attr]
    generator = torch.Generator(device=device).manual_seed(int(seed))
    sample = diffusion.sample(
        condition,
        steps=int(steps),
        eta=float(eta),
        active_logit_threshold=float(active_logit_threshold),
        availability_mask=availability,
        clamp_values_min=0.0,
        generator=generator,
    )
    if not bool(torch.isfinite(sample.dense_values).all()) or not bool(
        torch.isfinite(sample.active_logits).all()
    ):
        raise FloatingPointError("preview sampler produced non-finite predictions")
    support_mask = sample.active_mask
    generated_unit = sample.dense_values.clamp(0.0, 1.0)
    generated_dense_log = scaler.from_unit_tensor(generated_unit)
    generated_log = torch.where(
        support_mask,
        generated_dense_log,
        torch.zeros_like(generated_dense_log),
    )
    if not bool(torch.isfinite(generated_log).all()):
        raise FloatingPointError("generated log-count output contains NaN or Inf")
    measured = availability.bool()
    active_mask = generated_log > 0
    real_zero_fraction = float((~active & measured).sum() / measured.sum().clamp_min(1))
    generated_zero_fraction = float(
        ((~active_mask) & measured).sum() / measured.sum().clamp_min(1)
    )
    support_value_disagreement = float(
        ((support_mask != active_mask) & measured).sum()
        / measured.sum().clamp_min(1)
    )
    block_ids = np.asarray(list(batch["block_id"][:take]), dtype=np.str_)  # type: ignore[index]
    recipe_k = batch["recipe_k"][:take].detach().cpu().numpy()  # type: ignore[index,union-attr]
    block_index = batch["block_index"][:take].detach().cpu().numpy()  # type: ignore[index,union-attr]
    return {
        "generated_log1p": generated_log.cpu().numpy().astype(np.float32),
        "generated_counts": torch.expm1(generated_log)
        .clamp_min(0.0)
        .cpu()
        .numpy()
        .astype(np.float32),
        "real_log1p": np.asarray(x_log_real, dtype=np.float32),
        "real_counts": np.expm1(np.asarray(x_log_real, dtype=np.float32)).astype(
            np.float32
        ),
        "active_mask": active_mask.cpu().numpy(),
        "support_mask": support_mask.cpu().numpy(),
        "availability_mask": availability.bool().cpu().numpy(),
        "active_logits": sample.active_logits.cpu().numpy().astype(np.float32),
        "block_id": block_ids,
        "block_index": np.asarray(block_index, dtype=np.int64),
        "recipe_k": np.asarray(recipe_k, dtype=np.int64),
        "real_zero_fraction": real_zero_fraction,
        "generated_zero_fraction": generated_zero_fraction,
        "support_value_disagreement_fraction": support_value_disagreement,
        "cells": take,
        "steps": int(steps),
        "eta": float(eta),
        "active_logit_threshold": float(active_logit_threshold),
        "seed": int(seed),
    }


def attach_gene_metadata(
    result: Mapping[str, object],
    streams: PBISCE2Streams,
    *,
    source: str,
    checkpoint_path: Path | None = None,
) -> dict[str, object]:
    panel = streams.data_module.gene_panel
    enriched = dict(result)
    enriched.update(
        {
            "gene_positions": streams.data_module.panel_positions.astype(np.int64),
            "gene_ids": panel["gene_id"].astype(str).to_numpy(dtype=np.str_),
            "gene_symbols": panel["gene_symbol"].astype(str).to_numpy(dtype=np.str_),
            "gene_measurement_coverage": streams.data_module.measurement_coverage.astype(
                np.float32
            ),
            "source": str(source),
            "corpus_fingerprint": streams.data_module.corpus_fingerprint,
        }
    )
    if checkpoint_path is not None:
        resolved = Path(checkpoint_path).expanduser().resolve()
        enriched["checkpoint"] = str(resolved)
        if resolved.is_file():
            enriched["checkpoint_sha256"] = _sha256_file(resolved)
    return enriched


@dataclass(frozen=True)
class ExternalBulkConditions:
    """Validated full-panel bulks and selected-panel generation masks."""

    base_conditions: np.ndarray
    bulk_measurement_fraction: np.ndarray
    output_availability_mask: np.ndarray
    recipe_k: np.ndarray
    cells_per_bulk: np.ndarray
    sample_ids: np.ndarray
    input_path: Path
    input_sha256: str


def _strict_binary_array(value: object, *, name: str) -> np.ndarray:
    raw = np.asarray(value)
    if raw.dtype.kind not in "buif":
        raise ValueError(f"{name} must be a numeric or boolean array")
    if raw.dtype.kind == "f" and not np.isfinite(raw).all():
        raise ValueError(f"{name} contains NaN or Inf")
    if not bool(((raw == 0) | (raw == 1)).all()):
        raise ValueError(f"{name} must contain exact binary 0/1 values")
    return raw.astype(bool, copy=False)


def _strict_unit_interval_array(value: object, *, name: str) -> np.ndarray:
    raw = np.asarray(value)
    if raw.dtype.kind not in "buif":
        raise ValueError(f"{name} must be numeric")
    numeric = raw.astype(np.float32, copy=False)
    if not np.isfinite(numeric).all() or bool(((numeric < 0) | (numeric > 1)).any()):
        raise ValueError(f"{name} must contain finite values in [0, 1]")
    return numeric


def _strict_positive_integer_vector(value: object, *, name: str) -> np.ndarray:
    raw = np.asarray(value)
    if raw.dtype.kind not in "uif":
        raise ValueError(f"{name} must be numeric")
    numeric = raw.astype(np.float64, copy=False).reshape(-1)
    if not np.isfinite(numeric).all() or not bool((numeric == np.floor(numeric)).all()):
        raise ValueError(f"{name} must contain finite integer values")
    if bool((numeric <= 0).any()):
        raise ValueError(f"{name} must contain positive values")
    return numeric.astype(np.int64)


def load_external_bulk_conditions(
    path: Path | str,
    streams: PBISCE2Streams,
    *,
    default_cells_per_bulk: int,
    require_gene_ids: bool = True,
) -> ExternalBulkConditions:
    """Load external bulk conditions in canonical full-panel PBISC order.

    The NPZ must contain one of ``bulk_log1p`` or ``bulk_counts``, plus
    ``bulk_measurement_fraction``, ``output_availability_mask``, and ``recipe_k``.
    The two masks are deliberately separate: genes absent from a bulk assay are
    not automatically forbidden in the desired single-cell output assay.
    Bulks may be ``[20097]`` or ``[samples, 20097]``.  A ``gene_ids`` vector is
    required by default to prevent silent vocabulary/order drift.
    """

    input_path = Path(path).expanduser().resolve()
    if not input_path.is_file():
        raise FileNotFoundError(f"external condition NPZ not found: {input_path}")
    input_sha256 = _sha256_file(input_path)
    with np.load(input_path, allow_pickle=False) as payload:
        has_log = "bulk_log1p" in payload
        has_counts = "bulk_counts" in payload
        if has_log == has_counts:
            raise ValueError("NPZ must contain exactly one of bulk_log1p or bulk_counts")
        full_panel = streams.data_module.base.gene_panel
        full_gene_ids = full_panel["gene_id"].astype(str).to_numpy(dtype=np.str_)
        n_full = int(len(full_gene_ids))
        if require_gene_ids:
            if "gene_ids" not in payload:
                raise ValueError(
                    "external NPZ requires gene_ids; use the explicit CLI override "
                    "only when canonical PBISC gene order is independently guaranteed"
                )
            observed_ids = np.asarray(payload["gene_ids"]).astype(str)
            if observed_ids.shape != (n_full,) or not np.array_equal(
                observed_ids, full_gene_ids
            ):
                raise ValueError("external gene_ids do not match canonical PBISC order")
        raw_bulk = np.asarray(
            payload["bulk_log1p"] if has_log else payload["bulk_counts"],
            dtype=np.float32,
        )
        if raw_bulk.ndim == 1:
            raw_bulk = raw_bulk[None, :]
        if raw_bulk.ndim != 2 or raw_bulk.shape[1] != n_full:
            raise ValueError(
                f"external bulk must have shape [samples, {n_full}], got {raw_bulk.shape}"
            )
        if not np.isfinite(raw_bulk).all() or bool((raw_bulk < 0).any()):
            raise ValueError("external bulk contains negative, NaN, or Inf values")
        bulk_log1p = raw_bulk if has_log else np.log1p(raw_bulk).astype(np.float32)

        if "bulk_measurement_fraction" not in payload:
            raise ValueError("external NPZ requires bulk_measurement_fraction")
        bulk_fraction = _strict_unit_interval_array(
            payload["bulk_measurement_fraction"],
            name="bulk_measurement_fraction",
        )
        if bulk_fraction.ndim == 1:
            bulk_fraction = bulk_fraction[None, :]
        if bulk_fraction.shape == (1, n_full) and bulk_log1p.shape[0] > 1:
            bulk_fraction = np.repeat(bulk_fraction, bulk_log1p.shape[0], axis=0)
        if bulk_fraction.shape != bulk_log1p.shape:
            raise ValueError("bulk_measurement_fraction must match external bulks")
        if bool((bulk_log1p[bulk_fraction == 0] != 0).any()):
            raise ValueError("unmeasured external bulk coordinates must be stored as zero")

        if "output_availability_mask" not in payload:
            raise ValueError("external NPZ requires output_availability_mask")
        output_mask = _strict_binary_array(
            payload["output_availability_mask"], name="output_availability_mask"
        )
        if output_mask.ndim == 1:
            output_mask = output_mask[None, :]
        if output_mask.shape == (1, n_full) and bulk_log1p.shape[0] > 1:
            output_mask = np.repeat(output_mask, bulk_log1p.shape[0], axis=0)
        if output_mask.shape != bulk_log1p.shape:
            raise ValueError("output_availability_mask must match external bulks")

        if "recipe_k" not in payload:
            raise ValueError("external NPZ requires recipe_k")
        recipe_k = _strict_positive_integer_vector(
            payload["recipe_k"], name="recipe_k"
        )
        if recipe_k.size == 1 and bulk_log1p.shape[0] > 1:
            recipe_k = np.repeat(recipe_k, bulk_log1p.shape[0])
        if recipe_k.shape != (bulk_log1p.shape[0],):
            raise ValueError("recipe_k must contain one positive value per bulk")

        if "cells_per_bulk" in payload:
            cells = _strict_positive_integer_vector(
                payload["cells_per_bulk"], name="cells_per_bulk"
            )
            if cells.size == 1 and bulk_log1p.shape[0] > 1:
                cells = np.repeat(cells, bulk_log1p.shape[0])
        else:
            cells = np.full(
                bulk_log1p.shape[0], int(default_cells_per_bulk), dtype=np.int64
            )
        if cells.shape != recipe_k.shape:
            raise ValueError("cells_per_bulk must contain one positive value per bulk")

        if "sample_ids" in payload:
            sample_ids = np.asarray(payload["sample_ids"]).astype(str).reshape(-1)
        else:
            sample_ids = np.asarray(
                [f"external_{index:04d}" for index in range(bulk_log1p.shape[0])],
                dtype=np.str_,
            )
        if sample_ids.shape != recipe_k.shape or any(not value for value in sample_ids):
            raise ValueError("sample_ids must contain one non-empty ID per bulk")
    if _sha256_file(input_path) != input_sha256:
        raise RuntimeError("external condition NPZ changed while it was being loaded")

    base_conditions = np.stack(
        [
            streams.conditioner.transform(row, int(k))
            for row, k in zip(bulk_log1p, recipe_k, strict=True)
        ],
        axis=0,
    ).astype(np.float32, copy=False)
    positions = streams.data_module.panel_positions
    selected_bulk_fraction = bulk_fraction[:, positions].astype(
        np.float32, copy=False
    )
    selected_output_mask = output_mask[:, positions].astype(bool, copy=False)
    return ExternalBulkConditions(
        base_conditions=base_conditions,
        bulk_measurement_fraction=selected_bulk_fraction,
        output_availability_mask=selected_output_mask,
        recipe_k=recipe_k,
        cells_per_bulk=cells,
        sample_ids=sample_ids,
        input_path=input_path,
        input_sha256=input_sha256,
    )


@torch.no_grad()
def generate_external_bulks(
    diffusion: SparseDataDiffusion,
    external: ExternalBulkConditions,
    *,
    device: torch.device,
    scaler: SparseValueScaler,
    batch_size: int,
    steps: int,
    eta: float,
    active_logit_threshold: float,
    seed: int,
) -> dict[str, object]:
    """Generate requested cells for each external bulk without expanding all conditions."""

    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if float(eta) != 0.0:
        raise ValueError(
            "external generation currently requires eta=0 so results remain "
            "independent of sampling batch size"
        )
    generated: list[np.ndarray] = []
    generated_counts: list[np.ndarray] = []
    masks: list[np.ndarray] = []
    support_masks: list[np.ndarray] = []
    logits: list[np.ndarray] = []
    sample_index: list[np.ndarray] = []
    for index, (base, available, count) in enumerate(
        zip(
            external.base_conditions,
            external.output_availability_mask,
            external.cells_per_bulk,
            strict=True,
        )
    ):
        remaining = int(count)
        produced_for_bulk = 0
        while remaining > 0:
            take = min(int(batch_size), remaining)
            base_tensor = torch.from_numpy(np.repeat(base[None, :], take, axis=0)).to(
                device=device, dtype=torch.float32
            )
            availability_tensor = torch.from_numpy(
                np.repeat(available[None, :], take, axis=0)
            ).to(device=device, dtype=torch.bool)
            bulk_mask_tensor = torch.from_numpy(
                np.repeat(
                    external.bulk_measurement_fraction[index][None, :], take, axis=0
                )
            ).to(device=device, dtype=torch.float32)
            condition = condition_with_measurement_context(
                base_tensor, bulk_mask_tensor, availability_tensor
            )
            noise_rows: list[torch.Tensor] = []
            for local_index in range(take):
                cell_seed = (
                    int(seed)
                    + index * 10_000_019
                    + produced_for_bulk
                    + local_index
                ) % (2**63 - 1)
                cell_generator = torch.Generator(device=device).manual_seed(cell_seed)
                noise_rows.append(
                    torch.randn(
                        diffusion.state_dim,
                        device=device,
                        dtype=condition.dtype,
                        generator=cell_generator,
                    )
                )
            initial_noise = torch.stack(noise_rows, dim=0)
            sample = diffusion.sample(
                condition,
                steps=int(steps),
                eta=float(eta),
                active_logit_threshold=float(active_logit_threshold),
                availability_mask=availability_tensor,
                initial_noise=initial_noise,
                clamp_values_min=0.0,
            )
            if not bool(torch.isfinite(sample.dense_values).all()) or not bool(
                torch.isfinite(sample.active_logits).all()
            ):
                raise FloatingPointError(
                    "external sampler produced non-finite predictions"
                )
            dense_log = scaler.from_unit_tensor(sample.dense_values.clamp(0.0, 1.0))
            output = torch.where(
                sample.active_mask, dense_log, torch.zeros_like(dense_log)
            )
            if not bool(torch.isfinite(output).all()):
                raise FloatingPointError("external generation produced NaN or Inf")
            generated.append(output.cpu().numpy().astype(np.float32))
            generated_counts.append(
                torch.expm1(output).clamp_min(0.0).cpu().numpy().astype(np.float32)
            )
            masks.append((output > 0).cpu().numpy())
            support_masks.append(sample.active_mask.cpu().numpy())
            logits.append(sample.active_logits.cpu().numpy().astype(np.float32))
            sample_index.append(np.full(take, index, dtype=np.int64))
            remaining -= take
            produced_for_bulk += take
    return {
        "generated_log1p": np.concatenate(generated, axis=0),
        "generated_counts": np.concatenate(generated_counts, axis=0),
        "active_mask": np.concatenate(masks, axis=0),
        "support_mask": np.concatenate(support_masks, axis=0),
        "active_logits": np.concatenate(logits, axis=0),
        "bulk_index": np.concatenate(sample_index, axis=0),
        "sample_ids": external.sample_ids,
        "bulk_measurement_fraction": external.bulk_measurement_fraction,
        "output_availability_mask": external.output_availability_mask,
        "recipe_k": external.recipe_k,
        "cells_per_bulk": external.cells_per_bulk,
        "input_condition_path": str(external.input_path),
        "input_condition_sha256": external.input_sha256,
        "batch_size": int(batch_size),
        "steps": int(steps),
        "eta": float(eta),
        "active_logit_threshold": float(active_logit_threshold),
        "seed": int(seed),
    }


def save_preview(path: Path, preview: Mapping[str, object]) -> None:
    arrays = {
        key: value
        for key, value in preview.items()
        if isinstance(value, np.ndarray)
    }
    metadata = {
        key: value
        for key, value in preview.items()
        if not isinstance(value, np.ndarray)
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays, metadata_json=json.dumps(metadata))


def run_training(
    config: Mapping[str, object],
    *,
    resume_checkpoint: Path | None = None,
) -> dict[str, object]:
    paths = _require_mapping(config, "paths")
    data = _require_mapping(config, "data")
    training = _require_mapping(config, "training")
    sampling = _require_mapping(config, "sampling")
    output_dir = Path(str(paths["output_dir"])).expanduser().resolve()
    if (
        resume_checkpoint is None
        and output_dir.exists()
        and any(output_dir.iterdir())
    ):
        raise FileExistsError(
            f"refusing to overwrite non-empty run directory: {output_dir}; "
            "choose a new output_dir or pass --resume"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    seed = int(training.get("seed", 20260627))
    seed_everything(seed)
    device = resolve_device(str(training.get("device", "auto")))
    memory_policy = configure_memory_safety(training, device)
    resume_payload: dict[str, object] | None = None
    checkpoint_config: Mapping[str, object] | None = None
    if resume_checkpoint is not None:
        resolved_resume = Path(resume_checkpoint).expanduser().resolve()
        expected_latest = output_dir / "checkpoint_latest.pt"
        if resolved_resume != expected_latest:
            raise ValueError(
                "resume must use the run's checkpoint_latest.pt; older, best, "
                "or copied checkpoints could overwrite newer epoch artifacts"
            )
        resume_payload = load_checkpoint(resume_checkpoint, device=device)
        checkpoint_config = resume_payload.get("experiment_config")  # type: ignore[assignment]
        if not isinstance(checkpoint_config, Mapping):
            raise ValueError("resume checkpoint lacks experiment_config")
        _validate_resume_config(config, checkpoint_config)

    run_started_at = datetime.now(timezone.utc).isoformat()
    run_wall_started = time.monotonic()
    atomic_json(
        output_dir / "RUN_STATUS.json",
        {
            "schema_version": STATUS_SCHEMA,
            "status": "RUNNING",
            "started_at": run_started_at,
            "device": str(device),
            "memory_policy": memory_policy,
        },
    )
    atomic_json(output_dir / "effective_config.json", dict(config))
    atomic_json(
        output_dir / "PROCESS.json",
        {
            "schema_version": "pbisc_sdd.process.v1",
            "pid": os.getpid(),
            "hostname": socket.gethostname(),
            "argv": sys.argv,
            "started_at": run_started_at,
            "output_dir": str(output_dir),
            "memory_policy": memory_policy,
        },
    )
    try:
        streams = build_streams_from_config(
            config,
            gene_positions=(
                None if resume_payload is None else resume_payload["gene_positions"]
            ),
            conditioner_state=(
                None if resume_payload is None else resume_payload["conditioner_state"]
            ),
            measurement_coverage=(
                None
                if resume_payload is None
                else resume_payload["full_gene_measurement_coverage"]
            ),
        )
        if resume_payload is not None:
            _validate_checkpoint_binding(resume_payload, streams)
        panel = streams.data_module.gene_panel
        atomic_json(
            output_dir / "data_audit.json",
            {
                "schema_version": "pbisc_sdd.data_audit.v1",
                "corpus_fingerprint": streams.data_module.corpus_fingerprint,
                "gene_positions": streams.data_module.panel_positions,
                "gene_ids": panel["gene_id"].astype(str).tolist(),
                "gene_symbols": panel["gene_symbol"].astype(str).tolist(),
                "gene_measurement_coverage": (
                    streams.data_module.measurement_coverage
                ),
                "measurement_coverage_source": (
                    "checkpoint_full_panel"
                    if resume_payload is not None
                    else "train_split_sample_pools_and_zarr_masks"
                ),
                "conditioner_state_schema": streams.conditioner.state_dict().get(
                    "schema_version"
                ),
            },
        )
        base_condition_dim = int(streams.conditioner.dim)
        n_genes = int(len(streams.data_module.panel_positions))
        denoiser, diffusion, scaler = build_model_from_config(
            config, n_genes=n_genes, base_condition_dim=base_condition_dim
        )
        diffusion.to(device)
        ema = ExponentialMovingAverage(
            denoiser,
            decay=float(training.get("ema_decay", 0.9999)),
            update_every=int(training.get("ema_update_every", 1)),
        )
        optimizer = torch.optim.AdamW(
            diffusion.parameters(),
            lr=float(training["lr"]),
            weight_decay=float(training.get("weight_decay", 0.0)),
        )
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        start_epoch = 0
        global_step = 0
        best_validation_loss = math.inf
        history: list[dict[str, object]] = []
        if resume_payload is not None:
            if resume_payload.get("schema_version") != CHECKPOINT_SCHEMA:
                raise ValueError(
                    "legacy checkpoints may be sampled but cannot resume the EMA-enabled trainer"
                )
            if asdict(denoiser.config) != dict(resume_payload["model_config"]):
                raise ValueError("resume checkpoint model configuration differs")
            if asdict(diffusion.config) != dict(resume_payload["diffusion_config"]):
                raise ValueError("resume checkpoint diffusion configuration differs")
            denoiser.load_state_dict(resume_payload["model_state"], strict=True)
            ema_state = resume_payload.get("ema_state")
            if not isinstance(ema_state, Mapping):
                raise ValueError("resume checkpoint lacks EMA state")
            ema.load_state_dict(ema_state)
            optimizer.load_state_dict(resume_payload["optimizer_state"])
            start_epoch = int(resume_payload["epoch"]) + 1
            global_step = int(resume_payload["global_step"])
            best_validation_loss = float(resume_payload["best_validation_loss"])
            stored_history = resume_payload.get("history")
            if not isinstance(stored_history, list) or len(stored_history) != start_epoch:
                raise ValueError("resume checkpoint history does not match checkpoint epoch")
            history = list(stored_history)
            if resume_payload.get("training_device_type") != device.type:
                raise ValueError(
                    "resume must use the same device type as the checkpoint so "
                    "its training-generator RNG state remains exact"
                )
            completed_snapshot = output_dir / f"checkpoint_epoch_{start_epoch:03d}.pt"
            if completed_snapshot.exists():
                snapshot_payload = load_checkpoint(completed_snapshot, device=device)
                if (
                    int(snapshot_payload.get("epoch", -1)) != start_epoch - 1
                    or int(snapshot_payload.get("global_step", -1)) != global_step
                    or snapshot_payload.get("created_at")
                    != resume_payload.get("created_at")
                ):
                    raise ValueError(
                        "latest checkpoint and immutable epoch snapshot disagree"
                    )
            else:
                # latest is the transaction commit point; recreate a snapshot
                # if a crash occurred immediately after committing latest.
                atomic_torch_save(completed_snapshot, resume_payload)

        train_loader = _loader(streams.train, training)
        validation_loader = _loader(streams.validation, training)
        epochs = int(training["epochs"])
        generator = torch.Generator(device=device).manual_seed(seed + 43)
        if resume_payload is not None:
            _restore_rng_state(resume_payload, generator)
        for epoch in range(start_epoch, epochs):
            epoch_checkpoint_path = output_dir / f"checkpoint_epoch_{epoch + 1:03d}.pt"
            if epoch_checkpoint_path.exists():
                raise FileExistsError(
                    f"refusing to overwrite existing epoch checkpoint: {epoch_checkpoint_path}"
                )
            streams.train.set_epoch(epoch)
            epoch_started = time.monotonic()
            estimated_steps = len(train_loader)
            max_train_steps = int(training.get("max_train_steps", 0))
            if max_train_steps:
                estimated_steps = min(estimated_steps, max_train_steps)

            def heartbeat(step: int, latest_loss: float, grad_norm: float) -> None:
                elapsed = max(time.monotonic() - epoch_started, 1.0e-9)
                rate = step / elapsed
                remaining = max(estimated_steps - step, 0)
                memory = enforce_memory_floor(training, device)
                atomic_json(
                    output_dir / "RUN_STATUS.json",
                    {
                        "schema_version": STATUS_SCHEMA,
                        "status": "RUNNING",
                        "started_at": run_started_at,
                        "updated_at": datetime.now(timezone.utc).isoformat(),
                        "device": str(device),
                        "epoch": epoch + 1,
                        "epochs": epochs,
                        "step_in_epoch": step,
                        "estimated_steps_in_epoch": estimated_steps,
                        "global_step": global_step + step,
                        "latest_loss": latest_loss,
                        "latest_grad_norm": grad_norm,
                        "steps_per_second": rate,
                        "estimated_seconds_remaining_in_epoch": (
                            remaining / max(rate, 1.0e-12)
                        ),
                        "last_checkpoint": (
                            None
                            if epoch == 0
                            else str(
                                output_dir / f"checkpoint_epoch_{epoch:03d}.pt"
                            )
                        ),
                        "memory": memory,
                    },
                )

            train_metrics = train_one_epoch(
                diffusion,
                train_loader,
                optimizer,
                device=device,
                scaler=scaler,
                grad_clip=float(training["grad_clip"]),
                max_steps=max_train_steps,
                generator=generator,
                amp_mode=str(training.get("amp", "none")),
                heartbeat_every_steps=int(
                    training.get("heartbeat_every_steps", 50)
                ),
                heartbeat_callback=heartbeat,
                ema=ema,
            )
            train_seconds = time.monotonic() - epoch_started
            epoch_memory = enforce_memory_floor(training, device)
            global_step += int(train_metrics["batches"])
            if ema.num_updates != global_step:
                raise RuntimeError("EMA update count diverged from optimizer global_step")
            atomic_json(
                output_dir / "RUN_STATUS.json",
                {
                    "schema_version": STATUS_SCHEMA,
                    "status": "VALIDATING",
                    "started_at": run_started_at,
                    "updated_at": datetime.now(timezone.utc).isoformat(),
                    "device": str(device),
                    "epoch": epoch + 1,
                    "epochs": epochs,
                    "global_step": global_step,
                },
            )
            streams.validation.set_epoch(0)
            ema_diffusion = SparseDataDiffusion(ema.model, diffusion.config).to(device)
            validation_started = time.monotonic()
            validation_metrics = evaluate(
                ema_diffusion,
                validation_loader,
                device=device,
                scaler=scaler,
                max_steps=int(training.get("max_validation_steps", 0)),
                seed=seed + 99,
                amp_mode=str(training.get("amp", "none")),
            )
            validation_seconds = time.monotonic() - validation_started
            record: dict[str, object] = {
                "epoch": epoch + 1,
                "global_step": global_step,
                "train": train_metrics,
                "validation": validation_metrics,
                "train_seconds": train_seconds,
                "validation_seconds": validation_seconds,
                "memory": epoch_memory,
            }
            history.append(record)
            checkpoint = _checkpoint_payload(
                config=config,
                streams=streams,
                denoiser=denoiser,
                diffusion=diffusion,
                scaler=scaler,
                optimizer=optimizer,
                epoch=epoch,
                global_step=global_step,
                best_validation_loss=min(
                    best_validation_loss, float(validation_metrics["loss"])
                ),
                history=history,
                training_generator=generator,
                ema=ema,
            )
            improved = float(validation_metrics["loss"]) < best_validation_loss
            if improved:
                atomic_torch_save(output_dir / "checkpoint_best.pt", checkpoint)
            # `latest` is the commit point. If the process dies after this
            # atomic write, resume can reconstruct the immutable epoch copy.
            atomic_torch_save(output_dir / "checkpoint_latest.pt", checkpoint)
            atomic_torch_save(epoch_checkpoint_path, checkpoint)
            if improved:
                best_validation_loss = float(validation_metrics["loss"])
            atomic_json(output_dir / "history.json", {"epochs": history})
            atomic_json(
                output_dir / "RUN_STATUS.json",
                {
                    "schema_version": STATUS_SCHEMA,
                    "status": "RUNNING",
                    "started_at": run_started_at,
                    "updated_at": datetime.now(timezone.utc).isoformat(),
                    "device": str(device),
                    "epochs_completed": epoch + 1,
                    "epochs": epochs,
                    "global_step": global_step,
                    "best_validation_loss": best_validation_loss,
                    "last_checkpoint": str(epoch_checkpoint_path),
                },
            )

        preview_diffusion = SparseDataDiffusion(ema.model, diffusion.config).to(device)
        preview = generate_preview(
            preview_diffusion,
            validation_loader,
            device=device,
            scaler=scaler,
            cells=int(sampling["preview_cells"]),
            steps=int(sampling["steps"]),
            eta=float(sampling.get("eta", 0.0)),
            active_logit_threshold=float(
                sampling.get("active_logit_threshold", 0.0)
            ),
            seed=seed + 313,
        )
        last_epoch = int(history[-1]["epoch"]) if history else start_epoch
        immutable_checkpoint = output_dir / f"checkpoint_epoch_{last_epoch:03d}.pt"
        preview = attach_gene_metadata(
            preview,
            streams,
            source="pbisc_e2_validation",
            checkpoint_path=(
                immutable_checkpoint
                if immutable_checkpoint.is_file()
                else output_dir / "checkpoint_latest.pt"
            ),
        )
        save_preview(output_dir / "sample_preview.npz", preview)
        result = {
            "schema_version": STATUS_SCHEMA,
            "status": "COMPLETED",
            "completed_at": datetime.now(timezone.utc).isoformat(),
            "device": str(device),
            "epochs_completed": len(history),
            "global_step": global_step,
            "best_validation_loss": best_validation_loss,
            "n_genes": n_genes,
            "condition_dim": denoiser.condition_dim,
            "parameters": sum(parameter.numel() for parameter in diffusion.parameters()),
            "corpus_fingerprint": streams.data_module.corpus_fingerprint,
            "real_zero_fraction_preview": preview["real_zero_fraction"],
            "generated_zero_fraction_preview": preview["generated_zero_fraction"],
            "support_value_disagreement_fraction_preview": preview[
                "support_value_disagreement_fraction"
            ],
            "wall_seconds": time.monotonic() - run_wall_started,
            "memory_policy": memory_policy,
            "memory": enforce_memory_floor(training, device),
        }
        atomic_json(output_dir / "RUN_STATUS.json", result)
        return result
    except BaseException as error:
        last_progress: dict[str, object] = {}
        status_path = output_dir / "RUN_STATUS.json"
        if status_path.is_file():
            try:
                observed = json.loads(status_path.read_text(encoding="utf-8"))
                if isinstance(observed, dict):
                    last_progress = observed
            except (OSError, ValueError):
                last_progress = {}
        latest_path = output_dir / "checkpoint_latest.pt"
        atomic_json(
            status_path,
            {
                "schema_version": STATUS_SCHEMA,
                "status": "FAILED_PRESERVED",
                "failed_at": datetime.now(timezone.utc).isoformat(),
                "error_type": type(error).__name__,
                "error": str(error),
                "wall_seconds": time.monotonic() - run_wall_started,
                "last_progress": last_progress,
                "resume_checkpoint": (
                    str(latest_path) if latest_path.is_file() else None
                ),
            },
        )
        raise


def restore_for_sampling(
    checkpoint_path: Path | str,
    *,
    device: torch.device,
    progress: bool = False,
    use_ema: bool = True,
) -> tuple[
    dict[str, object],
    PBISCE2Streams,
    ConditionalSBDDenoiser,
    SparseDataDiffusion,
    SparseValueScaler,
]:
    payload = load_checkpoint(checkpoint_path, device=device)
    config = payload.get("experiment_config")
    if not isinstance(config, Mapping):
        raise ValueError("checkpoint lacks experiment_config")
    streams = build_streams_from_config(
        config,
        gene_positions=payload["gene_positions"],
        conditioner_state=payload["conditioner_state"],
        measurement_coverage=payload["full_gene_measurement_coverage"],
        progress=progress,
    )
    _validate_checkpoint_binding(payload, streams)
    denoiser_config = SparseDenoiserConfig(
        **{
            **dict(payload["model_config"]),
            "hidden_dims": tuple(payload["model_config"]["hidden_dims"]),
        }
    )
    denoiser = ConditionalSBDDenoiser(denoiser_config).to(device)
    model_state = payload["model_state"]
    if use_ema and isinstance(payload.get("ema_state"), Mapping):
        model_state = payload["ema_state"]["model_state"]  # type: ignore[index]
    denoiser.load_state_dict(model_state, strict=True)
    diffusion = SparseDataDiffusion(
        denoiser, SparseDiffusionConfig(**dict(payload["diffusion_config"]))
    ).to(device)
    scaler = SparseValueScaler.from_state_dict(payload["value_scaler"])
    return dict(config), streams, denoiser, diffusion, scaler


__all__ = [
    "CHECKPOINT_SCHEMA",
    "CONFIG_SCHEMA",
    "REFERENCE_COMMIT",
    "REFERENCE_REPOSITORY",
    "STATUS_SCHEMA",
    "atomic_json",
    "atomic_torch_save",
    "attach_gene_metadata",
    "build_model_from_config",
    "build_streams_from_config",
    "evaluate",
    "ExternalBulkConditions",
    "generate_external_bulks",
    "generate_preview",
    "load_checkpoint",
    "load_external_bulk_conditions",
    "load_experiment_config",
    "resolve_device",
    "restore_for_sampling",
    "run_training",
    "save_preview",
    "seed_everything",
    "train_one_epoch",
]
