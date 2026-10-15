"""Additive broad-panel training utilities for the PBISC SDD ladder.

This module deliberately leaves :mod:`pbisc_sdd.experiment` unchanged because
completed experiments bind that implementation byte-for-byte. The
``train_sdd_ladder.py`` entry point consumes the public functions here while
continuing to use the stable module for model, data, metric, and checkpoint
schema definitions.

The ladder contract differs only where explicitly documented:

* ``gradient_accumulation_steps`` defaults to one and counts microbatches per
  optimizer update. ``max_steps``, heartbeat steps, ``optimizer_steps``, EMA
  updates, and the eventual checkpoint ``global_step`` all count optimizer
  updates.
* ``prefetch_factor`` is optional, positive, and only legal with one or more
  DataLoader workers.
* checkpoints are always staged on CPU (memory-mapped when supported).  A
  sampling restore copies only the selected raw or EMA model to the requested
  device; optimizer state and the unused model copy never move there.

Ladder checkpoints use ``pbisc_sdd.checkpoint.ladder.v1`` so the stable trainer
cannot silently resume them while ignoring accumulation semantics.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import copy
import gc
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import random
import socket
import sys
import time
from typing import Any, Callable, Mapping, cast

import numpy as np
import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader

from . import experiment as _stable
from .data import PBISCE2Streams, SparseValueScaler
from .diffusion import SparseDataDiffusion, SparseDiffusionConfig
from .ema import ExponentialMovingAverage
from .model import ConditionalSBDDenoiser, SparseDenoiserConfig
from .proportion_hint import ProportionHint


CHECKPOINT_SCHEMA = "pbisc_sdd.checkpoint.ladder.v1"
STABLE_CHECKPOINT_SCHEMA = _stable.CHECKPOINT_SCHEMA
CONFIG_SCHEMA = _stable.CONFIG_SCHEMA
LADDER_TRAINING_CONTRACT = "pbisc_sdd.ladder_training.v1"


@dataclass(frozen=True)
class LadderTrainingOptions:
    """Validated options that alter broad-panel loading or update cadence."""

    num_workers: int
    gradient_accumulation_steps: int
    prefetch_factor: int | None

    @classmethod
    def from_mapping(cls, training: Mapping[str, object]) -> "LadderTrainingOptions":
        workers = _stable._non_negative_int(
            training.get("num_workers", 0), name="training.num_workers"
        )
        accumulation = _stable._positive_int(
            training.get("gradient_accumulation_steps", 1),
            name="training.gradient_accumulation_steps",
        )
        prefetch: int | None = None
        if "prefetch_factor" in training:
            prefetch = _stable._positive_int(
                training["prefetch_factor"], name="training.prefetch_factor"
            )
            if workers == 0:
                raise ValueError(
                    "training.prefetch_factor is valid only when "
                    "training.num_workers > 0"
                )
        return cls(
            num_workers=workers,
            gradient_accumulation_steps=accumulation,
            prefetch_factor=prefetch,
        )


def load_experiment_config(path: Path | str) -> dict[str, object]:
    """Load the stable v1 config and validate additive ladder options."""

    payload = _stable.load_experiment_config(path)
    LadderTrainingOptions.from_mapping(_stable._require_mapping(payload, "training"))
    return payload


def build_loader(
    dataset: object, training: Mapping[str, object]
) -> DataLoader[Any]:
    """Build a loader without passing worker-only options in single-process mode."""

    options = LadderTrainingOptions.from_mapping(training)
    kwargs: dict[str, object] = {
        "dataset": dataset,
        "batch_size": int(training["batch_size"]),
        "num_workers": options.num_workers,
        "pin_memory": bool(training.get("pin_memory", False)),
        "persistent_workers": options.num_workers > 0,
        "drop_last": False,
    }
    if options.prefetch_factor is not None:
        kwargs["prefetch_factor"] = options.prefetch_factor
    return DataLoader(**kwargs)  # type: ignore[arg-type]


def optimizer_steps_for_batches(
    batches: int, *, gradient_accumulation_steps: int, max_steps: int = 0
) -> int:
    """Return the optimizer updates produced by a number of microbatches."""

    batch_count = _stable._non_negative_int(batches, name="batches")
    accumulation = _stable._positive_int(
        gradient_accumulation_steps, name="gradient_accumulation_steps"
    )
    limit = _stable._non_negative_int(max_steps, name="max_steps")
    updates = math.ceil(batch_count / accumulation)
    return min(updates, limit) if limit else updates


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
    gradient_accumulation_steps: int = 1,
    amp_mode: str = "none",
    heartbeat_every_steps: int = 0,
    heartbeat_callback: Callable[[int, float, float], None] | None = None,
    ema: ExponentialMovingAverage[nn.Module] | None = None,
) -> dict[str, float | int]:
    """Train for one epoch with optimizer-step accumulation semantics.

    ``batches`` in the returned metrics is the number of consumed
    microbatches, preserving the stable metric meaning. ``optimizer_steps`` is
    the number to add to checkpoint ``global_step``.  EMA is updated exactly
    once after every successful optimizer update.  A final incomplete group is
    normalized by its actual microbatch count before clipping and stepping.
    """

    accumulation_target = _stable._positive_int(
        gradient_accumulation_steps, name="gradient_accumulation_steps"
    )
    step_limit = _stable._non_negative_int(max_steps, name="max_steps")
    heartbeat_interval = _stable._non_negative_int(
        heartbeat_every_steps, name="heartbeat_every_steps"
    )
    diffusion.train()
    value_error_sum = support_error_sum = 0.0
    value_elements = support_elements = 0
    counts = (0, 0, 0, 0)
    batches = 0
    optimizer_steps = 0
    accumulated = 0
    latest_loss = 0.0
    optimizer.zero_grad(set_to_none=True)

    def apply_update(group_size: int) -> float:
        nonlocal optimizer_steps
        inverse_group_size = 1.0 / float(group_size)
        for parameter in diffusion.parameters():
            if parameter.grad is not None:
                parameter.grad.mul_(inverse_group_size)
        norm = nn.utils.clip_grad_norm_(diffusion.parameters(), float(grad_clip))
        if not bool(torch.isfinite(norm)):
            raise FloatingPointError("gradient norm became non-finite")
        optimizer.step()
        if ema is not None:
            ema.update(diffusion.denoiser)
        optimizer_steps += 1
        optimizer.zero_grad(set_to_none=True)
        norm_value = float(norm.detach())
        if (
            heartbeat_callback is not None
            and heartbeat_interval > 0
            and optimizer_steps % heartbeat_interval == 0
        ):
            heartbeat_callback(optimizer_steps, latest_loss, norm_value)
        return norm_value

    for batch in loader:
        values, condition, availability, active = _stable._prepare_batch(
            batch, device=device, scaler=scaler
        )
        with _stable._autocast_context(device, amp_mode):
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
        batches += 1
        accumulated += 1
        latest_loss = float(result.loss.detach())

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
        counts = _stable._accumulate_counts(
            counts,
            _stable._support_counts(
                result.prediction.active_logits.detach(), active, availability
            ),
        )

        if accumulated == accumulation_target:
            apply_update(accumulated)
            accumulated = 0
            if step_limit and optimizer_steps >= step_limit:
                break

    if batches == 0:
        raise RuntimeError("training stream emitted no batches")
    if accumulated:
        apply_update(accumulated)

    metrics = _stable._metric_summary(
        value_error_sum=value_error_sum,
        value_elements=value_elements,
        support_error_sum=support_error_sum,
        support_elements=support_elements,
        value_loss_weight=float(diffusion.config.value_loss_weight),
        support_loss_weight=float(diffusion.config.support_loss_weight),
        batches=batches,
        counts=counts,
    )
    metrics["optimizer_steps"] = optimizer_steps
    return metrics


def _load_cpu_mmap(checkpoint_path: Path) -> object:
    try:
        return torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=False,
            mmap=True,
        )
    except TypeError:  # pragma: no cover - compatibility with older PyTorch
        try:
            return torch.load(
                checkpoint_path, map_location="cpu", weights_only=False
            )
        except TypeError:  # pragma: no cover
            return torch.load(checkpoint_path, map_location="cpu")
    except RuntimeError as error:
        # mmap is unavailable for checkpoints written with PyTorch's legacy
        # serialization. Retain CPU staging in that compatibility case.
        if "mmap can only be used" not in str(error):
            raise
        try:  # pragma: no cover - current checkpoints use zip serialization
            return torch.load(
                checkpoint_path, map_location="cpu", weights_only=False
            )
        except TypeError:  # pragma: no cover
            return torch.load(checkpoint_path, map_location="cpu")


def _validate_stable_v5_migration(payload: Mapping[str, object]) -> None:
    config = payload.get("experiment_config")
    if not isinstance(config, Mapping):
        raise ValueError("stable v5 migration checkpoint lacks experiment_config")
    training = _stable._require_mapping(config, "training")
    options = LadderTrainingOptions.from_mapping(training)
    if options.gradient_accumulation_steps != 1:
        raise ValueError(
            "stable v5 migration is safe only when "
            "gradient_accumulation_steps is absent or one"
        )
    ema_state = payload.get("ema_state")
    if not isinstance(ema_state, Mapping):
        raise ValueError("stable v5 migration checkpoint lacks EMA state")
    global_step = payload.get("global_step")
    ema_updates = ema_state.get("num_updates")
    if (
        isinstance(global_step, bool)
        or not isinstance(global_step, int)
        or isinstance(ema_updates, bool)
        or not isinstance(ema_updates, int)
        or global_step != ema_updates
    ):
        raise ValueError(
            "stable v5 migration requires EMA num_updates to equal global_step"
        )


def load_checkpoint(
    path: Path | str,
    *,
    device: torch.device,
    allow_stable_v5_migration: bool = False,
) -> dict[str, object]:
    """Load a trusted checkpoint into CPU-backed storage regardless of target.

    ``device`` remains in the signature for drop-in use by a later ladder run
    loop, but accelerator placement happens only when a selected state is
    restored into a module. This prevents optimizer and alternate model state
    from being deserialized wholesale into CUDA memory.
    """

    # Validate the caller's value without using it as torch.load map_location.
    torch.device(device)
    checkpoint_path = Path(path).expanduser().resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"checkpoint not found: {checkpoint_path}")
    payload = _load_cpu_mmap(checkpoint_path)
    if not isinstance(payload, dict):
        raise ValueError("unsupported PBISC-SDD checkpoint")
    schema = payload.get("schema_version")
    if schema == CHECKPOINT_SCHEMA:
        if payload.get("training_contract") != LADDER_TRAINING_CONTRACT:
            raise ValueError(
                "ladder checkpoint training contract is missing or invalid"
            )
    elif schema == STABLE_CHECKPOINT_SCHEMA and allow_stable_v5_migration:
        _validate_stable_v5_migration(payload)
    else:
        raise ValueError("unsupported PBISC-SDD ladder checkpoint")
    return payload


def restore_rng_state(
    payload: Mapping[str, object], generator: torch.Generator
) -> None:
    """Restore all checkpoint RNG streams immediately before epoch iteration."""

    _stable._restore_rng_state(payload, generator)


def _resume_contract(config: Mapping[str, object]) -> dict[str, object]:
    contract = _stable._resume_contract(config)
    training = cast(dict[str, object], contract["training"])
    # An explicit default remains compatible with a pre-ladder checkpoint that
    # omitted the additive field. Any non-default change remains fail-closed.
    training["gradient_accumulation_steps"] = _stable._positive_int(
        training.get("gradient_accumulation_steps", 1),
        name="training.gradient_accumulation_steps",
    )
    # ``prefetch_factor`` only bounds how many batches each worker prepares
    # ahead of the training loop. The DataLoader still returns batches in
    # worker round-robin order, and each worker's stream is fixed by its
    # worker id and the epoch, so the collated batch sequence is identical for
    # any prefetch depth (verified on the all-cell stream, 2026-09-16). It is
    # therefore a scheduling knob, not part of the exact-replay contract.
    # ``num_workers`` stays in the contract: it changes the worker sharding.
    training.pop("prefetch_factor", None)
    return contract


def validate_resume_config(
    current: Mapping[str, object], checkpoint_config: Mapping[str, object]
) -> None:
    """Validate exact replay, allowing only epochs/sampling and an explicit default."""

    LadderTrainingOptions.from_mapping(_stable._require_mapping(current, "training"))
    LadderTrainingOptions.from_mapping(
        _stable._require_mapping(checkpoint_config, "training")
    )
    current_contract = json.dumps(
        _stable._json_ready(_resume_contract(current)),
        sort_keys=True,
        separators=(",", ":"),
    )
    checkpoint_contract = json.dumps(
        _stable._json_ready(_resume_contract(checkpoint_config)),
        sort_keys=True,
        separators=(",", ":"),
    )
    if current_contract != checkpoint_contract:
        raise ValueError(
            "resume configuration differs from the checkpoint; only "
            "training.epochs and sampling settings may change"
        )
    completed_target = int(checkpoint_config["training"]["epochs"])  # type: ignore[index]
    requested_target = int(current["training"]["epochs"])  # type: ignore[index]
    if requested_target < completed_target:
        raise ValueError("resume training.epochs cannot move backwards")


def _seed_for_device(seed: int, device: torch.device) -> None:
    """Seed the selected training device without initializing an unused CUDA path."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)


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
    """Build a ladder payload without probing CUDA for a CPU-only run."""

    panel = streams.data_module.gene_panel
    training_device = next(denoiser.parameters()).device
    return {
        "schema_version": CHECKPOINT_SCHEMA,
        "training_contract": LADDER_TRAINING_CONTRACT,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "reference_repository": _stable.REFERENCE_REPOSITORY,
        "reference_commit": _stable.REFERENCE_COMMIT,
        "experiment_config": dict(config),
        "model_config": asdict(denoiser.config),
        "diffusion_config": asdict(diffusion.config),
        "value_scaler": scaler.state_dict(),
        "model_state": denoiser.state_dict(),
        "ema_state": ema.state_dict(),
        "training_device_type": training_device.type,
        "optimizer_state": optimizer.state_dict(),
        "conditioner_state": streams.conditioner.state_dict(),
        "proportion_hint": _stable.active_proportion_hint_contract(),
        "gene_positions": streams.data_module.panel_positions.copy(),
        "gene_ids": panel["gene_id"].astype(str).tolist(),
        "gene_symbols": panel["gene_symbol"].astype(str).tolist(),
        "gene_measurement_coverage": (
            streams.data_module.measurement_coverage.copy()
        ),
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
            torch.cuda.get_rng_state_all() if training_device.type == "cuda" else []
        ),
        "training_generator_state": training_generator.get_state(),
        "python_random_state": random.getstate(),
        "numpy_random_state": np.random.get_state(),
    }


def run_training(
    config: Mapping[str, object],
    *,
    resume_checkpoint: Path | None = None,
    allow_stable_v5_migration: bool = False,
    _stream_builder: Callable[..., PBISCE2Streams] | None = None,
) -> dict[str, object]:
    """Run the versioned trainer with ladder-only checkpoint semantics.

    The private stream-builder seam lets CPU contract tests supply tiny streams;
    production callers always use the stable PBISC E2 builder.
    """

    paths = _stable._require_mapping(config, "paths")
    _stable._require_mapping(config, "data")
    training = _stable._require_mapping(config, "training")
    sampling = _stable._require_mapping(config, "sampling")
    options = LadderTrainingOptions.from_mapping(training)
    output_dir = Path(str(paths["output_dir"])).expanduser().resolve()
    if resume_checkpoint is None and output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(
            f"refusing to overwrite non-empty run directory: {output_dir}; "
            "choose a new output_dir or pass --resume"
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    seed = int(training.get("seed", 20260627))
    device = _stable.resolve_device(str(training.get("device", "auto")))
    _seed_for_device(seed, device)
    memory_policy = _stable.configure_memory_safety(training, device)
    resume_payload: dict[str, object] | None = None
    if resume_checkpoint is not None:
        resolved_resume = Path(resume_checkpoint).expanduser().resolve()
        expected_latest = output_dir / "checkpoint_latest.pt"
        if resolved_resume != expected_latest:
            raise ValueError(
                "resume must use the run's checkpoint_latest.pt; older, best, "
                "or copied checkpoints could overwrite newer epoch artifacts"
            )
        resume_payload = load_checkpoint(
            resolved_resume,
            device=device,
            allow_stable_v5_migration=allow_stable_v5_migration,
        )
        checkpoint_config = resume_payload.get("experiment_config")
        if not isinstance(checkpoint_config, Mapping):
            raise ValueError("resume checkpoint lacks experiment_config")
        validate_resume_config(config, checkpoint_config)

    run_started_at = datetime.now(timezone.utc).isoformat()
    run_wall_started = time.monotonic()
    _stable.atomic_json(
        output_dir / "RUN_STATUS.json",
        {
            "schema_version": _stable.STATUS_SCHEMA,
            "status": "RUNNING",
            "started_at": run_started_at,
            "device": str(device),
            "memory_policy": memory_policy,
            "training_contract": LADDER_TRAINING_CONTRACT,
        },
    )
    _stable.atomic_json(output_dir / "effective_config.json", dict(config))
    _stable.atomic_json(
        output_dir / "PROCESS.json",
        {
            "schema_version": "pbisc_sdd.process.v1",
            "pid": os.getpid(),
            "hostname": socket.gethostname(),
            "argv": sys.argv,
            "started_at": run_started_at,
            "output_dir": str(output_dir),
            "memory_policy": memory_policy,
            "training_contract": LADDER_TRAINING_CONTRACT,
        },
    )

    try:
        hint = ProportionHint.from_data_section(_stable._require_mapping(config, "data"))
        _stable.set_active_proportion_hint(hint)
        stream_builder = _stable.build_streams_from_config
        if _stream_builder is not None:
            stream_builder = _stream_builder
        streams = stream_builder(
            config,
            gene_positions=(
                None if resume_payload is None else resume_payload["gene_positions"]
            ),
            conditioner_state=(
                None
                if resume_payload is None
                else resume_payload["conditioner_state"]
            ),
            measurement_coverage=(
                None
                if resume_payload is None
                else resume_payload["full_gene_measurement_coverage"]
            ),
        )
        if resume_payload is not None:
            _stable._validate_checkpoint_binding(resume_payload, streams)
            _validate_hint_binding(resume_payload, hint)
        panel = streams.data_module.gene_panel
        _stable.atomic_json(
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
                "training_contract": LADDER_TRAINING_CONTRACT,
            },
        )

        base_condition_dim = int(streams.conditioner.dim)
        if hint is not None:
            base_recipes = getattr(getattr(streams.data_module, "base", None), "recipes", None)
            if base_recipes is not None:
                coverage = hint.verify_coverage(
                    base_recipes["block_id"].astype(str).tolist(), base_recipes["sample_key"].astype(str).tolist()
                )
            else:
                coverage = {"skipped": "data module exposes no recipe frame"}
            base_condition_dim += int(hint.dim)
            print(json.dumps({"proportion_hint": {"dim": hint.dim, "labels": list(hint.labels), "coverage": coverage}}), flush=True)
        n_genes = int(len(streams.data_module.panel_positions))
        denoiser, diffusion, scaler = _stable.build_model_from_config(
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
            resume_schema = resume_payload.get("schema_version")
            if resume_schema not in {CHECKPOINT_SCHEMA, STABLE_CHECKPOINT_SCHEMA}:
                raise ValueError(
                    "only ladder checkpoints or explicitly migrated stable v5 "
                    "checkpoints may resume the ladder trainer"
                )
            if asdict(denoiser.config) != dict(resume_payload["model_config"]):
                raise ValueError("resume checkpoint model configuration differs")
            if asdict(diffusion.config) != dict(resume_payload["diffusion_config"]):
                raise ValueError("resume checkpoint diffusion configuration differs")
            if resume_payload.get("training_device_type") != device.type:
                raise ValueError(
                    "resume must use the same device type as the checkpoint so "
                    "its training-generator RNG state remains exact"
                )
            # ``torch.load(mmap=True)`` maps the whole checkpoint once, so deleting
            # individual tensors cannot unmap it: every page touched while copying
            # to the device stays charged to the job's cgroup until the last
            # reference to that payload is gone. Holding model, EMA and optimizer
            # state in one mapping therefore needs the full checkpoint resident --
            # 44.6 GB for the 2.79B-parameter 16x run, which exceeded the unit's
            # memory ceiling and thrashed the loader to a standstill (2026-09-17).
            # Each large component now gets its own short-lived mapping, so only
            # that component's pages are ever resident. Values are unchanged.
            migrate_component = resume_schema == STABLE_CHECKPOINT_SCHEMA

            def consume_component(
                key: str, apply: Callable[[Mapping[str, Tensor]], object]
            ) -> None:
                component_payload = load_checkpoint(
                    resolved_resume,
                    device=device,
                    allow_stable_v5_migration=migrate_component,
                )
                try:
                    state = component_payload.get(key)
                    if not isinstance(state, Mapping):
                        raise ValueError(f"resume checkpoint lacks {key}")
                    apply(cast(Mapping[str, Tensor], state))
                    del state
                finally:
                    component_payload.clear()
                    del component_payload
                    gc.collect()
                    # The mapping is gone, but its pages stay in the page cache
                    # and keep counting against the host's available memory
                    # until the kernel reclaims them. Dropping them here keeps
                    # the resident set at roughly one component instead of the
                    # whole checkpoint (measured 2026-09-17: without this the
                    # 16x resume bottomed out at 20 GiB available).
                    try:
                        with open(resolved_resume, "rb") as handle:
                            os.posix_fadvise(
                                handle.fileno(), 0, 0, os.POSIX_FADV_DONTNEED
                            )
                    except (OSError, AttributeError):  # pragma: no cover
                        pass
                    if device.type == "cuda":
                        torch.cuda.empty_cache()

            def stage_optimizer_state(chunk: int = 8) -> dict[str, object]:
                """Move the saved optimizer state to the device a few entries at a time.

                The optimizer state is the largest component (22.3 GiB for the 16x
                run). Copying it through a single mapping keeps its pages resident
                while the device copy grows, so both are held at once and the host
                runs out of memory (measured 2026-09-17). Re-mapping every ``chunk``
                entries bounds the resident pages instead. ``load_state_dict`` still
                performs its own id mapping and dtype policy; the tensors handed to
                it are already on the device, so its cast is a no-op. ``step`` stays
                exactly as saved because PyTorch keeps it on the CPU unless the
                optimizer is capturable or fused.
                """
                payload = load_checkpoint(
                    resolved_resume,
                    device=device,
                    allow_stable_v5_migration=migrate_component,
                )
                saved = payload.get("optimizer_state")
                if not isinstance(saved, Mapping):
                    raise ValueError("resume checkpoint lacks optimizer_state")
                param_groups = copy.deepcopy(saved["param_groups"])
                entry_keys = list(cast(Mapping[Any, Any], saved["state"]).keys())
                payload.clear()
                del payload, saved
                gc.collect()
                staged: dict[Any, dict[Any, Any]] = {}
                for start in range(0, len(entry_keys), max(int(chunk), 1)):
                    window = entry_keys[start : start + max(int(chunk), 1)]
                    chunk_payload = load_checkpoint(
                        resolved_resume,
                        device=device,
                        allow_stable_v5_migration=migrate_component,
                    )
                    try:
                        chunk_state = cast(
                            Mapping[Any, Any],
                            cast(Mapping[str, Any], chunk_payload["optimizer_state"])[
                                "state"
                            ],
                        )
                        for entry_key in window:
                            staged[entry_key] = {
                                name: (
                                    value.to(device)
                                    if name != "step" and isinstance(value, Tensor)
                                    else value.clone() if isinstance(value, Tensor) else value
                                )
                                for name, value in chunk_state[entry_key].items()
                            }
                        del chunk_state
                    finally:
                        chunk_payload.clear()
                        del chunk_payload
                        gc.collect()
                        try:
                            with open(resolved_resume, "rb") as handle:
                                os.posix_fadvise(
                                    handle.fileno(), 0, 0, os.POSIX_FADV_DONTNEED
                                )
                        except (OSError, AttributeError):  # pragma: no cover
                            pass
                return {"state": staged, "param_groups": param_groups}

            consume_component(
                "model_state",
                lambda state: denoiser.load_state_dict(state, strict=True),
            )
            consume_component("ema_state", ema.load_state_dict)
            optimizer.load_state_dict(
                cast(Mapping[str, Tensor], stage_optimizer_state())
            )
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()
            start_epoch = int(resume_payload["epoch"]) + 1
            global_step = int(resume_payload["global_step"])
            best_validation_loss = float(resume_payload["best_validation_loss"])
            stored_history = resume_payload.get("history")
            if (
                not isinstance(stored_history, list)
                or len(stored_history) != start_epoch
            ):
                raise ValueError(
                    "resume checkpoint history does not match checkpoint epoch"
                )
            history = list(stored_history)
            if ema.num_updates != global_step:
                raise ValueError(
                    "resume checkpoint EMA update count differs from global_step"
                )
            completed_snapshot = output_dir / f"checkpoint_epoch_{start_epoch:03d}.pt"
            if completed_snapshot.exists():
                snapshot_payload = load_checkpoint(
                    completed_snapshot,
                    device=device,
                    allow_stable_v5_migration=(
                        resume_schema == STABLE_CHECKPOINT_SCHEMA
                    ),
                )
                try:
                    if (
                        int(snapshot_payload.get("epoch", -1)) != start_epoch - 1
                        or int(snapshot_payload.get("global_step", -1)) != global_step
                        or snapshot_payload.get("created_at")
                        != resume_payload.get("created_at")
                    ):
                        raise ValueError(
                            "latest checkpoint and immutable epoch snapshot disagree"
                        )
                finally:
                    snapshot_payload.clear()
            else:
                _stable.atomic_torch_save(completed_snapshot, resume_payload)

        train_loader = build_loader(streams.train, training)
        validation_loader = build_loader(streams.validation, training)
        epochs = int(training["epochs"])
        generator = torch.Generator(device=device).manual_seed(seed + 43)
        if resume_payload is not None:
            restore_rng_state(resume_payload, generator)
            # Model, EMA, optimizer, history, and RNG have been copied. Drop the
            # mmap-backed checkpoint graph before the first accelerator batch.
            resume_payload.clear()
            resume_payload = None

        for epoch in range(start_epoch, epochs):
            epoch_checkpoint_path = output_dir / f"checkpoint_epoch_{epoch + 1:03d}.pt"
            if epoch_checkpoint_path.exists():
                raise FileExistsError(
                    f"refusing to overwrite existing epoch checkpoint: "
                    f"{epoch_checkpoint_path}"
                )
            streams.train.set_epoch(epoch)
            epoch_started = time.monotonic()
            max_train_steps = int(training.get("max_train_steps", 0))
            estimated_steps = optimizer_steps_for_batches(
                len(train_loader),
                gradient_accumulation_steps=options.gradient_accumulation_steps,
                max_steps=max_train_steps,
            )

            def heartbeat(step: int, latest_loss: float, grad_norm: float) -> None:
                elapsed = max(time.monotonic() - epoch_started, 1.0e-9)
                rate = step / elapsed
                remaining = max(estimated_steps - step, 0)
                memory = _stable.enforce_memory_floor(training, device)
                _stable.atomic_json(
                    output_dir / "RUN_STATUS.json",
                    {
                        "schema_version": _stable.STATUS_SCHEMA,
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
                            else str(output_dir / f"checkpoint_epoch_{epoch:03d}.pt")
                        ),
                        "memory": memory,
                        "training_contract": LADDER_TRAINING_CONTRACT,
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
                gradient_accumulation_steps=(
                    options.gradient_accumulation_steps
                ),
                amp_mode=str(training.get("amp", "none")),
                heartbeat_every_steps=int(
                    training.get("heartbeat_every_steps", 50)
                ),
                heartbeat_callback=heartbeat,
                ema=ema,
            )
            train_seconds = time.monotonic() - epoch_started
            epoch_memory = _stable.enforce_memory_floor(training, device)
            optimizer_steps = int(train_metrics["optimizer_steps"])
            global_step += optimizer_steps
            if ema.num_updates != global_step:
                raise RuntimeError(
                    "EMA update count diverged from optimizer global_step"
                )
            _stable.atomic_json(
                output_dir / "RUN_STATUS.json",
                {
                    "schema_version": _stable.STATUS_SCHEMA,
                    "status": "VALIDATING",
                    "started_at": run_started_at,
                    "updated_at": datetime.now(timezone.utc).isoformat(),
                    "device": str(device),
                    "epoch": epoch + 1,
                    "epochs": epochs,
                    "global_step": global_step,
                    "training_contract": LADDER_TRAINING_CONTRACT,
                },
            )
            streams.validation.set_epoch(0)
            ema_diffusion = SparseDataDiffusion(ema.model, diffusion.config).to(device)
            validation_started = time.monotonic()
            validation_metrics = _stable.evaluate(
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
                _stable.atomic_torch_save(
                    output_dir / "checkpoint_best.pt", checkpoint
                )
            # `latest` is the commit point. Resume can reconstruct an immutable
            # epoch snapshot if the process dies after this atomic write.
            _stable.atomic_torch_save(
                output_dir / "checkpoint_latest.pt", checkpoint
            )
            _stable.atomic_torch_save(epoch_checkpoint_path, checkpoint)
            if improved:
                best_validation_loss = float(validation_metrics["loss"])
            _stable.atomic_json(
                output_dir / "history.json",
                {
                    "training_contract": LADDER_TRAINING_CONTRACT,
                    "epochs": history,
                },
            )
            _stable.atomic_json(
                output_dir / "RUN_STATUS.json",
                {
                    "schema_version": _stable.STATUS_SCHEMA,
                    "status": "RUNNING",
                    "started_at": run_started_at,
                    "updated_at": datetime.now(timezone.utc).isoformat(),
                    "device": str(device),
                    "epochs_completed": epoch + 1,
                    "epochs": epochs,
                    "global_step": global_step,
                    "best_validation_loss": best_validation_loss,
                    "last_checkpoint": str(epoch_checkpoint_path),
                    "training_contract": LADDER_TRAINING_CONTRACT,
                },
            )

        preview_diffusion = SparseDataDiffusion(ema.model, diffusion.config).to(device)
        preview = _stable.generate_preview(
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
        preview = _stable.attach_gene_metadata(
            preview,
            streams,
            source="pbisc_e2_validation",
            checkpoint_path=(
                immutable_checkpoint
                if immutable_checkpoint.is_file()
                else output_dir / "checkpoint_latest.pt"
            ),
        )
        _stable.save_preview(output_dir / "sample_preview.npz", preview)
        result = {
            "schema_version": _stable.STATUS_SCHEMA,
            "status": "COMPLETED",
            "completed_at": datetime.now(timezone.utc).isoformat(),
            "device": str(device),
            "epochs_completed": len(history),
            "global_step": global_step,
            "best_validation_loss": best_validation_loss,
            "n_genes": n_genes,
            "condition_dim": denoiser.condition_dim,
            "parameters": sum(
                parameter.numel() for parameter in diffusion.parameters()
            ),
            "corpus_fingerprint": streams.data_module.corpus_fingerprint,
            "real_zero_fraction_preview": preview["real_zero_fraction"],
            "generated_zero_fraction_preview": preview[
                "generated_zero_fraction"
            ],
            "support_value_disagreement_fraction_preview": preview[
                "support_value_disagreement_fraction"
            ],
            "wall_seconds": time.monotonic() - run_wall_started,
            "memory_policy": memory_policy,
            "memory": _stable.enforce_memory_floor(training, device),
            "training_contract": LADDER_TRAINING_CONTRACT,
        }
        _stable.atomic_json(output_dir / "RUN_STATUS.json", result)
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
        _stable.atomic_json(
            status_path,
            {
                "schema_version": _stable.STATUS_SCHEMA,
                "status": "FAILED_PRESERVED",
                "failed_at": datetime.now(timezone.utc).isoformat(),
                "error_type": type(error).__name__,
                "error": str(error),
                "wall_seconds": time.monotonic() - run_wall_started,
                "last_progress": last_progress,
                "resume_checkpoint": (
                    str(latest_path) if latest_path.is_file() else None
                ),
                "training_contract": LADDER_TRAINING_CONTRACT,
            },
        )
        raise
    finally:
        _stable.set_active_proportion_hint(None)


def _validate_hint_binding(payload: Mapping[str, object], hint: "ProportionHint | None") -> None:
    """A hint-trained checkpoint resumes only with the same table/labels; a hint-free one only without."""

    recorded = payload.get("proportion_hint")
    if recorded is None and hint is None:
        return
    if (recorded is None) != (hint is None):
        raise ValueError(
            "proportion hint mismatch: checkpoint "
            + ("carries" if recorded is not None else "lacks")
            + " a proportion_hint contract but the config "
            + ("lacks" if hint is None else "carries")
            + " data.proportion_hint"
        )
    assert recorded is not None and hint is not None
    current = hint.contract()
    for key in ("labels", "dim", "table_sha256", "column_prefix"):
        if recorded.get(key) != current.get(key):
            raise ValueError(f"proportion hint mismatch on {key}: checkpoint {recorded.get(key)!r} vs config {current.get(key)!r}")


def restore_for_sampling(
    checkpoint_path: Path | str,
    *,
    device: torch.device,
    progress: bool = False,
    use_ema: bool = True,
    allow_stable_v5_migration: bool = False,
) -> tuple[
    dict[str, object],
    PBISCE2Streams,
    ConditionalSBDDenoiser,
    SparseDataDiffusion,
    SparseValueScaler,
]:
    """Restore only the requested model state, moving it to ``device`` last."""

    payload = load_checkpoint(
        checkpoint_path,
        device=device,
        allow_stable_v5_migration=allow_stable_v5_migration,
    )
    config = payload.get("experiment_config")
    if not isinstance(config, Mapping):
        raise ValueError("checkpoint lacks experiment_config")
    if payload.get("proportion_hint") is not None:
        raise ValueError(
            "checkpoint was trained with a cell-type proportion hint; this restore path has no "
            "hint source — sample it with scripts/sample_sdd_standalone.py --proportion-csv"
        )
    streams = _stable.build_streams_from_config(
        config,
        gene_positions=payload["gene_positions"],
        conditioner_state=payload["conditioner_state"],
        measurement_coverage=payload["full_gene_measurement_coverage"],
        progress=progress,
    )
    _stable._validate_checkpoint_binding(payload, streams)

    raw_model_config = payload.get("model_config")
    raw_diffusion_config = payload.get("diffusion_config")
    raw_scaler = payload.get("value_scaler")
    if not isinstance(raw_model_config, Mapping):
        raise ValueError("checkpoint lacks model_config")
    if not isinstance(raw_diffusion_config, Mapping):
        raise ValueError("checkpoint lacks diffusion_config")
    if not isinstance(raw_scaler, Mapping):
        raise ValueError("checkpoint lacks value_scaler")

    model_state: object = payload.get("model_state")
    selected_ema = False
    if use_ema and isinstance(payload.get("ema_state"), Mapping):
        model_state = payload["ema_state"].get("model_state")  # type: ignore[union-attr]
        selected_ema = True
    if not isinstance(model_state, Mapping):
        source = "EMA" if selected_ema else "raw"
        raise ValueError(f"checkpoint lacks a valid {source} model state")

    denoiser_config = SparseDenoiserConfig(
        **{
            **dict(raw_model_config),
            "hidden_dims": tuple(raw_model_config["hidden_dims"]),
        }
    )
    diffusion_config = SparseDiffusionConfig(**dict(raw_diffusion_config))
    scaler = SparseValueScaler.from_state_dict(raw_scaler)
    denoiser = ConditionalSBDDenoiser(denoiser_config)
    denoiser.load_state_dict(cast(Mapping[str, Tensor], model_state), strict=True)

    # Drop all mapped checkpoint references before accelerator transfer. The
    # selected state has already been copied into the CPU module.
    del model_state, raw_model_config, raw_diffusion_config, raw_scaler
    payload.clear()
    diffusion = SparseDataDiffusion(denoiser, diffusion_config).to(device)
    return dict(config), streams, denoiser, diffusion, scaler


__all__ = [
    "CHECKPOINT_SCHEMA",
    "CONFIG_SCHEMA",
    "LADDER_TRAINING_CONTRACT",
    "STABLE_CHECKPOINT_SCHEMA",
    "LadderTrainingOptions",
    "build_loader",
    "load_checkpoint",
    "load_experiment_config",
    "optimizer_steps_for_batches",
    "restore_for_sampling",
    "restore_rng_state",
    "run_training",
    "train_one_epoch",
    "validate_resume_config",
]
