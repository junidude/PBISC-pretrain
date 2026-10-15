#!/usr/bin/env python3
"""Resume sealed all-cell SDD at explicit global-epoch boundaries.

Packaged from the original bounded continuation trainer. This version
retains the ladder orchestration loop, reusing its model, optimizer, EMA,
checkpoint, and RNG helpers. Every segment is an incomplete study; a scheduler
must independently decide saturation. The sealed 16-slot recipe and target
order repeats, while training RNG and epoch metrics continue globally.
"""
from __future__ import annotations

import argparse
import copy
from dataclasses import replace
from datetime import datetime, timezone
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Mapping

import numpy as np

from .allcell import (
    POLICY, batch_count, combine_recipe_frames, digest, file_sha,
    load_topup_block, managed_gate, pool_digest, validate_stream_contract, write_json,
)

if TYPE_CHECKING:
    from pbisc_sdd.data import PBISCE2Streams

SCHEDULE_SLOTS = 16
ROUNDROBIN_CONTRACT = "pbisc_sdd.allcell_roundrobin.v1"
VALIDATION_PROTOCOL = (
    "unchanged: validation.set_epoch(0), seed=training.seed+99, "
    "existing max_validation_steps and loss weights"
)
BASE_TRAINER_SHA256 = "0b59d4cdb29ef42d2b4ecbc7a2feaa2152c91185171e0b5937601eeff6508456"
BASE_LADDER_SHA256 = "b2cbd12bfb65a6af4ee69f9cb27b4a3268ba1a58522f22471fbefd019ef105cc"


def schedule_slot(global_epoch: int) -> int:
    """Map a zero-based global epoch to the unchanged 16-slot data schedule."""
    if isinstance(global_epoch, bool) or not isinstance(global_epoch, int) or global_epoch < 0:
        raise ValueError("global epoch must be a nonnegative integer")
    return global_epoch % SCHEDULE_SLOTS


def recipes_for_global_epoch(data_module: Any, *, split: str, seed: int,
                             platform: str | None, topups: Any, global_epoch: int) -> Any:
    slot = schedule_slot(global_epoch)
    registered = data_module.split_manifest(
        split, max_blocks=8192, seed=seed + 1000003 * slot, platform=platform)
    return combine_recipe_frames(registered, topups, slot)


def worker_seed(seed: int, global_epoch: int, worker: int) -> int:
    """Replay full sealed data order; the training generator remains separate."""
    return seed + 1000003 * schedule_slot(global_epoch) + 7919 + 104729 * worker


def runtime_config(sealed_config: Mapping[str, Any], stop_after_epoch: int) -> dict[str, Any]:
    if isinstance(stop_after_epoch, bool) or not isinstance(stop_after_epoch, int) or stop_after_epoch <= 0:
        raise ValueError("stop-after-epoch must be a positive global epoch")
    config = copy.deepcopy(dict(sealed_config))
    config["training"]["epochs"] = max(int(config["training"]["epochs"]), stop_after_epoch)
    return config


def _binding(path: Path) -> dict[str, str]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return {"path": str(path.resolve()), "sha256": file_sha(path)}


def _finite_metrics(value: object) -> None:
    if isinstance(value, Mapping):
        for item in value.values():
            _finite_metrics(item)
    elif isinstance(value, (float, np.floating)) and not math.isfinite(float(value)):
        raise ValueError("Epoch metric is non-finite")


def verify_epoch_commit(out: Path, epoch: int, history: list[dict[str, Any]],
                        schedules: list[dict[str, Any]], config_sha256: str) -> dict[str, Any]:
    """Commit evidence only after full exhaustion, validation, and durable saves."""
    if epoch <= 0 or len(history) != epoch or [row["epoch"] for row in history] != list(range(1, epoch + 1)):
        raise ValueError("Epoch history is not contiguous through the requested boundary")
    slot = schedule_slot(epoch - 1)
    schedule = schedules[slot]
    expected_workers = schedule["target_cells_by_worker"]
    if len(expected_workers) != 4:
        raise ValueError("All-cell commits require four worker proofs")
    workers = []
    for worker, expected_cells in enumerate(expected_workers):
        path = out / "data_observation" / f"epoch_{epoch:03d}_worker_{worker}.json"
        row = json.loads(path.read_text())
        recipe_ids = schedule["recipe_ids"][worker::4]
        recipe_order = hashlib.sha256(b"".join(str(rid).encode("ascii") + b"\n" for rid in recipe_ids)).hexdigest()
        if (row.get("status") != "EXHAUSTED" or row.get("epoch") != epoch
                or row.get("worker") != worker or row.get("target_cells") != expected_cells
                or row.get("recipes") != len(recipe_ids) or row.get("recipe_order_sha256") != recipe_order):
            raise ValueError(f"Worker {worker} epoch {epoch} exhaustion or recipe coverage differs")
        target_hash = str(row.get("target_order_sha256", ""))
        if len(target_hash) != 64 or any(c not in "0123456789abcdef" for c in target_hash):
            raise ValueError("Worker target-order digest is invalid")
        workers.append({**row, **_binding(path)})
    record = history[-1]
    _finite_metrics(record["train"])
    _finite_metrics(record["validation"])
    expected_steps = batch_count(expected_workers, 256)
    if int(record["train"]["optimizer_steps"]) != expected_steps:
        raise ValueError("Optimizer steps differ from complete worker streams")
    expected_global_step = sum(int(row["train"]["optimizer_steps"]) for row in history)
    if int(record["global_step"]) != expected_global_step:
        raise ValueError("Global optimizer count differs from continuous history")
    persisted_history = json.loads((out / "history.json").read_text())["epochs"]
    if persisted_history != history:
        raise ValueError("Persisted history differs from checkpoint history")
    commit = {
        "schema_version": "pbisc_sdd.allcell_epoch_commit.v1", "status": "EPOCH_COMMITTED",
        "epoch": epoch, "schedule_slot": slot, "schedule_cycle": (epoch - 1) // SCHEDULE_SLOTS + 1,
        "study_complete": False, "global_step": expected_global_step,
        "config_sha256": config_sha256, "trainer_sha256": file_sha(Path(__file__)),
        "expected_optimizer_steps": expected_steps, "observed_optimizer_steps": expected_steps,
        "expected_target_cells": sum(expected_workers),
        "observed_target_cells": sum(row["target_cells"] for row in workers),
        "worker_observations": workers, "validation_protocol": VALIDATION_PROTOCOL,
        "checkpoint_latest": _binding(out / "checkpoint_latest.pt"),
        "checkpoint_epoch": _binding(out / f"checkpoint_epoch_{epoch:03d}.pt"),
        "history_sha256": file_sha(out / "history.json"),
    }
    path = out / "epoch_commits" / f"epoch_{epoch:03d}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if json.loads(path.read_text()) != commit:
            raise ValueError(f"Preserve conflicting immutable epoch commit: {path}")
    else:
        write_json(path, commit)
    return {**_binding(path), "epoch": epoch}

def run_training(
    config: Mapping[str, object],
    *,
    stop_after_epoch: int,
    _epoch_commit_callback: Callable[[int, list[dict[str, Any]]], None] | None = None,
    resume_checkpoint: Path | None = None,
    allow_stable_v5_migration: bool = False,
    _stream_builder: Callable[..., PBISCE2Streams] | None = None,
) -> dict[str, object]:
    """Execute a bounded segment with the sealed ladder state transition order."""
    import os
    import socket
    import time
    from dataclasses import asdict
    from typing import cast
    import torch
    from torch import Tensor
    from pbisc_sdd import experiment as _stable
    from pbisc_sdd.diffusion import SparseDataDiffusion
    from pbisc_sdd.ema import ExponentialMovingAverage
    from pbisc_sdd.ladder_experiment import (
        CHECKPOINT_SCHEMA, STABLE_CHECKPOINT_SCHEMA, LADDER_TRAINING_CONTRACT,
        LadderTrainingOptions, _seed_for_device, _checkpoint_payload,
        build_loader, load_checkpoint, validate_resume_config,
        optimizer_steps_for_batches, restore_rng_state, train_one_epoch,
    )
    config = runtime_config(config, stop_after_epoch)
    paths = _stable._require_mapping(config, "paths")
    _stable._require_mapping(config, "data")
    training = _stable._require_mapping(config, "training")
    _stable._require_mapping(config, "sampling")
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
            denoiser.load_state_dict(
                cast(Mapping[str, Tensor], resume_payload["model_state"]),
                strict=True,
            )
            ema_state = resume_payload.get("ema_state")
            if not isinstance(ema_state, Mapping):
                raise ValueError("resume checkpoint lacks EMA state")
            ema.load_state_dict(ema_state)
            optimizer_state = resume_payload.get("optimizer_state")
            if not isinstance(optimizer_state, Mapping):
                raise ValueError("resume checkpoint lacks optimizer state")
            optimizer.load_state_dict(optimizer_state)
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

        if stop_after_epoch <= start_epoch:
            raise ValueError("stop-after-epoch must advance beyond the committed checkpoint")
        for epoch in range(start_epoch, stop_after_epoch):
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

            if _epoch_commit_callback is not None:
                _epoch_commit_callback(epoch + 1, history)

        result = {
            "schema_version": _stable.STATUS_SCHEMA,
            "status": "BOUNDARY_SAVED",
            "study_complete": False,
            "start_epoch": start_epoch + 1,
            "end_epoch": stop_after_epoch,
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



def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--expected-config-sha256", required=True)
    parser.add_argument("--resume", type=Path, help="This run's committed checkpoint_latest.pt")
    parser.add_argument("--stop-after-epoch", type=int, required=True,
                        help="Stop cleanly after this 1-based GLOBAL epoch; may exceed16")
    args = parser.parse_args(argv)
    if file_sha(args.config) != args.expected_config_sha256:
        raise ValueError("Configuration seal differs")
    raw = json.loads(args.config.read_text())
    if raw["data"].get("profile_worst_k", False):
        raise ValueError("Round-robin continuation requires full all-cell training")
    runtime_config(raw, args.stop_after_epoch)
    out = Path(raw["paths"]["output_dir"]).resolve()
    if out.exists() and args.resume is None:
        raise FileExistsError(f"No overwrite or implicit resume: {out}")
    if args.resume is not None and (args.resume.resolve() != out / "checkpoint_latest.pt" or not args.resume.is_file()):
        raise ValueError("Resume must use the existing run's latest checkpoint")
    data = raw["data"]
    audit_path = Path(data["coverage_audit_summary"])
    if file_sha(audit_path) != data["coverage_audit_sha256"]:
        raise ValueError("Coverage audit changed")
    audit = json.loads(audit_path.read_text())
    if audit.get("status") != "PASS" or audit["train_universe"]["target_exposures"] != 193644000:
        raise ValueError("All-cell coverage audit did not pass")
    if data.get("priming_policy") != POLICY:
        raise ValueError("Unsupported priming policy")
    topup_path = Path(data["topup_manifest"])
    if file_sha(topup_path) != data["topup_manifest_sha256"]:
        raise ValueError("Top-up manifest changed")
    topup = json.loads(topup_path.read_text())
    from .portable import resolve_artifact
    for field in ("cell_ids", "recipes"):
        topup[field + "_path"] = str(resolve_artifact(topup[field + "_path"], raw))
    if (topup.get("status") != "PASS" or topup.get("schema_version") != "pbisc_sdd.allcell_topup.v1"
            or topup["coverage_summary_sha256"] != data["coverage_audit_sha256"]):
        raise ValueError("Top-up coverage contract differs")
    for field in ("cell_ids", "recipes"):
        if file_sha(Path(topup[field + "_path"])) != topup[field + "_sha256"]:
            raise ValueError(f"Top-up artifact changed: {field}")
    offsets_path = topup_path.parent / "shard_offsets.csv"
    if file_sha(offsets_path) != topup["artifacts_sha256"]["shard_offsets.csv"]:
        raise ValueError("Top-up source coordinate map changed")
    from pbisc_sdd.experiment import require_single_process
    require_single_process()
    if str(raw["training"].get("device", "auto")) != "cpu":
        managed_gate()
    import torch
    from torch.utils.data import get_worker_info
    from pbisc_sdd import experiment as stable, ladder_experiment as ladder
    from pbisc_sdd.data import PBISCSparseCellDataset
    import pandas as pd

    config = runtime_config(ladder.load_experiment_config(args.config), args.stop_after_epoch)
    training = config["training"]
    resume_epoch = 0
    if args.resume is not None:
        resume_metadata = ladder.load_checkpoint(args.resume, device=torch.device("cpu"))
        ladder.validate_resume_config(config, resume_metadata["experiment_config"])
        resume_epoch = int(resume_metadata["epoch"]) + 1
        resume_metadata.clear()
        if resume_epoch >= args.stop_after_epoch:
            raise ValueError("stop-after-epoch must advance beyond the committed checkpoint")
        recovery = out / "recovery_records" / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        recovery.mkdir(parents=True, exist_ok=False)
        for name in ("RUN_STATUS.json", "ALLCELL_DATA_CONTRACT.json", "ROUNDROBIN_DATA_CONTRACT.json", "ROUNDROBIN_SEGMENT.json"):
            path = out / name
            if path.exists():
                (recovery / name).write_bytes(path.read_bytes())
        existing_observations = out / "data_observation"
        for path in existing_observations.glob("*.json"):
            row = json.loads(path.read_text())
            if int(row.get("epoch", 0)) > resume_epoch or path.name.startswith("progress_"):
                path.rename(recovery / path.name)
    prefetch_factor = validate_stream_contract(training, data)
    if (int(raw["training"]["epochs"]) != 16 or int(training["max_train_steps"]) != 0
            or int(data["train_recipes"]) != 8192):
        raise ValueError("The sealed source config must define the complete16-slot schedule")
    proof: dict[str, Any] = {}
    observations = out / "data_observation"
    topup_cells = np.load(topup["cell_ids_path"], mmap_mode="r", allow_pickle=False)
    topup_recipes = pd.read_csv(topup["recipes_path"])
    if len(topup_cells) != int(topup["added_target_cells"]) or len(topup_recipes) != int(topup["topup_recipes"]):
        raise ValueError("Top-up cardinality differs")

    def explicit_block(module, recipe, rng):
        return load_topup_block(module, recipe, rng, topup_cells,
                                proof["shard_starts"], proof["global_pools"])

    class AllCellsDataset(PBISCSparseCellDataset):
        def recipes_for_epoch(self, epoch: int):
            return recipes_for_global_epoch(self.data_module, split=self.split,
                seed=self.seed, platform=self.platform, topups=topup_recipes, global_epoch=epoch)

        def __len__(self):
            with self._shared_epoch.get_lock():
                epoch = int(self._shared_epoch.value)
            return int(self.recipes_for_epoch(epoch)["k"].sum())

        def __iter__(self):
            worker = get_worker_info()
            worker_id = 0 if worker is None else int(worker.id)
            workers = 1 if worker is None else int(worker.num_workers)
            with self._shared_epoch.get_lock():
                epoch = int(self._shared_epoch.value)
            recipes = self.recipes_for_epoch(epoch).iloc[worker_id::workers].reset_index(drop=True)
            self.data_module.prime_pools(recipes[~recipes["is_topup"].astype(bool)])
            rng = np.random.default_rng(worker_seed(self.seed, epoch, worker_id))
            targets = recipe_count = 0
            membership = hashlib.sha256()
            recipe_order = hashlib.sha256()
            for _, recipe in recipes.iterrows():
                k = int(recipe["k"])
                shadow = np.random.default_rng()
                shadow.bit_generator.state = copy.deepcopy(rng.bit_generator.state)
                chosen = shadow.choice(k, size=k, replace=False)
                if bool(recipe["is_topup"]):
                    block, ids = explicit_block(self.data_module, recipe, rng)
                else:
                    refs, pools = self.data_module.base._recipe_refs_and_pools(recipe)
                    block = self.data_module.load_training_block(recipe, 4000, rng)
                    ids = np.asarray([proof["shard_offsets"][str(pools[segment]["shard_id"])] + int(row)
                                      for segment, row in refs], dtype="<i8")[chosen]
                if block.counts.shape[0] != k or rng.bit_generator.state != shadow.bit_generator.state:
                    raise RuntimeError("All-K membership or target permutation differs")
                rid = str(recipe["recipe_id"]).encode("ascii")
                membership.update(rid + b"\0" + ids.astype("<i8", copy=False).tobytes())
                recipe_order.update(rid + b"\n")
                condition = self.conditioner.transform(block.bulk_log1p, block.n_cells)
                for counts, availability in zip(block.counts, block.availability_mask, strict=True):
                    targets += 1
                    yield {"x_log": np.log1p(counts).astype(np.float32, copy=False),
                           "condition": np.asarray(condition, dtype=np.float32),
                           "availability_mask": availability.astype(np.float32, copy=False),
                           "bulk_measurement_fraction": block.bulk_measurement_fraction,
                           "block_index": np.int64(recipe["stream_recipe_index"]),
                           "block_id": block.block_id, "recipe_k": np.int64(block.n_cells)}
                recipe_count += 1
                if recipe_count % 32 == 0:
                    write_json(observations / f"progress_worker_{worker_id}.json",
                               {"epoch": epoch + 1, "recipes": recipe_count, "target_cells": targets})
            path = observations / f"epoch_{epoch+1:03d}_worker_{worker_id}.json"
            if path.exists():
                raise FileExistsError(f"Preserve existing worker proof: {path}")
            write_json(path, {"epoch": epoch + 1, "worker": worker_id, "recipes": recipe_count,
                             "target_cells": targets, "target_order_sha256": membership.hexdigest(),
                             "recipe_order_sha256": recipe_order.hexdigest(), "status": "EXHAUSTED"})

    def build_streams(*positional, **kwargs):
        streams = stable.build_streams_from_config(*positional, **kwargs)
        base = streams.data_module.base
        if base.corpus_fingerprint != audit["corpus_fingerprint"]:
            raise RuntimeError("Runtime source corpus differs from coverage audit")
        if base._pools_by_sample:
            raise RuntimeError("Global priming requires a fresh empty pool cache")
        train = base.recipes[base.recipes["split"].astype(str).eq("train")]
        keys = sorted(set(train["sample_key"].astype(str)))
        base.prime_pools(train)
        observed_pool_hash = pool_digest(base._pools_by_sample, keys)
        if observed_pool_hash != audit["pool_order_sha256"]:
            raise RuntimeError("Runtime train pool order differs from audited order")
        validation_recipes = base.split_manifest("evaluation", max_blocks=int(data["validation_recipes"]),
                                               seed=streams.validation.seed)
        if set(validation_recipes["sample_key"].astype(str)) & set(keys):
            raise RuntimeError("Train/validation sample pools overlap")
        base.prime_pools(validation_recipes)
        if pool_digest(base._pools_by_sample, keys) != observed_pool_hash:
            raise RuntimeError("Validation priming changed train pool membership")
        offset, offsets = 0, {}
        for shard_id in sorted(base._shards_by_id):
            offsets[str(shard_id)] = offset
            offset += int(base._shards_by_id[shard_id]["n_cells"])
        saved_offsets = pd.read_csv(offsets_path)
        expected_offsets = [{"shard_id": str(key), "global_offset": offsets[str(key)],
                             "n_cells": int(base._shards_by_id[key]["n_cells"])}
                            for key in sorted(base._shards_by_id)]
        if saved_offsets[["shard_id", "global_offset", "n_cells"]].to_dict("records") != expected_offsets:
            raise RuntimeError("Runtime source coordinates differ from explicit top-up map")
        proof["shard_offsets"] = offsets
        proof["shard_starts"] = np.asarray(list(offsets.values()), dtype=np.int64)
        proof["global_pools"] = [base._shards_by_id[key] for key in sorted(base._shards_by_id)]
        observations.mkdir(parents=True, exist_ok=args.resume is not None)
        target = AllCellsDataset(streams.data_module, split="train", recipes_per_epoch=8192,
                                cells_per_block=4000, conditioner=streams.conditioner,
                                seed=20260627, progress=False)
        schedules, expected_cells, expected_steps, worker_cells = [], [], [], []
        for epoch in range(SCHEDULE_SLOTS):
            recipes = target.recipes_for_epoch(epoch)
            counts = [int(recipes.iloc[w::4]["k"].sum()) for w in range(4)]
            worker_cells.append(counts)
            expected_cells.append(sum(counts))
            expected_steps.append(batch_count(counts, 256))
            schedules.append({"epoch": epoch+1, "recipe_ids": recipes["recipe_id"].astype(str).tolist(),
                              "rng_seeds_decimal": [str(int(v)) for v in recipes["rng_seed"]],
                              "target_cells_by_worker": counts})
        total_expected = 193644000 + int(topup["added_target_cells"])
        if (sum(expected_cells) != total_expected
                or len({rid for s in schedules for rid in s["recipe_ids"]}) != 131072 + int(topup["topup_recipes"])):
            raise RuntimeError("Full schedule coverage differs from all sealed recipes")
        pairing = {"source_corpus_fingerprint": base.corpus_fingerprint,
                   "source_recipe_sha256": base.build_metadata["recipe_parts_content_sha256"],
                   "source_pool_sha256": base.build_metadata["sample_pools_sha256"],
                   "pool_order_sha256": observed_pool_hash, "recipe_schedule_sha256": digest(schedules),
                   "coverage_audit_sha256": data["coverage_audit_sha256"], "seed": 20260627,
                   "topup_manifest_sha256": data["topup_manifest_sha256"],
                   "target_policy": "all-registered-K-plus-unseen-train-topup-v1",
                   "epochs": SCHEDULE_SLOTS, "recipes_per_epoch": 8192,
                   "registered_recipes_per_epoch": 8192,
                   "total_recipe_counts_by_epoch": [len(s["recipe_ids"]) for s in schedules],
                   "cells_per_block": 4000, "batch_size": 256, "gradient_accumulation_steps": 1,
                   "num_workers": 4, "prefetch_factor": prefetch_factor,
                   "prefetch_factor_note": "scheduling depth only; batch sequence independent of it",
                   "priming_policy": POLICY}
        proof.update(pairing=pairing, schedules=schedules, expected_target_cells_by_epoch=expected_cells,
                     expected_optimizer_steps_by_epoch=expected_steps, worker_cells=worker_cells)
        write_json(out / "ROUNDROBIN_DATA_CONTRACT.json", {"schema_version": "pbisc_sdd.allcell_data_contract.v1",
                   "status": "RUNNING", "pairing": pairing,
                   "expected_target_cells_by_epoch": expected_cells,
                   "expected_optimizer_steps_by_epoch": expected_steps,
                   "original_recipe_coverage": audit["train_universe"],
                   "planned_coverage_after_topup": topup["summary"]})
        schedule_path = out / "EPOCH_SCHEDULE.json"
        if schedule_path.exists():
            if json.loads(schedule_path.read_text()) != schedules:
                raise ValueError("Resume epoch schedule differs")
        else:
            write_json(schedule_path, schedules)
        return replace(streams, train=target)

    commits: list[dict[str, Any]] = []

    def commit_epoch(epoch: int, history: list[dict[str, Any]]) -> None:
        commits.append(verify_epoch_commit(out, epoch, history, proof["schedules"],
                                          args.expected_config_sha256))

    result = run_training(config, stop_after_epoch=args.stop_after_epoch,
                          resume_checkpoint=args.resume, _stream_builder=build_streams,
                          _epoch_commit_callback=commit_epoch)
    if int(result["epochs_completed"]) != args.stop_after_epoch:
        raise RuntimeError("Training did not reach the requested global boundary")
    receipt = {
        **result, "schema_version": ROUNDROBIN_CONTRACT, "status": "SEGMENT_COMPLETED",
        "study_complete": False, "completed_schedule_cycles": args.stop_after_epoch // SCHEDULE_SLOTS,
        "config_sha256": args.expected_config_sha256,
        "runtime_training_epochs": int(config["training"]["epochs"]),
        "trainer_sha256": file_sha(Path(__file__)), "epoch_commits": commits,
        "checkpoint_latest": _binding(out / "checkpoint_latest.pt"),
        "checkpoint_epoch": _binding(out / f"checkpoint_epoch_{args.stop_after_epoch:03d}.pt"),
        "history_sha256": file_sha(out / "history.json"), "validation_protocol": VALIDATION_PROTOCOL,
    }
    write_json(out / "ROUNDROBIN_DATA_CONTRACT.json", {
        "schema_version": ROUNDROBIN_CONTRACT, "status": "SEGMENT_COMPLETED",
        "study_complete": False, "pairing": proof["pairing"],
        "epochs_completed": args.stop_after_epoch,
        "completed_schedule_cycles": args.stop_after_epoch // SCHEDULE_SLOTS,
        "expected_target_cells_by_epoch": proof["expected_target_cells_by_epoch"],
        "expected_optimizer_steps_by_epoch": proof["expected_optimizer_steps_by_epoch"],
        "schedule_rule": "zero-based global epoch modulo16; fixed recipe and target order; continuous training RNG",
    })
    write_json(out / "ROUNDROBIN_SEGMENT.json", receipt)
    write_json(out / "RUN_STATUS.json", receipt)
    print(json.dumps(receipt), flush=True)


if __name__ == "__main__":
    main()
