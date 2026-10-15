from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import random
from types import SimpleNamespace
from typing import Any, Mapping, cast

import numpy as np
import pandas as pd
import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from pbisc_sdd.data import SparseValueScaler
from pbisc_sdd.ema import ExponentialMovingAverage
from pbisc_sdd.experiment import load_checkpoint as load_stable_checkpoint
from pbisc_sdd.ladder_experiment import (
    CHECKPOINT_SCHEMA,
    LADDER_TRAINING_CONTRACT,
    STABLE_CHECKPOINT_SCHEMA,
    build_loader,
    load_checkpoint,
    load_experiment_config,
    optimizer_steps_for_batches,
    restore_rng_state,
    run_training,
    train_one_epoch,
    validate_resume_config,
)
from pbisc_sdd.model import DenoiserOutput


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _sample(x_log: float) -> dict[str, object]:
    return {
        "x_log": torch.tensor([x_log], dtype=torch.float32),
        "condition": torch.tensor([0.0], dtype=torch.float32),
        "availability_mask": torch.tensor([1], dtype=torch.uint8),
        "bulk_measurement_fraction": torch.tensor([1.0], dtype=torch.float32),
        "block_index": torch.tensor(0),
        "block_id": "block-a",
        "recipe_k": torch.tensor(1),
    }


class _QuadraticDiffusion(nn.Module):
    """Small deterministic stand-in that exercises the trainer contract."""

    def __init__(self) -> None:
        super().__init__()
        self.denoiser = nn.Linear(1, 1, bias=False)
        self.denoiser.weight.data.fill_(1.0)
        self.config = SimpleNamespace(
            value_loss_mode="all_measured",
            value_loss_weight=1.0,
            support_loss_weight=0.0,
        )

    def training_loss(
        self,
        values: torch.Tensor,
        condition: torch.Tensor,
        *,
        availability_mask: torch.Tensor,
        active_target: torch.Tensor,
        generator: torch.Generator,
    ) -> SimpleNamespace:
        del condition, availability_mask, active_target, generator
        prediction = self.denoiser(values)
        value_loss = prediction.square().mean()
        return SimpleNamespace(
            loss=value_loss,
            value_loss=value_loss,
            support_loss=value_loss * 0.0,
            prediction=DenoiserOutput(
                value_prediction=prediction,
                active_logits=torch.zeros_like(prediction),
            ),
        )


def test_accumulation_steps_optimizer_and_ema_and_normalizes_tail() -> None:
    diffusion = _QuadraticDiffusion()
    optimizer = torch.optim.SGD(diffusion.parameters(), lr=0.1)
    ema = ExponentialMovingAverage(diffusion.denoiser, decay=0.9)
    heartbeats: list[int] = []
    metrics = train_one_epoch(
        diffusion,  # type: ignore[arg-type]
        DataLoader([_sample(1.0), _sample(2.0), _sample(3.0)], batch_size=1),
        optimizer,
        device=torch.device("cpu"),
        scaler=SparseValueScaler(max_log_value=3.0),
        grad_clip=100.0,
        max_steps=0,
        generator=torch.Generator().manual_seed(7),
        gradient_accumulation_steps=2,
        heartbeat_every_steps=1,
        heartbeat_callback=lambda step, loss, norm: heartbeats.append(step),
        ema=ema,
    )

    # First update averages x={1/3, 2/3}; the incomplete tail uses x=1 with
    # divisor one. Dividing that tail by the configured two would yield 0.85.
    assert float(diffusion.denoiser.weight.item()) == pytest.approx(34.0 / 45.0)
    assert metrics["batches"] == 3
    assert metrics["optimizer_steps"] == 2
    assert ema.num_updates == 2
    assert heartbeats == [1, 2]


def test_max_steps_caps_optimizer_updates_not_microbatches() -> None:
    diffusion = _QuadraticDiffusion()
    metrics = train_one_epoch(
        diffusion,  # type: ignore[arg-type]
        DataLoader([_sample(float(index)) for index in range(1, 6)], batch_size=1),
        torch.optim.SGD(diffusion.parameters(), lr=0.01),
        device=torch.device("cpu"),
        scaler=SparseValueScaler(max_log_value=5.0),
        grad_clip=100.0,
        max_steps=1,
        generator=torch.Generator().manual_seed(9),
        gradient_accumulation_steps=2,
    )

    assert metrics["batches"] == 2
    assert metrics["optimizer_steps"] == 1
    assert optimizer_steps_for_batches(
        5, gradient_accumulation_steps=2, max_steps=2
    ) == 2


@pytest.mark.parametrize("bad_value", [0, -1, 1.5, True, float("inf")])
def test_config_rejects_invalid_gradient_accumulation(
    tmp_path: Path, bad_value: object
) -> None:
    payload = json.loads((PROJECT_ROOT / "configs" / "training" / "synthetic-contract.json").read_text())
    payload["training"]["gradient_accumulation_steps"] = bad_value
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="gradient_accumulation_steps"):
        load_experiment_config(path)


def test_prefetch_factor_requires_workers_and_is_forwarded(tmp_path: Path) -> None:
    payload = json.loads((PROJECT_ROOT / "configs" / "training" / "synthetic-contract.json").read_text())
    payload["training"]["prefetch_factor"] = 3
    path = tmp_path / "bad-prefetch.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="only when.*num_workers > 0"):
        load_experiment_config(path)

    training = {
        "batch_size": 1,
        "num_workers": 1,
        "pin_memory": False,
        "prefetch_factor": 3,
    }
    loader = build_loader([torch.tensor(1.0)], training)
    assert loader.prefetch_factor == 3
    assert loader.persistent_workers is True

    single_process = build_loader(
        [torch.tensor(1.0)], {"batch_size": 1, "num_workers": 0}
    )
    assert single_process.prefetch_factor is None
    assert single_process.persistent_workers is False


def test_resume_accepts_explicit_default_but_rejects_cadence_change() -> None:
    checkpoint = json.loads(
        (PROJECT_ROOT / "configs" / "training" / "synthetic-contract.json").read_text(encoding="utf-8")
    )
    current = deepcopy(checkpoint)
    current["training"]["gradient_accumulation_steps"] = 1
    current["training"]["epochs"] += 1
    current["sampling"]["steps"] += 1
    validate_resume_config(current, checkpoint)

    current["training"]["gradient_accumulation_steps"] = 2
    with pytest.raises(ValueError, match="resume configuration differs"):
        validate_resume_config(current, checkpoint)


def test_resume_treats_prefetch_depth_as_scheduling_but_keeps_worker_sharding() -> None:
    checkpoint = json.loads(
        (PROJECT_ROOT / "configs" / "training" / "synthetic-contract.json").read_text(encoding="utf-8")
    )
    checkpoint["training"]["num_workers"] = 4
    checkpoint["training"]["prefetch_factor"] = 1

    # Prefetch depth only changes how far workers run ahead; the collated
    # batch sequence is identical, so resume must accept any depth change
    # (including dropping back to the loader default).
    current = deepcopy(checkpoint)
    current["training"]["prefetch_factor"] = 32
    validate_resume_config(current, checkpoint)
    del current["training"]["prefetch_factor"]
    validate_resume_config(current, checkpoint)

    # Worker count changes the per-worker recipe shards and therefore the
    # batch stream: it stays fail-closed.
    current = deepcopy(checkpoint)
    current["training"]["num_workers"] = 8
    with pytest.raises(ValueError, match="resume configuration differs"):
        validate_resume_config(current, checkpoint)


def test_checkpoint_load_is_cpu_staged_and_memory_mapped(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "checkpoint.pt"
    torch.save(
        {
            "schema_version": CHECKPOINT_SCHEMA,
            "training_contract": LADDER_TRAINING_CONTRACT,
            "model_state": {"weight": torch.ones(2)},
            "optimizer_state": {"state": {0: {"moment": torch.ones(4)}}},
        },
        path,
    )
    original_load = torch.load
    observed: list[dict[str, object]] = []

    def recording_load(*args: object, **kwargs: object) -> object:
        observed.append(dict(kwargs))
        return original_load(*args, **kwargs)

    monkeypatch.setattr(torch, "load", recording_load)
    payload = load_checkpoint(path, device=torch.device("cuda"))

    assert observed[0]["map_location"] == "cpu"
    assert observed[0]["mmap"] is True
    assert payload["model_state"]["weight"].device.type == "cpu"  # type: ignore[index]
    optimizer_state = payload["optimizer_state"]  # type: ignore[assignment]
    assert optimizer_state["state"][0]["moment"].device.type == "cpu"  # type: ignore[index]


def test_ladder_checkpoint_requires_root_training_contract(tmp_path: Path) -> None:
    path = tmp_path / "unbound-ladder.pt"
    torch.save({"schema_version": CHECKPOINT_SCHEMA}, path)

    with pytest.raises(ValueError, match="training contract"):
        load_checkpoint(path, device=torch.device("cpu"))


def test_rng_restore_replays_every_cpu_rng_stream(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    random.seed(101)
    np.random.seed(102)
    torch.manual_seed(103)
    generator = torch.Generator().manual_seed(104)
    payload: dict[str, object] = {
        "torch_rng_state": torch.get_rng_state(),
        "cuda_rng_state_all": [],
        "training_generator_state": generator.get_state(),
        "python_random_state": random.getstate(),
        "numpy_random_state": np.random.get_state(),
    }
    expected = (
        random.random(),
        float(np.random.random()),
        float(torch.rand(())),
        float(torch.rand((), generator=generator)),
    )

    random.seed(201)
    np.random.seed(202)
    torch.manual_seed(203)
    generator.manual_seed(204)
    restore_rng_state(payload, generator)
    observed = (
        random.random(),
        float(np.random.random()),
        float(torch.rand(())),
        float(torch.rand((), generator=generator)),
    )

    assert observed == expected


class _EpochDataset(Dataset[dict[str, object]]):
    def __init__(self, values: list[float]) -> None:
        self.values = values
        self.epochs: list[int] = []

    def __len__(self) -> int:
        return len(self.values)

    def __getitem__(self, index: int) -> dict[str, object]:
        return _sample(self.values[index])

    def set_epoch(self, epoch: int) -> None:
        self.epochs.append(int(epoch))


class _TinyConditioner:
    dim = 1

    def state_dict(self) -> dict[str, object]:
        return {"schema_version": "test.conditioner.v1"}


def _tiny_streams(config: Mapping[str, object], **kwargs: object) -> Any:
    del config, kwargs
    panel = pd.DataFrame({"gene_id": ["g0"], "gene_symbol": ["G0"]})
    data_module = SimpleNamespace(
        gene_panel=panel,
        panel_positions=np.asarray([0], dtype=np.int64),
        measurement_coverage=np.asarray([1.0], dtype=np.float32),
        full_measurement_coverage=np.asarray([1.0], dtype=np.float32),
        corpus_fingerprint="f" * 64,
    )
    return SimpleNamespace(
        train=_EpochDataset([0.25, 0.5, 0.75]),
        validation=_EpochDataset([0.5]),
        data_module=data_module,
        conditioner=_TinyConditioner(),
    )


def _run_config(output_dir: Path, *, epochs: int) -> dict[str, object]:
    return {
        "schema_version": "pbisc_sdd.train.v1",
        "experiment_id": "ladder-loop-test",
        "paths": {
            "model7_root": "/unused/model7",
            "conditioner_cache": "/unused/conditioner.pt",
            "output_dir": str(output_dir),
        },
        "data": {
            "n_genes": 1,
            "gene_selection": "coverage_with_immune_markers",
            "train_recipes": 1,
            "validation_recipes": 1,
            "test_recipes": 1,
            "train_cells_per_block": 1,
            "validation_cells_per_block": 1,
            "test_cells_per_block": 1,
            "include_availability_in_condition": True,
            "include_bulk_measurement_in_condition": True,
        },
        "model": {
            "hidden_dims": [4],
            "time_embedding_dim": 4,
            "context_dim": 4,
            "dropout": 0.0,
        },
        "diffusion": {
            "max_log1p_value": 1.0,
            "center_values": True,
            "clip_denoised": True,
            "support_scale": 1.0,
            "cosine_offset": 0.008,
            "log_snr_clip": 20.0,
            "value_loss_mode": "all_measured",
            "value_loss_weight": 1.0,
            "support_loss_weight": 1.0,
            "self_condition_probability": 0.0,
        },
        "training": {
            "epochs": epochs,
            "batch_size": 1,
            "gradient_accumulation_steps": 2,
            "max_train_steps": 0,
            "max_validation_steps": 1,
            "lr": 0.001,
            "weight_decay": 0.0,
            "grad_clip": 5.0,
            "num_workers": 0,
            "pin_memory": False,
            "device": "cpu",
            "amp": "none",
            "seed": 1234,
            "heartbeat_every_steps": 1,
            "ema_decay": 0.9,
            "ema_update_every": 1,
            "min_system_memory_available_gib": 0.0,
            "min_cuda_memory_available_gib": 0.0,
        },
        "sampling": {
            "steps": 1,
            "eta": 0.0,
            "active_logit_threshold": 0.0,
            "preview_cells": 1,
        },
    }


def _state_tensors(payload: dict[str, object], key: str) -> dict[str, torch.Tensor]:
    state = payload[key]
    assert isinstance(state, dict)
    return cast(dict[str, torch.Tensor], state)


def test_run_training_counts_optimizer_steps_and_exactly_resumes(
    tmp_path: Path,
) -> None:
    uninterrupted_dir = tmp_path / "uninterrupted"
    resumed_dir = tmp_path / "resumed"
    uninterrupted = run_training(
        _run_config(uninterrupted_dir, epochs=2),
        _stream_builder=_tiny_streams,
    )
    first = run_training(
        _run_config(resumed_dir, epochs=1),
        _stream_builder=_tiny_streams,
    )
    resumed = run_training(
        _run_config(resumed_dir, epochs=2),
        resume_checkpoint=resumed_dir / "checkpoint_latest.pt",
        _stream_builder=_tiny_streams,
    )

    assert first["global_step"] == 2
    assert uninterrupted["global_step"] == resumed["global_step"] == 4
    uninterrupted_checkpoint = load_checkpoint(
        uninterrupted_dir / "checkpoint_latest.pt", device=torch.device("cpu")
    )
    resumed_checkpoint = load_checkpoint(
        resumed_dir / "checkpoint_latest.pt", device=torch.device("cpu")
    )
    assert uninterrupted_checkpoint["schema_version"] == CHECKPOINT_SCHEMA
    assert resumed_checkpoint["schema_version"] == CHECKPOINT_SCHEMA
    assert (
        resumed_checkpoint["training_contract"] == LADDER_TRAINING_CONTRACT
    )
    assert resumed_checkpoint["global_step"] == 4
    ema_state = resumed_checkpoint["ema_state"]
    assert isinstance(ema_state, dict)
    assert ema_state["num_updates"] == 4
    history = resumed_checkpoint["history"]
    assert isinstance(history, list)
    assert [record["global_step"] for record in history] == [2, 4]
    assert [record["train"]["batches"] for record in history] == [3, 3]
    assert [record["train"]["optimizer_steps"] for record in history] == [2, 2]

    for state_key in ("model_state",):
        expected = _state_tensors(uninterrupted_checkpoint, state_key)
        observed = _state_tensors(resumed_checkpoint, state_key)
        assert expected.keys() == observed.keys()
        assert all(torch.equal(expected[name], observed[name]) for name in expected)
    uninterrupted_ema = uninterrupted_checkpoint["ema_state"]
    resumed_ema = resumed_checkpoint["ema_state"]
    assert isinstance(uninterrupted_ema, dict)
    assert isinstance(resumed_ema, dict)
    expected_ema = uninterrupted_ema["model_state"]
    observed_ema = resumed_ema["model_state"]
    assert isinstance(expected_ema, dict)
    assert isinstance(observed_ema, dict)
    assert all(
        torch.equal(expected_ema[name], observed_ema[name])
        for name in expected_ema
    )
    expected_optimizer = uninterrupted_checkpoint["optimizer_state"]
    observed_optimizer = resumed_checkpoint["optimizer_state"]
    assert expected_optimizer["param_groups"] == observed_optimizer["param_groups"]
    assert expected_optimizer["state"].keys() == observed_optimizer["state"].keys()
    for key, expected_state in expected_optimizer["state"].items():
        for name, expected_value in expected_state.items():
            actual = observed_optimizer["state"][key][name]
            assert torch.equal(expected_value, actual) if isinstance(expected_value, torch.Tensor) else expected_value == actual
    history_payload = json.loads((resumed_dir / "history.json").read_text())
    assert history_payload["training_contract"] == LADDER_TRAINING_CONTRACT
    assert [record["global_step"] for record in history_payload["epochs"]] == [2, 4]
    assert (resumed_dir / "sample_preview.npz").is_file()
    assert (resumed_dir / "checkpoint_epoch_002.pt").is_file()
    with pytest.raises(ValueError, match="unsupported PBISC-SDD checkpoint"):
        load_stable_checkpoint(
            resumed_dir / "checkpoint_latest.pt", device=torch.device("cpu")
        )


def test_stable_v5_migration_is_explicit_and_accumulation_safe(
    tmp_path: Path,
) -> None:
    path = tmp_path / "stable-v5.pt"
    payload: dict[str, object] = {
        "schema_version": STABLE_CHECKPOINT_SCHEMA,
        "experiment_config": {"training": {"num_workers": 0}},
        "global_step": 4,
        "ema_state": {"num_updates": 4},
    }
    torch.save(payload, path)

    with pytest.raises(ValueError, match="unsupported.*ladder checkpoint"):
        load_checkpoint(path, device=torch.device("cpu"))
    migrated = load_checkpoint(
        path,
        device=torch.device("cpu"),
        allow_stable_v5_migration=True,
    )
    assert migrated["schema_version"] == STABLE_CHECKPOINT_SCHEMA

    payload["experiment_config"] = {
        "training": {
            "num_workers": 0,
            "gradient_accumulation_steps": 2,
        }
    }
    torch.save(payload, path)
    with pytest.raises(ValueError, match="absent or one"):
        load_checkpoint(
            path,
            device=torch.device("cpu"),
            allow_stable_v5_migration=True,
        )
