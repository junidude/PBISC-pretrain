"""CPU-only state-continuity, epoch-boundary and 16-slot cycling regressions."""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping

import numpy as np
import pandas as pd
import pytest
import torch
from torch.utils.data import Dataset

from pbisc_sdd import experiment as stable, ladder_experiment as ladder
from pbisc.training import allcell_roundrobin as rr


@pytest.fixture(autouse=True)
def tiny_cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


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


def assert_nested_equal(left, right):
    if torch.is_tensor(left):
        assert torch.equal(left, right)
    elif isinstance(left, np.ndarray):
        np.testing.assert_array_equal(left, right)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            assert_nested_equal(left[key], right[key])
    elif isinstance(left, (list, tuple)):
        assert len(left) == len(right)
        for a, b in zip(left, right, strict=True):
            assert_nested_equal(a, b)
    else:
        assert left == right


def checkpoint(path):
    return ladder.load_checkpoint(path / "checkpoint_latest.pt", device=torch.device("cpu"))


def assert_training_state_equal(left, right):
    for key in ("model_state", "ema_state", "optimizer_state", "global_step", "epoch",
                "best_validation_loss", "training_generator_state", "torch_rng_state",
                "python_random_state", "numpy_random_state", "cuda_rng_state_all"):
        assert_nested_equal(left[key], right[key])
    for a, b in zip(left["history"], right["history"], strict=True):
        for key in ("epoch", "global_step", "train", "validation"):
            assert_nested_equal(a[key], b[key])


def test_exact_resume_and_clean_boundary_without_next_epoch_or_preview(tmp_path, monkeypatch):
    observed_streams = []
    def streams(config, **kwargs):
        value = _tiny_streams(config, **kwargs)
        observed_streams.append(value)
        return value
    def forbidden_preview(*args, **kwargs):
        raise AssertionError("An epoch boundary must not run a sampling preview")
    monkeypatch.setattr(stable, "generate_preview", forbidden_preview)
    full = tmp_path / "full"
    segmented = tmp_path / "segmented"
    rr.run_training(_run_config(full, epochs=16), stop_after_epoch=3, _stream_builder=streams)
    assert observed_streams[-1].train.epochs == [0, 1, 2]
    commits = []
    def commit(epoch, history):
        saved = checkpoint(segmented)
        assert saved["epoch"] == epoch - 1
        assert (segmented / f"checkpoint_epoch_{epoch:03d}.pt").is_file()
        assert json.loads((segmented / "history.json").read_text())["epochs"] == history
        commits.append(epoch)
    first = rr.run_training(_run_config(segmented, epochs=16), stop_after_epoch=1,
                           _stream_builder=streams, _epoch_commit_callback=commit)
    assert first["status"] == "BOUNDARY_SAVED" and first["study_complete"] is False
    assert observed_streams[-1].train.epochs == [0]
    assert commits == [1]
    assert not (segmented / "checkpoint_epoch_002.pt").exists()
    original_snapshot_hash = rr.file_sha(segmented / "checkpoint_epoch_001.pt")
    resumed = rr.run_training(_run_config(segmented, epochs=16), stop_after_epoch=3,
                             resume_checkpoint=segmented / "checkpoint_latest.pt",
                             _stream_builder=streams, _epoch_commit_callback=commit)
    assert resumed["start_epoch"] == 2 and resumed["end_epoch"] == 3
    assert observed_streams[-1].train.epochs == [1, 2]
    assert observed_streams[-1].validation.epochs == [0, 0]
    assert commits == [1, 2, 3]
    assert not (segmented / "checkpoint_epoch_004.pt").exists()
    assert rr.file_sha(segmented / "checkpoint_epoch_001.pt") == original_snapshot_hash
    assert_training_state_equal(checkpoint(full), checkpoint(segmented))
    assert checkpoint(segmented)["experiment_config"]["training"]["epochs"] == 16


def test_first_sixteen_match_sealed_trainer_and_resume_crosses_cycle(tmp_path):
    old = tmp_path / "sealed"
    segmented = tmp_path / "segmented"
    full = tmp_path / "full"
    ladder.run_training(_run_config(old, epochs=16), _stream_builder=_tiny_streams)
    boundary = rr.run_training(_run_config(segmented, epochs=16), stop_after_epoch=16,
                               _stream_builder=_tiny_streams)
    assert boundary["study_complete"] is False
    assert_training_state_equal(checkpoint(old), checkpoint(segmented))
    continued = rr.run_training(_run_config(segmented, epochs=16), stop_after_epoch=17,
                               resume_checkpoint=segmented / "checkpoint_latest.pt",
                               _stream_builder=_tiny_streams)
    rr.run_training(_run_config(full, epochs=16), stop_after_epoch=17, _stream_builder=_tiny_streams)
    assert continued["start_epoch"] == continued["end_epoch"] == 17
    assert continued["epochs_completed"] == 17 and continued["global_step"] == 34
    assert checkpoint(segmented)["experiment_config"]["training"]["epochs"] == 17
    assert_training_state_equal(checkpoint(full), checkpoint(segmented))


def test_nonadvancing_resume_fails_and_preserves_checkpoint(tmp_path):
    config = _run_config(tmp_path / "run", epochs=16)
    rr.run_training(config, stop_after_epoch=1, _stream_builder=_tiny_streams)
    path = Path(config["paths"]["output_dir"])
    before = rr.file_sha(path / "checkpoint_latest.pt")
    with pytest.raises(ValueError, match="must advance"):
        rr.run_training(config, stop_after_epoch=1, resume_checkpoint=path / "checkpoint_latest.pt",
                        _stream_builder=_tiny_streams)
    assert rr.file_sha(path / "checkpoint_latest.pt") == before
    assert not (path / "checkpoint_epoch_002.pt").exists()


def test_recipe_cycle_keeps_exact_first16_schedule_and_topups():
    seed = 20260627
    calls = []
    def split_manifest(split, max_blocks, seed, platform):
        calls.append(seed)
        slot = (seed - 20260627) // 1000003
        return pd.DataFrame({"recipe_id": [f"registered-{slot}"], "k": [2],
                             "rng_seed": np.asarray([2**64 - 1], dtype=np.uint64)})
    module = SimpleNamespace(split_manifest=split_manifest)
    topups = pd.DataFrame({"recipe_id": [f"extra-{i}" for i in range(16)],
                           "schedule_slot": list(range(16)), "k": [1] * 16})
    recipes = []
    for epoch in range(48):
        value = rr.recipes_for_global_epoch(module, split="train", seed=seed, platform=None,
                                            topups=topups, global_epoch=epoch)
        recipes.append(value)
        assert value.recipe_id.tolist() == [f"registered-{epoch % 16}", f"extra-{epoch % 16}"]
        assert value.rng_seed.tolist() == [2**64 - 1, 0]
    assert calls[:16] == [seed + 1000003 * i for i in range(16)]
    assert calls[:16] == calls[16:32] == calls[32:]
    for slot in range(16):
        pd.testing.assert_frame_equal(recipes[slot], recipes[slot + 16])


@pytest.mark.parametrize("value", [-1, 0, True, 1.5])
def test_invalid_stop_epoch_rejected(value):
    with pytest.raises(ValueError, match="positive"):
        rr.runtime_config({"training": {"epochs": 16}}, value)


def test_runtime_extension_does_not_modify_sealed_config():
    sealed = {"training": {"epochs": 16, "lr": 0.0001}, "data": {"train_recipes": 8192}}
    before = copy.deepcopy(sealed)
    for end in (3, 16, 17, 33):
        current = rr.runtime_config(sealed, end)
        assert current["training"]["epochs"] == max(16, end)
    assert sealed == before
    assert [rr.schedule_slot(e) for e in (0, 15, 16, 31, 32)] == [0, 15, 0, 15, 0]


def test_worker_target_order_repeats_entire_original16_slot_protocol():
    for epoch in range(16):
        for worker in range(4):
            seed = 20260627 + 1000003 * epoch + 7919 + 104729 * worker
            assert rr.worker_seed(20260627, epoch, worker) == seed
            original = np.random.default_rng(seed)
            repeated = np.random.default_rng(rr.worker_seed(20260627, epoch + 16, worker))
            for k in (250, 17, 4000):
                np.testing.assert_array_equal(original.choice(k, k, replace=False),
                                              repeated.choice(k, k, replace=False))


def commit_fixture(tmp_path, epoch=1):
    out = tmp_path / "run"
    observations = out / "data_observation"
    observations.mkdir(parents=True)
    schedules = [{"recipe_ids": [f"r{slot}-{w}" for w in range(4)],
                  "target_cells_by_worker": [1, 2, 3, 4]} for slot in range(16)]
    for worker, count in enumerate((1, 2, 3, 4)):
        rid = schedules[(epoch - 1) % 16]["recipe_ids"][worker]
        rr.write_json(observations / f"epoch_{epoch:03d}_worker_{worker}.json", {
            "status": "EXHAUSTED", "epoch": epoch, "worker": worker, "target_cells": count,
            "recipes": 1, "target_order_sha256": "a" * 64,
            "recipe_order_sha256": hashlib.sha256((rid + "\n").encode()).hexdigest()})
    history = [{"epoch": n, "global_step": 4 * n, "train": {"optimizer_steps": 4, "loss": 1.0},
                "validation": {"loss": 1.0}} for n in range(1, epoch + 1)]
    rr.write_json(out / "history.json", {"epochs": history})
    (out / "checkpoint_latest.pt").write_bytes(b"tiny checkpoint latest")
    (out / f"checkpoint_epoch_{epoch:03d}.pt").write_bytes(b"tiny checkpoint epoch")
    return out, history, schedules


def test_commit_only_after_worker_evidence_and_all_durable_hashes(tmp_path):
    out, history, schedules = commit_fixture(tmp_path, epoch=17)
    receipt = rr.verify_epoch_commit(out, 17, history, schedules, "b" * 64)
    saved = json.loads(Path(receipt["path"]).read_text())
    assert saved["schedule_slot"] == 0 and saved["schedule_cycle"] == 2
    assert saved["study_complete"] is False and saved["global_step"] == 68
    assert len(saved["worker_observations"]) == 4
    for binding in [saved["checkpoint_latest"], saved["checkpoint_epoch"], *saved["worker_observations"], receipt]:
        assert binding["sha256"] == rr.file_sha(Path(binding["path"]))
    assert rr.verify_epoch_commit(out, 17, history, schedules, "b" * 64) == receipt


@pytest.mark.parametrize("damage", ["missing_worker", "counts", "recipe_order", "steps", "nonfinite", "history", "checkpoint"])
def test_bad_epoch_evidence_cannot_create_commit(tmp_path, damage):
    out, history, schedules = commit_fixture(tmp_path)
    path = out / "data_observation/epoch_001_worker_0.json"
    if damage == "missing_worker":
        path.unlink()
    elif damage in ("counts", "recipe_order"):
        row = json.loads(path.read_text())
        row["target_cells" if damage == "counts" else "recipe_order_sha256"] = 99
        rr.write_json(path, row)
    elif damage == "steps":
        history[0]["train"]["optimizer_steps"] = 3
    elif damage == "nonfinite":
        history[0]["validation"]["loss"] = float("nan")
    elif damage == "history":
        rr.write_json(out / "history.json", {"epochs": []})
    else:
        (out / "checkpoint_epoch_001.pt").unlink()
    with pytest.raises((ValueError, FileNotFoundError)):
        rr.verify_epoch_commit(out, 1, history, schedules, "b" * 64)
    assert not (out / "epoch_commits/epoch_001.json").exists()


@pytest.mark.parametrize("name", ["6k-x1", "6k-trunk4x", "7k-trunk4x", "8k-trunk4x"])
def test_bounded_main_accepts_actual_production_prefetch32_before_loading_counts(tmp_path, monkeypatch, name):
    root = Path(__file__).resolve().parents[1]
    config = json.loads((root / "configs/original" / f"{name}.json").read_text())
    assert config["training"]["prefetch_factor"] == 32
    config["paths"]["output_dir"] = str(tmp_path / "fresh-run")
    config["training"]["device"] = "cpu"
    audit_path = tmp_path / "audit.json"
    audit_path.write_text(json.dumps({"status": "PASS", "train_universe": {"target_exposures": 193644000}}))
    config["data"]["coverage_audit_summary"] = str(audit_path)
    config["data"]["coverage_audit_sha256"] = rr.file_sha(audit_path)
    ids, recipes, offsets = tmp_path / "ids.npy", tmp_path / "recipes.csv", tmp_path / "shard_offsets.csv"
    np.save(ids, np.asarray([], dtype=np.int64))
    recipes.write_text("schedule_slot,k\n")
    offsets.write_text("shard_id,global_offset,n_cells\n")
    topup_path = tmp_path / "manifest.json"
    topup_path.write_text(json.dumps({
        "schema_version": "pbisc_sdd.allcell_topup.v1", "status": "PASS",
        "coverage_summary_sha256": config["data"]["coverage_audit_sha256"],
        "cell_ids_path": str(ids), "cell_ids_sha256": rr.file_sha(ids),
        "recipes_path": str(recipes), "recipes_sha256": rr.file_sha(recipes),
        "artifacts_sha256": {"shard_offsets.csv": rr.file_sha(offsets)},
    }))
    config["data"]["topup_manifest"] = str(topup_path)
    config["data"]["topup_manifest_sha256"] = rr.file_sha(topup_path)
    config_path = tmp_path / "input.json"
    config_path.write_text(json.dumps(config))
    monkeypatch.setattr(ladder, "load_experiment_config", lambda path: config)
    called = []
    original_gate = rr.validate_stream_contract
    def record_gate(training, data):
        result = original_gate(training, data)
        called.append(result)
        return result
    monkeypatch.setattr(rr, "validate_stream_contract", record_gate)
    class ReachedCountLoading(Exception):
        pass
    def stop_at_count_loading(*args, **kwargs):
        raise ReachedCountLoading
    monkeypatch.setattr(rr.np, "load", stop_at_count_loading)
    with pytest.raises(ReachedCountLoading):
        rr.main(["--config", str(config_path), "--expected-config-sha256", rr.file_sha(config_path),
                 "--stop-after-epoch", "1"])
    assert called == [32]
