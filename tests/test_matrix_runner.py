from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import subprocess
import sys

import pytest

from pbisc_pretrain import matrix


def _write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


@pytest.fixture
def plan(tmp_path, monkeypatch):
    assets, data, output = tmp_path / "assets", tmp_path / "data", tmp_path / "runs"
    base = assets / "configs/base/allcell-v2.json"
    _write(base, {"training": {"epochs": 16}})
    _write(data / "relocation.json", {"schema_version": "fixture"})
    inputs = {}
    for field in ("gene_positions_path", "gene_selection_manifest", "coverage_audit_summary", "topup_manifest"):
        path = data / f"{field}.json"
        _write(path, {"input": field})
        inputs[field] = str(path)
        hash_key = {"gene_positions_path": "gene_positions_sha256",
                    "gene_selection_manifest": "gene_selection_manifest_sha256",
                    "coverage_audit_summary": "coverage_audit_sha256", "topup_manifest": "topup_manifest_sha256"}[field]
        inputs[hash_key] = matrix._sha(path)
    rows = []
    for n in (7000, 8000, 10000):
        for name, trunk, context in matrix._VARIANTS:
            identifier = f"{n // 1000}k-{name}"
            path = assets / f"configs/matrix/{identifier}.json"
            _write(path, {"schema_version": "pbisc.pretrain.profile.v2",
                          "base_config": "../base/allcell-v2.json", "n_genes": n})
            rows.append({"id": identifier, "profile": str(path.relative_to(assets)),
                         "profile_sha256": matrix._sha(path), "n_genes": n,
                         "trunk_multiplier": trunk, "context_multiplier": context})
    source = assets / "configs/matrix.json"
    _write(source, {"schema_version": matrix.SCHEMA, "experiments": rows})
    monkeypatch.setattr(matrix, "asset_root", lambda: assets)

    def prepare(path, *, dataset_root, output_dir):
        profile = json.loads(Path(path).read_text())
        variant = Path(path).stem.split("-", 1)[1]
        _, trunk, context = next(row for row in matrix._VARIANTS if row[0] == variant)
        model = {"hidden_dims": [width * trunk for width in (1024, 1024, 512, 256)],
                 "context_dim": 512 * context, "time_embedding_dim": 128, **profile.get("model", {})}
        config = {"paths": {"output_dir": str(output_dir)}, "training": {"epochs": 16},
                  "model": model,
                  "data": {**inputs, "n_genes": profile["n_genes"]},
                  "_release": {"dataset_root": str(dataset_root)}}
        return config, "allcell"

    import pbisc.training.portable
    monkeypatch.setattr(pbisc.training.portable, "prepare_config", prepare)
    return SimpleNamespace(assets=assets, data=data, output=output, rows=rows, source=source, base=base,
                           prepare=prepare, inputs=inputs)


def _args(plan, command="run", *extras):
    return [command, "--data-root", str(plan.data), "--output-root", str(plan.output), *extras]


def _receipt(output: Path, config_sha: str, inputs: dict, *, end=16, start=1):
    output.mkdir(parents=True, exist_ok=True)
    history = [{"epoch": epoch, "global_step": epoch * 4, "train": {"optimizer_steps": 4, "loss": 1.0},
                "validation": {"loss": 1.0}} for epoch in range(1, end + 1)]
    _write(output / "history.json", {"epochs": history})
    schedule = [{"epoch": slot + 1, "target_cells_by_worker": [256] * 4,
                 "recipe_ids": [str(w) for w in range(4)]} for slot in range(16)]
    _write(output / "EPOCH_SCHEDULE.json", schedule)
    for name in ("checkpoint_latest.pt", f"checkpoint_epoch_{end:03d}.pt"):
        (output / name).write_bytes(b"fixture checkpoint, never deserialized")
    latest = {"path": str(output / "checkpoint_latest.pt"), "sha256": matrix._sha(output / "checkpoint_latest.pt")}
    epoch_checkpoint = {"path": str(output / f"checkpoint_epoch_{end:03d}.pt"),
                        "sha256": matrix._sha(output / f"checkpoint_epoch_{end:03d}.pt")}
    commits = []
    trainer_sha256 = matrix._implementation_bindings()["pbisc.training.allcell_roundrobin"]
    for epoch in range(start, end + 1):
        proofs = []
        for worker in range(4):
            proof = {"status": "EXHAUSTED", "epoch": epoch, "worker": worker, "target_cells": 256,
                     "recipes": 1, "recipe_order_sha256": hashlib.sha256(f"{worker}\n".encode()).hexdigest(),
                     "target_order_sha256": "a" * 64}
            path = output / "data_observation" / f"epoch_{epoch:03d}_worker_{worker}.json"
            _write(path, proof)
            proofs.append({**proof, "path": str(path), "sha256": matrix._sha(path)})
        commit = {"schema_version": "pbisc_sdd.allcell_epoch_commit.v1", "status": "EPOCH_COMMITTED",
                  "epoch": epoch, "schedule_slot": (epoch - 1) % 16, "config_sha256": config_sha,
                  "trainer_sha256": trainer_sha256,
                  "expected_optimizer_steps": 4, "observed_optimizer_steps": 4,
                  "expected_target_cells": 1024, "observed_target_cells": 1024,
                  "global_step": epoch * 4, "worker_observations": proofs,
                  "checkpoint_latest": latest, "checkpoint_epoch": epoch_checkpoint,
                  "history_sha256": matrix._sha(output / "history.json")}
        path = output / "epoch_commits" / f"epoch_{epoch:03d}.json"
        _write(path, commit)
        commits.append({"path": str(path), "sha256": matrix._sha(path), "epoch": epoch})
    _write(output / "ROUNDROBIN_DATA_CONTRACT.json", {
        "schema_version": matrix.ROUNDROBIN_SCHEMA, "status": "SEGMENT_COMPLETED", "epochs_completed": end,
        "pairing": {"recipe_schedule_sha256": hashlib.sha256(json.dumps(schedule, sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
                    "coverage_audit_sha256": inputs["coverage_audit_sha256"],
                    "topup_manifest_sha256": inputs["topup_manifest_sha256"]},
        "expected_target_cells_by_epoch": [1024] * 16, "expected_optimizer_steps_by_epoch": [4] * 16})
    receipt = {"schema_version": matrix.ROUNDROBIN_SCHEMA, "status": "SEGMENT_COMPLETED", "study_complete": False,
               "start_epoch": start, "end_epoch": end, "epochs_completed": end, "global_step": end * 4,
               "n_genes": 10000 if output.name.startswith("10k") else int(output.name[0]) * 1000,
               "config_sha256": config_sha, "epoch_commits": commits, "checkpoint_latest": latest,
               "trainer_sha256": trainer_sha256,
               "checkpoint_epoch": epoch_checkpoint, "history_sha256": matrix._sha(output / "history.json")}
    _write(output / "ROUNDROBIN_SEGMENT.json", receipt)
    _write(output / "RUN_STATUS.json", receipt)


def _children(plan, monkeypatch, *, fail_id=None, fail_command="train", complete=True):
    calls = []

    def run(argv, *, check):
        assert check is False
        command = argv[3]
        profile = Path(argv[argv.index("--config") + 1])
        output = Path(argv[argv.index("--output-dir") + 1])
        calls.append((command, output.name, list(argv)))
        assert argv[:3] == [sys.executable, "-m", "pbisc_pretrain.cli"]
        if output.name == fail_id and command == fail_command:
            return SimpleNamespace(returncode=17)
        if command == "prepare":
            assert not output.exists(), "controller must leave run directory creation to trainer"
            config, _ = plan.prepare(profile, dataset_root=plan.data, output_dir=output)
            _write(matrix._materialized(output), config)
        elif complete:
            end = int(argv[argv.index("--stop-after-epoch") + 1])
            _receipt(output, matrix._sha(matrix._materialized(output)), plan.inputs, end=end)
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(matrix.subprocess, "run", run)
    return calls


def test_actual_manifest_has_15_unique_reviewed_axes():
    _, rows = matrix.load_matrix()
    assert len({r["id"] for r in rows}) == 15
    assert [(r["n_genes"], r["trunk_multiplier"], r["context_multiplier"]) for r in rows] == [
        (n, trunk, context) for n in (7000, 8000, 10000) for _, trunk, context in matrix._VARIANTS]


def test_matrix_help_and_list_are_torch_free():
    code = ("from pbisc_pretrain.matrix import main, _implementation_bindings; import sys; "
            "main(['list']); assert _implementation_bindings(); assert 'torch' not in sys.modules")
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_run_serial_order_and_verified_skip(plan, monkeypatch, capsys):
    calls = _children(plan, monkeypatch)
    args = _args(plan, "run", "--only", "7k-1x", "--only", "7k-trunk2x")
    assert matrix.main(args) == 0
    assert [(c, name) for c, name, _ in calls] == [("prepare", "7k-1x"), ("train", "7k-1x"),
                                                    ("prepare", "7k-trunk2x"), ("train", "7k-trunk2x")]
    assert all(argv[argv.index("--stop-after-epoch") + 1] == "16" for c, _, argv in calls if c == "train")
    calls.clear()
    assert matrix.main(args) == 0
    assert calls == []
    assert json.loads(capsys.readouterr().out.splitlines()[-1])["status"] == "MATRIX_COMPLETED"


def test_all_15_children_run_once_in_manifest_order(plan, monkeypatch):
    calls = _children(plan, monkeypatch)
    assert matrix.main(_args(plan)) == 0
    assert [(command, name) for command, name, _ in calls] == [
        (command, row["id"]) for row in plan.rows for command in ("prepare", "train")]


def test_entire_selection_is_checked_before_any_child(plan, monkeypatch):
    calls = _children(plan, monkeypatch)
    (plan.output / "7k-trunk2x").mkdir(parents=True)
    assert matrix.main(_args(plan)) == 2
    assert not calls


@pytest.mark.parametrize(("field", "wrong"), [("hidden_dims", [2048, 2048, 1024, 512]),
                                              ("context_dim", 1024), ("time_embedding_dim", 256)])
def test_resealed_profile_cannot_mislabel_runtime_architecture_axes(plan, monkeypatch, field, wrong):
    calls = _children(plan, monkeypatch)
    profile_path = plan.assets / plan.rows[0]["profile"]
    profile = json.loads(profile_path.read_text())
    profile["model"] = {field: wrong}
    _write(profile_path, profile)
    plan.rows[0]["profile_sha256"] = matrix._sha(profile_path)
    _write(plan.source, {"schema_version": matrix.SCHEMA, "experiments": plan.rows})
    assert matrix.main(_args(plan, "run", "--only", "7k-1x")) == 2
    assert not calls
    assert not (plan.output / ".matrix/7k-1x.json").exists()


def test_controller_lock_prevents_overlapping_operators(plan, monkeypatch):
    calls = _children(plan, monkeypatch)
    with matrix._locked(plan.output):
        assert matrix.main(_args(plan, "prepare", "--only", "7k-1x")) == 2
    assert not calls


@pytest.mark.parametrize("failed_command", ["prepare", "train"])
def test_child_failure_stops_immediately_without_retry(plan, monkeypatch, failed_command):
    calls = _children(plan, monkeypatch, fail_id="7k-1x", fail_command=failed_command)
    assert matrix.main(_args(plan)) == 17
    assert all(name == "7k-1x" for _, name, _ in calls)
    assert len(calls) == (1 if failed_command == "prepare" else 2)


def test_foreign_cwd_resolves_assets_and_user_roots_once(plan, monkeypatch):
    foreign = plan.output.parent / "foreign"
    foreign.mkdir()
    monkeypatch.chdir(foreign)
    calls = _children(plan, monkeypatch)
    assert matrix.main(["prepare", "--data-root", "../data", "--output-root", "local-runs", "--only", "7k-1x"]) == 0
    argv = calls[0][2]
    assert argv[argv.index("--data-root") + 1] == str(plan.data)
    assert argv[argv.index("--output-dir") + 1] == str(foreign / "local-runs/7k-1x")
    assert not (foreign / "local-runs/7k-1x").exists()


def test_unbound_run_and_orphan_config_cannot_be_adopted(plan, monkeypatch):
    calls = _children(plan, monkeypatch)
    output = plan.output / "7k-1x"
    output.mkdir(parents=True)
    assert matrix.main(_args(plan, "run", "--only", "7k-1x", "--resume-existing")) == 2
    output.rmdir()
    _write(matrix._materialized(output), {})
    assert matrix.main(_args(plan, "run", "--only", "7k-1x", "--resume-existing")) == 2
    assert not calls


def test_partial_requires_explicit_same_run_resume(plan, monkeypatch):
    calls = _children(plan, monkeypatch)
    only = ["--only", "7k-1x"]
    assert matrix.main(_args(plan, "prepare", *only)) == 0
    output = plan.output / "7k-1x"
    output.mkdir()
    _write(output / "RUN_STATUS.json", {"status": "FAILED"})
    calls.clear()
    assert matrix.main(_args(plan, "run", *only)) == 2
    assert matrix.main(_args(plan, "run", *only, "--resume-existing")) == 2
    (output / "checkpoint_latest.pt").write_bytes(b"own checkpoint")
    assert matrix.main(_args(plan, "run", *only, "--resume-existing")) == 0
    assert len(calls) == 1 and calls[0][:2] == ("train", "7k-1x")
    argv = calls[0][2]
    assert argv[argv.index("--resume") + 1] == str(output / "checkpoint_latest.pt")


@pytest.mark.parametrize("artifact", ["RUN_STATUS.json", "ROUNDROBIN_SEGMENT.json", "checkpoint_latest.pt",
                                     "checkpoint_epoch_016.pt", "history.json", "epoch_commits/epoch_016.json",
                                     "data_observation/epoch_016_worker_0.json", "EPOCH_SCHEDULE.json"])
def test_completion_requires_actual_bound_evidence(plan, monkeypatch, artifact):
    calls = _children(plan, monkeypatch)
    args = _args(plan, "run", "--only", "7k-1x")
    assert matrix.main(args) == 0
    calls.clear()
    path = plan.output / "7k-1x" / artifact
    path.write_bytes(b"{}" if path.suffix == ".json" else b"changed checkpoint")
    assert matrix.main(args) == 2
    assert not calls


@pytest.mark.parametrize("changed", ["profile", "base", "panel", "runtime", "relocation"])
def test_changed_bound_inputs_are_conflicts(plan, monkeypatch, changed):
    calls = _children(plan, monkeypatch)
    args = _args(plan, "prepare", "--only", "7k-1x")
    assert matrix.main(args) == 0
    calls.clear()
    path = {"profile": plan.assets / plan.rows[0]["profile"], "base": plan.base,
            "panel": Path(plan.inputs["gene_positions_path"]),
            "runtime": plan.output / "7k-1x.input_config.json", "relocation": plan.data / "relocation.json"}[changed]
    path.write_text(path.read_text() + " ")
    assert matrix.main(args) == 2
    assert not calls


def test_irrelevant_legacy_extension_does_not_conflict_with_v2_resume(plan, monkeypatch):
    calls = _children(plan, monkeypatch)
    args = _args(plan, "run", "--only", "7k-1x")
    assert matrix.main([*args, "--stop-after-epoch", "1"]) == 0
    legacy = plan.data / "relocation.pretrain.json"
    legacy.write_text("an unrelated legacy extension is not a v2 input")
    calls.clear()
    assert matrix.main([*args, "--resume-existing"]) == 0
    assert len(calls) == 1 and calls[0][0] == "train"
    legacy.write_text("changed unrelated extension")
    calls.clear()
    assert matrix.main(args) == 0
    assert not calls
    binding = json.loads((plan.output / ".matrix/7k-1x.json").read_text())
    assert str(legacy) not in binding["inputs_sha256"]


@pytest.mark.parametrize("module", ["pbisc.training.portable", "pbisc.training.allcell_roundrobin",
                                    "pbisc_pretrain.matrix", "pbisc_sdd.model", "pbisc_sdd.collection_panel",
                                    "model7_no_annotation.data_e2"])
def test_changed_runtime_implementation_refuses_resume(plan, monkeypatch, module):
    original_source = matrix._module_source
    copy = plan.assets / "implementation.py"
    copy.write_bytes(original_source(module).read_bytes())
    monkeypatch.setattr(matrix, "_module_source", lambda name: copy if name == module else original_source(name))
    calls = _children(plan, monkeypatch)
    args = _args(plan, "prepare", "--only", "7k-1x")
    assert matrix.main(args) == 0
    binding = json.loads((plan.output / ".matrix/7k-1x.json").read_text())
    assert binding["runtime_implementations"][module] == matrix._sha(copy)
    assert all(matrix._digest(value) for value in binding["runtime_implementations"].values())
    copy.write_text(copy.read_text() + "\n# changed implementation\n")
    calls.clear()
    assert matrix.main(_args(plan, "run", "--only", "7k-1x", "--resume-existing")) == 2
    assert not calls


def test_forged_trainer_digest_cannot_establish_completion(plan, monkeypatch):
    calls = _children(plan, monkeypatch)
    args = _args(plan, "run", "--only", "7k-1x")
    assert matrix.main(args) == 0
    output = plan.output / "7k-1x"
    receipt = json.loads((output / "ROUNDROBIN_SEGMENT.json").read_text())
    receipt["trainer_sha256"] = "0" * 64
    for name in ("ROUNDROBIN_SEGMENT.json", "RUN_STATUS.json"):
        _write(output / name, receipt)
    calls.clear()
    assert matrix.main(args) == 2
    assert not calls


def test_implementation_change_during_prepare_prevents_training(plan, monkeypatch):
    original_source = matrix._module_source
    copy = plan.assets / "implementation.py"
    copy.write_bytes(original_source("pbisc_sdd.model").read_bytes())
    monkeypatch.setattr(matrix, "_module_source", lambda name: copy if name == "pbisc_sdd.model" else original_source(name))
    calls = _children(plan, monkeypatch)
    original_child = matrix.subprocess.run

    def changed_child(argv, *, check):
        result = original_child(argv, check=check)
        if argv[3] == "prepare":
            copy.write_text(copy.read_text() + "\n# changed after planning\n")
        return result

    monkeypatch.setattr(matrix.subprocess, "run", changed_child)
    assert matrix.main(_args(plan, "run", "--only", "7k-1x")) == 2
    assert [(command, name) for command, name, _ in calls] == [("prepare", "7k-1x")]


def test_stop_boundary_is_not_part_of_binding_and_new_epoch_commits_suffice(plan, monkeypatch):
    calls = _children(plan, monkeypatch)
    args = _args(plan, "run", "--only", "7k-1x")
    assert matrix.main([*args, "--stop-after-epoch", "1"]) == 0
    calls.clear()
    assert matrix.main([*args, "--resume-existing"]) == 0
    assert len(calls) == 1 and calls[0][0] == "train"
    output = plan.output / "7k-1x"
    _receipt(output, matrix._sha(matrix._materialized(output)), plan.inputs, start=2)
    calls.clear()
    assert matrix.main(args) == 0
    assert not calls


def test_child_zero_exit_without_receipt_does_not_complete(plan, monkeypatch):
    calls = _children(plan, monkeypatch, complete=False)
    assert matrix.main(_args(plan, "run", "--only", "7k-1x")) == 2
    assert len(calls) == 2
