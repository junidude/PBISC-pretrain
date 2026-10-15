"""Prepare and execute the reviewed ablation matrix, one child process at a time.

The controller never imports torch or reads checkpoint payloads. Completion
verification streams checkpoint bytes to SHA-256; it can take time on large runs.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import fcntl
import hashlib
from importlib.machinery import PathFinder
from importlib.util import find_spec
import json
import math
from pathlib import Path
import re
import subprocess
import sys
from typing import Any, Iterator

from .resources import asset_root, resolve_asset

SCHEMA = "pbisc.pretrain.matrix.v1"
BINDING_SCHEMA = "pbisc.pretrain.matrix_binding.v1"
ROUNDROBIN_SCHEMA = "pbisc_sdd.allcell_roundrobin.v1"
_VARIANTS = (("1x", 1, 1), ("trunk2x", 2, 1), ("trunk4x", 4, 1),
             ("ctx2x", 1, 2), ("ctx4x", 1, 4))
_RUNTIME_PACKAGES = ("pbisc", "pbisc_pretrain", "pbisc_sdd", "model7_no_annotation")


def _sha(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def _asset(value: str | Path) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        root = asset_root().resolve()
        path = (root / path).resolve()
        if not path.is_relative_to(root):
            raise ValueError("matrix asset escapes the installed asset root")
    return resolve_asset(path)


def load_matrix(path: str | Path = "configs/matrix.json") -> tuple[Path, list[dict[str, Any]]]:
    """Validate the complete, ordered 3-panel by 5-variant reviewed plan."""
    source = _asset(path)
    manifest = _json(source)
    if manifest.get("schema_version") != SCHEMA:
        raise ValueError("unsupported matrix schema")
    rows = manifest.get("experiments")
    expected = [(f"{n // 1000}k-{name}", n, trunk, context)
                for n in (7000, 8000, 10000) for name, trunk, context in _VARIANTS]
    if not isinstance(rows, list) or len(rows) != len(expected):
        raise ValueError("matrix must contain exactly 15 ordered experiments")
    for row, (identifier, n, trunk, context) in zip(rows, expected, strict=True):
        if (not isinstance(row, dict) or row.get("id") != identifier
                or (row.get("n_genes"), row.get("trunk_multiplier"), row.get("context_multiplier"))
                != (n, trunk, context)):
            raise ValueError("matrix experiment order or axes differ from the reviewed 15-run plan")
        profile = row.get("profile")
        if not isinstance(profile, str) or Path(profile).is_absolute():
            raise ValueError("matrix profiles must use asset-relative paths")
        if not _digest(row.get("profile_sha256")) or _sha(_asset(profile)) != row["profile_sha256"]:
            raise ValueError(f"matrix profile digest differs: {identifier}")
        if _json(_asset(profile)).get("schema_version") != "pbisc.pretrain.profile.v2":
            raise ValueError("matrix accepts only new portable v2 profiles")
    return source, rows


def _digest(value: object) -> bool:
    return isinstance(value, str) and re.fullmatch("[0-9a-f]{64}", value) is not None


def _materialized(output: Path) -> Path:
    return output.parent / f"{output.name}.input_config.json"


def _module_source(name: str) -> Path:
    """Locate the active installation without executing torch-bearing packages."""
    loaded = sys.modules.get(name)
    spec = getattr(loaded, "__spec__", None)
    if spec is None:
        parts = name.split(".")
        spec = find_spec(parts[0])
        for index in range(1, len(parts)):
            if spec is None or spec.submodule_search_locations is None:
                raise ValueError(f"runtime package cannot be resolved: {name}")
            # Unlike find_spec('package.child'), PathFinder does not import the
            # package __init__ to obtain its search path.
            spec = PathFinder.find_spec(".".join(parts[:index + 1]), spec.submodule_search_locations)
    if spec is None or not spec.origin or not spec.origin.endswith(".py"):
        raise ValueError(f"runtime Python source cannot be resolved: {name}")
    return Path(spec.origin).resolve(strict=True)


def _implementation_bindings() -> dict[str, str]:
    records: dict[str, str] = {}
    for package in _RUNTIME_PACKAGES:
        root = _module_source(package).parent
        for candidate in sorted(root.rglob("*.py")):
            relative = candidate.relative_to(root).with_suffix("")
            parts = relative.parts[:-1] if relative.name == "__init__" else relative.parts
            name = ".".join((package, *parts))
            # Bind the import system's resolved source, including an already
            # loaded module, rather than looking for files in the caller's cwd.
            records[name] = _sha(_module_source(name))
    return records


def _check_implementation(binding: dict[str, Any]) -> None:
    if _implementation_bindings() != binding["runtime_implementations"]:
        raise ValueError("runtime implementation changed after matrix planning")


def _binding(source: Path, row: dict[str, Any], data: Path, output: Path) -> dict[str, Any]:
    # This adapter imports only configuration helpers; training imports stay in
    # the public CLI child. The encoded bytes match pbisc.training.main exactly.
    from pbisc.training.portable import prepare_config

    profile = _asset(row["profile"])
    config, trainer = prepare_config(profile, dataset_root=data, output_dir=output)
    if trainer != "allcell" or config["data"]["n_genes"] != row["n_genes"]:
        raise ValueError(f"matrix runtime trainer or gene count differs: {row['id']}")
    model = config.get("model", {})
    if (model.get("hidden_dims") != [width * row["trunk_multiplier"] for width in (1024, 1024, 512, 256)]
            or model.get("context_dim") != 512 * row["context_multiplier"]
            or model.get("time_embedding_dim") != 128):
        raise ValueError(f"matrix runtime model differs from its declared architecture axes: {row['id']}")
    encoded = json.dumps(config, indent=2, sort_keys=True) + "\n"
    inputs: dict[str, str] = {str(profile): _sha(profile)}
    release = config.get("_release", {})
    relocation = data / "relocation.json"
    inputs[str(relocation)] = _sha(relocation)
    for key in ("gene_positions_path", "gene_selection_manifest", "coverage_audit_summary", "topup_manifest"):
        if key in config["data"]:
            path = Path(config["data"][key]).resolve(strict=True)
            actual = _sha(path)
            hash_key = {"gene_positions_path": "gene_positions_sha256",
                        "gene_selection_manifest": "gene_selection_manifest_sha256",
                        "coverage_audit_summary": "coverage_audit_sha256",
                        "topup_manifest": "topup_manifest_sha256"}[key]
            if actual != config["data"].get(hash_key):
                raise ValueError(f"matrix input contract digest differs: {path}")
            inputs[str(path)] = actual
    if release.get("profile_asset_root"):
        root = Path(release["profile_asset_root"]).resolve()
        for relative in release.get("profile_asset_paths", {}).values():
            path = (root / relative).resolve(strict=True)
            if not path.is_relative_to(root):
                raise ValueError("matrix profile asset escapes its root")
            inputs[str(path)] = _sha(path)
    raw_profile = _json(profile)
    if "base_config" in raw_profile:
        base = (profile.parent / raw_profile["base_config"]).resolve(strict=True)
        inputs[str(base)] = _sha(base)
    return {"schema_version": BINDING_SCHEMA, "experiment": row,
            "matrix": str(source), "matrix_sha256": _sha(source),
            "data_root": str(data), "output_dir": str(output),
            "materialized_config": str(_materialized(output)),
            "materialized_config_sha256": hashlib.sha256(encoded.encode()).hexdigest(),
            "data_contract": {key: config["data"].get(key) for key in
                              ("coverage_audit_sha256", "topup_manifest_sha256", "gene_context_collection_contract")},
            "runtime_implementations": _implementation_bindings(),
            "inputs_sha256": inputs}


@contextmanager
def _locked(output_root: Path) -> Iterator[None]:
    control = output_root / ".matrix"
    control.mkdir(parents=True, exist_ok=True)
    with (control / "controller.lock").open("a") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(f"another matrix controller owns {output_root}") from exc
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def _check_binding(path: Path, expected: dict[str, Any], output: Path) -> bool:
    if path.exists():
        if _json(path) != expected:
            raise ValueError(f"preserve conflicting matrix binding: {path}")
        materialized = _materialized(output)
        if not materialized.is_file() or _sha(materialized) != expected["materialized_config_sha256"]:
            raise ValueError(f"materialized input config changed or is missing: {materialized}")
        return True
    if output.exists() or _materialized(output).exists():
        raise FileExistsError(f"refuse to adopt an unbound existing run or input config: {output}")
    return False


def _check_file(record: object, expected_path: Path) -> None:
    if (not isinstance(record, dict) or record.get("path") != str(expected_path.resolve())
            or not _digest(record.get("sha256")) or not expected_path.is_file()
            or _sha(expected_path) != record["sha256"]):
        raise ValueError(f"completion file binding differs: {expected_path}")


def _finite(value: object) -> None:
    if isinstance(value, dict):
        for item in value.values():
            _finite(item)
    elif isinstance(value, list):
        for item in value:
            _finite(item)
    elif isinstance(value, float) and not math.isfinite(value):
        raise ValueError("non-finite completion metrics")


def _completed(output: Path, binding: dict[str, Any], target: int) -> bool:
    _check_implementation(binding)
    if _sha(_materialized(output)) != binding["materialized_config_sha256"]:
        raise ValueError("completion materialized config differs from immutable matrix binding")
    status_path = output / "RUN_STATUS.json"
    if not status_path.is_file():
        return False
    status = _json(status_path)
    if status.get("status") != "SEGMENT_COMPLETED":
        return False
    receipt = _json(output / "ROUNDROBIN_SEGMENT.json")
    end, start = receipt.get("end_epoch"), receipt.get("start_epoch")
    trainer_sha256 = binding["runtime_implementations"]["pbisc.training.allcell_roundrobin"]
    if (receipt != status or receipt.get("schema_version") != ROUNDROBIN_SCHEMA
            or type(end) is not int or type(start) is not int or not 1 <= start <= end
            or receipt.get("epochs_completed") != end
            or receipt.get("config_sha256") != binding["materialized_config_sha256"]
            or receipt.get("trainer_sha256") != trainer_sha256
            or receipt.get("n_genes") != binding["experiment"]["n_genes"]):
        raise ValueError(f"completion receipt differs from matrix configuration: {output}")
    history_path = output / "history.json"
    history = _json(history_path).get("epochs")
    if (not isinstance(history, list) or len(history) != end
            or any(not isinstance(row, dict) for row in history)
            or [row.get("epoch") for row in history] != list(range(1, end + 1))
            or _sha(history_path) != receipt.get("history_sha256")):
        raise ValueError("completion history is missing, changed or noncontiguous")
    _finite(history)
    _check_file(receipt.get("checkpoint_latest"), output / "checkpoint_latest.pt")
    _check_file(receipt.get("checkpoint_epoch"), output / f"checkpoint_epoch_{end:03d}.pt")
    schedule = json.loads((output / "EPOCH_SCHEDULE.json").read_text())
    contract = _json(output / "ROUNDROBIN_DATA_CONTRACT.json")
    if (not isinstance(schedule, list) or len(schedule) != 16
            or contract.get("schema_version") != ROUNDROBIN_SCHEMA
            or contract.get("status") != "SEGMENT_COMPLETED" or contract.get("epochs_completed") != end):
        raise ValueError("completion schedule or data contract differs")
    pairing = contract.get("pairing", {})
    expected_contract = binding["data_contract"]
    schedule_digest = hashlib.sha256(json.dumps(schedule, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    expected_cells, expected_steps = [], []
    for slot, scheduled in enumerate(schedule):
        if not isinstance(scheduled, dict):
            raise ValueError("completion schedule entry is not an object")
        cells = scheduled.get("target_cells_by_worker")
        if (scheduled.get("epoch") != slot + 1 or not isinstance(cells, list) or len(cells) != 4
                or any(type(n) is not int or n < 0 for n in cells)):
            raise ValueError("completion schedule has invalid epoch or worker counts")
        expected_cells.append(sum(cells))
        expected_steps.append(sum(math.ceil(n / 256) for n in cells))
    fingerprint = (expected_contract.get("gene_context_collection_contract") or {}).get("corpus_fingerprint")
    if (not isinstance(pairing, dict) or pairing.get("recipe_schedule_sha256") != schedule_digest
            or pairing.get("coverage_audit_sha256") != expected_contract["coverage_audit_sha256"]
            or pairing.get("topup_manifest_sha256") != expected_contract["topup_manifest_sha256"]
            or (fingerprint and (pairing.get("source_corpus_fingerprint") != fingerprint
                                 or receipt.get("corpus_fingerprint") != fingerprint))
            or contract.get("expected_target_cells_by_epoch") != expected_cells
            or contract.get("expected_optimizer_steps_by_epoch") != expected_steps):
        raise ValueError("completion data pairing differs from input contract or schedule")
    cumulative_steps = 0
    for epoch, record in enumerate(history, 1):
        steps = expected_steps[(epoch - 1) % 16]
        cumulative_steps += steps
        if (not isinstance(record.get("train"), dict) or record["train"].get("optimizer_steps") != steps
                or record.get("global_step") != cumulative_steps):
            raise ValueError("completion history optimizer counts differ from full worker streams")
    if receipt.get("global_step") != cumulative_steps:
        raise ValueError("completion optimizer count differs from history")
    commits = receipt.get("epoch_commits")
    if (not isinstance(commits, list) or any(not isinstance(row, dict) for row in commits)
            or [row.get("epoch") for row in commits] != list(range(start, end + 1))):
        raise ValueError("completion lacks contiguous newly executed epoch commits")
    for record in commits:
        epoch = record["epoch"]
        path = output / "epoch_commits" / f"epoch_{epoch:03d}.json"
        _check_file(record, path)
        commit = _json(path)
        slot = (epoch - 1) % 16
        row = schedule[slot]
        cells = row.get("target_cells_by_worker")
        recipes = row.get("recipe_ids")
        if not isinstance(cells, list) or len(cells) != 4 or not isinstance(recipes, list):
            raise ValueError("completion worker schedule differs")
        steps = sum(math.ceil(n / 256) for n in cells)
        workers = commit.get("worker_observations")
        if (commit.get("schema_version") != "pbisc_sdd.allcell_epoch_commit.v1"
                or commit.get("status") != "EPOCH_COMMITTED" or commit.get("epoch") != epoch
                or commit.get("schedule_slot") != slot
                or commit.get("config_sha256") != binding["materialized_config_sha256"]
                or commit.get("trainer_sha256") != trainer_sha256
                or commit.get("expected_optimizer_steps") != steps
                or commit.get("observed_optimizer_steps") != steps
                or commit.get("expected_target_cells") != sum(cells)
                or commit.get("observed_target_cells") != sum(cells)
                or commit.get("global_step") != history[epoch - 1].get("global_step")
                or history[epoch - 1].get("train", {}).get("optimizer_steps") != steps
                or not isinstance(workers, list) or len(workers) != 4):
            raise ValueError("completion epoch proof differs from configuration or schedule")
        for worker, proof in enumerate(workers):
            proof_path = output / "data_observation" / f"epoch_{epoch:03d}_worker_{worker}.json"
            _check_file(proof, proof_path)
            observed = _json(proof_path)
            expected_order = hashlib.sha256(b"".join(
                str(recipe).encode("ascii") + b"\n" for recipe in recipes[worker::4])).hexdigest()
            if (proof != {**observed, "path": str(proof_path.resolve()), "sha256": _sha(proof_path)}
                    or observed.get("status") != "EXHAUSTED" or observed.get("epoch") != epoch
                    or observed.get("worker") != worker or observed.get("target_cells") != cells[worker]
                    or observed.get("recipes") != len(recipes[worker::4])
                    or observed.get("recipe_order_sha256") != expected_order
                    or not _digest(observed.get("target_order_sha256"))):
                raise ValueError("completion worker exhaustion proof differs")
        if epoch == end:
            if (commit.get("checkpoint_latest") != receipt["checkpoint_latest"]
                    or commit.get("checkpoint_epoch") != receipt["checkpoint_epoch"]
                    or commit.get("history_sha256") != receipt["history_sha256"]):
                raise ValueError("final epoch commit differs from completion receipt")
    return end >= target


def _child(command: str, row: dict[str, Any], data: Path, output: Path,
           *, target: int, resume: bool = False) -> int:
    argv = [sys.executable, "-m", "pbisc_pretrain.cli", command,
            "--config", str(_asset(row["profile"])), "--data-root", str(data),
            "--output-dir", str(output)]
    if command == "train":
        argv += ["--stop-after-epoch", str(target)]
    if resume:
        argv += ["--resume", str(output / "checkpoint_latest.pt")]
    return subprocess.run(argv, check=False).returncode


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="pbisc-pretrain matrix", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for command in ("list", "prepare", "run"):
        child = sub.add_parser(command)
        child.add_argument("--matrix", default="configs/matrix.json", type=Path)
        child.add_argument("--only", action="append", help="experiment ID; repeat to select several")
        if command != "list":
            child.add_argument("--data-root", required=True, type=Path)
            child.add_argument("--output-root", default=Path("runs/ablation"), type=Path)
        if command == "run":
            child.add_argument("--resume-existing", action="store_true")
            child.add_argument("--stop-after-epoch", default=16, type=int)
    args = parser.parse_args(argv)
    results: list[dict[str, Any]] = []
    try:
        source, rows = load_matrix(args.matrix)
        only = set(args.only or [])
        if only - {row["id"] for row in rows}:
            raise ValueError(f"unknown matrix experiment IDs: {sorted(only - {row['id'] for row in rows})}")
        selected = [row for row in rows if not only or row["id"] in only]
        if args.command == "list":
            print(json.dumps({"schema_version": SCHEMA, "experiments": selected}, separators=(",", ":")))
            return 0
        data = args.data_root.expanduser().resolve()
        output_root = args.output_root.expanduser().resolve()
        target = getattr(args, "stop_after_epoch", 16)
        if target <= 0:
            raise ValueError("stop-after-epoch must be positive")
        with _locked(output_root):
            # Validate the entire selection before launching its first child.
            plans = []
            for row in selected:
                output = output_root / row["id"]
                expected = _binding(source, row, data, output)
                control = output_root / ".matrix" / f"{row['id']}.json"
                bound = _check_binding(control, expected, output)
                complete = output.exists() and _completed(output, expected, target)
                if output.exists() and not complete and args.command == "run":
                    if not args.resume_existing:
                        raise FileExistsError(f"partial run requires explicit --resume-existing: {output}")
                    if not (output / "checkpoint_latest.pt").is_file():
                        raise ValueError(f"partial run lacks its own checkpoint_latest.pt: {output}")
                plans.append((row, output, expected, control, bound, complete))
            for row, output, expected, control, bound, complete in plans:
                _check_implementation(expected)
                if complete and args.command == "run":
                    results.append({"id": row["id"], "status": "VERIFIED_COMPLETE"})
                    continue
                if not bound:
                    code = _child("prepare", row, data, output, target=target)
                    if code:
                        print(json.dumps({"status": "CHILD_FAILED", "id": row["id"], "command": "prepare",
                                          "returncode": code, "experiments": results}, separators=(",", ":")))
                        return code if code > 0 else 1
                    if output.exists() or not _materialized(output).is_file():
                        raise ValueError(f"prepare created a run directory or omitted its input config: {output}")
                    if _sha(_materialized(output)) != expected["materialized_config_sha256"]:
                        raise ValueError("prepared runtime config differs from matrix binding")
                    with control.open("x") as handle:
                        handle.write(json.dumps(expected, indent=2, sort_keys=True) + "\n")
                if args.command == "prepare":
                    results.append({"id": row["id"], "status": "PREPARED"})
                    continue
                _check_implementation(expected)
                code = _child("train", row, data, output, target=target, resume=output.exists())
                if code:
                    print(json.dumps({"status": "CHILD_FAILED", "id": row["id"], "command": "train",
                                      "returncode": code, "experiments": results}, separators=(",", ":")))
                    return code if code > 0 else 1
                if not _completed(output, expected, target):
                    raise ValueError(f"training child returned without verified completion: {output}")
                results.append({"id": row["id"], "status": "VERIFIED_COMPLETE"})
        print(json.dumps({"status": "PREPARED" if args.command == "prepare" else "MATRIX_COMPLETED",
                          "experiments": results}, separators=(",", ":")))
        return 0
    except (OSError, ValueError, RuntimeError, KeyError, TypeError) as exc:
        print(json.dumps({"status": "MATRIX_CONFLICT", "error": str(exc), "experiments": results},
                         separators=(",", ":")), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
