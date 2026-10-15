"""Small, lazy-import command line for downloading and pretraining PBISC."""
from __future__ import annotations

import argparse
import importlib.metadata
import json
from pathlib import Path
import platform
import subprocess
import sys

from . import __version__


def doctor() -> dict[str, object]:
    packages = {}
    for name in ("torch", "numpy", "pandas", "pyarrow", "zarr", "huggingface-hub", "nvidia-ml-py"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = "MISSING"
    try:
        gpu = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=15, check=True,
        ).stdout.strip().splitlines()
    except (FileNotFoundError, subprocess.SubprocessError):
        gpu = []
    return {"python": platform.python_version(), "platform": platform.platform(),
            "packages": packages, "gpu_inventory": gpu, "cuda_initialized": False,
            "distributed_supported": False, "pbisc_pretrain": __version__}


def _training_args(argv: list[str], *, prepare: bool) -> int:
    parser = argparse.ArgumentParser(prog=f"pbisc-pretrain {'prepare' if prepare else 'train'}")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--data-root", "--dataset-root", dest="data_root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--stop-after-epoch", type=int)
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args(argv)
    from .resources import resolve_asset
    from .safety import require_single_process
    require_single_process()
    config = resolve_asset(args.config)
    corpus_only = json.loads(config.read_text()).get("schema_version") == "pbisc.pretrain.profile.v2"
    preparing = prepare or args.prepare_only
    if not preparing:
        from .dataset import assert_training_ready
        assert_training_ready(args.data_root, corpus_only=corpus_only)
        from .safety import check_gpu_launch_policy
        # All committed production profiles use CUDA. CPU contract work uses `smoke`/tests.
        check_gpu_launch_policy()
    forwarded = ["--config", str(config), "--dataset-root", str(args.data_root),
                 "--output-dir", str(args.output_dir)]
    if args.resume:
        forwarded += ["--resume", str(args.resume)]
    if args.stop_after_epoch is not None:
        forwarded += ["--stop-after-epoch", str(args.stop_after_epoch)]
    if preparing:
        forwarded += ["--prepare-only"]
    from pbisc.training import main as train_main
    return int(train_main(forwarded) or 0)


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version", action="version", version=__version__)
    parser.add_argument("command", nargs="?", choices=("doctor", "data", "prepare", "train", "smoke", "matrix"))
    if not argv or argv[0] in {"-h", "--help", "--version"}:
        parser.parse_args(argv)
        if not argv:
            parser.print_help()
        return 0
    command, *remaining = argv
    if command == "doctor":
        argparse.ArgumentParser(prog="pbisc-pretrain doctor").parse_args(remaining)
        print(json.dumps(doctor(), indent=2))
        return 0
    if command == "data":
        from .dataset import main as data_main
        return int(data_main(remaining) or 0)
    if command == "matrix":
        from .matrix import main as matrix_main
        return int(matrix_main(remaining) or 0)
    if command in {"prepare", "train"}:
        return _training_args(remaining, prepare=command == "prepare")
    if command == "smoke":
        from .smoke import main as smoke_main
        return smoke_main(remaining)
    parser.error(f"unknown command: {command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
