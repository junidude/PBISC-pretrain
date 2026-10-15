from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from pbisc_pretrain.cli import main
from pbisc_pretrain.safety import require_single_process


def test_cli_help_does_not_import_torch():
    result = subprocess.run([sys.executable, "-c",
        "import sys; from pbisc_pretrain.cli import main; main(['--help']); "
        "assert 'torch' not in sys.modules"], text=True, capture_output=True)
    assert result.returncode == 0, result.stderr


def test_doctor_does_not_initialize_cuda(capsys):
    assert main(["doctor"]) == 0
    assert json.loads(capsys.readouterr().out)["cuda_initialized"] is False


def test_distributed_launch_is_rejected(monkeypatch):
    monkeypatch.setenv("WORLD_SIZE", "2")
    with pytest.raises(RuntimeError, match="single-GPU"):
        require_single_process()


def test_gpu_flock_requires_the_matching_inode_and_launch_ancestor(tmp_path):
    from pbisc_pretrain.safety import _lock_owned_by_ancestor
    lock = tmp_path / "lock"
    lock.touch()
    stat = lock.stat()
    record = f"1: FLOCK ADVISORY WRITE 123 {os.major(stat.st_dev):02x}:{os.minor(stat.st_dev):02x}:{stat.st_ino} 0 EOF"
    assert _lock_owned_by_ancestor(stat, record, {123, 124})
    assert not _lock_owned_by_ancestor(stat, record, {999})
    assert not _lock_owned_by_ancestor(stat, record.replace(str(stat.st_ino), str(stat.st_ino + 1)), {123})


def test_synthetic_smoke_replays_optimizer_and_ema(tmp_path: Path):
    from pbisc_pretrain.smoke import run_smoke
    report = run_smoke(tmp_path / "smoke", "cpu")
    assert report["exact_resumed_model"] and report["exact_resumed_ema"]
    assert report["unavailable_output_zero"] and not report["cuda_initialized"]
    with pytest.raises(FileExistsError):
        run_smoke(tmp_path / "smoke", "cpu")
