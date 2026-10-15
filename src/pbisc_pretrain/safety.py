"""Launch checks shared by the CLI and the synthetic smoke test."""
from __future__ import annotations

import os
from pathlib import Path
import subprocess


def require_single_process() -> None:
    if int(os.environ.get("WORLD_SIZE", "1")) != 1 or "LOCAL_RANK" in os.environ:
        raise RuntimeError("This trainer is single-GPU only; distributed/torchrun launch is unsupported")


def _ancestor_pids() -> set[int]:
    ancestors = set()
    pid = os.getpid()
    while pid > 0 and pid not in ancestors:
        ancestors.add(pid)
        try:
            status = (Path("/proc") / str(pid) / "status").read_text()
        except (FileNotFoundError, PermissionError):
            break
        pid = int(next(line.split()[1] for line in status.splitlines() if line.startswith("PPid:")))
    return ancestors


def _lock_owned_by_ancestor(lock_stat: os.stat_result, entries: str, ancestors: set[int]) -> bool:
    for line in entries.splitlines():
        fields = line.split()
        if len(fields) < 6 or fields[1] != "FLOCK" or fields[3] != "WRITE":
            continue
        major, minor, inode = fields[5].split(":")
        if (int(major, 16) == os.major(lock_stat.st_dev)
                and int(minor, 16) == os.minor(lock_stat.st_dev)
                and int(inode) == lock_stat.st_ino and int(fields[4]) in ancestors):
            return True
    return False


def check_gpu_launch_policy() -> None:
    """Preserve the mandatory GB10 guard without imposing its services on H100 hosts."""
    require_single_process()
    guard = Path.home() / ".local/libexec/ml-memory-guard"
    query = subprocess.run(
        ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
        capture_output=True, text=True, check=True, timeout=15,
    ).stdout
    is_gb10 = "GB10" in query.upper()
    if guard.exists() or is_gb10:
        if not guard.is_file():
            raise RuntimeError("GB10 requires the installed memory guard and ml-run launcher")
        from pbisc.training.allcell import managed_gate
        managed_gate()
        subprocess.run([str(guard), "--check"], check=True, timeout=15)
        processes = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory", "--format=csv,noheader"],
            capture_output=True, text=True, check=True, timeout=15,
        ).stdout
        if processes.strip():
            raise RuntimeError("GB10 already has a CUDA compute process; wait rather than overlap jobs")
        lock = Path(f"/run/user/{os.getuid()}/ml-gpu0.lock")
        if (not lock.is_file() or lock.is_symlink()
                or not _lock_owned_by_ancestor(lock.stat(), Path("/proc/locks").read_text(), _ancestor_pids())):
            raise RuntimeError("GB10 requires the shared GPU flock to be held by this launch ancestry; use ml-run")
