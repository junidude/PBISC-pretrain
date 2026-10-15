#!/usr/bin/env python3
"""Scan tracked candidates without displaying any matched credential value."""
from __future__ import annotations

from pathlib import Path
import re
import subprocess
import sys


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    listing = subprocess.run(["git", "ls-files", "-z", "--cached", "--others", "--exclude-standard"],
                             cwd=root, capture_output=True, check=True).stdout
    patterns = [re.compile(rb"hf_[A-Za-z0-9]{25,}"),
                re.compile(rb"gh[pousr]_[A-Za-z0-9]{25,}"),
                re.compile(rb"-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY-----")]
    forbidden = {".pt", ".pth", ".ckpt", ".safetensors", ".h5ad", ".parquet", ".npz", ".key", ".pem"}
    errors = []
    files = sorted({name.decode() for name in listing.split(b"\0") if name})
    for relative in files:
        path = root / relative
        if not path.is_file() or path.is_symlink():
            errors.append((relative, "not a regular file"))
            continue
        panel_axis = (relative.startswith(("panels/", "gene_context/panels/", "gene_context/historical/"))
                      and path.name == "gene_positions.npy")
        ranked_axis = relative == "gene_context/inputs/ranked_gene_positions.npy"
        if path.suffix in forbidden or (path.suffix == ".npy" and not (panel_axis or ranked_axis)):
            errors.append((relative, "unapproved data/checkpoint/credential file"))
        if path.stat().st_size > 5 * 1024 * 1024:
            errors.append((relative, "file exceeds reviewed 5 MiB source limit"))
        if any(pattern.search(path.read_bytes()) for pattern in patterns):
            errors.append((relative, "credential pattern detected; value redacted"))
    for relative, reason in errors:
        print(f"FAIL {relative}: {reason}", file=sys.stderr)
    print(f"Repository scan: {'FAIL' if errors else 'PASS'} ({len(files)} files)")
    return int(bool(errors))


if __name__ == "__main__":
    raise SystemExit(main())
