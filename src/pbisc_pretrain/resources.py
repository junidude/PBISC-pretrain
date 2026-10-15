"""Locate reviewed assets in a checkout or an installed wheel."""
from __future__ import annotations

from pathlib import Path
import sys


def asset_root() -> Path:
    checkout = Path(__file__).resolve().parents[2]
    if (checkout / "pyproject.toml").is_file() and (checkout / "configs").is_dir():
        return checkout
    installed = Path(sys.prefix) / "share" / "pbisc-pretrain"
    if (installed / "configs").is_dir() and (installed / "manifests").is_dir():
        return installed
    raise FileNotFoundError("PBISC-pretrain assets are missing; reinstall the complete wheel")


def resolve_asset(value: str | Path) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute() or path.is_file():
        return path.resolve(strict=True)
    root = asset_root().resolve()
    candidate = (root / path).resolve(strict=True)
    if not candidate.is_relative_to(root):
        raise ValueError("relative asset escapes the installed PBISC-pretrain root")
    return candidate
