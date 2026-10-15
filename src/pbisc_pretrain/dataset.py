"""Pinned, resumable HF transport and hash-verified restoration for pretraining.

The frozen release and CSR transport implementation are preserved. This adapter
adds final 6k/7k/8k panel bindings and a full-restoration training gate. A bounded
shard readback deliberately does not download the complete recipe metadata.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import fcntl
import hashlib
import json
from pathlib import Path
import re
import shutil
from typing import Any, Iterator, Sequence

from pbisc.data_release import (
    SCHEMA, _copy_record, _json, _safe_join, _stat_record, _verify_asset,
    _write_json, safe_extract_archive, sha256_file,
)

LOCK_SCHEMA = "pbisc.pretrain.dataset_lock.v1"
RECEIPT_SCHEMA = "pbisc.pretrain.dataset_verification.v1"


def _digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def load_lock(path: Path | str | None = None) -> dict[str, Any]:
    if path is None:
        from .resources import asset_root
        path = asset_root() / "manifests/dataset.lock.json"
    lock = _json(Path(path))
    if lock.get("schema_version") != LOCK_SCHEMA or not re.fullmatch(r"[0-9a-f]{40}", lock.get("revision", "")):
        raise ValueError("dataset lock requires the supported schema and immutable 40-character revision")
    return lock


def transport_root(cache_dir: Path | str, lock: dict[str, Any]) -> Path:
    return Path(cache_dir).expanduser().resolve() / lock["revision"]


def _roots(data_root: Path | str, cache_dir: Path | str, lock: dict[str, Any]) -> tuple[Path, Path]:
    root, cache = Path(data_root).expanduser().resolve(), transport_root(cache_dir, lock)
    if root == cache or root.is_relative_to(cache) or cache.is_relative_to(root):
        raise ValueError("data root and transport cache must be separate, non-nested directories")
    return root, cache


def _release(root: Path, lock: dict[str, Any]) -> dict[str, Any]:
    _verify_asset(root, lock["release"])
    release = _json(root / lock["release"]["path"])
    if release.get("schema_version") != SCHEMA or not release.get("complete"):
        raise ValueError("pinned release must be complete")
    if (release.get("corpus_fingerprint") != lock["corpus_fingerprint"]
            or release.get("source_fingerprint") != lock["source_fingerprint"]
            or len(release["assets"]) != lock["release_asset_count"]
            or len(release["shards"]) != lock["count_shards"]):
        raise ValueError("pinned release corpus, source or asset/shard contract differs")
    paths = [a["path"] for a in release["assets"]]
    if len(paths) != len(set(paths)):
        raise ValueError("duplicate release asset paths")
    for relative in paths:
        _safe_join(root, relative)
    by_path = {a["path"]: a for a in release["assets"]}
    for asset in [lock["relocation"], *lock.get("required_root_assets", [])]:
        if any(by_path.get(asset["path"], {}).get(k) != asset[k] for k in ("path", "sha256", "size_bytes")):
            raise ValueError(f"pinned required asset differs: {asset['path']}")
    return release


def _assets(release: dict[str, Any], shard_ids: Sequence[str] | None) -> list[dict[str, Any]]:
    if shard_ids is None:
        return release["assets"]
    known = {s["shard_id"] for s in release["shards"]}
    selected = set(shard_ids)
    if not selected or not selected.issubset(known):
        raise ValueError("unknown or empty shard selection")
    archives = [a for a in release["assets"] if a["kind"] == "tar" and a["shard_id"] in selected]
    if len(archives) != len(selected):
        raise ValueError("requested shards have not been exported")
    wanted = {"relocation.json", *(a["path"] for a in archives), *(a["members_path"] for a in archives)}
    return [a for a in release["assets"] if a["path"] in wanted]


def _nearest_existing(path: Path) -> Path:
    while not path.exists():
        path = path.parent
    return path


def _disk_plan(root: Path, cache: Path, cache_bytes: int, restored_bytes: int, headroom: int) -> dict[str, Any]:
    groups: dict[int, dict[str, Any]] = {}
    for role, path, needed in (("cache", cache, cache_bytes), ("data", root, restored_bytes)):
        existing = _nearest_existing(path)
        device = existing.stat().st_dev
        group = groups.setdefault(device, {"device": device, "paths": {}, "required_bytes": headroom,
                                          "headroom_bytes": headroom, "available_bytes": shutil.disk_usage(existing).free})
        group["paths"][role] = str(path)
        group["required_bytes"] += needed
    for group in groups.values():
        group["sufficient"] = group["available_bytes"] >= group["required_bytes"]
    return {"filesystems": list(groups.values()), "sufficient": all(g["sufficient"] for g in groups.values())}


def plan_dataset(data_root: Path | str, cache_dir: Path | str, *, shard_ids: Sequence[str] | None = None,
                 lock: dict[str, Any] | None = None, corpus_only: bool = False) -> dict[str, Any]:
    """Plan without payload downloads; full sizing is available entirely offline."""
    lock = load_lock() if lock is None else lock
    root, cache = _roots(data_root, cache_dir, lock)
    budget = lock["byte_budgets"]
    if shard_ids is None:
        cache_bytes, restored_bytes = budget["cache_total"], budget["restored_total"]
        if corpus_only:
            extra = budget.get("supplemental_assets", sum(a["size_bytes"] for a in lock["supplemental_assets"]))
            cache_bytes, restored_bytes = cache_bytes - extra, restored_bytes - extra
    else:
        if not (cache / "release.json").is_file():
            raise ValueError("subset planning needs the pinned release.json in the transport cache; download fetches it first")
        assets = _assets(_release(cache, lock), shard_ids)
        cache_bytes = sum(a["size_bytes"] for a in assets) + lock["release"]["size_bytes"]
        # Tar transport size is a conservative bound for the extracted payload.
        restored_bytes = cache_bytes
    return {"schema_version": "pbisc.pretrain.dataset_plan.v1", "repo_id": lock["repo_id"], "revision": lock["revision"],
            "scope": "full" if shard_ids is None else "subset", "training_ready_after_subset": False,
            "data_scope": "canonical_corpus" if corpus_only else "legacy_with_panels",
            "cache_bytes": cache_bytes, "restored_bytes": restored_bytes,
            "sizing": "conservative fresh-copy bytes; existing files are checked and reused during download",
            **_disk_plan(root, cache, cache_bytes, restored_bytes, budget["headroom_per_filesystem"])}


@contextmanager
def _locked(root: Path, *, shared: bool = False) -> Iterator[None]:
    state = _safe_join(root, ".pretrain")
    state.mkdir(parents=True, exist_ok=True)
    path = _safe_join(root, ".pretrain/data.lock")
    with path.open("a+") as handle:
        fcntl.flock(handle, (fcntl.LOCK_SH if shared else fcntl.LOCK_EX) | fcntl.LOCK_NB)
        yield


def _fetch(cache: Path, asset: dict[str, Any], lock: dict[str, Any], *, remote_path: str | None = None) -> Path:
    target = _safe_join(cache, asset["path"])
    if target.exists():
        return _verify_asset(cache, asset)
    from huggingface_hub import hf_hub_download
    # local_dir streams into this explicit cache, avoiding a second full HF blob cache.
    downloaded = Path(hf_hub_download(lock["repo_id"], remote_path or asset["path"], repo_type="dataset",
                                    revision=lock["revision"], local_dir=cache))
    if downloaded.resolve() != target.resolve():
        # Supplemental HF paths differ from their portable restored destinations.
        _verify_asset(cache, {**asset, "path": remote_path or asset["path"]})
        target.parent.mkdir(parents=True, exist_ok=True)
        downloaded.replace(target)
    return _verify_asset(cache, asset)


def _copy_verified(source: Path, root: Path, asset: dict[str, Any]) -> None:
    target = _safe_join(root, asset["path"])
    if target.exists():
        _verify_asset(root, asset)
        return
    copied = _copy_record(_stat_record(source, asset["path"]), target)
    if copied["sha256"] != asset["sha256"]:
        raise ValueError(f"source changed during copy: {asset['path']}")


def _extension(root: Path, lock: dict[str, Any]) -> dict[str, Any]:
    relocation = _json(_verify_asset(root, lock["relocation"]))
    mapping = dict(relocation["asset_paths"])
    for asset in lock["supplemental_assets"]:
        original, destination = asset["original_path"], asset["destination"]
        if original in mapping and mapping[original] != destination:
            raise ValueError("supplemental relocation conflicts with preserved relocation")
        mapping[original] = destination
    return {**relocation, "schema_version": "pbisc.pretrain_dataset_relocation.v1", "asset_paths": mapping,
            "original_relocation_sha256": lock["relocation"]["sha256"],
            "pretrain_extension": {"schema_version": "pbisc.pretrain.relocation_extension.v1",
                                   "base_relocation_sha256": lock["relocation"]["sha256"], "lock_sha256": _digest(lock)}}


def _members(cache: Path, archive: dict[str, Any], by_path: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    return _json(_verify_asset(cache, by_path[archive["members_path"]]))["members"]


def download_dataset(data_root: Path | str, cache_dir: Path | str, *, shard_ids: Sequence[str] | None = None,
                     lock: dict[str, Any] | None = None, corpus_only: bool = False) -> dict[str, Any]:
    """Download and restore the pinned corpus, or perform a bounded shard readback.

    Existing files with differing bytes are rejected. Completed files are rehashed
    before reuse. HF authentication uses its normal local credentials/environment.
    """
    lock = load_lock() if lock is None else lock
    root, cache = _roots(data_root, cache_dir, lock)
    cache.mkdir(parents=True, exist_ok=True)
    with _locked(cache), _locked(root):
        _fetch(cache, lock["release"], lock)
        release = _release(cache, lock)
        assets = _assets(release, shard_ids)
        supplemental = lock["supplemental_assets"] if shard_ids is None and not corpus_only else []
        # Charge only missing cache files. Destination credit is granted only to
        # files that were previously hashed and whose stat seals still match.
        missing_cache = sum(a["size_bytes"] for a in assets if not _safe_join(cache, a["path"]).exists())
        missing_cache += sum(a["size_bytes"] for a in supplemental if not _safe_join(cache, a["destination"]).exists())
        saved = _restored_credit(root, cache, assets, supplemental, lock, corpus_only=corpus_only)
        restored_bound = (lock["byte_budgets"]["restored_total"] if shard_ids is None
                          else sum(a["size_bytes"] for a in assets) + lock["release"]["size_bytes"])
        if corpus_only and shard_ids is None:
            restored_bound -= lock["byte_budgets"].get(
                "supplemental_assets", sum(a["size_bytes"] for a in lock["supplemental_assets"]))
        disk = _disk_plan(root, cache, missing_cache, max(0, restored_bound - saved),
                          lock["byte_budgets"]["headroom_per_filesystem"])
        if not disk["sufficient"]:
            raise ValueError("insufficient disk space for transport + restoration + filesystem headroom: " + json.dumps(disk))
        for asset in assets:
            _fetch(cache, asset, lock)
        _copy_verified(cache / "release.json", root, lock["release"])
        by_path = {a["path"]: a for a in assets}
        for asset in assets:
            if asset["kind"] != "tar":
                _copy_verified(_verify_asset(cache, asset), root, asset)
        for archive in (a for a in assets if a["kind"] == "tar"):
            members = _members(cache, archive, by_path)
            missing = False
            for member in members:
                target = _safe_join(root, member["path"])
                if target.exists():
                    _verify_asset(root, member)
                else:
                    missing = True
            if missing:
                safe_extract_archive(_verify_asset(cache, archive), members, root)
        for asset in supplemental:
            portable = {**asset, "path": asset["destination"]}
            _copy_verified(_fetch(cache, portable, lock, remote_path=asset["path"]), root, portable)
        if shard_ids is None and not corpus_only:
            extension = _extension(root, lock)
            extension_path = _safe_join(root, "relocation.pretrain.json")
            if extension_path.exists() and _json(extension_path) != extension:
                raise ValueError("existing pretrain relocation conflicts with pinned extension")
            _write_json(extension_path, extension)
        return _verify_restored(root, lock, shard_ids, corpus_only=corpus_only)


def _seal(root: Path, asset: dict[str, Any]) -> dict[str, Any]:
    path = _verify_asset(root, asset)
    stat = path.stat()
    return {"path": asset["path"], "sha256": asset["sha256"], "size_bytes": stat.st_size,
            "mtime_ns": stat.st_mtime_ns, "ctime_ns": stat.st_ctime_ns, "inode": stat.st_ino, "device": stat.st_dev}


def _check_seals(root: Path, records: list[dict[str, Any]]) -> None:
    for record in records:
        path = _safe_join(root, record["path"])
        if not path.is_file():
            raise ValueError(f"verified restored file is missing: {record['path']}")
        stat = path.stat()
        if any(actual != record[key] for actual, key in (
            (stat.st_size, "size_bytes"), (stat.st_mtime_ns, "mtime_ns"), (stat.st_ctime_ns, "ctime_ns"),
            (stat.st_ino, "inode"), (stat.st_dev, "device"))):
            raise ValueError(f"verified restored file changed; run data verify: {record['path']}")


def _ledger(root: Path, lock: dict[str, Any], *, corpus_only: bool = False) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    prefix = "corpus-" if corpus_only else ""
    receipt = _json(_safe_join(root, f".pretrain/{prefix}verification.json"))
    ledger_path = _safe_join(root, f".pretrain/{prefix}verified-files.json")
    if (receipt.get("schema_version") != RECEIPT_SCHEMA or receipt.get("revision") != lock["revision"]
            or receipt.get("lock_sha256") != _digest(lock) or sha256_file(ledger_path) != receipt.get("ledger_sha256")):
        raise ValueError("verification receipt or file ledger is not bound to the pinned lock")
    if receipt.get("data_scope", "legacy_with_panels") != ("canonical_corpus" if corpus_only else "legacy_with_panels"):
        raise ValueError("verification receipt data scope differs")
    records = _json(ledger_path)["files"]
    if len(records) != receipt.get("verified_files") or len({r["path"] for r in records}) != len(records):
        raise ValueError("verification file ledger count differs")
    _check_seals(root, records)
    return receipt, records


def _sealed_bytes(root: Path, lock: dict[str, Any], *, corpus_only: bool = False) -> int:
    try:
        receipt, records = _ledger(root, lock, corpus_only=corpus_only)
    except (OSError, ValueError, KeyError):
        return 0
    if receipt.get("status") != "TRAIN_READY" or receipt.get("verified_shards") != lock["count_shards"]:
        return 0
    return sum(r["size_bytes"] for r in records)


def _restored_credit(root: Path, cache: Path, assets: list[dict[str, Any]], supplemental: list[dict[str, Any]],
                     lock: dict[str, Any], *, corpus_only: bool = False) -> int:
    """Credit only verified existing bytes, including an interrupted restoration."""
    sealed = _sealed_bytes(root, lock, corpus_only=corpus_only)
    if sealed:
        return sealed
    expected = [lock["release"], *(a for a in assets if a["kind"] != "tar")]
    expected.extend({**a, "path": a["destination"]} for a in supplemental)
    by_path = {a["path"]: a for a in assets}
    for archive in (a for a in assets if a["kind"] == "tar"):
        manifest = by_path[archive["members_path"]]
        for location in (root, cache):
            if _safe_join(location, manifest["path"]).exists():
                expected.extend(_json(_verify_asset(location, manifest))["members"])
                break
    total = 0
    for asset in expected:
        if _safe_join(root, asset["path"]).exists():
            _verify_asset(root, asset)
            total += asset["size_bytes"]
    return total


def _verify_restored(root: Path, lock: dict[str, Any], shard_ids: Sequence[str] | None,
                     *, corpus_only: bool = False) -> dict[str, Any]:
    release = _release(root, lock)
    assets = _assets(release, shard_ids)
    by_path = {a["path"]: a for a in assets}
    expected = [lock["release"], *(a for a in assets if a["kind"] != "tar")]
    for archive in (a for a in assets if a["kind"] == "tar"):
        expected.extend(_members(root, archive, by_path))
    if shard_ids is None and not corpus_only:
        expected.extend({**a, "path": a["destination"]} for a in lock["supplemental_assets"])
        path = _safe_join(root, "relocation.pretrain.json")
        if _json(path) != _extension(root, lock):
            raise ValueError("pretrain relocation extension differs from pinned assets")
        expected.append({"path": "relocation.pretrain.json", "sha256": sha256_file(path), "size_bytes": path.stat().st_size})
    paths = [a["path"] for a in expected]
    if len(paths) != len(set(paths)):
        raise ValueError("duplicate or colliding restored file paths")
    records = [_seal(root, a) for a in expected]
    prefix = "corpus-" if corpus_only else ""
    ledger = _safe_join(root, f".pretrain/{prefix}verified-files.json")
    _write_json(ledger, {"schema_version": "pbisc.pretrain.file_seals.v1", "files": records})
    receipt = {"schema_version": RECEIPT_SCHEMA, "repo_id": lock["repo_id"], "revision": lock["revision"],
               "lock_sha256": _digest(lock), "release_sha256": lock["release"]["sha256"],
               "data_scope": "canonical_corpus" if corpus_only else "legacy_with_panels",
               "corpus_fingerprint": lock["corpus_fingerprint"],
               "status": "TRAIN_READY" if shard_ids is None else "SUBSET_READY", "complete": shard_ids is None,
               "verified_shards": sum(a["kind"] == "tar" for a in assets), "verified_files": len(records),
               "verified_bytes": sum(r["size_bytes"] for r in records), "ledger_sha256": sha256_file(ledger),
               "verification_scope": "SHA256 of all declared restored files; archive members verified individually; no CUDA/training"}
    _write_json(_safe_join(root, f".pretrain/{prefix}verification.json"), receipt)
    return receipt


def verify_dataset(data_root: Path | str, *, shard_ids: Sequence[str] | None = None,
                   lock: dict[str, Any] | None = None, corpus_only: bool = False) -> dict[str, Any]:
    """Rehash all restored files (including CSR bytes), independent of cache."""
    lock = load_lock() if lock is None else lock
    root = Path(data_root).expanduser().resolve()
    with _locked(root):
        return _verify_restored(root, lock, shard_ids, corpus_only=corpus_only)


def assert_training_ready(data_root: Path | str, *, lock: dict[str, Any] | None = None,
                          corpus_only: bool = False) -> dict[str, Any]:
    """Require full hash-audit evidence and unchanged file seals before training.

    A full `data verify` streams the entire corpus once. Subsequent training
    starts check size/mtime/ctime/inode/device seals instead of rehashing 111 GB.
    Moving or modifying a restored root requires another full verification.
    """
    lock = load_lock() if lock is None else lock
    root = Path(data_root).expanduser().resolve()
    try:
        with _locked(root, shared=True):
            receipt, records = _ledger(root, lock, corpus_only=corpus_only)
            if (receipt.get("status") != "TRAIN_READY" or receipt.get("complete") is not True
                    or receipt.get("verified_shards") != lock["count_shards"]):
                raise ValueError("partial/metadata-only restoration cannot train; complete data download and full data verify are required")
            # Reconstruct exact expected file coverage from hash-bound manifests;
            # a mutable status flag or truncated ledger cannot authorize training.
            release = _release(root, lock)
            expected_assets = [lock["release"]]
            if not corpus_only:
                extension_path = _safe_join(root, "relocation.pretrain.json")
                expected_assets.append({"path": "relocation.pretrain.json",
                                       "sha256": sha256_file(extension_path), "size_bytes": extension_path.stat().st_size})
                expected_assets.extend({**a, "path": a["destination"]} for a in lock["supplemental_assets"])
            by_path = {a["path"]: a for a in release["assets"]}
            for asset in release["assets"]:
                if asset["kind"] == "tar":
                    expected_assets.extend(_members(root, asset, by_path))
                else:
                    expected_assets.append(asset)
            expected = {a["path"]: a for a in expected_assets}
            if {r["path"] for r in records} != set(expected) or len(expected) != len(expected_assets):
                raise ValueError("verification ledger does not cover the full pinned corpus")
            if any(r[k] != expected[r["path"]][k] for r in records for k in ("sha256", "size_bytes")):
                raise ValueError("verification ledger hashes differ from pinned manifests")
            if not corpus_only and _json(root / "relocation.pretrain.json") != _extension(root, lock):
                raise ValueError("pinned relocation extension changed")
            return receipt
    except (OSError, KeyError) as exc:
        raise ValueError("dataset is not TRAIN_READY; run complete data download and data verify") from exc


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for command in ("plan", "download", "verify"):
        sub = commands.add_parser(command)
        sub.add_argument("--data-root", required=True)
        sub.add_argument("--cache-dir", required=command != "verify")
        sub.add_argument("--shard-id", action="append", help="bounded readback; never enables training")
        sub.add_argument("--corpus-only", action="store_true", help="full canonical corpus; gene panels supplied by the repository")
    args = parser.parse_args(argv)
    try:
        if args.command == "plan":
            result = plan_dataset(args.data_root, args.cache_dir, shard_ids=args.shard_id, corpus_only=args.corpus_only)
        elif args.command == "download":
            result = download_dataset(args.data_root, args.cache_dir, shard_ids=args.shard_id, corpus_only=args.corpus_only)
        else:
            result = verify_dataset(args.data_root, shard_ids=args.shard_id, corpus_only=args.corpus_only)
    except (OSError, ValueError) as exc:
        parser.exit(2, f"data {args.command}: {exc}\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
