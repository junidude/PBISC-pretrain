"""Portable, lossless transport of the manifest-selected PBISC training corpus.

Inventory reads table footers and directory metadata only. Export copies bytes and
streams one deterministic tar per CSR shard; it never decodes expression arrays.
Nothing in this module uploads data or changes the original research corpus.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import tarfile
import tempfile
from typing import Any, BinaryIO, Iterable, Sequence

SCHEMA = "pbisc.dataset_release.v1"
INVENTORY_SCHEMA = "pbisc.dataset_inventory.v1"
BUFFER_SIZE = 1024 * 1024


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(BUFFER_SIZE), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".partial")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def _safe_join(root: Path, relative: str) -> Path:
    parts = PurePosixPath(relative)
    if not relative or parts.is_absolute() or "\\" in relative or any(p in {"", ".", ".."} for p in relative.split("/")):
        raise ValueError(f"unsafe relative path: {relative!r}")
    result = root.joinpath(*parts.parts)
    for ancestor in (result, *result.parents):
        if ancestor == root.parent:
            break
        if ancestor.is_symlink():
            raise ValueError(f"symlinks are not accepted: {ancestor}")
    result.resolve().relative_to(root.resolve())
    return result


def _stat_record(path: Path, relative: str) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"expected regular source file: {path}")
    stat = path.stat()
    return {"source": str(path), "path": relative, "size_bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def _unchanged(record: dict[str, Any]) -> Path:
    path = Path(record["source"])
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"source disappeared or is a symlink: {path}")
    stat = path.stat()
    if (stat.st_size, stat.st_mtime_ns) != (record["size_bytes"], record["mtime_ns"]):
        raise ValueError(f"source changed since inventory: {path}")
    return path


def _files(root: Path) -> Iterable[Path]:
    for directory, dirs, names in os.walk(root, followlinks=False):
        dirs.sort()
        for name in dirs:
            if (Path(directory) / name).is_symlink():
                raise ValueError(f"symlink in source tree: {Path(directory) / name}")
        for name in sorted(names):
            yield Path(directory) / name


def _uri(uri: str, source: Path, base: Path) -> tuple[Path, str]:
    for scheme, root, prefix in (("strict://", source, "source"), ("base://", base, "base")):
        if uri.startswith(scheme):
            relative = uri[len(scheme):]
            return _safe_join(root, relative), f"{prefix}/{relative}"
    raise ValueError(f"unsupported source URI: {uri}")


def build_inventory(*, source_root: str | Path, index_root: str | Path,
                    base_root: str | Path, topup_root: str | Path | None = None,
                    panels: Sequence[str | Path] = (),
                    extra_assets: Sequence[tuple[str | Path, str]] = ()) -> dict[str, Any]:
    """Enumerate only referenced assets without hashing/reading expression data."""
    import pyarrow.parquet as pq

    source, index, base = (Path(p).expanduser().resolve() for p in (source_root, index_root, base_root))
    source_manifest = _json(source / "dataset_manifest.json")
    index_manifest = _json(index / "dataset_manifest.json")
    rows = pq.read_table(source / "metadata/sc_shards.parquet").to_pylist()
    index_rows = pq.read_table(index / "metadata/source_shards.parquet").to_pylist()
    by_id = {str(r["shard_id"]): r for r in rows}
    if len(by_id) != len(rows) or set(by_id) != {str(r["shard_id"]) for r in index_rows}:
        raise ValueError("source/index shard coverage differs or IDs are duplicated")
    for row in index_rows:
        if any(row[key] != by_id[str(row["shard_id"])][key] for key in ("dataset_id", "n_cells", "n_genes", "count_zarr", "obs_part")):
            raise ValueError(f"source/index shard contract differs: {row['shard_id']}")
    metadata: dict[str, dict[str, Any]] = {}
    def add_file(path: Path, relative: str) -> None:
        record = _stat_record(path, relative)
        if relative in metadata and record != metadata[relative]:
            raise ValueError(f"colliding metadata destination: {relative}")
        metadata[relative] = record
    for name in ("dataset_manifest.json", "BUILD_COMPLETE.json", "metadata/sc_shards.parquet", "metadata/var.parquet"):
        add_file(source / name, f"source/{name}")
    for name in ("source_audit_prebuild.json", "metadata/source_audit.json"):
        if (source / name).is_file():
            add_file(source / name, f"source/{name}")
    for name in ("dataset_manifest.json", "COMPLETE"):
        add_file(index / name, f"index/{name}")
    if (index / "audit.json").is_file():
        add_file(index / "audit.json", "index/audit.json")
    for path in sorted((index / "metadata").glob("*.parquet")):
        add_file(path, f"index/metadata/{path.name}")
    for record in index_manifest["recipe_part_files"]:
        relative = str(record["recipe_part"])
        add_file(_safe_join(index, relative), f"index/{relative}")
    shards = []
    for row in sorted(rows, key=lambda r: str(r["shard_id"])):
        count_path, count_relative = _uri(str(row["count_zarr"]), source, base)
        obs_path, obs_relative = _uri(str(row["obs_part"]), source, base)
        if not count_path.is_dir():
            raise ValueError(f"missing CSR shard: {count_path}")
        for component in ("data", "indices", "indptr", "measurement_mask"):
            if not (count_path / component).is_dir():
                raise ValueError(f"missing CSR/mask component {component}: {count_path}")
        zarr_format = 3 if (count_path / "zarr.json").is_file() else 2 if (count_path / ".zgroup").is_file() else None
        if zarr_format is None:
            raise ValueError(f"unrecognized Zarr format: {count_path}")
        if pq.ParquetFile(obs_path).metadata.num_rows != int(row["n_cells"]):
            raise ValueError(f"row-aligned obs length differs: {obs_path}")
        add_file(obs_path, obs_relative)
        members = [_stat_record(p, f"{count_relative}/{p.relative_to(count_path).as_posix()}") for p in _files(count_path)]
        identifier = hashlib.sha256(str(row["shard_id"]).encode()).hexdigest()[:24]
        role = "base" if str(row["count_zarr"]).startswith("base://") else "overlay"
        shards.append({"shard_id": str(row["shard_id"]), "dataset_id": str(row["dataset_id"]),
                       "n_cells": int(row["n_cells"]), "n_genes": int(row["n_genes"]),
                       "count_source": str(row.get("count_source", "unspecified")), "role": role,
                       "zarr_format": zarr_format, "archive": f"counts/{role}/{identifier}.tar",
                       "count_path": count_relative, "source_directory": str(count_path),
                       "obs_path": obs_relative, "size_bytes": sum(m["size_bytes"] for m in members), "files": members})
    original_roots = {"source_root": str(source), "base_source_root": str(base), "index_root": str(index),
                      "gene_panel_path": str(source / "metadata/var.parquet")}
    relative_roots = {"source_root": "source", "base_source_root": "base", "index_root": "index", "gene_panel_path": "source/metadata/var.parquet"}
    if topup_root is not None:
        topup = Path(topup_root).expanduser().resolve()
        manifest = _json(topup / "manifest.json")
        names = {"manifest.json", "shard_offsets.csv", "topup_cell_ids.npy", "topup_recipes.csv", *manifest.get("artifacts_sha256", {})}
        for name in sorted(names):
            add_file(_safe_join(topup, name), f"topup/{name}")
        original_roots["topup_root"] = str(topup)
        relative_roots["topup_root"] = "topup"
    for raw_panel in panels:
        panel = Path(raw_panel).expanduser().resolve()
        add_file(panel, f"panels/{panel.name}")
    for raw_source, relative in extra_assets:
        _safe_join(Path("/portable-root"), relative)
        add_file(Path(raw_source).expanduser().resolve(), relative)
    asset_paths = {r["source"]: r["path"] for r in metadata.values()}
    rights = []
    for dataset_id in sorted({s["dataset_id"] for s in shards}):
        subset = [s for s in shards if s["dataset_id"] == dataset_id]
        accession = re.search(r"GSE\d+", dataset_id)
        rights.append({"dataset_id": dataset_id, "accession_hint": accession.group(0) if accession else "",
                       "source_url": "", "license": "", "permission_evidence": "", "rights_status": "unverified",
                       "public_redistribution": "not_authorized_by_this_inventory", "n_shards": len(subset),
                       "count_bytes": sum(s["size_bytes"] for s in subset)})
    relocation = {"schema_version": "pbisc.dataset_relocation.v1", "original_roots": original_roots,
                  "relative_roots": relative_roots, "asset_paths": asset_paths,
                  "original_sha256": {"source_manifest": sha256_file(source / "dataset_manifest.json"),
                                      "source_complete": sha256_file(source / "BUILD_COMPLETE.json"),
                                      "index_manifest": sha256_file(index / "dataset_manifest.json")}}
    result = {"schema_version": INVENTORY_SCHEMA, "relocation": relocation,
              "source_fingerprint": source_manifest.get("source_fingerprint_sha256"),
              "corpus_fingerprint": index_manifest.get("corpus_fingerprint"),
              "metadata": sorted(metadata.values(), key=lambda r: r["path"]), "shards": shards, "rights": rights,
              "summary": {"n_datasets": len(rights), "n_shards": len(shards),
                          "n_base_shards": sum(s["role"] == "base" for s in shards),
                          "n_overlay_shards": sum(s["role"] == "overlay" for s in shards),
                          "n_cells": sum(s["n_cells"] for s in shards),
                          "n_genes": sorted({s["n_genes"] for s in shards}),
                          "metadata_bytes": sum(r["size_bytes"] for r in metadata.values()),
                          "count_bytes": sum(s["size_bytes"] for s in shards),
                          "source_files": len(metadata) + sum(len(s["files"]) for s in shards),
                          "validation_scope": "metadata/footer coverage only; values and rights not audited"}}
    result["inventory_sha256"] = _digest(result)
    return result


def write_inventory(inventory: dict[str, Any], output: str | Path) -> None:
    output = Path(output)
    _write_json(output, inventory)
    _write_rights(output.with_name("rights.tsv"), inventory["rights"])


def _write_rights(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]) if rows else ["dataset_id", "rights_status"], delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


class _HashReader:
    def __init__(self, stream: BinaryIO):
        self.stream = stream
        self.digest = hashlib.sha256()
    def read(self, size: int = -1) -> bytes:
        block = self.stream.read(size)
        self.digest.update(block)
        return block


class _HashWriter:
    def __init__(self, stream: BinaryIO):
        self.stream = stream
        self.digest = hashlib.sha256()
    def write(self, block: bytes) -> int:
        self.digest.update(block)
        return self.stream.write(block)


def _copy_record(record: dict[str, Any], destination: Path) -> dict[str, Any]:
    source = _unchanged(record)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".partial")
    digest = hashlib.sha256()
    with source.open("rb") as src, temporary.open("wb") as dst:
        for block in iter(lambda: src.read(BUFFER_SIZE), b""):
            digest.update(block)
            dst.write(block)
        dst.flush()
        os.fsync(dst.fileno())
    _unchanged(record)
    temporary.replace(destination)
    return {"path": record["path"], "size_bytes": record["size_bytes"], "sha256": digest.hexdigest(), "kind": "file"}


def _archive_shard(shard: dict[str, Any], output: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    destination = _safe_join(output, shard["archive"])
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".partial")
    actual = {str(p) for p in _files(Path(shard["source_directory"]))}
    if actual != {r["source"] for r in shard["files"]}:
        raise ValueError(f"shard file set changed since inventory: {shard['shard_id']}")
    members = []
    with temporary.open("wb") as destination_stream:
        writer = _HashWriter(destination_stream)
        with tarfile.open(fileobj=writer, mode="w|", format=tarfile.PAX_FORMAT) as archive:
            for record in sorted(shard["files"], key=lambda r: r["path"]):
                source = _unchanged(record)
                member = tarfile.TarInfo(record["path"])
                member.size = record["size_bytes"]
                member.mode = 0o644
                member.mtime = member.uid = member.gid = 0
                member.uname = member.gname = ""
                with source.open("rb") as stream:
                    reader = _HashReader(stream)
                    archive.addfile(member, reader)
                    digest = reader.digest.hexdigest()
                _unchanged(record)
                members.append({"path": record["path"], "size_bytes": record["size_bytes"], "sha256": digest})
        destination_stream.flush()
        os.fsync(destination_stream.fileno())
    temporary.replace(destination)
    members_path = shard["archive"] + ".members.json"
    _write_json(_safe_join(output, members_path), {"schema_version": "pbisc.archive_members.v1", "members": members})
    archive_asset = {"path": shard["archive"], "size_bytes": destination.stat().st_size,
                     "sha256": writer.digest.hexdigest(), "kind": "tar", "shard_id": shard["shard_id"], "members_path": members_path}
    member_file = _safe_join(output, members_path)
    return archive_asset, {"path": members_path, "size_bytes": member_file.stat().st_size, "sha256": sha256_file(member_file), "kind": "archive_members"}


def _verify_asset(root: Path, asset: dict[str, Any]) -> Path:
    path = _safe_join(root, asset["path"])
    if not path.is_file() or path.stat().st_size != asset["size_bytes"] or sha256_file(path) != asset["sha256"]:
        raise ValueError(f"checksum/size mismatch: {asset['path']}")
    return path


def export_dataset(inventory_path: str | Path, output: str | Path, *, max_shards: int | None = None,
                   metadata_only: bool = False) -> dict[str, Any]:
    """Export immutable bytes, resumably at file/shard boundaries.

    max_shards bounds *new* archives in this invocation. Run again to continue.
    A partial export is never marked complete. Existing completed assets are
    rehashed before reuse. The .export/ directory is local state, not for upload.
    """
    inventory = _json(Path(inventory_path))
    expected = inventory.pop("inventory_sha256")
    if inventory.get("schema_version") != INVENTORY_SCHEMA or _digest(inventory) != expected:
        raise ValueError("inventory fingerprint or schema differs")
    inventory["inventory_sha256"] = expected
    output = Path(output).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    state_root = output / ".export"
    state_root.mkdir(exist_ok=True)
    lock_path = state_root / "export.lock"
    import fcntl
    with lock_path.open("a+") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        state_path = state_root / "state.json"
        state = _json(state_path) if state_path.exists() else {"inventory_sha256": expected, "assets": {}}
        if state["inventory_sha256"] != expected:
            raise ValueError("destination was initialized with another inventory")
        assets = state["assets"]
        def record(asset: dict[str, Any]) -> None:
            assets[asset["path"]] = asset
            _write_json(state_path, state)
            with (state_root / "events.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(json.dumps({"event": "asset_complete", **asset}, sort_keys=True) + "\n")
        for item in inventory["metadata"]:
            _unchanged(item)
            if item["path"] in assets:
                _verify_asset(output, assets[item["path"]])
            else:
                record(_copy_record(item, _safe_join(output, item["path"])))
        new_shards = 0
        for shard in inventory["shards"]:
            if shard["archive"] in assets:
                for member in shard["files"]:
                    _unchanged(member)
                _verify_asset(output, assets[shard["archive"]])
                _verify_asset(output, assets[shard["archive"] + ".members.json"])
                continue
            if metadata_only or (max_shards is not None and new_shards >= max_shards):
                continue
            archive_asset, members_asset = _archive_shard(shard, output)
            record(members_asset)
            record(archive_asset)
            new_shards += 1
        _write_json(output / "relocation.json", inventory["relocation"])
        _write_rights(output / "provenance/rights.tsv", inventory["rights"])
        for relative in ("relocation.json", "provenance/rights.tsv"):
            path = output / relative
            record({"path": relative, "size_bytes": path.stat().st_size, "sha256": sha256_file(path), "kind": "file"})
        completed = [s["shard_id"] for s in inventory["shards"] if s["archive"] in assets]
        release = {"schema_version": SCHEMA, "inventory_sha256": expected,
                   "complete": len(completed) == len(inventory["shards"]), "completed_shards": len(completed),
                   "summary": inventory["summary"], "source_fingerprint": inventory["source_fingerprint"],
                   "corpus_fingerprint": inventory["corpus_fingerprint"],
                   "rights_status": "unverified; inventory is not redistribution permission",
                   "shards": [{k: v for k, v in shard.items() if k not in {"files", "source_directory"}} for shard in inventory["shards"]],
                   "assets": sorted(assets.values(), key=lambda r: r["path"])}
        _write_json(output / "release.json", release)
        return release


def safe_extract_archive(archive_path: Path, members: list[dict[str, Any]], destination: Path) -> None:
    """Extract only regular declared members, checking each digest and size."""
    expected = {m["path"]: m for m in members}
    if len(expected) != len(members):
        raise ValueError("duplicate member names in archive manifest")
    for relative in expected:
        _safe_join(destination, relative)
    seen: set[str] = set()
    with tarfile.open(archive_path, mode="r|*") as archive:
        for member in archive:
            if not member.isfile() or member.name not in expected or member.name in seen:
                raise ValueError(f"unexpected or unsafe archive member: {member.name}")
            contract = expected[member.name]
            if member.size != contract["size_bytes"]:
                raise ValueError(f"archive member size differs: {member.name}")
            target = _safe_join(destination, member.name)
            target.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(dir=target.parent, prefix=".pbisc-extract-", delete=False) as stream:
                temporary = Path(stream.name)
                try:
                    digest = hashlib.sha256()
                    source = archive.extractfile(member)
                    if source is None:
                        raise ValueError(f"unreadable archive member: {member.name}")
                    with source:
                        for block in iter(lambda: source.read(BUFFER_SIZE), b""):
                            digest.update(block)
                            stream.write(block)
                    stream.flush()
                    os.fsync(stream.fileno())
                    if digest.hexdigest() != contract["sha256"]:
                        raise ValueError(f"archive member checksum differs: {member.name}")
                except BaseException:
                    temporary.unlink(missing_ok=True)
                    raise
            temporary.chmod(0o644)
            temporary.replace(target)
            seen.add(member.name)
    if seen != set(expected):
        raise ValueError("archive is missing declared members")


def _selected_assets(release: dict[str, Any], shard_ids: Sequence[str] | None) -> list[dict[str, Any]]:
    if release.get("schema_version") != SCHEMA:
        raise ValueError("unsupported release schema")
    known = {s["shard_id"] for s in release["shards"]}
    selected = known if shard_ids is None else set(shard_ids)
    if not selected.issubset(known):
        raise ValueError(f"unknown shards: {sorted(selected - known)}")
    available = {a["shard_id"] for a in release["assets"] if a["kind"] == "tar"}
    if not selected.issubset(available):
        raise ValueError("requested shards have not been exported")
    archives = {a["path"] for a in release["assets"] if a["kind"] == "tar" and a["shard_id"] in selected}
    member_paths = {a["members_path"] for a in release["assets"] if a["path"] in archives}
    assets = [a for a in release["assets"] if a["kind"] == "file" or a["path"] in archives | member_paths]
    paths = [a["path"] for a in assets]
    if len(paths) != len(set(paths)):
        raise ValueError("duplicate release asset paths")
    return assets


def verify_dataset(release_root: str | Path, *, shard_ids: Sequence[str] | None = None) -> dict[str, Any]:
    """Verify every selected transport asset against its SHA256 and byte size."""
    root = Path(release_root).resolve()
    release = _json(root / "release.json")
    assets = _selected_assets(release, shard_ids)
    for asset in assets:
        _verify_asset(root, asset)
    return {"schema_version": SCHEMA, "verified_assets": len(assets),
            "release_sha256": sha256_file(root / "release.json"),
            "scope": "transport bytes only; no biological or redistribution-rights assessment"}


def restore_dataset(release_root: str | Path, destination: str | Path, *, shard_ids: Sequence[str] | None = None) -> dict[str, Any]:
    release_root, destination = Path(release_root).resolve(), Path(destination).resolve()
    if release_root == destination:
        raise ValueError("restore destination must differ from transport directory")
    release = _json(release_root / "release.json")
    assets = _selected_assets(release, shard_ids)
    destination.mkdir(parents=True, exist_ok=True)
    for asset in assets:
        source = _verify_asset(release_root, asset)
        if asset["kind"] == "file":
            record = _stat_record(source, asset["path"])
            copied = _copy_record(record, _safe_join(destination, asset["path"]))
            if copied["sha256"] != asset["sha256"]:
                raise ValueError(f"source changed during restore: {source}")
    by_path = {asset["path"]: asset for asset in assets}
    for asset in assets:
        if asset["kind"] == "tar":
            member_path = _verify_asset(release_root, by_path[asset["members_path"]])
            members = _json(member_path)["members"]
            safe_extract_archive(_safe_join(release_root, asset["path"]), members, destination)
    restored = {"schema_version": "pbisc.dataset_restored.v1", "release_sha256": sha256_file(release_root / "release.json"),
                "corpus_fingerprint": release["corpus_fingerprint"],
                "shards": sorted(a["shard_id"] for a in assets if a["kind"] == "tar"),
                "complete": sum(a["kind"] == "tar" for a in assets) == len(release["shards"])}
    _write_json(destination / "restored.json", restored)
    return restored


def download_dataset(repo_id: str, revision: str, cache_dir: str | Path, *,
                     shard_ids: Sequence[str] | None = None, destination: str | Path | None = None) -> dict[str, Any]:
    """Download checksum-bound assets from an explicitly pinned HF dataset SHA."""
    if not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError("revision must be a full 40-character dataset commit SHA")
    from huggingface_hub import hf_hub_download
    cache = Path(cache_dir).resolve()
    cache.mkdir(parents=True, exist_ok=True)
    hf_hub_download(repo_id, "release.json", repo_type="dataset", revision=revision, local_dir=cache)
    release = _json(cache / "release.json")
    for asset in _selected_assets(release, shard_ids):
        _safe_join(cache, asset["path"])
        hf_hub_download(repo_id, asset["path"], repo_type="dataset", revision=revision, local_dir=cache)
        _verify_asset(cache, asset)
    if destination is not None:
        return restore_dataset(cache, destination, shard_ids=shard_ids)
    return {"repo_id": repo_id, "revision": revision, "release_root": str(cache), "download_verified": True}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    inventory = commands.add_parser("inventory", help="metadata-only inventory; no expression hashing")
    for name in ("source-root", "index-root", "base-root", "output"):
        inventory.add_argument(f"--{name}", required=True)
    inventory.add_argument("--topup-root")
    inventory.add_argument("--panel", action="append", default=[])
    inventory.add_argument("--asset", action="append", default=[], metavar="SOURCE=RELATIVE_PATH",
                           help="Additional conditioner/panel/implementation assets with portable destinations")
    export = commands.add_parser("export")
    export.add_argument("--inventory", required=True)
    export.add_argument("--output", required=True)
    export.add_argument("--max-shards", type=int)
    export.add_argument("--metadata-only", action="store_true")
    restore = commands.add_parser("restore")
    restore.add_argument("--release-root", required=True)
    restore.add_argument("--destination", required=True)
    restore.add_argument("--shard-id", action="append")
    verify = commands.add_parser("verify", help="verify selected transport files by SHA256")
    verify.add_argument("--release-root", required=True)
    verify.add_argument("--shard-id", action="append")
    download = commands.add_parser("download")
    download.add_argument("--repo-id", required=True)
    download.add_argument("--revision", required=True)
    download.add_argument("--cache-dir", required=True)
    download.add_argument("--destination")
    download.add_argument("--shard-id", action="append")
    args = parser.parse_args(argv)
    if args.command == "inventory":
        result = build_inventory(source_root=args.source_root, index_root=args.index_root, base_root=args.base_root,
                                 topup_root=args.topup_root, panels=args.panel,
                                 extra_assets=[tuple(item.split("=", 1)) for item in args.asset])
        write_inventory(result, args.output)
        print(json.dumps(result["summary"], indent=2))
    elif args.command == "export":
        if args.max_shards is not None and args.max_shards < 0:
            parser.error("--max-shards must be nonnegative")
        result = export_dataset(args.inventory, args.output, max_shards=args.max_shards, metadata_only=args.metadata_only)
        print(json.dumps({"complete": result["complete"], "completed_shards": result["completed_shards"], "assets": len(result["assets"])}, indent=2))
    elif args.command == "restore":
        print(json.dumps(restore_dataset(args.release_root, args.destination, shard_ids=args.shard_id), indent=2))
    elif args.command == "verify":
        print(json.dumps(verify_dataset(args.release_root, shard_ids=args.shard_id), indent=2))
    else:
        print(json.dumps(download_dataset(args.repo_id, args.revision, args.cache_dir, shard_ids=args.shard_id, destination=args.destination), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
