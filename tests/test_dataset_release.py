from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path
import tarfile

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from pbisc.data_release import (
    build_inventory, download_dataset, export_dataset, restore_dataset,
    safe_extract_archive, sha256_file, verify_dataset, write_inventory,
)


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def make_corpus(tmp_path: Path) -> dict:
    source, index, base = (tmp_path / name for name in ("source", "index", "base"))
    (source / "metadata").mkdir(parents=True)
    (index / "metadata").mkdir(parents=True)
    (index / "recipes").mkdir()
    base.mkdir()
    rows = []
    for i, root in enumerate((base, source)):
        name = f"shard{i}"
        count = root / "counts" / (name + ".zarr")
        count.mkdir(parents=True)
        write_json(count / ("zarr.json" if i == 0 else ".zgroup"), {"zarr_format": 3 if i == 0 else 2})
        for component in ("data", "indices", "indptr", "measurement_mask"):
            (count / component).mkdir()
            (count / component / "0").write_bytes(bytes([i, 0, 3, 0, 255]))
        obs_name = f"metadata/{name}.parquet"
        pq.write_table(pa.table({"dataset_id": ["study", "study"], "donor_id": ["a", "b"],
                                 "sample_id": ["s1", "s2"], "row_index": [0, 1]}), source / obs_name)
        rows.append({"shard_id": name, "dataset_id": "study_GSE123", "n_cells": 2, "n_genes": 3,
                     "count_zarr": f"{'base' if i == 0 else 'strict'}://counts/{name}.zarr",
                     "obs_part": f"strict://{obs_name}", "count_source": "raw/X"})
    pq.write_table(pa.Table.from_pylist(rows), source / "metadata/sc_shards.parquet")
    pq.write_table(pa.Table.from_pylist(rows), index / "metadata/source_shards.parquet")
    pq.write_table(pa.table({"gene_id": ["g1", "g2", "g3"]}), source / "metadata/var.parquet")
    pq.write_table(pa.table({"sample_key": ["sample"], "shard_id": ["shard0"],
                             "row_indices": pa.array([[0, 1]], type=pa.list_(pa.int32()))}), index / "metadata/sample_pools.parquet")
    pq.write_table(pa.table({"seed": pa.array([2**64 - 1], type=pa.uint64())}), index / "recipes/train.parquet")
    write_json(source / "dataset_manifest.json", {"source_fingerprint_sha256": "a" * 64, "source_root": str(source)})
    write_json(source / "BUILD_COMPLETE.json", {"status": "complete"})
    write_json(index / "dataset_manifest.json", {"corpus_fingerprint": "b" * 64,
               "recipe_part_files": [{"recipe_part": "recipes/train.parquet"}]})
    (index / "COMPLETE").write_text("b" * 64)
    topup = tmp_path / "topup"
    topup.mkdir()
    for name in ("shard_offsets.csv", "topup_cell_ids.npy", "topup_recipes.csv"):
        (topup / name).write_bytes(b"fixture")
    write_json(topup / "manifest.json", {"artifacts_sha256": {}})
    panel = tmp_path / "gene_positions.npy"
    panel.write_bytes(b"panel")
    conditioner = tmp_path / "conditioner_cache.pt"
    conditioner.write_bytes(b"opaque-checkpoint")
    inventory = build_inventory(source_root=source, index_root=index, base_root=base,
                                topup_root=topup, panels=[panel],
                                extra_assets=[(conditioner, "conditioner/conditioner_cache.pt")])
    path = tmp_path / "inventory.json"
    write_inventory(inventory, path)
    return {"inventory": inventory, "path": path, "source": source, "index": index, "base": base}


def test_inventory_and_roundtrip_preserve_all_bytes_and_uint64(tmp_path):
    corpus = make_corpus(tmp_path)
    inventory = corpus["inventory"]
    assert inventory["summary"]["n_base_shards"] == 1
    assert inventory["summary"]["n_overlay_shards"] == 1
    assert inventory["rights"][0]["rights_status"] == "unverified"
    release_root, destination = tmp_path / "export", tmp_path / "restored"
    release = export_dataset(corpus["path"], release_root)
    assert release["complete"]
    verify_dataset(release_root)
    assert restore_dataset(release_root, destination)["complete"]
    for record in inventory["metadata"] + [r for s in inventory["shards"] for r in s["files"]]:
        assert Path(record["source"]).read_bytes() == (destination / record["path"]).read_bytes()
    restored = pq.read_table(destination / "index/recipes/train.parquet")
    assert restored["seed"].to_pylist() == [2**64 - 1]
    mapping = json.loads((destination / "relocation.json").read_text())
    assert mapping["asset_paths"][str(tmp_path / "conditioner_cache.pt")] == "conditioner/conditioner_cache.pt"
    assert mapping["original_sha256"]["source_manifest"] == sha256_file(corpus["source"] / "dataset_manifest.json")


def test_deterministic_resume_bounded_export(tmp_path):
    corpus = make_corpus(tmp_path)
    root1, root2 = tmp_path / "a", tmp_path / "b"
    assert not export_dataset(corpus["path"], root1, metadata_only=True)["complete"]
    partial = export_dataset(corpus["path"], root1, max_shards=1)
    assert not partial["complete"] and partial["completed_shards"] == 1
    with pytest.raises(ValueError, match="not been exported"):
        restore_dataset(root1, tmp_path / "incomplete")
    finished = export_dataset(corpus["path"], root1, max_shards=1)
    assert finished["complete"]
    independent = export_dataset(corpus["path"], root2)
    assert finished == independent
    for a in finished["assets"]:
        assert (root1 / a["path"]).read_bytes() == (root2 / a["path"]).read_bytes()


def test_selected_download_restore_and_corruption(tmp_path):
    corpus = make_corpus(tmp_path)
    root = tmp_path / "export"
    release = export_dataset(corpus["path"], root)
    restored = restore_dataset(root, tmp_path / "one", shard_ids=["shard0"])
    assert restored["shards"] == ["shard0"] and not restored["complete"]
    archive = next(a for a in release["assets"] if a["kind"] == "tar")
    with (root / archive["path"]).open("ab") as stream:
        stream.write(b"corrupt")
    with pytest.raises(ValueError, match="checksum/size mismatch"):
        verify_dataset(root)
    with pytest.raises(ValueError, match="checksum/size mismatch"):
        export_dataset(corpus["path"], root)


def test_source_change_invalidates_export(tmp_path):
    corpus = make_corpus(tmp_path)
    record = corpus["inventory"]["shards"][0]["files"][0]
    Path(record["source"]).write_bytes(b"changed")
    with pytest.raises(ValueError, match="changed since inventory"):
        export_dataset(corpus["path"], tmp_path / "export")


@pytest.mark.parametrize("name,typeflag", [("../outside", tarfile.REGTYPE), ("/absolute", tarfile.REGTYPE),
                                           ("file", tarfile.SYMTYPE), ("file", tarfile.LNKTYPE)])
def test_safe_extraction_rejects_traversal_and_links(tmp_path, name, typeflag):
    path = tmp_path / "unsafe.tar"
    info = tarfile.TarInfo(name)
    info.type = typeflag
    info.linkname = "../outside" if typeflag != tarfile.REGTYPE else ""
    info.size = 1 if typeflag == tarfile.REGTYPE else 0
    with tarfile.open(path, "w") as archive:
        archive.addfile(info, io.BytesIO(b"x") if info.size else None)
    member = {"path": name, "size_bytes": info.size, "sha256": hashlib.sha256(b"x").hexdigest()}
    with pytest.raises(ValueError):
        safe_extract_archive(path, [member], tmp_path / "dest")
    assert not (tmp_path / "outside").exists()


def test_safe_extraction_rejects_existing_symlink(tmp_path):
    destination = tmp_path / "dest"
    destination.mkdir()
    (destination / "link").symlink_to(tmp_path, target_is_directory=True)
    with pytest.raises(ValueError, match="symlinks"):
        safe_extract_archive(tmp_path / "unused.tar", [{"path": "link/escape", "size_bytes": 0, "sha256": ""}], destination)


def test_downloader_requires_commit_revision(tmp_path):
    with pytest.raises(ValueError, match="40-character"):
        download_dataset("example/data", "main", tmp_path)


def test_inventory_rejects_mismatched_shards(tmp_path):
    corpus = make_corpus(tmp_path)
    pq.write_table(pa.table({"shard_id": ["other"]}), corpus["index"] / "metadata/source_shards.parquet")
    with pytest.raises(ValueError, match="coverage"):
        build_inventory(source_root=corpus["source"], index_root=corpus["index"], base_root=corpus["base"])
