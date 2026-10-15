from __future__ import annotations

import json
from pathlib import Path
import shutil

import pytest

from pbisc.data_release import export_dataset, sha256_file
from pbisc_pretrain import dataset
from test_dataset_release import make_corpus


@pytest.fixture
def remote(tmp_path, monkeypatch):
    corpus = make_corpus(tmp_path / "research")
    remote = tmp_path / "remote"
    release = export_dataset(corpus["path"], remote)
    by_path = {a["path"]: a for a in release["assets"]}
    overlay = remote / "supplements/panel.npy"
    overlay.parent.mkdir()
    overlay.write_bytes(b"selected-panel")
    asset = {"path": "supplements/panel.npy", "destination": "panels/6k/gene_positions.npy",
             "original_path": "/research/final6k/gene_positions.npy", "size_bytes": overlay.stat().st_size,
             "sha256": sha256_file(overlay)}
    lock = {"schema_version": dataset.LOCK_SCHEMA, "repo_id": "example/fixture", "revision": "a" * 40,
            "release": {"path": "release.json", "size_bytes": (remote / "release.json").stat().st_size,
                        "sha256": sha256_file(remote / "release.json")},
            "relocation": by_path["relocation.json"], "source_fingerprint": release["source_fingerprint"],
            "corpus_fingerprint": release["corpus_fingerprint"], "release_asset_count": len(release["assets"]),
            "count_shards": len(release["shards"]), "required_root_assets": [by_path["index/dataset_manifest.json"]],
            "supplemental_assets": [asset],
            "byte_budgets": {"cache_total": 200000, "restored_total": 200000, "headroom_per_filesystem": 0}}
    calls = []

    def fetch(repo_id, filename, *, repo_type, revision, local_dir):
        assert repo_id == lock["repo_id"] and revision == lock["revision"] and repo_type == "dataset"
        calls.append(filename)
        target = Path(local_dir) / filename
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(remote / filename, target)
        return str(target)

    monkeypatch.setattr("huggingface_hub.hf_hub_download", fetch)
    return {"root": remote, "lock": lock, "calls": calls, "release": release,
            "data": tmp_path / "data", "cache": tmp_path / "cache"}


def test_full_roundtrip_preserves_frozen_manifests_and_gates_training(remote):
    r = remote
    receipt = dataset.download_dataset(r["data"], r["cache"], lock=r["lock"])
    assert receipt["status"] == "TRAIN_READY" and receipt["verified_shards"] == 2
    assert dataset.assert_training_ready(r["data"], lock=r["lock"]) == receipt
    for relative in ("relocation.json", "source/dataset_manifest.json", "index/dataset_manifest.json"):
        assert (r["data"] / relative).read_bytes() == (r["root"] / relative).read_bytes()
    extension = json.loads((r["data"] / "relocation.pretrain.json").read_text())
    assert extension["schema_version"] == "pbisc.pretrain_dataset_relocation.v1"
    assert extension["original_relocation_sha256"] == r["lock"]["relocation"]["sha256"]
    assert extension["asset_paths"]["/research/final6k/gene_positions.npy"] == "panels/6k/gene_positions.npy"
    assert dataset.verify_dataset(r["data"], lock=r["lock"])["status"] == "TRAIN_READY"


def test_download_resumes_verified_files_without_network_or_rewrites(remote):
    r = remote
    dataset.download_dataset(r["data"], r["cache"], lock=r["lock"])
    stat = (r["data"] / "source/dataset_manifest.json").stat()
    calls = list(r["calls"])
    dataset.download_dataset(r["data"], r["cache"], lock=r["lock"])
    assert r["calls"] == calls
    assert (r["data"] / "source/dataset_manifest.json").stat().st_mtime_ns == stat.st_mtime_ns


def test_interrupted_full_restore_credits_new_verified_files_after_subset(remote, monkeypatch):
    r = remote
    dataset.download_dataset(r["data"], r["cache"], shard_ids=["shard0"], lock=r["lock"])
    extract = dataset.safe_extract_archive

    def interrupt_after_extract(*args):
        extract(*args)
        raise RuntimeError("simulated interruption after completing a shard")

    monkeypatch.setattr(dataset, "safe_extract_archive", interrupt_after_extract)
    with pytest.raises(RuntimeError, match="simulated interruption"):
        dataset.download_dataset(r["data"], r["cache"], lock=r["lock"])
    monkeypatch.setattr(dataset, "safe_extract_archive", extract)
    cache = dataset.transport_root(r["cache"], r["lock"])
    credit = dataset._restored_credit(r["data"], cache, r["release"]["assets"],
                                       r["lock"]["supplemental_assets"], r["lock"])
    # A stale subset receipt must not hide subsequently completed full files.
    subset = json.loads((r["data"] / ".pretrain/verification.json").read_text())
    assert credit > subset["verified_bytes"]
    remaining = r["lock"]["byte_budgets"]["restored_total"] - credit
    remaining += sum(a["size_bytes"] for a in r["lock"]["supplemental_assets"])
    monkeypatch.setattr(dataset.shutil, "disk_usage", lambda _: type("Usage", (), {"free": remaining + 1})())
    assert dataset.download_dataset(r["data"], r["cache"], lock=r["lock"])["status"] == "TRAIN_READY"


def test_subset_is_bounded_and_never_training_ready(remote):
    r = remote
    receipt = dataset.download_dataset(r["data"], r["cache"], shard_ids=["shard0"], lock=r["lock"])
    assert receipt["status"] == "SUBSET_READY" and not receipt["complete"]
    assert len(r["calls"]) == 4  # release, relocation, one tar, its member manifest
    assert not (r["data"] / "index/dataset_manifest.json").exists()
    assert not (r["data"] / "panels/6k/gene_positions.npy").exists()
    with pytest.raises(ValueError, match="partial/metadata-only"):
        dataset.assert_training_ready(r["data"], lock=r["lock"])
    with pytest.raises((ValueError, FileNotFoundError)):
        dataset.verify_dataset(r["data"], lock=r["lock"])
    assert dataset.verify_dataset(r["data"], shard_ids=["shard0"], lock=r["lock"])["status"] == "SUBSET_READY"
    assert dataset.download_dataset(r["data"], r["cache"], lock=r["lock"])["status"] == "TRAIN_READY"


def test_metadata_only_or_forged_status_cannot_train(remote):
    r = remote
    with pytest.raises(ValueError, match="not TRAIN_READY"):
        dataset.assert_training_ready(r["data"], lock=r["lock"])
    dataset.download_dataset(r["data"], r["cache"], shard_ids=["shard0"], lock=r["lock"])
    path = r["data"] / ".pretrain/verification.json"
    value = json.loads(path.read_text())
    value.update(status="TRAIN_READY", complete=True, verified_shards=2)
    path.write_text(json.dumps(value))
    with pytest.raises((ValueError, FileNotFoundError)):
        dataset.assert_training_ready(r["data"], lock=r["lock"])


def test_corrupt_cache_refused_without_overwriting(remote):
    r = remote
    cache = dataset.transport_root(r["cache"], r["lock"])
    cache.mkdir(parents=True)
    (cache / "release.json").write_bytes(b"conflict")
    with pytest.raises(ValueError, match="checksum/size mismatch"):
        dataset.download_dataset(r["data"], r["cache"], lock=r["lock"])
    assert not r["calls"]
    assert (cache / "release.json").read_bytes() == b"conflict"


def test_changed_counts_fail_seals_and_full_hash_audit(remote):
    r = remote
    dataset.download_dataset(r["data"], r["cache"], lock=r["lock"])
    target = r["data"] / "base/counts/shard0.zarr/data/0"
    original = target.read_bytes()
    target.write_bytes(bytes([99]) + original[1:])
    with pytest.raises(ValueError, match="changed"):
        dataset.assert_training_ready(r["data"], lock=r["lock"])
    with pytest.raises(ValueError, match="checksum/size mismatch"):
        dataset.verify_dataset(r["data"], lock=r["lock"])
    with pytest.raises(ValueError, match="checksum/size mismatch"):
        dataset.download_dataset(r["data"], r["cache"], lock=r["lock"])
    assert target.read_bytes()[0] == 99


def test_ledger_is_bound_to_hashes_not_only_mutable_receipt(remote):
    r = remote
    dataset.download_dataset(r["data"], r["cache"], lock=r["lock"])
    ledger_path = r["data"] / ".pretrain/verified-files.json"
    ledger = json.loads(ledger_path.read_text())
    ledger["files"][0]["sha256"] = "0" * 64
    ledger_path.write_text(json.dumps(ledger))
    receipt_path = r["data"] / ".pretrain/verification.json"
    receipt = json.loads(receipt_path.read_text())
    receipt["ledger_sha256"] = sha256_file(ledger_path)
    receipt_path.write_text(json.dumps(receipt))
    with pytest.raises(ValueError, match="hashes differ"):
        dataset.assert_training_ready(r["data"], lock=r["lock"])


def test_disk_plan_groups_shared_partition_and_separates_other_partition(tmp_path, monkeypatch):
    monkeypatch.setattr(dataset.shutil, "disk_usage", lambda _: type("Usage", (), {"free": 200})())
    result = dataset._disk_plan(tmp_path / "data", tmp_path / "cache", 100, 100, 10)
    assert len(result["filesystems"]) == 1 and not result["sufficient"]
    assert result["filesystems"][0]["required_bytes"] == 210
    # /dev/shm uses another device on Linux and exercises separate accounting.
    if Path("/dev/shm").is_dir() and Path("/dev/shm").stat().st_dev != tmp_path.stat().st_dev:
        result = dataset._disk_plan(tmp_path / "data", Path("/dev/shm/pbisc-fixture"), 100, 100, 10)
        assert len(result["filesystems"]) == 2 and result["sufficient"]
        assert all(g["required_bytes"] == 110 for g in result["filesystems"])


def test_full_download_refuses_insufficient_disk_before_payload(remote, monkeypatch):
    r = remote
    monkeypatch.setattr(dataset.shutil, "disk_usage", lambda _: type("Usage", (), {"free": 1})())
    with pytest.raises(ValueError, match="insufficient disk space"):
        dataset.download_dataset(r["data"], r["cache"], lock=r["lock"])
    assert r["calls"] == ["release.json"]


def test_unknown_shards_and_symlinks_fail_closed(remote, tmp_path):
    r = remote
    with pytest.raises(ValueError, match="unknown"):
        dataset.download_dataset(r["data"], r["cache"], shard_ids=["missing"], lock=r["lock"])
    r["data"].mkdir(exist_ok=True)
    (r["data"] / "base").symlink_to(tmp_path, target_is_directory=True)
    with pytest.raises(ValueError, match="symlink"):
        dataset.download_dataset(r["data"], r["cache"], shard_ids=["shard0"], lock=r["lock"])


def test_committed_lock_has_immutable_revision_and_safe_totals():
    path = Path(__file__).resolve().parents[1] / "manifests/dataset.lock.json"
    lock = dataset.load_lock(path)
    assert lock["revision"] == "bfe88d1ff4cc1869946cdb7afda11f7927424ae5"
    assert lock["release_asset_count"] == 1346 and lock["count_shards"] == 432
    assert lock["byte_budgets"]["transport_assets"] == 111510563626
    assert len(lock["supplemental_assets"]) == 13
    assert not any(k in path.read_text().lower() for k in ("hf_token", "sample_id", "donor_id"))


def test_lock_rejects_mutable_revision(tmp_path):
    path = tmp_path / "lock.json"
    path.write_text(json.dumps({"schema_version": dataset.LOCK_SCHEMA, "revision": "main"}))
    with pytest.raises(ValueError, match="immutable"):
        dataset.load_lock(path)
