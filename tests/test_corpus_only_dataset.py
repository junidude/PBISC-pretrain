from __future__ import annotations

import json

import pytest

from pbisc_pretrain import dataset
import test_pretrain_dataset

remote = test_pretrain_dataset.remote


def test_corpus_download_uses_no_external_panel_assets(remote):
    r = remote
    (r["root"] / "supplements/panel.npy").unlink()
    receipt = dataset.download_dataset(r["data"], r["cache"], lock=r["lock"], corpus_only=True)
    assert receipt["status"] == "TRAIN_READY"
    assert receipt["data_scope"] == "canonical_corpus"
    assert not any("supplements/" in path for path in r["calls"])
    assert not (r["data"] / "relocation.pretrain.json").exists()
    assert not (r["data"] / "panels/6k/gene_positions.npy").exists()
    assert dataset.assert_training_ready(r["data"], lock=r["lock"], corpus_only=True) == receipt
    with pytest.raises(ValueError, match="not TRAIN_READY"):
        dataset.assert_training_ready(r["data"], lock=r["lock"])
    with pytest.raises(FileNotFoundError):
        dataset.verify_dataset(r["data"], lock=r["lock"])


def test_corpus_and_legacy_receipts_coexist(remote):
    r = remote
    legacy = dataset.download_dataset(r["data"], r["cache"], lock=r["lock"])
    assert dataset.download_dataset(r["data"], r["cache"], lock=r["lock"], corpus_only=True)["status"] == "TRAIN_READY"
    assert dataset.assert_training_ready(r["data"], lock=r["lock"]) == legacy
    assert dataset.assert_training_ready(r["data"], lock=r["lock"], corpus_only=True)["data_scope"] == "canonical_corpus"
    # Explicitly changing the scope cannot repurpose an otherwise valid ledger.
    path = r["data"] / ".pretrain/corpus-verification.json"
    value = json.loads(path.read_text())
    value["data_scope"] = "legacy_with_panels"
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="scope differs"):
        dataset.assert_training_ready(r["data"], lock=r["lock"], corpus_only=True)


def test_subset_corpus_never_authorizes_training(remote):
    r = remote
    receipt = dataset.download_dataset(r["data"], r["cache"], lock=r["lock"],
                                       corpus_only=True, shard_ids=["shard0"])
    assert receipt["status"] == "SUBSET_READY"
    with pytest.raises(ValueError, match="partial/metadata-only"):
        dataset.assert_training_ready(r["data"], lock=r["lock"], corpus_only=True)


def test_changed_corpus_bytes_require_new_hash_verification(remote):
    r = remote
    dataset.download_dataset(r["data"], r["cache"], lock=r["lock"], corpus_only=True)
    target = r["data"] / "base/counts/shard0.zarr/data/0"
    target.write_bytes(b"broken")
    with pytest.raises(ValueError, match="changed"):
        dataset.assert_training_ready(r["data"], lock=r["lock"], corpus_only=True)
    with pytest.raises(ValueError, match="checksum/size mismatch"):
        dataset.verify_dataset(r["data"], lock=r["lock"], corpus_only=True)


def test_corpus_plan_excludes_legacy_supplemental_bytes(remote):
    r = remote
    old = dataset.plan_dataset(r["data"], r["cache"], lock=r["lock"])
    new = dataset.plan_dataset(r["data"], r["cache"], lock=r["lock"], corpus_only=True)
    extra = sum(a["size_bytes"] for a in r["lock"]["supplemental_assets"])
    assert new["cache_bytes"] == old["cache_bytes"] - extra
    assert new["restored_bytes"] == old["restored_bytes"] - extra
