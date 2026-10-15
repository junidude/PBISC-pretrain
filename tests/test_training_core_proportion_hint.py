"""CPU tests for the optional cell-type proportion hint (pbisc_sdd.proportion_hint)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from pbisc_sdd import experiment as stable
from pbisc_sdd.data import SparseValueScaler
from pbisc_sdd.proportion_hint import ProportionHint, ProportionHintConfig, load_hint_csv

LABELS = ["T cell", "B cell", "Monocyte"]


def _fixture(tmp_path: Path):
    tmp_path.mkdir(parents=True, exist_ok=True)
    table = pd.DataFrame(
        {
            "sample_key": ["s1", "s2", "s3"],
            "split": ["train", "train", "val"],
            "prop_l1::T cell": [0.6, 0.2, 0.5],
            "prop_l1::B cell": [0.3, 0.2, 0.25],
            "prop_l1::Monocyte": [0.1, 0.6, 0.25],
        }
    )
    table.to_parquet(tmp_path / "props.parquet", index=False)
    recipes = tmp_path / "recipes"
    recipes.mkdir()
    pd.DataFrame({"block_id": ["b1", "b2"], "sample_key": ["s1", "s2"]}).to_parquet(recipes / "train.parquet", index=False)
    pd.DataFrame({"block_id": ["b3"], "sample_key": ["s3"]}).to_parquet(recipes / "val.parquet", index=False)
    pd.DataFrame({"block_id": ["t1"], "sample_key": ["s2"], "k": [5]}).to_csv(tmp_path / "topup.csv", index=False)
    return {
        "table_parquet": str(tmp_path / "props.parquet"),
        "labels": LABELS,
        "recipe_sources": [str(recipes), str(tmp_path / "topup.csv")],
        "fallback": "train_mean",
        "max_unknown_fraction": 0.0,
    }


def test_lookup_and_contract(tmp_path):
    hint = ProportionHint(ProportionHintConfig.from_mapping(_fixture(tmp_path)))
    assert hint.dim == 3 and hint.labels == tuple(LABELS)
    vec = hint.vectors_for(["b1", "t1", "b3"])
    assert vec.shape == (3, 3) and vec.dtype == np.float32
    np.testing.assert_allclose(vec[0], [0.6, 0.3, 0.1], atol=1e-6)
    np.testing.assert_allclose(vec[1], [0.2, 0.2, 0.6], atol=1e-6)  # top-up block -> s2
    np.testing.assert_allclose(vec[2], [0.5, 0.25, 0.25], atol=1e-6)
    np.testing.assert_allclose(hint.fallback, [0.4, 0.25, 0.35], atol=1e-6)  # train mean
    contract = hint.contract()
    assert contract["dim"] == 3 and contract["labels"] == LABELS and contract["n_blocks_mapped"] == 4
    json.dumps(contract)
    report = hint.verify_coverage(["b1", "b2", "b3", "t1"], ["s1", "s2", "s3"])
    assert report["unknown_blocks"] == 0 and report["samples_without_proportions"] == 0


def test_unknown_block_fails_closed(tmp_path):
    hint = ProportionHint(ProportionHintConfig.from_mapping(_fixture(tmp_path)))
    with pytest.raises(RuntimeError):
        hint.verify_coverage(["b1", "zzz"], ["s1"])
    # streaming rows: fallback is used but the fraction check trips once >= 4096 rows were seen
    with pytest.raises(RuntimeError):
        for _ in range(100):
            hint.vectors_for(["zzz"] * 64)


def test_prepare_batch_appends_hint(tmp_path):
    hint = ProportionHint(ProportionHintConfig.from_mapping(_fixture(tmp_path)))
    g = 4
    batch = {
        "x_log": torch.zeros(2, g),
        "condition": torch.ones(2, g + 6),
        "availability_mask": torch.ones(2, g),
        "bulk_measurement_fraction": torch.ones(2, g),
        "block_id": ["b1", "b2"],
    }
    scaler = SparseValueScaler(max_log_value=8.0)
    stable.set_active_proportion_hint(None)
    try:
        _, cond0, _, _ = stable._prepare_batch(batch, device=torch.device("cpu"), scaler=scaler)
        stable.set_active_proportion_hint(hint)
        _, cond1, _, _ = stable._prepare_batch(batch, device=torch.device("cpu"), scaler=scaler)
    finally:
        stable.set_active_proportion_hint(None)
    assert cond0.shape == (2, g + 6 + 2 * g)
    assert cond1.shape == (2, g + 6 + 3 + 2 * g)
    torch.testing.assert_close(cond1[:, : g + 6], cond0[:, : g + 6])
    np.testing.assert_allclose(cond1[0, g + 6 : g + 9].numpy(), [0.6, 0.3, 0.1], atol=1e-6)
    torch.testing.assert_close(cond1[:, g + 9 :], cond0[:, g + 6 :])


def test_load_hint_csv_contract(tmp_path):
    csv = tmp_path / "hint.csv"
    pd.DataFrame({"sample_id": ["d2", "d1"], "T cell": [0.5, 0.7], "B cell": [0.3, 0.2], "Monocyte": [0.2, 0.1]}).to_csv(csv, index=False)
    values = load_hint_csv(csv, LABELS, sample_ids=["d1", "d2"])
    np.testing.assert_allclose(values, [[0.7, 0.2, 0.1], [0.5, 0.3, 0.2]], atol=1e-6)
    with pytest.raises(ValueError):
        load_hint_csv(csv, LABELS, sample_ids=["d1", "d3"])
    with pytest.raises(ValueError):
        load_hint_csv(csv, LABELS + ["DC"], sample_ids=["d1"])


# ---------------------------------------------------------------------------
# Trainer integration: checkpoint contract, resume binding, restore refusal
# ---------------------------------------------------------------------------


def _hint_streams(config, **kwargs):
    from test_training_ladder import _EpochDataset, _sample, _tiny_streams

    class _BlockDataset(_EpochDataset):
        def __getitem__(self, index: int) -> dict[str, object]:
            row = _sample(self.values[index])
            row["block_id"] = ["b1", "b2", "t1"][index % 3]
            return row

    streams = _tiny_streams(config, **kwargs)
    streams.train = _BlockDataset([0.25, 0.5, 0.75])
    streams.validation = _BlockDataset([0.5])
    return streams


def test_run_training_records_the_hint_and_binds_resume(tmp_path):
    from test_training_ladder import _run_config

    from pbisc_sdd.ladder_experiment import (
        _validate_hint_binding,
        load_checkpoint,
        restore_for_sampling,
        run_training,
    )

    block = _fixture(tmp_path / "hint")
    out = tmp_path / "run"
    cfg = _run_config(out, epochs=1)
    cfg["data"]["proportion_hint"] = block
    first = run_training(cfg, _stream_builder=_hint_streams)
    assert stable.ACTIVE_PROPORTION_HINT is None  # reset after the run
    payload = load_checkpoint(out / "checkpoint_latest.pt", device=torch.device("cpu"))
    contract = payload["proportion_hint"]
    assert contract["labels"] == LABELS and contract["dim"] == 3
    assert payload["model_config"]["condition_dim"] == 1 + 3 + 2 * 1  # base 1 + hint 3 + 2G

    cfg2 = _run_config(out, epochs=2)
    cfg2["data"]["proportion_hint"] = block
    resumed = run_training(cfg2, resume_checkpoint=out / "checkpoint_latest.pt", _stream_builder=_hint_streams)
    assert resumed["global_step"] > first["global_step"]

    with pytest.raises(ValueError, match="proportion hint mismatch"):
        _validate_hint_binding(payload, None)
    other = dict(block, table_parquet=str(tmp_path / "other.parquet"))
    pd.read_parquet(block["table_parquet"]).assign(**{"prop_l1::T cell": [0.5, 0.5, 0.5]}).to_parquet(other["table_parquet"], index=False)
    with pytest.raises(ValueError, match="table_sha256"):
        _validate_hint_binding(payload, ProportionHint(ProportionHintConfig.from_mapping(other)))
    with pytest.raises(ValueError, match="proportion hint"):
        restore_for_sampling(out / "checkpoint_latest.pt", device=torch.device("cpu"))

    plain = _run_config(tmp_path / "plain", epochs=1)
    run_training(plain, _stream_builder=_hint_streams)
    plain_payload = load_checkpoint(tmp_path / "plain" / "checkpoint_latest.pt", device=torch.device("cpu"))
    assert plain_payload["proportion_hint"] is None
    assert plain_payload["model_config"]["condition_dim"] == 1 + 2 * 1
