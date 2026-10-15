"""CPU checks for exact expanded-panel identity and immutable recipe contracts."""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pbisc_sdd.expanded_panel import MODE, SCHEMA, validate_expanded_panel
from pbisc_sdd.experiment import require_single_process, resolve_device
from pbisc_sdd.ladder_experiment import validate_resume_config
from test_training_ladder import _run_config


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fixture(tmp_path: Path) -> tuple[dict, dict, Path]:
    axis = tmp_path / "source/metadata/var.parquet"
    axis.parent.mkdir(parents=True)
    pd.DataFrame({"gene_id": ["ENSG0", "ENSG1", "ENSG2", "ENSG3"]}).to_parquet(axis)
    selected = tmp_path / "panels/7k/selected_genes.csv"
    selected.parent.mkdir(parents=True)
    pd.DataFrame({"panel_position": [0, 1, 3], "tensor_index": [0, 1, 2],
                  "gene_id": ["ENSG0", "ENSG1", "ENSG3"]}).to_csv(selected, index=False)
    positions = selected.parent / "gene_positions.npy"
    np.save(positions, np.asarray([0, 1, 3], dtype=np.int64))
    old = tmp_path / "panels/6k/selected_genes.csv"
    old.parent.mkdir(parents=True)
    pd.DataFrame({"panel_position": [1, 3], "gene_id": ["ENSG1", "ENSG3"]}).to_csv(old, index=False)
    old_positions = old.parent / "gene_positions.npy"
    np.save(old_positions, np.asarray([1, 3], dtype=np.int64))
    remap = selected.parent / "old_to_new.csv"
    pd.DataFrame({"gene_id": ["ENSG1", "ENSG3"], "raw_panel_position": [1, 3],
                  "old_tensor_index": [0, 1], "new_tensor_index": [1, 2]}).to_csv(remap, index=False)
    manifest = {
        "schema": SCHEMA, "legacy_loader_compatible_manifest": False,
        "n_selected_genes": 3, "old_n_genes": 2,
        "literature_scope": "endpoint-informed external references",
        "known_limitations": ["candidate selection does not establish biological identity"],
        "selected_genes_path": "/historic/panels/7k/selected_genes.csv", "selected_genes_sha256": _sha(selected),
        "gene_positions_path": "/historic/panels/7k/gene_positions.npy", "gene_positions_sha256": _sha(positions),
        "old_to_new": {"path": "/historic/panels/7k/old_to_new.csv", "sha256": _sha(remap)},
        "source_bindings": [
            {"path": "/historic/source/metadata/var.parquet", "sha256": _sha(axis)},
            {"path": "/historic/tiers/6k/selected_genes.csv", "sha256": _sha(old)},
            {"path": "/historic/tiers/6k/gene_positions.npy", "sha256": _sha(old_positions)},
        ],
    }
    manifest_path = selected.parent / "selection_manifest.json"
    manifest_path.write_text(json.dumps(manifest))
    config = _run_config(tmp_path / "run", epochs=16)
    config["data"].update(n_genes=3, gene_selection=MODE, gene_positions_path=str(positions),
                          gene_positions_sha256=_sha(positions), gene_selection_manifest=str(manifest_path),
                          gene_selection_manifest_sha256=_sha(manifest_path),
                          expanded_panel_contract={"corpus_fingerprint": "f" * 64, "canonical_panel_sha256": _sha(axis)})
    config["e2_data"] = {"gene_panel_path": str(axis)}
    config["_release"] = {"dataset_root": str(tmp_path), "asset_paths": {
        "/historic/panels/7k/selected_genes.csv": "panels/7k/selected_genes.csv",
        "/historic/panels/7k/old_to_new.csv": "panels/7k/old_to_new.csv",
        "/historic/tiers/6k/selected_genes.csv": "panels/6k/selected_genes.csv",
        "/historic/tiers/6k/gene_positions.npy": "panels/6k/gene_positions.npy",
    }}
    return config, manifest, manifest_path


def _reseal(config: dict, manifest: dict, path: Path) -> None:
    path.write_text(json.dumps(manifest))
    config["data"]["gene_selection_manifest_sha256"] = _sha(path)


def test_expanded_panel_binds_gene_ids_positions_corpus_and_original_members(tmp_path):
    config, _, _ = _fixture(tmp_path)
    validate_expanded_panel(config, gene_panel=pd.DataFrame({"gene_id": ["ENSG0", "ENSG1", "ENSG3"]}),
                            corpus_fingerprint="f" * 64)
    with pytest.raises(ValueError, match="corpus fingerprint"):
        validate_expanded_panel(config, corpus_fingerprint="e" * 64)
    with pytest.raises(ValueError, match="runtime Ensembl"):
        validate_expanded_panel(config, gene_panel=pd.DataFrame({"gene_id": ["ENSG1", "ENSG0", "ENSG3"]}))


def test_expanded_panel_rejects_resealed_wrong_ensembl_mapping(tmp_path):
    config, manifest, path = _fixture(tmp_path)
    selected_path = path.parent / "selected_genes.csv"
    selected = pd.read_csv(selected_path)
    selected.loc[0, "gene_id"] = "ENSG2"
    selected.to_csv(selected_path, index=False)
    manifest["selected_genes_sha256"] = _sha(selected_path)
    _reseal(config, manifest, path)
    with pytest.raises(ValueError, match="canonical raw coordinates"):
        validate_expanded_panel(config)


def test_expanded_panel_rejects_resealed_old_member_reordering(tmp_path):
    config, manifest, path = _fixture(tmp_path)
    remap_path = path.parent / "old_to_new.csv"
    remap = pd.read_csv(remap_path)
    remap["new_tensor_index"] = [2, 1]
    remap.to_csv(remap_path, index=False)
    manifest["old_to_new"]["sha256"] = _sha(remap_path)
    _reseal(config, manifest, path)
    with pytest.raises(ValueError, match="original6k preservation"):
        validate_expanded_panel(config)


@pytest.mark.parametrize("coordinates", [[0, 0, 3], [0, 3, 1], [0.0, 1.0, 3.0]])
def test_expanded_panel_rejects_duplicate_unsorted_or_float_positions(tmp_path, coordinates):
    config, manifest, path = _fixture(tmp_path)
    positions = Path(config["data"]["gene_positions_path"])
    np.save(positions, np.asarray(coordinates))
    manifest["gene_positions_sha256"] = config["data"]["gene_positions_sha256"] = _sha(positions)
    _reseal(config, manifest, path)
    with pytest.raises(ValueError, match="ascending integer"):
        validate_expanded_panel(config)


def test_expanded_panel_cannot_claim_legacy_loader_compatibility(tmp_path):
    config, manifest, path = _fixture(tmp_path)
    manifest["legacy_loader_compatible_manifest"] = True
    _reseal(config, manifest, path)
    with pytest.raises(ValueError, match="legacy compatibility"):
        validate_expanded_panel(config)


def test_new_gene_count_cannot_resume_old_model(tmp_path):
    original = _run_config(tmp_path / "run", epochs=16)
    expanded = deepcopy(original)
    expanded["data"]["n_genes"] += 1
    with pytest.raises(ValueError, match="resume configuration differs"):
        validate_resume_config(expanded, original)


@pytest.mark.parametrize("key,value", [("WORLD_SIZE", "2"), ("LOCAL_WORLD_SIZE", "2"),
                                        ("LOCAL_RANK", "0"), ("RANK", "0")])
def test_distributed_launch_fails_before_any_cuda_query(monkeypatch, key, value):
    import torch
    monkeypatch.setenv(key, value)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: pytest.fail("CUDA must not be queried"))
    with pytest.raises(RuntimeError, match="process|torchrun"):
        resolve_device("cuda")


def test_single_process_cpu_has_no_cuda_query(monkeypatch):
    import torch
    for key in ("WORLD_SIZE", "LOCAL_WORLD_SIZE", "LOCAL_RANK", "RANK"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: pytest.fail("CPU must not query CUDA"))
    require_single_process()
    assert resolve_device("cpu") == torch.device("cpu")


def test_four_profiles_preserve_exact_allcell_numeric_stream_contract():
    from pbisc.training.allcell import validate_stream_contract
    root = Path(__file__).resolve().parents[1]
    for name in ("6k-x1", "6k-trunk4x", "7k-trunk4x", "8k-trunk4x"):
        profile_path = root / "configs/training" / f"{name}.json"
        profile = json.loads(profile_path.read_text())
        original = (profile_path.parent / profile["original_config"]).resolve()
        assert _sha(original) == profile["original_config_sha256"]
        config = json.loads(original.read_text())
        assert profile["trainer"] == "allcell"
        assert config["training"]["epochs"] == 16
        assert config["data"]["train_recipes"] == 8192
        validate_stream_contract(config["training"], config["data"])


def test_expanded_schema_is_rejected_by_legacy_mode_during_prepare(tmp_path):
    from pbisc_sdd.ladder_experiment import load_experiment_config
    config, _, _ = _fixture(tmp_path)
    config["data"]["gene_selection"] = "explicit_train_only_hvg_manifest"
    path = tmp_path / "masquerading.json"
    path.write_text(json.dumps(config))
    with pytest.raises(ValueError, match="legacy HVG selection"):
        load_experiment_config(path)
