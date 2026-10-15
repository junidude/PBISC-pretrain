"""CPU-only portability checks using the actual fifteen matrix profiles.

Corpus fixtures contain metadata only. Their raw-axis serialization is sealed
locally; no count archive, original host path, CUDA, or network is consulted.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from pbisc.training.portable import prepare_config, resolve_artifact
from pbisc_sdd.collection_panel import MODE, validate_collection_panel
from pbisc_sdd.ladder_experiment import load_experiment_config, validate_resume_config

ROOT = Path(__file__).resolve().parents[1]
SIZES = ("7k", "8k", "10k")
VARIANTS = ("1x", "trunk2x", "trunk4x", "ctx2x", "ctx4x")


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n")


def _layout(tmp_path: Path, layout: str = "checkout") -> tuple[Path, Path]:
    assets = tmp_path / {"checkout": "repository", "relocated": "deep/project/tree",
                         "wheel": "environment/share/pbisc-pretrain"}[layout]
    shutil.copytree(ROOT / "configs", assets / "configs")
    shutil.copytree(ROOT / "gene_context", assets / "gene_context")
    data = tmp_path / "separate-restored-corpus"
    axis_path = data / "source/metadata/var.parquet"
    axis_path.parent.mkdir(parents=True)
    universe = pd.read_csv(assets / "gene_context/inputs/gene_universe.csv")
    universe[["gene_id", "gene_symbol"]].to_parquet(axis_path, index=False)
    axis_contract = assets / "gene_context/inputs/raw_axis_contract.json"
    contract = json.loads(axis_contract.read_text())
    contract["raw_var_sha256"] = _sha(axis_path)
    _json(axis_contract, contract)
    profiles = list((assets / "configs/matrix").glob("*.json"))
    assert len(profiles) == 15
    for path in profiles:
        profile = json.loads(path.read_text())
        profile["raw_axis_contract_sha256"] = _sha(axis_contract)
        _json(path, profile)
    original_hashes = {}
    for key, relative in (("source_manifest", "source/dataset_manifest.json"),
                          ("source_complete", "source/BUILD_COMPLETE.json"),
                          ("index_manifest", "index/dataset_manifest.json")):
        artifact = data / relative
        _json(artifact, {"fixture": "frozen provenance", "role": key})
        original_hashes[key] = _sha(artifact)
    relocation = {
        "schema_version": "pbisc.dataset_relocation.v1",
        "original_roots": {"source_root": "/historical/source", "base_source_root": "/historical/base",
                           "index_root": "/historical/index", "gene_panel_path": "/historical/source/metadata/var.parquet"},
        "relative_roots": {"source_root": "source", "base_source_root": "base",
                           "index_root": "index", "gene_panel_path": "source/metadata/var.parquet"},
        "original_sha256": original_hashes,
        "asset_paths": {"/historical/topup/payload.json": "topup/payload.json"},
    }
    _json(data / "relocation.json", relocation)
    for relative in ("base", "conditioner", "topup", "provenance/coverage_v1"):
        (data / relative).mkdir(parents=True, exist_ok=True)
    return assets, data


@pytest.mark.parametrize("layout", ("checkout", "relocated", "wheel"))
def test_all_fifteen_profiles_prepare_and_validate_without_cwd_or_remote_panels(tmp_path, monkeypatch, layout):
    assets, data = _layout(tmp_path, layout)
    arbitrary_cwd = tmp_path / "unrelated-working-directory"
    arbitrary_cwd.mkdir()
    monkeypatch.chdir(arbitrary_cwd)
    from model7_no_annotation.data_e2 import E2FullDataConfig, _validated_relocation
    from pbisc.training.allcell import validate_stream_contract
    for size in SIZES:
        for variant in VARIANTS:
            path = assets / f"configs/matrix/{size}-{variant}.json"
            config, trainer = prepare_config(path, dataset_root=data, output_dir=tmp_path / f"runs/{size}-{variant}")
            assert trainer == "allcell"
            assert config["data"]["gene_selection"] == MODE
            assert config["data"]["n_genes"] == int(size[:-1]) * 1000
            assert Path(config["data"]["gene_positions_path"]).is_relative_to(assets / "gene_context")
            assert Path(config["data"]["gene_selection_manifest"]).is_relative_to(assets / "gene_context")
            assert Path(config["paths"]["conditioner_cache"]) == data / "conditioner/conditioner_cache.pt"
            assert Path(config["paths"]["model7_root"]) == data
            assert resolve_artifact("/historical/topup/payload.json", config) == data / "topup/payload.json"
            assert not (data / "panels").exists()
            _validated_relocation(E2FullDataConfig(**config["e2_data"]))
            validate_stream_contract(config["training"], config["data"])
            materialized = tmp_path / "materialized.json"
            _json(materialized, config)
            assert load_experiment_config(materialized)["model"] == config["model"]


def test_matrix_paths_relocate_but_numeric_resume_stays_strict(tmp_path):
    first_assets, first_data = _layout(tmp_path / "first")
    second_assets, second_data = _layout(tmp_path / "second", "wheel")
    name = "configs/matrix/8k-trunk4x.json"
    first, _ = prepare_config(first_assets / name, dataset_root=first_data, output_dir=tmp_path / "run-one")
    second, _ = prepare_config(second_assets / name, dataset_root=second_data, output_dir=tmp_path / "run-two")
    validate_resume_config(second, first)
    for section, key, value in (("training", "lr", 0.1), ("training", "batch_size", 128),
                                ("data", "train_cells_per_block", 3999),
                                ("model", "hidden_dims", [1024])):
        changed = deepcopy(second)
        changed[section][key] = value
        with pytest.raises(ValueError, match="resume configuration differs"):
            validate_resume_config(changed, first)


@pytest.mark.parametrize("field,value", (("panel_manifest", "../../outside.json"),
                                          ("base_config", "../../../outside.json"),
                                          ("panel_manifest", "/host-specific/panel.json"),
                                          ("asset_root", "/host-specific")))
def test_matrix_profile_rejects_absolute_paths_and_traversal_before_writes(tmp_path, field, value):
    assets, data = _layout(tmp_path)
    path = assets / "configs/matrix/7k-1x.json"
    profile = json.loads(path.read_text())
    profile[field] = value
    _json(path, profile)
    output = tmp_path / "must-not-be-created"
    with pytest.raises(ValueError, match="relative|escapes"):
        prepare_config(path, dataset_root=data, output_dir=output)
    assert not output.exists()


def test_collection_rejects_changed_gene_hash_and_raw_axis(tmp_path):
    assets, data = _layout(tmp_path)
    config, _ = prepare_config(assets / "configs/matrix/10k-trunk4x.json", dataset_root=data, output_dir=tmp_path / "run")
    positions = Path(config["data"]["gene_positions_path"])
    original = positions.read_bytes()
    changed = np.load(positions, allow_pickle=False)
    changed[0] += 1
    np.save(positions, changed, allow_pickle=False)
    with pytest.raises(ValueError, match="SHA-256"):
        validate_collection_panel(config)
    positions.write_bytes(original)
    axis = data / "source/metadata/var.parquet"
    table = pd.read_parquet(axis)
    table.loc[0, "gene_id"] = "ENSG_INVALID"
    table.to_parquet(axis, index=False)
    with pytest.raises(ValueError, match="full raw gene axis"):
        validate_collection_panel(config)


def test_collection_map_cannot_escape_asset_root_or_fall_back_to_corpus(tmp_path):
    assets, data = _layout(tmp_path)
    config, _ = prepare_config(assets / "configs/matrix/7k-1x.json", dataset_root=data, output_dir=tmp_path / "run")
    config["_release"]["profile_asset_paths"]["panels/7k/selected_genes.csv"] = "../outside.csv"
    with pytest.raises(ValueError, match="escapes"):
        validate_collection_panel(config)
    del config["_release"]["profile_asset_paths"]["panels/7k/selected_genes.csv"]
    with pytest.raises(ValueError, match="not bound"):
        validate_collection_panel(config)


def test_profile_cannot_hide_numeric_change_in_relocation_provenance(tmp_path):
    assets, data = _layout(tmp_path)
    path = assets / "configs/matrix/7k-1x.json"
    profile = json.loads(path.read_text())
    profile["overrides"]["training"] = {"batch_size": 128}
    _json(path, profile)
    with pytest.raises(ValueError, match="unsupported matrix profile override"):
        prepare_config(path, dataset_root=data, output_dir=tmp_path / "run")


def test_collection_runtime_dispatch_binds_corpus_and_full_axis_before_selection(tmp_path, monkeypatch):
    from pbisc_sdd import experiment

    assets, data = _layout(tmp_path)
    config, _ = prepare_config(assets / "configs/matrix/10k-1x.json", dataset_root=data, output_dir=tmp_path / "run")
    selected = pd.read_csv(assets / "gene_context/panels/10k/selected_genes.csv")
    fingerprint = config["data"]["gene_context_collection_contract"]["corpus_fingerprint"]
    streams = SimpleNamespace(data_module=SimpleNamespace(gene_panel=selected, corpus_fingerprint=fingerprint))

    def build(**kwargs):
        assert kwargs["n_genes"] == 10000
        assert kwargs["data_config"]["gene_panel_path"] == str(data / "source/metadata/var.parquet")
        assert len(pd.read_parquet(kwargs["data_config"]["gene_panel_path"])) == 20097
        assert kwargs["conditioner_cache"] == data / "conditioner/conditioner_cache.pt"
        return streams

    monkeypatch.setattr(experiment, "build_pbisc_e2_streams", build)
    assert experiment.build_streams_from_config(config) is streams
    streams.data_module.corpus_fingerprint = "0" * 64
    with pytest.raises(ValueError, match="runtime corpus fingerprint"):
        experiment.build_streams_from_config(config)


def test_collection_schema_cannot_masquerade_as_legacy_hvg_selection(tmp_path):
    assets, data = _layout(tmp_path)
    config, _ = prepare_config(assets / "configs/matrix/7k-1x.json", dataset_root=data, output_dir=tmp_path / "run")
    config["data"]["gene_selection"] = "explicit_train_only_hvg_manifest"
    path = tmp_path / "invalid-mode.json"
    _json(path, config)
    with pytest.raises(ValueError, match="legacy HVG selection"):
        load_experiment_config(path)


def test_cross_tier_evidence_and_resealed_index_cannot_bypass_profile_pin(tmp_path):
    assets, data = _layout(tmp_path)
    path = assets / "configs/matrix/7k-1x.json"
    config, _ = prepare_config(path, dataset_root=data, output_dir=tmp_path / "run")
    index_path = assets / "gene_context/MANIFEST.json"
    index = json.loads(index_path.read_text())
    changed_paths = {"panels/10k/selected_genes.csv", "panels/10k/additions.csv"}
    changed_gene = None
    for relative in sorted(changed_paths):
        artifact = assets / "gene_context" / relative
        table = pd.read_csv(artifact)
        if changed_gene is None:
            changed_gene = table.loc[table["evidence_primary_source_url"].notna(), "gene_id"].iloc[0]
        table.loc[table["gene_id"] == changed_gene, "evidence_primary_source_url"] = "https://example.invalid/altered-evidence"
        table.to_csv(artifact, index=False)
        record = next(record for record in index["files"] if record["path"] == relative)
        record.update(sha256=_sha(artifact), bytes=artifact.stat().st_size)
    _json(index_path, index)
    with pytest.raises(ValueError, match="collection index"):
        validate_collection_panel(config)
    with pytest.raises(ValueError, match="collection index hash differs from profile"):
        prepare_config(path, dataset_root=data, output_dir=tmp_path / "new-run")
    assert not (tmp_path / "new-run").exists()


def test_matrix_profile_ignores_legacy_supplemental_relocation_extension(tmp_path):
    assets, data = _layout(tmp_path)
    # Matrix profiles need only the original corpus map. An unrelated legacy
    # overlay can be absent or stale without making vendored panels remote.
    _json(data / "relocation.pretrain.json", {"schema_version": "obsolete-legacy-overlay"})
    config, _ = prepare_config(assets / "configs/matrix/7k-1x.json", dataset_root=data, output_dir=tmp_path / "run")
    assert "pretrain_relocation_sha256" not in config["_release"]
    validate_collection_panel(config)


def test_actual_v2_preparation_does_not_import_torch_in_controller_process(tmp_path):
    assets, data = _layout(tmp_path)
    code = """
import json
import sys
from pbisc.training.portable import prepare_config
assert 'torch' not in sys.modules
config, trainer = prepare_config(sys.argv[1], dataset_root=sys.argv[2], output_dir=sys.argv[3])
assert 'torch' not in sys.modules
print(json.dumps({'trainer': trainer, 'n_genes': config['data']['n_genes']}))
"""
    unrelated = tmp_path / "unrelated-controller-cwd"
    unrelated.mkdir()
    result = subprocess.run(
        [sys.executable, "-c", code, str(assets / "configs/matrix/10k-trunk4x.json"),
         str(data), str(tmp_path / "child-owned-output")],
        cwd=unrelated, text=True, capture_output=True, check=True,
    )
    assert json.loads(result.stdout) == {"trainer": "allcell", "n_genes": 10000}
