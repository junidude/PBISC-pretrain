from copy import deepcopy
import hashlib
import json
from pathlib import Path

import pytest

from pbisc.training.portable import canonical_resume_config, prepare_config, resolve_artifact
from pbisc_sdd.ladder_experiment import validate_resume_config
from test_training_ladder import _run_config


def test_relocated_resume_preserves_all_numeric_contracts(tmp_path):
    original = _run_config(tmp_path / "original", epochs=1)
    current = deepcopy(original)
    old = current["paths"]["output_dir"]
    current["paths"]["output_dir"] = str(tmp_path / "restored")
    current["_release"] = {"original_path_values": {"paths.output_dir": old}}
    validate_resume_config(current, original)
    current["training"]["lr"] *= 2
    with pytest.raises(ValueError, match="resume configuration differs"):
        validate_resume_config(current, original)


def test_path_provenance_cannot_override_model_or_training(tmp_path):
    original = _run_config(tmp_path / "run", epochs=1)
    for dotted, value in (("training.lr", 0.5), ("model.hidden_dims", [4])):
        bad = deepcopy(original)
        bad["_release"] = {"original_path_values": {dotted: value}}
        with pytest.raises(ValueError, match="only path fields"):
            canonical_resume_config(bad)
        path = tmp_path / "bad.json"
        path.write_text(json.dumps(bad))
        with pytest.raises(ValueError, match="only path fields"):
            prepare_config(path)


def test_artifact_mapping_is_explicit_and_contained(tmp_path):
    cfg = {"_release": {"dataset_root": str(tmp_path), "asset_paths": {"/old/a": "data/a"}}}
    assert resolve_artifact("/old/a", cfg) == tmp_path / "data/a"
    with pytest.raises(ValueError, match="does not bind"):
        resolve_artifact("/unmapped/a", cfg)
    cfg["_release"]["asset_paths"]["/old/a"] = "../outside"
    with pytest.raises(ValueError, match="escapes"):
        resolve_artifact("/old/a", cfg)


def test_model_math_files_match_extraction_source_hashes():
    root = Path(__file__).resolve().parents[1]
    manifest = json.loads((root / "provenance/core_sources.json").read_text())
    for name in manifest["unchanged_math_modules"]:
        actual = hashlib.sha256((root / name).read_bytes()).hexdigest()
        assert actual == manifest["files"][name]["original_sha256"]


def test_allcell_resume_dispatches_bounded_runner_without_old_worker_proofs(tmp_path, monkeypatch):
    from pbisc import training
    from pbisc.training import allcell, allcell_roundrobin
    from pbisc_sdd import ladder_experiment
    config = _run_config(tmp_path / "restored-run", epochs=16)
    config["e2_data"] = {}
    seen = []
    monkeypatch.setattr(training, "prepare_config", lambda *a, **k: (config, "allcell"))
    monkeypatch.setattr(ladder_experiment, "load_experiment_config", lambda p: config)
    monkeypatch.setattr(allcell_roundrobin, "main", lambda args: seen.extend(args))
    monkeypatch.setattr(allcell, "main", lambda args: pytest.fail("requires unavailable historical proof sidecars"))
    resume = tmp_path / "restored-run/checkpoint_latest.pt"
    assert training.main(["--config", "unused.json", "--resume", str(resume)]) == 0
    assert seen[seen.index("--resume") + 1] == str(resume)
    assert seen[seen.index("--stop-after-epoch") + 1] == "16"


def test_release_profile_requires_new_output_before_reading_original_paths(tmp_path):
    profile = tmp_path / "profile.json"
    profile.write_text(json.dumps({"schema_version": "pbisc.release_training_profile.v1"}))
    with pytest.raises(ValueError, match="explicit output_dir"):
        prepare_config(profile)
    assert list(tmp_path.iterdir()) == [profile]


def test_relocation_extension_must_preserve_original_map_and_frozen_hash(tmp_path):
    config = _run_config(tmp_path / "run", epochs=1)
    config["e2_data"] = {}
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config))
    relocation_path = tmp_path / "relocation.json"
    relocation = {"schema_version": "pbisc.dataset_relocation.v1", "asset_paths": {"/old/file": "kept"},
                  "relative_roots": {key: key for key in ("source_root", "base_source_root", "index_root", "gene_panel_path")}}
    relocation_path.write_text(json.dumps(relocation))
    extension_path = tmp_path / "relocation.pretrain.json"
    extension = {"schema_version": "pbisc.pretrain_dataset_relocation.v1", "original_relocation_sha256": "0" * 64,
                 "asset_paths": {"/old/file": "kept"}}
    extension_path.write_text(json.dumps(extension))
    with pytest.raises(ValueError, match="does not bind the original"):
        prepare_config(config_path, dataset_root=tmp_path)
    extension["original_relocation_sha256"] = hashlib.sha256(relocation_path.read_bytes()).hexdigest()
    extension["asset_paths"]["/old/file"] = "changed"
    extension_path.write_text(json.dumps(extension))
    with pytest.raises(ValueError, match="changed an original artifact"):
        prepare_config(config_path, dataset_root=tmp_path)
