"""Requirements for the requested 3 panels × 5 independent architecture settings."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_requested_fifteen_configs_have_exactly_the_declared_single_variable_changes():
    matrix = json.loads((ROOT / "configs/matrix.json").read_text())
    experiments = matrix["experiments"]
    assert len(experiments) == len({e["id"] for e in experiments}) == 15
    expected = {(n, variant) for n in (7000, 8000, 10000)
                for variant in ("1x", "trunk2x", "trunk4x", "ctx2x", "ctx4x")}
    assert {(e["n_genes"], e["id"].split("-", 1)[1]) for e in experiments} == expected
    base_paths = set()
    for entry in experiments:
        path = ROOT / entry["profile"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == entry["profile_sha256"]
        profile = json.loads(path.read_text())
        base_path = (path.parent / profile["base_config"]).resolve()
        base_paths.add(base_path)
        assert hashlib.sha256(base_path.read_bytes()).hexdigest() == profile["base_config_sha256"]
        overrides = profile["overrides"]
        tm, cm = entry["trunk_multiplier"], entry["context_multiplier"]
        assert tm == 1 or cm == 1
        assert overrides["model"] == {"hidden_dims": [v * tm for v in (1024, 1024, 512, 256)],
                                      "context_dim": 512 * cm}
        assert overrides["data"] == {"n_genes": entry["n_genes"]}
        assert set(overrides) == {"experiment_id", "data", "model"}
        base = json.loads(base_path.read_text())
        assert base["model"]["time_embedding_dim"] == 128
        assert base["training"]["seed"] == 20260627
        assert base["training"]["epochs"] == 16
        assert base["training"]["batch_size"] == 256
        assert base["training"]["num_workers"] == 4
        assert base["training"]["gradient_accumulation_steps"] == 1
        assert base["data"]["train_recipes"] == 8192
        assert base["training"]["max_cuda_memory_gib"] == 24
    assert len(base_paths) == 1


def test_new_profile_and_base_configs_contain_no_absolute_paths():
    paths = [ROOT / "configs/base/allcell-v2.json", *sorted((ROOT / "configs/matrix").glob("*.json"))]

    def check(value):
        if isinstance(value, dict):
            for item in value.values():
                check(item)
        elif isinstance(value, list):
            for item in value:
                check(item)
        elif isinstance(value, str):
            assert not value.startswith("/"), value
            assert "/home/sj_server" not in value

    for path in paths:
        check(json.loads(path.read_text()))
