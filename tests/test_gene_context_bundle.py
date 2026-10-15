from __future__ import annotations

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest


ROOT = Path(__file__).resolve().parents[1]
BUNDLE = ROOT / "gene_context"
EXPECTED_POSITIONS = {
    6: "653ffa45ac519000baa41086c52408009391d6e68cfb218bc6a50d34163fcb24",
    7: "f4bda0b94e2f5d6fcc7e44e34765a1b77b31dce85de49925bc95b3906ee91e10",
    8: "a6d8acb7ef27ae1c2733d50072e2d21cdb54b20a38910a5d10c4c8f9d72305d8",
    10: "14bfe526b4bb4b096db060b4b1a683ccfa30f28e2739c05d3a7342e1d0f47c82",
}


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _verifier():
    spec = importlib.util.spec_from_file_location("bundled_context_verifier", BUNDLE / "verify_collection.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_gene_context_bundle_seals_all_payload_and_source_exact_copies():
    assert (BUNDLE / "BUNDLE_MANIFEST.json").is_file(), "gene-context payload must be bundled"
    manifest = json.loads((BUNDLE / "BUNDLE_MANIFEST.json").read_text())
    expected_paths = {
        p.relative_to(BUNDLE).as_posix()
        for p in BUNDLE.rglob("*")
        if p.is_file() and "__pycache__" not in p.parts and p.name != "BUNDLE_MANIFEST.json"
    }
    assert {item["path"] for item in manifest["files"]} == expected_paths
    for item in manifest["files"]:
        relative = Path(item["path"])
        assert not relative.is_absolute() and ".." not in relative.parts
        path = BUNDLE / relative
        assert not path.is_symlink()
        assert path.stat().st_size == item["bytes"]
        assert _sha(path) == item["sha256"], relative
    provenance = json.loads((ROOT / "provenance/gene_context_bundle.json").read_text())
    for item in provenance["source_exact_files"]:
        assert _sha(ROOT / item["path"]) == item["sha256"]
    assert _sha(BUNDLE / "build_collection.py") == (
        "137ca4bf62a772a4f7fecf56ca63a0ea565d0576006734449abcf8ec77bdc5cf"
    )


def test_gene_context_identity_raw_coordinates_txt_order_and_nesting():
    universe = pd.read_csv(BUNDLE / "inputs/gene_universe.csv")
    assert universe.gene_id.is_unique and len(universe) == 20097
    assert universe.panel_position.tolist() == list(range(20097))
    previous = set()
    for k in (6, 7, 8, 10):
        folder = BUNDLE / "panels" / f"{k}k"
        table = pd.read_csv(folder / "selected_genes.csv")
        positions = np.load(folder / "gene_positions.npy", allow_pickle=False)
        assert len(table) == table.gene_id.nunique() == k * 1000
        assert positions.dtype == np.int64 and np.all(np.diff(positions) > 0)
        assert positions.tolist() == table.panel_position.tolist()
        assert universe.iloc[positions].gene_id.tolist() == table.gene_id.tolist()
        assert (folder / "gene_ids.txt").read_text().splitlines() == table.gene_id.tolist()
        assert (folder / "gene_symbols.txt").read_text().splitlines() == table.gene_symbol.tolist()
        assert _sha(folder / "gene_positions.npy") == EXPECTED_POSITIONS[k]
        selected = set(table.gene_id)
        assert previous < selected
        previous = selected
    assert table.gene_symbol.nunique() == 9985


def test_gene_context_frozen_recipe_and_original6k_tensor_mapping():
    universe = pd.read_csv(BUNDLE / "inputs/gene_universe.csv")
    priority = pd.read_csv(BUNDLE / "inputs/literature_addition_order.csv")
    ranked = np.load(BUNDLE / "inputs/ranked_gene_positions.npy", allow_pickle=False)
    assert ranked.tolist() == universe.sort_values("universal_rank").panel_position.tolist()
    assert len(priority) == priority.gene_id.nunique() == 2091
    assert priority.addition_priority_rank.tolist() == list(range(1, 2092))
    panels = {k: pd.read_csv(BUNDLE / "panels" / f"{k}k/selected_genes.csv") for k in (6, 7, 8, 10)}
    base = set(panels[6].gene_id)
    assert not base.intersection(priority.gene_id)
    for k in (7, 8):
        assert set(panels[k].gene_id) == base.union(priority.head(k * 1000 - 6000).gene_id)
    eligible = universe[universe.eligible & universe.measured_cells.gt(0)].sort_values("universal_rank")
    reserved = base.union(priority.gene_id)
    backfill = [gene for gene in eligible.gene_id if gene not in reserved][:1909]
    assert set(panels[10].gene_id) == base.union(priority.gene_id, backfill)
    assert panels[10].selection_reason.value_counts().to_dict() == {
        "preserved_final6k": 6000,
        "frozen_primary_literature_curated": 2091,
        "train_only_HVG_ranked_backfill_not_literature_curated": 1909,
    }
    additions_from8 = panels[10][~panels[10].gene_id.isin(panels[8].gene_id)]
    assert additions_from8.selection_reason.value_counts().to_dict() == {
        "frozen_primary_literature_curated": 91,
        "train_only_HVG_ranked_backfill_not_literature_curated": 1909,
    }
    mapping = pd.read_csv(BUNDLE / "panels/10k/old6k_to_10k.csv")
    assert mapping.old_tensor_index.tolist() == list(range(6000))
    assert mapping.gene_id.tolist() == panels[6].gene_id.tolist()
    assert mapping.raw_panel_position.tolist() == panels[6].panel_position.tolist()
    assert panels[10].iloc[mapping.new_tensor_index].gene_id.tolist() == panels[6].gene_id.tolist()


def test_gene_context_raw_axis_contract_binds_snapshots_without_raw_data():
    contract = json.loads((BUNDLE / "inputs/raw_axis_contract.json").read_text())
    assert contract["n_genes"] == 20097
    assert contract["raw_var_sha256"] == "64f4e30e17763bef37793e7229edb6caa0a7ffe3f3bfa5a08e1d4ca8ea7ea1dc"
    assert contract["canonical_panel_sha256"] == (
        "9f6dd441e00d4e00168d69d6133ddb2802ab9e87a20c35ddfe1bcfa84f5ebcce"
    )
    for key in ("gene_universe", "ranked_gene_positions"):
        binding = contract[key]
        assert not Path(binding["path"]).is_absolute()
        assert _sha(BUNDLE / binding["path"]) == binding["sha256"]
    assert not list(BUNDLE.rglob("*.parquet"))


def test_gene_context_rebuild_is_exact_after_relocation_from_unrelated_cwd(tmp_path):
    relocated = tmp_path / "checkout/gene_context"
    shutil.copytree(BUNDLE, relocated, ignore=shutil.ignore_patterns("__pycache__"))
    cwd = tmp_path / "unrelated"
    cwd.mkdir()
    output = tmp_path / "rebuilt"
    env = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
    subprocess.run(
        [sys.executable, str(relocated / "build_collection.py"), "--output", str(output)],
        cwd=cwd, env=env, check=True, capture_output=True, text=True,
    )
    manifest = json.loads((relocated / "MANIFEST.json").read_text())
    for item in manifest["files"]:
        assert _sha(output / item["path"]) == item["sha256"], item["path"]
    assert _sha(output / "MANIFEST.json") == _sha(relocated / "MANIFEST.json")
    receipt = tmp_path / "verification.json"
    subprocess.run(
        [sys.executable, str(relocated / "verify_collection.py"), "--root", str(output),
         "--receipt", str(receipt)],
        cwd=cwd, env=env, check=True, capture_output=True, text=True,
    )
    result = json.loads(receipt.read_text())
    assert result["status"] == "PASS"
    assert result["original_source_panel_files_rechecked"] == 0


def test_gene_context_verifier_uses_only_bundled_sources_and_is_read_only(monkeypatch):
    verifier = _verifier()
    source = json.loads((BUNDLE / "inputs/source_bindings.json").read_text())
    historical = {Path(item["original_path"]) for item in source["source_records"]}
    original_is_file = Path.is_file

    def no_historical_probe(path):
        assert path not in historical, f"runtime dereferenced historical source: {path}"
        return original_is_file(path)

    monkeypatch.setattr(Path, "is_file", no_historical_probe)
    receipt_hash = _sha(BUNDLE / "VERIFIED.json")
    result = verifier.verify(BUNDLE)
    assert result["status"] == "PASS"
    assert result["original_source_panel_files_rechecked"] == 0
    assert _sha(BUNDLE / "VERIFIED.json") == receipt_hash


def test_gene_context_verifier_rejects_tampered_payload(tmp_path):
    destination = tmp_path / "bundle"
    shutil.copytree(BUNDLE, destination, ignore=shutil.ignore_patterns("__pycache__"))
    ids = destination / "panels/10k/gene_ids.txt"
    ids.write_bytes(ids.read_bytes().replace(b"ENSG", b"FAKE", 1))
    with pytest.raises(AssertionError):
        _verifier().verify(destination)
