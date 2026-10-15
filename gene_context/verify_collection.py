"""Verify collection-local inputs without reading historical source paths.

The bundled historical VERIFIED.json is preserved. Use --receipt PATH to write
a separate receipt for the current verification; no model imports are needed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1048576), b""):
            h.update(chunk)
    return h.hexdigest()


def verify(root: Path, receipt: Path | None = None) -> dict:
    root = root.resolve()
    manifest = json.loads((root / "MANIFEST.json").read_text())
    source = json.loads((root / "inputs/source_bindings.json").read_text())
    for record in manifest["files"]:
        path = root / record["path"]
        assert path.stat().st_size == record["bytes"], str(path)
        assert sha(path) == record["sha256"], str(path)
    universe = pd.read_csv(root / "inputs/gene_universe.csv")
    priority = pd.read_csv(root / "inputs/literature_addition_order.csv")
    assert len(universe) == 20097 == universe.gene_id.nunique()
    assert universe.gene_symbol.notna().all()
    assert sorted(universe.universal_rank) == list(range(1, 20098))
    assert universe.panel_position.tolist() == list(range(20097))
    assert len(priority) == priority.gene_id.nunique() == 2091
    assert priority.addition_priority_rank.tolist() == list(range(1, 2092))
    main = {}
    outcomes = []
    for category, sizes in [("panels", [2, 3, 6, 7, 8, 10]), ("historical", [2, 3])]:
        for k in sizes:
            folder = root / category / f"{k}k"
            table = pd.read_csv(folder / "selected_genes.csv")
            positions = np.load(folder / "gene_positions.npy", allow_pickle=False)
            assert len(table) == table.gene_id.nunique() == k * 1000
            assert len(positions) == len(table) and positions.dtype == np.int64
            assert np.all(np.diff(positions) > 0)
            assert np.array_equal(table.panel_position, positions)
            assert np.array_equal(table.gene_id, universe.iloc[positions].gene_id)
            ids_txt = (folder / "gene_ids.txt").read_text().splitlines()
            symbols_txt = (folder / "gene_symbols.txt").read_text().splitlines()
            assert ids_txt == table.gene_id.astype(str).tolist()
            assert symbols_txt == table.gene_symbol.astype(str).tolist()
            assert len(symbols_txt) == len(table)  # duplicate symbols retained
            tier_manifest = json.loads((folder / "selection_manifest.json").read_text())
            assert tier_manifest["n_genes"] == len(table)
            for record in tier_manifest["files"]:
                assert sha(root / record["path"]) == record["sha256"]
            if category == "panels":
                main[k] = table
            outcomes.append({"panel": f"{category}/{k}k", "genes": len(table), "unique_symbols": table.gene_symbol.nunique(), "row_order_and_coordinates": "PASS", "txt_order_and_duplicate_symbols": "PASS"})
    for a, b in zip([2, 3, 6, 7, 8], [3, 6, 7, 8, 10]):
        assert set(main[a].gene_id) < set(main[b].gene_id), (a, b)
    rank_table = universe[universe.eligible].sort_values("universal_rank")
    assert len(rank_table) == 15114
    for k in [2, 3, 6]:
        assert set(rank_table.head(k * 1000).gene_id) == set(main[k].gene_id)
    assert sha(root / "panels/6k/gene_positions.npy") == "653ffa45ac519000baa41086c52408009391d6e68cfb218bc6a50d34163fcb24"
    assert sha(root / "panels/3k/gene_positions.npy") == "8ecfbc9c1bb2ee61eb6d9c7c2aabeedaceb8a4f478f9a93e707fd826edcc1121"
    base = set(main[6].gene_id)
    literature = set(priority.gene_id)
    assert not literature & base
    for k in [7, 8]:
        assert set(main[k].gene_id) == base | set(priority.head(k * 1000 - 6000).gene_id)
    expected_ten = base | literature
    chosen_backfill = []
    for row in rank_table.itertuples(index=False):
        if len(expected_ten) == 10000:
            break
        if row.gene_id not in expected_ten and row.measured_cells > 0:
            expected_ten.add(row.gene_id)
            chosen_backfill.append(row.gene_id)
    assert len(chosen_backfill) == 1909
    assert set(main[10].gene_id) == expected_ten
    counts = main[10].selection_reason.value_counts().to_dict()
    assert counts == {"preserved_final6k": 6000, "frozen_primary_literature_curated": 2091, "train_only_HVG_ranked_backfill_not_literature_curated": 1909}
    new = main[10][~main[10].gene_id.isin(base)]
    assert new.measured_cells.gt(0).all() and new.detected_cells.gt(0).all()
    mapping = pd.read_csv(root / "panels/10k/old6k_to_10k.csv")
    assert len(mapping) == 6000 and mapping.old_tensor_index.tolist() == list(range(6000))
    assert np.array_equal(mapping.gene_id, main[6].gene_id)
    assert np.array_equal(main[10].iloc[mapping.new_tensor_index].gene_id, main[6].gene_id)
    membership = pd.read_csv(root / "gene_membership.csv")
    assert np.array_equal(membership.gene_id, universe.gene_id)
    for outcome in outcomes:
        path = outcome["panel"]
        panel = pd.read_csv(root / path / "selected_genes.csv")
        assert np.array_equal(membership[path.replace("/", "_")], membership.gene_id.isin(panel.gene_id))
    frozen_count = 0
    for record in source["source_records"]:
        if record["role"] != "frozen_panel_copy":
            continue
        assert sha(root / record["collection_copy"]) == record["sha256"]
        frozen_count += 1
        # original_path is historical provenance, never a runtime dependency.
    result = {"status": "PASS", "main_nested_sizes": [2000, 3000, 6000, 7000, 8000, 10000], "panels": outcomes,
              "ten_k_reason_counts": counts, "ten_k_all_new_genes_train_sample_measured_and_detected": True,
              "frozen_panel_files_exact": frozen_count, "original_source_panel_files_rechecked": 0,
              "current2k3k6k_rank_replay": "PASS", "frozen7k8k_order_replay": "PASS", "ten_k_extension_replay": "PASS",
              "old6k_mapping": "PASS", "wide_membership": "PASS", "no_training_or_inference": True,
              "manifest_sha256": sha(root / "MANIFEST.json"), "verifier_sha256": sha(Path(__file__))}
    if receipt is not None:
        receipt.write_text(json.dumps(result, indent=2) + "\n")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--receipt", type=Path, help="Write a new receipt to this explicit path")
    args = parser.parse_args()
    print(json.dumps(verify(args.root, receipt=args.receipt), indent=2))
