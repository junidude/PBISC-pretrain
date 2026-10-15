"""Reproduce this collection from its small input snapshots and sealed panel copies.

Default output is this directory. --output DIR builds a separate reproducibility
copy, without reading the original project, papers, expression, or checkpoints.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
BASE_COLUMNS = ["panel_position", "gene_id", "gene_symbol", "mean_log1p", "variance_log1p", "normalized_dispersion", "measured_cells", "detected_cells"]
FROZEN_TIERS = ["panels/2k", "panels/6k", "panels/7k", "panels/8k", "historical/2k", "historical/3k"]
EXPECTED_FINAL6K = "653ffa45ac519000baa41086c52408009391d6e68cfb218bc6a50d34163fcb24"
EXPECTED_CURRENT3K = "8ecfbc9c1bb2ee61eb6d9c7c2aabeedaceb8a4f478f9a93e707fd826edcc1121"


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1048576), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def file_record(path: Path, root: Path) -> dict:
    return {"path": str(path.relative_to(root)), "bytes": path.stat().st_size, "sha256": sha(path)}


def copy_exact(src: Path, dst: Path, expected: str) -> None:
    if sha(src) != expected:
        raise ValueError(f"Frozen input changed: {src}")
    dst.parent.mkdir(parents=True, exist_ok=True)
    if src.resolve() != dst.resolve():
        shutil.copyfile(src, dst)
    if sha(dst) != expected:
        raise ValueError(f"Copy does not match: {dst}")


def build(output: Path) -> dict:
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    source_path = ROOT / "inputs/source_bindings.json"
    source = json.loads(source_path.read_text())
    # All actual build inputs are collection-local. Original absolute paths in
    # source_bindings are provenance and are never dereferenced here.
    for item in source["derived_snapshots"]:
        copy_exact(ROOT / item["path"], output / item["path"], item["sha256"])
    copy_exact(source_path, output / "inputs/source_bindings.json", sha(source_path))
    frozen_records = [r for r in source["source_records"] if r["role"] == "frozen_panel_copy"]
    for item in frozen_records:
        copy_exact(ROOT / item["collection_copy"], output / item["collection_copy"], item["sha256"])
    universe = pd.read_csv(ROOT / "inputs/gene_universe.csv")
    literature = pd.read_csv(ROOT / "inputs/literature_addition_order.csv").fillna("")
    assert len(universe) == 20097 and universe.gene_id.is_unique
    assert universe.panel_position.tolist() == list(range(20097))
    assert literature.gene_id.is_unique
    literature = literature.sort_values(["addition_priority_rank", "gene_id"])
    assert literature.addition_priority_rank.tolist() == list(range(1, 2092))
    assert len(literature) == 2091
    eligible = universe[universe.eligible & universe.measured_cells.gt(0)].sort_values(["universal_rank", "gene_id"])
    assert len(eligible) == 15114
    baseline = pd.read_csv(ROOT / "panels/6k/selected_genes.csv")
    baseline_ids = set(baseline.gene_id)
    assert len(baseline_ids) == 6000
    assert sha(ROOT / "panels/6k/gene_positions.npy") == EXPECTED_FINAL6K
    for n in [2000, 6000]:
        old = pd.read_csv(ROOT / "panels" / f"{n // 1000}k/selected_genes.csv")
        assert set(eligible.head(n).gene_id) == set(old.gene_id)
    assert not set(literature.gene_id) & baseline_ids
    assert set(literature.gene_id) <= set(universe.loc[universe.measured_cells.gt(0), "gene_id"])

    # Current 3k uses the same eligible universal ranking as current 2k/6k.
    three = universe[universe.gene_id.isin(eligible.head(3000).gene_id)].sort_values("panel_position")
    three_dir = output / "panels/3k"
    three_dir.mkdir(parents=True, exist_ok=True)
    three[BASE_COLUMNS].to_csv(three_dir / "selected_genes.csv", index=False)
    np.save(three_dir / "gene_positions.npy", three.panel_position.to_numpy(dtype=np.int64), allow_pickle=False)
    assert sha(three_dir / "gene_positions.npy") == EXPECTED_CURRENT3K

    # Exact extension of the frozen curated addition order. No source ranking,
    # literature search, native expression audit, or threshold is recomputed.
    literature_ids = literature.gene_id.tolist()
    for n in [7000, 8000]:
        fixed = pd.read_csv(ROOT / "panels" / f"{n // 1000}k/selected_genes.csv")
        assert set(fixed.gene_id) == baseline_ids | set(literature_ids[:n - 6000])
    backfill = [g for g in eligible.gene_id if g not in baseline_ids and g not in set(literature_ids)][:1909]
    assert len(backfill) == 1909
    additions = literature_ids + backfill
    ten_ids = baseline_ids | set(additions)
    assert len(ten_ids) == 10000
    ten = universe[universe.gene_id.isin(ten_ids)].sort_values("panel_position").copy()
    ten.insert(0, "tensor_index", np.arange(10000))
    ten["selection_reason"] = np.where(ten.gene_id.isin(baseline_ids), "preserved_final6k", np.where(ten.gene_id.isin(literature_ids), "frozen_primary_literature_curated", "train_only_HVG_ranked_backfill_not_literature_curated"))
    ten["addition_priority_rank"] = ten.gene_id.map({g: i + 1 for i, g in enumerate(additions)})
    ten = ten.merge(literature.drop(columns="addition_priority_rank"), on="gene_id", how="left", validate="one_to_one").fillna({c: "" for c in literature.columns if c.startswith("evidence_")})
    ten_dir = output / "panels/10k"
    ten_dir.mkdir(parents=True, exist_ok=True)
    ten.to_csv(ten_dir / "selected_genes.csv", index=False)
    np.save(ten_dir / "gene_positions.npy", ten.panel_position.to_numpy(dtype=np.int64), allow_pickle=False)
    ten[~ten.gene_id.isin(baseline_ids)].to_csv(ten_dir / "additions.csv", index=False)
    index_of = ten.set_index("gene_id").tensor_index
    mapping = pd.DataFrame({"gene_id": baseline.gene_id, "gene_symbol": baseline.gene_symbol, "raw_panel_position": baseline.panel_position, "old_tensor_index": np.arange(6000), "new_tensor_index": baseline.gene_id.map(index_of)})
    mapping.to_csv(ten_dir / "old6k_to_10k.csv", index=False)

    definitions = [
        ("panels/2k", "current_floor01_1m", "existing_current_panel_exact_copy", "Eligible universal rank prefix 2000; copied byte-exact."),
        ("panels/3k", "current_floor01_1m", "newly_materialized_current_panel_no_new_trained_model", "Eligible universal rank prefix 3000; canonical tensor order."),
        ("panels/6k", "current_floor01_1m", "existing_final6k_panel_exact_copy", "Eligible universal rank prefix 6000; copied byte-exact."),
        ("panels/7k", "frozen_rare_program_expansion", "existing_research_candidate_exact_copy_no_training", "Final6k plus first1000 frozen literature additions; no HVG backfill."),
        ("panels/8k", "frozen_rare_program_expansion", "existing_research_candidate_exact_copy_no_training", "Final6k plus first2000 frozen literature additions; no HVG backfill."),
        ("panels/10k", "frozen_rare_program_expansion_plus_train_HVG", "new_research_candidate_no_training", "Final6k plus all2091 frozen literature additions plus1909 remaining eligible train-HVG ranks."),
        ("historical/2k", "historical_train256x128", "historical_experiment_panel_exact_copy", "Original historical2k; not the main harmonized current2k."),
        ("historical/3k", "historical_train256x128", "historical_experiment_panel_exact_copy", "Original historical3k; not the new harmonized current3k."),
    ]
    membership = universe[["panel_position", "gene_id", "gene_symbol"]].copy()
    versions = {
        "current_floor01_1m": "panel_ladder_floor01_1m_20260923",
        "frozen_rare_program_expansion": "rare_gene_context_expansion_20261015",
        "frozen_rare_program_expansion_plus_train_HVG": "rare_gene_context_expansion_20261015 + panel_ladder_floor01_1m_20260923",
        "historical_train256x128": "hvg_train256x128_seed20260627",
    }
    index_rows = []
    outputs = [output / "inputs/source_bindings.json", *[output / item["path"] for item in source["derived_snapshots"]]]
    prior_ids: set[str] = set()
    for directory, family, status, rule in definitions:
        dest = output / directory
        panel = pd.read_csv(dest / "selected_genes.csv")
        size = len(panel)
        ids = set(panel.gene_id)
        assert panel.gene_id.is_unique
        assert np.array_equal(np.load(dest / "gene_positions.npy", allow_pickle=False), panel.panel_position.to_numpy(dtype=np.int64))
        if directory.startswith("panels/"):
            assert prior_ids < ids
            prior_ids = ids
        (dest / "gene_ids.txt").write_text("\n".join(panel.gene_id.astype(str)) + "\n")
        # Symbols deliberately retain duplicate locus labels and row order.
        (dest / "gene_symbols.txt").write_text("\n".join(panel.gene_symbol.astype(str)) + "\n")
        membership[directory.replace("/", "_")] = membership.gene_id.isin(ids)
        original = [r for r in frozen_records if str(Path(r["collection_copy"]).parent) == directory]
        paths = [dest / "selected_genes.csv", dest / "gene_positions.npy", dest / "gene_ids.txt", dest / "gene_symbols.txt"]
        if directory == "panels/10k":
            paths += [dest / "additions.csv", dest / "old6k_to_10k.csv"]
        manifest = {
            "schema": "pbisc.gene_context_collection.tier.v1", "panel": directory, "n_genes": size,
            "family": family, "status": status, "selection_rule": rule,
            "tensor_order": "CSV row and TXT line; ascending original raw panel_position",
            "gene_id_key": "unique Ensembl ID; symbols are not deduplicated",
            "corpus_fingerprint": source["corpus_fingerprint"], "trained_checkpoint_included": False,
            "loader_ready": False,
            "loader_gate": "Collection-local manifests are provenance records, not a replacement for the existing loader schema. New current3k and curated7k/8k/10k require reviewed loader/config bindings; 7k/8k/10k also require appropriate new or explicitly migrated/retrained weights.",
            "endpoint_informed": directory in {"panels/7k", "panels/8k", "panels/10k"},
            "gene_specificity_limit": "Literature programs include shared genes; HVG backfill is explicitly not literature-curated or rare-cell-specific.",
            "original_sources": original,
            "input_bindings": [file_record(output / item["path"], output) for item in source["derived_snapshots"]] + [file_record(output / "inputs/source_bindings.json", output)],
            "builder_sha256": sha(Path(__file__)), "files": [file_record(f, output) for f in paths],
        }
        if directory == "panels/10k":
            manifest["components"] = {"preserved_final6k": 6000, "frozen_literature_additions": 2091, "train_HVG_backfill": 1909, "extra_vs8k_literature": 91, "extra_vs8k_HVG": 1909}
        write_json(dest / "selection_manifest.json", manifest)
        outputs += paths + [dest / "selection_manifest.json"]
        index_rows.append({"panel": directory, "genes": size, "family": family, "source_version": versions[family], "status": status, "selection_rule": rule, "selected_genes": f"{directory}/selected_genes.csv", "gene_positions": f"{directory}/gene_positions.npy", "gene_ids": f"{directory}/gene_ids.txt", "gene_symbols": f"{directory}/gene_symbols.txt", "manifest": f"{directory}/selection_manifest.json", "selected_genes_sha256": sha(dest / "selected_genes.csv"), "gene_positions_sha256": sha(dest / "gene_positions.npy"), "symbols_are_unique": bool(panel.gene_symbol.is_unique)})
    membership.to_csv(output / "gene_membership.csv", index=False)
    pd.DataFrame(index_rows).to_csv(output / "INDEX.csv", index=False)
    outputs += [output / "gene_membership.csv", output / "INDEX.csv"]
    result = {
        "schema": "pbisc.gene_context_collection.v1", "date": "2026-10-15", "status": "built_not_model_validation",
        "main_sizes": [2000, 3000, 6000, 7000, 8000, 10000], "historical_sizes": [2000, 3000],
        "small_panel_choice": source["small_panel_choice"], "main_sets_nested": True,
        "selector_transition": "2k/3k/6k current train-HVG family;7k/8k literature expansion;10k completes literature then train-HVG backfill",
        "ten_k_components": {"preserved_final6k": 6000, "literature_additions": 2091, "train_HVG_backfill": 1909},
        "frozen_final6k_positions_sha256": EXPECTED_FINAL6K, "current3k_positions_sha256": EXPECTED_CURRENT3K,
        "no_new_research_or_expression_statistics": True, "no_training_or_inference": True,
        "input_source_bindings": file_record(output / "inputs/source_bindings.json", output),
        "builder_sha256": sha(Path(__file__)), "files": [file_record(p, output) for p in sorted(set(outputs))],
    }
    write_json(output / "MANIFEST.json", result)
    return {"status": result["status"], "main_sizes": result["main_sizes"], "ten_k_components": result["ten_k_components"], "ten_k_positions_sha256": sha(ten_dir / "gene_positions.npy"), "current3k_positions_sha256": sha(three_dir / "gene_positions.npy"), "output": str(output)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT)
    args = parser.parse_args()
    print(json.dumps(build(args.output), indent=2))
