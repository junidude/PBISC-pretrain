"""Build review-only 7k/8k nested panels from frozen evidence and train HVG stats.

Never edits active configs; never imports torch; never reads expression arrays.
Canonical raw coordinates stay panel_position; CSV row number is new tensor axis.
"""
from __future__ import annotations

import collections
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from audit_local import ANNOT, LADDER, M, OUT, RAW, binding, sha, write_json

HGNC = OUT / "reference/hgnc_complete_set_downloaded.tsv"


def make_resolver(raw: pd.DataFrame):
    hgnc = pd.read_csv(HGNC, sep="\t", usecols=["symbol", "alias_symbol", "prev_symbol", "ensembl_gene_id", "status"], dtype=str).fillna("")
    approved, aliases = collections.defaultdict(set), collections.defaultdict(set)
    raw_ids = set(raw.gene_id)
    symbols = raw.groupby("gene_symbol").gene_id.apply(set).to_dict()
    for row in hgnc.itertuples(index=False):
        if row.status != "Approved" or not row.ensembl_gene_id:
            continue
        approved[row.symbol].add(row.ensembl_gene_id)
        for name in [row.symbol, *row.alias_symbol.split("|"), *row.prev_symbol.split("|")]:
            if name:
                aliases[name].add(row.ensembl_gene_id)

    def resolve(symbol: str) -> tuple[str, str, str]:
        direct = symbols.get(symbol, set())
        if len(direct) == 1:
            if approved.get(symbol) and not direct & approved[symbol]:
                return "", "raw_symbol_conflicts_current_HGNC_Ensembl_review_required", ";".join(sorted(direct | approved[symbol]))
            return next(iter(direct)), "exact_raw_symbol", ""
        if len(direct) > 1:
            main = direct & approved.get(symbol, set())
            if len(main) == 1:
                return next(iter(main)), "HGNC_primary_Ensembl_resolves_multiple_raw_loci", ";".join(sorted(direct))
            return "", "ambiguous_raw_symbol_unresolved", ";".join(sorted(direct))
        known = approved.get(symbol, set()) or aliases.get(symbol, set())
        if len(known) > 1:
            return "", "ambiguous_HGNC_alias_unresolved", ";".join(sorted(known))
        match = known & raw_ids
        if len(match) == 1:
            return next(iter(match)), "downloaded_HGNC_symbol_or_alias_to_Ensembl", ""
        if len(match) > 1:
            return "", "ambiguous_HGNC_alias_unresolved", ";".join(sorted(match))
        return "", "HGNC_gene_absent_raw_axis" if known else "unresolved_symbol_absent_raw_axis", ";".join(sorted(known))

    return resolve


def evidence_rank(row: pd.Series) -> tuple:
    """Order within one source program; do not compare cross-study effect sizes."""
    stat = {}
    if row.get("source_statistic", ""):
        try:
            stat = json.loads(row.source_statistic)
        except (ValueError, TypeError):
            pass
    for key in ["source_rank", "rank"]:
        value = row.get(key, "") if key == "source_rank" else stat.get(key, "")
        try:
            if value != "" and math.isfinite(float(value)):
                return (0, float(value), row.gene_id)
        except (ValueError, TypeError):
            pass
    for key in ["log2FC", "avg_logFC", "avg_diff"]:
        value = stat.get(key, row.get(key, ""))
        try:
            if value != "" and math.isfinite(float(value)):
                return (1, -abs(float(value)), row.gene_id)
        except (ValueError, TypeError):
            pass
    return (2, int(row.evidence_order), row.gene_id)


def round_robin(groups: dict[str, list[str]]) -> list[str]:
    queues = {key: collections.deque(value) for key, value in sorted(groups.items())}
    result, seen = [], set()
    while queues:
        for key in list(queues):
            queue = queues[key]
            while queue and queue[0] in seen:
                queue.popleft()
            if queue:
                gene = queue.popleft()
                result.append(gene)
                seen.add(gene)
            if not queue:
                del queues[key]
    return result


def main() -> None:
    raw = pd.read_parquet(RAW)
    stats = pd.read_csv(OUT / "audited_gene_universe.csv")
    old = pd.read_csv(LADDER / "tiers/6k/selected_genes.csv")
    old_ids = set(old.gene_id)
    resolve = make_resolver(raw)
    frames, input_sources = [], []
    for path in sorted(OUT.glob("research/*/evidence.csv")):
        e = pd.read_csv(path, dtype=str).fillna("")
        e["evidence_file"] = str(path.relative_to(OUT))
        e["evidence_csv_row"] = np.arange(len(e)) + 2
        frames.append(e)
        input_sources.append(binding(path))
    assert len(frames) == 2, "Both research agents' evidence.csv files are required"
    e = pd.concat(frames, ignore_index=True).fillna("")
    e["evidence_order"] = np.arange(len(e))
    e["gene_symbol"] = e.gene_symbol.str.strip()
    resolved = [resolve(symbol) for symbol in e.gene_symbol]
    e[["gene_id", "resolution", "ambiguity_or_absent_gene_ids"]] = pd.DataFrame(resolved, index=e.index)
    e["canonical_gene_symbol"] = e.gene_id.map(raw.set_index("gene_id").gene_symbol).fillna("")
    e["raw_available"] = e.gene_id.ne("")
    e["in_final6k"] = e.gene_id.isin(old_ids)
    e["symbol_family_in_final6k"] = e.gene_symbol.isin(old.gene_symbol)
    e["in_frozen_1005_annotator"] = e.gene_id.isin(pd.read_csv(ANNOT / "genes.csv").gene_id)
    for field in ["panel_position", "measured_cells", "detected_cells", "eligible", "universal_rank"]:
        e["train1m_" + field] = e.gene_id.map(stats.set_index("gene_id")[field])
    e["train_support_status"] = np.select([~e.raw_available, e.train1m_measured_cells.eq(0), e.train1m_detected_cells.eq(0)], ["raw_unavailable_or_unresolved", "unmeasured_in_train1m", "measured_but_zero_detected_in_train1m_not_full_corpus_absence"], default="positive_in_train1m")
    # Disease authors' bounded ranked programs get the same fair queue level as
    # supporting blood identity programs, rather than all losing to a large DC list.
    tiers = {"core": 0, "supporting": 1, "optional": 1, "optional_state": 2, "optional_tissue": 3}
    unknown = set(e.priority) - set(tiers)
    assert not unknown, f"Unrecognized evidence priority: {unknown}"
    e["selection_tier"] = e.priority.map(tiers)
    e["program_key"] = e.apply(lambda r: "|".join([r.get("source_id", "") or r.doi or r.primary_source_url, r.get("program", "") or r.population]), axis=1)
    e["source_scope_class"] = e.apply(lambda r: "same_benchmark_source_endpoint_informed" if "same_benchmark" in str(r.get("selection_scope", "")) or any(x in str(r.get("population", "")) for x in ["Ye_", "Imperial_", "Giroux", "Ji_"]) else "external_reference_exposure_not_fully_audited", axis=1)
    sentinel_path = OUT / "research/ilc_dc/protected_sentinels.csv"
    sentinel_symbols = set()
    if sentinel_path.exists():
        sentinels = pd.read_csv(sentinel_path).fillna("")
        sentinel_symbols = set(sentinels.gene_symbol)
        assert sentinel_symbols <= set(e.gene_symbol), "Sentinels need source-supported evidence rows"
        input_sources.append(binding(sentinel_path))
        # Protect assay capability without relabelling tissue/cytokine evidence as
        # universal core identity or as constitutive resting-blood expression.
        e.loc[e.gene_symbol.isin(sentinel_symbols), "selection_tier"] = 0
    e["protected_functional_sentinel"] = e.gene_symbol.isin(sentinel_symbols)
    e["coverage_class"] = np.select([
        e.priority.eq("optional_tissue"),
        e.priority.eq("optional_state"),
        e.source_scope_class.eq("same_benchmark_source_endpoint_informed"),
        e.priority.eq("core") & e.role.str.contains("state|program", case=False, regex=True),
        e.priority.eq("core"),
    ], ["optional_tissue", "independent_optional_state", "same_benchmark_targeted_state", "core_state_program", "core_identity_boundary"], default="blood_supporting")
    available = e[e.raw_available & e.train1m_measured_cells.gt(0)].copy()
    priority_order = []
    for tier in sorted(available.selection_tier.unique()):
        populations = {}
        for population, group in available[available.selection_tier.eq(tier)].groupby("population", sort=True):
            programs = {}
            for program, pg in group.groupby("program_key", sort=True):
                rows = sorted([r for _, r in pg.iterrows()], key=evidence_rank)
                programs[program] = list(dict.fromkeys(r.gene_id for r in rows if r.gene_id not in old_ids))
            populations[population] = round_robin(programs)
        priority_order.extend(round_robin(populations))
    priority_order = list(dict.fromkeys(priority_order))
    e["addition_priority_rank"] = e.gene_id.map({g: i + 1 for i, g in enumerate(priority_order)})
    sources = input_sources + [binding(RAW), binding(HGNC), binding(OUT / "reference/hgnc_download_receipt.json"), binding(LADDER / "manifest.json"), binding(LADDER / "full_gene_statistics.csv"), binding(LADDER / "ranked_gene_positions.npy"), binding(LADDER / "tiers/6k/selected_genes.csv"), binding(LADDER / "tiers/6k/gene_positions.npy"), binding(OUT / "audited_gene_universe.csv"), binding(Path(__file__))]
    backfill = stats[stats.eligible & stats.measured_cells.gt(0)].sort_values(["universal_rank", "gene_id"]).gene_id.tolist()
    all_additions = list(dict.fromkeys(priority_order + [g for g in backfill if g not in old_ids]))
    assert len(all_additions) >= 2000
    chosen = {6000: old_ids}
    panels = []
    for size in [7000, 8000]:
        additions = all_additions[:size - 6000]
        selected_ids = old_ids | set(additions)
        chosen[size] = selected_ids
        assert len(selected_ids) == size
        selected = stats[stats.gene_id.isin(selected_ids)].sort_values("panel_position").copy()
        selected["tensor_index"] = np.arange(size)
        selected["selection_reason"] = np.where(selected.gene_id.isin(old_ids), "preserved_final6k", np.where(selected.gene_id.isin(priority_order), "primary_literature_curated", "train_only_HVG_ranked_backfill_not_literature_curated"))
        selected["addition_priority_rank"] = selected.gene_id.map({g: i + 1 for i, g in enumerate(all_additions)})
        subset = e[e.raw_available]
        for column in ["population", "priority", "role", "primary_source_url", "source_scope_class"]:
            lookup = subset.groupby("gene_id")[column].agg(lambda x: ";".join(sorted(set(v for v in x if v)))).to_dict()
            selected["evidence_" + column] = selected.gene_id.map(lookup).fillna("")
        dest = OUT / "panels" / f"{size // 1000}k"
        dest.mkdir(parents=True, exist_ok=True)
        csv_path, pos_path = dest / "selected_genes.csv", dest / "gene_positions.npy"
        selected.to_csv(csv_path, index=False)
        np.save(pos_path, selected.panel_position.to_numpy(dtype=np.int64), allow_pickle=False)
        positions = selected.set_index("gene_id").tensor_index
        mapping = pd.DataFrame({"gene_id": old.gene_id, "gene_symbol": old.gene_symbol, "raw_panel_position": old.panel_position, "old_tensor_index": np.arange(6000), "new_tensor_index": old.gene_id.map(positions)})
        mapping.to_csv(dest / "old_to_new.csv", index=False)
        selected[~selected.gene_id.isin(old_ids)].to_csv(dest / "additions.csv", index=False)
        reason_counts = selected.selection_reason.value_counts().to_dict()
        tier_manifest = {"schema": "pbisc.rare_gene_expansion.candidate.v1", "status": "research_candidate_only_no_training_or_active_config_change", "n_selected_genes": size, "old_n_genes": 6000, "tensor_order": "ascending canonical raw panel_position; CSV row=tensor axis", "gene_id_key": "Ensembl stable ID; duplicate biological symbols are distinct loci, flagged separately", "source_preservation": "all original 6000 Ensembl IDs preserved", "gene_positions_path": str(pos_path), "gene_positions_sha256": sha(pos_path), "selected_genes_path": str(csv_path), "selected_genes_sha256": sha(csv_path), "old_to_new": binding(dest / "old_to_new.csv"), "reason_counts": reason_counts, "source_bindings": sources, "legacy_loader_compatible_manifest": False, "loader_gate": "The existing explicit_train_only_hvg_manifest loader rejects this evidence-driven selector. A reviewed loader/manifest extension is required before training; do not disguise it as train-only HVG.", "gene_selection_expression_scope": "frozen pretraining-train-only HVG statistics; no local held-out expression accessed", "literature_scope": "includes benchmark-author endpoint-informed markers; not fully independent unbiased feature selection", "priority_policy": "core first; fair population and source-program round robin among supporting blood and bounded disease lists; optional stimulation then optional tissue; eligible train-only HVG backfill only if needed", "known_limitations": ["A source-defined program is a candidate feature list, not a complete or sufficient identity test", "1M sampled detection is not population-specific, not full-corpus abundance, and zero is inconclusive", "Official HGNC snapshot fetched 2026-10-15; raw canonical axis remains the existing project freeze", "Raw-unavailable genes require new preprocessing and cannot be appended as zero-valued columns"]}
        (dest / "selection_manifest.json").write_text(json.dumps(tier_manifest, indent=2) + "\n")
        panels.append({"size": size, "n_literature_additions": int(sum(g in priority_order for g in additions)), "n_backfill_additions": int(sum(g not in priority_order for g in additions)), "selected_genes_sha256": sha(csv_path), "gene_positions_sha256": sha(pos_path), "manifest": binding(dest / "selection_manifest.json")})
    assert chosen[6000] < chosen[7000] < chosen[8000]
    for size in [7000, 8000]:
        e[f"in_{size // 1000}k"] = e.gene_id.isin(chosen[size])
    e.to_csv(OUT / "evidence_resolved.csv", index=False)
    e[~e.raw_available].to_csv(OUT / "unavailable_or_ambiguous_evidence.csv", index=False)
    coverage = []
    for population, group in e.groupby("population", sort=True):
        unique = group.drop_duplicates("gene_symbol")
        core = group[group.priority.eq("core")]
        available_ids = set(group.loc[group.raw_available, "gene_id"])
        core_ids = set(core.loc[core.raw_available, "gene_id"])
        row = {"population": population, "core_required_symbols": core.gene_symbol.nunique(), "all_candidate_program_symbols": unique.gene_symbol.nunique(), "raw_available_Ensembl_genes": len(available_ids), "raw_unavailable_or_unresolved_symbols": ";".join(sorted(set(group.loc[~group.raw_available, "gene_symbol"]))), "core_raw_unavailable_or_unresolved_symbols": ";".join(sorted(set(core.loc[~core.raw_available, "gene_symbol"]))), "train1m_positive_genes": group.loc[group.train_support_status.eq("positive_in_train1m"), "gene_id"].nunique(), "train1m_measured_zero_detected_genes": group.loc[group.train_support_status.str.startswith("measured_but_zero"), "gene_id"].nunique(), "gene_specific_donor_platform_signal": "not persisted in frozen pooled statistics; unknown", "source_scope": ";".join(sorted(set(group.source_scope_class)))}
        for size in [6000, 7000, 8000]:
            suffix = str(size // 1000) + "k"
            row["available_" + suffix] = len(available_ids & chosen[size])
            row["core_available_" + suffix] = len(core_ids & chosen[size])
            row["core_missing_available_raw_" + suffix] = ";".join(sorted(set(core.loc[core.raw_available & ~core.gene_id.isin(chosen[size]), "gene_symbol"])))
            row["program_missing_available_raw_" + suffix] = ";".join(sorted(set(group.loc[group.raw_available & ~group.gene_id.isin(chosen[size]), "gene_symbol"])))
        row["program_genes_gained_8k_over_7k"] = len(available_ids & (chosen[8000] - chosen[7000]))
        coverage.append(row)
    coverage_df = pd.DataFrame(coverage)
    coverage_df.to_csv(OUT / "coverage.csv", index=False)
    category_rows = []
    for category, group in e.groupby("coverage_class", sort=True):
        ids = set(group.loc[group.raw_available, "gene_id"])
        row = {"coverage_class": category, "requested_symbols": group.gene_symbol.nunique(), "available_raw_genes": len(ids), "core_available_raw_genes": group.loc[group.raw_available & group.priority.eq("core"), "gene_id"].nunique(), "unresolved_symbols": group.loc[~group.raw_available, "gene_symbol"].nunique(), "interpretation": "overlapping gene sets; category counts must not be added"}
        for size in [6000, 7000, 8000]:
            row[f"available_{size // 1000}k"] = len(ids & chosen[size])
        category_rows.append(row)
    pd.DataFrame(category_rows).to_csv(OUT / "coverage_by_evidence_class.csv", index=False)
    # Every panel gene, including HVG-only backfill, receives measured-source audit fields.
    masks = pd.read_csv(OUT / "cohort_measurement_masks.csv")
    panel_gene_audit = stats[stats.gene_id.isin(chosen[8000])].merge(masks.drop(columns="gene_symbol"), on="gene_id", validate="one_to_one")
    panel_gene_audit["in_7k"] = panel_gene_audit.gene_id.isin(chosen[7000])
    panel_gene_audit["in_8k"] = True
    panel_gene_audit.to_csv(OUT / "panel_gene_measurement_audit.csv", index=False)
    mask_coverage = []
    for cohort in ["ye2026_sepsis", "imperial_eskd", "giroux2022", "ji2025_chili"]:
        for size in [6000, 7000, 8000]:
            d = masks[masks.gene_id.isin(chosen[size])]
            a, b = d[f"{cohort}_singlecell_fraction"], d[f"{cohort}_bulk_fraction"]
            mask_coverage.append({"cohort": cohort, "panel_genes": size, "sc_measured_all_audited_masks": int(a.eq(1).sum()), "bulk_measured_all_audited_masks": int(b.eq(1).sum()), "both_measured_all_audited_masks": int((a.eq(1) & b.eq(1)).sum()), "unavailable_evaluation_policy": "NA or masked; never zero-filled as true absence", "used_for_selection": False})
    pd.DataFrame(mask_coverage).to_csv(OUT / "panel_cohort_measurement_coverage.csv", index=False)
    required_ids = set(available.loc[available.priority.eq("core"), "gene_id"])
    protected_ids = set(available.loc[available.selection_tier.eq(0), "gene_id"])
    positive_ids = set(available.loc[available.train1m_detected_cells.gt(0), "gene_id"])
    summary = {"schema": "pbisc.rare_gene_expansion.v1", "status": "review_only_panels_built", "evidence_rows": len(e), "evidence_requested_symbols": e.gene_symbol.nunique(), "resolved_Ensembl_genes": available.gene_id.nunique(), "curated_missing_from_final6k": len(priority_order), "curated_missing_train1m_positive": len(set(priority_order) & positive_ids), "curated_missing_train1m_zero_detected": len(set(priority_order) - positive_ids), "core_raw_available_unique_genes": len(required_ids), "core_missing_from6k": len(required_ids - old_ids), "protected_functional_sentinel_symbols": sorted(sentinel_symbols), "minimal6k_plus_core_count": len(old_ids | required_ids), "minimal6k_plus_all_available_program_count": len(old_ids | set(priority_order)), "panels": panels, "core_loss_7k": len(required_ids - chosen[7000]), "core_loss_8k": len(required_ids - chosen[8000]), "remaining_full_program_genes_7k": len(set(priority_order) - chosen[7000]), "remaining_full_program_genes_8k": len(set(priority_order) - chosen[8000]), "decision_rule": "Both preserve core. Choose 7k for compact capability, or 8k for additional documented blood/state program coverage; neither panel size is proven necessary or sufficient for biological recovery. Full optional literature union is not a requirement.", "no_active_configs_modified": True, "training_launched": False, "source_bindings": sources, "selection_uses_local_heldout_expression": False, "selection_includes_benchmark_published_endpoint_markers": True}
    write_json("selection_manifest.json", summary)
    summary["minimal6k_plus_core_and_protected_sentinels_count"] = len(old_ids | protected_ids)
    summary["core_and_protected_loss_7k"] = len(protected_ids - chosen[7000])
    summary["core_and_protected_loss_8k"] = len(protected_ids - chosen[8000])
    write_json("selection_manifest.json", summary)
    print(json.dumps({k: v for k, v in summary.items() if k != "source_bindings"}, indent=2))


if __name__ == "__main__":
    main()
