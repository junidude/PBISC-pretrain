# Gene-context collection — version 20261015

The main collection uses the current floor01 baseline for **2k, 3k and 6k**, preserves the exact curated **7k/8k** candidates, and extends their frozen recipe to **10k**. Original historical 2k/3k lists are preserved separately, as explicitly requested. No training, inference, new literature search or expression-statistics pass was performed.

| Main panel | Selection and status | Gene list |
|---|---|---|
| 2k | Exact current floor01 rank-prefix copy | [CSV](panels/2k/selected_genes.csv) · [IDs](panels/2k/gene_ids.txt) · [symbols](panels/2k/gene_symbols.txt) |
| 3k | Newly materialized prefix of the same current ranking; **not a newly trained model** | [CSV](panels/3k/selected_genes.csv) · [IDs](panels/3k/gene_ids.txt) · [symbols](panels/3k/gene_symbols.txt) |
| 6k | Exact final floor01 baseline | [CSV](panels/6k/selected_genes.csv) · [IDs](panels/6k/gene_ids.txt) · [symbols](panels/6k/gene_symbols.txt) |
| 7k | Exact frozen 6k + first 1,000 curated additions | [CSV](panels/7k/selected_genes.csv) · [IDs](panels/7k/gene_ids.txt) · [symbols](panels/7k/gene_symbols.txt) |
| 8k | Exact frozen 6k + first 2,000 curated additions | [CSV](panels/8k/selected_genes.csv) · [IDs](panels/8k/gene_ids.txt) · [symbols](panels/8k/gene_symbols.txt) |
| 10k | New candidate: 6,000 baseline + 2,091 curated + 1,909 train-HVG backfill | [CSV](panels/10k/selected_genes.csv) · [IDs](panels/10k/gene_ids.txt) · [symbols](panels/10k/gene_symbols.txt) |

Historical experiment lists: [historical 2k](historical/2k/selected_genes.csv) and [historical 3k](historical/3k/selected_genes.csv). They are byte-exact copies of the old 256×128-sample selector outputs and must not be relabelled as the main current panels.

[INDEX.csv](INDEX.csv) records versions, selection families, status, paths and hashes. [gene_membership.csv](gene_membership.csv) provides one row per canonical Ensembl ID and a membership column for each main and historical list. Each tier also includes `gene_positions.npy` and a collection-local provenance manifest.

## Why 10k has these components

The 10,000-gene total is the requested budget, not an experimentally optimized size. The frozen rare-program design contains 2,091 available additions beyond 6k. The builder keeps their **existing `addition_priority_rank` order unchanged**, includes all 2,091, then adds the next 1,909 eligible genes from the existing train-only universal HVG ranking after excluding genes already selected. Consequently the extension from 8k to 10k consists of **91 remaining literature-program genes plus 1,909 HVG genes**.

The original two-level source-program/population round-robin was already frozen upstream; this builder does not repeat curation, revise priorities, use the later native ILC audit to rank genes, or resample statistics. The current 3k similarly takes the first 3,000 eligible universal ranks and writes them in canonical order. Exact current 2k and 6k were independently replayed from that same ranking.

All main sets are nested: **2k ⊂ 3k ⊂ 6k ⊂ 7k ⊂ 8k ⊂ 10k**. This is a mixed-selector sequence: 2k/3k/6k use current train-HVG selection, 7k/8k use targeted literature expansion, and 10k completes that expansion before adding HVGs. It is **not one pure HVG gene-number ablation**. The literature programs include shared genes and benchmark-author endpoint-informed targets; the extra 4,000 genes in 10k are not 4,000 rare-cell-specific markers. No local held-out expression informed the expansion, but that does not make the entire feature-selection process blind or wholly train-only.

## Coordinate and model contracts

Ensembl ID is the unique key. **Do not deduplicate the symbol TXT files**: distinct raw loci can share a symbol. CSV row order, TXT line order and NPY order agree; `panel_position` is the original 20,097-axis coordinate. The 10k list contains 10,000 distinct Ensembl IDs and 9,985 distinct symbols. All original 6k genes are preserved, but they need not occupy the first 6,000 tensor rows; use [old6k_to_10k.csv](panels/10k/old6k_to_10k.csv).

The current2k/final6k, curated7k/8k and historical2k/3k CSV/NPY files are copied byte-exact. Original files, frozen manifests, reports and active configurations remain unchanged. Collection manifests use collection-relative output paths and retain the original source bindings separately; they do not pretend that old manifests automatically point to relocated files.

These lists are **not runnable model releases**. Collection manifests are provenance records rather than accepted legacy loader adapters. New 3k bindings need review; 7k/8k/10k additionally need appropriately trained weights or explicitly migrated and retrained weights. A 6k checkpoint cannot generate additional coordinates merely by changing a gene list. No checkpoint or adapter was created, changed or included here.

## Reproduction and verification

The small frozen snapshots in `inputs/` total about 2.55 MB. They contain the gene statistics/ranks and the already-resolved literature-addition order; papers, HGNC payloads and expression data are not duplicated. Original absolute paths in [source_bindings.json](inputs/source_bindings.json) are provenance records. The builder reads its collection-local snapshots and sealed panel copies, so it does not need those external paths to reproduce the outputs.

From this directory, `python build_collection.py --output /path/to/rebuild` writes a separate data-package reconstruction; `python verify_collection.py --root /path/to/rebuild` checks it. Python with pandas and NumPy is required. [VERIFIED.json](VERIFIED.json) checks dimensions, identities, raw coordinates, nesting, TXT order, retained duplicate symbols, source-copy hashes, exact 3k/10k recipes, membership and the 6k→10k mapping. [DETERMINISM.json](DETERMINISM.json) records a bit-identical rebuild under the recorded runtime; it does not claim portability across untested library versions or biological performance.


## PBISC-pretrain portable bundle

This directory vendors the 20261015 collection as real files. The exact frozen
builder and all collection inputs and panel files are included. Current 2k/3k
and historical 2k/3k are retained because the original builder checks and emits
those lists; they are not additional training profiles in the 7k/8k/10k matrix.
The `review/` administrative records are omitted. No expression matrix,
checkpoint, raw Parquet metadata, downloaded paper or HGNC payload is bundled.

The source builder, `MANIFEST.json`, panel manifests, panel CSV/NPY/TXT files,
resolved literature order, gene-universe snapshot and historical verification
receipts are byte-exact copies. The additional
`inputs/ranked_gene_positions.npy` is the exact frozen full ranking named by
`inputs/source_bindings.json`. `inputs/raw_axis_contract.json` binds the compact
snapshot to the original raw-var and canonical-panel hashes without including
raw corpus data. `BUNDLE_MANIFEST.json` seals this portable payload; the
repository's `provenance/gene_context_bundle.json` records the source hashes.

Run either command from any working directory, with the paths adjusted to the
checkout location:

```bash
python /path/to/checkout/gene_context/build_collection.py --output /tmp/gene-context-rebuild
python /path/to/checkout/gene_context/verify_collection.py --root /tmp/gene-context-rebuild --receipt /tmp/gene-context-verified.json
```

The vendored verifier differs from its source only to make verification
read-only by default, permit an explicit receipt path, and avoid probing any
historical absolute `original_path`. The historical `VERIFIED.json` and
`DETERMINISM.json` describe the source collection's recorded Python 3.12 run;
current checks create a separate receipt and do not overwrite those records.
The original builder emits the original collection data and manifests; added
portable-bundle metadata and code are not copied into a reconstruction.

The original collection manifests remain provenance records. PBISC-pretrain
uses a separate strict collection loader binding before training; this does
not make the upstream `loader_ready: false` provenance field a claim that
expanded trained weights are included. Every matrix profile starts a fresh
run, and no trained 7k/8k/10k checkpoint is bundled.
