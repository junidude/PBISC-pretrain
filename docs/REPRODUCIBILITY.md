# Preserved pretraining contract

These profiles retain the existing E2/all-cell data and diffusion training
semantics. Portability changes paths and packaging; it does not authorize
rebuilding splits, recipes, normalization, or scientific settings. Source/audit
evidence should be consulted alongside the executable tests and locked metadata.

## Three-state measurement semantics

Each gene coordinate is measured-and-detected, measured zero, or unmeasured/NA.
An unmeasured coordinate must never become a measured zero. Preserve source
measurement masks and keep supervised losses restricted to measured coordinates.
The value-loss profile is `all_measured`; measured zeros remain valid observed
values. Do not infer gene availability from a zero-filled expression matrix.

Preserve original fractional values, sparse structure, row/gene order, donor
splits, sample-pool row indices, exact uint64 recipe seeds, and explicit top-up
coordinates. Do not round expression values, merge donor identities across
studies, regenerate split membership, or rebuild the conditioner/panel silently.

Condition normalization uses the **full 20,097-gene source panel before the
training gene subset is selected**. Normalizing a 6k/7k/8k panel independently
changes library-size factors and scalar features. Preserve the selected bulk
expression features, availability and bulk-measurement features, and six scalar
features: the final condition input has dimension `3G + 6` for G training genes.

## Fixed All-K schedule

| Setting | Value |
| --- | ---: |
| Seed | 20260627 |
| Batch size | 256 |
| Data-loader workers | 4 |
| Gradient accumulation | 1 |
| `train_cells_per_block` parameter | 4,000 |
| Registered recipes per schedule slot | 8,192 |
| Schedule slots | 16 |
| Registered recipes across the schedule | 131,072 |
| Registered target-cell exposures | 193,644,000 |
| Additional top-up recipes | 3,658 |
| Additional top-up target-cell exposures | 1,995,930 |
| Total planned target-cell exposures | 195,639,930 |

The 4,000-cell setting is a sampling parameter, not a claim that every
registered recipe contributes exactly 4,000 distinct cells. The exposure totals
come from the sealed coverage/top-up metadata. Exposures are training targets,
not a count of unique biological cells.

One epoch corresponds to one schedule slot. **Sixteen slots are not sixteen
complete corpus passes.** Preserve the registered recipe and top-up accounting,
worker ownership, validation behavior, and the schedule's boundary semantics.
Do not replace this schedule with a generic shuffled DataLoader over every cell.

## Models and expanded panels

The 6k x1 and 6k trunk4x profiles preserve their bound original configurations.
The 7k and 8k profiles use the pinned expanded panels and the same trunk4x widths.
Panel positions, selection manifests, selected-gene tables, and old-to-new
mappings are tracked as small metadata assets. Their hashes bind panel identity.
Expanded panels create different model shapes and therefore require fresh
training, not a renamed 6k checkpoint.

The 8k-trunk4x profile is a proposed starting candidate. Panel coverage and code
contract tests do not establish improved rare-cell performance, downstream
quality, or a successful full H100 pretraining run.

## Evidence levels

- A source audit and CPU unit tests support preserved implementation behavior.
- A synthetic smoke supports the tiny model/optimizer/checkpoint path.
- Data hash verification supports the reported byte identity and completeness.
- Preparation supports profile, path, and artifact binding checks.
- A successful H100 bounded run supports the specific hardware/software/data
  path executed; a full schedule and resume require their own recorded evidence.

Keep these conclusions separate when reporting progress. No matching seed,
single smoke, or preparation-only result proves CUDA numerical identity or
biological quality.
