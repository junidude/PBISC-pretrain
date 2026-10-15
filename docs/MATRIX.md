# Gene-context and architecture matrix

The manifest `configs/matrix.json` enumerates exactly these fresh experiments:

| Genes | Baseline | Trunk 2x | Trunk 4x | Context 2x | Context 4x |
|---:|---|---|---|---|---|
| 7,000 | `7k-1x` | `7k-trunk2x` | `7k-trunk4x` | `7k-ctx2x` | `7k-ctx4x` |
| 8,000 | `8k-1x` | `8k-trunk2x` | `8k-trunk4x` | `8k-ctx2x` | `8k-ctx4x` |
| 10,000 | `10k-1x` | `10k-trunk2x` | `10k-trunk4x` | `10k-ctx2x` | `10k-ctx4x` |

Each profile lives in `configs/matrix/` and seals the shared relative base config,
its collection-local panel manifest, the complete collection index, and the raw
axis contract. The new profiles contain no source-server absolute paths.

## Widths and fixed settings

Baseline descending trunk widths are `[1024,1024,512,256]`, context width 512,
and time embedding 128. Trunk2x/4x multiply only the four trunk widths. Ctx2x/4x
set context to 1024/2048 and keep the baseline trunk. Context controls the
conditioning/time encoders, mixer, and conditioning projections into blocks.
The raw condition remains `3G+6` at a fixed gene count.

All 15 share seed 20260627, BF16, batch 256, four workers, accumulation one,
AdamW settings, EMA, validation policy, 16 slots and explicit top-ups. A 24 GiB
allocation safety ceiling is uniform across profiles. It is not a measured
H100 memory requirement. Preserve the full 20,097-gene normalization before
subsetting. A new panel or architecture is a new run, not a compatible resume.

## Parameter counts

These counts come from constructing the actual denoiser on PyTorch's metadata
device. They measure tensor shapes, not H100 memory peaks or training time.

| Genes | 1x | Trunk2x | Trunk4x | Ctx2x | Ctx4x |
|---:|---:|---:|---:|---:|---:|
| 7k | 61,867,184 | 121,021,616 | 266,069,168 | 77,804,208 | 114,396,848 |
| 8k | 69,549,184 | 134,847,616 | 292,183,168 | 87,022,208 | 126,686,848 |
| 10k | 84,913,184 | 162,499,616 | 344,411,168 | 105,458,208 | 151,266,848 |

## Bundled selection evidence

`gene_context/` contains the actual frozen files and reconstruction inputs,
including CSV/NPY/TXT axes and the original 6k mappings. Its original source
collection is retained with a portable verifier. The collection contains some
smaller/historical tiers needed for exact reconstruction; only the 15 rows above
are scheduled by this matrix.

The 7k/8k CSV and NPY files are byte-exact source copies. The 10k selection is
6,000 final baseline genes plus all 2,091 frozen literature-program additions
plus 1,909 remaining eligible train-HVG genes. It adds 91 literature-program
and 1,909 HVG genes beyond 8k. Ensembl ID is the unique key; repeated symbols
must not be deduplicated. Source exposure and endpoint-informed caveats remain.
The source tier manifests' `loader_ready: false` describe their original
provenance-package status. The separately reviewed v2 runtime binding validates
and accepts them without rewriting those historical manifests.

## Data and paths

For this matrix use `data plan`, `download`, and `verify` with `--corpus-only`.
All canonical count shards, splits, recipes, top-ups and conditioner are still
required. Repository-owned gene contexts are resolved independently and do not
require remote 7k/8k panel supplements. Canonical-corpus and legacy-with-panels
verification receipts are separate and cannot authorize each other's scope.

Profiles resolve paths relative to their installed asset tree. Runtime data and
output arguments resolve once relative to the caller's working directory. The
same tree works in a checkout, a moved checkout, or the wheel's shared assets.
Original absolute strings in frozen source provenance are not opened as paths.

```bash
pbisc-pretrain data download --corpus-only --data-root data/pbisc --cache-dir cache/pbisc
pbisc-pretrain data verify --corpus-only --data-root data/pbisc
pbisc-pretrain matrix prepare --data-root data/pbisc --output-root runs/ablation
pbisc-pretrain matrix run --data-root data/pbisc --output-root runs/ablation
```

## Sequential execution and recovery

The controller does not initialize CUDA. It starts one public training CLI
subprocess at a time, binds the selected profile, effective input, source code,
corpus relocation and bundled assets, and stops on the first nonzero child exit.
A controller lock prevents two controllers from writing the same output root.
Control files live under `runs/ablation/.matrix/`; each model writes its own
`runs/ablation/<ID>/` directory. Do not pre-create per-model run directories.

First-slot validation can target one model:

```bash
pbisc-pretrain matrix run --data-root data/pbisc --output-root runs/ablation \
  --only 7k-1x --stop-after-epoch 1
```

Inspect real full-model losses, memory and saved evidence, then continue the
same experiment selection explicitly:

```bash
pbisc-pretrain matrix run --data-root data/pbisc --output-root runs/ablation \
  --only 7k-1x --resume-existing
```

`--stop-after-epoch` is an absolute completed-slot target. The default is 16.
Omit `--only` to continue that run and start the remaining requested matrix entries.
Existing partial runs require `--resume-existing`, valid source/input bindings,
and their own trusted latest checkpoint. Fully verified runs already at the
target are skipped. Completion verification streams checkpoint hashes without
loading tensors; expect disk I/O up to roughly twice checkpoint size. Status
flags alone, altered artifacts, missing worker/commit proofs, stale code or
conflicting configurations cannot trigger an automatic skip. Failure is never
automatically retried. Resume occurs at completed epoch boundaries.

Legacy `configs/training/` profiles remain available for reproducing the earlier
private release and require their original with-panels data verification. They
are outside this new 15-run matrix.

## Verification

The update passed234 CPU tests. All15 profiles passed relocated checkout and installed-share layouts, including strict canonical-axis and source-collection checks. All15 denoisers were instantiated on the metadata device to inspect parameter shapes without allocating model weights. A separately installed wheel supplied all15 profiles and all bundled collection files from an unrelated working directory. See [matrix validation](../provenance/matrix_validation.json), [current source identities](../provenance/matrix_sources.json), and [independent review](MATRIX_REVIEW.md). NoH100CUDA or modeltraining was executed here.
