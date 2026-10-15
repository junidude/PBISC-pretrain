# PBISC-pretrain

Single-GPU PBISC-Diffusion pretraining with a pinned corpus, bundled gene
contexts, relative training profiles, and verified checkpoint resume.

The requested experiment matrix has **15 fresh runs**: 7k, 8k and 10k genes,
each with baseline 1x, trunk2x, trunk4x, ctx2x and ctx4x. Actual gene CSVs,
Ensembl IDs, symbols, coordinate arrays, selection manifests and reconstruction
inputs are committed under [`gene_context/`](gene_context/README.md).

| Variant, for each gene panel | Descending trunk widths | Context width |
|---|---|---:|
| `1x` | 1024, 1024, 512, 256 | 512 |
| `trunk2x` | 2048, 2048, 1024, 512 | 512 |
| `trunk4x` | 4096, 4096, 2048, 1024 | 512 |
| `ctx2x` | 1024, 1024, 512, 256 | 1024 |
| `ctx4x` | 1024, 1024, 512, 256 | 2048 |

Time embedding stays 128. Each architecture ablation changes one width axis.
All runs use the same sealed 16-slot All-K plus top-up schedule. The profiles
and gene contexts work in a relocated checkout or installed wheel. Source and
historical manifests retain their provenance; runtime does not consult the
original research host.

## Install on one allocated H100

Use an authenticated GitHub account with private-repository access. The target
is Linux x86_64, Python 3.11, glibc 2.28 or newer, and a CUDA 12.8-compatible
NVIDIA driver. Multi-GPU/DDP is unsupported.

```bash
git clone https://github.com/junidude/PBISC-pretrain.git
cd PBISC-pretrain
python3.11 -m venv .venv
. .venv/bin/activate
python -m pip install torch==2.9.0 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements-h100.txt
python -m pip install -e '.[dev]'
pbisc-pretrain doctor
```

See [environment details](docs/ENVIRONMENT.md). The managed GB10 source host
must use the guard and `ml-run` policy in [AGENTS.md](AGENTS.md).

## Download the corpus and verify the bundled contexts

All example data and output paths are relative to the invocation directory.
Allow at least 250 GB for transport/restoration, plus checkpoint storage for
15 runs. Gene contexts come from this repository; `--corpus-only` downloads the
fixed full canonical corpus without the old remote panel supplements.

```bash
pbisc-pretrain data plan --corpus-only --data-root data/pbisc --cache-dir cache/pbisc
pbisc-pretrain data download --corpus-only --data-root data/pbisc --cache-dir cache/pbisc
pbisc-pretrain data verify --corpus-only --data-root data/pbisc
python gene_context/verify_collection.py --root gene_context
```

The corpus must be `TRAIN_READY` in canonical-corpus scope. An isolated shard
or metadata-only restoration cannot train. Authentication, when needed, uses
normal HF credentials or runtime `HF_TOKEN`; no credentials are committed.

## Prepare and execute the 15-run matrix

```bash
pbisc-pretrain matrix list
pbisc-pretrain smoke --device cpu --output-dir runs/smoke-cpu
pbisc-pretrain smoke --device cuda --output-dir runs/smoke-h100
pbisc-pretrain matrix prepare --data-root data/pbisc --output-root runs/gene-context
pbisc-pretrain matrix run --data-root data/pbisc --output-root runs/gene-context
```

Run CUDA commands only inside the authorized GPU allocation. The matrix uses
one subprocess at a time and stops on failure. The default target is 16 schedule
slots per run. `--only 7k-1x` selects one run; repeat `--only` to select several.
Use `--stop-after-epoch 1` for a bounded first slot when validating the full
model. After inspecting a stopped or interrupted run, explicitly authorize
continuation with `--resume-existing`. Verified completed targets are skipped;
unknown, conflicting or damaged outputs are rejected.

See the [matrix runbook](docs/MATRIX.md), [training semantics](docs/TRAINING.md),
and canonical [AGENTS.md](AGENTS.md). `CLAUDE.md` points to the same instructions.

## Evidence and publication

The 10k panel contains 6,000 baseline genes, all 2,091 frozen literature additions
and 1,909 train-HVG additions. This mixed selection is not a pure HVG size
ablation, and feature coverage does not establish rare-cell recovery. Expanded
panels and changed widths need fresh training.

See [source audit](docs/SOURCE_AUDIT.md), [matrix validation](provenance/matrix_validation.json), [independent matrix review](docs/MATRIX_REVIEW.md),
[reproducibility](docs/REPRODUCIBILITY.md), [licensing](docs/LICENSING.md), and the
[publication checklist](docs/PUBLISHING.md). Datasets, credentials, checkpoints,
generated cells and run logs stay outside Git. PBISC licensing remains an owner
decision; upstream notices are retained.
