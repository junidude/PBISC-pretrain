# PBISC-pretrain

Minimal, single-GPU PBISC diffusion pretraining code with a pinned data download,
portable training profiles, and checkpoint resume. This repository is intended
to be cloned onto a separate training machine; it does not require the original
research workspace.

The recommended starting candidate is **8k genes with a 4× trunk**. Its quality
and full-run H100 memory requirements have not been established. The 6k and 7k
profiles support controlled comparisons. A CPU smoke test verifies a small
synthetic training contract; it does not validate full-corpus training or model
quality.

## Install on an allocated H100 machine

Use Linux x86_64, Python 3.11, glibc 2.28 or newer, and an NVIDIA driver compatible
with CUDA 12.8. The initial target is one H100; multi-GPU/DDP is unsupported.
See [environment details](docs/ENVIRONMENT.md) before running on another host.

```bash
git clone git@github.com:junidude/PBISC-pretrain.git
cd PBISC-pretrain
python3.11 -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch==2.9.0 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements-h100.txt
python -m pip install -e '.[dev]'
pbisc-pretrain doctor
```

`doctor` does not initialize CUDA by default. Installation and CPU checks are
separate from successful H100 execution. Preserve the resulting environment
report and run the GPU smoke test below inside the allocated job.

## Download and verify the pinned corpus

The CLI uses the dataset revision and hashes committed in this repository.
Choose separate transport-cache, restored-data, and run directories with ample
free space. The full transport plus extracted corpus needs approximately twice
the corpus storage; see [data and storage](docs/DATA.md) for the exact inventory.

```bash
pbisc-pretrain data plan --data-root /data/pbisc --cache-dir /scratch/pbisc-download
pbisc-pretrain data download --data-root /data/pbisc --cache-dir /scratch/pbisc-download
pbisc-pretrain data verify --data-root /data/pbisc --cache-dir /scratch/pbisc-download
```

Hugging Face account settings are not a download endpoint. If access requires
authentication, use `hf auth login` or supply `HF_TOKEN` through the runtime
environment. Never place tokens in source, command arguments, or logs. Access
and public redistribution rights are separate; see [licensing](docs/LICENSING.md).

## Prepare, smoke-test, and train

Run these CUDA commands only within your GPU allocation, with one job using the
GPU. On the managed GB10 source host, use the mandatory guarded launcher in
[AGENTS.md](AGENTS.md).

```bash
pbisc-pretrain smoke --output-dir /work/pbisc-smoke-cpu --device cpu
pbisc-pretrain smoke --output-dir /work/pbisc-smoke-h100 --device cuda
pbisc-pretrain prepare --config configs/training/8k-trunk4x.json \
  --data-root /data/pbisc --output-dir /work/runs/8k-trunk4x
pbisc-pretrain train --config configs/training/8k-trunk4x.json \
  --data-root /data/pbisc --output-dir /work/runs/8k-trunk4x \
  --stop-after-epoch 1
```

Inspect the first completed slot's logs, checkpoint, finite losses, and memory
before continuing. Resume the same run to execute the remaining schedule:

```bash
pbisc-pretrain train --config configs/training/8k-trunk4x.json \
  --data-root /data/pbisc --output-dir /work/runs/8k-trunk4x \
  --resume /work/runs/8k-trunk4x/checkpoint_latest.pt
```

An epoch is one of **16 schedule slots**, not one full-corpus pass. Resume starts
at a completed epoch boundary; an interrupted epoch is replayed. The 7k and 8k
models have different shapes and cannot resume a 6k checkpoint. Read the
[training runbook](docs/TRAINING.md) and [scientific contract](docs/REPRODUCIBILITY.md)
before changing a profile.

## Repository scope

This repository contains pretraining code and metadata. Downloaded expression
matrices, datasets, checkpoints, generated samples, and run logs stay outside
Git. It does not include adaptation, evaluation claims, or a manuscript release.

- [AGENTS.md](AGENTS.md): canonical instructions for coding and training agents;
  [CLAUDE.md](CLAUDE.md) points to it.
- [Environment](docs/ENVIRONMENT.md), [data](docs/DATA.md),
  [training](docs/TRAINING.md), and [reproducibility](docs/REPRODUCIBILITY.md).
- [Contributing](CONTRIBUTING.md), [security](SECURITY.md),
  [licensing](docs/LICENSING.md), and [publication checklist](docs/PUBLISHING.md).
- [Source audit](docs/SOURCE_AUDIT.md), [validation evidence](docs/VALIDATION.md),
  and [independent review](docs/INDEPENDENT_REVIEW.md).

PBISC-authored code has no newly assigned blanket license. The upstream MIT
notice is retained in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
