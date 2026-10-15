# Training environment

## Target H100 environment

The initial portable target is Linux x86_64, Python 3.11, glibc 2.28 or newer,
one allocated NVIDIA H100, and the official PyTorch 2.9.0 CUDA 12.8 wheel. This
is a proposed environment that requires a successful local CUDA smoke test;
the source host's CPU checks do not establish H100 execution.

Create an isolated environment after obtaining the GPU allocation:

```bash
python3.11 -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch==2.9.0 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements-h100.txt
python -m pip install -e '.[dev]'
python -m pip check
pbisc-pretrain doctor
```

The CUDA wheel command follows the [official PyTorch 2.9.0 installation matrix](https://pytorch.org/get-started/previous-versions/).
Install the PyTorch wheel first so a generic dependency resolution does not
silently select another CUDA build. `nvidia-ml-py==13.580.82` provides NVML access
for device and memory observation; it is distinct from PyTorch CUDA execution.

The [NVIDIA CUDA 12.8 Update 1 release notes](https://docs.nvidia.com/cuda/archive/12.8.1/cuda-toolkit-release-notes/index.html)
document driver compatibility. R570 with Linux driver 570.124.06 or newer is a
conservative recommended baseline for this environment; it is not a universal
minimum for every CUDA minor-compatibility configuration. Follow the cluster's
driver policy. Record the actual driver and `torch.version.cuda`; the CUDA value
shown by `nvidia-smi` describes driver capability, not the installed PyTorch
runtime.

The trainer's precision behavior is part of the run configuration. PyTorch's
[AMP documentation](https://docs.pytorch.org/docs/2.9/amp.html) describes the
underlying mixed-precision API. Do not change precision, enable a new compiler,
or substitute an optimizer to fit hardware without recording a new experiment.

## CPU verification

For CPU-only tests, install the official CPU wheel in a separate environment:

```bash
python3.11 -m venv .venv-cpu
. .venv-cpu/bin/activate
python -m pip install --upgrade pip
python -m pip install torch==2.9.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e '.[dev]'
python -m pip check
pytest -q
pbisc-pretrain smoke --output-dir /tmp/pbisc-pretrain-smoke --device cpu
```

`doctor` reports installed software and hardware without initializing CUDA; it
does not validate CUDA execution. Run `smoke --device cuda` only inside the
allocated GPU job and after applicable
host-safety checks. The synthetic smoke uses a tiny model and generated inputs;
it checks the training/checkpoint path without downloading expression matrices.

## Resource planning and evidence

Full 8k-trunk4x H100 peak memory, throughput, and training quality are unmeasured
until a real run is recorded. H100 memory capacity varies by model; check the
allocated device rather than assuming every H100 has the same capacity. Avoid
starting another job on the same GPU. Keep the data-loader workers' host RAM and
filesystem cache in the memory budget, and provision transport plus restored
data space as described in [DATA.md](DATA.md).

The historical training environment used Python 3.12 and
`torch==2.12.0.dev20260306+cu128` on a GB10. A clean PyTorch 2.9.0 H100 run is a
separate environment, so bit-identical continuation is not promised. Preserve
the Git commit, dependency report, GPU/driver report, command, config hash, data
verification result, logs, and checkpoint metadata with each run.

When a smoke or training run fails, retain its output directory and report the
failure. Inspect memory evidence before a restart. Do not automatically reduce
batch size, alter the panel, or enter a restart loop.
