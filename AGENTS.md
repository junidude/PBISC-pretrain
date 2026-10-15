# PBISC-pretrain agent instructions

This file is the canonical instruction source for all coding and training
agents in this repository. `CLAUDE.md` must point here rather than duplicate it.
Higher-priority user and host instructions still apply.

## Scope and autonomy

Keep this repository focused on reproducible PBISC pretraining. Preserve the
pinned data revision, hashes, measurement semantics, numeric configuration, and
checkpoint contract. Do not silently substitute datasets, panels, models,
precisions, or training schedules. Keep manuscript, adaptation, evaluation, and
deployment changes outside this task unless requested.

Within an explicitly authorized and allocated training job, an agent should
complete installation, download, verification, preparation, a bounded smoke
test, and training without asking again at every reversible step. A GPU being
visible is not a resource allocation. Respect the operator's scheduler, GPU
assignment, time, storage, and spending limits. Do not kill another job.

## First steps on a new machine

1. Read `README.md` and `docs/ENVIRONMENT.md`; inspect the actual CLI with
   `pbisc-pretrain --help` and each relevant subcommand's `--help`.
2. Record Git commit, Python/dependency versions, GPU and driver information,
   available RAM/storage, and allocation limits. `pbisc-pretrain doctor` avoids
   CUDA initialization by default.
3. Follow the pinned data `plan` → `download` → `verify` flow in `docs/DATA.md`.
   Require complete verification before claiming the training corpus is ready.
4. Run CPU smoke, then CUDA smoke inside the authorized GPU allocation. Preserve
   reports and failures. A tiny synthetic smoke is not a full-corpus validation.
5. Prepare the selected profile, run one bounded schedule slot, inspect outputs
   and memory, then resume within the authorized allocation. See
   `docs/TRAINING.md` for interruption and resume semantics.

Use the 8k-trunk4x profile as the proposed starting candidate unless the operator
selected another profile. Its biological quality and full H100 resource use are
unproven. Report evidence without upgrading synthetic, CPU, or preparation-only
results into a successful H100 training claim.

## GPU and memory safety

Run one training/inference/evaluation CUDA job per allocated GPU. This trainer
is single GPU and rejects `WORLD_SIZE > 1`; do not invoke `torchrun`, DDP, or a
multi-GPU launcher. On ordinary H100 machines, use the site's scheduler or
resource manager, select the assigned device, and preserve room for the OS and
data workers. Stop on memory failure and inspect evidence before restarting.
Do not create automatic OOM/restart loops or change batch size to conceal a
failure in a fixed profile.

### Managed GB10 source host

On the managed `sj_server_1` NVIDIA DGX Spark/GB10 host, or any environment
governed by its `ml-memory-guard`, this policy overrides generic launch examples.
GB10 CPU and GPU share unified memory; cgroup limits alone do not contain CUDA
allocations. Before anything that initializes CUDA or may consume substantial
memory, run all three checks:

```bash
systemctl --user is-active ml-memory-guard.service
/home/sj_server_1/.local/libexec/ml-memory-guard --check
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader
```

If the guard is inactive, run
`systemctl --user restart ml-memory-guard.service` once, repeat the checks, and
report failure if it remains unavailable. Launch GPU work only through
`ml-run COMMAND [ARG ...]`
or an already verified equivalent that uses `ml.slice`, the shared
`/run/user/1000/ml-gpu0.lock`, `MemorySwapMax=0`, and `OOMPolicy=kill`. CPU data
work that may consume substantial memory also needs the guard preflight.

Do not launch CUDA directly with Python, `torchrun`, `accelerate`, `deepspeed`,
`vllm`, `nohup`, `setsid`, or tmux. Use tmux only to observe logs. Do not use sudo
or root-owned GPU containers. The coordinating agent owns GPU launches;
subagents must not launch GPU commands independently. Queue when the shared
lock or a compute PID is present.

Never disable the guard, remove its lock, enable ML swap, lower a job's OOM
score, or raise host memory limits without explicit operator approval. Training
uses a positive OOM score adjustment. Preserve checkpoints, append-only logs,
manifests, and status files after any termination. Inspect live policy files
instead of relying on old limits:

- `/home/sj_server_1/.config/systemd/user/ml.slice`
- `/home/sj_server_1/.config/systemd/user/ml-memory-guard.service`
- `/home/sj_server_1/.local/libexec/ml-memory-guard`
- `/home/sj_server_1/.local/bin/ml-run`

## Scientific and checkpoint invariants

- Missing/unmeasured genes are NA, never measured zero. Preserve measurement
  masks, row/gene order, train/validation membership, recipe seeds, and top-ups.
- Normalize over the full 20,097-gene source representation before choosing a
  training panel. Do not renormalize the selected panel.
- Preserve the `3G + 6` conditioning input and measured-only loss behavior.
- Original all-cell schedule: seed 20260627, batch size 256, four data workers,
  accumulation 1, 4,000 training cells per registered recipe, 16 schedule slots.
  Slots are not whole-corpus passes. See `docs/REPRODUCIBILITY.md` for accounting.
- Resume only trusted full training checkpoints with compatible numeric/model
  and data bindings. Numeric differences fail closed. Path relocation does not
  authorize a scientific change. Resume is at epoch boundaries; interrupted
  epochs replay. Different panel/model shapes require a fresh run.

## Changes and verification

Keep changes narrow and review the diff. Run the relevant CPU tests and smoke
checks before claiming completion; do not start CUDA to validate documentation
or metadata changes. Preserve upstream attribution and audit records. Record
what was actually run, on which hardware and software, and what remains untested.
Do not claim identical CUDA trajectories across environments from matching
seeds alone.

Never commit data, count archives, credentials, checkpoints, generated samples,
run logs, local environment files, or new runtime machine-specific paths. Locked
historical paths are retained only as source provenance. Read credentials
only at runtime. Public repository visibility and publication are owner actions;
do not change them automatically. Preserve study-specific rights and do not
invent authors, a DOI, a license, or permission to redistribute data.
