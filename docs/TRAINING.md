# Training and resume runbook

This is a single-GPU trainer. Use one allocated GPU and one training job. The
entry point rejects `WORLD_SIZE > 1`; do not use `torchrun` or DDP. For the
managed GB10 source host, follow the guard checks and `ml-run` requirement in
[AGENTS.md](../AGENTS.md) before any CUDA work.

## Select a profile

| Profile | Genes | Descending trunk widths | Purpose |
| --- | ---: | --- | --- |
| `configs/training/6k-x1.json` | 6,000 | 1024, 1024, 512, 256 | Original-width comparison |
| `configs/training/6k-trunk4x.json` | 6,000 | 4096, 4096, 2048, 1024 | Wider-trunk comparison |
| `configs/training/7k-trunk4x.json` | 7,000 | 4096, 4096, 2048, 1024 | Expanded-panel candidate |
| `configs/training/8k-trunk4x.json` | 8,000 | 4096, 4096, 2048, 1024 | Recommended starting candidate |

The 7k/8k profiles are candidates; wider coverage is not evidence of better
downstream performance. Full H100 memory, throughput, and training quality need
measurement. Profiles preserve the all-cell schedule and named scientific
settings; changes require a separately named experiment.

Each portable profile binds its underlying config and panel metadata by hash.
Use `--data-root` to relocate the locked corpus and `--output-dir` to choose a
new run directory. Do not edit original absolute paths or remove hash checks.
Installed packages also provide the named profiles; the CLI resolves
`configs/training/...` outside a checkout through its installed shared assets.

## Prepare and run a bounded first slot

After full data verification and environment checks:

```bash
pbisc-pretrain smoke --output-dir /work/pbisc-smoke-cpu --device cpu
pbisc-pretrain smoke --output-dir /work/pbisc-smoke-h100 --device cuda
pbisc-pretrain prepare --config configs/training/8k-trunk4x.json \
  --data-root /data/pbisc --output-dir /work/runs/8k-trunk4x
pbisc-pretrain train --config configs/training/8k-trunk4x.json \
  --data-root /data/pbisc --output-dir /work/runs/8k-trunk4x \
  --stop-after-epoch 1
```

`smoke` exercises a tiny synthetic model, an optimizer step, and checkpoint
save/reload. It does not exercise the real count-shard pipeline. `prepare` checks
profile/artifact bindings and prepares the portable run. It does not execute
training or replace complete data verification. `train --prepare-only` offers
the equivalent preparation-only route.

`--stop-after-epoch N` means stop once completed epoch N has been reached; it is
an absolute boundary, not N additional epochs on resume. An epoch is a schedule
slot. Check the command report, finite training/validation losses, checkpoint
write, worker proofs, and available GPU/host memory after the first slot. Keep
the logs and environment evidence with the run.

## Continue the same run

Resume the trusted latest checkpoint with the same profile and output directory:

```bash
pbisc-pretrain train --config configs/training/8k-trunk4x.json \
  --data-root /data/pbisc --output-dir /work/runs/8k-trunk4x \
  --resume /work/runs/8k-trunk4x/checkpoint_latest.pt
```

Omitting `--stop-after-epoch` runs the remaining configured schedule. To perform
only one more slot after epoch 1, use `--stop-after-epoch 2`. Resume validation
retains the model, optimizer, EMA, RNG, history, and completed schedule position.
Only documented path relocation is permitted; changed numeric configuration,
model shape, panel/data identity, or incompatible checkpoint content fails
closed. Full PyTorch resume checkpoints must come from a trusted source.

The 7k and 8k models have different input/output shapes. A 6k checkpoint cannot
resume either profile, and a 7k checkpoint cannot resume 8k. Start a fresh run
directory for a different panel or model. A weights-only export is not a full
training-resume checkpoint.

## Interruption and failure

Checkpoint resume is **at completed epoch boundaries**. A partially completed
epoch is replayed from the last completed checkpoint; there is no promise of
mid-epoch continuation. Preserve the interrupted run and its append-only
evidence. Inspect the failure cause before resuming. Do not delete worker
proofs, modify receipts, or overwrite a checkpoint to make validation pass.

For a guard kill, OOM, corrupt data, non-finite loss, or disk failure, inspect the
corresponding logs and memory/storage evidence before any restart. Do not enable
automatic restart loops or silently reduce batch size/workers/precision. Record
any deliberate new experiment separately.

Matching seeds and checkpoint contents do not guarantee bit-identical CUDA
trajectories across PyTorch, CUDA, hardware, or kernel versions. See
[ENVIRONMENT.md](ENVIRONMENT.md) and preserve the actual environment per run.
