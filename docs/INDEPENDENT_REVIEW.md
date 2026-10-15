> Historical review of the initial `ff4d88f` / v0.1.0 release. Its source SHA table applies to that release; the 15-run v0.2.0 matrix has separate current review evidence.

# Independent PBISC-pretrain review

Reviewed repository: `/home/sj_server_1/PBISC/PBISC-pretrain`.
Review date: 2026-10-16. Review role: read-only independent source/contract review;
this report is the only file written by this reviewer.

**Verdict: PASS for the reviewed code and bounded CPU contracts after the P1
finding below was corrected. No unresolved functional blocker was found.** This
does not certify a complete HF restoration, a destination-machine H100 run, full
schedule execution, GPU numerical parity, memory capacity, or biological quality.
The coordinator is separately validating final wheel contents and execution
outside the checkout; this report does not substitute for that check.

## Resolved finding

### P1 — documented bounded first slot and every resume rejected production profiles

The original reviewed `src/pbisc/training/allcell_roundrobin.py` required
`training.prefetch_factor == 1`. All four committed production profiles use 32.
`src/pbisc/training/__init__.py:53` sends `--stop-after-epoch` and every resume
through this runner, so the documented first-slot command failed with
`All-cell stream contract differs` before starting the data/model loop.

The final runner imports and calls the shared `validate_stream_contract`, which
accepts the supported range 1–64 while retaining seed, batch size, worker count,
accumulation, and all-member constraints. The emitted pairing now records the
actual prefetch depth. The change preserves the mathematical and data stream
contracts rather than silently changing the original profile to depth 1.

Regression: `tests/test_training_allcell_roundrobin.py:359` runs bounded `main`
with each actual committed production config and confirms that prefetch 32
passes the stream gate and reaches the next loading boundary. Its metadata and
count loading are deliberately bounded test seams; this is not real-corpus GPU
training evidence. The reviewer reran the updated bounded runner and CLI tests:
27 passed in 2.34 seconds.

## Independently checked contracts

- Traced public CLI training through `assert_training_ready`, profile binding,
  portable configuration, and the full or bounded All-K runner. Full receipts
  reconstruct exact expected files from the locked manifests. `SUBSET_READY`, a
  changed restored file, or a ledger with changed hashes does not authorize
  training. Full count members are hashed during verification; later starts
  check the recorded file seals.
- Reviewed immutable HF revision and hashes, separate non-nested cache/data
  roots, disk budgeting, verified file reuse, restoration after interruption,
  guarded archive member extraction, unsafe member/link/path rejection, and
  full-versus-subset readiness. `relocation.pretrain.json` extends the preserved
  original map and binds its original SHA rather than rewriting frozen source
  or index manifests.
- Checked both original 6k config snapshots against the live source paths in
  `provenance/core_sources.json`: each is byte-identical. All four production
  profiles bind their original configs by SHA and retain 16 slots, 8,192
  registered recipes per slot, seed 20260627, batch 256, four workers, and
  accumulation one. The All-K/top-up implementations retain exact uint64 seeds,
  full registered K targets, explicit top-up coordinates, and the total
  195,639,930 target occurrences. These are exposures, not unique-cell counts.
- Independently ran `validate_expanded_panel` for the committed 7k and 8k
  configurations against the actual canonical
  `model_7_no_annotation_sh_hwang/data/e2_full_source/metadata/var.parquet`,
  actual historical selected-gene tables, and actual old-to-new maps. Both
  passed exact Ensembl ID, raw coordinate, tensor order, canonical-axis hash,
  locked corpus fingerprint, and original6k identity/order checks. The expanded
  schema remains separate from the legacy train-only HVG schema, with exposure
  and endpoint caveats retained.
- Confirmed full-panel normalization precedes panel selection:
  `SubsetAnnotationFreeConditioner.transform` transforms the complete source
  representation and then selects genes while retaining six scalar features.
  Measurement masks remain separate from measured-zero values, and conditioning
  remains `3G + 6`.
- Confirmed `model.py`, `diffusion.py`, `ema.py`, and `proportion_hint.py` are
  byte-identical to both the extraction source and the current active Model 8
  source. All recorded packaged source hashes match final files.
- Reviewed epoch-boundary resume, model/optimizer/EMA/RNG/history restoration,
  numeric/model/data compatibility checks, same-run latest-checkpoint routing,
  preservation of immutable epoch snapshots and failed-attempt evidence, and
  newly executed worker-proof checks. Changed panel dimensions require a fresh
  run. CPU continuation tests cover the bounded state transition order; this
  does not establish identical CUDA trajectories across environments.
- Reviewed trusted-pickle boundaries: full training checkpoints and the
  conditioner cache use explicit trusted PyTorch artifact loading; the public
  data gate binds the supplied conditioner hash before training. Documentation
  states that resume checkpoints must be trusted.
- Reviewed single-process rejection before training device initialization and
  the public GB10 CUDA launch check. The latter now requires the managed gate,
  guard `--check`, no existing CUDA compute PID, and an active kernel FLOCK for
  the exact shared lock inode/device held by the process launch ancestry. This
  matches the installed `ml-run` flock-parent layout. CPU fixtures verify the
  lock matcher; no CUDA launch was used to validate it. Ordinary H100 hosts do
  not inherit source-host systemd service requirements.
- Reviewed H100 instructions and dependencies: one GPU, official PyTorch 2.9.0
  CUDA 12.8 wheel installed first, pinned scientific dependencies, and
  `nvidia-ml-py` for the trainer's utilization/temperature calls. These are a
  proposed installation target and require destination-host execution.

## Reviewer execution evidence

Environment: repository `.venv`, Python 3.11, `torch==2.9.0+cpu`,
`numpy==1.26.4`, `pytest==9.0.2`, `zarr==3.1.5`.
Tests ran with `CUDA_VISIBLE_DEVICES=''` and BLAS/OpenMP/MKL thread limits of 1.

1. Dataset restoration/gating, expanded-panel, portable resume, all-cell, and
   round-robin test files: **67 passed in 5.17 seconds**.
2. Updated public CLI and bounded runner test files after the prefetch and
   launch-lock changes: **27 passed in 2.34 seconds**. These runs overlap and
   should not be summed as a distinct-test count.
3. Actual small-panel metadata checks, exact 6k source-config comparison, and
   source math/provenance SHA checks described above passed independently.

The coordinator separately reported 166 full CPU tests passing in 13.13 seconds
and successful sdist/wheel construction. The reviewer did not independently run
the entire suite, install the final wheel, download the complete HF corpus, read
full expression shards, initialize CUDA, or run training/inference. The later
coordinator wheel verification and bounded data readback receipts retain their
own evidence scope.

## Final reviewed core identities

| File | SHA-256 |
|---|---|
| `src/pbisc/training/allcell.py` | `0b254b1f18db3f39a417f6499281167d36b13320bfe8e3cad7438d5ca20b26ce` |
| `src/pbisc/training/allcell_roundrobin.py` | `46486bb2b2ebf3035666dbc46cc2e8c35ef44588b7c5d6511618108fea935aff` |
| `src/pbisc/training/portable.py` | `6e44c0e60842d8867acf4a694c5fbb6805cdc30b7516b01b9569cc0cf0569f4e` |
| `src/pbisc_sdd/expanded_panel.py` | `6c84a8d7d16b22024702a516ae801202d358c590a725e6ace3127d9576b1429a` |
| `src/pbisc_pretrain/dataset.py` | `ff31a7251e22d2d3a72f4969e1590fdd6424c75d928d8c9d0643fd017dc28d85` |
| `src/pbisc_pretrain/safety.py` | `cc4403a757e0286305608d9548d6acb417026f54de1f37defbac993012128866` |
| `manifests/dataset.lock.json` | `d37dff04a83858ca2c4140d5317ef94a4804de1400833eac9242707d7a267ba3` |
| `provenance/core_sources.json` | `35eea14b671e3f24e77dc6af07efbcc147edb3b13cc5f5f8621f5f60e118eb0c` |

## Publication scope

The authored-code license remains an explicit owner decision, and source-study
rights remain separate from repository/data accessibility. `docs/LICENSING.md`
and `docs/PUBLISHING.md` state these boundaries and retain the upstream MIT
notice. Pending owner choices do not block the authorized private GitHub
repository. A public code-availability release and public visibility change
still require the owner decisions documented there; no license, author list,
publication DOI, data permission, or H100 validation should be inferred from
this review.
