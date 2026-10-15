# PBISC-pretrain matrix independent review

Review scope: repository-owned 7k/8k/10k gene-context profiles, portable loader, canonical-corpus readiness, serial matrix controller, and package asset inclusion. This report is outside the training repository. No repository source edits, Git mutations, CUDA, training, external raw-data access, or network calls were performed by this reviewer.

## Review status

Data/profile and post-fix matrix-runner review completed. No open blockers remain in this bounded scope. The continuation-example correction has also been verified in the final documentation.

## Verified evidence

- `tests/test_matrix_portability.py`: **13 passed in 7.76s**, using the repository `.venv`, hidden CUDA devices, one CPU BLAS/OpenMP thread, disabled bytecode writes and pytest cache. Includes relocated checkout and wheel-layout fixtures, explicit collection-schema dispatch, corpus-axis binding, path escape rejection, and strict numeric resume compatibility.
- Independent static check of all 15 committed profiles: profile SHA-256 values match the matrix; each actual model override equals the declared trunk/context multiplier; only the panel count and declared architecture axis change. Baseline `[1024,1024,512,256]`, context 512, time 128; all profiles retain 16 slots, seed 20260627, batch 256 and four workers.
- Independent packaging-glob check: all **56** gene-context payload files are covered by setuptools `data-files` destinations. This establishes declared wheel coverage; an actual wheel build was not performed by this reviewer.
- Scratch canonical-corpus probe: removing a verified ledger entry and updating both the receipt count and ledger SHA still fails with `verification ledger does not cover the full pinned corpus`.
- Scratch subset probe: changing subset receipt flags to `TRAIN_READY`, `complete=true` and the full shard count still fails against hash-bound complete-release metadata.
- Portable v2 preparation now ignores `relocation.pretrain.json`, and resolves its panel payload from the repository collection while the canonical count roots and metadata remain under the restored dataset root.

## Actionable findings sent to coordinator

1. **P2, closed: matrix binding included the unused legacy relocation extension.** At the first runner snapshot, `_binding()` hashed `data/relocation.pretrain.json` whenever present. This made adding, removing, or changing an irrelevant legacy overlay extension invalidate a v2 matrix preparation/resume. The final matrix binding excludes the extension and has regression coverage, as detailed below.

2. **Provenance finding, closed:** the first `_binding()` recorded profiles, base config, panel payload and selected metadata but no executing trainer/loader Python implementation digests. The final binding records all active runtime Python module sources and enforces the roundrobin trainer digest when verifying completion, as detailed below.

3. **P2, closed: declared and actual architecture axes were not compared.** Correct committed profiles were independently confirmed, but a resealed changed profile could disagree with its matrix row. The final runner checks actual trunk, context and time widths before writing bindings or dispatching children, with regressions for all three mismatch cases.

## Initial reviewed snapshots

| File | SHA-256 |
| --- | --- |
| `src/pbisc/training/portable.py` | `211a1924490c2751e05d1687ae57b86f2401ba7f2ce509cf28d6cde35f1e4c53` |
| `src/pbisc_pretrain/dataset.py` | `19889a0b675ac795d863905042a196f7ccaa267e02f62a2e0d8e1f10ccd9c341` |
| `src/pbisc_pretrain/matrix.py` | `6e10a100f518c4b8d03129a87384f9cf649a5ee82f04beafd20d07870971b6ce` |
| `configs/matrix.json` | `dde8b4fd6a9bc0dc00b0e7db9e977b2cb0a150e3bd79077b8b32c1c9c09f7a90` |
| `pyproject.toml` | `4895c0fb232f2bda30eb030a39256d38b54dddbb1421fa792ac6106df0bd0db4` |

## Post-fix review

- Focused final CPU suite: `tests/test_matrix_runner.py tests/test_matrix_definition.py tests/test_matrix_portability.py` — **56 passed in 9.96s**. Same CPU-only environment constraints as above; subprocess execution is stubbed in the runner tests.
- The unused legacy relocation extension is absent from matrix input bindings. The regression covers changing that irrelevant file without changing the prepared v2 run binding.
- Source identity is now bound across all **31 active runtime Python module sources** in `pbisc`, `pbisc_pretrain`, `pbisc_sdd`, and `model7_no_annotation`. Independent execution confirmed the discovery and hashing process does not import `torch`.
- Source changes are rejected before train dispatch and when reading completion evidence. Segment and epoch commit receipts must match the bound roundrobin trainer SHA-256. Tests cover altered trainer/loader/model/collection/controller source and a source change during preparation.
- Runtime `hidden_dims`, `context_dim`, and time embedding are checked against each matrix row before a control binding is written or a child is launched. Resealed profiles with mismatched dimensions fail closed. The original architecture-axis validation finding is closed.
- All 15 profiles now explicitly pin the complete collection `MANIFEST.json`; portable preparation checks it and the strict runtime loader checks it again against `gene_context_collection_contract`. Thus cross-tier source files used by the collection validator come from a profile-pinned index. Raw-axis, corpus fingerprint, endpoint-informed exposure and selection caveats remain independently checked.
- The runner validates the complete selected plan before launching any child, resolves user paths once against the invocation directory, uses one synchronous child process at a time, rejects unbound outputs and partial implicit recovery, stops immediately on child failure, and requires actual checkpoint/history/worker/epoch proof hashes for completion skipping.
- Packaging declarations still cover all 56 bundled payload files. This reviewer did not run an actual wheel build or actual-corpus metadata check; those are assigned to the coordinator.

### Documentation correction verified

The initial `docs/MATRIX.md` continuation example omitted the selector after demonstrating a single-model first slot. The final example now includes `--only 7k-1x --resume-existing`; the next paragraph explicitly describes omitting `--only` to start the remaining matrix entries. Independently reread and verified after the documentation owner's correction.

### Final reviewed source snapshots

| File | SHA-256 |
| --- | --- |
| `src/pbisc/training/portable.py` | `1c79b38ca92221206d1d39e25ba265296f9547f1eeaaaf03c73b5acd69892137` |
| `src/pbisc/training/panel_contracts.py` | `d4f928f97ab1f50615e399b86feb2c33eac2fbf1b62e1dce66379192c58e7672` |
| `src/pbisc_pretrain/dataset.py` | `19889a0b675ac795d863905042a196f7ccaa267e02f62a2e0d8e1f10ccd9c341` |
| `src/pbisc_pretrain/matrix.py` | `f02bc0d3577efd21f9e82063822a76cc65d4200d9ff9106884c1bbf5dee13470` |
| `src/pbisc_sdd/collection_panel.py` | `6881a278f6407e91665e67ba3a6e15b83c3a94dd4a670f8a6771818b3f0d5962` |

## Verdict

**PASS within the stated scope. No open actionable blockers.** All reported code findings are closed by implementation and regression evidence, and the documentation selector issue is closed. Final source SHA-256 values were rechecked after the coordinator's freeze signal and remain identical to the tested snapshots above. No CUDA, full-model training, H100 resource, biological quality, actual wheel install, or full-corpus restoration claim follows from this review. The coordinator separately reported 234 full-suite tests passing in 25.60s and lint/repository scans passing; those broader checks were not rerun by this reviewer.
