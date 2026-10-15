# Contributing

Read [AGENTS.md](AGENTS.md) before changing code or running training. Keep the
repository limited to the pretraining path and keep datasets and run artifacts
outside Git.

For a code change, describe the observable problem, the resulting behavior, and
the checks actually run. Run the relevant CPU tests and a CPU synthetic smoke
when the training/checkpoint path changes. GPU checks need an authorized GPU
allocation and applicable host-safety preflight; documentation edits do not
require launching CUDA.

Preserve data hashes, the three-state measurement contract, full-gene
normalization, recipe/split membership, and fail-closed resume validation. A
scientific configuration change is a new experiment: give it a new named
profile and explain the changed assumption. Do not silently edit a fixed
profile to make a run fit a machine.

Use small, reviewable diffs. Include dependency changes and environment evidence
when relevant. Do not include private paths, credentials, expressions, weights,
generated samples, or logs in commits. Keep [CLAUDE.md](CLAUDE.md) as a pointer to
the canonical [AGENTS.md](AGENTS.md).
