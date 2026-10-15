> Initial v0.1.0 validation snapshot. Current matrix checks are recorded separately.

# Validation evidence

The initial preparation passed **166 CPU tests** using Python 3.11.14 and
PyTorch 2.9.0+cpu. A wheel and source distribution were built. The wheel was
installed in a separate environment and exercised from outside the checkout:
all four production profile seals, the pinned data lock, packaged agent guide,
and synthetic model/optimizer/EMA continuation passed.

The synthetic smoke performs two optimizer updates, saves and restores model,
optimizer, EMA and RNG state, reproduces the second update exactly, and checks
finite sampled values and zero output at unavailable genes. This is a small
software contract test, not a quality benchmark.

A real 1.4 MB count archive was downloaded from the pinned HF revision and all
nine members were hash-verified and restored. Repeating that subset restoration
reused verified files without network calls. The full-training gate rejected
the subset. An additional read-only CPU slice used 64 actual target cells with
8,000 genes, the full canonical conditioner, measurement context, and a tiny
hidden-width diffusion forward; it passed without a backward or optimizer step.

See [local validation](../provenance/local_validation.json),
[source identities](../provenance/core_sources.json),
[data readback](../provenance/dataset_download_verification.json), and the
[independent review](INDEPENDENT_REVIEW.md). The independent review found and
closed a production-prefetch mismatch in the bounded/resume runner.

The GitHub workflow repeats CPU tests, source scanning, and installed-wheel
checks on Linux x86_64. Consult the check attached to the exact commit rather
than treating an earlier successful check as evidence for later code.

**Not executed in this preparation:** full corpus restoration on the destination
host, H100 CUDA/BF16 smoke, full-sized model memory profiling, or pretraining.
The destination agent must verify the complete restored data and execute its
allocated H100 smoke before beginning the full schedule. CPU success does not
establish GPU numerical equivalence or biological recovery.
