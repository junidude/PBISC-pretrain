# Source extraction and scientific contract

This repository extracts the PBISC-Diffusion pretraining dependency closure from
the existing portable release, with explicit updates from the active Model 8
implementation. It excludes adaptation campaigns, figure builders, manuscript
sources, downloaded expression data, and model checkpoints.

The machine-readable source map is
[`provenance/core_sources.json`](../provenance/core_sources.json). It records
original and packaged SHA-256 values, including portability changes. Original
absolute paths in that record and preserved configuration snapshots are historical
provenance; runtime paths are supplied through the verified dataset relocation.

## Preserved implementation

The denoiser, diffusion objective and sampler, EMA implementation, and proportion
hint math are byte-identical to the audited active source. Package-local
`model7_no_annotation` contains only the required data reader and conditioner;
the former Model 7 VAE is not imported or distributed here.

The portable All-K runner is essential. The plain ladder trainer alone would
change target-cell exposure. The preserved schedule contains:

| Component | Recipes | Target-cell occurrences |
|---|---:|---:|
| Registered training recipes | 131,072 | 193,644,000 |
| Explicit top-up recipes | 3,658 | 1,995,930 |
| Combined | 134,730 | 195,639,930 |

An occurrence can revisit a cell. These counts are not distinct corpus cells.
The 16 epochs are recipe schedule slots. The fixed stream uses seed 20260627,
batch size 256, four workers, accumulation one, and up to 4,000 target members per
recipe. Prefetch depth changes scheduling and must stay in the validated 1–64
range. Integer recipe seeds remain exact during concatenation.

## Explicit changes

- Package-local imports and authenticated, immutable-revision data transport
  replace dependencies on the original workspace.
- The exact sparse-row gatherer is retained to avoid repeated scattered Zarr
  reads. It preserves CSR values and requested row order.
- Resume loads model, EMA, and optimizer components with bounded CPU staging.
  Path relocation does not override numeric training contracts.
- The bounded All-K runner accepts the validated prefetch range used by the
  production profiles, including their depth of 32.
- Literature-derived 7k/8k manifests have a separate validation path. They are
  never relabelled as the older train-only HVG selector. Their coordinates,
  Ensembl IDs, original 6k retention, and artifact hashes are checked.
- Single-process launch checks reject unsupported distributed invocation. The
  public launcher retains managed-host GB10 guard and shared-lock requirements;
  an ordinary H100 host uses its own GPU allocation.

## Inputs, panels and normalization

The pinned corpus has 20,097 canonical genes. The full-gene bulk transform and
six scalar features are computed before selecting the model panel. For G genes,
the condition has 3G+6 values. Unmeasured genes remain excluded by masks;
unmeasured and measured-zero observations are distinct.

The original 6k profiles are retained as reference configurations. The nested
7k/8k panels are literature-derived research candidates; their feature coverage
does not establish improved generation. Existing 6k weights are not exact-resume
checkpoints for those larger tensor shapes. The source evidence and candidate
manifests retain their original hashes and methodological qualifications.

## What the tests establish

CPU tests check model/diffusion contracts, measurement masking, actual sparse E2
fixtures, all-member/top-up membership, exact integer seeds, expanded-panel
integrity, relocation, and optimizer/EMA/RNG continuation. Tests also exercise
download verification and incomplete-restoration rejection. The installed wheel
is exercised outside the checkout.

The data readback receipt is
[`provenance/dataset_download_verification.json`](../provenance/dataset_download_verification.json).
A real small count archive and all of its members were verified and restored.
This does not substitute for full-corpus verification on the destination host.

The original runs used a development PyTorch build on GB10. This repository's
CPU baseline uses PyTorch 2.9.0; its H100 environment specifies the official CUDA
12.8 wheel. Cross-platform bitwise GPU equivalence, H100 memory use, throughput,
and model quality require destination-host evidence. Record those observations
with Git commit, package versions, config hash, data revision, hardware, and
driver rather than interpreting CPU tests as H100 validation.
