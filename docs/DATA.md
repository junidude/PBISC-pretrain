# Pinned data download and restoration

The dataset is a transport release of the existing E2 corpus. It is not an
ordinary Hugging Face table, and `datasets.load_dataset()` does not restore its
CSR Zarr count archives. Use the repository CLI so revision, hashes, shard
membership, masks, panels, and relocation remain bound together.

## Identity and storage

The committed [dataset lock](../manifests/dataset.lock.json) selects:

| Item | Value |
| --- | --- |
| Hugging Face dataset | `junidude14/PBISC-data` |
| Revision | `bfe88d1ff4cc1869946cdb7afda11f7927424ae5` |
| Transport assets | 1,346 assets; 111,510,563,626 bytes, about 103.84 GiB |
| Count shards | 432: 404 base shards and 28 strict overlay shards |
| Transport cache including supplemental assets | 111,518,298,563 bytes |
| Restored data including supplemental assets | 111,263,071,293 bytes |

Reserve **at least 250 GB free for cache plus restored data**, then add a separate
budget for checkpoints and run outputs. The planner calculates filesystem-specific
requirements and headroom from the lock; its result takes precedence over these
rounded planning numbers. Cache and restored data may be on separate disks. Do
not assume the transport archives disappear after extraction.

The pinned revision is a file identity, independent of the repository's current
default branch or visibility. At preparation time the dataset was publicly
reachable; future access may require authentication. Repository visibility does
not establish source-study redistribution permission. Preserve
`provenance/rights.tsv`; see [LICENSING.md](LICENSING.md).

## Download flow

Choose paths outside the checkout, then run:

```bash
pbisc-pretrain data plan --data-root /data/pbisc --cache-dir /scratch/pbisc-download
pbisc-pretrain data download --data-root /data/pbisc --cache-dir /scratch/pbisc-download
pbisc-pretrain data verify --data-root /data/pbisc --cache-dir /scratch/pbisc-download
```

`plan` reports required storage before the full transfer. `download` retrieves
the locked revision, checks assets, restores count archives, and installs the
required supplemental panel metadata. `verify` rehashes all declared restored
files, including individual count-archive members; it does not require the
transport cache. Read its report and require `status: TRAIN_READY` with
`complete: true` before training. Use the same paths
when continuing an interrupted download. Preserve failure reports and partial
files while investigating; do not manually rename an incomplete receipt into a
completed one.

If authentication is needed, log in with `hf auth login` or provide `HF_TOKEN`
through the job's runtime environment. Do not paste a token into commands,
tracked files, manifests, or logs. Use the actual dataset endpoint selected by
the lock. A Hugging Face account settings URL is not a download URL.

Hashing and extraction can require substantial disk I/O. A completed download
or metadata-only preparation does not prove that all restored count members are
present and intact. Reverification can be slow because it reads the required
bytes; do not replace it with an existence check.

Verification records `.pretrain/verification.json` and a hash-bound file ledger.
Training checks that the complete receipt matches the pinned lock and that file
size, timestamps, inode, and device seals still match. Moving or modifying the
restored root requires another full `data verify`. A bounded `--shard-id`
download/verification records `SUBSET_READY` and never enables training.

## Restored layout and preservation

The restored root contains the source/index metadata, recipe pools, donor
splits, top-up coordinates, conditioner cache, and count shards, plus the panel
assets used by these profiles. The transport preserves base Zarr v3 and strict
overlay Zarr v2 bytes rather than re-encoding their values.

```text
/data/pbisc/
  release.json
  relocation.json
  relocation.pretrain.json
  source/                 # strict source metadata and overlay counts
  base/                   # base CSR count shards
  index/                  # pools, split membership, and recipes
  topup/                  # explicit top-up coordinates
  conditioner/            # supplied full-panel conditioner cache
  panels/6k/              # pinned legacy panel
  panels/7k/              # pinned expanded panel
  panels/8k/              # pinned expanded panel
  provenance/             # rights, coverage, and implementation provenance
```

Original `release.json`, `relocation.json`, source manifests, and index manifests
remain byte-identical. `relocation.pretrain.json` extends supplemental asset
mappings without rewriting original scientific metadata. Historical absolute
paths in locked originals record provenance; the portable mapping resolves them
under the chosen data root. Recreating the original server directory tree is
unnecessary.

Restore verifies archive and member sizes/hashes and rejects unsafe paths,
links, duplicate members, missing members, and unexpected archive entries. Keep
that guarded restore path. Do not use unchecked extraction or convert fractional
expression values into integer counts.

## What verification establishes

Successful verification establishes the locked transport/restoration identity
that the command reports. It does not establish biological suitability,
publication rights, donor identity beyond the source metadata, or successful
full-corpus GPU execution. The source workspace has not been used here to
perform a second complete download and restore on an H100 machine.

Only a complete restored corpus is suitable for these fixed training profiles.
A partial shard restore, metadata-only copy, successful account login, or one
downloaded shard must not be described as full data readiness. Keep count
archives, restored data, caches, receipts, and logs outside Git.
