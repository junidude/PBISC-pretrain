# Code publication checklist

The owner made this repository public on 28 November 2026 under the MIT License,
as the companion of the PBISC code repository. Publication remains an explicit
owner action: passing tests does not authorize a release.

Before a public release:

- Review the final tracked-file allowlist. Exclude count matrices, archives,
  downloaded metadata with restricted information, model weights, samples, logs,
  credentials, new runtime machine paths, caches, and run outputs. Review any
  historical paths retained inside checksum-bound source provenance explicitly.
- Choose the PBISC-authored code license and preserve the exact upstream MIT
  notice. Review study-specific rights independently of the code license.
- Confirm repository title, author/corporate-author metadata, version, citation,
  and any paper/DOI links with the owner. Do not invent publication metadata.
- Pin the data revision and every required asset hash. Confirm a new machine can
  obtain the intended files using the documented download endpoint and access
  method. A settings page or an account login is not a data download test.
- Run CPU tests, check links and CLI examples, and inspect the diff for secrets.
  Record the actual evidence. A full H100 run may only be described as verified
  if its hardware/software, data verification, smoke, training, and resume
  evidence exists.
- Tag the reviewed code commit and preserve the environment and verification
  records. Publish only the explicitly approved code/assets, and verify access
  as a reader afterward.

Do not rewrite frozen research releases or edit the manuscript as part of this
checklist. Data and model publication need their own reviewed scope.
