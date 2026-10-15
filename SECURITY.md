# Security and private artifacts

Report vulnerabilities to the repository owner through a private GitHub channel
available to authorized collaborators. Do not publish exploit details,
credentials, donor information, or private data paths in a public issue. No
separate security email address is asserted here.

Keep Hugging Face and GitHub credentials in their supported credential stores or
runtime environment. Never commit them or put them in command arguments, debug
logs, notebooks, reports, or generated manifests. If a credential is exposed,
revoke it and notify the owner; removing it from the current file alone is not a
complete response.

Load full PyTorch training checkpoints only from trusted sources: they may
contain pickled Python objects. Verify pinned data and checkpoint hashes before
use. The data restore path must reject unsafe archive paths and unexpected
members; do not replace it with an unrestricted `tar` extraction of an
unverified download.

Downloaded source data, counts, checkpoints, generated samples, and run logs
belong outside the repository. Preserve source-specific data access and rights
requirements even when an upstream download is publicly reachable.
