"""Portable entry points for the preserved stable, ladder and All-K trainers."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .portable import prepare_config


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--conditioner-cache", type=Path)
    parser.add_argument("--trainer", choices=("stable", "ladder", "allcell"))
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--stop-after-epoch", type=int)
    parser.add_argument("--prepare-only", action="store_true", help="Validate and save configuration without loading counts or initializing CUDA")
    args = parser.parse_args(argv)
    from pbisc_sdd.experiment import require_single_process
    require_single_process()
    config, inferred = prepare_config(args.config, dataset_root=args.dataset_root,
                                     output_dir=args.output_dir, conditioner_cache=args.conditioner_cache)
    trainer = args.trainer or inferred
    if args.stop_after_epoch is not None and trainer != "allcell":
        parser.error("--stop-after-epoch is available only for the allcell trainer")
    out = Path(config["paths"]["output_dir"])
    out.parent.mkdir(parents=True, exist_ok=True)
    materialized = out.parent / f"{out.name}.input_config.json"
    encoded = json.dumps(config, indent=2, sort_keys=True) + "\n"
    if materialized.exists() and materialized.read_text() != encoded:
        raise FileExistsError(f"preserve conflicting training input: {materialized}")
    materialized.write_text(encoded)
    from pbisc_sdd import experiment, ladder_experiment
    loader = experiment if trainer == "stable" else ladder_experiment
    validated = loader.load_experiment_config(materialized)
    from model7_no_annotation.data_e2 import E2FullDataConfig, _validated_relocation
    _validated_relocation(E2FullDataConfig(**config["e2_data"]))
    if args.prepare_only:
        print(json.dumps({"status": "CONFIGURATION_VALIDATED", "trainer": trainer,
                          "config": str(materialized), "counts_validated": False}))
        return 0
    from .allcell import file_sha, managed_gate
    if str(validated["training"]["device"]) != "cpu":
        managed_gate()
    if trainer == "allcell":
        from . import allcell, allcell_roundrobin
        training_args = ["--config", str(materialized), "--expected-config-sha256", file_sha(materialized)]
        if args.resume:
            training_args.extend(["--resume", str(args.resume)])
        if args.stop_after_epoch is not None or args.resume is not None:
            # A released resume checkpoint embeds history, but cannot supply
            # prior worker-proof sidecars. Audit newly executed epochs through
            # the bounded runner instead of requiring proofs for earlier runs.
            stop = args.stop_after_epoch if args.stop_after_epoch is not None else int(validated["training"]["epochs"])
            training_args.extend(["--stop-after-epoch", str(stop)])
            allcell_roundrobin.main(training_args)
        else:
            allcell.main(training_args)
    else:
        result = loader.run_training(validated, resume_checkpoint=args.resume)
        print(json.dumps(result))
    return 0


__all__ = ["main", "prepare_config"]
