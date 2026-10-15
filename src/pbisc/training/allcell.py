#!/usr/bin/env python3
"""Train immutable SDD core on every K-member of the sealed recipe pool."""
from __future__ import annotations

import argparse
import copy
from dataclasses import replace
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
from typing import Any

import numpy as np

POLICY = "all-train-first-then-validation-v1"


def digest(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def file_sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for part in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(part)
    return h.hexdigest()


def write_json(path: Path, value: object) -> None:
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temp.replace(path)


def pool_digest(pools: dict[str, list[dict[str, Any]]], keys: list[str]) -> str:
    records = []
    for key in sorted(keys):
        records.append({"sample_key": key, "segments": [
            {"shard_id": str(p["shard_id"]), "length": len(p["row_indices"]),
             "row_indices_sha256": hashlib.sha256(np.asarray(p["row_indices"], dtype="<i8").tobytes()).hexdigest()}
            for p in pools[key]]})
    return digest(records)


def batch_count(worker_cells: list[int], batch_size: int) -> int:
    return sum(math.ceil(n / batch_size) for n in worker_cells)


def combine_recipe_frames(registered, topups, epoch: int):
    """Preserve every uint64 seed exactly when adding rows without RNG seeds."""
    import pandas as pd
    original_seeds = [int(v) for v in registered["rng_seed"]]
    main = registered.copy()
    main["rng_seed"] = main["rng_seed"].astype(object)
    main["is_topup"] = False
    extras = topups[topups["schedule_slot"].astype(int).eq(epoch)].copy()
    extras["rng_seed"] = pd.Series([0] * len(extras), index=extras.index, dtype=object)
    extras["is_topup"] = True
    combined = pd.concat([main, extras], ignore_index=True)
    combined["stream_recipe_index"] = np.arange(len(combined), dtype=np.int64)
    if ([int(v) for v in combined["rng_seed"].iloc[:len(main)]] != original_seeds
            or any(not isinstance(v, (int, np.integer)) for v in combined["rng_seed"])):
        raise RuntimeError("Recipe concatenation changed integer RNG seed values")
    return combined


def load_topup_block(module, recipe, rng, topup_cells, shard_starts, global_pools):
    """Load explicit same-sample members using the original count/mask semantics."""
    from pbisc_sdd.data import MeasurementAwareRecipeBlock
    begin, end = int(recipe["topup_start"]), int(recipe["topup_stop"])
    ids = np.asarray(topup_cells[begin:end], dtype=np.int64)
    k = int(recipe["k"])
    universe = int(shard_starts[-1]) + int(global_pools[-1]["n_cells"])
    if (begin < 0 or end > len(topup_cells) or len(ids) != k or not 1 <= k <= 4000
            or len(np.unique(ids)) != k or np.any(ids < 0) or np.any(ids >= universe)):
        raise ValueError("Explicit top-up cell range or unique membership differs")
    segments = np.searchsorted(shard_starts, ids, side="right") - 1
    refs = [(int(seg), int(cell - shard_starts[seg])) for seg, cell in zip(segments, ids)]
    full = module.base._load_sparse_refs(refs, global_pools).tocsr()
    bulk_log = np.log1p(np.asarray(full.sum(axis=0), dtype=np.float64).reshape(-1)).astype(np.float32)
    selected = full[:, module.source_gene_indices].tocsr()
    chosen = rng.choice(k, size=k, replace=False)
    masks = np.stack([module._measurement_mask(global_pools[refs[int(i)][0]]) for i in chosen]).astype(np.float32)
    counts = selected[chosen].toarray().astype(np.float32)
    fraction = np.zeros(len(module.source_gene_indices), dtype=np.float64)
    for seg, count in zip(*np.unique(segments, return_counts=True)):
        fraction += int(count) * module._measurement_mask(global_pools[int(seg)])
    block = MeasurementAwareRecipeBlock(str(recipe["block_id"]), counts, masks,
                (fraction / k).astype(np.float32), bulk_log, k)
    if np.any(counts[masks == 0] != 0):
        raise RuntimeError("Top-up has expression at an unmeasured gene")
    return block, ids[chosen]


def validate_stream_contract(training: dict[str, Any], data: dict[str, Any]) -> int:
    """Check the sealed all-cell stream contract and return the loader prefetch depth.

    Seed, cells per block, accumulation, batch size and the 4-worker sharding fix
    the collated batch sequence. ``prefetch_factor`` only bounds how far each
    worker runs ahead (the loader still returns batches in worker round-robin
    order), so any depth in [1, 64] yields the identical stream; depth 1
    serialized the workers, hence larger depths are allowed since 2026-09-16.
    """
    if (int(training["seed"]) != 20260627 or int(data["train_cells_per_block"]) != 4000
            or int(training["gradient_accumulation_steps"]) != 1
            or int(training["batch_size"]) != 256 or int(training["num_workers"]) != 4):
        raise ValueError("All-cell stream contract differs")
    prefetch_factor = int(training["prefetch_factor"])
    if not 1 <= prefetch_factor <= 64:
        raise ValueError("All-cell stream prefetch_factor must be in [1, 64]")
    return prefetch_factor


def managed_gate() -> None:
    # Apply the host policy only where its compensating guard is installed.
    if not (Path.home() / ".local/libexec/ml-memory-guard").is_file():
        return
    groups = [line[3:] for line in Path("/proc/self/cgroup").read_text().splitlines() if line.startswith("0::")]
    if len(groups) != 1 or "ml.slice" not in Path(groups[0]).parts:
        raise RuntimeError("Training must be inside ml.slice")
    group = Path("/sys/fs/cgroup") / groups[0].lstrip("/")
    if (group / "memory.swap.max").read_text().strip() != "0":
        raise RuntimeError("Training swap must be zero")
    services = [part for part in Path(groups[0]).parts if part.endswith(".service")]
    if not services:
        raise RuntimeError("A managed training service is required")
    policy = subprocess.check_output(["systemctl", "--user", "show", services[-1], "--property=OOMPolicy", "--value"], text=True).strip()
    if policy != "kill":
        raise RuntimeError("OOMPolicy must be kill")
    subprocess.run(["systemctl", "--user", "is-active", "--quiet", "ml-memory-guard.service"], check=True)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--expected-config-sha256", required=True)
    parser.add_argument("--resume", type=Path, help="Manual recovery only: this run's checkpoint_latest.pt")
    args = parser.parse_args(argv)
    if file_sha(args.config) != args.expected_config_sha256:
        raise ValueError("Configuration seal differs")
    raw = json.loads(args.config.read_text())
    out = Path(raw["paths"]["output_dir"]).resolve()
    if out.exists() and args.resume is None:
        raise FileExistsError(f"No overwrite or implicit resume: {out}")
    if args.resume is not None and (args.resume.resolve() != out / "checkpoint_latest.pt" or not args.resume.is_file()):
        raise ValueError("Resume must use the existing run's latest checkpoint")
    data = raw["data"]
    audit_path = Path(data["coverage_audit_summary"])
    if file_sha(audit_path) != data["coverage_audit_sha256"]:
        raise ValueError("Coverage audit changed")
    audit = json.loads(audit_path.read_text())
    if audit.get("status") != "PASS" or audit["train_universe"]["target_exposures"] != 193644000:
        raise ValueError("All-cell coverage audit did not pass")
    if data.get("priming_policy") != POLICY:
        raise ValueError("Unsupported priming policy")
    topup_path = Path(data["topup_manifest"])
    if file_sha(topup_path) != data["topup_manifest_sha256"]:
        raise ValueError("Top-up manifest changed")
    topup = json.loads(topup_path.read_text())
    from .portable import resolve_artifact
    for field in ("cell_ids", "recipes"):
        topup[field + "_path"] = str(resolve_artifact(topup[field + "_path"], raw))
    if (topup.get("status") != "PASS" or topup.get("schema_version") != "pbisc_sdd.allcell_topup.v1"
            or topup["coverage_summary_sha256"] != data["coverage_audit_sha256"]):
        raise ValueError("Top-up coverage contract differs")
    for field in ("cell_ids", "recipes"):
        if file_sha(Path(topup[field + "_path"])) != topup[field + "_sha256"]:
            raise ValueError(f"Top-up artifact changed: {field}")
    offsets_path = topup_path.parent / "shard_offsets.csv"
    if file_sha(offsets_path) != topup["artifacts_sha256"]["shard_offsets.csv"]:
        raise ValueError("Top-up source coordinate map changed")
    from pbisc_sdd.experiment import require_single_process
    require_single_process()
    if str(raw["training"].get("device", "auto")) != "cpu":
        managed_gate()
    import torch
    from torch.utils.data import get_worker_info
    from pbisc_sdd import experiment as stable, ladder_experiment as ladder
    from pbisc_sdd.data import PBISCSparseCellDataset
    import pandas as pd

    config = ladder.load_experiment_config(args.config)
    training = config["training"]
    profile = bool(data.get("profile_worst_k", False))
    resume_epoch = 0
    if args.resume is not None:
        if profile:
            raise ValueError("Profiles do not resume")
        resume_metadata = ladder.load_checkpoint(args.resume, device=torch.device("cpu"))
        ladder.validate_resume_config(config, resume_metadata["experiment_config"])
        resume_epoch = int(resume_metadata["epoch"]) + 1
        resume_metadata.clear()
        if resume_epoch >= int(training["epochs"]):
            raise ValueError("Run already completed its epoch target; verify instead of resuming")
        recovery = out / "recovery_records" / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        recovery.mkdir(parents=True, exist_ok=False)
        for name in ("RUN_STATUS.json", "ALLCELL_DATA_CONTRACT.json"):
            path = out / name
            if path.exists():
                (recovery / name).write_bytes(path.read_bytes())
        existing_observations = out / "data_observation"
        for path in existing_observations.glob("*.json"):
            row = json.loads(path.read_text())
            if int(row.get("epoch", 0)) > resume_epoch or path.name.startswith("progress_"):
                path.rename(recovery / path.name)
    prefetch_factor = validate_stream_contract(training, data)
    if not profile and (int(training["epochs"]) != 16 or int(training["max_train_steps"]) != 0
                        or int(data["train_recipes"]) != 8192):
        raise ValueError("Full run must exhaust all16 recipe slots")
    if profile and (int(training["epochs"]) != 1 or int(training["max_train_steps"]) != 128):
        raise ValueError("Profile must use one128-step interval")
    proof: dict[str, Any] = {}
    observations = out / "data_observation"
    topup_cells = np.load(topup["cell_ids_path"], mmap_mode="r", allow_pickle=False)
    topup_recipes = pd.read_csv(topup["recipes_path"])
    if len(topup_cells) != int(topup["added_target_cells"]) or len(topup_recipes) != int(topup["topup_recipes"]):
        raise ValueError("Top-up cardinality differs")

    def explicit_block(module, recipe, rng):
        return load_topup_block(module, recipe, rng, topup_cells,
                                proof["shard_starts"], proof["global_pools"])

    class AllCellsDataset(PBISCSparseCellDataset):
        def recipes_for_epoch(self, epoch: int):
            recipes = self.data_module.split_manifest(self.split, max_blocks=8192,
                seed=self.seed + 1000003 * epoch, platform=self.platform)
            if profile:
                recipes = recipes[recipes["k"].astype(int).eq(4000)].head(16).reset_index(drop=True)
                if len(recipes) != 16:
                    raise ValueError("Worst-K profile requires16 K4000 recipes")
                recipes["is_topup"] = False
            else:
                recipes = combine_recipe_frames(recipes, topup_recipes, epoch)
            return recipes

        def __len__(self):
            with self._shared_epoch.get_lock():
                epoch = int(self._shared_epoch.value)
            return int(self.recipes_for_epoch(epoch)["k"].sum())

        def __iter__(self):
            worker = get_worker_info()
            worker_id = 0 if worker is None else int(worker.id)
            workers = 1 if worker is None else int(worker.num_workers)
            with self._shared_epoch.get_lock():
                epoch = int(self._shared_epoch.value)
            recipes = self.recipes_for_epoch(epoch).iloc[worker_id::workers].reset_index(drop=True)
            self.data_module.prime_pools(recipes[~recipes["is_topup"].astype(bool)])
            rng = np.random.default_rng(self.seed + 1000003 * epoch + 7919 + 104729 * worker_id)
            targets = recipe_count = 0
            membership = hashlib.sha256()
            recipe_order = hashlib.sha256()
            for _, recipe in recipes.iterrows():
                k = int(recipe["k"])
                shadow = np.random.default_rng()
                shadow.bit_generator.state = copy.deepcopy(rng.bit_generator.state)
                chosen = shadow.choice(k, size=k, replace=False)
                if bool(recipe["is_topup"]):
                    block, ids = explicit_block(self.data_module, recipe, rng)
                else:
                    refs, pools = self.data_module.base._recipe_refs_and_pools(recipe)
                    block = self.data_module.load_training_block(recipe, 4000, rng)
                    ids = np.asarray([proof["shard_offsets"][str(pools[segment]["shard_id"])] + int(row)
                                      for segment, row in refs], dtype="<i8")[chosen]
                if block.counts.shape[0] != k or rng.bit_generator.state != shadow.bit_generator.state:
                    raise RuntimeError("All-K membership or target permutation differs")
                rid = str(recipe["recipe_id"]).encode("ascii")
                membership.update(rid + b"\0" + ids.astype("<i8", copy=False).tobytes())
                recipe_order.update(rid + b"\n")
                condition = self.conditioner.transform(block.bulk_log1p, block.n_cells)
                for counts, availability in zip(block.counts, block.availability_mask, strict=True):
                    targets += 1
                    yield {"x_log": np.log1p(counts).astype(np.float32, copy=False),
                           "condition": np.asarray(condition, dtype=np.float32),
                           "availability_mask": availability.astype(np.float32, copy=False),
                           "bulk_measurement_fraction": block.bulk_measurement_fraction,
                           "block_index": np.int64(recipe["stream_recipe_index"]),
                           "block_id": block.block_id, "recipe_k": np.int64(block.n_cells)}
                recipe_count += 1
                if recipe_count % 32 == 0:
                    write_json(observations / f"progress_worker_{worker_id}.json",
                               {"epoch": epoch + 1, "recipes": recipe_count, "target_cells": targets})
            path = observations / f"epoch_{epoch+1:03d}_worker_{worker_id}.json"
            if path.exists():
                raise FileExistsError(f"Preserve existing worker proof: {path}")
            write_json(path, {"epoch": epoch + 1, "worker": worker_id, "recipes": recipe_count,
                             "target_cells": targets, "target_order_sha256": membership.hexdigest(),
                             "recipe_order_sha256": recipe_order.hexdigest(), "status": "EXHAUSTED"})

    def build_streams(*positional, **kwargs):
        streams = stable.build_streams_from_config(*positional, **kwargs)
        base = streams.data_module.base
        if base.corpus_fingerprint != audit["corpus_fingerprint"]:
            raise RuntimeError("Runtime source corpus differs from coverage audit")
        if base._pools_by_sample:
            raise RuntimeError("Global priming requires a fresh empty pool cache")
        train = base.recipes[base.recipes["split"].astype(str).eq("train")]
        keys = sorted(set(train["sample_key"].astype(str)))
        base.prime_pools(train)
        observed_pool_hash = pool_digest(base._pools_by_sample, keys)
        if observed_pool_hash != audit["pool_order_sha256"]:
            raise RuntimeError("Runtime train pool order differs from audited order")
        validation_recipes = base.split_manifest("evaluation", max_blocks=int(data["validation_recipes"]),
                                               seed=streams.validation.seed)
        if set(validation_recipes["sample_key"].astype(str)) & set(keys):
            raise RuntimeError("Train/validation sample pools overlap")
        base.prime_pools(validation_recipes)
        if pool_digest(base._pools_by_sample, keys) != observed_pool_hash:
            raise RuntimeError("Validation priming changed train pool membership")
        offset, offsets = 0, {}
        for shard_id in sorted(base._shards_by_id):
            offsets[str(shard_id)] = offset
            offset += int(base._shards_by_id[shard_id]["n_cells"])
        saved_offsets = pd.read_csv(offsets_path)
        expected_offsets = [{"shard_id": str(key), "global_offset": offsets[str(key)],
                             "n_cells": int(base._shards_by_id[key]["n_cells"])}
                            for key in sorted(base._shards_by_id)]
        if saved_offsets[["shard_id", "global_offset", "n_cells"]].to_dict("records") != expected_offsets:
            raise RuntimeError("Runtime source coordinates differ from explicit top-up map")
        proof["shard_offsets"] = offsets
        proof["shard_starts"] = np.asarray(list(offsets.values()), dtype=np.int64)
        proof["global_pools"] = [base._shards_by_id[key] for key in sorted(base._shards_by_id)]
        observations.mkdir(parents=True, exist_ok=args.resume is not None)
        target = AllCellsDataset(streams.data_module, split="train", recipes_per_epoch=8192,
                                cells_per_block=4000, conditioner=streams.conditioner,
                                seed=20260627, progress=False)
        schedules, expected_cells, expected_steps, worker_cells = [], [], [], []
        for epoch in range(int(training["epochs"])):
            recipes = target.recipes_for_epoch(epoch)
            counts = [int(recipes.iloc[w::4]["k"].sum()) for w in range(4)]
            worker_cells.append(counts)
            expected_cells.append(sum(counts))
            expected_steps.append(batch_count(counts, 256))
            schedules.append({"epoch": epoch+1, "recipe_ids": recipes["recipe_id"].astype(str).tolist(),
                              "rng_seeds_decimal": [str(int(v)) for v in recipes["rng_seed"]],
                              "target_cells_by_worker": counts})
        total_expected = 193644000 + int(topup["added_target_cells"])
        if not profile and (sum(expected_cells) != total_expected
                or len({rid for s in schedules for rid in s["recipe_ids"]}) != 131072 + int(topup["topup_recipes"])):
            raise RuntimeError("Full schedule coverage differs from all sealed recipes")
        pairing = {"source_corpus_fingerprint": base.corpus_fingerprint,
                   "source_recipe_sha256": base.build_metadata["recipe_parts_content_sha256"],
                   "source_pool_sha256": base.build_metadata["sample_pools_sha256"],
                   "pool_order_sha256": observed_pool_hash, "recipe_schedule_sha256": digest(schedules),
                   "coverage_audit_sha256": data["coverage_audit_sha256"], "seed": 20260627,
                   "topup_manifest_sha256": data["topup_manifest_sha256"],
                   "target_policy": "all-registered-K-plus-unseen-train-topup-v1",
                   "epochs": int(training["epochs"]), "recipes_per_epoch": 8192,
                   "registered_recipes_per_epoch": 8192,
                   "total_recipe_counts_by_epoch": [len(s["recipe_ids"]) for s in schedules],
                   "cells_per_block": 4000, "batch_size": 256, "gradient_accumulation_steps": 1,
                   "num_workers": 4, "prefetch_factor": prefetch_factor,
                   "prefetch_factor_note": "scheduling depth only; batch sequence independent of it",
                   "priming_policy": POLICY}
        proof.update(pairing=pairing, expected_target_cells_by_epoch=expected_cells,
                     expected_optimizer_steps_by_epoch=expected_steps, worker_cells=worker_cells)
        write_json(out / "ALLCELL_DATA_CONTRACT.json", {"schema_version": "pbisc_sdd.allcell_data_contract.v1",
                   "status": "PROFILE_RUNNING" if profile else "RUNNING", "pairing": pairing,
                   "expected_target_cells_by_epoch": expected_cells,
                   "expected_optimizer_steps_by_epoch": expected_steps,
                   "original_recipe_coverage": audit["train_universe"],
                   "planned_coverage_after_topup": topup["summary"]})
        schedule_path = out / "EPOCH_SCHEDULE.json"
        if schedule_path.exists():
            if json.loads(schedule_path.read_text()) != schedules:
                raise ValueError("Resume epoch schedule differs")
        else:
            write_json(schedule_path, schedules)
        return replace(streams, train=target)

    result = ladder.run_training(config, resume_checkpoint=args.resume, _stream_builder=build_streams)
    if profile:
        write_json(out / "ALLCELL_PROFILE.json", {"status": "COMPLETED", "result": result,
                   "data_contract": "16 worst-K recipes available; step-limited profile, not a complete epoch",
                   "config_sha256": args.expected_config_sha256})
        print(json.dumps(result), flush=True)
        return
    history = json.loads((out / "history.json").read_text())
    observed_cells, observed_steps, worker_hashes = [], [], []
    for epoch in range(16):
        rows = [json.loads((observations / f"epoch_{epoch+1:03d}_worker_{w}.json").read_text()) for w in range(4)]
        if any(r["status"] != "EXHAUSTED" or r["target_cells"] != proof["worker_cells"][epoch][w] for w,r in enumerate(rows)):
            raise RuntimeError("Worker stream coverage failed")
        observed_cells.append(sum(r["target_cells"] for r in rows))
        observed_steps.append(int(history["epochs"][epoch]["train"]["optimizer_steps"]))
        worker_hashes.append([r["target_order_sha256"] for r in rows])
    if observed_cells != proof["expected_target_cells_by_epoch"] or observed_steps != proof["expected_optimizer_steps_by_epoch"]:
        raise RuntimeError("Observed cell/optimizer counts differ from complete schedule")
    pairing = {**proof["pairing"], "emitted_target_order_sha256": digest(worker_hashes)}
    write_json(out / "ALLCELL_DATA_CONTRACT.json", {"schema_version": "pbisc_sdd.allcell_data_contract.v1",
               "status": "COMPLETED", "pairing": pairing,
               "expected_target_cells_by_epoch": proof["expected_target_cells_by_epoch"],
               "observed_target_cells_by_epoch": observed_cells,
               "expected_optimizer_steps_by_epoch": proof["expected_optimizer_steps_by_epoch"],
               "observed_optimizer_steps_by_epoch": observed_steps, "total_target_cells": sum(observed_cells),
               "original_recipe_coverage": audit["train_universe"],
               "coverage_after_topup": topup["summary"]})
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
