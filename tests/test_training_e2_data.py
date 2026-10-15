from __future__ import annotations

import importlib.util
import hashlib
import json
from dataclasses import replace
from pathlib import Path
import shutil
import sys

import numpy as np
import pandas as pd
import pytest
from scipy import sparse
import zarr


from model7_no_annotation.data_e2 import (  # noqa: E402
    E2FullDataConfig,
    E2IndexContractError,
    _csr_rows,
    _resolve_source_uri,
    build_e2_full_data_module,
    build_e2_full_datasets,
    fit_e2_full_conditioner,
)


from pbisc.training import build_e2_index as BUILDER


def _write_csr_zarr(path: Path, matrix: sparse.csr_matrix) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    group = zarr.open_group(str(path), mode="w")
    group.create_array("data", data=matrix.data.astype(np.float32), chunks=(17,))
    group.create_array("indices", data=matrix.indices.astype(np.int32), chunks=(17,))
    group.create_array("indptr", data=matrix.indptr.astype(np.int64), chunks=(7,))


def _dataset_rows(dataset_id: str, *, scale: bool) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    donors = ["single"] if scale else ["d0", "d1", "d2", ""]
    for donor_number, donor in enumerate(donors):
        for sample_number in range(2):
            for _ in range(6):
                rows.append(
                    {
                        "dataset_id": dataset_id,
                        "donor_id": donor,
                        "sample_id": f"s{donor_number}-{sample_number}",
                        "source_row_index": len(rows) + 100,
                        "row_index": len(rows),
                    }
                )
    return pd.DataFrame(rows)


@pytest.fixture(scope="module")
def tiny_e2(tmp_path_factory: pytest.TempPathFactory) -> dict[str, object]:
    root = tmp_path_factory.mktemp("e2-data")
    strict_root = root / "strict"
    base_root = root / "base"
    index_root = root / "index"
    (strict_root / "metadata" / "obs_sc.parquet").mkdir(parents=True)
    (strict_root / "metadata").mkdir(parents=True, exist_ok=True)
    base_root.mkdir()

    datasets = (
        ("001_tiny_10x", False, "base"),
        ("304_tiny_parse", False, "strict"),
        ("010_tiny_scale", True, "base"),
        ("081_tiny_seqwell", False, "strict"),
    )
    shards: list[dict[str, object]] = []
    total_cells = 0
    for dataset_number, (dataset_id, scale, storage) in enumerate(datasets):
        obs = _dataset_rows(dataset_id, scale=scale)
        shard_id = f"shard-{dataset_number}"
        obs_path = strict_root / "metadata" / "obs_sc.parquet" / f"{shard_id}.parquet"
        obs.to_parquet(obs_path, index=False)
        n_cells = len(obs)
        values = np.arange(n_cells * 5, dtype=np.float32).reshape(n_cells, 5)
        values = (values % 7).astype(np.float32)
        matrix = sparse.csr_matrix(values)
        count_relative = Path("counts") / "sc_counts" / f"{shard_id}.zarr"
        count_root = base_root if storage == "base" else strict_root
        _write_csr_zarr(count_root / count_relative, matrix)
        shards.append(
            {
                "dataset_id": dataset_id,
                "shard_id": shard_id,
                "n_cells": n_cells,
                "n_genes": 5,
                "count_zarr": f"{storage}://{count_relative.as_posix()}",
                "obs_part": (
                    f"strict://metadata/obs_sc.parquet/{shard_id}.parquet"
                ),
            }
        )
        total_cells += n_cells
    pd.DataFrame(shards).to_parquet(
        strict_root / "metadata" / "sc_shards.parquet", index=False
    )
    shard_manifest_path = strict_root / "metadata" / "sc_shards.parquet"
    panel = pd.DataFrame(
        {
            "panel_index": np.arange(5, dtype=np.int64),
            "feature_name": [f"g{i}" for i in range(5)],
        }
    )
    gene_panel_path = strict_root / "metadata" / "var.parquet"
    panel.to_parquet(gene_panel_path, index=False)

    fingerprint = "a" * 64
    panel_sha256 = hashlib.sha256(gene_panel_path.read_bytes()).hexdigest()
    strict_metadata = {
        "schema_version": "model7.e2_strict_source.v1",
        "status": "complete",
        "strict_source_root": str(strict_root.resolve()),
        "base_source_root": str(base_root.resolve()),
        "source_fingerprint_sha256": fingerprint,
        "n_datasets": 4,
        "logical_cells": total_cells,
        "recipe_pool_annotation_free": True,
        "annotation_columns_referenced_by_builder": [],
        "annotation_columns_used_for_selection": [],
        "annotation_columns_used_for_qc": [],
        "tabula_ref_excluded": True,
        "gene_panel": {
            "path": str(gene_panel_path.resolve()),
            "sha256": panel_sha256,
            "n_genes": 5,
        },
        "gene_panel_sha256": panel_sha256,
        "shard_manifest_sha256": hashlib.sha256(
            shard_manifest_path.read_bytes()
        ).hexdigest(),
    }
    manifest_path = strict_root / "dataset_manifest.json"
    manifest_path.write_text(
        json.dumps(strict_metadata), encoding="utf-8"
    )
    complete_metadata = {
        "schema_version": "model7.e2_strict_source.v1",
        "status": "complete",
        "base_source_root": str(base_root.resolve()),
        "source_fingerprint_sha256": fingerprint,
        "n_datasets": 4,
        "logical_cells": total_cells,
        "gene_panel_sha256": panel_sha256,
        "dataset_manifest_sha256": hashlib.sha256(
            manifest_path.read_bytes()
        ).hexdigest(),
    }
    (strict_root / "BUILD_COMPLETE.json").write_text(
        json.dumps(complete_metadata), encoding="utf-8"
    )

    config_payload = {
        "paths": {
            "method1_root": str(root / "unused-method1"),
            "source_root": str(strict_root),
            "base_source_root": str(base_root),
            "index_root": str(index_root),
            "gene_panel_path": str(gene_panel_path),
        },
        "data": {
            "train_split": "train",
            "eval_split": "val",
            "split_seed": 19,
            "split_fractions": {"train": 0.5, "val": 0.25, "test": 0.25},
            "sampling_policy": "random",
            "platform_sampling_policy": "sqrt_cells",
            "platforms": ["10x", "Parse", "ScaleBio", "Seq-Well"],
            "k_values": [2, 4],
            "k_weights": [1, 2],
            "train_recipe_pool_size": 32,
            "recipes_per_epoch": 8,
            "fit_blocks": 4,
            "max_eval_blocks": 6,
            "validation_recipes_per_platform": 2,
            "cells_per_block": 2,
            "validation_cells_per_block": 1,
            "expected_datasets": 4,
            "expected_cells": total_cells,
        },
        "training": {"seed": 29},
        "conditioner": {"mode": "bio_bulk_depth"},
    }
    config_path = root / "e2.json"
    config_path.write_text(json.dumps(config_payload), encoding="utf-8")
    manifest = BUILDER.build_index(config_path)
    config = E2FullDataConfig(
        project_root=root / "unused-method1",
        source_root=strict_root,
        base_source_root=base_root,
        index_root=index_root,
        gene_panel_path=gene_panel_path,
        train_recipes_per_epoch=8,
        validation_recipes=6,
        validation_recipes_per_platform=2,
        fit_blocks=4,
        train_cells_per_block=2,
        validation_cells_per_block=1,
        seed=29,
        k_values=(2, 4),
        k_weights=(1, 2),
        split_seed=19,
        split_fractions=(0.5, 0.25, 0.25),
        train_recipe_pool_size=32,
        expected_datasets=4,
        expected_cells=total_cells,
        progress=False,
    )
    return {
        "root": root,
        "strict_root": strict_root,
        "base_root": base_root,
        "index_root": index_root,
        "config": config,
        "manifest": manifest,
    }


def test_uri_containment_rejects_traversal_and_escaping_symlink(tmp_path: Path) -> None:
    strict = tmp_path / "strict"
    base = tmp_path / "base"
    strict.mkdir()
    base.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    (strict / "escape").symlink_to(outside, target_is_directory=True)

    with pytest.raises(E2IndexContractError, match="escapes"):
        _resolve_source_uri(
            "strict://../outside", strict_root=strict, base_root=base, description="test"
        )
    with pytest.raises(E2IndexContractError, match="escapes"):
        _resolve_source_uri(
            "strict://escape/file", strict_root=strict, base_root=base, description="test"
        )
    with pytest.raises(E2IndexContractError, match="must use"):
        _resolve_source_uri(
            "/tmp/unpinned", strict_root=strict, base_root=base, description="test"
        )


def test_chunked_csr_rows_preserve_order_values_and_duplicates(tmp_path: Path) -> None:
    dense = np.asarray(
        [[0, 1, 2], [3, 0, 4], [5, 6, 0], [0, 7, 8], [9, 0, 0]],
        dtype=np.float32,
    )
    matrix = sparse.csr_matrix(dense)
    path = tmp_path / "matrix.zarr"
    _write_csr_zarr(path, matrix)
    group = zarr.open_group(str(path), mode="r")
    rows = np.asarray([4, 1, 2, 2, 3], dtype=np.int64)

    observed = _csr_rows(group, rows, dense.shape[1]).toarray()

    np.testing.assert_array_equal(observed, dense[rows])


def test_builder_contract_splits_quotas_and_annotation_absence(tiny_e2: dict[str, object]) -> None:
    index_root = tiny_e2["index_root"]
    assert isinstance(index_root, Path)
    manifest = tiny_e2["manifest"]
    assert isinstance(manifest, dict)
    assert manifest["source_obs_columns_read"] == [
        "dataset_id",
        "donor_id",
        "sample_id",
        "row_index",
    ]
    assert manifest["annotation_fields_never_requested_by_index_builder"] is True
    assert manifest["selection_uses_annotation_values"] is False
    assert manifest["split_unit"] == "dataset_donor"

    splits = pd.read_parquet(index_root / "metadata" / "donor_splits.parquet")
    assert (
        splits.groupby(["dataset_id", "split_key_type", "split_key_value"])["split"]
        .nunique()
        .max()
        == 1
    )
    fallback = splits[splits["split_key_type"].eq("sample")]
    assert not fallback.empty
    assert fallback["split_key_value"].str.startswith("s").all()

    scale = splits[splits["platform"].eq("ScaleBio")]
    assert set(scale["split"]) == {"train"}
    counts = manifest["recipe_counts_by_split_platform"]
    assert counts["val"].get("ScaleBio", 0) == 0
    assert counts["diagnostic"]["ScaleBio"] == 2
    assert manifest["platform_validation_contract"]["ScaleBio"] == {
        "split": "diagnostic",
        "scope": "in_corpus_diagnostic_singleton_donor",
        "heldout": False,
        "heldout_available": False,
    }


def test_builder_has_exact_sqrt_k_quotas_and_four_disjoint_epochs(
    tiny_e2: dict[str, object],
) -> None:
    index_root = tiny_e2["index_root"]
    assert isinstance(index_root, Path)
    manifest = tiny_e2["manifest"]
    assert isinstance(manifest, dict)
    assert sum(manifest["platform_quotas_per_epoch"].values()) == 8
    assert sum(manifest["k_quotas_per_epoch"].values()) == 8

    recipe_parts = pd.read_parquet(index_root / "metadata" / "recipe_parts.parquet")
    train = pd.concat(
        [
            pd.read_parquet(index_root / relative)
            for relative in recipe_parts.loc[recipe_parts["split"].eq("train"), "recipe_part"]
        ],
        ignore_index=True,
    )
    assert train.groupby("schedule_slot").size().eq(8).all()
    assert train["recipe_id"].is_unique
    slot_sets = [set(frame["recipe_id"]) for _, frame in train.groupby("schedule_slot")]
    assert len(slot_sets) == 4
    for left in range(len(slot_sets)):
        for right in range(left + 1, len(slot_sets)):
            assert slot_sets[left].isdisjoint(slot_sets[right])
    expected_platform = manifest["platform_quotas_per_epoch"]
    expected_k = {int(k): count for k, count in manifest["k_quotas_per_epoch"].items()}
    for _, frame in train.groupby("schedule_slot"):
        assert frame["platform"].value_counts().to_dict() == expected_platform
        assert frame["k"].value_counts().to_dict() == expected_k


def test_end_to_end_loader_conditioner_and_batch_schema(tiny_e2: dict[str, object]) -> None:
    config = tiny_e2["config"]
    assert isinstance(config, E2FullDataConfig)
    data_module = build_e2_full_data_module(config)
    counts = data_module.valid_recipe_counts()
    assert counts["evaluation"]["ScaleBio"] == 0
    assert counts["diagnostic"]["ScaleBio"] == 2
    assert data_module.split_manifest("evaluation")["platform"].isin(
        ["10x", "Parse", "Seq-Well"]
    ).all()

    conditioner = fit_e2_full_conditioner(data_module, config)
    assert conditioner.dim == len(data_module.gene_panel) + 6
    datasets = build_e2_full_datasets(data_module, conditioner, config)
    datasets.train.set_epoch(0)
    epoch_recipes = data_module.split_manifest(
        "train", max_blocks=config.train_recipes_per_epoch, seed=config.seed
    )
    assert len(datasets.train) == int(
        np.minimum(epoch_recipes["k"].to_numpy(), config.train_cells_per_block).sum()
    )
    sample = next(iter(datasets.train))
    assert set(sample) == {
        "x_log",
        "condition",
        "block_index",
        "block_id",
        "recipe_k",
    }
    assert "platform" not in sample
    assert np.asarray(sample["x_log"]).shape == (5,)
    assert np.asarray(sample["condition"]).shape == (11,)
    assert int(sample["recipe_k"]) in config.k_values
    diagnostic = next(iter(datasets.validation_by_platform["ScaleBio"]))
    assert set(diagnostic) == set(sample)


def test_loader_never_rescans_source_obs_after_index_build(
    tiny_e2: dict[str, object], monkeypatch: pytest.MonkeyPatch
) -> None:
    config = tiny_e2["config"]
    assert isinstance(config, E2FullDataConfig)
    data_module = build_e2_full_data_module(config)
    recipes = data_module.split_manifest("train", max_blocks=2, seed=config.seed)

    def forbidden_read(*args: object, **kwargs: object) -> object:
        raise AssertionError("runtime loader must use persisted sample_pools.parquet")

    monkeypatch.setattr(pd, "read_parquet", forbidden_read)
    data_module.prime_pools(recipes)


@pytest.mark.parametrize(
    "artifact",
    (
        "metadata/sample_pools.parquet",
        "metadata/sample_index.parquet",
        "metadata/donor_splits.parquet",
        "metadata/source_shards.parquet",
        "metadata/recipe_parts.parquet",
        "first_recipe_part",
    ),
)
def test_loader_fails_closed_when_bound_index_artifact_changes(
    tiny_e2: dict[str, object], tmp_path: Path, artifact: str
) -> None:
    original = tiny_e2["index_root"]
    config = tiny_e2["config"]
    assert isinstance(original, Path)
    assert isinstance(config, E2FullDataConfig)
    copied = tmp_path / "index"
    shutil.copytree(original, copied)
    if artifact == "first_recipe_part":
        parts = pd.read_parquet(copied / "metadata" / "recipe_parts.parquet")
        target = copied / str(parts.iloc[0]["recipe_part"])
    else:
        target = copied / artifact
    with target.open("ab") as handle:
        handle.write(b"index-drift")

    with pytest.raises(E2IndexContractError, match="fingerprint"):
        build_e2_full_data_module(replace(config, index_root=copied))


def test_relocation_preserves_recipes_counts_and_legacy_root_checks(tiny_e2, tmp_path):
    config = tiny_e2["config"]
    source = tmp_path / "source"
    base = tmp_path / "base"
    index = tmp_path / "index"
    for old, new in ((config.source_root, source), (config.base_source_root, base), (config.index_root, index)):
        shutil.copytree(old, new)
    panel = source / "metadata/var.parquet"
    relocated = replace(config, source_root=source, base_source_root=base,
                        index_root=index, gene_panel_path=panel)
    with pytest.raises(E2IndexContractError, match="root differs"):
        build_e2_full_data_module(relocated)
    mapping = {
        "schema_version": "pbisc.dataset_relocation.v1",
        "original_roots": {key: str(getattr(config, key)) for key in ("source_root", "base_source_root", "index_root", "gene_panel_path")},
        "relative_roots": {"source_root": "source", "base_source_root": "base", "index_root": "index", "gene_panel_path": "source/metadata/var.parquet"},
        "original_sha256": {key: hashlib.sha256(path.read_bytes()).hexdigest() for key, path in (
            ("source_manifest", source / "dataset_manifest.json"),
            ("source_complete", source / "BUILD_COMPLETE.json"),
            ("index_manifest", index / "dataset_manifest.json"))},
    }
    map_path = tmp_path / "relocation.json"
    map_path.write_text(json.dumps(mapping))
    original = build_e2_full_data_module(config)
    restored = build_e2_full_data_module(replace(relocated, relocation_manifest=map_path))
    assert restored.corpus_fingerprint == original.corpus_fingerprint
    pd.testing.assert_frame_equal(restored.recipes, original.recipes)
    recipe = original.split_manifest("train").iloc[0]
    for module in (original, restored):
        module.prime_pools(original.split_manifest("train"))
    refs1, pools1 = original._recipe_refs_and_pools(recipe)
    refs2, pools2 = restored._recipe_refs_and_pools(recipe)
    assert refs1 == refs2
    np.testing.assert_array_equal(original._load_sparse_refs(refs1, pools1).toarray(),
                                  restored._load_sparse_refs(refs2, pools2).toarray())
    mapping["original_sha256"]["index_manifest"] = "0" * 64
    map_path.write_text(json.dumps(mapping))
    with pytest.raises(E2IndexContractError, match="fingerprint differs"):
        build_e2_full_data_module(replace(relocated, relocation_manifest=map_path))
