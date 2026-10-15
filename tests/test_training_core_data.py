from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import sparse
import torch

from pbisc_sdd.data import (
    MeasurementAwareE2DataModule,
    SparseValueScaler,
    SubsetAnnotationFreeConditioner,
    condition_with_availability,
    condition_with_measurement_context,
    resolve_gene_positions,
)


class _FakeBase:
    def __init__(self) -> None:
        self.gene_panel = pd.DataFrame(
            {
                "panel_index": [0, 1, 2],
                "gene_id": ["g0", "g1", "g2"],
                "gene_symbol": ["G0", "G1", "G2"],
            }
        )
        self.gene_indices = np.arange(3, dtype=np.int64)
        self.source_n_genes = 3
        self.corpus_fingerprint = "a" * 64
        self.groups = {
            "first": {"measurement_mask": np.array([1, 0, 1], dtype=np.uint8)},
            "second": {"measurement_mask": np.array([0, 1, 1], dtype=np.uint8)},
        }

    def _recipe_refs_and_pools(self, recipe):
        del recipe
        return [(0, 0), (1, 0)], [
            {"count_zarr": "first"},
            {"count_zarr": "second"},
        ]

    def _load_sparse_refs(self, refs, pools):
        del refs, pools
        return sparse.csr_matrix(np.array([[3, 0, 0], [0, 2, 0]], dtype=np.float32))

    def _zarr_group(self, path):
        return self.groups[path]


class _FakeFullConditioner:
    n_genes = 5
    fitted_ = True
    feature_names = [f"g{i}" for i in range(5)] + [f"s{i}" for i in range(6)]

    def transform(self, bulk_log1p, recipe_k):
        assert np.asarray(bulk_log1p).shape == (5,)
        assert recipe_k > 0
        return np.arange(11, dtype=np.float32)

    def state_dict(self):
        return {"schema_version": "fake"}


def test_measurement_aware_block_aligns_each_cell_with_its_shard_mask() -> None:
    module = MeasurementAwareE2DataModule(_FakeBase(), [0, 1, 2])
    block = module.load_training_block(
        {"k": 2, "block_id": "b0"},
        cells_per_block=2,
        rng=np.random.default_rng(3),
    )

    assert block.counts.shape == (2, 3)
    assert block.availability_mask.shape == block.counts.shape
    assert np.all(block.availability_mask[block.counts > 0] == 1)
    assert np.allclose(block.bulk_measurement_fraction, [0.5, 0.5, 1.0])
    assert block.bulk_log1p.shape == (3,)
    assert np.allclose(np.expm1(block.bulk_log1p), [3.0, 2.0, 0.0])


def test_subset_conditioner_selects_after_full_panel_transform() -> None:
    conditioner = SubsetAnnotationFreeConditioner(
        _FakeFullConditioner(), np.array([1, 4], dtype=np.int64)
    )
    result = conditioner.transform(np.zeros(5, dtype=np.float32), recipe_k=10)

    assert result.tolist() == [1.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
    assert conditioner.dim == 8


def test_value_scaler_and_condition_mask_contract() -> None:
    scaler = SparseValueScaler(max_log_value=4.0)
    x_log = torch.tensor([[0.0, 2.0, 8.0]])
    unit = scaler.to_unit_tensor(x_log)
    restored = scaler.from_unit_tensor(unit)
    condition = condition_with_availability(
        torch.tensor([[0.25, -0.5]]), torch.tensor([[True, False, True]])
    )

    assert torch.equal(unit, torch.tensor([[0.0, 0.5, 1.0]]))
    assert torch.equal(restored, torch.tensor([[0.0, 2.0, 4.0]]))
    assert torch.equal(condition, torch.tensor([[0.25, -0.5, 1.0, -1.0, 1.0]]))
    measurement_condition = condition_with_measurement_context(
        torch.tensor([[0.25, -0.5]]),
        torch.tensor([[1.0, 0.5, 0.0]]),
        torch.tensor([[True, False, True]]),
    )
    assert torch.equal(
        measurement_condition,
        torch.tensor([[0.25, -0.5, 1.0, 0.0, -1.0, 1.0, -1.0, 1.0]]),
    )


def test_coverage_gene_selection_is_stable_and_keeps_marker() -> None:
    panel = pd.DataFrame(
        {
            "measurement_frequency_sc": [10, 9, 1, 8],
            "is_immune_marker": [False, False, True, False],
            "is_interferon_gene": [False, False, False, False],
        }
    )
    positions = resolve_gene_positions(panel, n_genes=2)

    assert positions.tolist() == [0, 2]
