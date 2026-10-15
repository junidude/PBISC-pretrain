from __future__ import annotations

import numpy as np

from pbisc_sdd.gene_selection import rank_hvg_statistics


def test_hvg_ranking_is_finite_train_statistic_and_keeps_marker() -> None:
    measured = np.full(8, 100, dtype=np.int64)
    detected = np.full(8, 20, dtype=np.int64)
    means = np.asarray([0.1, 0.2, 0.3, 0.4, 1.0, 1.1, 1.2, 1.3])
    variances = np.asarray([0.1, 0.2, 3.0, 0.1, 0.2, 0.3, 4.0, 0.1])
    value_sum = means * measured
    square_sum = (variances + means**2) * measured
    marker = np.zeros(8, dtype=bool)
    marker[0] = True

    positions, estimated_mean, estimated_variance, score = rank_hvg_statistics(
        value_sum=value_sum,
        value_square_sum=square_sum,
        measured_cells=measured,
        detected_cells=detected,
        marker_mask=marker,
        n_genes=3,
        min_measured_cells=10,
        min_detected_cells=3,
        mean_bins=2,
    )

    assert 0 in positions
    assert positions.shape == (3,)
    assert np.allclose(estimated_mean, means)
    assert np.allclose(estimated_variance, variances)
    assert np.isfinite(score[positions]).all()


def test_hvg_ranking_rejects_insufficient_eligible_genes() -> None:
    zeros = np.zeros(4)
    try:
        rank_hvg_statistics(
            value_sum=zeros,
            value_square_sum=zeros,
            measured_cells=np.asarray([100, 0, 0, 0]),
            detected_cells=np.asarray([10, 0, 0, 0]),
            marker_mask=np.zeros(4, dtype=bool),
            n_genes=2,
            min_measured_cells=10,
            min_detected_cells=3,
        )
    except ValueError as error:
        assert "fewer than 2" in str(error)
    else:  # pragma: no cover
        raise AssertionError("expected insufficient eligible genes to fail")
