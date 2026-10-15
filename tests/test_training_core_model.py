from __future__ import annotations

import pytest
import torch

from pbisc_sdd.model import (
    ConditionalSBDDenoiser,
    DenoiserOutput,
    SparseDenoiserConfig,
)


N_GENES = 5
CONDITION_DIM = 3
BATCH_SIZE = 2


def small_config() -> SparseDenoiserConfig:
    return SparseDenoiserConfig(
        n_genes=N_GENES,
        condition_dim=CONDITION_DIM,
        hidden_dims=(16, 12),
        time_embedding_dim=8,
        context_dim=8,
        dropout=0.0,
    )


def test_conditional_denoiser_forward_shapes() -> None:
    torch.manual_seed(7)
    model = ConditionalSBDDenoiser(small_config()).eval()
    noisy_state = torch.randn(BATCH_SIZE, 2 * N_GENES)
    time = torch.tensor([0.1, 0.8])
    condition = torch.randn(BATCH_SIZE, CONDITION_DIM)

    output = model(noisy_state, time, condition)
    self_conditioned = model(
        noisy_state,
        time,
        condition,
        self_condition=torch.zeros_like(noisy_state),
    )

    assert output.value_prediction.shape == (BATCH_SIZE, N_GENES)
    assert output.active_logits.shape == (BATCH_SIZE, N_GENES)
    assert output.continuous_x0().shape == noisy_state.shape
    assert self_conditioned.value_prediction.shape == (BATCH_SIZE, N_GENES)
    assert self_conditioned.active_logits.shape == (BATCH_SIZE, N_GENES)
    assert torch.isfinite(output.value_prediction).all()
    assert torch.isfinite(output.active_logits).all()


def test_continuous_support_is_probability_consistent() -> None:
    log_nine = torch.log(torch.tensor(9.0))
    output = DenoiserOutput(
        value_prediction=torch.zeros(1, 3),
        active_logits=torch.tensor([[log_nine, -log_nine, 0.0]]),
    )

    continuous = output.continuous_x0()

    assert torch.allclose(
        continuous[:, 3:],
        torch.tensor([[0.8, -0.8, 0.0]]),
        atol=1e-6,
    )


@pytest.mark.parametrize(
    ("noisy_state", "time", "condition", "self_condition"),
    [
        (
            torch.zeros(BATCH_SIZE, 2 * N_GENES - 1),
            torch.zeros(BATCH_SIZE),
            torch.zeros(BATCH_SIZE, CONDITION_DIM),
            None,
        ),
        (
            torch.zeros(BATCH_SIZE, 2 * N_GENES),
            torch.zeros(BATCH_SIZE, 1),
            torch.zeros(BATCH_SIZE, CONDITION_DIM),
            None,
        ),
        (
            torch.zeros(BATCH_SIZE, 2 * N_GENES),
            torch.zeros(BATCH_SIZE),
            torch.zeros(BATCH_SIZE, CONDITION_DIM - 1),
            None,
        ),
        (
            torch.zeros(BATCH_SIZE, 2 * N_GENES),
            torch.zeros(BATCH_SIZE),
            torch.zeros(BATCH_SIZE, CONDITION_DIM),
            torch.zeros(BATCH_SIZE, 2 * N_GENES - 1),
        ),
    ],
    ids=("state-width", "time-rank", "condition-width", "self-condition-width"),
)
def test_conditional_denoiser_rejects_invalid_shapes(
    noisy_state: torch.Tensor,
    time: torch.Tensor,
    condition: torch.Tensor,
    self_condition: torch.Tensor | None,
) -> None:
    model = ConditionalSBDDenoiser(small_config())

    with pytest.raises(ValueError):
        model(noisy_state, time, condition, self_condition=self_condition)


def test_conditional_denoiser_rejects_batch_mismatch() -> None:
    model = ConditionalSBDDenoiser(small_config())

    with pytest.raises(ValueError):
        model(
            torch.zeros(BATCH_SIZE, 2 * N_GENES),
            torch.zeros(BATCH_SIZE),
            torch.zeros(BATCH_SIZE + 1, CONDITION_DIM),
        )
