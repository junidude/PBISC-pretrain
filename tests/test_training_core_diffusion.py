from __future__ import annotations

from collections.abc import Callable

import pytest
import torch
from torch import nn

from pbisc_sdd.diffusion import SparseDataDiffusion, SparseDiffusionConfig
from pbisc_sdd.model import (
    ConditionalSBDDenoiser,
    DenoiserOutput,
    SparseDenoiserConfig,
)


N_GENES = 5
CONDITION_DIM = 3
BATCH_SIZE = 2
PredictionFn = Callable[
    [torch.Tensor, torch.Tensor], tuple[torch.Tensor, torch.Tensor]
]


def small_batch() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    values = torch.tensor(
        [
            [1.0, 0.0, 2.0, 0.0, 8.0],
            [0.0, 3.0, 0.0, 4.0, 0.0],
        ]
    )
    measurement_mask = torch.tensor(
        [
            [True, True, True, False, False],
            [True, True, False, True, True],
        ]
    )
    active_target = torch.where(values > 0, 1.0, -1.0)
    condition = torch.tensor([[0.2, -0.3, 0.5], [-0.4, 0.1, 0.7]])
    return values, active_target, measurement_mask, condition


class RecordingDenoiser(nn.Module):
    """Small deterministic denoiser used to isolate diffusion contracts."""

    def __init__(
        self,
        prediction: PredictionFn | None = None,
    ) -> None:
        super().__init__()
        self.config = SparseDenoiserConfig(
            n_genes=N_GENES,
            condition_dim=CONDITION_DIM,
            hidden_dims=(8,),
            time_embedding_dim=8,
            context_dim=8,
            dropout=0.0,
        )
        self.n_genes = self.config.n_genes
        self.condition_dim = self.config.condition_dim
        self.prediction = prediction
        self.calls: list[dict[str, torch.Tensor | None]] = []

    def forward(
        self,
        noisy_state: torch.Tensor,
        time: torch.Tensor,
        condition: torch.Tensor,
        self_condition: torch.Tensor | None = None,
    ) -> DenoiserOutput:
        self.calls.append(
            {
                "time": time.detach().clone(),
                "condition": condition.detach().clone(),
                "self_condition": (
                    None if self_condition is None else self_condition.detach().clone()
                ),
            }
        )
        if self.prediction is None:
            values = noisy_state.new_zeros((noisy_state.shape[0], N_GENES))
            logits = noisy_state.new_zeros((noisy_state.shape[0], N_GENES))
        else:
            values, logits = self.prediction(noisy_state, condition)
        return DenoiserOutput(value_prediction=values, active_logits=logits)


def test_forward_corruption_shapes_and_missing_neutrality() -> None:
    values, active_target, measurement_mask, _ = small_batch()
    diffusion = SparseDataDiffusion(RecordingDenoiser())
    clean_state = diffusion.make_clean_state(
        values,
        active_target,
        measurement_mask=measurement_mask,
    )
    time = torch.tensor([0.15, 0.75])
    noise = torch.randn(
        clean_state.shape,
        dtype=clean_state.dtype,
        device=clean_state.device,
        generator=torch.Generator().manual_seed(11),
    )
    corruption = diffusion.q_sample(clean_state, time, noise=noise)

    assert clean_state.shape == (BATCH_SIZE, 2 * N_GENES)
    assert torch.equal(clean_state[:, :N_GENES][~measurement_mask], torch.zeros(3))
    assert torch.equal(clean_state[:, N_GENES:][~measurement_mask], torch.zeros(3))
    assert corruption.noisy_state.shape == clean_state.shape
    assert corruption.noise.shape == clean_state.shape
    assert corruption.log_snr.shape == (BATCH_SIZE, 1)
    assert corruption.alpha.shape == (BATCH_SIZE, 1)
    assert corruption.sigma.shape == (BATCH_SIZE, 1)
    assert torch.equal(corruption.noise, noise)
    assert torch.isfinite(corruption.noisy_state).all()


def test_centered_value_mode_maps_to_minus_one_one_and_inverts_sampling() -> None:
    values = torch.tensor([[0.0, 1.0, 2.0, 3.0, 4.0]]).repeat(BATCH_SIZE, 1)
    measured = torch.ones_like(values, dtype=torch.bool)
    diffusion = SparseDataDiffusion(
        RecordingDenoiser(),
        SparseDiffusionConfig(value_scale=4.0, center_values=True),
    )
    clean = diffusion.make_clean_state(values, measurement_mask=measured)
    assert torch.allclose(
        clean[:, :N_GENES],
        torch.tensor([[-1.0, -0.5, 0.0, 0.5, 1.0]]).repeat(BATCH_SIZE, 1),
    )

    analog = clean[:, :N_GENES].clone()

    def fixed_prediction(noisy_state, condition):
        del noisy_state, condition
        logits = torch.full_like(analog, 8.0)
        return analog, logits

    sampled = SparseDataDiffusion(
        RecordingDenoiser(fixed_prediction),
        SparseDiffusionConfig(value_scale=4.0, center_values=True),
    ).sample(
        torch.zeros(BATCH_SIZE, CONDITION_DIM),
        steps=1,
        initial_noise=torch.zeros(BATCH_SIZE, 2 * N_GENES),
    )
    assert torch.allclose(sampled.dense_values, values)


def test_sampling_clip_denoised_bounds_centered_x0_each_step() -> None:
    def out_of_range_prediction(noisy_state, condition):
        del condition
        batch = noisy_state.shape[0]
        return (
            noisy_state.new_full((batch, N_GENES), 2.0),
            noisy_state.new_full((batch, N_GENES), 8.0),
        )

    condition = torch.zeros(BATCH_SIZE, CONDITION_DIM)
    noise = torch.zeros(BATCH_SIZE, 2 * N_GENES)
    clipped = SparseDataDiffusion(
        RecordingDenoiser(out_of_range_prediction),
        SparseDiffusionConfig(center_values=True, clip_denoised=True),
    ).sample(condition, steps=2, initial_noise=noise)
    unclipped = SparseDataDiffusion(
        RecordingDenoiser(out_of_range_prediction),
        SparseDiffusionConfig(center_values=True, clip_denoised=False),
    ).sample(condition, steps=2, initial_noise=noise)

    assert torch.equal(clipped.dense_values, torch.ones_like(clipped.dense_values))
    assert torch.equal(
        unclipped.dense_values,
        torch.full_like(unclipped.dense_values, 1.5),
    )


def test_small_end_to_end_loss_is_finite_and_backpropagates() -> None:
    torch.manual_seed(17)
    values, active_target, measurement_mask, condition = small_batch()
    model = ConditionalSBDDenoiser(
        SparseDenoiserConfig(
            n_genes=N_GENES,
            condition_dim=CONDITION_DIM,
            hidden_dims=(16, 12),
            time_embedding_dim=8,
            context_dim=8,
            dropout=0.0,
        )
    )
    diffusion = SparseDataDiffusion(
        model,
        SparseDiffusionConfig(self_condition_probability=1.0),
    )
    result = diffusion(
        values,
        condition,
        measurement_mask=measurement_mask,
        active_target=active_target,
        time=torch.tensor([0.2, 0.7]),
        noise=torch.randn(BATCH_SIZE, 2 * N_GENES),
    )

    for scalar in (result.loss, result.value_loss, result.support_loss):
        assert scalar.ndim == 0
        assert torch.isfinite(scalar)
    result.loss.backward()
    for head in (model.value_head, model.active_head):
        assert head.weight.grad is not None
        assert torch.isfinite(head.weight.grad).all()


def test_unmeasured_target_changes_do_not_change_training_loss() -> None:
    torch.manual_seed(13)
    values, active_target, measurement_mask, condition = small_batch()
    diffusion = SparseDataDiffusion(
        RecordingDenoiser(),
        SparseDiffusionConfig(self_condition_probability=0.0),
    )
    time = torch.tensor([0.25, 0.65])
    noise = torch.randn(BATCH_SIZE, 2 * N_GENES)

    baseline = diffusion.training_loss(
        values,
        condition,
        measurement_mask=measurement_mask,
        active_target=active_target,
        time=time,
        noise=noise,
    )
    changed_values = values.clone()
    changed_values[~measurement_mask] = torch.tensor([101.0, -57.0, 999.0])
    changed_active = active_target.clone()
    changed_active[~measurement_mask] *= -1
    changed = diffusion.training_loss(
        changed_values,
        condition,
        measurement_mask=measurement_mask,
        active_target=changed_active,
        time=time,
        noise=noise,
    )

    assert torch.equal(baseline.clean_state, changed.clean_state)
    assert torch.equal(baseline.corruption.noisy_state, changed.corruption.noisy_state)
    assert torch.equal(baseline.value_loss, changed.value_loss)
    assert torch.equal(baseline.support_loss, changed.support_loss)
    assert torch.equal(baseline.loss, changed.loss)


def test_active_only_value_loss_ignores_measured_zero_prediction_error() -> None:
    values = torch.tensor([[2.0, 0.0, 1.0, 0.0, 3.0]]).repeat(BATCH_SIZE, 1)
    active_target = torch.tensor([[1.0, -1.0, 1.0, -1.0, 1.0]]).repeat(
        BATCH_SIZE, 1
    )
    measurement_mask = torch.ones_like(values, dtype=torch.bool)
    condition = torch.zeros(BATCH_SIZE, CONDITION_DIM)
    time = torch.full((BATCH_SIZE,), 0.4)
    noise = torch.zeros(BATCH_SIZE, 2 * N_GENES)

    def prediction_with_inactive_value(
        inactive_value: float,
    ) -> Callable[[torch.Tensor, torch.Tensor], tuple[torch.Tensor, torch.Tensor]]:
        def predict(
            noisy_state: torch.Tensor, condition: torch.Tensor
        ) -> tuple[torch.Tensor, torch.Tensor]:
            del condition
            prediction = noisy_state.new_zeros((noisy_state.shape[0], N_GENES))
            prediction[:, [1, 3]] = inactive_value
            logits = noisy_state.new_zeros((noisy_state.shape[0], N_GENES))
            return prediction, logits

        return predict

    def value_loss(mode: str, inactive_value: float) -> torch.Tensor:
        diffusion = SparseDataDiffusion(
            RecordingDenoiser(prediction_with_inactive_value(inactive_value)),
            SparseDiffusionConfig(
                value_loss_mode=mode,
                self_condition_probability=0.0,
            ),
        )
        return diffusion.training_loss(
            values,
            condition,
            measurement_mask=measurement_mask,
            active_target=active_target,
            time=time,
            noise=noise,
        ).value_loss

    active_baseline = value_loss("active_only", 0.0)
    active_with_error = value_loss("active_only", 50.0)
    measured_baseline = value_loss("all_measured", 0.0)
    measured_with_error = value_loss("all_measured", 50.0)

    assert torch.equal(active_baseline, active_with_error)
    assert measured_with_error > measured_baseline


def test_support_bce_uses_positive_logit_for_nonzero_target() -> None:
    values, active_target, measurement_mask, condition = small_batch()
    time = torch.full((BATCH_SIZE,), 0.5)
    noise = torch.zeros(BATCH_SIZE, 2 * N_GENES)

    def support_prediction(sign: float) -> RecordingDenoiser:
        def predict(
            noisy_state: torch.Tensor, condition: torch.Tensor
        ) -> tuple[torch.Tensor, torch.Tensor]:
            del condition
            batch = noisy_state.shape[0]
            logits = sign * 10.0 * active_target.to(noisy_state)
            return noisy_state.new_zeros((batch, N_GENES)), logits

        return RecordingDenoiser(predict)

    def support_loss(sign: float) -> torch.Tensor:
        diffusion = SparseDataDiffusion(
            support_prediction(sign),
            SparseDiffusionConfig(self_condition_probability=0.0),
        )
        return diffusion.training_loss(
            values,
            condition,
            measurement_mask=measurement_mask,
            active_target=active_target,
            time=time,
            noise=noise,
        ).support_loss

    correctly_signed = support_loss(1.0)
    flipped = support_loss(-1.0)

    assert correctly_signed < 1e-3
    assert flipped > 5.0


def test_training_self_conditioning_uses_detached_first_prediction() -> None:
    values, active_target, measurement_mask, condition = small_batch()
    denoiser = RecordingDenoiser()
    diffusion = SparseDataDiffusion(
        denoiser,
        SparseDiffusionConfig(self_condition_probability=1.0),
    )
    result = diffusion.training_loss(
        values,
        condition,
        measurement_mask=measurement_mask,
        active_target=active_target,
        time=torch.tensor([0.3, 0.6]),
        noise=torch.zeros(BATCH_SIZE, 2 * N_GENES),
    )

    assert len(denoiser.calls) == 2
    assert denoiser.calls[0]["self_condition"] is None
    second_self_condition = denoiser.calls[1]["self_condition"]
    assert second_self_condition is not None
    assert second_self_condition.shape == (BATCH_SIZE, 2 * N_GENES)
    assert result.self_condition is not None
    assert torch.equal(result.self_condition, second_self_condition)
    assert not result.self_condition.requires_grad


def test_ddim_sampling_applies_support_mask_as_exact_zeros() -> None:
    active_logits = torch.tensor([-8.0, 8.0, -8.0, 8.0, -8.0])

    def fixed_prediction(
        noisy_state: torch.Tensor, condition: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        del condition
        batch = noisy_state.shape[0]
        values = noisy_state.new_full((batch, N_GENES), 2.0)
        logits = active_logits.to(noisy_state).expand(batch, -1)
        return values, logits

    diffusion = SparseDataDiffusion(RecordingDenoiser(fixed_prediction))
    condition = torch.zeros(BATCH_SIZE, CONDITION_DIM)
    sample = diffusion.sample(
        condition,
        steps=4,
        eta=0.0,
        initial_noise=torch.zeros(BATCH_SIZE, 2 * N_GENES),
    )

    expected_mask = torch.tensor([False, True, False, True, False]).expand(
        BATCH_SIZE, -1
    )
    assert torch.equal(sample.active_mask, expected_mask)
    assert torch.equal(sample.values[~sample.active_mask], torch.zeros(6))
    assert torch.equal(sample.values, sample.dense_values * sample.active_mask)
    assert torch.count_nonzero(sample.values[:, [0, 2, 4]]) == 0


def test_sampling_measurement_aliases_override_learned_active_mask() -> None:
    def all_active_prediction(
        noisy_state: torch.Tensor, condition: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        del condition
        shape = (noisy_state.shape[0], N_GENES)
        return noisy_state.new_full(shape, 2.0), noisy_state.new_full(shape, 8.0)

    diffusion = SparseDataDiffusion(RecordingDenoiser(all_active_prediction))
    condition = torch.zeros(BATCH_SIZE, CONDITION_DIM)
    initial_noise = torch.zeros(BATCH_SIZE, 2 * N_GENES)
    measurement_mask = torch.tensor(
        [
            [True, False, True, False, True],
            [False, True, False, True, False],
        ]
    )

    measured_sample = diffusion.sample(
        condition,
        steps=1,
        eta=0.0,
        initial_noise=initial_noise,
        measurement_mask=measurement_mask,
    )
    available_sample = diffusion.sample(
        condition,
        steps=1,
        eta=0.0,
        initial_noise=initial_noise,
        availability_mask=measurement_mask,
    )

    assert torch.equal(measured_sample.active_mask, measurement_mask)
    assert torch.equal(available_sample.active_mask, measurement_mask)
    assert torch.equal(measured_sample.values, available_sample.values)
    assert torch.equal(
        measured_sample.values[~measurement_mask],
        torch.zeros_like(measured_sample.values[~measurement_mask]),
    )


def test_sampling_rejects_both_measurement_mask_aliases() -> None:
    diffusion = SparseDataDiffusion(RecordingDenoiser())
    condition = torch.zeros(BATCH_SIZE, CONDITION_DIM)
    measurement_mask = torch.ones(BATCH_SIZE, N_GENES, dtype=torch.bool)

    with pytest.raises(ValueError, match="only one"):
        diffusion.sample(
            condition,
            steps=1,
            measurement_mask=measurement_mask,
            availability_mask=measurement_mask,
        )


def test_ddim_inactive_nonfinite_dense_predictions_become_exact_zeros() -> None:
    def nonfinite_prediction(
        noisy_state: torch.Tensor, condition: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        del condition
        values = torch.tensor(
            [float("inf"), float("-inf"), float("nan"), 2.0, 3.0],
            device=noisy_state.device,
            dtype=noisy_state.dtype,
        ).expand(noisy_state.shape[0], -1)
        logits = torch.tensor(
            [-8.0, -8.0, -8.0, 8.0, 8.0],
            device=noisy_state.device,
            dtype=noisy_state.dtype,
        ).expand(noisy_state.shape[0], -1)
        return values, logits

    diffusion = SparseDataDiffusion(RecordingDenoiser(nonfinite_prediction))
    sample = diffusion.sample(
        torch.zeros(BATCH_SIZE, CONDITION_DIM),
        steps=1,
        eta=0.0,
        initial_noise=torch.zeros(BATCH_SIZE, 2 * N_GENES),
        clamp_values_min=None,
    )

    assert torch.equal(
        sample.active_mask[:, :3],
        torch.zeros(BATCH_SIZE, 3, dtype=torch.bool),
    )
    assert torch.equal(sample.values[:, :3], torch.zeros(BATCH_SIZE, 3))
    assert torch.isfinite(sample.values).all()


def test_ddim_eta_zero_is_reproducible_with_seeded_generator() -> None:
    def echo_prediction(
        noisy_state: torch.Tensor, condition: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        del condition
        return noisy_state[:, :N_GENES], noisy_state[:, N_GENES:]

    diffusion = SparseDataDiffusion(RecordingDenoiser(echo_prediction))
    condition = torch.zeros(BATCH_SIZE, CONDITION_DIM)
    sample_a = diffusion.sample(
        condition,
        steps=5,
        eta=0.0,
        generator=torch.Generator().manual_seed(101),
    )
    sample_b = diffusion.sample(
        condition,
        steps=5,
        eta=0.0,
        generator=torch.Generator().manual_seed(101),
    )
    sample_c = diffusion.sample(
        condition,
        steps=5,
        eta=0.0,
        generator=torch.Generator().manual_seed(102),
    )

    for field in ("values", "dense_values", "active_logits", "final_state"):
        assert torch.equal(getattr(sample_a, field), getattr(sample_b, field))
    assert torch.equal(sample_a.active_mask, sample_b.active_mask)
    assert not torch.equal(sample_a.final_state, sample_c.final_state)


def test_sampling_propagates_changed_condition_to_every_denoising_step() -> None:
    def condition_prediction(
        noisy_state: torch.Tensor, condition: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        value = condition[:, :1].expand(-1, N_GENES)
        logits = noisy_state.new_full((noisy_state.shape[0], N_GENES), 8.0)
        return value, logits

    denoiser = RecordingDenoiser(condition_prediction)
    diffusion = SparseDataDiffusion(denoiser)
    initial_noise = torch.zeros(BATCH_SIZE, 2 * N_GENES)
    condition_a = torch.zeros(BATCH_SIZE, CONDITION_DIM)
    condition_b = torch.ones(BATCH_SIZE, CONDITION_DIM)

    sample_a = diffusion.sample(
        condition_a,
        steps=4,
        eta=0.0,
        initial_noise=initial_noise,
    )
    assert denoiser.calls
    assert all(torch.equal(call["condition"], condition_a) for call in denoiser.calls)

    denoiser.calls.clear()
    sample_b = diffusion.sample(
        condition_b,
        steps=4,
        eta=0.0,
        initial_noise=initial_noise,
    )
    assert denoiser.calls
    assert all(torch.equal(call["condition"], condition_b) for call in denoiser.calls)
    assert not torch.equal(sample_a.dense_values, sample_b.dense_values)


def test_training_loss_rejects_invalid_mask_shape_and_alias_collision() -> None:
    values, active_target, measurement_mask, condition = small_batch()
    diffusion = SparseDataDiffusion(RecordingDenoiser())

    with pytest.raises(ValueError):
        diffusion.training_loss(
            values,
            condition,
            measurement_mask=measurement_mask[:, :-1],
            active_target=active_target,
        )
    with pytest.raises(ValueError):
        diffusion.training_loss(
            values,
            condition,
            measurement_mask=measurement_mask,
            availability_mask=measurement_mask,
            active_target=active_target,
        )


def test_unmeasured_active_target_garbage_is_ignored() -> None:
    values, active_target, measurement_mask, condition = small_batch()
    active_target = active_target.clone()
    active_target[~measurement_mask] = torch.tensor(
        [float("nan"), 37.0, -999.0]
    )
    diffusion = SparseDataDiffusion(
        RecordingDenoiser(),
        SparseDiffusionConfig(self_condition_probability=0.0),
    )

    result = diffusion.training_loss(
        values,
        condition,
        measurement_mask=measurement_mask,
        active_target=active_target,
        time=torch.tensor([0.25, 0.75]),
        noise=torch.zeros(BATCH_SIZE, 2 * N_GENES),
    )

    assert torch.isfinite(result.loss)
    assert torch.equal(
        result.clean_state[:, N_GENES:][~measurement_mask],
        torch.zeros(3),
    )


@pytest.mark.parametrize("invalid_target", [float("nan"), 7.0])
def test_measured_active_target_garbage_is_rejected(invalid_target: float) -> None:
    values, active_target, measurement_mask, condition = small_batch()
    active_target = active_target.clone()
    active_target[0, 0] = invalid_target
    diffusion = SparseDataDiffusion(RecordingDenoiser())

    with pytest.raises(ValueError, match="active_target"):
        diffusion.training_loss(
            values,
            condition,
            measurement_mask=measurement_mask,
            active_target=active_target,
        )


@pytest.mark.parametrize(
    ("row", "gene", "contradictory_target"),
    [(0, 0, -1.0), (0, 1, 1.0)],
    ids=("nonzero-marked-inactive", "zero-marked-active"),
)
def test_measured_value_and_support_contradiction_is_rejected(
    row: int,
    gene: int,
    contradictory_target: float,
) -> None:
    values, active_target, measurement_mask, condition = small_batch()
    active_target = active_target.clone()
    active_target[row, gene] = contradictory_target
    diffusion = SparseDataDiffusion(RecordingDenoiser())

    with pytest.raises(ValueError, match=r"active_target.*values"):
        diffusion.training_loss(
            values,
            condition,
            measurement_mask=measurement_mask,
            active_target=active_target,
        )


def test_half_precision_large_measured_mask_has_finite_correct_mean() -> None:
    n_genes = 70_000

    class WideZeroDenoiser(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.n_genes = n_genes
            self.condition_dim = 1

        def forward(
            self,
            noisy_state: torch.Tensor,
            time: torch.Tensor,
            condition: torch.Tensor,
            self_condition: torch.Tensor | None = None,
        ) -> DenoiserOutput:
            del time, condition, self_condition
            shape = (noisy_state.shape[0], self.n_genes)
            return DenoiserOutput(
                value_prediction=noisy_state.new_zeros(shape),
                active_logits=noisy_state.new_zeros(shape),
            )

    values = torch.ones(1, n_genes, dtype=torch.float16)
    diffusion = SparseDataDiffusion(
        WideZeroDenoiser(),
        SparseDiffusionConfig(self_condition_probability=0.0),
    )
    result = diffusion.training_loss(
        values,
        torch.zeros(1, 1, dtype=torch.float16),
        measurement_mask=torch.ones(1, n_genes, dtype=torch.bool),
        active_target=torch.ones(1, n_genes, dtype=torch.bool),
        time=torch.full((1,), 0.5, dtype=torch.float16),
        noise=torch.zeros(1, 2 * n_genes, dtype=torch.float16),
    )

    assert torch.isfinite(result.loss)
    assert float(result.value_loss) == pytest.approx(1.0)
    assert float(result.support_loss) == pytest.approx(0.693147, abs=1e-3)
