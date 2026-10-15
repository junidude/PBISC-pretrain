"""Continuous sparse-data diffusion objective and DDIM sampler.

Values and active/support variables are diffused jointly in the original gene
space.  A measured active target is represented by ``+support_scale`` and a
measured inactive target by ``-support_scale``.  Unmeasured coordinates are
neutralized to zero and excluded from both loss terms; they are never taught
as biological zeros.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Literal, cast

import torch
from torch import nn
import torch.nn.functional as F

from .model import DenoiserOutput


ValueLossMode = Literal["all_measured", "active_only"]


def _finite_positive(name: str, value: object) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be finite and positive, got {value!r}")
    try:
        result = float(cast(float, value))
    except (TypeError, ValueError) as error:
        raise ValueError(
            f"{name} must be finite and positive, got {value!r}"
        ) from error
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be finite and positive, got {value!r}")
    return result


def _finite_non_negative(name: str, value: object) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be finite and non-negative, got {value!r}")
    try:
        result = float(cast(float, value))
    except (TypeError, ValueError) as error:
        raise ValueError(
            f"{name} must be finite and non-negative, got {value!r}"
        ) from error
    if not math.isfinite(result) or result < 0.0:
        raise ValueError(f"{name} must be finite and non-negative, got {value!r}")
    return result


@dataclass(frozen=True)
class SparseDiffusionConfig:
    """Objective and continuous cosine-schedule configuration.

    ``value_scale`` converts a PBISC value into model space by division.  It
    must therefore be fixed from training data and reused at generation time.
    """

    value_scale: float = 1.0
    center_values: bool = False
    clip_denoised: bool = True
    support_scale: float = 1.0
    cosine_offset: float = 0.008
    log_snr_clip: float = 20.0
    value_loss_mode: ValueLossMode = "active_only"
    value_loss_weight: float = 1.0
    support_loss_weight: float = 1.0
    self_condition_probability: float = 0.5

    def __post_init__(self) -> None:
        _finite_positive("value_scale", self.value_scale)
        if not isinstance(self.center_values, bool):
            raise ValueError("center_values must be boolean")
        if not isinstance(self.clip_denoised, bool):
            raise ValueError("clip_denoised must be boolean")
        _finite_positive("support_scale", self.support_scale)
        _finite_positive("log_snr_clip", self.log_snr_clip)
        offset = _finite_non_negative("cosine_offset", self.cosine_offset)
        if offset >= 1.0:
            raise ValueError("cosine_offset must be smaller than 1")
        if self.value_loss_mode not in ("all_measured", "active_only"):
            raise ValueError(
                "value_loss_mode must be 'all_measured' or 'active_only'"
            )
        _finite_non_negative("value_loss_weight", self.value_loss_weight)
        _finite_non_negative("support_loss_weight", self.support_loss_weight)
        probability = _finite_non_negative(
            "self_condition_probability", self.self_condition_probability
        )
        if probability > 1.0:
            raise ValueError("self_condition_probability must be in [0, 1]")


@dataclass(frozen=True)
class ForwardCorruption:
    """A sample from q(x_t | x_0), with broadcast-ready schedule terms."""

    noisy_state: torch.Tensor
    noise: torch.Tensor
    log_snr: torch.Tensor
    alpha: torch.Tensor
    sigma: torch.Tensor
    time: torch.Tensor


@dataclass(frozen=True)
class SparseDiffusionLoss:
    """Weighted objective plus inspectable training intermediates."""

    loss: torch.Tensor
    value_loss: torch.Tensor
    support_loss: torch.Tensor
    prediction: DenoiserOutput
    corruption: ForwardCorruption
    clean_state: torch.Tensor
    self_condition: torch.Tensor | None

    @property
    def total_loss(self) -> torch.Tensor:
        return self.loss

    @property
    def self_condition_used(self) -> bool:
        return self.self_condition is not None


@dataclass(frozen=True)
class SparseSample:
    """Final exact-sparse sample and its dense/support diagnostics."""

    values: torch.Tensor
    dense_values: torch.Tensor
    active_mask: torch.Tensor
    active_logits: torch.Tensor
    final_state: torch.Tensor
    steps: int
    eta: float

    @property
    def generated_values(self) -> torch.Tensor:
        return self.values

    @property
    def value_prediction(self) -> torch.Tensor:
        return self.dense_values


def cosine_log_snr(
    time: torch.Tensor,
    *,
    offset: float = 0.008,
    clip: float = 20.0,
) -> torch.Tensor:
    """Return a normalized continuous cosine log signal-to-noise ratio.

    Normalization makes the un-clipped cumulative alpha equal to one at
    ``time=0``.  The returned log-SNR is clipped only for endpoint numerical
    stability.
    """

    if not isinstance(time, torch.Tensor):
        raise TypeError("time must be a torch.Tensor")
    if not torch.is_floating_point(time):
        raise TypeError("time must use a floating-point dtype")
    if time.ndim != 1:
        raise ValueError(f"time must have shape [batch], got {tuple(time.shape)}")
    if not bool(torch.isfinite(time).all()):
        raise ValueError("time must contain only finite values")
    if bool(((time < 0.0) | (time > 1.0)).any()):
        raise ValueError("time values must lie in [0, 1]")
    offset_value = _finite_non_negative("offset", offset)
    if offset_value >= 1.0:
        raise ValueError("offset must be smaller than 1")
    clip_value = _finite_positive("clip", clip)

    # Keep schedule arithmetic in float32 when a low-precision model is used.
    schedule_time = (
        time.float()
        if time.dtype in (torch.float16, torch.bfloat16)
        else time
    )
    start_angle = offset_value / (1.0 + offset_value) * math.pi * 0.5
    angle = (
        (schedule_time + offset_value)
        / (1.0 + offset_value)
        * math.pi
        * 0.5
    )
    alpha_bar = angle.cos().square() / (math.cos(start_angle) ** 2)
    # A fixed epsilon avoids bfloat16's very coarse endpoint epsilon because
    # schedule_time has already been promoted above.
    alpha_bar = alpha_bar.clamp(1.0e-12, 1.0 - 1.0e-12)
    log_snr = torch.log(alpha_bar) - torch.log1p(-alpha_bar)
    return log_snr.clamp(-clip_value, clip_value)


def log_snr_to_alpha_sigma(
    log_snr: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert log-SNR to coefficients satisfying alpha^2 + sigma^2 = 1."""

    if not isinstance(log_snr, torch.Tensor):
        raise TypeError("log_snr must be a torch.Tensor")
    if not torch.is_floating_point(log_snr):
        raise TypeError("log_snr must use a floating-point dtype")
    alpha = torch.sigmoid(log_snr).sqrt()
    sigma = torch.sigmoid(-log_snr).sqrt()
    return alpha, sigma


def _as_binary_mask(
    tensor: torch.Tensor,
    *,
    name: str,
    expected_shape: torch.Size,
    device: torch.device,
) -> torch.Tensor:
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if tensor.shape != expected_shape:
        raise ValueError(
            f"{name} must have shape {tuple(expected_shape)}, got {tuple(tensor.shape)}"
        )
    if tensor.device != device:
        raise ValueError(f"{name} must be on device {device}, got {tensor.device}")
    if tensor.dtype == torch.bool:
        return tensor
    if torch.is_floating_point(tensor) and not bool(torch.isfinite(tensor).all()):
        raise ValueError(f"{name} must contain only finite values")
    is_binary = (tensor == 0) | (tensor == 1)
    if not bool(is_binary.all()):
        raise ValueError(f"{name} must contain only 0/1 or boolean values")
    return tensor.to(dtype=torch.bool)


def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    # FP16 cannot represent counts above 65,504, which a modest full-panel
    # PBISC batch already exceeds.  Accumulate reductions in float32 while
    # retaining float64 when the caller deliberately uses it.
    accumulation_dtype = (
        torch.float32
        if values.dtype in (torch.float16, torch.bfloat16)
        else values.dtype
    )
    accumulation_values = values.to(dtype=accumulation_dtype)
    weights = mask.to(dtype=accumulation_dtype)
    return (accumulation_values * weights).sum() / weights.sum().clamp_min(1.0)


class SparseDataDiffusion(nn.Module):
    """PBISC value-support diffusion with a measured-only training objective."""

    def __init__(
        self,
        denoiser: nn.Module,
        config: SparseDiffusionConfig | None = None,
        *,
        n_genes: int | None = None,
    ) -> None:
        super().__init__()
        if not isinstance(denoiser, nn.Module):
            raise TypeError("denoiser must be a torch.nn.Module")
        self.denoiser = denoiser
        self.config = config if config is not None else SparseDiffusionConfig()
        if not isinstance(self.config, SparseDiffusionConfig):
            raise TypeError("config must be a SparseDiffusionConfig or None")

        model_n_genes = getattr(denoiser, "n_genes", None)
        if n_genes is None:
            n_genes = model_n_genes
        if isinstance(n_genes, bool) or not isinstance(n_genes, int) or n_genes <= 0:
            raise ValueError(
                "n_genes must be a positive integer or exposed by denoiser.n_genes"
            )
        if model_n_genes is not None and model_n_genes != n_genes:
            raise ValueError(
                f"n_genes={n_genes} disagrees with denoiser.n_genes={model_n_genes}"
            )
        self.n_genes = n_genes
        self.state_dim = 2 * n_genes

    @property
    def model(self) -> nn.Module:
        """Compatibility alias for code following the reference repository."""

        return self.denoiser

    def _validate_values_condition(
        self,
        values: torch.Tensor,
        condition: torch.Tensor,
    ) -> None:
        if not isinstance(values, torch.Tensor):
            raise TypeError("values must be a torch.Tensor")
        if not torch.is_floating_point(values):
            raise TypeError("values must use a floating-point dtype")
        if values.ndim != 2 or values.shape[1] != self.n_genes:
            raise ValueError(
                f"values must have shape [batch, {self.n_genes}], "
                f"got {tuple(values.shape)}"
            )
        if not isinstance(condition, torch.Tensor):
            raise TypeError("condition must be a torch.Tensor")
        if not torch.is_floating_point(condition):
            raise TypeError("condition must use a floating-point dtype")
        if condition.ndim != 2 or condition.shape[0] != values.shape[0]:
            raise ValueError(
                "condition must have shape [batch, condition_features] with the "
                f"same batch as values, got {tuple(condition.shape)}"
            )
        if condition.device != values.device:
            raise ValueError("values and condition must be on the same device")
        expected_condition_dim = getattr(self.denoiser, "condition_dim", None)
        if (
            isinstance(expected_condition_dim, int)
            and condition.shape[1] != expected_condition_dim
        ):
            raise ValueError(
                f"condition width must be {expected_condition_dim}, "
                f"got {condition.shape[1]}"
            )
        if not bool(torch.isfinite(condition).all()):
            raise ValueError("condition must contain only finite values")

    def _resolve_measurement_mask(
        self,
        values: torch.Tensor,
        *,
        measurement_mask: torch.Tensor | None,
        availability_mask: torch.Tensor | None,
        required: bool,
    ) -> torch.Tensor:
        if measurement_mask is not None and availability_mask is not None:
            raise ValueError(
                "pass only one of measurement_mask or availability_mask; "
                "they are semantic aliases"
            )
        selected = (
            measurement_mask
            if measurement_mask is not None
            else availability_mask
        )
        if selected is None:
            if required:
                raise ValueError(
                    "measurement_mask (or availability_mask alias) is required"
                )
            return torch.ones_like(values, dtype=torch.bool)
        name = (
            "measurement_mask"
            if measurement_mask is not None
            else "availability_mask"
        )
        return _as_binary_mask(
            selected,
            name=name,
            expected_shape=values.shape,
            device=values.device,
        )

    def _resolve_active_target(
        self,
        values: torch.Tensor,
        active_target: torch.Tensor | None,
        *,
        measurement_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if active_target is None:
            return values != 0
        if not isinstance(active_target, torch.Tensor):
            raise TypeError("active_target must be a torch.Tensor")
        if active_target.shape != values.shape:
            raise ValueError(
                "active_target must have shape "
                f"{tuple(values.shape)}, got {tuple(active_target.shape)}"
            )
        if active_target.device != values.device:
            raise ValueError("active_target and values must be on the same device")
        if active_target.dtype == torch.bool:
            return active_target
        selected = (
            torch.ones_like(values, dtype=torch.bool)
            if measurement_mask is None
            else measurement_mask
        )
        observed_target = active_target[selected]
        if torch.is_floating_point(observed_target) and not bool(
            torch.isfinite(observed_target).all()
        ):
            raise ValueError(
                "active_target must contain finite values at measured coordinates"
            )
        zero_one = ((observed_target == 0) | (observed_target == 1)).all()
        signed = ((observed_target == -1) | (observed_target == 1)).all()
        if not bool(zero_one | signed):
            raise ValueError(
                "measured active_target values must use boolean, 0/1, or "
                "-1/+1 active encoding"
            )
        return active_target > 0

    def make_clean_state(
        self,
        values: torch.Tensor,
        active_target: torch.Tensor | None = None,
        *,
        measurement_mask: torch.Tensor | None = None,
        availability_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Construct ``[scaled_value, analog_active_support]`` in model space.

        The mask is optional for standalone use and defaults to all measured.
        During training it is required by :meth:`training_loss`.
        """

        if not isinstance(values, torch.Tensor):
            raise TypeError("values must be a torch.Tensor")
        if not torch.is_floating_point(values):
            raise TypeError("values must use a floating-point dtype")
        if values.ndim != 2 or values.shape[1] != self.n_genes:
            raise ValueError(
                f"values must have shape [batch, {self.n_genes}], "
                f"got {tuple(values.shape)}"
            )
        measured = self._resolve_measurement_mask(
            values,
            measurement_mask=measurement_mask,
            availability_mask=availability_mask,
            required=False,
        )
        active = self._resolve_active_target(
            values,
            active_target,
            measurement_mask=measured,
        )
        measured_values = values[measured]
        if not bool(torch.isfinite(measured_values).all()):
            raise ValueError("measured values must contain only finite values")
        if bool((measured_values < 0.0).any()):
            raise ValueError("PBISC measured values must be non-negative")
        expected_active = values != 0
        if bool((active[measured] != expected_active[measured]).any()):
            raise ValueError(
                "active_target must equal (values != 0) at measured coordinates"
            )

        value_scale = float(self.config.value_scale)
        scaled_values = values / value_scale
        if self.config.center_values:
            scaled_values = scaled_values * 2.0 - 1.0
        scaled_values = torch.where(measured, scaled_values, torch.zeros_like(values))
        support_scale = float(self.config.support_scale)
        analog_support = torch.where(
            active,
            torch.full_like(values, support_scale),
            torch.full_like(values, -support_scale),
        )
        # Missing assay coordinates are unknown, not observed zeros.  A neutral
        # target also prevents their filler value from leaking into q(x_t|x0).
        analog_support = torch.where(
            measured,
            analog_support,
            torch.zeros_like(analog_support),
        )
        return torch.cat((scaled_values, analog_support), dim=-1)

    def q_sample(
        self,
        clean_state: torch.Tensor,
        time: torch.Tensor,
        noise: torch.Tensor | None = None,
        *,
        generator: torch.Generator | None = None,
    ) -> ForwardCorruption:
        """Sample the continuous cosine forward process at per-row times."""

        if not isinstance(clean_state, torch.Tensor):
            raise TypeError("clean_state must be a torch.Tensor")
        if not torch.is_floating_point(clean_state):
            raise TypeError("clean_state must use a floating-point dtype")
        if clean_state.ndim != 2 or clean_state.shape[1] != self.state_dim:
            raise ValueError(
                f"clean_state must have shape [batch, {self.state_dim}], "
                f"got {tuple(clean_state.shape)}"
            )
        batch_size = clean_state.shape[0]
        if not isinstance(time, torch.Tensor):
            raise TypeError("time must be a torch.Tensor")
        if not torch.is_floating_point(time):
            raise TypeError("time must use a floating-point dtype")
        if time.shape != (batch_size,):
            raise ValueError(
                f"time must have shape [{batch_size}], got {tuple(time.shape)}"
            )
        if time.device != clean_state.device:
            raise ValueError("time and clean_state must be on the same device")
        if noise is None:
            noise = torch.randn(
                clean_state.shape,
                device=clean_state.device,
                dtype=clean_state.dtype,
                generator=generator,
            )
        else:
            if not isinstance(noise, torch.Tensor):
                raise TypeError("noise must be a torch.Tensor or None")
            if noise.shape != clean_state.shape:
                raise ValueError("noise shape must match clean_state")
            if noise.device != clean_state.device:
                raise ValueError("noise and clean_state must be on the same device")
            if not torch.is_floating_point(noise):
                raise TypeError("noise must use a floating-point dtype")
            noise = noise.to(dtype=clean_state.dtype)

        log_snr_vector = cosine_log_snr(
            time,
            offset=self.config.cosine_offset,
            clip=self.config.log_snr_clip,
        )
        alpha, sigma = log_snr_to_alpha_sigma(log_snr_vector)
        alpha = alpha[:, None].to(dtype=clean_state.dtype)
        sigma = sigma[:, None].to(dtype=clean_state.dtype)
        noisy_state = alpha * clean_state + sigma * noise
        return ForwardCorruption(
            noisy_state=noisy_state,
            noise=noise,
            log_snr=log_snr_vector[:, None],
            alpha=alpha,
            sigma=sigma,
            time=time,
        )

    def _coerce_prediction(
        self,
        raw_output: object,
        *,
        batch_size: int,
        device: torch.device,
    ) -> DenoiserOutput:
        if isinstance(raw_output, DenoiserOutput):
            prediction = raw_output
        elif isinstance(raw_output, torch.Tensor):
            if raw_output.shape != (batch_size, self.state_dim):
                raise ValueError(
                    "tensor denoiser output must have shape "
                    f"[{batch_size}, {self.state_dim}]"
                )
            value_prediction, active_logits = raw_output.split(self.n_genes, dim=-1)
            prediction = DenoiserOutput(value_prediction, active_logits)
        elif isinstance(raw_output, (tuple, list)) and len(raw_output) == 2:
            value_prediction, active_logits = raw_output
            if not isinstance(value_prediction, torch.Tensor) or not isinstance(
                active_logits, torch.Tensor
            ):
                raise TypeError("both tuple denoiser outputs must be tensors")
            prediction = DenoiserOutput(value_prediction, active_logits)
        else:
            raise TypeError(
                "denoiser must return DenoiserOutput, a [B,2G] tensor, or a "
                "(value_prediction, active_logits) pair"
            )

        expected = (batch_size, self.n_genes)
        if prediction.value_prediction.shape != expected:
            raise ValueError(
                f"value_prediction must have shape {expected}, "
                f"got {tuple(prediction.value_prediction.shape)}"
            )
        if prediction.active_logits.shape != expected:
            raise ValueError(
                f"active_logits must have shape {expected}, "
                f"got {tuple(prediction.active_logits.shape)}"
            )
        for name, tensor in (
            ("value_prediction", prediction.value_prediction),
            ("active_logits", prediction.active_logits),
        ):
            if tensor.device != device:
                raise ValueError(f"{name} must be on the denoiser input device")
            if not torch.is_floating_point(tensor):
                raise TypeError(f"{name} must use a floating-point dtype")
        return prediction

    def _predict(
        self,
        noisy_state: torch.Tensor,
        time: torch.Tensor,
        condition: torch.Tensor,
        self_condition: torch.Tensor | None,
    ) -> DenoiserOutput:
        raw_output = self.denoiser(noisy_state, time, condition, self_condition)
        return self._coerce_prediction(
            raw_output,
            batch_size=noisy_state.shape[0],
            device=noisy_state.device,
        )

    def training_loss(
        self,
        values: torch.Tensor,
        condition: torch.Tensor,
        *,
        measurement_mask: torch.Tensor | None = None,
        availability_mask: torch.Tensor | None = None,
        active_target: torch.Tensor | None = None,
        time: torch.Tensor | None = None,
        noise: torch.Tensor | None = None,
        generator: torch.Generator | None = None,
    ) -> SparseDiffusionLoss:
        """Compute measured-only value x0 loss and active-support BCE loss."""

        self._validate_values_condition(values, condition)
        measured = self._resolve_measurement_mask(
            values,
            measurement_mask=measurement_mask,
            availability_mask=availability_mask,
            required=True,
        )
        active = self._resolve_active_target(
            values,
            active_target,
            measurement_mask=measured,
        )
        clean_state = self.make_clean_state(
            values,
            active,
            measurement_mask=measured,
        )
        batch_size = values.shape[0]
        if time is None:
            time = torch.rand(
                (batch_size,),
                device=values.device,
                dtype=values.dtype,
                generator=generator,
            )
        corruption = self.q_sample(
            clean_state,
            time,
            noise,
            generator=generator,
        )

        self_condition: torch.Tensor | None = None
        probability = float(self.config.self_condition_probability)
        use_self_condition = probability >= 1.0
        if 0.0 < probability < 1.0:
            draw = torch.rand((), device=values.device, generator=generator)
            use_self_condition = bool(draw < probability)
        if use_self_condition:
            with torch.no_grad():
                preliminary = self._predict(
                    corruption.noisy_state,
                    time,
                    condition,
                    None,
                )
                self_condition = preliminary.continuous_x0(
                    support_scale=self.config.support_scale
                ).detach()

        prediction = self._predict(
            corruption.noisy_state,
            time,
            condition,
            self_condition,
        )
        scaled_target = clean_state[:, : self.n_genes]
        if self.config.value_loss_mode == "all_measured":
            value_mask = measured
        else:
            value_mask = measured & active
        value_loss = _masked_mean(
            (prediction.value_prediction - scaled_target).square(),
            value_mask,
        )
        support_element_loss = F.binary_cross_entropy_with_logits(
            prediction.active_logits,
            active.to(dtype=prediction.active_logits.dtype),
            reduction="none",
        )
        support_loss = _masked_mean(support_element_loss, measured)
        loss = (
            float(self.config.value_loss_weight) * value_loss
            + float(self.config.support_loss_weight) * support_loss
        )
        return SparseDiffusionLoss(
            loss=loss,
            value_loss=value_loss,
            support_loss=support_loss,
            prediction=prediction,
            corruption=corruption,
            clean_state=clean_state,
            self_condition=self_condition,
        )

    def forward(
        self,
        values: torch.Tensor,
        condition: torch.Tensor,
        *,
        measurement_mask: torch.Tensor | None = None,
        availability_mask: torch.Tensor | None = None,
        active_target: torch.Tensor | None = None,
        time: torch.Tensor | None = None,
        noise: torch.Tensor | None = None,
        generator: torch.Generator | None = None,
    ) -> SparseDiffusionLoss:
        return self.training_loss(
            values,
            condition,
            measurement_mask=measurement_mask,
            availability_mask=availability_mask,
            active_target=active_target,
            time=time,
            noise=noise,
            generator=generator,
        )

    @torch.no_grad()
    def sample(
        self,
        condition: torch.Tensor,
        *,
        steps: int = 50,
        eta: float = 0.0,
        active_logit_threshold: float = 0.0,
        measurement_mask: torch.Tensor | None = None,
        availability_mask: torch.Tensor | None = None,
        initial_noise: torch.Tensor | None = None,
        clamp_values_min: float | None = 0.0,
        generator: torch.Generator | None = None,
    ) -> SparseSample:
        """Generate exact-sparse values with continuous-time DDIM.

        With ``eta=0``, the result is deterministic for fixed ``initial_noise``
        or for generators initialized with the same seed.
        """

        if isinstance(steps, bool) or not isinstance(steps, int) or steps <= 0:
            raise ValueError("steps must be a positive integer")
        eta_value = _finite_non_negative("eta", eta)
        if eta_value > 1.0:
            raise ValueError("eta must be in [0, 1]")
        threshold = float(active_logit_threshold)
        if not math.isfinite(threshold):
            raise ValueError("active_logit_threshold must be finite")
        if clamp_values_min is not None and not math.isfinite(
            float(clamp_values_min)
        ):
            raise ValueError("clamp_values_min must be finite or None")
        if not isinstance(condition, torch.Tensor):
            raise TypeError("condition must be a torch.Tensor")
        if not torch.is_floating_point(condition):
            raise TypeError("condition must use a floating-point dtype")
        if condition.ndim != 2 or condition.shape[0] <= 0:
            raise ValueError("condition must have shape [batch, condition_features]")
        expected_condition_dim = getattr(self.denoiser, "condition_dim", None)
        if (
            isinstance(expected_condition_dim, int)
            and condition.shape[1] != expected_condition_dim
        ):
            raise ValueError(
                f"condition width must be {expected_condition_dim}, "
                f"got {condition.shape[1]}"
            )
        if not bool(torch.isfinite(condition).all()):
            raise ValueError("condition must contain only finite values")

        reference_parameter = next(
            (
                parameter
                for parameter in self.denoiser.parameters()
                if torch.is_floating_point(parameter)
            ),
            None,
        )
        if (
            reference_parameter is not None
            and condition.device != reference_parameter.device
        ):
            raise ValueError(
                f"condition is on {condition.device}, but denoiser is on "
                f"{reference_parameter.device}"
            )
        sampling_dtype = (
            reference_parameter.dtype
            if reference_parameter is not None
            else condition.dtype
        )
        condition = condition.to(dtype=sampling_dtype)

        batch_size = condition.shape[0]
        state_shape = (batch_size, self.state_dim)
        if measurement_mask is not None and availability_mask is not None:
            raise ValueError(
                "pass only one of measurement_mask or availability_mask; "
                "they are semantic aliases"
            )
        requested_measurement_mask = (
            measurement_mask
            if measurement_mask is not None
            else availability_mask
        )
        generation_mask: torch.Tensor | None = None
        if requested_measurement_mask is not None:
            generation_mask = _as_binary_mask(
                requested_measurement_mask,
                name=(
                    "measurement_mask"
                    if measurement_mask is not None
                    else "availability_mask"
                ),
                expected_shape=torch.Size((batch_size, self.n_genes)),
                device=condition.device,
            )
        if initial_noise is None:
            state = torch.randn(
                state_shape,
                device=condition.device,
                dtype=sampling_dtype,
                generator=generator,
            )
        else:
            if not isinstance(initial_noise, torch.Tensor):
                raise TypeError("initial_noise must be a torch.Tensor or None")
            if initial_noise.shape != state_shape:
                raise ValueError(
                    f"initial_noise must have shape {state_shape}, "
                    f"got {tuple(initial_noise.shape)}"
                )
            if initial_noise.device != condition.device:
                raise ValueError("initial_noise and condition must share a device")
            if not torch.is_floating_point(initial_noise):
                raise TypeError("initial_noise must use a floating-point dtype")
            state = initial_noise.to(dtype=sampling_dtype).clone()

        times = torch.linspace(
            1.0,
            0.0,
            steps + 1,
            device=condition.device,
            dtype=torch.float32,
        )
        schedule_log_snr = cosine_log_snr(
            times,
            offset=self.config.cosine_offset,
            clip=self.config.log_snr_clip,
        )
        schedule_alpha, schedule_sigma = log_snr_to_alpha_sigma(schedule_log_snr)
        self_condition: torch.Tensor | None = None
        final_prediction: DenoiserOutput | None = None
        final_value_prediction: torch.Tensor | None = None
        was_training = self.denoiser.training
        self.denoiser.eval()
        try:
            for index in range(steps):
                time = times[index].expand(batch_size)
                prediction = self._predict(
                    state,
                    time,
                    condition,
                    self_condition,
                )
                value_prediction = prediction.value_prediction
                if self.config.clip_denoised:
                    minimum = -1.0 if self.config.center_values else 0.0
                    value_prediction = value_prediction.clamp(minimum, 1.0)
                analog_support = torch.tanh(
                    prediction.active_logits * 0.5
                ) * float(self.config.support_scale)
                x0 = torch.cat((value_prediction, analog_support), dim=-1).to(
                    dtype=state.dtype
                )
                final_prediction = prediction
                final_value_prediction = value_prediction
                self_condition = x0.detach()

                if index == steps - 1:
                    state = x0
                    continue

                calculation_dtype = (
                    torch.float32
                    if state.dtype in (torch.float16, torch.bfloat16)
                    else state.dtype
                )
                alpha = schedule_alpha[index].to(dtype=calculation_dtype).reshape(1, 1)
                sigma = schedule_sigma[index].to(dtype=calculation_dtype).reshape(1, 1)
                next_alpha = schedule_alpha[index + 1].to(
                    dtype=calculation_dtype
                ).reshape(1, 1)
                next_sigma = schedule_sigma[index + 1].to(
                    dtype=calculation_dtype
                ).reshape(1, 1)
                calculation_state = state.to(dtype=calculation_dtype)
                calculation_x0 = x0.to(dtype=calculation_dtype)

                predicted_noise = (
                    calculation_state - alpha * calculation_x0
                ) / sigma.clamp_min(1.0e-8)
                alpha_bar = alpha.square()
                next_alpha_bar = next_alpha.square()
                ddim_variance = (
                    next_sigma.square()
                    / sigma.square().clamp_min(1.0e-12)
                    * (1.0 - alpha_bar / next_alpha_bar.clamp_min(1.0e-12))
                ).clamp_min(0.0)
                ddim_sigma = eta_value * ddim_variance.sqrt()
                direction_scale = (
                    next_sigma.square() - ddim_sigma.square()
                ).clamp_min(0.0).sqrt()
                if eta_value > 0.0:
                    step_noise = torch.randn(
                        state_shape,
                        device=state.device,
                        dtype=calculation_dtype,
                        generator=generator,
                    )
                else:
                    step_noise = torch.zeros_like(
                        calculation_state,
                        dtype=calculation_dtype,
                    )
                state = (
                    next_alpha * calculation_x0
                    + direction_scale * predicted_noise
                    + ddim_sigma * step_noise
                ).to(dtype=sampling_dtype)
        finally:
            self.denoiser.train(was_training)

        if final_prediction is None or final_value_prediction is None:
            raise RuntimeError("DDIM sampler produced no prediction")
        dense_values = final_value_prediction
        if self.config.center_values:
            dense_values = (dense_values + 1.0) * 0.5
        dense_values = dense_values * float(self.config.value_scale)
        if clamp_values_min is not None:
            dense_values = dense_values.clamp_min(float(clamp_values_min))
        active_mask = final_prediction.active_logits > threshold
        if generation_mask is not None:
            active_mask = active_mask & generation_mask
        # ``where`` guarantees literal zeros even if an inactive dense
        # prediction is NaN or infinite; multiplication would yield NaN.
        values = torch.where(
            active_mask,
            dense_values,
            torch.zeros_like(dense_values),
        )
        return SparseSample(
            values=values,
            dense_values=dense_values,
            active_mask=active_mask,
            active_logits=final_prediction.active_logits,
            final_state=state,
            steps=steps,
            eta=eta_value,
        )


__all__ = [
    "ForwardCorruption",
    "SparseDataDiffusion",
    "SparseDiffusionConfig",
    "SparseDiffusionLoss",
    "SparseSample",
    "ValueLossMode",
    "cosine_log_snr",
    "log_snr_to_alpha_sigma",
]
