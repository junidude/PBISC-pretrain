"""Conditioned MLP denoiser for PBISC sparse-data diffusion.

The network follows the skip-connected ``Cell_Unet`` idea from
PhilSid/sparse-data-diffusion, but exposes two semantically distinct heads:
one predicts the clean (scaled) expression values and one predicts logits for
the event that a measured gene is active.  The latter are classification
logits, not VAE latents and not "zero" logits.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import nn


def _positive_int(name: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer, got {value!r}")
    return value


@dataclass(frozen=True)
class SparseDenoiserConfig:
    """Architecture of :class:`ConditionalSBDDenoiser`.

    ``hidden_dims`` describes the descending half of the symmetric MLP.  For
    example ``(512, 512, 256, 128)`` descends to a 128-wide internal layer and
    then returns through additive skip connections.  This is ordinary hidden
    computation inside the denoiser, not a sampled or persisted latent code.
    """

    n_genes: int
    condition_dim: int
    hidden_dims: tuple[int, ...] = (512, 512, 256, 128)
    time_embedding_dim: int = 128
    context_dim: int = 256
    dropout: float = 0.0
    max_time_period: float = 10_000.0

    def __post_init__(self) -> None:
        _positive_int("n_genes", self.n_genes)
        _positive_int("condition_dim", self.condition_dim)
        _positive_int("time_embedding_dim", self.time_embedding_dim)
        _positive_int("context_dim", self.context_dim)
        if not isinstance(self.hidden_dims, tuple) or not self.hidden_dims:
            raise ValueError("hidden_dims must be a non-empty tuple")
        for index, width in enumerate(self.hidden_dims):
            _positive_int(f"hidden_dims[{index}]", width)
        dropout = float(self.dropout)
        if not math.isfinite(dropout) or not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be finite and in [0, 1)")
        period = float(self.max_time_period)
        if not math.isfinite(period) or period <= 1.0:
            raise ValueError("max_time_period must be finite and greater than 1")


@dataclass(frozen=True)
class DenoiserOutput:
    """Clean-value prediction and active/support classification logits."""

    value_prediction: torch.Tensor
    active_logits: torch.Tensor

    @property
    def support_logits(self) -> torch.Tensor:
        """Alias emphasizing that active logits predict the support mask."""

        return self.active_logits

    def continuous_x0(self, *, support_scale: float = 1.0) -> torch.Tensor:
        """Return the joint continuous x0 used by a diffusion transition.

        BCE training needs unconstrained logits, while the forward support
        variable has finite analog values.  ``2 * sigmoid(logit) - 1`` is the
        probability-consistent signed expectation and equals
        ``tanh(logit / 2)`` without changing the sign used for masking.
        """

        scale = float(support_scale)
        if not math.isfinite(scale) or scale <= 0.0:
            raise ValueError("support_scale must be finite and positive")
        if self.value_prediction.shape != self.active_logits.shape:
            raise ValueError(
                "value_prediction and active_logits must have equal shapes"
            )
        if self.value_prediction.ndim != 2:
            raise ValueError("denoiser outputs must be rank-2 [batch, genes] tensors")
        analog_support = torch.tanh(self.active_logits * 0.5) * scale
        return torch.cat((self.value_prediction, analog_support), dim=-1)


def sinusoidal_time_embedding(
    time: torch.Tensor,
    dimension: int,
    *,
    max_period: float = 10_000.0,
) -> torch.Tensor:
    """Embed continuous normalized times with fixed sinusoidal features."""

    _positive_int("dimension", dimension)
    if time.ndim != 1:
        raise ValueError(f"time must have shape [batch], got {tuple(time.shape)}")
    if not torch.is_floating_point(time):
        raise TypeError("time must use a floating-point dtype")
    half = dimension // 2
    if half == 0:
        return time[:, None]
    denominator = max(half - 1, 1)
    frequencies = torch.exp(
        -math.log(float(max_period))
        * torch.arange(half, device=time.device, dtype=time.dtype)
        / denominator
    )
    arguments = time[:, None] * frequencies[None, :] * (2.0 * math.pi)
    embedding = torch.cat((arguments.cos(), arguments.sin()), dim=-1)
    if dimension % 2:
        embedding = torch.cat((embedding, time[:, None]), dim=-1)
    return embedding


class _ConditionedBlock(nn.Module):
    """Linear block receiving the shared time/condition context."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        context_dim: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.input_projection = nn.Linear(input_dim, output_dim)
        self.context_projection = nn.Sequential(
            nn.SiLU(),
            nn.Linear(context_dim, output_dim),
        )
        self.normalization = nn.LayerNorm(output_dim)
        self.activation = nn.SiLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, inputs: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        hidden = self.input_projection(inputs) + self.context_projection(context)
        hidden = self.normalization(hidden)
        hidden = self.activation(hidden)
        return self.dropout(hidden)


class ConditionalSBDDenoiser(nn.Module):
    """Skip-connected conditional MLP predicting value x0 and active logits.

    Input and self-conditioning tensors both live in the expanded data space
    ``[scaled_value, analog_active_support]`` and therefore have width ``2G``.
    ``time`` is a normalized continuous diffusion time in ``[0, 1]``.
    """

    def __init__(self, config: SparseDenoiserConfig) -> None:
        super().__init__()
        if not isinstance(config, SparseDenoiserConfig):
            raise TypeError("config must be a SparseDenoiserConfig")
        self.config = config
        self.n_genes = config.n_genes
        self.condition_dim = config.condition_dim
        self.state_dim = 2 * config.n_genes

        self.time_encoder = nn.Sequential(
            nn.Linear(config.time_embedding_dim, config.context_dim),
            nn.SiLU(),
            nn.Linear(config.context_dim, config.context_dim),
        )
        self.condition_encoder = nn.Sequential(
            nn.Linear(config.condition_dim, config.context_dim),
            nn.LayerNorm(config.context_dim),
            nn.SiLU(),
            nn.Linear(config.context_dim, config.context_dim),
        )
        self.context_mixer = nn.Sequential(
            nn.LayerNorm(config.context_dim),
            nn.SiLU(),
            nn.Linear(config.context_dim, config.context_dim),
        )

        widths = config.hidden_dims
        down_blocks: list[nn.Module] = [
            _ConditionedBlock(
                2 * self.state_dim,
                widths[0],
                config.context_dim,
                config.dropout,
            )
        ]
        down_blocks.extend(
            _ConditionedBlock(
                widths[index],
                widths[index + 1],
                config.context_dim,
                config.dropout,
            )
            for index in range(len(widths) - 1)
        )
        self.down_blocks = nn.ModuleList(down_blocks)

        self.up_blocks = nn.ModuleList(
            _ConditionedBlock(
                widths[index + 1],
                widths[index],
                config.context_dim,
                config.dropout,
            )
            for index in reversed(range(len(widths) - 1))
        )
        self.output_trunk = nn.Sequential(
            nn.Linear(widths[0], widths[0]),
            nn.LayerNorm(widths[0]),
            nn.SiLU(),
            nn.Dropout(config.dropout),
        )
        self.value_head = nn.Linear(widths[0], config.n_genes)
        self.active_head = nn.Linear(widths[0], config.n_genes)

    def _validate_inputs(
        self,
        noisy_state: torch.Tensor,
        time: torch.Tensor,
        condition: torch.Tensor,
        self_condition: torch.Tensor | None,
    ) -> None:
        if not isinstance(noisy_state, torch.Tensor):
            raise TypeError("noisy_state must be a torch.Tensor")
        if not torch.is_floating_point(noisy_state):
            raise TypeError("noisy_state must use a floating-point dtype")
        if noisy_state.ndim != 2 or noisy_state.shape[1] != self.state_dim:
            raise ValueError(
                f"noisy_state must have shape [batch, {self.state_dim}], "
                f"got {tuple(noisy_state.shape)}"
            )
        batch_size = noisy_state.shape[0]
        if not isinstance(time, torch.Tensor):
            raise TypeError("time must be a torch.Tensor")
        if not torch.is_floating_point(time):
            raise TypeError("time must use a floating-point dtype")
        if time.shape != (batch_size,):
            raise ValueError(
                f"time must have shape [{batch_size}], got {tuple(time.shape)}"
            )
        if not isinstance(condition, torch.Tensor):
            raise TypeError("condition must be a torch.Tensor")
        if not torch.is_floating_point(condition):
            raise TypeError("condition must use a floating-point dtype")
        if condition.shape != (batch_size, self.condition_dim):
            raise ValueError(
                f"condition must have shape [{batch_size}, {self.condition_dim}], "
                f"got {tuple(condition.shape)}"
            )
        if self_condition is not None:
            if not isinstance(self_condition, torch.Tensor):
                raise TypeError("self_condition must be a torch.Tensor or None")
            if not torch.is_floating_point(self_condition):
                raise TypeError("self_condition must use a floating-point dtype")
            if self_condition.shape != noisy_state.shape:
                raise ValueError(
                    "self_condition shape must match noisy_state; "
                    f"got {tuple(self_condition.shape)} and {tuple(noisy_state.shape)}"
                )
        tensors = [time, condition]
        if self_condition is not None:
            tensors.append(self_condition)
        if any(tensor.device != noisy_state.device for tensor in tensors):
            raise ValueError("all denoiser inputs must be on the same device")
        model_device = self.value_head.weight.device
        if noisy_state.device != model_device:
            raise ValueError(
                f"denoiser inputs are on {noisy_state.device}, but model is on "
                f"{model_device}"
            )

    def forward(
        self,
        noisy_state: torch.Tensor,
        time: torch.Tensor,
        condition: torch.Tensor,
        self_condition: torch.Tensor | None = None,
    ) -> DenoiserOutput:
        self._validate_inputs(noisy_state, time, condition, self_condition)
        model_dtype = self.value_head.weight.dtype
        noisy_state = noisy_state.to(dtype=model_dtype)
        if self_condition is None:
            self_condition = torch.zeros_like(noisy_state)
        else:
            self_condition = self_condition.to(dtype=model_dtype)

        time_features = sinusoidal_time_embedding(
            time.to(dtype=model_dtype),
            self.config.time_embedding_dim,
            max_period=self.config.max_time_period,
        )
        time_context = self.time_encoder(time_features)
        condition_context = self.condition_encoder(condition.to(dtype=model_dtype))
        context = self.context_mixer(time_context + condition_context)

        hidden = torch.cat((noisy_state, self_condition), dim=-1)
        history: list[torch.Tensor] = []
        for block in self.down_blocks:
            hidden = block(hidden, context)
            history.append(hidden)

        history.pop()  # narrowest hidden layer has no same-width skip
        for block in self.up_blocks:
            hidden = block(hidden, context)
            hidden = hidden + history.pop()

        hidden = self.output_trunk(hidden)
        return DenoiserOutput(
            value_prediction=self.value_head(hidden),
            active_logits=self.active_head(hidden),
        )


# Concise alias used by early design notes and downstream experiment code.
SparseDataDenoiser = ConditionalSBDDenoiser
ConditionalSBDDenoiserConfig = SparseDenoiserConfig


__all__ = [
    "ConditionalSBDDenoiser",
    "ConditionalSBDDenoiserConfig",
    "DenoiserOutput",
    "SparseDataDenoiser",
    "SparseDenoiserConfig",
    "sinusoidal_time_embedding",
]
