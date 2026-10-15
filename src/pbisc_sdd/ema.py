"""Exponential moving averages for stable model evaluation."""

from __future__ import annotations

from copy import deepcopy
import math
from typing import Generic, Mapping, TypeVar, cast

import torch
from torch import Tensor, nn


ModelT = TypeVar("ModelT", bound=nn.Module)

EMA_STATE_SCHEMA = "pbisc_sdd.ema.v1"
_STATE_KEYS = {
    "schema_version",
    "decay",
    "update_every",
    "num_updates",
    "model_state",
}


class ExponentialMovingAverage(Generic[ModelT]):
    """Maintain a frozen evaluation copy of a PyTorch model.

    ``num_updates`` counts calls to :meth:`update`. The shadow model is updated
    whenever that count is divisible by ``update_every``. The first applied
    update copies the trained model exactly so a short run does not retain a
    large contribution from random initialization. Later floating-point
    parameters use ``shadow = decay * shadow + (1 - decay) * source``;
    non-floating parameters and all buffers are copied exactly.
    """

    def __init__(
        self,
        model: ModelT,
        *,
        decay: float,
        update_every: int = 1,
    ) -> None:
        if isinstance(decay, bool) or not isinstance(decay, (int, float)):
            raise ValueError("decay must be a finite number in [0, 1)")
        decay_value = float(decay)
        if not math.isfinite(decay_value) or not 0.0 <= decay_value < 1.0:
            raise ValueError("decay must be a finite number in [0, 1)")
        if (
            isinstance(update_every, bool)
            or not isinstance(update_every, int)
            or update_every <= 0
        ):
            raise ValueError("update_every must be a positive integer")

        self.decay = decay_value
        self.update_every = update_every
        self.num_updates = 0
        self.model = deepcopy(model)
        self._freeze_shadow()

    @property
    def shadow_model(self) -> ModelT:
        """Return the frozen model copy used for evaluation and sampling."""

        return self.model

    def _freeze_shadow(self) -> None:
        self.model.eval()
        self.model.requires_grad_(False)

    @staticmethod
    def _validate_tensor_structure(
        shadow: Mapping[str, Tensor],
        source: Mapping[str, Tensor],
        *,
        kind: str,
    ) -> None:
        if shadow.keys() != source.keys():
            missing = sorted(shadow.keys() - source.keys())
            unexpected = sorted(source.keys() - shadow.keys())
            raise ValueError(
                f"source model {kind} differ from the EMA model "
                f"(missing={missing}, unexpected={unexpected})"
            )
        for name, shadow_tensor in shadow.items():
            source_tensor = source[name]
            if shadow_tensor.shape != source_tensor.shape:
                raise ValueError(
                    f"source model {kind} {name!r} has shape "
                    f"{tuple(source_tensor.shape)}, expected {tuple(shadow_tensor.shape)}"
                )
            if shadow_tensor.dtype != source_tensor.dtype:
                raise ValueError(
                    f"source model {kind} {name!r} has dtype "
                    f"{source_tensor.dtype}, expected {shadow_tensor.dtype}"
                )

    @torch.no_grad()
    def update(self, model: nn.Module) -> bool:
        """Process one training update and return whether the EMA was applied."""

        next_update = self.num_updates + 1
        if next_update % self.update_every != 0:
            self.num_updates = next_update
            self._freeze_shadow()
            return False

        shadow_parameters = dict(self.model.named_parameters())
        source_parameters = dict(model.named_parameters())
        shadow_buffers = dict(self.model.named_buffers())
        source_buffers = dict(model.named_buffers())
        self._validate_tensor_structure(
            shadow_parameters, source_parameters, kind="parameters"
        )
        self._validate_tensor_structure(shadow_buffers, source_buffers, kind="buffers")

        first_applied_update = self.num_updates < self.update_every
        blend_weight = 1.0 - self.decay
        for name, shadow_parameter in shadow_parameters.items():
            source_parameter = source_parameters[name].detach()
            if shadow_parameter.is_floating_point() and not first_applied_update:
                shadow_parameter.lerp_(source_parameter, blend_weight)
            else:
                shadow_parameter.copy_(source_parameter)
        for name, shadow_buffer in shadow_buffers.items():
            shadow_buffer.copy_(source_buffers[name].detach())

        self.num_updates = next_update
        self._freeze_shadow()
        return True

    def state_dict(self) -> dict[str, object]:
        """Return the shadow weights and the configuration required to restore them."""

        return {
            "schema_version": EMA_STATE_SCHEMA,
            "decay": self.decay,
            "update_every": self.update_every,
            "num_updates": self.num_updates,
            "model_state": self.model.state_dict(),
        }

    def load_state_dict(self, state: Mapping[str, object]) -> None:
        """Restore an EMA state after strictly validating its configuration."""

        actual_keys = set(state)
        if actual_keys != _STATE_KEYS:
            missing = sorted(_STATE_KEYS - actual_keys)
            unexpected = sorted(actual_keys - _STATE_KEYS)
            raise ValueError(
                "invalid EMA state keys "
                f"(missing={missing}, unexpected={unexpected})"
            )
        if state["schema_version"] != EMA_STATE_SCHEMA:
            raise ValueError(
                f"EMA state schema must be {EMA_STATE_SCHEMA!r}"
            )

        state_decay = state["decay"]
        if isinstance(state_decay, bool) or not isinstance(state_decay, (int, float)):
            raise ValueError("EMA state decay must be numeric")
        if float(state_decay) != self.decay:
            raise ValueError(
                f"EMA decay mismatch: state has {float(state_decay)}, "
                f"instance expects {self.decay}"
            )

        state_update_every = state["update_every"]
        if (
            isinstance(state_update_every, bool)
            or not isinstance(state_update_every, int)
            or state_update_every != self.update_every
        ):
            raise ValueError(
                f"EMA update_every mismatch: state has {state_update_every!r}, "
                f"instance expects {self.update_every}"
            )

        state_num_updates = state["num_updates"]
        if (
            isinstance(state_num_updates, bool)
            or not isinstance(state_num_updates, int)
            or state_num_updates < 0
        ):
            raise ValueError("EMA state num_updates must be a non-negative integer")
        model_state = state["model_state"]
        if not isinstance(model_state, Mapping):
            raise ValueError("EMA state model_state must be a mapping")

        self.model.load_state_dict(
            cast(Mapping[str, Tensor], model_state), strict=True
        )
        self.num_updates = state_num_updates
        self._freeze_shadow()


__all__ = ["EMA_STATE_SCHEMA", "ExponentialMovingAverage"]
