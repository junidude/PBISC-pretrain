from __future__ import annotations

from copy import deepcopy

import pytest
import torch
from torch import nn

from pbisc_sdd.ema import EMA_STATE_SCHEMA, ExponentialMovingAverage


class _ModelWithBuffer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(2, 1, bias=True)
        self.register_buffer("running_value", torch.tensor([1.0, 2.0]))
        self.register_buffer("batch_count", torch.tensor(0, dtype=torch.long))


def _filled_model(value: float) -> _ModelWithBuffer:
    model = _ModelWithBuffer()
    with torch.no_grad():
        model.linear.weight.fill_(value)
        model.linear.bias.fill_(value + 1.0)
    return model


def test_exact_parameter_update_and_frozen_eval_shadow() -> None:
    source = _filled_model(2.0)
    ema = ExponentialMovingAverage(source, decay=0.75)

    assert ema.model is not source
    assert not ema.model.training
    assert all(not parameter.requires_grad for parameter in ema.model.parameters())

    with torch.no_grad():
        source.linear.weight.fill_(6.0)
        source.linear.bias.fill_(8.0)

    assert ema.update(source)
    assert ema.num_updates == 1
    assert torch.equal(ema.model.linear.weight, torch.full((1, 2), 6.0))
    assert torch.equal(ema.model.linear.bias, torch.tensor([8.0]))

    with torch.no_grad():
        source.linear.weight.fill_(10.0)
        source.linear.bias.fill_(12.0)
    ema.update(source)
    assert torch.equal(ema.model.linear.weight, torch.full((1, 2), 7.0))
    assert torch.equal(ema.model.linear.bias, torch.tensor([9.0]))


def test_update_every_counts_calls_and_skips_intermediate_models() -> None:
    source = _filled_model(1.0)
    ema = ExponentialMovingAverage(source, decay=0.5, update_every=2)

    with torch.no_grad():
        source.linear.weight.fill_(3.0)
    assert not ema.update(source)
    assert ema.num_updates == 1
    assert torch.equal(ema.model.linear.weight, torch.ones(1, 2))

    with torch.no_grad():
        source.linear.weight.fill_(5.0)
    assert ema.update(source)
    assert ema.num_updates == 2
    assert torch.equal(ema.model.linear.weight, torch.full((1, 2), 5.0))


def test_buffers_are_copied_instead_of_averaged() -> None:
    source = _filled_model(1.0)
    ema = ExponentialMovingAverage(source, decay=0.99)

    source.running_value.fill_(9.0)
    source.batch_count.fill_(7)
    ema.update(source)

    assert torch.equal(ema.model.running_value, torch.full((2,), 9.0))
    assert torch.equal(ema.model.batch_count, torch.tensor(7))


def test_state_dict_roundtrip_restores_shadow_and_counter() -> None:
    source = _filled_model(1.0)
    ema = ExponentialMovingAverage(source, decay=0.8, update_every=2)
    source.linear.weight.data.fill_(5.0)
    source.running_value.fill_(4.0)
    ema.update(source)
    ema.update(source)

    state = deepcopy(ema.state_dict())
    restored = ExponentialMovingAverage(
        _filled_model(-3.0), decay=0.8, update_every=2
    )
    restored.load_state_dict(state)

    assert restored.num_updates == 2
    assert not restored.model.training
    assert all(not parameter.requires_grad for parameter in restored.model.parameters())
    for name, value in ema.model.state_dict().items():
        assert torch.equal(restored.model.state_dict()[name], value)


@pytest.mark.parametrize(
    ("instance_kwargs", "state_change"),
    [
        ({"decay": 0.7, "update_every": 2}, {"decay": 0.8}),
        ({"decay": 0.7, "update_every": 2}, {"update_every": 3}),
        (
            {"decay": 0.7, "update_every": 2},
            {"schema_version": "pbisc_sdd.ema.unknown"},
        ),
    ],
    ids=("decay", "update-every", "schema"),
)
def test_load_state_dict_rejects_config_mismatch(
    instance_kwargs: dict[str, float | int],
    state_change: dict[str, object],
) -> None:
    ema = ExponentialMovingAverage(_filled_model(1.0), **instance_kwargs)
    state = ema.state_dict() | state_change

    with pytest.raises(ValueError):
        ema.load_state_dict(state)


def test_load_state_dict_requires_exact_keys() -> None:
    ema = ExponentialMovingAverage(_filled_model(1.0), decay=0.9)
    state = ema.state_dict()
    state["extra"] = EMA_STATE_SCHEMA

    with pytest.raises(ValueError, match="invalid EMA state keys"):
        ema.load_state_dict(state)
