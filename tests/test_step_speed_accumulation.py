"""Verify that one timed step is one correctly accumulated optimizer update."""

import copy
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts/step_speed"))

from benchmark import backward_accumulated, case_name, grid, step


class ToyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(3, 2)
        self.calls = 0

    def forward(self, input_ids, labels):
        self.calls += 1
        return SimpleNamespace(loss=torch.nn.functional.mse_loss(self.linear(input_ids), labels))


@pytest.mark.parametrize("accumulation", [2, 4])
def test_accumulated_gradient_and_loss_equal_full_batch(accumulation):
    torch.manual_seed(9)
    model = ToyModel()
    reference = copy.deepcopy(model)
    batch = {"input_ids": torch.randn(8, 3), "labels": torch.randn(8, 2)}
    expected_loss = reference(**batch).loss
    expected_loss.backward()
    actual_loss = backward_accumulated(model, batch, accumulation)
    torch.testing.assert_close(actual_loss, expected_loss.detach())
    for actual, expected in zip(model.parameters(), reference.parameters()):
        torch.testing.assert_close(actual.grad, expected.grad)
    assert model.calls == accumulation


@pytest.mark.parametrize("accumulation", [2, 4])
def test_one_clip_update_and_clear_per_effective_batch(accumulation):
    torch.manual_seed(10)
    model = ToyModel()
    reference = copy.deepcopy(model)
    params = list(model.parameters())
    optimizer = torch.optim.SGD(params, lr=.1)
    ref_params = list(reference.parameters())
    ref_optimizer = torch.optim.SGD(ref_params, lr=.1)
    batch = {"input_ids": torch.randn(8, 3), "labels": torch.randn(8, 2)}
    reference(**batch).loss.backward()
    torch.nn.utils.clip_grad_norm_(ref_params, 1., foreach=True)
    ref_optimizer.step()
    with patch.object(optimizer, "step", wraps=optimizer.step) as update, \
         patch.object(optimizer, "zero_grad", wraps=optimizer.zero_grad) as clear, \
         patch("torch.nn.utils.clip_grad_norm_", wraps=torch.nn.utils.clip_grad_norm_) as clip:
        step(model, optimizer, params, batch, accumulation)
    assert update.call_count == clear.call_count == clip.call_count == 1
    assert model.calls == accumulation
    assert all(parameter.grad is None for parameter in params)
    for actual, expected in zip(params, ref_params):
        torch.testing.assert_close(actual, expected)


def test_grid_contains_all336_unique_cases():
    shards = [grid(shard) for shard in range(4)]
    names = [case_name(case) for shard in shards for case in shard]
    assert len(names) == len(set(names)) == 336
    assert [len(shard) for shard in shards] == [84, 84, 84, 84]
    assert {case["batch"] for shard in shards for case in shard} == {1, 2, 4, 8, 16, 32, 64, 128}


@pytest.mark.parametrize("accumulation", [0, 3])
def test_invalid_accumulation_rejected(accumulation):
    with pytest.raises(ValueError):
        backward_accumulated(ToyModel(), {"input_ids": torch.randn(8, 3)}, accumulation)
