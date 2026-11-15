"""Analytical regression oracles for the corrected LoRA-Pro update."""

import pytest
import torch

from optimizers.lora_pro import (
    LoRAPro,
    _equivalent_gradient,
    _factor_direction,
    _gram_inverses,
    _solve_symmetric_sylvester,
)


def _factors(seed=7):
    generator = torch.Generator().manual_seed(seed)
    A = torch.randn(2, 5, generator=generator)
    B = torch.randn(6, 2, generator=generator)
    G = torch.randn(6, 5, generator=generator)
    return A, B, G


@pytest.mark.parametrize("scaling", [0.5, 1.0, 2.0])
def test_equivalent_gradient_matches_dense_tangent_projection(scaling):
    A, B, G = _factors()
    _, _, _, _, inv_AAt, inv_BtB = _gram_inverses(A, B, 1e-10)
    GA, GB = scaling * B.T @ G, scaling * G @ A.T
    actual = _equivalent_gradient(A, B, GA, GB, inv_AAt, inv_BtB, scaling)
    A64, B64, G64 = A.double(), B.double(), G.double()
    projector_B = B64 @ torch.linalg.solve(B64.T @ B64, B64.T)
    projector_A = A64.T @ torch.linalg.solve(A64 @ A64.T, A64)
    expected = projector_B @ G64 + G64 @ projector_A - projector_B @ G64 @ projector_A
    torch.testing.assert_close(actual.double(), expected, atol=5e-7, rtol=3e-6)


def test_b_gradient_changes_step_when_a_gradient_is_zero():
    updates = []
    for sign in (-1, 1):
        A = torch.nn.Parameter(torch.eye(4)[:2].clone())
        B = torch.nn.Parameter(0.1 * torch.eye(4)[:, :2])
        G = torch.zeros(4, 4)
        G[2, 0] = sign
        A.grad = B.detach().T @ G
        B.grad = G @ A.detach().T
        before = B.detach().clone()
        optimizer = LoRAPro([{"params": [A, B]}], lr=5e-5, weight_decay=0)
        optimizer.step()
        updates.append(B.detach() - before)
    assert updates[0].norm() > 0
    torch.testing.assert_close(updates[0], -updates[1])


@pytest.mark.parametrize("weight_decay", [0.0, 0.1])
def test_zero_b_start_uses_nonzero_b_gradient(weight_decay):
    A = torch.nn.Parameter(torch.eye(4)[:2].clone())
    B = torch.nn.Parameter(torch.zeros(4, 2))
    target = torch.zeros(4, 4)
    target[2, 0] = 1
    before = 0.5 * (B @ A - target).square().sum()
    before.backward()
    assert A.grad.norm() == 0 and B.grad.norm() > 0
    optimizer = LoRAPro([{"params": [A, B]}], lr=5e-5, weight_decay=weight_decay)
    optimizer.step()
    after = 0.5 * (B @ A - target).square().sum()
    assert B.norm() > 0 and after < before
    assert optimizer.last_diagnostics["zero_b_groups"] == 1
    assert optimizer.last_diagnostics["zero_update_groups"] == 0


def test_sylvester_matches_kronecker_linear_system():
    generator = torch.Generator().manual_seed(11)
    left = torch.randn(3, 3, dtype=torch.float64, generator=generator)
    left = left @ left.T + 0.1 * torch.eye(3, dtype=torch.float64)
    right = torch.randn(3, 3, dtype=torch.float64, generator=generator)
    right = right @ right.T + 0.1 * torch.eye(3, dtype=torch.float64)
    rhs = torch.randn(3, 3, dtype=torch.float64, generator=generator)
    actual = _solve_symmetric_sylvester(left, right, rhs)
    eye = torch.eye(3, dtype=torch.float64)
    # Row-major vec(left X + X right) = (left kron I + I kron right.T) vec(X).
    system = torch.kron(left.contiguous(), eye) + torch.kron(eye, right.T.contiguous())
    expected = torch.linalg.solve(system, rhs.reshape(-1)).reshape(3, 3)
    torch.testing.assert_close(actual, expected, rtol=1e-11, atol=1e-11)
    torch.testing.assert_close(left @ actual + actual @ right, rhs, rtol=1e-11, atol=1e-11)


@pytest.mark.parametrize("scaling", [0.5, 1.0, 2.0])
def test_factor_mapping_preserves_projected_weight_direction(scaling):
    A, B, G = _factors()
    _, _, AAt, BtB, inv_AAt, inv_BtB = _gram_inverses(A, B, 1e-10)
    dA, dB = _factor_direction(A, B, G, AAt, BtB, inv_AAt, inv_BtB, scaling)
    actual = scaling * (B.double() @ dA + dB @ A.double())
    expected = -_equivalent_gradient(
        A, B, B.T @ G, G @ A.T, inv_AAt, inv_BtB, 1.0
    )
    torch.testing.assert_close(actual, expected.double(), rtol=3e-6, atol=5e-7)
    # The Sylvester choice is the minimum-factor-norm representative, at negligible ridge.
    torch.testing.assert_close(dA @ A.double().T, B.double().T @ dB, rtol=3e-6, atol=5e-7)


def test_rank_deficient_factors_remain_finite_and_bounded():
    torch.manual_seed(5)
    A = torch.nn.Parameter(torch.ones(2, 5))
    B = torch.nn.Parameter(torch.ones(6, 2))
    A.grad = torch.randn_like(A)
    B.grad = torch.randn_like(B)
    before_A, before_B = A.detach().clone(), B.detach().clone()
    optimizer = LoRAPro([{"params": [A, B]}], lr=1.0, max_delta_norm=0.1)
    optimizer.step()
    assert torch.isfinite(A).all() and torch.isfinite(B).all()
    assert (A - before_A).norm() <= 0.100001
    assert (B - before_B).norm() <= 0.100001
    assert optimizer.last_diagnostics["clipped_factors"] > 0


def test_nonfinite_gradient_is_not_silently_zeroed():
    A, B, _ = _factors()
    A, B = torch.nn.Parameter(A), torch.nn.Parameter(B)
    A.grad = torch.full_like(A, float("nan"))
    B.grad = torch.ones_like(B)
    optimizer = LoRAPro([{"params": [A, B]}])
    with pytest.raises(FloatingPointError, match="Nonfinite"):
        optimizer.step()


def test_finite_gradient_overflow_of_fp32_adam_state_is_not_silently_zeroed():
    A = torch.nn.Parameter(torch.eye(4)[:2].clone())
    B = torch.nn.Parameter(0.1 * torch.eye(4)[:, :2])
    A.grad = torch.full_like(A, 1e30)
    B.grad = torch.full_like(B, 1e30)
    before_A, before_B = A.detach().clone(), B.detach().clone()
    optimizer = LoRAPro([{"params": [A, B]}], lr=5e-5, weight_decay=0.1)
    with pytest.raises(FloatingPointError, match="Adam moment"):
        optimizer.step()
    torch.testing.assert_close(A, before_A)
    torch.testing.assert_close(B, before_B)
    assert optimizer.last_diagnostics["nonfinite_groups"] == 1


def test_low_rank_regression_learns_from_zero_b():
    torch.manual_seed(17)
    A = torch.nn.Parameter(torch.randn(2, 5))
    B = torch.nn.Parameter(torch.zeros(6, 2))
    target = torch.randn(6, 2) @ torch.randn(2, 5)
    optimizer = LoRAPro([{"params": [A, B]}], lr=0.02, weight_decay=0,
                        max_delta_norm=1.0, log_optimizer_stats=False)
    start = (B @ A - target).square().mean().item()
    for _ in range(150):
        optimizer.zero_grad()
        loss = (B @ A - target).square().mean()
        loss.backward()
        optimizer.step()
    assert loss.item() < start * 0.1
    assert optimizer.state[A]["m1_W"].dtype == torch.float32
    assert optimizer.state[A]["m2_W"].dtype == torch.float32
