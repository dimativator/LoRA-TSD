"""Numerical contracts for the corrected LoRA-Rite preconditioner."""

import importlib.util
from pathlib import Path

import pytest
import torch


_spec = importlib.util.spec_from_file_location(
    "lora_rite_repair", Path(__file__).resolve().parents[1] / "src/optimizers/lora_rite.py"
)
_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_module)
LoRARite = _module.LoRARite
Helper = _module._LoraRiteHelper


def _apply(helper, gradient, covariance, epsilon=1e-6, escape=0, root=None):
    return helper.get_preconditioned_update(
        gradient, covariance, escape, epsilon, epsilon**2, False, escape != 0,
        covariance_root=root,
    )


def _svd_whitened_gradient(gradient, epsilon=1e-6):
    u, singular, vh = torch.linalg.svd(gradient.double(), full_matrices=False)
    coefficients = singular / (singular / gradient.shape[0]**0.5 + epsilon)
    return (u * coefficients) @ vh


@pytest.mark.parametrize("seed", range(10))
def test_rank_one_gradient_matches_svd_oracle_without_zeroing(seed):
    torch.manual_seed(seed)
    gradient = torch.randn(64, 1) @ torch.randn(1, 16)
    helper = Helper()
    covariance = helper.compute_second_moments(gradient)
    assert covariance.dtype == torch.float64
    root = helper.update_second_root(0, gradient, torch.zeros_like(covariance),
                                     torch.zeros_like(covariance), 0.999)
    actual = _apply(helper, gradient, covariance, root=root)
    expected = _svd_whitened_gradient(gradient)
    torch.testing.assert_close(actual.double(), expected, rtol=2e-6, atol=2e-7)
    assert torch.isfinite(actual).all()
    assert actual.norm() > 0


def test_square_root_state_matches_covariance_moment_recurrence():
    torch.manual_seed(9)
    helper = Helper()
    root = torch.zeros(4, 4, dtype=torch.float64)
    covariance = torch.zeros_like(root)
    for step in range(5):
        gradient = torch.randn(12, 4)
        rotation = torch.randn(4, 4)
        decay = helper.bias_corrected_decay(step, 0.999)
        expected = (decay * rotation.double() @ covariance @ rotation.double().T
                    + (1 - decay) * gradient.double().T @ gradient.double() / 12)
        root = helper.update_second_root(step, gradient, root, rotation, 0.999)
        covariance = root.T @ root
        torch.testing.assert_close(covariance, expected, rtol=1e-12, atol=1e-12)


def test_fp32_roundoff_covariance_does_not_generate_nan():
    torch.manual_seed(0)
    gradient = torch.randn(64, 1) @ torch.randn(1, 16)
    rounded = (gradient.T @ gradient) / gradient.shape[0]
    helper = Helper()
    actual = _apply(helper, gradient, rounded)
    assert torch.isfinite(actual).all()
    assert actual.norm() > 0
    assert helper.diagnostics["roundoff_eigenvalues_clamped"] > 0


def test_full_rank_matches_high_precision_inverse_action():
    torch.manual_seed(4)
    gradient = torch.randn(32, 6, dtype=torch.float64)
    factor = torch.randn(9, 6, dtype=torch.float64)
    covariance = factor.T @ factor + torch.eye(6, dtype=torch.float64)
    helper = Helper()
    values, vectors = torch.linalg.eigh(covariance)
    expected = gradient @ ((vectors / (values.sqrt() + 1e-6)) @ vectors.T)
    torch.testing.assert_close(_apply(helper, gradient, covariance), expected,
                               rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("epsilon", [0.0, 1e-6])
def test_zero_covariance_zero_gradient_stays_zero(epsilon):
    helper = Helper()
    actual = _apply(helper, torch.zeros(5, 3), torch.zeros(3, 3), epsilon)
    assert torch.isfinite(actual).all()
    assert torch.count_nonzero(actual) == 0


def test_escape_mass_is_applied_without_adding_an_extra_ridge():
    covariance = torch.diag(torch.tensor([0.0, 2.0], dtype=torch.float64))
    gradient = torch.ones(3, 2, dtype=torch.float64)
    actual = _apply(Helper(), gradient, covariance, escape=0.25)
    expected = gradient / (torch.tensor([0.25, 2.25]).sqrt() + 1e-6)
    torch.testing.assert_close(actual, expected.double(), rtol=2e-7, atol=1e-12)


@pytest.mark.parametrize("matrix", [
    torch.diag(torch.tensor([1.0, -0.1])),
    torch.tensor([[float("nan"), 0.0], [0.0, 1.0]]),
    torch.tensor([[float("inf"), 0.0], [0.0, 1.0]]),
])
def test_invalid_covariance_raises_instead_of_silently_zeroing(matrix):
    with pytest.raises(FloatingPointError):
        _apply(Helper(), torch.ones(4, 2), matrix)


def test_invalid_gradient_and_negative_escape_raise():
    with pytest.raises(FloatingPointError):
        _apply(Helper(), torch.full((3, 2), float("nan")), torch.eye(2))
    with pytest.raises(FloatingPointError):
        _apply(Helper(), torch.ones(3, 2), torch.eye(2), escape=-0.1)


def test_relative_epsilon_has_defined_scale_and_no_undefined_variable():
    helper = Helper()
    covariance = torch.diag(torch.tensor([1.0, 4.0], dtype=torch.float64))
    actual = helper.inverse_sqrt(covariance, 0, 0.1, 0.01, relative_epsilon=True)
    expected = torch.diag(1 / (torch.tensor([1.0, 2.0], dtype=torch.float64) + 0.2))
    torch.testing.assert_close(actual, expected)


def test_covariance_transport_preserves_double_precision_and_psd():
    torch.manual_seed(6)
    helper = Helper()
    gradient = torch.randn(16, 4)
    covariance = helper.compute_second_moments(gradient)
    rotation = torch.randn(4, 4)
    actual = helper.transform_second_moment_to_new_basis(covariance, rotation)
    expected = rotation.double() @ covariance @ rotation.double().T
    assert actual.dtype == torch.float64
    torch.testing.assert_close(actual, expected)
    assert torch.linalg.eigvalsh(actual).min() > -1e-12


@pytest.mark.parametrize("seed", range(5))
def test_real_optimizer_moves_from_zero_b_with_rank_one_gradient(seed):
    torch.manual_seed(seed)
    a = torch.nn.Parameter(torch.linalg.qr(torch.randn(64, 16)).Q.T.contiguous())
    b = torch.nn.Parameter(torch.zeros(32, 16))
    optimizer = LoRARite([{"params": [a, b]}], lr=1e-4, betas=(0.9, 0.999))
    for _ in range(3):
        a.grad = torch.zeros_like(a)
        b.grad = 0.01 * torch.randn(32, 1) @ torch.randn(1, 16)
        optimizer.step()
        assert torch.isfinite(a).all() and torch.isfinite(b).all()
        assert b.norm() > 0
        assert optimizer.state[a]["attr"].v_l.dtype == torch.float64
        assert optimizer.state[a]["attr"].v_r.dtype == torch.float64
        assert all(torch.isfinite(torch.tensor(v)) for v in optimizer.last_diagnostics.values())


def test_invalid_optimizer_gradient_is_not_masked():
    a = torch.nn.Parameter(torch.eye(3))
    b = torch.nn.Parameter(torch.eye(3))
    optimizer = LoRARite([{"params": [a, b]}], lr=1e-4, betas=(0.9, 0.999))
    a.grad = torch.full_like(a, float("nan"))
    b.grad = torch.ones_like(b)
    with pytest.raises(FloatingPointError):
        optimizer.step()


def test_corrected_optimizer_learns_a_small_linear_problem():
    torch.manual_seed(12)
    a = torch.nn.Parameter(torch.linalg.qr(torch.randn(12, 3)).Q.T.contiguous())
    b = torch.nn.Parameter(torch.zeros(8, 3))
    inputs = 0.1 * torch.randn(32, 12)
    targets = inputs @ (0.1 * torch.randn(12, 3) @ torch.randn(3, 8))
    optimizer = LoRARite([{"params": [a, b]}], lr=0.01, betas=(0.9, 0.999))
    losses = []
    for _ in range(80):
        optimizer.zero_grad()
        loss = (inputs @ a.T @ b.T - targets).square().mean()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    assert losses[-1] < 0.01 * losses[0]


def test_preconditioner_action_is_equivariant_under_rank_rotation():
    torch.manual_seed(8)
    gradient = torch.randn(12, 4, dtype=torch.float64)
    covariance = gradient.T @ gradient / gradient.shape[0]
    rotation = torch.linalg.qr(torch.randn(4, 4, dtype=torch.float64)).Q
    expected = _apply(Helper(), gradient, covariance) @ rotation
    actual = _apply(Helper(), gradient @ rotation, rotation.T @ covariance @ rotation)
    torch.testing.assert_close(actual, expected, rtol=1e-11, atol=1e-11)
