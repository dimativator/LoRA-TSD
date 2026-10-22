import pytest
import torch

from optimizers.retractions import RETRACTIONS, retract_lora, retract_candidate
from optimizers.lora_tsd import LoRATSD as LoRANormSpectralBall
from optimizers.riemannian_lora import FixedRank, Riemannion


def _inputs(dtype=torch.float64):
    torch.manual_seed(42)
    B = torch.linalg.qr(torch.randn(12, 3, dtype=dtype)).Q
    A = torch.randn(3, 10, dtype=dtype)
    # Canonical lift of a generic tangent vector, including both subspaces.
    Xi = torch.randn(12, 3, dtype=dtype) @ A + B @ torch.randn(3, 10, dtype=dtype)
    dB = Xi @ torch.linalg.pinv(A)
    dA = B.T @ (Xi - dB @ A)
    return A, B, dA, dB

@pytest.mark.parametrize("method", RETRACTIONS)
def test_centering_rank_and_first_derivative(method):
    A, B, dA, dB = _inputs()
    W = B @ A
    zero_A, zero_B, _ = retract_lora(A, B, 0 * dA, 0 * dB, method)
    torch.testing.assert_close(zero_B @ zero_A, W)
    Xi = dB @ A + B @ dA
    errors = []
    for h in (1e-3, 5e-4):
        new_A, new_B, fallback = retract_lora(A, B, h * dA, h * dB, method)
        assert not fallback
        assert torch.linalg.matrix_rank(new_B @ new_A) == 3
        errors.append(((new_B @ new_A - W) / h - Xi).norm())
    assert errors[1] < 0.6 * errors[0]

@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_svd_matches_dense_linear_step_and_excludes_cross_term(dtype):
    A, B, dA, dB = _inputs(dtype)
    dA, dB = 0.1 * dA, 0.1 * dB
    Y = B @ A + dB @ A + B @ dA
    U, S, Vt = torch.linalg.svd(Y, full_matrices=False)
    expected = (U[:, :3] * S[:3]) @ Vt[:3]
    new_A, new_B, _ = retract_lora(A, B, dA, dB, "svd")
    torch.testing.assert_close(new_B @ new_A, expected, rtol=1e-5, atol=1e-6)
    assert (new_B @ new_A - (B + dB) @ (A + dA)).norm() > 1e-3

def test_orthographic_correction_is_normal_at_start():
    A, B, dA, dB = _inputs()
    new_A, new_B, _ = retract_lora(A, B, 0.01 * dA, 0.01 * dB, "orthographic")
    correction = new_B @ new_A - (B @ A + 0.01 * (dB @ A + B @ dA))
    V = torch.linalg.qr(A.T).Q
    torch.testing.assert_close(B.T @ correction, torch.zeros(3, 10, dtype=A.dtype), atol=1e-12, rtol=0)
    torch.testing.assert_close(correction @ V, torch.zeros(12, 3, dtype=A.dtype), atol=1e-12, rtol=0)

@pytest.mark.parametrize("method", RETRACTIONS)
def test_common_gauge_preserves_product(method):
    A, B, dA, dB = _inputs()
    native_A, native_B, _ = retract_lora(A, B, .01 * dA, .01 * dB, method)
    new_A, new_B, _ = retract_lora(A, B, .01 * dA, .01 * dB, method, "left_orthogonal")
    torch.testing.assert_close(new_B @ new_A, native_B @ native_A)
    torch.testing.assert_close(new_B.T @ new_B, torch.eye(3, dtype=A.dtype))

@pytest.mark.parametrize("method", RETRACTIONS)
def test_zero_initialization_has_common_product(method):
    A, B, dA, dB = _inputs()
    B.zero_()
    dA.zero_()
    new_A, new_B, _ = retract_lora(A, B, dA, .01 * dB, method)
    torch.testing.assert_close(new_B @ new_A, .01 * dB @ A)

@pytest.mark.parametrize("method", RETRACTIONS)
def test_spectral_optimizer_step_matches_selected_map(method):
    A, B, _, _ = _inputs(torch.float32)
    # Capture identical factor increments from the unmodified direction path.
    pairs = [(torch.nn.Parameter(A.clone()), torch.nn.Parameter(B.clone())) for _ in range(2)]
    ambient_grad = torch.randn(12, 10)
    for a, b in pairs:
        a.grad = b.detach().T @ ambient_grad
        b.grad = ambient_grad @ a.detach().T
    ref = LoRANormSpectralBall([{"params": list(pairs[0])}], use_momentum=False,
                              lr_A=.01, ball_iters=1, max_delta_norm=.5, log_optimizer_stats=False)
    opt = LoRANormSpectralBall([{"params": list(pairs[1])}], use_momentum=False,
                              lr_A=.01, ball_iters=1, max_delta_norm=.5,
                              retraction=method, retraction_gauge="left_orthogonal",
                              log_optimizer_stats=False)
    ref.step()
    dA = pairs[0][0].detach() - A
    dB = pairs[0][1].detach() - B
    expected_A, expected_B, _ = retract_lora(A, B, dA, dB, method, "left_orthogonal")
    opt.step()
    torch.testing.assert_close(pairs[1][1] @ pairs[1][0], expected_B @ expected_A, atol=3e-6, rtol=1e-5)

@pytest.mark.parametrize("method", RETRACTIONS)
def test_riemannion_step_preserves_storage_and_orthogonality(method):
    A, B, _, _ = _inputs(torch.float32)
    a = torch.nn.Parameter(torch.cat([B.T, torch.zeros_like(B.T)], dim=0))
    b = torch.nn.Parameter(torch.cat([A.T, torch.linalg.qr(A.T).Q], dim=1))
    opt = Riemannion([{"params": [a, b], "param_names": ["lora_A.weight", "lora_B.weight"]}],
                     lr=1e-4, momentum=0, retraction=method)
    for _ in range(3):
        a.grad = torch.randn_like(a)
        b.grad = torch.randn_like(b)
        opt.step()
        left = a.T[:, :3]
        torch.testing.assert_close(left.T @ left, torch.eye(3), atol=1e-5, rtol=1e-5)
        assert torch.count_nonzero(a[3:]) == 0
        assert torch.isfinite(b).all()

def test_refactorization_rejects_untransported_spectral_momentum():
    A, B, _, _ = _inputs()
    with pytest.raises(ValueError, match="momentum"):
        LoRANormSpectralBall([{"params": [A, B]}], retraction="svd")

@pytest.mark.parametrize("method", RETRACTIONS)
def test_riemannion_step_uses_same_tangent_for_each_map(method):
    A, B, _, _ = _inputs(torch.float32)
    a = torch.nn.Parameter(torch.cat([B.T, torch.zeros_like(B.T)], dim=0))
    b = torch.nn.Parameter(torch.cat([A.T, torch.linalg.qr(A.T).Q], dim=1))
    opt = Riemannion([{"params": [a, b], "param_names": ["lora_A.weight", "lora_B.weight"]}],
                     lr=1e-4, momentum=0, retraction=method)
    a.grad, b.grad = torch.randn_like(a), torch.randn_like(b)
    manifold = FixedRank(3)
    point, grad = opt.get_deltas(a, b, manifold)
    dot_left, dot_right = opt.apply_muon(point, grad, manifold)
    lr = opt.adjust_lr_for_muon(1e-4, b.shape[0], a.shape[1], 3)
    left, right = manifold.embedding(point, [-lr * dot_left, A - lr * dot_right], ambient=False)
    expected_A, expected_B, _ = retract_candidate(A, B, left, right, method)
    opt.step()
    torch.testing.assert_close(a.T[:, :3] @ b.T[:3], expected_B @ expected_A,
                               atol=2e-6, rtol=1e-5)

@pytest.mark.parametrize("family", ["tsd", "riemannion"])
@pytest.mark.parametrize("method", RETRACTIONS)
def test_short_matrix_fit_reduces_loss(family, method):
    A, B, _, _ = _inputs(torch.float32)
    target = B @ A + torch.randn(12, 3) @ torch.randn(3, 10)
    if family == "tsd":
        a, b = torch.nn.Parameter(A.clone()), torch.nn.Parameter(B.clone())
        opt = LoRANormSpectralBall([{"params": [a, b]}], use_momentum=False,
                                  lr_A=.02, ball_iters=1, max_delta_norm=.5,
                                  retraction=method, retraction_gauge="left_orthogonal",
                                  log_optimizer_stats=False)
        product = lambda: b @ a
    else:
        a = torch.nn.Parameter(torch.cat([B.T, torch.zeros_like(B.T)]))
        b = torch.nn.Parameter(torch.cat([A.T, torch.linalg.qr(A.T).Q], dim=1))
        opt = Riemannion([{"params": [a, b], "param_names": ["lora_A.weight", "lora_B.weight"]}],
                         lr=.001, momentum=0, retraction=method)
        product = lambda: a.T[:, :3] @ b.T[:3]
    initial = ((product() - target) ** 2).mean().item()
    for _ in range(10):
        opt.zero_grad()
        loss = ((product() - target) ** 2).mean()
        loss.backward()
        opt.step()
    final = ((product() - target) ** 2).mean()
    assert torch.isfinite(final)
    assert final.item() < initial

@pytest.mark.parametrize("method", RETRACTIONS)
def test_projected_momentum_is_gauge_independent(method):
    A, B, _, _ = _inputs(torch.float32)
    Q = torch.linalg.qr(torch.randn(3, 3)).Q
    pairs = [(torch.nn.Parameter(A.clone()), torch.nn.Parameter(B.clone())),
             (torch.nn.Parameter(Q.T @ A), torch.nn.Parameter(B @ Q))]
    opts = [LoRANormSpectralBall([{"params": [a, b]}], use_momentum=True, momentum_cf=.95,
                               retraction=method, retraction_gauge="left_orthogonal",
                               retraction_momentum="projected", lr_A=.01, ball_iters=1,
                               log_optimizer_stats=False) for a, b in pairs]
    for _ in range(3):
        ambient_gradient = torch.randn(12, 10)
        for opt, (a, b) in zip(opts, pairs):
            a.grad = b.detach().T @ ambient_gradient
            b.grad = ambient_gradient @ a.detach().T
            opt.step()
        torch.testing.assert_close(pairs[0][1] @ pairs[0][0], pairs[1][1] @ pairs[1][0],
                                   atol=1e-5, rtol=1e-5)
        L0, R0 = opts[0].state[pairs[0][0]]["tangent_momentum"]
        L1, R1 = opts[1].state[pairs[1][0]]["tangent_momentum"]
        torch.testing.assert_close(L0 @ R0, L1 @ R1, atol=1e-5, rtol=1e-5)

def test_riemannion_transport_keeps_previous_point_snapshot():
    A, B, _, _ = _inputs(torch.float32)
    a = torch.nn.Parameter(torch.cat([B.T, torch.zeros_like(B.T)]))
    b = torch.nn.Parameter(torch.cat([A.T, torch.linalg.qr(A.T).Q], dim=1))
    opt = Riemannion([{"params": [a, b], "param_names": ["lora_A.weight", "lora_B.weight"]}],
                     momentum=.95, retraction="factor", transport_snapshot=True)
    a.grad, b.grad = torch.randn_like(a), torch.randn_like(b)
    opt.step()
    saved_left, saved_right = opt.state[a]["prev_point"]
    torch.testing.assert_close(saved_left, B)
    torch.testing.assert_close(saved_right, A)

def test_riemannion_zero_singular_modes_create_left_motion():
    A, B, _, _ = _inputs()
    manifold = FixedRank(3)
    grad = torch.randn_like(A)
    left, right = manifold.embedding([B, A], [torch.zeros_like(B), grad], ambient=False)
    Q_l, R_l = torch.linalg.qr(left)
    Q_r, R_r = torch.linalg.qr(right.T)
    U, S, Vt = torch.linalg.svd(R_l @ R_r.T, full_matrices=False)
    V = torch.linalg.qr(A.T).Q
    full = (Q_l @ U) @ (Vt @ Q_r.T)
    supported = (Q_l @ U[:, :3]) @ (Vt[:3] @ Q_r.T)
    full_motion = (full - B @ (B.T @ full)) @ V
    supported_motion = (supported - B @ (B.T @ supported)) @ V
    assert S[3:].abs().max() < 1e-12
    assert full_motion.norm() > 0.1
    assert supported_motion.norm() < 1e-12
