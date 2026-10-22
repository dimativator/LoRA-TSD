import pytest
import torch

import importlib.util
from pathlib import Path

_spec = importlib.util.spec_from_file_location("riemannian_layer", Path(__file__).resolve().parents[1] / "src/models/riemannian_lora_layer.py")
_layer_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_layer_module)
RiemannianLoRALinear = _layer_module.RiemannianLoRALinear
from optimizers.riemannian_lora import FixedRank
from optimizers.riemannion_original import RiemannionOriginal
from optimizers.retractions import RETRACTIONS


def _layer():
    torch.manual_seed(7)
    layer = RiemannianLoRALinear(torch.nn.Linear(9, 11, bias=False), 2, 2)
    with torch.no_grad():
        C = torch.randn(2, 9, dtype=torch.float32)
        layer.lora_B.weight[:, :2].copy_(C.T)
        layer.lora_B.weight[:, 2:].copy_(torch.linalg.qr(C.T).Q)
    return layer


def _optimizer(layer, **kwargs):
    return RiemannionOriginal([{
        'params': [layer.lora_A.weight, layer.lora_B.weight],
        'param_names': ['lora_A.weight', 'lora_B.weight'],
    }], lr=1e-3, **kwargs)


def test_auxiliary_forward_preserves_value_and_recovers_full_tangent_gradient():
    layer = _layer()
    x = torch.randn(4, 9, dtype=torch.float32)
    expected = layer(x)
    layer.tangent_forward = True
    actual = layer(x)
    torch.testing.assert_close(actual, expected)
    upstream_grad = torch.randn_like(actual)
    actual.backward(upstream_grad)
    optimizer = _optimizer(layer)
    manifold = FixedRank(2)
    point, vec = optimizer.get_deltas(layer.lora_A.weight, layer.lora_B.weight, manifold)
    U, C = point
    V = torch.linalg.qr(C.T).Q
    # The layer explicitly computes adapters in float32.
    G = (upstream_grad.float().T @ x.float())
    expected = U @ (U.T @ G) + (G @ V - U @ (U.T @ G @ V)) @ V.T
    torch.testing.assert_close(manifold.embedding(point, vec), expected,
                               rtol=1e-5, atol=1e-6)
    assert vec[0].norm() > 0


@pytest.mark.parametrize('retraction', RETRACTIONS)
def test_original_supports_retractions_with_momentum(retraction):
    layer = _layer()
    layer.tangent_forward = True
    optimizer = _optimizer(layer, retraction=retraction, momentum=0.9)
    for _ in range(2):
        optimizer.zero_grad()
        layer(torch.randn(4, 9, dtype=torch.float32)).square().sum().backward()
        optimizer.step()
        assert torch.isfinite(layer.lora_A.weight).all()
        assert torch.isfinite(layer.lora_B.weight).all()
        torch.testing.assert_close(layer.lora_A.weight[2:],
                                   torch.zeros_like(layer.lora_A.weight[2:]))


def test_default_is_svd_and_nesterov_matches_authors_heavy_ball():
    layer = _layer()
    optimizer = _optimizer(layer)
    assert optimizer.param_groups[0]['retraction'] == 'svd'
    U = layer.lora_A.weight.T[:, :2]
    C = layer.lora_B.weight.T[:2]
    point = [U, C]
    vec = [torch.randn_like(U), torch.randn_like(C)]
    old_vec = [torch.randn_like(U), torch.randn_like(C)]
    manifold = FixedRank(2)
    state = {'prev_point': point, 'momentum': old_vec}
    transported = manifold.transport(point, point, old_vec)
    result = optimizer.apply_momentum(point, vec, state, manifold, 0.9, True)
    for actual, previous, gradient in zip(result, transported, vec):
        torch.testing.assert_close(actual, 0.9 * previous + gradient)
    assert state['prev_point'] is point
