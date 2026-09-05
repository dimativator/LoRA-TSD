import torch
from torch.optim import Optimizer

from .utils import zeropower_via_newtonschulz5


@torch.no_grad()
def inv_sqrt_via_newtonschulz(P, steps=7, eps=1e-5, exact=False):
    """Compute a damped inverse square root of a small SPD matrix."""
    assert P.shape[0] == P.shape[1]
    r = P.shape[0]
    I = torch.eye(r, dtype=P.dtype, device=P.device)

    P = 0.5 * (P + P.T) + eps * I

    if exact:
        evals, evecs = torch.linalg.eigh(P)
        evals = evals.clamp_min(eps)
        return (evecs * evals.rsqrt()) @ evecs.T

    t = P.norm()
    Y = P / t
    Z = I.clone()
    for _ in range(steps):
        M = 0.5 * (3.0 * I - Z @ Y)
        Y = Y @ M
        Z = M @ Z

    return Z / t.sqrt()


class LoRAMuonManifold(Optimizer):
    """LoRA-Muon optimizer from arXiv:2606.12921."""

    def __init__(
        self,
        params,
        lr: float = 1e-2,
        momentum: float = 0.95,
        weight_decay: float = 0.0,
        ns_steps: int = 8,
        inv_sqrt_steps: int = 7,
        eps: float = 1e-5,
        exact_inv_sqrt: bool = False,
        max_delta_norm: float = 0,
        log_optimizer_stats: bool = True,
    ):
        defaults = dict(
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
            ns_steps=ns_steps,
            inv_sqrt_steps=inv_sqrt_steps,
            eps=eps,
            exact_inv_sqrt=exact_inv_sqrt,
            max_delta_norm=max_delta_norm,
            log_optimizer_stats=log_optimizer_stats,
        )
        self.last_logs = None

        super().__init__(params, defaults)

    def _momentum(self, param, grad, beta):
        state = self.state[param]
        buf = state.get("momentum_buffer", None)
        if buf is None:
            buf = torch.zeros_like(grad)
        buf.mul_(beta).add_(grad, alpha=1.0 - beta)
        state["momentum_buffer"] = buf
        return buf

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        step_logs = [] if any(
            group.get("log_optimizer_stats", True) for group in self.param_groups
        ) else None

        for group_idx, group in enumerate(self.param_groups):

            params = list(group["params"])
            if len(params) != 2:
                raise ValueError(
                    "Each param group for LoRAMuonManifold must contain exactly "
                    "two tensors: [A, B]."
                )

            A, B = params

            if A.grad is None or B.grad is None:
                continue

            lr = group["lr"]
            eps = group["eps"]
            inv_sqrt_steps = group["inv_sqrt_steps"]
            exact = group["exact_inv_sqrt"]

            A32 = A.float()
            B32 = B.float()

            M_A = self._momentum(A, A.grad.float(), group["momentum"])
            M_B = self._momentum(B, B.grad.float(), group["momentum"])

            S_A = A32 @ A32.T
            S_B = B32.T @ B32

            R_A = inv_sqrt_via_newtonschulz(S_A, inv_sqrt_steps, eps, exact)
            R_B = inv_sqrt_via_newtonschulz(S_B, inv_sqrt_steps, eps, exact)

            dA32 = -0.5 * lr * (
                R_B @ zeropower_via_newtonschulz5(R_B @ M_A, group["ns_steps"]).float()
            )
            dB32 = -0.5 * lr * (
                zeropower_via_newtonschulz5(M_B @ R_A, group["ns_steps"]).float() @ R_A
            )

            max_delta_norm = group["max_delta_norm"]
            if max_delta_norm > 0:
                for D in (dA32, dB32):
                    D_norm = D.norm()
                    if D_norm > max_delta_norm:
                        D.mul_(max_delta_norm / D_norm)

            wd = group["weight_decay"]
            s = (1.0 - wd * lr) ** 0.5 if wd > 0 else 1.0
            if wd > 0 and not s > 0:
                raise ValueError(
                    f"weight_decay * lr must be < 1 for the split rule "
                    f"(got {wd} * {lr} = {wd * lr})."
                )

            if s == 1.0:
                A.add_(dA32.to(A.dtype))
                B.add_(dB32.to(B.dtype))
            else:
                A.mul_(s).add_((dA32 / s).to(A.dtype))
                B.mul_(s).add_((dB32 / s).to(B.dtype))

            if step_logs is not None and group.get("log_optimizer_stats", True):
                step_logs.append(
                    {
                        "group_idx": group_idx,
                        "A_F_norm": A32.norm().item(),
                        "B_F_norm": B32.norm().item(),
                        "G_A_F_norm": M_A.norm().item(),
                        "G_B_F_norm": M_B.norm().item(),
                        "dA_F_norm": dA32.norm().item(),
                        "dB_F_norm": dB32.norm().item(),
                        "R_A_2_norm": torch.linalg.matrix_norm(R_A, ord=2).item(),
                        "R_B_2_norm": torch.linalg.matrix_norm(R_B, ord=2).item(),
                    }
                )

        self.last_logs = step_logs
        return loss
