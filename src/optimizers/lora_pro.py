import torch
from torch.optim import Optimizer


def _gram_inverses(A, B, eps_inv):
    """Keep the rank-sized solves in double precision, not the dense Adam state."""
    A64, B64 = A.double(), B.double()
    eye = torch.eye(A.shape[0], dtype=torch.float64, device=A.device)
    AAt = A64 @ A64.T + eps_inv * eye
    BtB = B64.T @ B64 + eps_inv * eye
    return A64, B64, AAt, BtB, torch.linalg.solve(AAt, eye), torch.linalg.solve(BtB, eye)


def _equivalent_gradient(A, B, G_A, G_B, inv_AAt, inv_BtB, scaling):
    """Reconstruct the equivalent weight gradient from both factor gradients."""
    left = inv_BtB @ G_A.double()
    right = (G_B.double() - B.double() @ (inv_BtB @ (B.double().T @ G_B.double()))) @ inv_AAt
    return (B @ left.to(B.dtype) + right.to(A.dtype) @ A) / scaling


def _solve_symmetric_sylvester(left, right, rhs):
    """Solve left @ X + X @ right = rhs for positive-definite rank matrices."""
    left_values, left_vectors = torch.linalg.eigh((left + left.T) * 0.5)
    right_values, right_vectors = torch.linalg.eigh((right + right.T) * 0.5)
    denominators = left_values[:, None] + right_values[None, :]
    if not torch.isfinite(denominators).all() or (denominators <= 0).any():
        raise FloatingPointError("LoRA-Pro Sylvester system is not positive definite")
    transformed = left_vectors.T @ rhs @ right_vectors
    return left_vectors @ (transformed / denominators) @ right_vectors.T


def _factor_direction(A, B, direction, AAt, BtB, inv_AAt, inv_BtB, scaling):
    """Authors' Sylvester-gauge factor mapping, with the configured ridge."""
    grad_A = (scaling * (B.T @ direction)).double()
    grad_B = (scaling * (direction @ A.T)).double()
    base_A = (inv_BtB @ grad_A) / scaling ** 2
    base_B = (grad_B - B.double() @ (inv_BtB @ (B.double().T @ grad_B))) @ inv_AAt
    base_B = base_B / scaling ** 2
    gauge = _solve_symmetric_sylvester(BtB, AAt, -base_A @ A.double().T)
    return -(base_A + gauge @ A.double()), -(base_B - B.double() @ gauge)


class LoRAPro(Optimizer):
    """LoRA-Pro optimizer from arXiv:2407.18242."""

    def __init__(
        self,
        params,
        lr: float = 2e-4,
        betas=(0.9, 0.999),
        eps_adam: float = 1e-6,
        eps_inv: float = 1e-6,
        weight_decay: float = 0.1,
        max_delta_norm: float = 1.0,
        log_ns_gap: bool = False,
        log_optimizer_stats: bool = True,
        lora_scaling: float = 1.0,
    ):
        if eps_inv <= 0 or eps_adam <= 0 or lora_scaling <= 0:
            raise ValueError("LoRA-Pro eps_inv, eps_adam and lora_scaling must be positive")
        defaults = dict(
            lr=lr,
            betas=betas,
            eps_adam=eps_adam,
            eps_inv=eps_inv,
            weight_decay=weight_decay,
            max_delta_norm=max_delta_norm,
            log_ns_gap=log_ns_gap,
            log_optimizer_stats=log_optimizer_stats,
            lora_scaling=lora_scaling,
        )
        self.last_logs = None
        self.last_diagnostics = None
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        step_logs = [] if any(
            group.get("log_optimizer_stats", True)
            for group in self.param_groups
        ) else None
        diagnostics = {"groups": 0, "zero_b_groups": 0, "zero_update_groups": 0,
                       "clipped_factors": 0, "nonfinite_groups": 0}
        update_norm_sq = 0.0
        equivalent_norm_sq = 0.0

        for group_idx, group in enumerate(self.param_groups):
            params = list(group["params"])
            if len(params) != 2:
                raise ValueError(
                    "Each param group for LoRAPro must contain exactly "
                    "two tensors: [A, B]."
                )

            A, B = params

            if A.grad is None or B.grad is None:
                continue

            if A.ndim != 2 or B.ndim != 2 or A.shape[0] != B.shape[1]:
                raise ValueError("LoRA-Pro expects A[r,in] and B[out,r]")

            A32 = A.float()
            B32 = B.float()
            G_A32 = A.grad.float()
            G_B32 = B.grad.float()
            if not all(torch.isfinite(value).all() for value in (A32, B32, G_A32, G_B32)):
                raise FloatingPointError("Nonfinite LoRA-Pro factor or input gradient")

            if group["weight_decay"] != 0:
                G_A32 = G_A32.add(A32, alpha=group["weight_decay"])
                G_B32 = G_B32.add(B32, alpha=group["weight_decay"])

            eps_inv = group["eps_inv"]
            eps_adam = group["eps_adam"]
            beta1, beta2 = group["betas"]
            scaling = group["lora_scaling"]

            _, _, AAt, BtB, inv_AAt, inv_BtB = _gram_inverses(A32, B32, eps_inv)
            G_W32 = _equivalent_gradient(A32, B32, G_A32, G_B32, inv_AAt, inv_BtB, scaling)
            if not torch.isfinite(G_W32).all():
                raise FloatingPointError("Nonfinite LoRA-Pro equivalent gradient")

            st = self.state[A]
            if "step" not in st:
                st["step"] = 0
                st["m1_W"] = torch.zeros_like(G_W32)
                st["m2_W"] = torch.zeros_like(G_W32)

            st["step"] += 1
            t = st["step"]

            m1 = st["m1_W"]
            m2 = st["m2_W"]

            m1.mul_(beta1).add_(G_W32, alpha=1.0 - beta1)
            m2.mul_(beta2).addcmul_(G_W32, G_W32, value=1.0 - beta2)
            if not torch.isfinite(m1).all() or not torch.isfinite(m2).all():
                diagnostics["nonfinite_groups"] += 1
                self.last_diagnostics = diagnostics.copy()
                raise FloatingPointError("Nonfinite LoRA-Pro Adam moment state")

            bias_corr1 = 1.0 - beta1 ** t
            bias_corr2 = 1.0 - beta2 ** t

            denom = (m2.sqrt() / (bias_corr2 ** 0.5)).add_(eps_adam)
            M_W = (group["lr"] / bias_corr1) * (m1 / denom)
            if not torch.isfinite(denom).all() or not torch.isfinite(M_W).all():
                diagnostics["nonfinite_groups"] += 1
                self.last_diagnostics = diagnostics.copy()
                raise FloatingPointError("Nonfinite LoRA-Pro Adam denominator or direction")

            dA32, dB32 = _factor_direction(
                A32, B32, M_W, AAt, BtB, inv_AAt, inv_BtB, scaling
            )
            if not torch.isfinite(dA32).all() or not torch.isfinite(dB32).all():
                raise FloatingPointError("Nonfinite LoRA-Pro factor update")

            for D in (dA32, dB32):
                D_norm = D.norm()
                if D_norm > group["max_delta_norm"]:
                    D.mul_(group["max_delta_norm"] / D_norm)
                    diagnostics["clipped_factors"] += 1

            diagnostics["groups"] += 1
            diagnostics["zero_b_groups"] += int(torch.count_nonzero(B32) == 0)
            norm_sq = (dA32.square().sum() + dB32.square().sum()).item()
            diagnostics["zero_update_groups"] += int(norm_sq == 0)
            update_norm_sq += norm_sq
            equivalent_norm_sq += torch.linalg.vector_norm(G_W32, dtype=torch.float64).item() ** 2

            A.add_(dA32.to(A.dtype))
            B.add_(dB32.to(B.dtype))

            if step_logs is not None and group.get("log_optimizer_stats", True):
                step_logs.append({
                    "group_idx": group_idx,
                    "A_F_norm": A32.norm().item(),
                    "B_F_norm": B32.norm().item(),
                    "G_A_F_norm": G_A32.norm().item(),
                    "G_B_F_norm": G_B32.norm().item(),
                    "dA_F_norm": dA32.norm().item(),
                    "dB_F_norm": dB32.norm().item(),
                    "G_W_F_norm": G_W32.norm().item(),
                    "M_W_F_norm": M_W.norm().item(),
                    "ns_gap_fro": None,
                })

        self.last_logs = step_logs
        diagnostics.update(update_norm=update_norm_sq ** 0.5,
                           equivalent_gradient_norm=equivalent_norm_sq ** 0.5)
        self.last_diagnostics = diagnostics
        return loss
