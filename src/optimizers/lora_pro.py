import torch
from torch.optim import Optimizer


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
    ):
        defaults = dict(
            lr=lr,
            betas=betas,
            eps_adam=eps_adam,
            eps_inv=eps_inv,
            weight_decay=weight_decay,
            max_delta_norm=max_delta_norm,
            log_ns_gap=log_ns_gap,
            log_optimizer_stats=log_optimizer_stats,
        )
        self.last_logs = None
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

            r = A.shape[0]
            dev = A.device

            A32 = A.float()
            B32 = B.float()
            G_A32 = A.grad.float()
            G_B32 = B.grad.float()

            if group["weight_decay"] != 0:
                G_A32 = G_A32.add(A32, alpha=group["weight_decay"])
                G_B32 = G_B32.add(B32, alpha=group["weight_decay"])

            eps_inv = group["eps_inv"]
            eps_adam = group["eps_adam"]
            beta1, beta2 = group["betas"]

            # Reconstruct G_W = B (B^T B + eps I)^(-1) G_A.
            BtB = B32.T @ B32 + eps_inv * torch.eye(r, device=dev, dtype=torch.float32)
            G_W32 = B32 @ torch.linalg.solve(BtB, G_A32)

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

            bias_corr1 = 1.0 - beta1 ** t
            bias_corr2 = 1.0 - beta2 ** t

            denom = (m2.sqrt() / (bias_corr2 ** 0.5)).add_(eps_adam)
            M_W = (group["lr"] / bias_corr1) * (m1 / denom)

            # Map the weight-space update back to the LoRA factors.
            dA32 = -torch.linalg.solve(BtB, B32.T @ M_W)

            AAt = A32 @ A32.T + eps_inv * torch.eye(r, device=dev, dtype=torch.float32)
            dB32 = -torch.linalg.solve(AAt, A32 @ M_W.T).T

            for D in (dA32, dB32):
                D_norm = D.norm()
                if D_norm > group["max_delta_norm"]:
                    D.mul_(group["max_delta_norm"] / D_norm)

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
        return loss
