import torch
from torch.optim import Optimizer

from .utils import zeropower_via_newtonschulz5


class LoRAAdam(Optimizer):
    def __init__(
        self,
        params,
        lr_A: float = 1e-3,
        lr_B: float = 1e-4,
        betas=(0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        max_delta_norm: float = 1,
        log_ns_gap: bool = False,
        log_optimizer_stats: bool = True,
    ):
        defaults = dict(
            lr_A=lr_A,
            lr_B=lr_B,
            betas=betas,
            eps=eps,
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
            or group.get("log_ns_gap", False)
            for group in self.param_groups
        ) else None

        for group_idx, group in enumerate(self.param_groups):
            params = list(group["params"])
            if len(params) != 2:
                raise ValueError(
                    "Each param group for LoRAAdam must contain exactly "
                    "two tensors: [A, B]."
                )

            A, B = params

            if A.grad is None or B.grad is None:
                continue

            A32 = A.float()
            B32 = B.float()
            G_A32 = A.grad.float()
            G_B32 = B.grad.float()

            if group["weight_decay"] != 0:
                G_A32 = G_A32.add(A32, alpha=group["weight_decay"])
                G_B32 = G_B32.add(B32, alpha=group["weight_decay"])

            G_W32 = None
            if group.get("log_ns_gap", False):
                W = group.get("W", None)
            else:
                W = None
            if W is not None:
                if getattr(W, "grad", None) is not None:
                    G_W32 = W.grad.float()
                else:
                    G_W32 = None

            beta1, beta2 = group["betas"]
            eps = group["eps"]

            stA = self.state[A]
            stB = self.state[B]
            if "step" not in stA:
                stA["step"] = 0
                stA["exp_avg"] = torch.zeros_like(G_A32)
                stA["exp_avg_sq"] = torch.zeros_like(G_A32)
            if "step" not in stB:
                stB["step"] = 0
                stB["exp_avg"] = torch.zeros_like(G_B32)
                stB["exp_avg_sq"] = torch.zeros_like(G_B32)

            stA["step"] += 1
            stB["step"] += 1

            exp_avg_A = stA["exp_avg"]
            exp_avg_sq_A = stA["exp_avg_sq"]
            exp_avg_B = stB["exp_avg"]
            exp_avg_sq_B = stB["exp_avg_sq"]

            exp_avg_A.mul_(beta1).add_(G_A32, alpha=1 - beta1)
            exp_avg_sq_A.mul_(beta2).addcmul_(G_A32, G_A32, value=1 - beta2)
            exp_avg_B.mul_(beta1).add_(G_B32, alpha=1 - beta1)
            exp_avg_sq_B.mul_(beta2).addcmul_(G_B32, G_B32, value=1 - beta2)

            bias_correction1_A = 1 - beta1 ** stA["step"]
            bias_correction2_A = 1 - beta2 ** stA["step"]
            bias_correction1_B = 1 - beta1 ** stB["step"]
            bias_correction2_B = 1 - beta2 ** stB["step"]

            step_size_A = group["lr_A"] / bias_correction1_A
            step_size_B = group["lr_B"] / bias_correction1_B

            denom_A = (exp_avg_sq_A.sqrt() / (bias_correction2_A**0.5)).add_(eps)
            denom_B = (exp_avg_sq_B.sqrt() / (bias_correction2_B**0.5)).add_(eps)

            dA32 = -step_size_A * (exp_avg_A / denom_A)
            dB32 = -step_size_B * (exp_avg_B / denom_B)

            ns_gap_fro = None
            if group.get("log_ns_gap", False) and G_W32 is not None:
                deltaW_lin32 = dB32 @ A32 + B32 @ dA32
                ns_gw32 = zeropower_via_newtonschulz5(
                    -G_W32,
                    steps=group.get("ns_steps", 10),
                    eps=group.get("ns_eps", 1e-7),
                ).float()

                Nn = ns_gw32 / (ns_gw32.norm("fro") + 1e-12)
                Dn = deltaW_lin32 / (deltaW_lin32.norm("fro") + 1e-12)
                ns_gap_fro = torch.linalg.norm(Nn - Dn, ord="fro").item()

            for D in (dA32, dB32):
                D_norm = D.norm()
                if D_norm > group["max_delta_norm"]:
                    D.mul_(group["max_delta_norm"] / D_norm)

            A.add_(dA32.to(A.dtype))
            B.add_(dB32.to(B.dtype))

            if step_logs is not None and (
                group.get("log_optimizer_stats", True) or ns_gap_fro is not None
            ):
                group_logs = {"group_idx": group_idx, "ns_gap_fro": ns_gap_fro}
                if group.get("log_optimizer_stats", True):
                    group_logs.update(
                        {
                            "A_F_norm": A32.norm().item(),
                            "B_F_norm": B32.norm().item(),
                            "G_A_F_norm": G_A32.norm().item(),
                            "G_B_F_norm": G_B32.norm().item(),
                            "dA_F_norm": dA32.norm().item(),
                            "dB_F_norm": dB32.norm().item(),
                        }
                    )
                step_logs.append(group_logs)

        self.last_logs = step_logs
        return loss
