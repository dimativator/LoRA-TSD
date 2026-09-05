import torch
from torch.optim import Optimizer

from .utils import zeropower_via_newtonschulz5, power_iteration_t


class LoRASGD(Optimizer):
    def __init__(
        self,
        params,
        lr_A: float = 1e-3,
        lr_B: float = 1e-4,
        max_delta_norm: float = 1,
        log_ns_gap: bool = False,
    ):
        defaults = dict(
            lr_A=lr_A,
            lr_B=lr_B,
            max_delta_norm=max_delta_norm,
            log_ns_gap=log_ns_gap,
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
                    "Each param group for LoRASGD must contain exactly "
                    "two tensors: [A, B]."
                )

            A, B = params

            if A.grad is None or B.grad is None:
                continue

            A32 = A.float()
            B32 = B.float()
            G_A32 = A.grad.float()
            G_B32 = B.grad.float()

            W = group.get("W", None)
            G_W32 = None
            if W is not None:
                if getattr(W, "grad", None) is not None:
                    G_W32 = W.grad.float()
                else:
                    G_W32 = None

            dA32 = -group["lr_A"] * G_A32.clone()
            dB32 = -group["lr_B"] * G_B32.clone()

            deltaW_lin32 = dB32 @ A32 + B32 @ dA32

            ns_gap_fro = None
            if group.get("log_ns_gap", False) and G_W32 is not None:
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

            if step_logs is not None:
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
