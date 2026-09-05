import torch
from torch.optim import Optimizer

from .utils import *


@torch.no_grad()
def _ns_sq(T: torch.Tensor, steps: int = 5) -> torch.Tensor:
    """Apply Newton-Schulz orthogonalization to a square matrix."""
    a, b, c = 3.4445, -4.7750, 2.0315
    X = T / (T.norm() + 1e-7)
    for _ in range(steps):
        A = X @ X.t()
        X = a * X + (b * A + c * A @ A) @ X
    return X


@torch.no_grad()
def _proj_T_factored(L: torch.Tensor, R: torch.Tensor,
                     U_B: torch.Tensor, V_A: torch.Tensor):
    """Apply the tangent projector to L @ R in factored form."""
    C = U_B.t() @ L
    D = R @ V_A
    L_new = torch.cat([U_B, L @ D], dim=1)
    R_new = torch.cat([C @ R - (C @ D) @ V_A.t(), V_A.t()], dim=0)
    return L_new, R_new


@torch.no_grad()
def f(A, B, dA, dB, G_A, G_B, alpha, lambda_l2_A, lambda_l2_B, norm_ord):
    """Evaluate the diagnostic inner objective."""
    if norm_ord == 2:
        norm_term, _, _ = top_singular_triplet_lora(A, B, dA, dB, iters=10)
    elif norm_ord == "fro":
        norm_term = torch.linalg.norm(dB @ A + B @ dA, ord=norm_ord)

    loss = (
        torch.trace(G_A.T @ dA)
        + torch.trace(G_B.T @ dB)
        + alpha * norm_term ** 2
        + lambda_l2_A * dA.pow(2).sum()
        + lambda_l2_B * dB.pow(2).sum()
    )
    return loss


@torch.no_grad()
def _spd_inv(M: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Invert a small SPD matrix after adding a relative ridge."""
    r = M.shape[0]
    ridge = eps * M.diagonal().mean().clamp_min(1e-12)
    return torch.linalg.inv(M + ridge * torch.eye(r, device=M.device, dtype=M.dtype))


@torch.no_grad()
def tangent_projected_grad(A, B, G_A, G_B, BtB_i=None, AAt_i=None):
    """Compute P_T(G_W) from the LoRA factor gradients.

        g = P_B G_W + G_W P_A - P_B G_W P_A
          = B (B^T B)^{-1} G_A
          + G_B (A A^T)^{-1} A
          - B (B^T B)^{-1} (G_A A^T) (A A^T)^{-1} A.
    """
    if BtB_i is None:
        BtB_i = _spd_inv(B.t() @ B)
    if AAt_i is None:
        AAt_i = _spd_inv(A @ A.t())

    g = (
        B @ (BtB_i @ G_A)
        + (G_B @ AAt_i) @ A
        - B @ (BtB_i @ (G_A @ A.t()) @ AAt_i) @ A
    )
    return g


class LoRATSD(Optimizer):
    """
    LoRA tangent-space spectral descent.

    The optimizer reconstructs the tangent-projected weight gradient from the
    factor gradients. Each inner iteration applies matrix-sign
    orthogonalization and projects the result back to the tangent space:

        X_{0} = g,   X_{k+1} = P_T( msign(X_k) ),

    Newton-Schulz runs on a small QR core. Each parameter group contains
    [A, B] with shapes (r, n) and (m, r).
    """

    def __init__(
        self,
        params,
        norm_ord: int = 2,
        use_momentum: bool = True,
        momentum_cf: float = 0.9,
        lr_A: float = 1e-3,
        lr_B: float = 1e-4,
        lr_A_scheduler=None,
        lr_B_scheduler=None,
        alpha=None,
        lambda_l2_A: float = 0.0,
        lambda_l2_B: float = 0.0,
        ball_iters: int = 10,
        ridge_eps: float = 1e-8,
        max_delta_norm: float = 1.0,
        ns_steps: int = 6,
        ns_eps: float = 1e-7,
        balance: bool = False,
        balance_mode: str = "norm",
        rebalance_every: int = 1,
        balance_eps: float = 1e-12,
        log_ns_gap: bool = False,
        log_f_inner: bool = False,
        log_grads: bool = False,
        log_optimizer_stats: bool = True,
    ):
        defaults = dict(
            norm_ord=norm_ord,
            lr_A=lr_A,
            lr_B=lr_B,
            alpha=(1 / lr_A + 1 / lr_B) / 2 if not alpha else alpha,
            lambda_l2_A=lambda_l2_A,
            lambda_l2_B=lambda_l2_B,
            ball_iters=ball_iters,
            ridge_eps=ridge_eps,
            max_delta_norm=max_delta_norm,
            ns_steps=ns_steps,
            ns_eps=ns_eps,
            log_ns_gap=log_ns_gap,
            log_f_inner=log_f_inner,
            log_grads=log_grads,
            log_optimizer_stats=log_optimizer_stats,
        )
        self.last_logs = None

        self.use_momentum = use_momentum
        self.momentum_cf = momentum_cf

        self.balance = balance
        self.balance_mode = balance_mode
        self.rebalance_every = max(1, int(rebalance_every))
        self.balance_eps = balance_eps

        self.lr_A_scheduler = lr_A_scheduler
        self.lr_B_scheduler = lr_B_scheduler
        self._sched_step = 0

        super().__init__(params, defaults)

        for g in self.param_groups:
            g.setdefault("base_lr_A", float(g["lr_A"]))
            g.setdefault("base_lr_B", float(g["lr_B"]))

    def _apply_scheduler_to_key(self, sched, key: str, step_idx: int, group_idx: int):
        if sched is None:
            return

        if callable(sched) and not hasattr(sched, "step"):
            base_key = "base_lr_A" if key == "lr_A" else "base_lr_B"
            base_lr = float(self.param_groups[group_idx].get(base_key, self.param_groups[group_idx][key]))
            try:
                new_lr = sched(step_idx, base_lr, group_idx)
            except TypeError:
                new_lr = sched(step_idx)
            self.param_groups[group_idx][key] = float(new_lr)
            return

        for g in self.param_groups:
            g["_saved_lr"] = g.get("lr", None)
            g["lr"] = float(g[key])
        sched.step()
        for g in self.param_groups:
            g[key] = float(g["lr"])
            saved = g.pop("_saved_lr")
            if saved is None:
                g.pop("lr", None)
            else:
                g["lr"] = saved

    @torch.no_grad()
    def _rebalance_group(self, A, B):
        """Rebalance A and B without changing their product."""
        A32 = A.data.float()
        B32 = B.data.float()
        nA = A32.norm()
        nB = B32.norm()
        if nA < self.balance_eps or nB < self.balance_eps:
            return

        bufA = self.state[A].get("momentum_buffer", None) if self.use_momentum else None
        bufB = self.state[B].get("momentum_buffer", None) if self.use_momentum else None

        if self.balance_mode == "norm":
            c = (nB / nA).sqrt()
            A.data.mul_(c.to(A.dtype))
            B.data.div_(c.to(B.dtype))
            if bufA is not None:
                bufA.div_(c)
            if bufB is not None:
                bufB.mul_(c)
        elif self.balance_mode == "whiten":
            r = A32.shape[0]
            I = torch.eye(r, device=A32.device, dtype=A32.dtype)

            def _sym_pow(S, p):
                S = 0.5 * (S + S.t())
                w, V = torch.linalg.eigh(S)
                w = w.clamp_min(1e-12)
                return (V * w.pow(p)) @ V.t()

            S_A = A32 @ A32.t()
            S_B = B32.t() @ B32
            SA_h = _sym_pow(S_A, 0.5)
            SA_ih = _sym_pow(S_A, -0.5)
            P = SA_ih @ _sym_pow(SA_h @ S_B @ SA_h, 0.5) @ SA_ih
            M = _sym_pow(P, 0.5)
            Minv = _sym_pow(P, -0.5)
            A.data.copy_((M @ A32).to(A.dtype))
            B.data.copy_((B32 @ Minv).to(B.dtype))
            if bufA is not None:
                bufA.copy_(Minv @ bufA)
            if bufB is not None:
                bufB.copy_(bufB @ M)
        else:
            raise ValueError(f"Unknown balance_mode: {self.balance_mode}")

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        self._sched_step += 1
        for gi in range(len(self.param_groups)):
            self._apply_scheduler_to_key(self.lr_A_scheduler, "lr_A", self._sched_step, gi)
            self._apply_scheduler_to_key(self.lr_B_scheduler, "lr_B", self._sched_step, gi)

        step_logs = [] if any(
            group.get("log_optimizer_stats", True)
            or group.get("log_ns_gap", False)
            or group.get("log_f_inner", False)
            or group.get("log_grads", False)
            for group in self.param_groups
        ) else None

        for group_idx, group in enumerate(self.param_groups):
            params = list(group["params"])
            if len(params) != 2:
                raise ValueError(
                    "Each param group for LoRATSD must contain "
                    "exactly two tensors: [A, B]."
                )

            A, B = params
            if A.grad is None or B.grad is None:
                continue

            A32 = A.float()
            B32 = B.float()
            G_A32 = A.grad.float()
            G_B32 = B.grad.float()

            raw_G_A = G_A32.detach().cpu() if group.get("log_grads") else None
            raw_G_B = G_B32.detach().cpu() if group.get("log_grads") else None

            if self.use_momentum:
                stA = self.state[A]
                bufA = stA.get("momentum_buffer", None)
                if bufA is None:
                    bufA = torch.zeros_like(G_A32)
                bufA.mul_(self.momentum_cf).add_(G_A32, alpha=(1.0 - self.momentum_cf))
                stA["momentum_buffer"] = bufA
                G_A32 = bufA

                stB = self.state[B]
                bufB = stB.get("momentum_buffer", None)
                if bufB is None:
                    bufB = torch.zeros_like(G_B32)
                bufB.mul_(self.momentum_cf).add_(G_B32, alpha=(1.0 - self.momentum_cf))
                stB["momentum_buffer"] = bufB
                G_B32 = bufB

            if group.get("log_ns_gap", False):
                W = group.get("W", None)
                G_W32 = W.grad.float() if (W is not None and getattr(W, "grad", None) is not None) else None
            else:
                G_W32 = None

            r = A32.shape[0]
            lr_A = group["lr_A"]
            lr_B = group["lr_B"]

            U_B = torch.linalg.qr(B32, mode="reduced")[0]
            V_A = torch.linalg.qr(A32.t(), mode="reduced")[0]
            BtB_i = _spd_inv(B32.t() @ B32, group["ridge_eps"])
            AAt_i = _spd_inv(A32 @ A32.t(), group["ridge_eps"])

            # Factor P_T(G_W) as L_g @ R_g.
            # Here M = (BtB)^(-1) (G_A A^T) (AAt)^(-1).
            M = BtB_i @ (G_A32 @ A32.t()) @ AAt_i
            L_g = torch.cat([B32, G_B32 @ AAt_i], dim=1)
            R_g = torch.cat([BtB_i @ G_A32 - M @ A32, A32], dim=0)

            if group["log_f_inner"]:
                dA_sgd = -lr_A * G_A32
                dB_sgd = -lr_B * G_B32
                f_start = f(A32, B32, dA_sgd, dB_sgd, G_A32, G_B32,
                            group["alpha"], group["lambda_l2_A"],
                            group["lambda_l2_B"], group["norm_ord"]).cpu()
            else:
                f_start = None

            # Apply msign through the small QR core, then project onto T.
            ns_steps = group["ns_steps"]
            L, R = L_g, R_g
            for _ in range(group["ball_iters"]):
                Q_L, R_L = torch.linalg.qr(L, mode="reduced")
                Q_R, R_R = torch.linalg.qr(R.t(), mode="reduced")
                T_orth = _ns_sq(R_L @ R_R.t(), steps=ns_steps)
                L, R = _proj_T_factored(Q_L @ T_orth, Q_R.t(), U_B, V_A)

            GLL = L.t() @ L
            GRR = R @ R.t()
            X_fro = (GLL * GRR).sum().clamp_min(0).sqrt()

            cos_grad = None
            dual_ratio = None
            if group.get("log_optimizer_stats", True):
                M1 = L_g.t() @ L
                dot_gX = (R_g * (M1 @ R)).sum()
                _, R_Lg = torch.linalg.qr(L_g, mode="reduced")
                _, R_Rg = torch.linalg.qr(R_g.t(), mode="reduced")
                g_core = R_Lg @ R_Rg.t()
                g_norm = g_core.norm()
                cos_grad = dot_gX / (g_norm * X_fro + 1e-12)
                if self._sched_step % 50 == 0:
                    g_nuclear_norm = torch.linalg.svdvals(g_core).sum()
                    dual_ratio = dot_gX / (g_nuclear_norm + 1e-12)

            # Reconstruct factor updates whose linearized product is -lr_A L R.
            RA_t = R @ A32.t()
            dB32 = -lr_A * (L @ RA_t) @ AAt_i
            BtL = B32.t() @ L
            BtdB = B32.t() @ dB32
            dA32 = BtB_i @ (-lr_A * (BtL @ R) - BtdB @ A32)

            if group["log_f_inner"]:
                f_end = f(A32, B32, dA32, dB32, G_A32, G_B32,
                          group["alpha"], group["lambda_l2_A"],
                          group["lambda_l2_B"], group["norm_ord"]).cpu()
            else:
                f_end = None

            S_spec = S_sgd = S_muon = None
            ball_flatness = None
            if group.get("log_ns_gap", False):
                G_A_ns = zeropower_via_newtonschulz5(
                    -G_A32, steps=group["ns_steps"], eps=group["ns_eps"]).float()
                G_B_ns = zeropower_via_newtonschulz5(
                    -G_B32, steps=group["ns_steps"], eps=group["ns_eps"]).float()

                _, S_spec, _ = lowrank_update_top_svd(A32, B32, dA32, dB32, k=r)
                _, S_sgd, _ = lowrank_update_top_svd(
                    A32, B32, -lr_A * G_A32, -lr_B * G_B32, k=r)
                _, S_muon, _ = lowrank_update_top_svd(
                    A32, B32, -lr_A * G_A_ns, -lr_B * G_B_ns, k=r)

                if S_spec is not None and S_spec.numel() > 1:
                    ball_flatness = float(S_spec[0] / S_spec[-1].clamp_min(1e-12))

            ns_gap_fro = None
            if group.get("log_ns_gap", False) and G_W32 is not None:
                deltaW_lin32 = dB32 @ A32 + B32 @ dA32
                ns_gw32 = zeropower_via_newtonschulz5(
                    -G_W32, steps=max(group["ns_steps"], 10), eps=group["ns_eps"]).float()
                Nn = ns_gw32 / (ns_gw32.norm("fro") + 1e-12)
                Dn = deltaW_lin32 / (deltaW_lin32.norm("fro") + 1e-12)
                ns_gap_fro = torch.linalg.norm(Nn - Dn, ord="fro").item()

            dW_norm = lr_A * X_fro
            if dW_norm > group["max_delta_norm"]:
                scale = group["max_delta_norm"] / dW_norm
                dA32.mul_(scale)
                dB32.mul_(scale)

            A.add_(dA32.to(A.dtype))
            B.add_(dB32.to(B.dtype))

            if self.balance and (self._sched_step % self.rebalance_every == 0):
                self._rebalance_group(A, B)

            if step_logs is not None:
                group_logs = {
                    "group_idx": group_idx,
                    "G_A": raw_G_A,
                    "G_B": raw_G_B,
                    "A": A32.detach().cpu() if group.get("log_grads") else None,
                    "B": B32.detach().cpu() if group.get("log_grads") else None,
                    "rho": None,
                    "f_start": f_start,
                    "f_end": f_end,
                    "f_losses": None,
                    "ns_gap_fro": ns_gap_fro,
                    "singular_values_spec_upd": S_spec,
                    "singular_values_sgd_upd": S_sgd,
                    "singular_values_muon_upd": S_muon,
                    "ball_flatness": ball_flatness,
                    "ball_dual_ratio": (
                        float(dual_ratio) if dual_ratio is not None else None
                    ),
                }
                if group.get("log_optimizer_stats", True):
                    group_logs.update(
                        {
                            "A_F_norm": A32.norm().item(),
                            "B_F_norm": B32.norm().item(),
                            "G_A_F_norm": G_A32.norm().item(),
                            "G_B_F_norm": G_B32.norm().item(),
                            "dA_F_norm": dA32.norm().item(),
                            "dB_F_norm": dB32.norm().item(),
                            "lr_A": lr_A,
                            "lr_B": lr_B,
                            "ball_cos_grad": float(cos_grad),
                            "dW_F_norm": dW_norm.item(),
                        }
                    )
                step_logs.append(group_logs)

        self.last_logs = step_logs
        return loss
