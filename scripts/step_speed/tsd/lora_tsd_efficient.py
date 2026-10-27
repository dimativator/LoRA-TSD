"""LoRA-TSD with fixed inner-loop bases and device-side update scaling."""

import torch

from .lora_tsd import LoRATSD, _ns_sq


class LoRATSDEfficient(LoRATSD):
    """Run LoRATSD's alternating projections in a fixed small core.

    The first r columns of each augmented QR span the current LoRA factor.
    In these coordinates the tangent projector only zeros the bottom-right
    block. Bases are rebuilt each step, but never inside the alternation loop.
    Scheduling, momentum, diagnostics and factor reconstruction are inherited.
    The iteration is equivalent in exact arithmetic. Rank-deficient inputs
    can differ from the original's rounding-sensitive repeated QR path.
    """

    def _prepare_bases(
        self, A: torch.Tensor, B: torch.Tensor, G_A: torch.Tensor,
        G_B: torch.Tensor, group: dict,
    ) -> tuple:
        if self.use_momentum and group.get("vector_transport", False):
            # Transport can introduce directions outside the fresh gradients.
            return super()._prepare_bases(A, B, G_A, G_B, group)
        r = A.shape[0]
        U = torch.linalg.qr(torch.cat([B, G_B], dim=1), mode="reduced")[0]
        V = torch.linalg.qr(torch.cat([A.t(), G_A.t()], dim=1), mode="reduced")[0]
        return U[:, :r], V[:, :r], (U, V)

    def _spectral_step(
        self, L: torch.Tensor, R: torch.Tensor, U_B: torch.Tensor,
        V_A: torch.Tensor, group: dict, bases: tuple | None,
    ) -> tuple:
        r = U_B.shape[1]
        if bases is None:
            # Projected momentum still has two rank-r blocks. Its complement
            # directions are available after transport, rather than before it.
            U = torch.linalg.qr(
                torch.cat([U_B, L[:, r:]], dim=1), mode="reduced",
            )[0]
            V = torch.linalg.qr(
                torch.cat([V_A, R[:r].t()], dim=1), mode="reduced",
            )[0]
        else:
            U, V = bases
        left = U.t() @ L
        right = R @ V
        # These blocks vanish by construction: L's first block lies in U_B,
        # and R's second block lies in V_A. Keep the exact zeros rather than
        # amplifying QR roundoff in null directions over repeated NS steps.
        left[r:, :r] = 0
        right[r:, r:] = 0
        core = left @ right
        for _ in range(group["ball_iters"]):
            core = _ns_sq(core, steps=group["ns_steps"])
            core[r:, r:] = 0
        return U @ core, V.t(), core.norm()

    def _clip_deltas(
        self, dA: torch.Tensor, dB: torch.Tensor, dW_norm: torch.Tensor,
        max_delta_norm: float,
    ) -> None:
        # Use a safe denominator even when the update is exactly zero.
        denominator = torch.where(dW_norm > 0, dW_norm, torch.ones_like(dW_norm))
        scale = torch.where(
            dW_norm > max_delta_norm,
            max_delta_norm / denominator,
            torch.ones_like(dW_norm),
        )
        dA.mul_(scale)
        dB.mul_(scale)

    @torch.no_grad()
    def _rebalance_group(self, A: torch.Tensor, B: torch.Tensor) -> None:
        A32, B32 = A.float(), B.float()
        nA, nB = A32.norm(), B32.norm()
        valid = (nA >= self.balance_eps) & (nB >= self.balance_eps)
        bufA = self.state[A].get("momentum_buffer") if self.use_momentum else None
        bufB = self.state[B].get("momentum_buffer") if self.use_momentum else None

        if self.balance_mode == "norm":
            safe_nA = torch.where(valid, nA, torch.ones_like(nA))
            safe_nB = torch.where(valid, nB, torch.ones_like(nB))
            c = (safe_nB / safe_nA).sqrt()
            A.mul_(c.to(A.dtype))
            B.div_(c.to(B.dtype))
            if bufA is not None:
                bufA.div_(c)
            if bufB is not None:
                bufB.mul_(c)
        elif self.balance_mode == "whiten":
            def sym_pow(S: torch.Tensor, power: float) -> torch.Tensor:
                w, V = torch.linalg.eigh(0.5 * (S + S.t()))
                return (V * w.clamp_min(1e-12).pow(power)) @ V.t()

            S_A, S_B = A32 @ A32.t(), B32.t() @ B32
            SA_h = sym_pow(S_A, 0.5)
            SA_ih = sym_pow(S_A, -0.5)
            P = SA_ih @ sym_pow(SA_h @ S_B @ SA_h, 0.5) @ SA_ih
            identity = torch.eye(A32.shape[0], device=A.device, dtype=A32.dtype)
            M = torch.where(valid, sym_pow(P, 0.5), identity)
            Minv = torch.where(valid, sym_pow(P, -0.5), identity)
            A.copy_((M @ A32).to(A.dtype))
            B.copy_((B32 @ Minv).to(B.dtype))
            if bufA is not None:
                bufA.copy_(Minv @ bufA)
            if bufB is not None:
                bufB.copy_(bufB @ M)
        else:
            raise ValueError(f"Unknown balance_mode: {self.balance_mode}")
