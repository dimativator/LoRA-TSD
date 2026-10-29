"""Controlled direction ablation with the existing LoRA-TSD factor retraction."""

import json
from pathlib import Path
import sys
import time

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/step_speed"))
from tsd.lora_tsd_efficient import LoRATSDEfficient

VARIANTS = ("frobenius", "one_step", "factorwise", "coupled")


@torch.no_grad()
def rectangular_sign(matrix: torch.Tensor, steps: int = 5) -> torch.Tensor:
    """Use the coupled method's FP32 quintic polynomial on a rectangular input."""
    transposed = matrix.shape[0] > matrix.shape[1]
    work = matrix.t() if transposed else matrix
    work = work / (work.norm() + 1e-7)
    for _ in range(steps):
        gram = work @ work.t()
        work = 3.4445 * work + (-4.7750 * gram + 2.0315 * gram @ gram) @ work
    return work.t() if transposed else work


@torch.no_grad()
def factored_singular_values(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    """Compute the spectrum of left @ right without allocating that product."""
    _, left_core = torch.linalg.qr(left, mode="reduced")
    _, right_core = torch.linalg.qr(right.t(), mode="reduced")
    return torch.linalg.svdvals(left_core @ right_core.t())


def spectrum_record(values: torch.Tensor) -> dict:
    values = values.detach().double().cpu()
    leading = float(values[0]) if values.numel() else 0.0
    normalized = values / leading if leading > 0 else torch.zeros_like(values)
    stable_rank = float(normalized.square().sum())
    return {"singular_values": values.tolist(), "normalized_singular_values": normalized.tolist(),
            "stable_rank": stable_rank, "spectral_norm": leading,
            "frobenius_norm": float(values.norm())}


class TangentAblation(LoRATSDEfficient):
    """Change direction construction while inheriting the complete TSD update path.

    Factorwise uses the same momentum gradients and matrix-sign polynomial as
    coupled TSD, but orthogonalizes G_A and G_B independently. Its induced
    direction B sign(G_A) + sign(G_B) A is reconstructed through the exact same
    canonical split as every other variant. All groups must use equal lr_A/B.
    """

    def __init__(self, params, variant: str, lr: float = .005,
                 spectrum_path: Path | str | None = None,
                 spectrum_steps: tuple | list | None = None, coupled_tau: int = 5, **kwargs):
        if not isinstance(coupled_tau, int) or coupled_tau < 1:
            raise ValueError("coupled_tau must be a positive integer")
        if variant not in VARIANTS:
            raise ValueError(f"Unknown variant: {variant}")
        if "ball_iters" in kwargs:
            raise ValueError("The direction variant determines ball_iters")
        options = {"lr_A": lr, "lr_B": lr, "use_momentum": True,
                   "momentum_cf": .95, "ns_steps": 5, "balance": True,
                   "max_delta_norm": .1, "log_optimizer_stats": False}
        options.update(kwargs)
        options["ball_iters"] = {"frobenius": 0, "one_step": 1,
                                 "factorwise": 0, "coupled": coupled_tau}[variant]
        super().__init__(params, **options)
        self.variant = variant
        self.coupled_tau = coupled_tau
        self.spectrum_path = Path(spectrum_path) if spectrum_path is not None else None
        self.spectrum_steps = set(spectrum_steps if spectrum_steps is not None
                                  else [1, *range(250, 3001, 250)])
        names = [group.get("name", f"group_{i}") for i, group in enumerate(self.param_groups)]
        candidates = [i for i, name in enumerate(names) if "q_proj" in name]
        if not candidates:
            candidates = list(range(len(names)))
        self.spectrum_groups = {candidates[i] for i in (0, len(candidates) // 2, len(candidates) - 1)}
        self._group_indices = {id(group): i for i, group in enumerate(self.param_groups)}
        self._captured = {}
        self._pending_spectra = {}
        self._direction_inputs = None
        for group in self.param_groups:
            if group["lr_A"] != group["lr_B"]:
                raise ValueError("Shared canonical retraction requires equal factor learning rates")

    def _prepare_bases(self, A, B, G_A, G_B, group):
        self._direction_inputs = (A, B, G_A, G_B, group)
        return super()._prepare_bases(A, B, G_A, G_B, group)

    def _spectral_step(self, L, R, U_B, V_A, group, bases):
        if self.variant == "factorwise":
            A, B, G_A, G_B, _ = self._direction_inputs
            sign_A = rectangular_sign(G_A, group["ns_steps"])
            sign_B = rectangular_sign(G_B, group["ns_steps"])
            L = torch.cat([B, sign_B], dim=1)
            R = torch.cat([sign_A, A], dim=0)
            norm = ((L.t() @ L) * (R @ R.t())).sum().clamp_min(0).sqrt()
        else:
            L, R, norm = super()._spectral_step(L, R, U_B, V_A, group, bases)
        index = self._group_indices[id(group)]
        if index in self._captured:
            self._pending_spectra[index] = {
                "direction": spectrum_record(factored_singular_values(L, R))}
        return L, R, norm

    def _clip_deltas(self, dA, dB, dW_norm, max_delta_norm):
        super()._clip_deltas(dA, dB, dW_norm, max_delta_norm)
        A, B, _, _, group = self._direction_inputs
        index = self._group_indices[id(group)]
        if index in self._captured:
            left = torch.cat([B, dB], dim=1)
            right = torch.cat([dA, A], dim=0)
            self._pending_spectra[index]["tangent_update"] = spectrum_record(
                factored_singular_values(left, right))

    @torch.no_grad()
    def step(self, closure=None):
        record_step = self._sched_step + 1
        self._captured = {}
        self._pending_spectra = {}
        if self.spectrum_path is not None and record_step in self.spectrum_steps:
            for index in self.spectrum_groups:
                A, B = self.param_groups[index]["params"]
                if A.grad is not None and B.grad is not None:
                    self._captured[index] = (A.detach().float().clone(), B.detach().float().clone())
        result = super().step(closure)
        if self._captured:
            self.spectrum_path.parent.mkdir(parents=True, exist_ok=True)
            with self.spectrum_path.open("a") as stream:
                for index, (old_A, old_B) in self._captured.items():
                    group = self.param_groups[index]
                    A, B = group["params"]
                    left = torch.cat([B.float(), -old_B], dim=1)
                    right = torch.cat([A.float(), old_A], dim=0)
                    row = {"timestamp": time.time(), "step": record_step, "variant": self.variant,
                           "group": group.get("name", f"group_{index}"), "group_index": index,
                           "rank": A.shape[0], "lr": group["lr_A"],
                           "ball_iters": group["ball_iters"], "coupled_tau": self.coupled_tau,
                           **self._pending_spectra[index],
                           "applied_update": spectrum_record(factored_singular_values(left, right))}
                    stream.write(json.dumps(row, allow_nan=False) + "\n")
        self._captured = {}
        self._pending_spectra = {}
        self._direction_inputs = None
        return result
