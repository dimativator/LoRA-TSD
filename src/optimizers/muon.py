"""Muon implementation adapted from https://github.com/KellerJordan/modded-nanogpt."""

import os

import torch
import torch.distributed as dist
from typing import Callable

from .utils import zeropower_via_newtonschulz5


class Muon(torch.optim.Optimizer):
    """Apply momentum followed by Newton-Schulz orthogonalization to matrix updates."""

    def __init__(
        self,
        muon_params,
        lr=0.02,
        momentum=0.95,
        nesterov=True,
        ns_steps=6,
        max_delta_norm=0.0,
        adamw_params=None,
        adamw_lr=3e-4,
        adamw_betas=(0.95, 0.95),
        adamw_eps=1e-8,
        adamw_wd=0,
    ):
        defaults = dict(
            lr=lr,
            momentum=momentum,
            nesterov=nesterov,
            ns_steps=ns_steps,
            max_delta_norm=max_delta_norm,
            adamw_lr=adamw_lr,
            adamw_lr_ratio=adamw_lr / lr,
            adamw_betas=adamw_betas,
            adamw_eps=adamw_eps,
            adamw_wd=adamw_wd,
        )

        params = list(muon_params)
        adamw_params = list(adamw_params) if adamw_params is not None else []
        params.extend(adamw_params)
        super().__init__(params, defaults)

        for p in muon_params:
            if p.ndim >= 2 and p.size(0) < 10000:
                self.state[p]["use_muon"] = True
            else:
                self.state[p]["use_muon"] = False
        for p in adamw_params:
            self.state[p]["use_muon"] = False

        if "WORLD_SIZE" in os.environ:
            self.world_size = int(os.environ["WORLD_SIZE"])
            self.rank = int(os.environ["RANK"])
        else:
            self.world_size = 1
            self.rank = 0

    def step(self, closure: Callable = None):
        """Perform one optimization step."""
        loss = None
        if closure is not None:
            loss = closure()
        for group in self.param_groups:
            params = [p for p in group["params"] if self.state[p]["use_muon"]]
            lr = group["lr"]
            momentum = group["momentum"]

            total_params = sum(p.numel() for p in params)
            updates_flat = torch.zeros(
                total_params, device="cuda", dtype=torch.bfloat16
            )
            curr_idx = 0
            for i, p in enumerate(params):
                if i % self.world_size == self.rank:
                    g = p.grad
                    if g.ndim > 2:
                        g = g.view(g.size(0), -1)
                    assert g is not None
                    state = self.state[p]
                    if "momentum_buffer" not in state:
                        state["momentum_buffer"] = torch.zeros_like(g)
                    buf = state["momentum_buffer"]
                    buf.mul_(momentum).add_(g)
                    if group["nesterov"]:
                        g = g.add(buf, alpha=momentum)
                    g = zeropower_via_newtonschulz5(
                        g, steps=group["ns_steps"], eps=group["adamw_eps"]
                    )
                    g *= max(1, g.size(0) / g.size(1)) ** 0.5
                    updates_flat[curr_idx : curr_idx + p.numel()] = g.flatten()
                curr_idx += p.numel()

            if self.world_size > 1:
                dist.all_reduce(updates_flat, op=dist.ReduceOp.SUM)

            curr_idx = 0
            for p in params:
                g = (
                    updates_flat[curr_idx : curr_idx + p.numel()]
                    .view_as(p.data)
                    .type_as(p.data)
                )
                if group["max_delta_norm"] and group["max_delta_norm"] > 0:
                    g_norm = g.norm()
                    if g_norm > group["max_delta_norm"]:
                        g.mul_(group["max_delta_norm"] / g_norm)
                p.data.add_(g, alpha=-lr)

                curr_idx += p.numel()

            params = [p for p in group["params"] if not self.state[p]["use_muon"]]
            lr = (
                group["adamw_lr_ratio"] * group["lr"]
            )
            beta1, beta2 = group["adamw_betas"]
            eps = group["adamw_eps"]
            weight_decay = group["adamw_wd"]

            for p in params:
                g = p.grad
                assert g is not None
                state = self.state[p]
                if "step" not in state:
                    state["step"] = 0
                    state["moment1"] = torch.zeros_like(g)
                    state["moment2"] = torch.zeros_like(g)
                state["step"] += 1
                step = state["step"]
                buf1 = state["moment1"]
                buf2 = state["moment2"]
                buf1.lerp_(g, 1 - beta1)
                buf2.lerp_(g.square(), 1 - beta2)

                g = buf1 / (eps + buf2.sqrt())

                bias_correction1 = 1 - beta1**step
                bias_correction2 = 1 - beta2**step
                scale = bias_correction1 / bias_correction2**0.5
                p.data.mul_(1 - lr * weight_decay)
                p.data.add_(g, alpha=-lr / scale)

        return loss


def separate_params(param_groups):
    param_groups_2d = []
    param_groups_non2d = []
    total_param_2d_count = 0
    total_param_non2d_count = 0

    if (
        isinstance(param_groups, list) and isinstance(param_groups[0], dict)
    ) or isinstance(param_groups, dict):
        if isinstance(param_groups, dict):
            param_groups = [param_groups]
        for group in param_groups:
            (
                params_2d,
                params_non2d,
                param_2d_count,
                param_non2d_count,
            ) = separate_params(group["params"])
            param_group_2d = {"params": params_2d}
            param_group_non2d = {"params": params_non2d}
            for k in group.keys():
                if k != "params":
                    param_group_2d[k] = group[k]
                    param_group_non2d[k] = group[k]

            param_groups_2d.append(param_group_2d)
            param_groups_non2d.append(param_group_non2d)
            total_param_2d_count += param_2d_count
            total_param_non2d_count += param_non2d_count

        return (
            param_groups_2d,
            param_groups_non2d,
            total_param_2d_count,
            total_param_non2d_count,
        )

    elif isinstance(param_groups, list) and isinstance(param_groups[0], torch.Tensor):
        params_2d = []
        params_non2d = []
        param_group = param_groups
        for param in param_group:
            if param.ndim == 2:
                params_2d.append(param)
            else:
                params_non2d.append(param)
        return params_2d, params_non2d, len(params_2d), len(params_non2d)
    else:
        raise TypeError("Expected optimizer parameter groups or a list of tensors.")
