"""LoRA layer using the doubled-rank representation of Riemannian optimizers."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class RiemannianLoRALinear(nn.Module):
    """Drop-in replacement for a frozen nn.Linear with Riemannian LoRA adapters."""

    def __init__(
        self,
        base_linear: nn.Linear,
        r: int,
        lora_alpha: int,
        lora_dropout: float = 0.0,
    ):
        super().__init__()

        out_features, in_features = base_linear.weight.shape
        self.r = r
        self.scaling = lora_alpha / r
        self.tangent_forward = False

        self.base_linear = base_linear
        for p in self.base_linear.parameters():
            p.requires_grad_(False)

        # The optimizer expects doubled-rank A and B weights.
        self.lora_A = nn.Linear(out_features, 2 * r, bias=False)
        self.lora_B = nn.Linear(2 * r, in_features, bias=False)

        self._init_weights()

        self.lora_dropout = nn.Dropout(lora_dropout) if lora_dropout > 0.0 else nn.Identity()

    def _init_weights(self):
        with torch.no_grad():
            out_features = self.lora_A.weight.shape[1]

            Q, _ = torch.linalg.qr(torch.randn(out_features, self.r,
                                                device=self.lora_A.weight.device,
                                                dtype=self.lora_A.weight.dtype))
            self.lora_A.weight.data.zero_()
            self.lora_A.weight.data[:self.r, :] = Q.T

            self.lora_B.weight.data.zero_()

    def forward(self, x):
        base_out = self.base_linear(x)

        A_param = self.lora_A.weight
        B_param = self.lora_B.weight

        A_eff = A_param.T[:, :self.r]
        B_eff = B_param.T[:self.r, :]

        x_f32 = self.lora_dropout(x).float()
        if self.tangent_forward:
            lora_out = F.linear(x_f32, B_param.T)
            lora_out = F.linear(lora_out, A_param.T)
        else:
            lora_out = F.linear(x_f32, B_eff)
            lora_out = F.linear(lora_out, A_eff)

        return base_out + lora_out.to(x.dtype) * self.scaling


def apply_riemannian_lora(model, r: int, lora_alpha: int, lora_dropout: float = 0.0):
    """Replace PEFT LoRA linear modules with Riemannian LoRA layers."""
    replacements = {}

    for name, module in model.named_modules():
        if not (hasattr(module, "base_layer") and
                hasattr(module, "lora_A") and
                hasattr(module, "lora_B")):
            continue

        base_linear = module.base_layer
        if not isinstance(base_linear, nn.Linear):
            continue

        device = base_linear.weight.device
        dtype  = base_linear.weight.dtype

        new_module = RiemannianLoRALinear(
            base_linear, r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout
        )
        new_module = new_module.to(device=device, dtype=dtype)
        new_module.lora_A.weight.data = new_module.lora_A.weight.data.to(torch.float32)
        new_module.lora_B.weight.data = new_module.lora_B.weight.data.to(torch.float32)

        replacements[name] = new_module

    for name, new_module in replacements.items():
        if "." in name:
            parent_name, child_name = name.rsplit(".", 1)
            parent = model.get_submodule(parent_name)
        else:
            parent, child_name = model, name
        setattr(parent, child_name, new_module)

    return model


def get_riemannian_param_groups(model):
    """Return parameter groups for a Riemannian LoRA optimizer."""
    param_groups = []
    for module in model.modules():
        if isinstance(module, RiemannianLoRALinear):
            module.lora_A.weight.requires_grad_(True)
            module.lora_B.weight.requires_grad_(True)
            param_groups.append({
                "params":      [module.lora_A.weight, module.lora_B.weight],
                "param_names": ["lora_A.weight", "lora_B.weight"],
            })
    return param_groups
