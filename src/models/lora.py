import torch
from peft import get_peft_model

from .model_fabric import LoRAModelFabric


def get_full_finetune_model(model_name, target_modules, to_float32=False):
    """Load a base model and unfreeze the selected linear weights."""
    base_model, tokenizer = LoRAModelFabric.get_model(model_name)
    base_model.gradient_checkpointing_enable()
    base_model.enable_input_require_grads()
    if not getattr(base_model, "hf_device_map", None):
        base_model = base_model.to("cuda")

    for p in base_model.parameters():
        p.requires_grad_(False)

    target_set = set(target_modules)
    for name, module in base_model.named_modules():
        if isinstance(module, torch.nn.Linear) and name.split(".")[-1] in target_set:
            if to_float32:
                module.weight.data = module.weight.data.to(torch.float32)
            module.weight = torch.nn.Parameter(module.weight.data)
            module.weight.requires_grad_(True)

    return base_model, tokenizer


def get_muon_full_param_groups(model, target_modules):
    """Return a flat list of weight tensors from target modules for Muon."""
    target_set = set(target_modules)
    params = []
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear) and name.split(".")[-1] in target_set:
            if module.weight.requires_grad:
                params.append(module.weight)
    return params


def get_lora_model(model_name, lora_config, to_float32=True, add_lora=True):
    base_model, tokenizer = LoRAModelFabric.get_model(model_name)
    sharded = bool(getattr(base_model, "hf_device_map", None))
    model = get_peft_model(base_model, lora_config)
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    if not sharded:
        model = model.to("cuda")

    if not add_lora:
        return model, tokenizer
    
    for module in model.modules():
        if hasattr(module, "lora_A") and hasattr(module, "lora_B"):
            for name, A_layer in module.lora_A.items():
                B_layer = module.lora_B[name]

                if to_float32:
                    A_layer.weight.data = A_layer.weight.data.to(torch.float32)
                    B_layer.weight.data = B_layer.weight.data.to(torch.float32)

                A_layer.weight = torch.nn.Parameter(A_layer.weight.data)
                B_layer.weight = torch.nn.Parameter(B_layer.weight.data)

    return model, tokenizer


def get_lora_param_groups(model, add_W=False):
    for p in model.parameters():
        p.requires_grad_(False)

    param_groups = []
    for module in model.modules():
        if (
            hasattr(module, "lora_A")
            and hasattr(module, "lora_B")
            and hasattr(module, "weight")
        ):

            if add_W:
                W = module.weight
                W.requires_grad_(True)
            else:
                W = None

            for adapter_name, A_layer in module.lora_A.items():
                B_layer = module.lora_B[adapter_name]
                A = A_layer.weight
                B = B_layer.weight
                A.requires_grad_(True)
                B.requires_grad_(True)

                param_groups.append({"params": [A, B], "W": W})

    return param_groups
