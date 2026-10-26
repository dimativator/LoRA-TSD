"""BoolQ-1B optimizer settings and generation-specific schedule normalization."""

import math

import torch

from args_parser import parse_config
from models import RiemannianLoRALinear
from optimizers import build_optimizer, make_linear_warmup_decay
from optimizers.schedulers import make_torch_linear_warmup_decay


METHODS = (
    "lora_sgd", "adamw", "lora_muon", "riemannian_sgd", "lora_rite",
    "riemannion_original", "lora_pro", "lora_muon_manifold", "lora_tsd", "muon_full",
)

PRESETS = {
    "lora_sgd": {"lr": 0.001, "flags": ["--use-momentum", "--momentum-cf", "0.9"],
                 "source": "lr_sweep_lora_sgd_loi.sh",
                 "notes": "Historical SGD implementation ignores momentum flags; plain factor SGD."},
    "adamw": {"lr": 0.001, "flags": ["--weight-decay", "0.01", "--grad-clip-norm", "1"],
              "source": "lr_sweep_lora_adamw_loi.sh",
              "notes": "Corrected original torch AdamW with independent A/B groups; no delta clipping."},
    "lora_muon": {"lr": 0.01, "flags": ["--grad-clip-norm", "1", "--max-delta-norm", "0.01"],
                  "source": "lr_sweep_lora_muon_loi.sh", "notes": "Factor-wise Muon, NS5, Nesterov."},
    "riemannian_sgd": {"lr": 0.002, "flags": [],
                       "source": "riemannian_sgd_1b_s01.sh", "notes": "Single LR; momentum 0.95."},
    "lora_rite": {"lr": 0.00008,
                  "flags": ["--grad-clip-norm", "1", "--max-delta-norm", "0.1",
                            "--use-momentum", "--momentum-cf", "0.95", "--ball-iters", "5",
                            "--ball-balance", "--ball-balance-mode", "norm", "--ball-rebalance-every", "1"],
                  "source": "lr_sweep_lora_rite_loi.sh",
                  "notes": "Preserves existing generation pilot, including global clip1; original BoolQ global clipping disabled."},
    "riemannion_original": {"lr": 0.00003, "flags": [], "source": "riemannion_loi.sh",
                            "notes": "Fixed authors' tangent gradients and previous-point momentum; exact QR/SVD, not paper NS6."},
    "lora_pro": {"lr": 0.0005,
                 "flags": ["--weight-decay", "0.1", "--adam-eps", "1e-6",
                           "--lora-pro-eps-inv", "1e-6", "--max-delta-norm", "0.1"],
                 "source": "lr_sweep_lora_pro_loi.sh",
                 "notes": "Historical PEFT init_lora_weights=None initializes both factors randomly, changing initial model; needed by historical G_W reconstruction, which freezes at B=0. No RSVD; single LR."},
    "lora_muon_manifold": {"lr": 0.003,
                           "flags": ["--weight-decay", "0.1", "--grad-clip-norm", "1",
                                     "--max-delta-norm", "0.1", "--lmm-momentum", "0.95",
                                     "--lmm-ns-steps", "8", "--lmm-inv-sqrt-steps", "7", "--lmm-eps", "1e-5"],
                           "source": "lr_sweep_lora_muon_manifold_loi_seed42.sh",
                           "notes": "BoolQ selected shared eta0.003; approximate inverse square root."},
    "lora_tsd": {"lr": 0.005,
                 "flags": ["--grad-clip-norm", "1", "--max-delta-norm", "0.1",
                           "--use-momentum", "--momentum-cf", "0.95", "--ball-iters", "5",
                           "--ball-balance", "--ball-balance-mode", "norm", "--ball-rebalance-every", "1"],
                 "source": "lora_tsd_balliters_llama1b.sh + paper selected tau5",
                 "notes": "Preserves existing generation pilot exactly."},
    "muon_full": {"lr": 0.03, "flags": [], "source": "muon_full_1b_s01.sh",
                  "notes": "Full q/k/v weight tuning, no adapters/RSVD; generation uses3000 versus historical2500 steps."},
}


def generation_config(method: str, dataset: str, steps: int, output_dir: str,
                      microbatch: int, max_length: int) -> dict:
    preset = PRESETS[method]
    lr = preset["lr"]
    flags = ["--run-name", output_dir.rstrip("/").split("/")[-1], "--optimizer", method,
             "--model", "meta-llama/Llama-3.2-1B-Instruct", "--dataset", dataset,
             "--steps", str(steps), "--seed", "42", "--output-dir", output_dir,
             "--lr-a", str(lr), "--lr-b", str(lr), "--checkpoint-every", "0",
             "--train-batch-size", str(microbatch), "--grad-accum-steps", str(32 // microbatch),
             "--eval-batch-size", "4", "--eval-every", "250", "--grad-checkpointing",
             "--muon-ns-steps", "5", "--weight-decay", "0"]
    if method not in {"lora_pro", "muon_full"}:
        flags += ["--rsvd-init", "--rsvd-q", "15", "--rsvd-scale", "0"]
    config = parse_config(flags + preset["flags"])
    config["lora_init"] = None if method == "lora_pro" else True
    torch_schedule = method in {"adamw", "riemannion_original"}
    config.update(max_sequence_length=max_length, effective_batch=32,
                  actual_schedule=("torch linear warmup10pct decay to1e-5; scheduler after optimizer"
                                   if torch_schedule else
                                   "linear warmup10pct decay to1e-5, applied before each update"),
                  schedule_normalization=("corrected original torch schedule" if torch_schedule else
                                          "common generation schedule; historical flags may have been ignored or absent"),
                  boolq_preset_source=preset["source"], optimizer_notes=preset["notes"],
                  initialization=("full q/k/v weights, no adapters" if method == "muon_full" else
                                  "historical PEFT init_lora_weights=None; both factors random, initial model changed; no RSVD" if method == "lora_pro" else
                                  "LOI RSVD q15 scale0"),
                  historical_schedule=("constant, scheduler flag absent" if method in {"lora_rite", "muon_full"} else
                                       "linear10pct warmup to1e-5" if method in {"adamw", "riemannion_original", "lora_tsd"} else
                                       "linear requested but unused by historical optimizer factory"),
                  checkpoints=False, save_final_model=False)
    return config


def enable_fixed_original_tangent(model, method: str) -> None:
    """Call only after RSVD, which relies on the initialization forward path."""
    if method == "riemannion_original":
        for module in model.modules():
            if isinstance(module, RiemannianLoRALinear):
                module.tangent_forward = True


def generation_optimizer(groups, config: dict) -> torch.optim.Optimizer:
    if config["optimizer"] != "adamw":
        return build_optimizer(groups, config)
    adamw_groups = []
    for group in groups:
        A, B = group["params"]
        adamw_groups.extend([{"params": [A], "lr": config["lr_A"], "lora_factor": "A"},
                              {"params": [B], "lr": config["lr_B"], "lora_factor": "B"}])
    return torch.optim.AdamW(adamw_groups,
                            betas=(config["adam_beta1"], config["adam_beta2"]),
                            eps=config["adam_eps"], weight_decay=config["weight_decay"])


def optimizer_parameters(optimizer: torch.optim.Optimizer) -> list:
    return [parameter for group in optimizer.param_groups for parameter in group["params"]]


def gradient_norm(parameters: list, clip_norm: float) -> torch.Tensor:
    # Zero means no clipping, while retaining the mandatory nonfinite check.
    limit = clip_norm if clip_norm > 0 else math.inf
    return torch.nn.utils.clip_grad_norm_(parameters, limit, error_if_nonfinite=True)


class GenerationSchedule:
    def __init__(self, optimizer: torch.optim.Optimizer, config: dict):
        self.optimizer = optimizer
        self.config = config
        self.torch_schedule = None
        if config["optimizer"] in {"adamw", "riemannion_original"}:
            self.torch_schedule = make_torch_linear_warmup_decay(
                optimizer, total_steps=config["steps"], warmup_frac=0.1, min_lr=1e-5)
        self.manual = {
            key: make_linear_warmup_decay(config[key], config["steps"], warmup_frac=0.1, min_lr=1e-5)
            for key in ("lr_A", "lr_B")
        }

    def before_step(self, step: int) -> float:
        if self.torch_schedule is None:
            for group in self.optimizer.param_groups:
                if "lr_A" in group:
                    for key in ("lr_A", "lr_B"):
                        group[key] = self.manual[key](step, self.config[key], 0)
                else:
                    group["lr"] = self.manual["lr_A"](step, self.config["lr_A"], 0)
        first = self.optimizer.param_groups[0]
        return float(first.get("lr", first.get("lr_A")))

    def after_step(self) -> None:
        if self.torch_schedule is not None:
            self.torch_schedule.step()
