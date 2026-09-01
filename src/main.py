import json
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
from peft import LoraConfig

import models
from args_parser import parse_config
from data import get_dataset
from optimizers import build_optimizer, make_linear_warmup_decay
from run import checkpoint_exists, train


RIEMANNIAN_OPTIMIZERS = {"riemannian_sgd", "riemannion"}
FULL_FINETUNE_OPTIMIZERS = {"muon_full"}


def set_seed(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)


def build_model(config):
    optimizer_name = config["optimizer"]
    if optimizer_name in FULL_FINETUNE_OPTIMIZERS:
        model, tokenizer = models.get_full_finetune_model(
            config["model"],
            config["lora_target_modules"],
        )
        groups = models.get_muon_full_param_groups(
            model,
            config["lora_target_modules"],
        )
        return model, tokenizer, groups

    lora_config = LoraConfig(
        r=config["lora_r"],
        lora_alpha=config["lora_alpha"],
        target_modules=config["lora_target_modules"],
        lora_dropout=config["lora_dropout"],
        init_lora_weights=config["lora_init"],
        bias=config["lora_bias"],
        task_type="CAUSAL_LM",
    )
    model, tokenizer = models.get_lora_model(
        config["model"],
        lora_config,
        to_float32=True,
        add_lora=True,
    )

    if optimizer_name in RIEMANNIAN_OPTIMIZERS:
        model = models.apply_riemannian_lora(
            model,
            r=config["lora_r"],
            lora_alpha=config["lora_alpha"],
            lora_dropout=config["lora_dropout"],
        )
        groups = models.get_riemannian_param_groups(model)
    else:
        groups = models.get_lora_param_groups(model, add_W=False)

    return model, tokenizer, groups


def configure_schedules(config) -> None:
    if config["lr_scheduler"] != "linear":
        return
    config["lr_A_scheduler"] = make_linear_warmup_decay(
        config["lr_A"],
        config["steps"],
        warmup_frac=0.1,
        min_lr=1e-5,
    )
    config["lr_B_scheduler"] = make_linear_warmup_decay(
        config["lr_B"],
        config["steps"],
        warmup_frac=0.1,
        min_lr=1e-5,
    )


def initialize_subspace(config, model, dataloader) -> None:
    if (
        not config["rsvd_init"]
        or config["optimizer"] in FULL_FINETUNE_OPTIMIZERS
        or checkpoint_exists(config)
    ):
        return
    if config["optimizer"] in RIEMANNIAN_OPTIMIZERS:
        models.rsvd_init(
            model,
            dataloader,
            q=config["rsvd_q"],
            device="cuda",
            scale=config["rsvd_scale"],
        )
    else:
        models.rsvd_init_standard_lora(
            model,
            dataloader,
            q=config["rsvd_q"],
            device="cuda",
            scale=config["rsvd_scale"],
        )


def main() -> None:
    config = parse_config()
    set_seed(config["seed"])
    model, tokenizer, parameter_groups = build_model(config)

    build_dataloader, evaluate = get_dataset(config["dataset"])
    data_config = {
        "seed": config["seed"],
        "train_batch_size": config["train_batch_size"],
    }
    dataloader = build_dataloader(tokenizer, data_config)
    initialize_subspace(config, model, dataloader)
    configure_schedules(config)
    optimizer = build_optimizer(parameter_groups, config)

    started = time.time()
    losses, validation_scores = train(
        config,
        model,
        optimizer,
        dataloader,
        tokenizer,
        evaluate,
    )
    elapsed = time.time() - started

    if hasattr(model, "gradient_checkpointing_disable"):
        model.gradient_checkpointing_disable()
    model.config.use_cache = True
    test_accuracy = evaluate(
        model,
        tokenizer,
        config.get("eval_batch_size")
        or max(config["train_batch_size"] // 2, 1),
    )

    best_validation_accuracy = max(
        (item["accuracy"] for item in validation_scores),
        default=None,
    )
    reported_accuracy = test_accuracy
    if best_validation_accuracy is not None:
        reported_accuracy = max(test_accuracy, best_validation_accuracy)

    result = {
        "run_name": config["run_name"],
        "optimizer": config["optimizer"],
        "model": config["model"],
        "dataset": config["dataset"],
        "seed": config["seed"],
        "steps": config["steps"],
        "best_validation_accuracy": best_validation_accuracy,
        "validation_scores": validation_scores,
        "test_accuracy": test_accuracy,
        "reported_accuracy": reported_accuracy,
        "mean_last_100_losses": (
            sum(losses[-100:]) / len(losses[-100:]) if losses else None
        ),
        "seconds_per_step": elapsed / config["steps"],
    }

    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"{config['run_name']}.json"
    output_file.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
