import os
import random
from pathlib import Path

import numpy as np
import torch


def checkpoint_path(config):
    if config.get("resume_from"):
        return Path(config["resume_from"]).expanduser()
    if not config.get("checkpoint_dir"):
        return None
    return Path(config["checkpoint_dir"]).expanduser() / "latest.pt"


def checkpoint_exists(config) -> bool:
    path = checkpoint_path(config)
    return path is not None and path.is_file()


def _trainable_parameter_state(model):
    return {
        name: parameter.detach().cpu()
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }


def _rng_state():
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def _restore_rng_state(state):
    if not state:
        return
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if torch.cuda.is_available() and "cuda" in state:
        torch.cuda.set_rng_state_all(state["cuda"])


def save_training_checkpoint(
    config,
    model,
    optimizer,
    global_step,
    epoch,
    next_batch_index,
    epoch_generator_state,
    losses,
    validation_scores,
    optimizer_lr_scheduler=None,
):
    path = checkpoint_path(config)
    if path is None:
        return None
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(path.name + ".tmp")
    state = {
        "version": 1,
        "global_step": int(global_step),
        "epoch": int(epoch),
        "next_batch_index": int(next_batch_index),
        "epoch_generator_state": epoch_generator_state,
        "model_trainable_state": _trainable_parameter_state(model),
        "optimizer_state": optimizer.state_dict(),
        "optimizer_lr_scheduler": optimizer_lr_scheduler.state_dict() if optimizer_lr_scheduler is not None else None,
        "optimizer_attrs": {
            name: getattr(optimizer, name)
            for name in ("_sched_step",)
            if hasattr(optimizer, name)
        },
        "losses": list(losses),
        "validation_scores": list(validation_scores),
        "rng_state": _rng_state(),
    }
    torch.save(state, temporary_path)
    os.replace(temporary_path, path)
    print(f"[checkpoint] saved step {global_step} to {path}", flush=True)
    return path


def load_training_checkpoint(config, model, optimizer, optimizer_lr_scheduler=None):
    path = checkpoint_path(config)
    if path is None or not path.is_file():
        return None
    state = torch.load(path, map_location="cpu", weights_only=False)
    if state.get("version") != 1:
        raise ValueError(f"Unsupported checkpoint version in {path}")

    named_parameters = dict(model.named_parameters())
    with torch.no_grad():
        for name, saved in state["model_trainable_state"].items():
            if name not in named_parameters:
                raise ValueError(f"Checkpoint parameter absent from model: {name}")
            parameter = named_parameters[name]
            if tuple(parameter.shape) != tuple(saved.shape):
                raise ValueError(
                    f"Checkpoint shape mismatch for {name}: "
                    f"model={tuple(parameter.shape)} checkpoint={tuple(saved.shape)}"
                )
            parameter.copy_(saved.to(parameter.device, parameter.dtype))

    optimizer.load_state_dict(state["optimizer_state"])
    if optimizer_lr_scheduler is not None and state.get("optimizer_lr_scheduler") is not None:
        optimizer_lr_scheduler.load_state_dict(state["optimizer_lr_scheduler"])
    for name, value in state.get("optimizer_attrs", {}).items():
        setattr(optimizer, name, value)
    _restore_rng_state(state.get("rng_state"))
    print(
        f"[checkpoint] resumed step {state['global_step']} from {path}",
        flush=True,
    )
    return state
