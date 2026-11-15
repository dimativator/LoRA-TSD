"""Disposable padded-memory probe of the repaired optimizer and actual initialization."""

import json
import os
from pathlib import Path
import sys
import time

from cloud_repair_common import STATE, TASKS, task_key


def synthetic_batch(torch, microbatch: int, pad_token_id: int, device: str) -> dict:
    ids = torch.randint(32, 1000, (microbatch, 1024), device=device)
    mask = torch.ones_like(ids)
    labels = torch.full_like(ids, -100)
    for row in range(microbatch):
        padding = row % 128 if microbatch > 1 else 1
        end = 1024 - padding
        if padding:
            mask[row, -padding:] = 0
            ids[row, -padding:] = pad_token_id
        labels[row, end - 8:end] = ids[row, end - 8:end]
    return {"input_ids": ids, "attention_mask": mask, "labels": labels}


def main(args, frozen: Path, materialized: dict) -> None:
    task = TASKS[args.worker_id]
    state = json.loads((STATE / "tasks" / (task_key(task) + ".json")).read_text())
    if (state["worker"] != args.worker_id or state["pid"] != os.getppid()
            or state["hostname"] != os.uname().nodename or state["phase"] != "profiling"):
        raise ValueError("Profile parent does not own this reserved task")
    sys.path.insert(0, str(frozen / "src"))
    import torch
    from args_parser import parse_config
    from main import build_model, initialize_subspace, set_seed
    from optimizers import build_optimizer
    from cloud_repair_runtime import arguments
    if torch.cuda.device_count() != 1:
        raise ValueError("Profile must see exactly one allocated GPU")
    set_seed(task["seed"])
    if task["optimizer"] == "lora_rite":
        torch.linalg.qr(torch.eye(2, dtype=torch.float32, device="cuda"))
        torch.cuda.synchronize()
    free_before, total = torch.cuda.mem_get_info()
    torch.cuda.reset_peak_memory_stats()
    config = parse_config(arguments(task, frozen, args.task.parent, args.microbatch))
    model, tokenizer, groups = build_model(config)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    model.config.use_cache = False
    model.train()
    batch = synthetic_batch(torch, args.microbatch, tokenizer.pad_token_id, "cuda")
    started = time.monotonic()
    initialize_subspace(config, model, [batch] * (2 * config["rsvd_q"] + 1))
    optimizer = build_optimizer(groups, config)
    for _ in range(2):
        optimizer.zero_grad(set_to_none=True)
        for _ in range(config["grad_accum_steps"]):
            loss = model(**batch).loss
            if not torch.isfinite(loss):
                raise RuntimeError("Nonfinite profiling loss")
            (loss / config["grad_accum_steps"]).backward()
        if config["grad_clip_norm"] > 0:
            parameters = [p for group in optimizer.param_groups for p in group["params"]]
            torch.nn.utils.clip_grad_norm_(parameters, config["grad_clip_norm"])
        optimizer.step()
        if not all(torch.isfinite(p).all() for group in optimizer.param_groups for p in group["params"]):
            raise RuntimeError("Nonfinite profiling factors")
        torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated()
    result = {"microbatch": args.microbatch, "sequence_length": 1024,
              "includes_rsvd_initialization": bool(config["rsvd_init"]),
              "initialization_batches": 2 * config["rsvd_q"] + 1 if config["rsvd_init"] else 0,
              "actual_initialization_preserved": True, "full_updates": 2,
              "padded_attention": True, "peak_allocated_bytes": peak,
              "free_before_bytes": free_before, "total_bytes": total,
              "fits": peak < .96 * free_before, "seconds": time.monotonic() - started}
    print(json.dumps(result), flush=True)
    if not result["fits"]:
        raise SystemExit(43)
