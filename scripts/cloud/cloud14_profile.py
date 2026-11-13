"""Include padded attention and RSVD initialization peaks in the disposable memory probe."""

import json
import os
from pathlib import Path
import sys
import time

from cloud14_core import STATE, task_key


def synthetic_batch(torch, microbatch: int, pad_token_id: int, device: str) -> dict:
    ids = torch.randint(32, 1000, (microbatch, 1024), device=device)
    mask = torch.ones_like(ids)
    labels = torch.full_like(ids, -100)
    for row in range(microbatch):
        padding = row % 128 if microbatch > 1 else 1
        if padding:
            mask[row, -padding:] = 0
            ids[row, -padding:] = pad_token_id
        end = 1024 - padding
        labels[row, end - 8:end] = ids[row, end - 8:end]
    return {"input_ids": ids, "attention_mask": mask, "labels": labels}


def main(args, frozen: Path, materialized: dict) -> None:
    key = task_key({"method": materialized["method"]["id"], "dataset": materialized["dataset"],
                    "seed": materialized["seed"]})
    state = json.loads((STATE / "tasks" / (key + ".json")).read_text())
    worker = int(args.task.parent.name.split("-", 1)[0].removeprefix("cloud"))
    assert state["worker"] == worker and state["pid"] == os.getppid()
    assert state["hostname"] == os.uname().nodename and state["phase"] == "profiling"
    sys.path.insert(0, str(frozen / "src"))
    sys.path.insert(0, str(frozen / "scripts/brain_lab"))
    import torch
    from args_parser import parse_config
    from main import build_model, initialize_subspace, set_seed
    from models import RiemannianLoRALinear
    from optimizers import build_optimizer
    import paper_8b_campaign
    set_seed(materialized["seed"])
    if materialized["method"]["optimizer"] == "lora_rite":
        torch.linalg.qr(torch.eye(2, dtype=torch.float32, device="cuda"))
        torch.cuda.synchronize()
    free_before, total = torch.cuda.mem_get_info()
    torch.cuda.reset_peak_memory_stats()
    config = parse_config(paper_8b_campaign.arguments(materialized, args.microbatch, args.task.parent))
    model, tokenizer, groups = build_model(config)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    model.config.use_cache = False
    model.train()
    batch = synthetic_batch(torch, args.microbatch, tokenizer.pad_token_id, "cuda")
    started = time.monotonic()
    # Real initialization is exercised only in this disposable profiling process.
    # The training child still reloads the frozen model and initializes from real data.
    initialize_subspace(config, model, [batch] * (2 * config["rsvd_q"] + 1))
    for module in model.modules():
        if isinstance(module, RiemannianLoRALinear):
            with torch.no_grad():
                right = .01 * torch.randn_like(module.lora_B.weight[:, :module.r])
                module.lora_B.weight[:, :module.r].copy_(right)
                module.lora_B.weight[:, module.r:].copy_(torch.linalg.qr(right).Q)
            if config["optimizer"] == "riemannion_original":
                module.tangent_forward = True
        elif hasattr(module, "lora_B"):
            with torch.no_grad():
                for layer in module.lora_B.values():
                    layer.weight.normal_(0, .01)
    optimizer = build_optimizer(groups, config)
    for _ in range(2):
        optimizer.zero_grad(set_to_none=True)
        for _ in range(config["grad_accum_steps"]):
            loss = model(**batch).loss
            if not torch.isfinite(loss):
                raise RuntimeError("Nonfinite profiling loss")
            (loss / config["grad_accum_steps"]).backward()
        if config["grad_clip_norm"] > 0:
            parameters = [parameter for group in optimizer.param_groups for parameter in group["params"]]
            torch.nn.utils.clip_grad_norm_(parameters, config["grad_clip_norm"])
        optimizer.step()
        torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated()
    result = {"microbatch": args.microbatch, "sequence_length": 1024,
              "includes_rsvd_initialization": bool(config["rsvd_init"]), "padded_attention": True,
              "peak_allocated_bytes": peak, "free_before_bytes": free_before,
              "total_bytes": total, "fits": peak < .96 * free_before,
              "seconds": time.monotonic() - started}
    print(json.dumps(result), flush=True)
    if not result["fits"]:
        raise SystemExit(43)
