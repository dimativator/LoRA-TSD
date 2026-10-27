"""Controlled full-step throughput benchmark, with fresh processes per case."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts/generation"))

import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, LlamaConfig, LlamaForCausalLM

from disk_guard import DiskGuard
from models import (RiemannianLoRALinear, apply_riemannian_lora,
                    get_lora_param_groups, get_riemannian_param_groups)
from optimizers.lora_muon import LoRAMuon
from optimizers.lora_muon_manifold import LoRAMuonManifold
from optimizers.riemannion_original import RiemannionOriginal
from tsd.lora_tsd_efficient import LoRATSDEfficient

METHODS = ("adamw", "muon_on_lora", "riemannion_original", "lora_muon",
           "tsd_tau1", "tsd_tau3", "tsd_tau5")
RANKS = (1, 2, 4, 8, 16, 32)
BATCHES = (1, 2, 4, 8, 16, 32, 64, 128)


def grid(shard: int, shards: int = 4) -> list[dict]:
    cases = []
    for ri, rank in enumerate(RANKS):
        for bi, batch in enumerate(BATCHES):
            if (ri + bi) % shards != shard:
                continue
            # Rotate method order to avoid always timing TSD last.
            offset = (ri + bi) % len(METHODS)
            order = METHODS[offset:] + METHODS[:offset]
            for method in order:
                cases.append({"method": method, "rank": rank, "batch": batch})
    return cases


def case_name(case: dict) -> str:
    return f"{case['method']}_r{case['rank']}_b{case['batch']}"


@torch.no_grad()
def initialize_adapters(model, rank: int, original: bool) -> list:
    """Give every method the same nonzero rank-r initial weight perturbation."""
    if original:
        model = apply_riemannian_lora(model, rank, rank, .05)
        groups = get_riemannian_param_groups(model)
    else:
        for module in model.modules():
            if hasattr(module, "lora_A") and hasattr(module, "lora_B"):
                for key in module.lora_A:
                    module.lora_A[key].weight.data = module.lora_A[key].weight.float()
                    module.lora_B[key].weight.data = module.lora_B[key].weight.float()
        groups = get_lora_param_groups(model)
    # CPU generator makes the shared initialization independent of adapter setup.
    generator = torch.Generator().manual_seed(42)
    for group in groups:
        A, B = group["params"]
        m, n = (A.shape[1], B.shape[0]) if original else (B.shape[0], A.shape[1])
        U = torch.linalg.qr(torch.randn(m, rank, generator=generator)).Q.to(A.device)
        V = torch.linalg.qr(torch.randn(n, rank, generator=generator)).Q.to(A.device)
        if original:
            A.zero_()
            B.zero_()
            A[:rank].copy_(U.T)
            B[:, :rank].copy_(1e-4 * V)
            B[:, rank:].copy_(V)
        else:
            A.copy_(.01 * V.T)
            B.copy_(.01 * U)
        group["log_optimizer_stats"] = False
    if original:
        for module in model.modules():
            if isinstance(module, RiemannianLoRALinear):
                module.tangent_forward = True
    return groups


def build(case: dict, model_path: str | None, device: str = "cuda") -> tuple:
    torch.manual_seed(42)
    if model_path is None:
        base = LlamaForCausalLM(LlamaConfig(
            vocab_size=256, hidden_size=128, intermediate_size=256,
            num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
            max_position_embeddings=512, use_cache=False,
        )).to(device)
    else:
        base = AutoModelForCausalLM.from_pretrained(
            model_path, torch_dtype=torch.bfloat16, attn_implementation="sdpa",
            local_files_only=True, low_cpu_mem_usage=True,
        ).to(device)
    base.config.use_cache = False
    model = get_peft_model(base, LoraConfig(
        r=case["rank"], lora_alpha=case["rank"], lora_dropout=.05,
        target_modules=["q_proj", "k_proj", "v_proj"],
        task_type="CAUSAL_LM", init_lora_weights=True,
    ))
    groups = initialize_adapters(model, case["rank"], case["method"] == "riemannion_original")
    params = [p for group in groups for p in group["params"]]
    method = case["method"]
    if method == "adamw":
        optimizer = torch.optim.AdamW(params, lr=.001, betas=(.9, .999),
                                     eps=1e-8, weight_decay=.01, fused=device == "cuda")
    elif method == "muon_on_lora":
        optimizer = LoRAMuon(groups, lr_A=.01, lr_B=.01, momentum=.95,
                             nesterov=True, ns_steps=5, max_delta_norm=.01,
                             log_optimizer_stats=False)
    elif method == "riemannion_original":
        optimizer = RiemannionOriginal(groups, lr=3e-5, momentum=.95,
                                       nesterov=False, weight_decay=0., retraction="svd")
    elif method == "lora_muon":
        optimizer = LoRAMuonManifold(groups, lr=.003, momentum=.95, weight_decay=.1,
                                     ns_steps=8, inv_sqrt_steps=7, eps=1e-5,
                                     max_delta_norm=.1, log_optimizer_stats=False)
    else:
        optimizer = LoRATSDEfficient(
            groups, lr_A=.005, lr_B=.005, ball_iters=int(method[-1]), ns_steps=5,
            use_momentum=True, momentum_cf=.95, max_delta_norm=.1,
            balance=True, balance_mode="norm", rebalance_every=1,
            log_optimizer_stats=False,
        )
    model.train()
    return model, optimizer, params


def step(model, optimizer, params: list, batch: dict) -> torch.Tensor:
    device_batch = {key: value.to(next(model.parameters()).device, non_blocking=True)
                    for key, value in batch.items()}
    loss = model(**device_batch).loss
    loss.backward()
    torch.nn.utils.clip_grad_norm_(params, 1., foreach=True)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    return loss.detach()


def stats(values: list[float]) -> dict:
    return {"mean": statistics.mean(values), "median": statistics.median(values),
            "min": min(values), "max": max(values),
            "stdev": statistics.stdev(values) if len(values) > 1 else 0., "samples": values}


def measure(args: argparse.Namespace) -> dict:
    case = {key: getattr(args, key) for key in ("method", "rank", "batch")}
    torch.set_num_threads(8)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    model, optimizer, params = build(case, args.model_path)
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.enable_input_require_grads()
    gen = torch.Generator().manual_seed(43)
    ids = torch.randint(4, model.config.vocab_size, (args.batch, args.length), generator=gen)
    batch = {"input_ids": ids.pin_memory(), "labels": ids.clone().pin_memory(),
             "attention_mask": torch.ones_like(ids).pin_memory()}
    for _ in range(args.warmup):
        loss = step(model, optimizer, params, batch)
    torch.cuda.synchronize()
    if not torch.isfinite(loss).item():
        raise RuntimeError("Nonfinite warmup loss")
    torch.cuda.reset_peak_memory_stats()
    windows = []
    for window in range(args.windows):
        torch.cuda.synchronize()
        started = time.perf_counter()
        for _ in range(args.steps_per_window):
            loss = step(model, optimizer, params, batch)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
        if not torch.isfinite(loss).item():
            raise RuntimeError("Nonfinite measured loss")
        windows.append(elapsed / args.steps_per_window)
        print("SPEED_WINDOW " + json.dumps({**case, "window": window,
                                             "seconds_per_step": windows[-1]}), flush=True)
    peak_allocated = torch.cuda.max_memory_allocated()
    peak_reserved = torch.cuda.max_memory_reserved()
    # Separate diagnostic pass avoids synchronization/event overhead in the
    # main full-step measurement. It includes the optimizer's CPU launch cost.
    optimizer_seconds = []
    for _ in range(5):
        device_batch = {key: value.to("cuda", non_blocking=True) for key, value in batch.items()}
        model(**device_batch).loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1., foreach=True)
        torch.cuda.synchronize()
        started = time.perf_counter()
        optimizer.step()
        torch.cuda.synchronize()
        optimizer_seconds.append(time.perf_counter() - started)
        optimizer.zero_grad(set_to_none=True)
    hardware = subprocess.run(["nvidia-smi", "--query-gpu=uuid,name,memory.total,driver_version",
                               "--format=csv,noheader"], capture_output=True, text=True, check=True).stdout.strip()
    properties = torch.cuda.get_device_properties(0)
    return {**case, "status": "completed", "full_step_seconds": stats(windows),
            "optimizer_seconds": stats(optimizer_seconds),
            "tokens_per_second": args.batch * args.length / statistics.mean(windows),
            "supervised_tokens_per_second": args.batch * (args.length - 1) / statistics.mean(windows),
            "peak_allocated_bytes": peak_allocated, "peak_reserved_bytes": peak_reserved,
            "final_loss": loss.item(), "trainable_parameters": sum(p.numel() for p in params),
            "gpu": hardware, "torch": torch.__version__, "cuda": torch.version.cuda,
            "allocated_gpu": {"name": properties.name, "total_memory": properties.total_memory,
                              "uuid": str(getattr(properties, "uuid", "unavailable"))},
            "optimizer_class": type(optimizer).__name__}


def smoke() -> dict:
    torch.set_num_threads(2)
    records = []
    for rank in (1, 32):
        for method in METHODS:
            case = {"method": method, "rank": rank, "batch": 2}
            model, optimizer, params = build(case, None, "cpu")
            ids = torch.randint(4, 256, (2, 16))
            for iteration in range(2):
                loss = model(input_ids=ids, labels=ids).loss
                loss.backward()
                if method == "riemannion_original" and iteration == 0:
                    auxiliary = [module.lora_A.weight.grad[rank:].norm() for module in model.modules()
                                 if isinstance(module, RiemannianLoRALinear)]
                    assert auxiliary and all(value > 0 for value in auxiliary)
                torch.nn.utils.clip_grad_norm_(params, 1., foreach=True)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            if not torch.isfinite(loss).item() or not all(torch.isfinite(p).all() for p in params):
                raise RuntimeError("Nonfinite CPU smoke: " + method)
            if method == "riemannion_original":
                for module in model.modules():
                    if isinstance(module, RiemannianLoRALinear):
                        assert module.tangent_forward
                        assert module.lora_A.weight.grad is None
            records.append({**case, "loss": loss.item(), "optimizer": type(optimizer).__name__})
    return {"status": "passed", "cases": records}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--method", choices=METHODS)
    parser.add_argument("--rank", type=int)
    parser.add_argument("--batch", type=int)
    parser.add_argument("--model-path")
    parser.add_argument("--output")
    parser.add_argument("--length", type=int, default=512)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--windows", type=int, default=3)
    parser.add_argument("--steps-per-window", type=int, default=10)
    parser.add_argument("--gradient-checkpointing", action="store_true")
    args = parser.parse_args()
    if args.smoke:
        print("SPEED_SMOKE " + json.dumps(smoke()), flush=True)
        return
    started = time.time()
    try:
        result = measure(args)
    except torch.cuda.OutOfMemoryError as error:
        result = {"status": "oom", "error": str(error)}
    except Exception as error:
        traceback.print_exc()
        result = {"status": "failed", "error": repr(error)}
    result.update(method=args.method, rank=args.rank, batch=args.batch, length=args.length,
                  warmup=args.warmup, windows=args.windows, steps_per_window=args.steps_per_window,
                  gradient_checkpointing=args.gradient_checkpointing, started_at=started,
                  ended_at=time.time(), pid=os.getpid(), seed=42,
                  model="meta-llama/Llama-3.2-1B-Instruct", base_dtype="bfloat16",
                  adapter_dtype="float32", attention="sdpa", tf32=False,
                  gradient_accumulation=1, global_grad_clip=1., lora_alpha=args.rank,
                  dropout=.05, targets=["q_proj", "k_proj", "v_proj"], checkpoints=False,
                  initialization="same nonzero rank-r perturbation: 1e-4 U V^T; initialization excluded",
                  measurement="3 synchronized windows of10 full steps; H2D/forward/backward/clip/update/zero",
                  optimizer_measurement="separate five-step diagnostic pass, synchronized around optimizer.step",
                  source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    destination = Path(args.output)
    guard = DiskGuard(destination.parent.parent, destination.stem)
    guard(pending_write_bytes=1_000_000, reason="before_result", write_path=destination.parent)
    destination.write_text(json.dumps(result, indent=2) + "\n")
    print("SPEED_RESULT " + json.dumps(result), flush=True)
    if result["status"] == "failed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
