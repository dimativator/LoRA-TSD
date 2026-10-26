"""Generation pilots using the paper's model, initialization, and optimizers."""

import argparse
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import torch
from args_parser import parse_config
from main import build_model, initialize_subspace, set_seed
from optimizers import build_optimizer, make_linear_warmup_decay
from disk_guard import DiskGuard
from generation_data import build_data
from generation_eval import check_evaluation_dependencies, evaluate


def append_metric(path: Path, row: dict) -> None:
    row = {"timestamp": time.time(), **row}
    with path.open("a") as stream:
        stream.write(json.dumps(row, allow_nan=False) + "\n")
    print("GEN_METRIC " + json.dumps(row, allow_nan=False), flush=True)


def next_batches(loader):
    while True:
        for batch in loader:
            yield batch


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["e2e_nlg", "smol_smoltalk_100k"], required=True)
    parser.add_argument("--method", choices=["lora_tsd", "lora_rite"], required=True)
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    run = args.output_dir
    root = Path(os.environ["GENERATION_ROOT"])
    guard = DiskGuard(root, run.name, run / "disk-trainer.jsonl")
    guard(pending_write_bytes=3_000_000_000, reason="model_cache_reserve",
          write_path=Path(os.environ["HF_HOME"]))
    microbatch = 4 if args.dataset == "e2e_nlg" else 2
    max_length = 512 if args.dataset == "e2e_nlg" else 2048
    lr = 0.005 if args.method == "lora_tsd" else 0.00008
    flags = ["--run-name", run.name, "--optimizer", args.method,
             "--model", "meta-llama/Llama-3.2-1B-Instruct", "--dataset", args.dataset,
             "--steps", str(args.steps), "--seed", "42", "--output-dir", str(run),
             "--lr-a", str(lr), "--lr-b", str(lr), "--checkpoint-every", "0",
             "--train-batch-size", str(microbatch), "--grad-accum-steps", str(32 // microbatch),
             "--eval-batch-size", "4", "--eval-every", "250", "--grad-checkpointing",
             "--rsvd-init", "--rsvd-q", "15", "--rsvd-scale", "0",
             "--grad-clip-norm", "1", "--max-delta-norm", "0.1",
             "--use-momentum", "--momentum-cf", "0.95", "--ball-iters", "5",
             "--ball-balance", "--ball-balance-mode", "norm", "--ball-rebalance-every", "1",
             "--muon-ns-steps", "5", "--weight-decay", "0"]
    config = parse_config(flags)
    config["lora_init"] = True
    config.update(max_sequence_length=max_length, effective_batch=32,
                  actual_schedule="linear warmup10pct decay to1e-5, applied before each update",
                  checkpoints=False, save_final_model=False)
    (run / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    gpu = subprocess.run(["nvidia-smi", "--query-gpu=index,name,memory.used,memory.free",
                          "--format=csv,noheader"], capture_output=True, text=True, timeout=15)
    processes = subprocess.run(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,used_memory",
                                "--format=csv,noheader"], capture_output=True, text=True, timeout=15)
    print("GEN_GPU_BEFORE_TRAIN " + json.dumps({"gpus": gpu.stdout, "processes": processes.stdout}), flush=True)
    if torch.cuda.device_count() != 1:
        raise RuntimeError("Exactly one visible GPU is required")
    set_seed(42)
    model, tokenizer, groups = build_model(config)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    model.config.use_cache = False
    guard(reason="model_loaded")
    data_dir = Path(os.environ.get("GENERATION_DATA_DIR", "/tmp/generation-data"))
    data_dir.mkdir(parents=True, exist_ok=True)
    def data_guard(pending_write_bytes: int = 0, reason: str = "periodic") -> dict:
        return guard(pending_write_bytes, reason, write_path=data_dir)
    train_loader, val_loader, generation_examples, manifest = build_data(
        tokenizer, args.dataset, 42, max_length, microbatch, data_dir, data_guard)
    (run / "data-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    check_evaluation_dependencies(generation_examples, data_guard)
    initialize_subspace(config, model, train_loader)
    set_seed(42)
    if train_loader.generator is not None:
        train_loader.generator.manual_seed(42)
    model.zero_grad(set_to_none=True)
    optimizer = build_optimizer(groups, config)
    parameters = [parameter for group in groups for parameter in group["params"]]
    schedule = make_linear_warmup_decay(lr, args.steps, warmup_frac=0.1, min_lr=1e-5)
    history = run / "history.jsonl"

    def validate(step: int) -> None:
        guard(reason="before_validation")
        begun = time.monotonic()
        model.gradient_checkpointing_disable()
        model.config.use_cache = True
        try:
            def eval_guard(pending_write_bytes: int = 0, reason: str = "periodic") -> dict:
                target = data_dir if reason == "ifeval_nltk_data" else run
                return guard(pending_write_bytes, reason, write_path=target)
            result = evaluate(model, tokenizer, args.dataset, val_loader,
                              generation_examples, run, step, eval_guard)
            append_metric(history, {"event": "validation", "step": step, **result,
                                    "validation_seconds": time.monotonic() - begun})
        finally:
            model.config.use_cache = False
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
            model.train()

    validate(0)
    iterator = next_batches(train_loader)
    training_started = time.monotonic()
    for step in range(1, args.steps + 1):
        guard(reason="before_training_update")
        begun = time.monotonic()
        batches = [next(iterator) for _ in range(config["grad_accum_steps"])]
        counts = [int((batch["labels"][:, 1:] != -100).sum()) for batch in batches]
        tokens = sum(counts)
        if tokens <= 0:
            raise RuntimeError("No supervised tokens in optimizer update")
        optimizer.zero_grad(set_to_none=True)
        accumulated_loss = 0.0
        model.train()
        for batch, count in zip(batches, counts):
            if count <= 0:
                continue
            batch = {key: value.to("cuda") for key, value in batch.items()}
            loss = model(**batch).loss
            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite loss at step {step}")
            weight = count / tokens
            (loss * weight).backward()
            accumulated_loss += float(loss.detach()) * weight
        norm = torch.nn.utils.clip_grad_norm_(parameters, config["grad_clip_norm"], error_if_nonfinite=True)
        used_lr = schedule(step, lr, 0)
        for group in optimizer.param_groups:
            if args.method == "lora_tsd":
                group["lr_A"] = used_lr
                group["lr_B"] = used_lr
            else:
                group["lr"] = used_lr
        optimizer.step()
        if not math.isfinite(accumulated_loss):
            raise RuntimeError("Non-finite aggregate loss")
        append_metric(history, {"event": "train", "step": step, "train/loss": accumulated_loss,
                                "lr_used": used_lr, "supervised_tokens": tokens,
                                "grad_norm": float(norm), "step_seconds": time.monotonic() - begun,
                                "elapsed_seconds": time.monotonic() - training_started,
                                "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                                "cuda_peak_reserved_bytes": torch.cuda.max_memory_reserved()})
        if step % config["eval_every"] == 0 or step == args.steps:
            validate(step)
    guard(reason="before_completion_log")
    (run / "completed.json").write_text(json.dumps({"status": "completed", "steps": args.steps,
                                                   "timestamp": time.time(), "checkpoints": False}) + "\n")


if __name__ == "__main__":
    main()
