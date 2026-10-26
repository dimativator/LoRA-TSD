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
from main import build_model, initialize_subspace, set_seed
from disk_guard import DiskGuard
from generation_data import build_data
from generation_eval import check_evaluation_dependencies, evaluate
from optimizer_presets import (METHODS, GenerationSchedule, enable_fixed_original_tangent,
                               generation_config, generation_optimizer, gradient_norm,
                               optimizer_parameters)


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
    parser.add_argument("--method", choices=METHODS, required=True)
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
    config = generation_config(args.method, args.dataset, args.steps, str(run), microbatch, max_length)
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
    enable_fixed_original_tangent(model, args.method)
    set_seed(42)
    if train_loader.generator is not None:
        train_loader.generator.manual_seed(42)
    model.zero_grad(set_to_none=True)
    optimizer = generation_optimizer(groups, config)
    parameters = optimizer_parameters(optimizer)
    schedule = GenerationSchedule(optimizer, config)
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
        norm = gradient_norm(parameters, config["grad_clip_norm"])
        used_lr = schedule.before_step(step)
        optimizer.step()
        schedule.after_step()
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
