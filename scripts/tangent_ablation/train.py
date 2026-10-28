"""Train one controlled direction ablation without writing model checkpoints."""

import argparse
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/generation"))
sys.path.insert(0, str(ROOT / "src"))

import torch
from main import build_model, initialize_subspace, set_seed
from disk_guard import DiskGuard
from optimizer_presets import GenerationSchedule, generation_config, gradient_norm
from optimizer import TangentAblation

_data_spec = importlib.util.spec_from_file_location("tangent_data", Path(__file__).with_name("data.py"))
_data_module = importlib.util.module_from_spec(_data_spec)
_data_spec.loader.exec_module(_data_module)
build_data = _data_module.build_data


def smoke() -> dict:
    """Exercise the real collator, PEFT forward/backward and every direction on CPU."""
    from peft import LoraConfig, get_peft_model
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast
    from data.boolq import _example_to_prompt_and_gold
    from data.data_common import GenericSFTCollator
    from models import get_lora_param_groups

    backend = Tokenizer(WordLevel({"[UNK]": 0, "[PAD]": 1, "True": 2, "False": 3}, unk_token="[UNK]"))
    backend.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, unk_token="[UNK]", pad_token="[PAD]")
    collator = GenericSFTCollator(tokenizer, _example_to_prompt_and_gold, max_length=128)
    batch = collator([{"passage": "A small test passage.", "question": "Is this a test?", "answer": True},
                      {"passage": "Another small passage.", "question": "Is this false?", "answer": False}])
    cases = []
    for variant in ("frobenius", "one_step", "factorwise", "coupled"):
        torch.manual_seed(42)
        base = LlamaForCausalLM(LlamaConfig(
            vocab_size=8, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
            num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=256,
            use_cache=False,
        ))
        model = get_peft_model(base, LoraConfig(
            r=2, lora_alpha=2, target_modules=["q_proj", "k_proj", "v_proj"],
            task_type="CAUSAL_LM", lora_dropout=.05,
        ))
        groups = get_lora_param_groups(model)
        optimizer = TangentAblation(groups, variant=variant, lr=.005, ns_steps=5,
                                   use_momentum=True, momentum_cf=.95, balance=True,
                                   max_delta_norm=.1, log_optimizer_stats=False)
        losses = []
        for _ in range(2):
            optimizer.zero_grad(set_to_none=True)
            loss = model(**batch).loss
            if not torch.isfinite(loss):
                raise RuntimeError("Nonfinite tiny-model smoke loss")
            loss.backward()
            gradient_norm([p for g in groups for p in g["params"]], 1.)
            optimizer.step()
            losses.append(float(loss.detach()))
        cases.append({"variant": variant, "updates": 2, "losses": losses})
    return {"status": "passed", "cases": cases, "real_sft_collator": True}


def append_metric(path: Path, record: dict) -> None:
    record = {"timestamp": time.time(), **record}
    serialized = json.dumps(record, allow_nan=False)
    with path.open("a") as stream:
        stream.write(serialized + "\n")
    print("TANGENT_METRIC " + serialized, flush=True)


def next_batches(loader):
    while True:
        yield from loader


def adapter_digest(groups: list) -> str:
    digest = hashlib.sha256()
    for group in groups:
        for parameter in group["params"]:
            digest.update(parameter.detach().float().cpu().numpy().tobytes())
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=["frobenius", "one_step", "factorwise", "coupled"], required=True)
    parser.add_argument("--lr", type=float, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--dataset", choices=["boolq", "siqa"], required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--eval-every", type=int, default=250)
    parser.add_argument("--eval-max-samples", type=int)
    args = parser.parse_args()
    if args.lr <= 0 or args.steps <= 0 or args.eval_every <= 0:
        parser.error("LR, steps and evaluation interval must be positive")
    if args.eval_max_samples is not None and args.eval_max_samples <= 0:
        parser.error("--eval-max-samples must be positive")
    run = args.output_dir
    run.mkdir(parents=True, exist_ok=True)
    if (run / "history.jsonl").exists() or (run / "result.json").exists():
        raise FileExistsError("Refusing to overwrite an existing training run: " + str(run))
    campaign = Path(os.environ["ABLATION_ROOT"])
    guard = DiskGuard(campaign, run.name, run / "disk-trainer.jsonl")
    guard(pending_write_bytes=3_000_000_000, reason="before_model_load",
          write_path=Path(os.environ["HF_HOME"]))
    config = generation_config("lora_tsd", args.dataset, args.steps, str(run), 8, 1024)
    config.update(seed=args.seed, lr_A=args.lr, lr_B=args.lr, eval_every=args.eval_every,
                  variant=args.variant, eval_max_samples=args.eval_max_samples,
                  selection_metric="max validation accuracy at scheduled post-update evaluations",
                  selection_excludes_step0=True, eval_batch_size=8,
                  trainer="tangent_ablation/train.py", loss_weighting="equal microbatch mean, as benchmark",
                  max_sequence_length=1024, checkpoints=False, save_final_model=False)
    (run / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    gpu = subprocess.run(["nvidia-smi", "--query-gpu=index,uuid,name,memory.used,memory.free",
                          "--format=csv,noheader"], capture_output=True, text=True, timeout=15)
    processes = subprocess.run(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,used_memory",
                                "--format=csv,noheader"], capture_output=True, text=True, timeout=15)
    print("TANGENT_GPU_BEFORE_TRAIN " + json.dumps({"gpus": gpu.stdout, "processes": processes.stdout}), flush=True)
    if torch.cuda.device_count() != 1:
        raise RuntimeError("Exactly one visible GPU is required")
    from huggingface_hub import snapshot_download
    from models.model_fabric import LoRAModelFabric
    revision = "9213176726f574b556790deb65791e0c5aa438b6"
    snapshot = snapshot_download(config["model"], revision=revision,
                                 allow_patterns=["*.json", "*.safetensors", "*.model", "*.jinja"],
                                 token=os.environ.get("HF_TOKEN"))
    LoRAModelFabric.MODEL_ALIASES[config["model"].lower()] = snapshot
    config["model_revision"] = revision
    (run / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    set_seed(args.seed)
    model, tokenizer, groups = build_model(config)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    model.config.use_cache = False
    data_dir = Path(os.environ.get("TANGENT_DATA_DIR", "/tmp/tangent-ablation-data"))
    loader, evaluator, manifest = build_data(
        tokenizer, args.dataset, args.seed, 8, data_dir, guard, args.eval_max_samples)
    (run / "data-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    guard(reason="before_loi_initialization")
    initialize_subspace(config, model, loader)
    initial_adapter_sha256 = adapter_digest(groups)
    (run / "initialization.json").write_text(json.dumps({
        "adapter_sha256": initial_adapter_sha256, "seed": args.seed,
        "method": "LOI RSVD q15 scale0", "rsvd_batches": 31,
    }, indent=2) + "\n")
    set_seed(args.seed)
    loader.generator.manual_seed(args.seed)
    model.zero_grad(set_to_none=True)
    names = {id(parameter): name for name, parameter in model.named_parameters()}
    for group in groups:
        group["name"] = names[id(group["params"][0])].split(".lora_A.")[0]
    spectrum_steps = [1, *range(args.eval_every, args.steps + 1, args.eval_every)]
    if args.steps not in spectrum_steps:
        spectrum_steps.append(args.steps)
    optimizer = TangentAblation(
        groups, variant=args.variant, lr=args.lr, spectrum_path=run / "spectra.jsonl",
        spectrum_steps=spectrum_steps, ns_steps=5, use_momentum=True, momentum_cf=.95,
        balance=True, balance_mode="norm", rebalance_every=1,
        max_delta_norm=.1, log_optimizer_stats=False,
    )
    parameters = [parameter for group in groups for parameter in group["params"]]
    schedule = GenerationSchedule(optimizer, config)
    history = run / "history.jsonl"
    evaluations = []

    def validate(step: int) -> None:
        guard(reason="before_validation")
        started = time.monotonic()
        model.gradient_checkpointing_disable()
        model.config.use_cache = True
        try:
            score = evaluator.evaluate(model, tokenizer, batch_size=8, use_logprob=True)
            if not math.isfinite(score):
                raise RuntimeError("Nonfinite validation score")
            record = {"event": "validation", "step": step, "accuracy": score,
                      "selection_eligible": step > 0,
                      "validation_seconds": time.monotonic() - started}
            append_metric(history, record)
            if step > 0:
                evaluations.append(record)
        finally:
            model.config.use_cache = False
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
            model.train()

    validate(0)
    iterator = next_batches(loader)
    training_started = time.monotonic()
    for step in range(1, args.steps + 1):
        guard(reason="before_training_update")
        started = time.monotonic()
        optimizer.zero_grad(set_to_none=True)
        model.train()
        total_loss = 0.
        supervised_tokens = 0
        for _ in range(config["grad_accum_steps"]):
            batch = next(iterator)
            token_count = int((batch["labels"][:, 1:] != -100).sum())
            if token_count == 0:
                raise RuntimeError(f"No supervised tokens in a microbatch at update {step}")
            supervised_tokens += token_count
            batch = {key: value.to("cuda") for key, value in batch.items()}
            loss = model(**batch).loss
            if not torch.isfinite(loss):
                raise RuntimeError(f"Nonfinite loss at update {step}")
            (loss / config["grad_accum_steps"]).backward()
            total_loss += float(loss.detach()) / config["grad_accum_steps"]
        norm = gradient_norm(parameters, 1.)
        used_lr = schedule.before_step(step)
        optimizer.step()
        schedule.after_step()
        append_metric(history, {"event": "train", "step": step, "train/loss": total_loss,
                                "lr_used": used_lr, "supervised_tokens": supervised_tokens,
                                "grad_norm": float(norm), "step_seconds": time.monotonic() - started,
                                "elapsed_seconds": time.monotonic() - training_started,
                                "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                                "cuda_peak_reserved_bytes": torch.cuda.max_memory_reserved()})
        if step % args.eval_every == 0 or step == args.steps:
            validate(step)
    best = max(evaluations, key=lambda item: item["accuracy"])
    result = {"status": "completed", "dataset": args.dataset, "variant": args.variant,
              "lr": args.lr, "seed": args.seed, "steps": args.steps,
              "eval_every": args.eval_every, "best_score": best["accuracy"],
              "best_step": best["step"], "evaluations": evaluations,
              "initial_adapter_sha256": initial_adapter_sha256,
              "data_manifest_sha256": manifest["manifest_sha256"],
              "checkpoints": False, "timestamp": time.time()}
    guard(pending_write_bytes=1_000_000, reason="before_result")
    (run / "result.tmp").write_text(json.dumps(result, indent=2) + "\n")
    (run / "result.tmp").replace(run / "result.json")
    print("TANGENT_RESULT " + json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
