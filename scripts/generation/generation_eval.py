"""Token-weighted held-out loss and generated-response evaluation."""

import importlib
import json
import logging
import math
from pathlib import Path
import sys
import time
from typing import Callable

import torch


logger = logging.getLogger(__name__)


def _reference_streams(examples: list[dict]) -> list[list[str]]:
    count = max(len(row["references"]) for row in examples)
    return [[row["references"][index] if index < len(row["references"])
             else row["references"][0] for row in examples] for index in range(count)]


def _prepare_ifeval_runtime(root: str, guard: Callable) -> object:
    """Use unmodified official checks, including their sentence tokenizer data."""
    import langdetect
    import nltk

    langdetect.DetectorFactory.seed = 0
    nltk_dir = Path(root) / "nltk_data"
    guard(pending_write_bytes=40_000_000, reason="ifeval_nltk_data")
    nltk_dir.mkdir(parents=True, exist_ok=True)
    nltk.data.path.insert(0, str(nltk_dir))
    for package, resource in [("punkt", "tokenizers/punkt"),
                              ("punkt_tab", "tokenizers/punkt_tab/english")]:
        try:
            nltk.data.find(resource)
        except LookupError:
            if not nltk.download(package, download_dir=str(nltk_dir), quiet=True):
                raise RuntimeError("Unable to prepare IFEval tokenizer resource " + package)
    sys.path.insert(0, root)
    return importlib.import_module("instruction_following_eval.evaluation_lib")


def check_evaluation_dependencies(generation_examples: dict, guard: Callable) -> None:
    """Fail before optimization if a required official scorer is unavailable."""
    if generation_examples["task"] == "e2e_nlg":
        import sacrebleu  # noqa: F401
    else:
        _prepare_ifeval_runtime(generation_examples["ifeval_root"], guard)


def _score_ifeval(examples: list[dict], responses: list[str], root: str,
                  guard: Callable) -> tuple[dict, list[dict]]:
    lib = _prepare_ifeval_runtime(root, guard)
    mapping = {example["prompt"]: response for example, response in zip(examples, responses)}
    scores = {}
    details = []
    for mode, evaluator in [("strict", lib.test_instruction_following_strict),
                             ("loose", lib.test_instruction_following_loose)]:
        outputs = []
        for example in examples:
            inp = lib.InputExample(key=example["key"], prompt=example["prompt"],
                                   instruction_id_list=example["instruction_id_list"],
                                   kwargs=example["kwargs"])
            outputs.append(evaluator(inp, mapping))
        instruction_total = sum(len(item.follow_instruction_list) for item in outputs)
        scores[f"ifeval/{mode}_prompt_accuracy"] = sum(
            item.follow_all_instructions for item in outputs) / len(outputs)
        scores[f"ifeval/{mode}_instruction_accuracy"] = sum(
            sum(item.follow_instruction_list) for item in outputs) / instruction_total
        for example, item in zip(examples, outputs):
            details.append({"key": example["key"], "mode": mode,
                            "instruction_id_list": item.instruction_id_list,
                            "follow_instruction_list": item.follow_instruction_list,
                            "follow_all_instructions": item.follow_all_instructions})
    return scores, details


@torch.inference_mode()
def evaluate(model, tokenizer, dataset: str, val_dataloader,
             generation_examples: dict, output_dir, step: int, guard: Callable) -> dict:
    """Evaluate all development examples, preserving prompts and generation logs."""
    started = time.monotonic()
    check_evaluation_dependencies(generation_examples, guard)
    output_dir = Path(output_dir)
    device = next(model.parameters()).device
    was_training = model.training
    had_checkpointing = bool(getattr(model, "is_gradient_checkpointing", False))
    previous_cache = model.config.use_cache
    previous_padding = tokenizer.padding_side
    model.eval()
    if had_checkpointing:
        model.gradient_checkpointing_disable()
    model.config.use_cache = False
    try:
        loss_sum = 0.0
        target_tokens = 0
        for batch in val_dataloader:
            guard()
            batch = {key: value.to(device) for key, value in batch.items()}
            count = int((batch["labels"][:, 1:] != -100).sum())
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16,
                                enabled=device.type == "cuda"):
                loss = model(**batch).loss
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite validation loss")
            loss_sum += float(loss) * count
            target_tokens += count
        if target_tokens == 0:
            raise ValueError("Validation set contains no supervised target tokens")
        metrics = {"val/loss": loss_sum / target_tokens,
                   "val/target_tokens": target_tokens}
        metrics["val/perplexity"] = math.exp(min(metrics["val/loss"], 50.0))
        examples = generation_examples["examples"]
        is_e2e = generation_examples["task"] == "e2e_nlg"
        max_new_tokens = generation_examples.get("max_new_tokens", 128 if is_e2e else 2048)
        batch_size = generation_examples.get("batch_size", 8)
        tokenizer.padding_side = "left"
        model.config.use_cache = True
        eos_ids = [tokenizer.eos_token_id]
        configured_eos = getattr(getattr(model, "generation_config", None), "eos_token_id", [])
        if isinstance(configured_eos, int):
            configured_eos = [configured_eos]
        for token in configured_eos or []:
            if token not in eos_ids:
                eos_ids.append(token)
        end_turn = tokenizer.convert_tokens_to_ids("<|eot_id|>")
        if end_turn is not None and end_turn != tokenizer.unk_token_id and end_turn not in eos_ids:
            eos_ids.append(end_turn)
        predictions = []
        details = []
        for offset in range(0, len(examples), batch_size):
            guard()
            chunk = examples[offset:offset + batch_size]
            prompts = [tokenizer.apply_chat_template(
                row["messages"], tokenize=False, add_generation_prompt=True,
            ) for row in chunk]
            inputs = tokenizer(prompts, return_tensors="pt", padding=True,
                               add_special_tokens=False, truncation=False).to(device)
            prefix_length = inputs["input_ids"].shape[1]
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16,
                                enabled=device.type == "cuda"):
                outputs = model.generate(**inputs, do_sample=False,
                                         max_new_tokens=max_new_tokens,
                                         eos_token_id=eos_ids,
                                         pad_token_id=tokenizer.pad_token_id,
                                         use_cache=True)
            generated = outputs[:, prefix_length:]
            responses = tokenizer.batch_decode(generated, skip_special_tokens=True)
            for row, response, ids in zip(chunk, responses, generated):
                token_list = ids.tolist()
                end_positions = [index for index, token in enumerate(token_list) if token in eos_ids]
                generated_length = end_positions[0] + 1 if end_positions else len(token_list)
                scored_response = response.strip() if is_e2e else response
                predictions.append(scored_response)
                details.append({"step": step, "prompt": row.get("prompt", row.get("mr")),
                                "response": scored_response, "generated_tokens": generated_length,
                                "hit_generation_limit": not end_positions and generated_length >= max_new_tokens,
                                "references": row.get("references"), "key": row.get("key")})
            logger.info("Generation validation step %s: %s/%s responses", step,
                        min(offset + batch_size, len(examples)), len(examples))
        if is_e2e:
            import sacrebleu
            references = _reference_streams(examples)
            bleu = sacrebleu.metrics.BLEU()
            chrf = sacrebleu.metrics.CHRF(word_order=2)
            metrics["e2e/bleu"] = bleu.corpus_score(predictions, references).score
            metrics["e2e/chrfpp"] = chrf.corpus_score(predictions, references).score
            metrics["e2e/bleu_signature"] = str(bleu.get_signature())
            metrics["e2e/chrfpp_signature"] = str(chrf.get_signature())
            check_details = []
        else:
            scores, check_details = _score_ifeval(
                examples, predictions, generation_examples["ifeval_root"], guard)
            metrics.update(scores)
        metrics.update({"generation/examples": len(examples),
                        "generation/max_new_tokens": max_new_tokens,
                        "generation/hit_limit_count": sum(row["hit_generation_limit"] for row in details),
                        "generation/mean_output_tokens": sum(row["generated_tokens"] for row in details) / len(details),
                        "eval/seconds": time.monotonic() - started})
        payload = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in details)
        checks = "".join(json.dumps(row) + "\n" for row in check_details)
        guard(pending_write_bytes=len(payload.encode()) + len(checks.encode()) + 20000,
              reason="validation_prediction_logs")
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / f"predictions-step{step:06d}.jsonl").write_text(payload)
        if checks:
            (output_dir / f"ifeval-checks-step{step:06d}.jsonl").write_text(checks)
        return metrics
    finally:
        tokenizer.padding_side = previous_padding
        model.config.use_cache = previous_cache
        if had_checkpointing:
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.train(was_training)
