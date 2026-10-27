"""CPU-only smoke test of real chat masking and official generation scorers."""

import json
import os
from pathlib import Path
import tempfile
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import torch
from transformers import AutoTokenizer
from transformers import LlamaConfig, LlamaForCausalLM
from peft import LoraConfig, get_peft_model
from models import (RiemannianLoRALinear, apply_riemannian_lora, get_lora_param_groups,
                    get_muon_full_param_groups, get_riemannian_param_groups,
                    rsvd_init, rsvd_init_standard_lora)

from optimizer_presets import (METHODS, GenerationSchedule, enable_fixed_original_tangent,
                               generation_config, generation_optimizer, gradient_norm,
                               optimizer_parameters)


def optimizer_smoke(methods=METHODS) -> dict:
    results = {}
    for method in methods:
        torch.manual_seed(42)
        config = generation_config(method, "smoke", 3000, "/tmp/smoke", 2, 16)
        tiny = LlamaForCausalLM(LlamaConfig(vocab_size=128, hidden_size=64,
                                           intermediate_size=128, num_hidden_layers=1,
                                           num_attention_heads=4, num_key_value_heads=2))
        if method == "muon_full":
            for parameter in tiny.parameters():
                parameter.requires_grad_(False)
            for name, module in tiny.named_modules():
                if isinstance(module, torch.nn.Linear) and name.split(".")[-1] in config["lora_target_modules"]:
                    module.weight.requires_grad_(True)
            groups = get_muon_full_param_groups(tiny, config["lora_target_modules"])
        else:
            tiny = get_peft_model(tiny, LoraConfig(r=config["lora_r"], lora_alpha=config["lora_alpha"],
                                                  lora_dropout=config["lora_dropout"],
                                                  init_lora_weights=config["lora_init"],
                                                  target_modules=config["lora_target_modules"],
                                                  task_type="CAUSAL_LM"))
            if method in {"riemannian_sgd", "riemannion_original"}:
                tiny = apply_riemannian_lora(tiny, config["lora_r"], config["lora_alpha"], config["lora_dropout"])
                groups = get_riemannian_param_groups(tiny)
            else:
                groups = get_lora_param_groups(tiny, add_W=False)
        ids = torch.randint(0, 128, (2, 16))
        batch = {"input_ids": ids, "labels": ids.clone(), "attention_mask": torch.ones_like(ids)}
        if config["rsvd_init"]:
            initialize = rsvd_init if method in {"riemannian_sgd", "riemannion_original"} else rsvd_init_standard_lora
            # Exercise the production initializer with a smaller iteration count on CPU.
            initialize(tiny, [batch] * 3, q=1, device="cpu", scale=config["rsvd_scale"])
        enable_fixed_original_tangent(tiny, method)
        tiny.zero_grad(set_to_none=True)
        optimizer = generation_optimizer(groups, config)
        if method == "lora_pro":
            assert all(torch.count_nonzero(group["params"][1]) > 0 for group in optimizer.param_groups), \
                "Historical Pro initialization requires nonzero B"
        parameters = optimizer_parameters(optimizer)
        original = [parameter.detach().clone() for parameter in parameters]
        schedule = GenerationSchedule(optimizer, config)
        losses = []
        auxiliary_gradient = None
        lrs = []
        for step in range(1, 3):
            optimizer.zero_grad(set_to_none=True)
            loss = tiny(**batch).loss
            assert torch.isfinite(loss)
            loss.backward()
            if method == "riemannion_original":
                auxiliary_gradient = sum(float(module.lora_A.weight.grad[module.r:].norm())
                                         for module in tiny.modules() if isinstance(module, RiemannianLoRALinear))
                assert auxiliary_gradient > 0, "Fixed original must receive auxiliary tangent gradients"
            norm = gradient_norm(parameters, config["grad_clip_norm"])
            assert torch.isfinite(norm) and norm > 0
            assert any(parameter.grad is not None and torch.count_nonzero(parameter.grad)
                       for parameter in parameters), "No-clipping preset must preserve gradients"
            lrs.append(schedule.before_step(step))
            optimizer.step()
            schedule.after_step()
            assert all(torch.isfinite(parameter).all() for parameter in parameters)
            losses.append(float(loss.detach()))
        assert any(not torch.equal(before, after) for before, after in zip(original, parameters)), method
        results[method] = {"losses": losses, "lr_used": lrs, "parameters_changed": True,
                           "auxiliary_gradient_norm": auxiliary_gradient,
                           "preset": config}
    return results


def main() -> None:
    import sacrebleu
    from disk_guard import DiskGuard
    from generation_data import ChatCollator, _prepare_ifeval, e2e_messages, encode_messages
    from generation_eval import _prepare_ifeval_runtime, _reference_streams, _score_ifeval

    root = Path(os.environ["GENERATION_ROOT"])
    guard = DiskGuard(root, "cpu_smoke")
    guard(pending_write_bytes=100_000_000, reason="tokenizer_and_smoke_data")
    optimizer_results = optimizer_smoke()
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B-Instruct",
                                              token=os.environ.get("HF_TOKEN"))
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    messages = [{"role": "user", "content": "UNSUPERVISED_FIRST_USER"},
                {"role": "assistant", "content": "FIRST_ASSISTANT_ANSWER"},
                {"role": "user", "content": "UNSUPERVISED_SECOND_USER"},
                {"role": "assistant", "content": "SECOND_ASSISTANT_ANSWER"}]
    encoded = encode_messages(tokenizer, messages, 2048)
    selected = [label for label in encoded["labels"] if label != -100]
    text = tokenizer.decode(selected, skip_special_tokens=True)
    assert "FIRST_ASSISTANT_ANSWER" in text and "SECOND_ASSISTANT_ANSWER" in text, text
    assert "UNSUPERVISED" not in text, text
    short = encode_messages(tokenizer, messages, 1)
    assert all(label == -100 for label in short["labels"])
    restaurant = e2e_messages("name[The Eagle], food[French]", "The Eagle serves French food.")
    other = encode_messages(tokenizer, restaurant, 512)
    collated = ChatCollator(tokenizer)([encoded, other])
    assert collated["input_ids"].shape == collated["labels"].shape
    assert int((collated["labels"] != -100).sum()) == len(selected) + sum(
        label != -100 for label in other["labels"])
    assert torch.all(collated["labels"][collated["attention_mask"] == 0] == -100)
    full = tokenizer.apply_chat_template(restaurant, tokenize=True, add_generation_prompt=False)
    prompt = tokenizer.apply_chat_template(restaurant[:-1], tokenize=True, add_generation_prompt=True)
    assert full[:len(prompt)] == prompt
    refs = _reference_streams([{"references": ["The Eagle serves French food.", "French food at The Eagle."]}])
    predictions = ["The Eagle serves French food."]
    bleu = sacrebleu.corpus_bleu(predictions, refs).score
    chrf = sacrebleu.corpus_chrf(predictions, refs, word_order=2).score
    assert bleu > 99 and chrf > 99
    with tempfile.TemporaryDirectory(prefix="generation-cpu-smoke-") as directory:
        runtime, prompts = _prepare_ifeval(Path(directory), guard)
        _prepare_ifeval_runtime(str(runtime), guard)
        scores, details = _score_ifeval(prompts[:2], ["This is a test response."] * 2,
                                       str(runtime), guard)
        assert len(details) == 4
        assert len(scores) == 4 and all(0 <= value <= 1 for value in scores.values())
    print("GEN_CPU_SMOKE " + json.dumps({"status": "passed", "assistant_targets": len(selected),
                                            "optimizer_losses": {method: result["losses"]
                                                                 for method, result in optimizer_results.items()},
                                            "optimizer_checks": {method: {key: value for key, value in result.items()
                                                                          if key not in {"preset", "losses"}}
                                                                 for method, result in optimizer_results.items()},
                                            "collated_shape": list(collated["input_ids"].shape),
                                            "e2e_bleu": bleu, "e2e_chrfpp": chrf,
                                            "ifeval_prompts": len(prompts),
                                            "ifeval_synthetic_scores": scores}), flush=True)


if __name__ == "__main__":
    main()
