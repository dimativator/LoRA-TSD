"""CPU-only smoke test of real chat masking and official generation scorers."""

import json
import os
from pathlib import Path
import tempfile

import sacrebleu
import torch
from transformers import AutoTokenizer

from disk_guard import DiskGuard
from generation_data import ChatCollator, _prepare_ifeval, e2e_messages, encode_messages
from generation_eval import _prepare_ifeval_runtime, _reference_streams, _score_ifeval


def main() -> None:
    root = Path(os.environ["GENERATION_ROOT"])
    guard = DiskGuard(root, "cpu_smoke")
    guard(pending_write_bytes=100_000_000, reason="tokenizer_and_smoke_data")
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
                                            "collated_shape": list(collated["input_ids"].shape),
                                            "e2e_bleu": bleu, "e2e_chrfpp": chrf,
                                            "ifeval_prompts": len(prompts),
                                            "ifeval_synthetic_scores": scores}), flush=True)


if __name__ == "__main__":
    main()
