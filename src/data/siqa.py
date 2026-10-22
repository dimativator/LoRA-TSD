import os
import re
from pathlib import Path
from functools import lru_cache

import torch
from datasets import Dataset, load_dataset
from torch.utils.data import DataLoader

from .base_eval import BaseLogprobEval
from .data_common import GenericSFTCollator


@lru_cache(maxsize=2)
def _load_split(split):
    root = os.environ.get("LORA_BENCHMARK_DATA_DIR")
    if root:
        import pyarrow.parquet as pq
        files = sorted((Path(root) / "siqa").rglob(f"{split}*.parquet"))
        if not files:
            raise FileNotFoundError(f"Missing local SIQA {split} parquet under {root}")
        return Dataset(pq.read_table([str(path) for path in files]))
    return load_dataset("allenai/social_i_qa", split=split)


def siqa_prompt(
    context: str,
    question: str,
    answerA: str,
    answerB: str,
    answerC: str,
    chat_template: bool = False,
) -> tuple:
    if chat_template:
        messages = [
            {
                "role": "system",
                "content": "Please choose the correct answer to the question based on the context provided. Follow the answer format, full answer not needed.",
            },
            {
                "role": "user",
                "content": f"Context: {context}\n\nQuestion: {question}\n\nA: {answerA}\n\nB: {answerB}\n\nC: {answerC}\n\nAnswer format: A/B/C",
            },
        ]
        return messages

    system_part = (
        "system: Please choose the correct answer to the question based on the context provided. Follow the answer format, full answer not needed."
    )
    user_part = (
        f"user: Context: {context}\n\nQuestion: {question}\n\n"
        f"A: {answerA}\n\nB: {answerB}\n\nC: {answerC}\n\n"
        f"Answer format: A/B/C"
    )
    assistant_part = "assistant: The correct answer is "
    return f"{system_part}\n\n{user_part}\n\n{assistant_part}"


def siqa_label_text(label: str) -> str:
    label_map = {"1": "A", "2": "B", "3": "C"}
    return label_map.get(str(label))


def _example_to_prompt_and_gold(ex, chat_template: bool = False):
    gold = siqa_label_text(ex["label"])

    if chat_template:
        messages = siqa_prompt(
            ex["context"],
            ex["question"],
            ex["answerA"],
            ex["answerB"],
            ex["answerC"],
            chat_template=True,
        )
        messages.append({"role": "assistant", "content": gold})
        return messages, gold

    prompt = siqa_prompt(
        ex["context"],
        ex["question"],
        ex["answerA"],
        ex["answerB"],
        ex["answerC"],
        chat_template=False,
    )
    return prompt, gold


def get_train_dataloader(tokenizer, config):
    train_dataset = _load_split("train")

    collator = GenericSFTCollator(
        tokenizer=tokenizer,
        example_to_prompt_and_gold=_example_to_prompt_and_gold,
        max_length=1024
    )

    g = torch.Generator()
    g.manual_seed(config["seed"])

    return DataLoader(
        train_dataset,
        batch_size=config["train_batch_size"],
        shuffle=True,
        generator=g,
        collate_fn=collator,
        pin_memory=torch.cuda.is_available(),
    )


_ANSWER_RE = re.compile(r"\b(answer\s*[abc])\b", re.IGNORECASE)
_LETTER_RE = re.compile(r"\b([abc])\b", re.IGNORECASE)


def _normalize_answer(text):
    if text is None:
        return None
    t = text.strip().lower()

    m2 = _LETTER_RE.search(t)
    if m2:
        letter = m2.group(1).lower()
        if letter == "a":
            return "A"
        elif letter == "b":
            return "B"
        elif letter == "c":
            return "C"

    return None


class SIQAEval(BaseLogprobEval):
    dataset_label = "SIQA"
    _CHOICES = ["A", "B", "C"]

    def load_eval_dataset(self):
        return _load_split("validation")

    def example_to_prompt_gold_choices(self, ex, use_chat_template: bool):
        gold = siqa_label_text(ex["label"])
        if use_chat_template:
            prompt = siqa_prompt(
                ex["context"], ex["question"],
                ex["answerA"], ex["answerB"], ex["answerC"],
                chat_template=True,
            )
        else:
            prompt = siqa_prompt(
                ex["context"], ex["question"],
                ex["answerA"], ex["answerB"], ex["answerC"],
                chat_template=False,
            )
        return prompt, gold, self._CHOICES

    def normalize_pred(self, pred: str) -> str:
        return pred.strip().upper()


def compute_siqa_accuracy(model, tokenizer, batch_size=32, use_logprob: bool = True) -> float:
    return SIQAEval().evaluate(model, tokenizer, batch_size=batch_size, use_logprob=use_logprob)
