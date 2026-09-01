import re

import torch
from datasets import load_dataset
from torch.utils.data import DataLoader

from .base_eval import BaseLogprobEval
from .data_common import GenericSFTCollator


def boolq_prompt(passage: str, question: str, chat_template: bool = False) -> tuple:
    if chat_template:
        messages = [
            {
                "role": "system",
                "content": "Please answer the following question with True or False. Follow the answer format, full answer not needed.",
            },
            {
                "role": "user",
                "content": f"Question: {question}\nPassage: {passage}\nAnswer format: True/False",
            },
        ]
        return messages

    system_part = (
        "system: Please answer the following question with True or False. "
        "Follow the answer format, full answer not needed."
    )
    user_part = (
        f"user: Question: {question}\nPassage: {passage}\nAnswer format: True/False"
    )
    assistant_part = "assistant: The correct answer is "
    return f"{system_part}\n\n{user_part}\n\n{assistant_part}"


def boolq_label_text(answer_bool: bool) -> str:
    return "True" if bool(answer_bool) else "False"


def _example_to_prompt_and_gold(ex, chat_template: bool = False):
    gold = boolq_label_text(ex["answer"])

    if chat_template:
        messages = boolq_prompt(ex["passage"], ex["question"], chat_template=True)
        messages.append({"role": "assistant", "content": gold})
        return messages, gold

    prompt = boolq_prompt(ex["passage"], ex["question"], chat_template=False)
    return prompt, gold


def get_train_dataloader(tokenizer, config):
    train_dataset = load_dataset("google/boolq")["train"]

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


_YESNO_RE = re.compile(r"\b(yes|no)\b", re.IGNORECASE)
_TRUEFALSE_RE = re.compile(r"\b(true|false)\b", re.IGNORECASE)

def _normalize_yesno(text):
    if text is None:
        return None
    t = text.strip().lower()

    m = _TRUEFALSE_RE.search(t)
    if m:
        return "True" if m.group(1).lower() == "true" else "False"

    m2 = _YESNO_RE.search(t)
    if m2:
        return "True" if m2.group(1).lower() == "yes" else "False"

    return None


class BoolQEval(BaseLogprobEval):
    dataset_label = "BoolQ"

    def load_eval_dataset(self):
        return load_dataset("google/boolq")["validation"]

    def example_to_prompt_gold_choices(self, ex, use_chat_template: bool):
        gold = boolq_label_text(ex["answer"])
        if use_chat_template:
            prompt = boolq_prompt(ex["passage"], ex["question"], chat_template=True)
        else:
            prompt = boolq_prompt(ex["passage"], ex["question"], chat_template=False)
        return prompt, gold, ["True", "False"]

    def normalize_pred(self, pred: str) -> str:
        return _normalize_yesno(pred) or pred.strip()


def compute_boolq_accuracy(model, tokenizer, batch_size=32, use_logprob=True) -> float:
    return BoolQEval().evaluate(model, tokenizer, batch_size=batch_size, use_logprob=use_logprob)
