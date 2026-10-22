import re

import torch
from datasets import load_dataset
from torch.utils.data import DataLoader

from .base_eval import BaseLogprobEval
from .data_common import GenericSFTCollator


def piqa_prompt(goal: str, sol1: str, sol2: str, chat_template: bool = False) -> tuple:
    if chat_template:
        messages = [
            {
                "role": "system",
                "content": "Please choose the correct solution to the question. Follow the answer format, full answer not needed.",
            },
            {
                "role": "user",
                "content": f"Question: {goal}\n\nSolution 1: {sol1}\n\nSolution 2: {sol2}\n\nAnswer format: Solution 1/Solution 2",
            },
        ]
        return messages

    system_part = "system: Please choose the correct solution to the question. Follow the answer format, full answer not needed."
    user_part = f"user: Goal: {goal}\n\nSolution 1: {sol1}\n\nSolution 2: {sol2}\n\nAnswer format: Solution 1/Solution 2"
    assistant_part = "assistant: The correct answer is "
    return f"{system_part}\n\n{user_part}\n\n{assistant_part}"


def piqa_label_text(label: int) -> str:
    return "Solution 1" if label == 0 else "Solution 2"


def _example_to_prompt_and_gold(ex, chat_template: bool = False):
    gold = piqa_label_text(ex["label"])

    if chat_template:
        messages = piqa_prompt(ex["goal"], ex["sol1"], ex["sol2"], chat_template=True)
        messages.append({"role": "assistant", "content": gold})
        return messages, gold

    prompt = piqa_prompt(ex["goal"], ex["sol1"], ex["sol2"], chat_template=False)
    return prompt, gold


def get_train_dataloader(tokenizer, config):
    train_dataset = load_dataset("ybisk/piqa", split="train")

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


_SOLUTION_RE = re.compile(r"\b(solution\s*[12])\b", re.IGNORECASE)
_NUMBER_RE = re.compile(r"\b([12])\b")


def _normalize_solution(text):
    if text is None:
        return None
    t = text.strip().lower()

    m = _SOLUTION_RE.search(t)
    if m:
        solution_text = m.group(1).lower()
        if "1" in solution_text:
            return "Solution 1"
        elif "2" in solution_text:
            return "Solution 2"

    m2 = _NUMBER_RE.search(t)
    if m2:
        return "Solution 1" if m2.group(1) == "1" else "Solution 2"

    return None


class PIQAEval(BaseLogprobEval):
    dataset_label = "PIQA"
    _CHOICES = ["Solution 1", "Solution 2"]

    def load_eval_dataset(self):
        return load_dataset("ybisk/piqa", split="validation")

    def example_to_prompt_gold_choices(self, ex, use_chat_template: bool):
        gold = piqa_label_text(ex["label"])
        if use_chat_template:
            prompt = piqa_prompt(ex["goal"], ex["sol1"], ex["sol2"], chat_template=True)
        else:
            prompt = piqa_prompt(ex["goal"], ex["sol1"], ex["sol2"], chat_template=False)
        return prompt, gold, self._CHOICES

    def normalize_pred(self, pred: str) -> str:
        return _normalize_solution(pred) or pred.strip()


def compute_piqa_accuracy(model, tokenizer, batch_size=32, use_logprob: bool = True) -> float:
    return PIQAEval().evaluate(model, tokenizer, batch_size=batch_size, use_logprob=use_logprob)
