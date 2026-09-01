import re

import torch
from datasets import load_dataset
from torch.utils.data import DataLoader

from .base_eval import BaseLogprobEval
from .data_common import GenericSFTCollator


def obqa_prompt(
    question_stem: str,
    choices_text: list,
    choices_label: list,
    chat_template: bool = False,
) -> tuple:
    choices_formatted = "\n".join(
        [f"{label}: {text}" for label, text in zip(choices_label, choices_text)]
    )

    if chat_template:
        messages = [
            {
                "role": "system",
                "content": "Please choose the correct answer for the given question. Follow the answer format, full answer not needed.",
            },
            {
                "role": "user",
                "content": f"Question: {question_stem}\n\n{choices_formatted}\n\nAnswer format: A/B/C/D",
            },
        ]
        return messages

    system_part = "system: Please choose the correct answer for the given question. Follow the answer format, full answer not needed."
    user_part = f"user: Question: {question_stem}\n\n{choices_formatted}\n\nAnswer format: A/B/C/D"
    assistant_part = "assistant: The correct answer is "
    return f"{system_part}\n\n{user_part}\n\n{assistant_part}"


def obqa_label_text(answerKey: str) -> str:
    return str(answerKey).upper()


def _example_to_prompt_and_gold(ex, chat_template: bool = False):
    gold = obqa_label_text(ex["answerKey"])

    if chat_template:
        messages = obqa_prompt(
            ex["question_stem"],
            ex["choices"]["text"],
            ex["choices"]["label"],
            chat_template=True,
        )
        messages.append({"role": "assistant", "content": gold})
        return messages, gold

    prompt = obqa_prompt(
        ex["question_stem"],
        ex["choices"]["text"],
        ex["choices"]["label"],
        chat_template=False,
    )
    return prompt, gold


def get_train_dataloader(tokenizer, config):
    train_dataset = load_dataset("allenai/openbookqa", "main")["train"]

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


_LETTER_RE = re.compile(r"\b([abcd])\b", re.IGNORECASE)


def _normalize_answer(text):
    if text is None:
        return None
    t = text.strip().upper()

    if t and t[0] in ["A", "B", "C", "D"]:
        return t[0]

    m = _LETTER_RE.search(text)
    if m:
        return m.group(1).upper()

    return None


class OBQAEval(BaseLogprobEval):
    dataset_label = "OBQA"

    def load_eval_dataset(self):
        return load_dataset("allenai/openbookqa", "main")["validation"]

    def example_to_prompt_gold_choices(self, ex, use_chat_template: bool):
        gold = obqa_label_text(ex["answerKey"])
        labels = list(ex["choices"]["label"])
        if use_chat_template:
            prompt = obqa_prompt(ex["question_stem"], ex["choices"]["text"], labels, chat_template=True)
        else:
            prompt = obqa_prompt(ex["question_stem"], ex["choices"]["text"], labels, chat_template=False)
        return prompt, gold, labels

    def normalize_pred(self, pred: str) -> str:
        return pred.strip().upper()


def compute_obqa_accuracy(model, tokenizer, batch_size=32, use_logprob: bool = True) -> float:
    return OBQAEval().evaluate(model, tokenizer, batch_size=batch_size, use_logprob=use_logprob)
