import re

import torch
from datasets import load_dataset
from dataclasses import dataclass
from torch.utils.data import DataLoader
from transformers import PreTrainedTokenizerBase

from .base_eval import BaseLogprobEval
from .data_common import format_label_for_prompt, select_best_choice


def qnli_prompt(text1: str, text2: str) -> str:
    return (
        "Determine whether the sentence entails the question.\n"
        "Reply with exactly one of: entailment, not entailment.\n\n"
        f"Question: {text1}\n"
        f"Sentence: {text2}\n"
        "Answer:"
    )


def qnli_label_text(label) -> str:
    if isinstance(label, str):
        t = label.strip().lower()
        if t == "entailment":
            return "entailment"
        if t in {"not entailment", "not_entailment", "not-entailment"}:
            return "not entailment"
        raise ValueError(f"Unexpected label_text: {label!r}")

    li = int(label)
    if li == 0:
        return "entailment"
    if li == 1:
        return "not entailment"
    raise ValueError(f"Unexpected label value: {label!r}")


def _example_to_prompt_and_gold(ex):
    if "prompt" in ex:
        prompt = ex["prompt"]
        gold = ex.get("label_text", None)
        if gold is None and "label" in ex:
            gold = qnli_label_text(ex["label"])
        return prompt, qnli_label_text(gold)

    prompt = qnli_prompt(ex["text1"], ex["text2"])
    if "label_text" in ex and ex["label_text"] is not None:
        gold = qnli_label_text(ex["label_text"])
    else:
        gold = qnli_label_text(ex["label"])
    return prompt, gold


@dataclass
class QNLISFTCollator:
    tokenizer: PreTrainedTokenizerBase
    max_length: int = 1024

    def __post_init__(self):
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def __call__(self, batch):
        input_ids_list = []
        attention_mask_list = []
        labels_list = []

        for ex in batch:
            prompt, gold = _example_to_prompt_and_gold(ex)

            prompt_enc = self.tokenizer(
                prompt,
                add_special_tokens=False,
                truncation=True,
                max_length=self.max_length,
            )
            label_text = format_label_for_prompt(prompt, gold)
            full_text = prompt + label_text
            full_enc = self.tokenizer(
                full_text,
                add_special_tokens=False,
                truncation=True,
                max_length=self.max_length,
            )

            input_ids = torch.tensor(full_enc["input_ids"], dtype=torch.long)
            attention_mask = torch.ones_like(input_ids, dtype=torch.long)

            labels = input_ids.clone()
            prompt_len = len(prompt_enc["input_ids"])
            labels[:prompt_len] = -100  # Mask prompt tokens.

            input_ids_list.append(input_ids)
            attention_mask_list.append(attention_mask)
            labels_list.append(labels)

        input_ids_padded = torch.nn.utils.rnn.pad_sequence(
            input_ids_list, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        attention_mask_padded = torch.nn.utils.rnn.pad_sequence(
            attention_mask_list, batch_first=True, padding_value=0
        )
        labels_padded = torch.nn.utils.rnn.pad_sequence(
            labels_list, batch_first=True, padding_value=-100
        )

        return {
            "input_ids": input_ids_padded,
            "attention_mask": attention_mask_padded,
            "labels": labels_padded,
        }


def get_train_dataloader(tokenizer, config):
    train_dataset = load_dataset("SetFit/qnli", split="train")

    collator = QNLISFTCollator(
        tokenizer=tokenizer,
        max_length=int(config.get("max_length", 1024)),
    )

    g = torch.Generator()
    g.manual_seed(int(config["seed"]))

    return DataLoader(
        train_dataset,
        batch_size=int(config["train_batch_size"]),
        shuffle=True,
        generator=g,
        collate_fn=collator,
        pin_memory=torch.cuda.is_available(),
    )


_NOT_ENT_RE = re.compile(r"\bnot[\s_-]?entailment\b", re.IGNORECASE)
_ENT_RE = re.compile(r"\bentailment\b", re.IGNORECASE)


def _normalize_qnli_output(text: str):
    if not text:
        return None
    t = text.strip()

    if _NOT_ENT_RE.search(t):
        return "not entailment"
    if _ENT_RE.search(t):
        return "entailment"
    return None


class QNLIEval(BaseLogprobEval):
    dataset_label = "QNLI"
    _CHOICES = ["entailment", "not entailment"]

    def load_eval_dataset(self):
        return load_dataset("SetFit/qnli", split="validation")

    def example_to_prompt_gold_choices(self, ex, use_chat_template: bool):
        prompt, gold = _example_to_prompt_and_gold(ex)
        return prompt, gold, self._CHOICES

    def normalize_pred(self, pred: str) -> str:
        return _normalize_qnli_output(pred) or pred.strip()


def compute_qnli_accuracy(model, tokenizer, batch_size=32, use_logprob: bool = True) -> float:
    return QNLIEval().evaluate(model, tokenizer, batch_size=batch_size, use_logprob=use_logprob)
