import re

import torch
from datasets import load_dataset
from dataclasses import dataclass
from torch.utils.data import DataLoader
from transformers import PreTrainedTokenizerBase

from .base_eval import BaseLogprobEval
from .data_common import format_label_for_prompt, select_best_choice


def multi_nli_prompt(premise: str, hypothesis: str) -> str:
    return (
        "Determine the relationship between the premise and the hypothesis.\n"
        "Reply with exactly one of: entailment, neutral, contradiction.\n\n"
        f"Premise: {premise}\n"
        f"Hypothesis: {hypothesis}\n"
        "Answer:"
    )


def multi_nli_label_text(label) -> str:
    if isinstance(label, str):
        t = label.strip().lower()
        if t in {"entailment", "neutral", "contradiction"}:
            return t
        raise ValueError(f"Unexpected label string: {label!r}")

    li = int(label)
    if li == 0:
        return "entailment"
    if li == 1:
        return "neutral"
    if li == 2:
        return "contradiction"
    if li == -1:
        raise ValueError(
            "Unlabeled example (label=-1). Filter these out before training/eval."
        )
    raise ValueError(f"Unexpected label value: {label!r}")


def _example_to_prompt_and_gold(ex):
    if "prompt" in ex:
        prompt = ex["prompt"]
        gold = ex.get("label_text", None)
        if gold is None and "label" in ex and int(ex["label"]) != -1:
            gold = multi_nli_label_text(ex["label"])
        if gold is None:
            raise ValueError(
                f"Cannot determine gold label for example with label={ex.get('label')}. "
                "Filter out label=-1 examples before calling this function."
            )
        return prompt, multi_nli_label_text(gold)

    label = ex["label"]
    if int(label) == -1:
        raise ValueError(
            "Unlabeled example (label=-1) reached _example_to_prompt_and_gold. "
            "Apply ds.filter(lambda ex: int(ex['label']) != -1) before iterating."
        )
    prompt = multi_nli_prompt(ex["premise"], ex["hypothesis"])
    gold = multi_nli_label_text(label)
    return prompt, gold


@dataclass
class MultiNLISFTCollator:
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
    ds = load_dataset("nyu-mll/multi_nli", split="train")
    ds = ds.filter(lambda ex: int(ex["label"]) != -1)

    collator = MultiNLISFTCollator(
        tokenizer=tokenizer,
        max_length=int(config.get("max_length", 1024)),
    )

    g = torch.Generator()
    g.manual_seed(int(config["seed"]))

    return DataLoader(
        ds,
        batch_size=int(config["train_batch_size"]),
        shuffle=True,
        generator=g,
        collate_fn=collator,
        pin_memory=torch.cuda.is_available(),
    )


_CONTR_RE = re.compile(r"\bcontradiction\b", re.IGNORECASE)
_NEUTRAL_RE = re.compile(r"\bneutral\b", re.IGNORECASE)
_ENT_RE = re.compile(r"\bentailment\b", re.IGNORECASE)
_DIGIT_RE = re.compile(r"\b([012])\b")


def _normalize_multi_nli_output(text: str):
    if not text:
        return None
    t = text.strip()

    if _CONTR_RE.search(t):
        return "contradiction"
    if _NEUTRAL_RE.search(t):
        return "neutral"
    if _ENT_RE.search(t):
        return "entailment"

    m = _DIGIT_RE.search(t)
    if m:
        d = int(m.group(1))
        return {0: "entailment", 1: "neutral", 2: "contradiction"}[d]

    return None


class MultiNLIEval(BaseLogprobEval):
    _CHOICES = ["entailment", "neutral", "contradiction"]

    def __init__(self, split: str = "validation_matched"):
        self.split = split
        self.dataset_label = f"MultiNLI/{split}"

    def load_eval_dataset(self):
        ds = load_dataset("nyu-mll/multi_nli", split=self.split)
        return ds.filter(lambda ex: int(ex["label"]) != -1)

    def example_to_prompt_gold_choices(self, ex, use_chat_template: bool):
        prompt, gold = _example_to_prompt_and_gold(ex)
        return prompt, gold, self._CHOICES

    def normalize_pred(self, pred: str) -> str:
        return _normalize_multi_nli_output(pred) or pred.strip()


def compute_multi_nli_accuracy(
    model,
    tokenizer,
    batch_size: int = 32,
    split: str = "validation_matched",
    use_logprob: bool = True,
) -> float:
    return MultiNLIEval(split=split).evaluate(
        model, tokenizer, batch_size=batch_size, use_logprob=use_logprob
    )


def compute_multi_nli_accuracy_both(model, tokenizer, batch_size: int = 32) -> float:
    acc_matched = compute_multi_nli_accuracy(model, tokenizer, batch_size=batch_size, split="validation_matched")
    acc_mismatched = compute_multi_nli_accuracy(model, tokenizer, batch_size=batch_size, split="validation_mismatched")
    return (float(acc_matched) + float(acc_mismatched)) / 2
