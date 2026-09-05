import logging
import torch
from dataclasses import dataclass
from typing import Callable
from transformers import PreTrainedTokenizerBase

logger = logging.getLogger(__name__)


def format_label_for_prompt(prompt: str, label: str) -> str:
    if label is None:
        return ""
    label_text = str(label)
    if prompt and prompt[-1].isspace():
        return label_text
    return " " + label_text


def get_chat_template_kwargs(tokenizer) -> dict:
    """Disable Qwen3 thinking so train and evaluation use the same prefix."""
    template = getattr(tokenizer, "chat_template", None) or ""
    if "enable_thinking" in template:
        return {"enable_thinking": False}
    return {}


@dataclass
class GenericSFTCollator:
    tokenizer: PreTrainedTokenizerBase
    example_to_prompt_and_gold: Callable
    max_length: int = 1024

    def __post_init__(self):
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def __call__(self, batch):
        input_ids_list = []
        attention_mask_list = []
        labels_list = []

        n_skipped = 0
        for ex in batch:
            if hasattr(self.tokenizer, "apply_chat_template") and self.tokenizer.chat_template is not None:
                messages, gold = self.example_to_prompt_and_gold(ex, chat_template=True)
                template_kwargs = get_chat_template_kwargs(self.tokenizer)

                full_text = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=False,
                    **template_kwargs,
                )

                user_messages = messages[:-1]
                user_text = self.tokenizer.apply_chat_template(
                    user_messages,
                    tokenize=False,
                    add_generation_prompt=True,
                    **template_kwargs,
                )

                full_ids = self.tokenizer(
                    full_text,
                    add_special_tokens=False,
                    truncation=True,
                    max_length=self.max_length,
                )["input_ids"]

                user_ids = self.tokenizer(
                    user_text,
                    add_special_tokens=False,
                    truncation=True,
                    max_length=self.max_length,
                )["input_ids"]

                input_ids = torch.tensor(full_ids, dtype=torch.long)
                attention_mask = torch.ones_like(input_ids, dtype=torch.long)

                labels = torch.full_like(input_ids, -100)
                labels[len(user_ids) :] = input_ids[len(user_ids) :]

            else:
                prompt, gold = self.example_to_prompt_and_gold(ex, chat_template=False)

                prompt_ids = self.tokenizer(
                    prompt,
                    add_special_tokens=False,
                    truncation=True,
                    max_length=self.max_length - 20,
                )["input_ids"]

                label_text = format_label_for_prompt(prompt, gold)
                gold_ids = self.tokenizer(
                    label_text,
                    add_special_tokens=False,
                )["input_ids"]

                if len(gold_ids) == 0:
                    n_skipped += 1
                    continue

                input_ids = prompt_ids + gold_ids
                if len(input_ids) > self.max_length:
                    input_ids = input_ids[: self.max_length]

                input_ids = torch.tensor(input_ids, dtype=torch.long)
                attention_mask = torch.ones_like(input_ids, dtype=torch.long)

                labels = torch.full_like(input_ids, -100)
                answer_start = len(prompt_ids)
                labels[answer_start:] = input_ids[answer_start:]

            input_ids_list.append(input_ids)
            attention_mask_list.append(attention_mask)
            labels_list.append(labels)

        if n_skipped > 0:
            logger.warning(
                "GenericSFTCollator: skipped %d/%d examples with empty gold token ids. "
                "This silently reduces your effective batch size.",
                n_skipped, len(batch),
            )

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


def compute_logprob_for_choice(prompt, choice_label, model, tokenizer):

    prompt_ids = tokenizer.encode(prompt, return_tensors="pt", add_special_tokens=False)
    choice_ids = tokenizer.encode(
        choice_label, return_tensors="pt", add_special_tokens=False
    )

    input_ids = torch.cat((prompt_ids, choice_ids), dim=1).to(model.device)

    with torch.no_grad():
        outputs = model(input_ids, labels=input_ids)

    logits = outputs.logits[:, :-1, :]
    target_ids = input_ids[:, 1:]

    log_probs = torch.nn.functional.log_softmax(logits, dim=-1)

    chosen_log_probs = log_probs.gather(2, target_ids.unsqueeze(-1)).squeeze(-1)

    choice_log_probs = chosen_log_probs[
        :, prompt_ids.shape[1] - 1 : prompt_ids.shape[1] + choice_ids.shape[1] - 1
    ]

    avg_logprob = choice_log_probs.sum().item() / choice_ids.shape[1]
    return avg_logprob


def select_best_choice(prompt, choice_labels, model, tokenizer):
    best_choice = None
    best_logprob = float("-inf")
    logprobs = []

    for label in choice_labels:
        lp = compute_logprob_for_choice(prompt, label, model, tokenizer)
        logprobs.append(lp)

        if lp > best_logprob:
            best_logprob = lp
            best_choice = label

    return best_choice, logprobs
