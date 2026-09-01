import logging
from abc import ABC, abstractmethod

import torch
from tqdm import tqdm

from .data_common import (
    format_label_for_prompt,
    get_chat_template_kwargs,
    select_best_choice,
)

logger = logging.getLogger(__name__)


class BaseLogprobEval(ABC):
    dataset_label: str = ""

    @abstractmethod
    def load_eval_dataset(self):
        pass

    @abstractmethod
    def example_to_prompt_gold_choices(self, ex, use_chat_template: bool):
        pass

    def normalize_pred(self, pred: str) -> str:
        return pred.strip()

    def normalize_gold(self, gold: str) -> str:
        return gold

    @torch.no_grad()
    def evaluate(self, model, tokenizer, batch_size: int = 32, use_logprob: bool = True) -> float:
        dataset = self.load_eval_dataset()
        device = next(model.parameters()).device
        model.eval()

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        is_enc_dec = getattr(model.config, "is_encoder_decoder", False)
        use_logprob = use_logprob and not is_enc_dec
        use_chat_template = hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template is not None
        template_kwargs = get_chat_template_kwargs(tokenizer)

        records = []
        for ex in dataset:
            raw_prompt, gold, choices = self.example_to_prompt_gold_choices(ex, use_chat_template)
            if isinstance(raw_prompt, list):
                prompt = tokenizer.apply_chat_template(
                    raw_prompt,
                    tokenize=False,
                    add_generation_prompt=True,
                    **template_kwargs,
                )
            else:
                prompt = raw_prompt
            records.append((prompt, gold, choices))

        correct = 0
        evaluated = 0
        desc = f"Evaluating {self.dataset_label}" if self.dataset_label else "Evaluating"

        for i in tqdm(range(0, len(records), batch_size), desc=desc):
            batch = records[i : i + batch_size]

            if use_logprob:
                for prompt, gold, choices in batch:
                    logprob_choices = [format_label_for_prompt(prompt, c) for c in choices]
                    pred_label, _ = select_best_choice(prompt, logprob_choices, model, tokenizer)
                    pred = self.normalize_pred(pred_label)
                    norm_gold = self.normalize_gold(gold)
                    evaluated += 1
                    correct += int(pred == norm_gold)
            else:
                for prompt, gold, choices in batch:
                    pred_text = self._predict_generation(model, tokenizer, prompt, device)
                    pred = self.normalize_pred(pred_text)
                    norm_gold = self.normalize_gold(gold)
                    evaluated += 1
                    correct += int(pred == norm_gold)

        acc = float(correct / evaluated) if evaluated else 0.0
        logger.info("ACC %s: %.4f", self.dataset_label or self.__class__.__name__, acc)
        print(f"ACC: {acc:.4f}")
        return acc

    def _predict_generation(self, model, tokenizer, prompt: str, device, max_new_tokens: int = 10) -> str:
        enc = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=1024,
            add_special_tokens=False,
        ).to(device)
        gen = model.generate(
            **enc,
            do_sample=False,
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.eos_token_id,
        )
        if getattr(model.config, "is_encoder_decoder", False):
            out_ids = gen[0]
        else:
            out_ids = gen[0, enc["input_ids"].shape[1]:]
        return tokenizer.decode(out_ids, skip_special_tokens=True)
