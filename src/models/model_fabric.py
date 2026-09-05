import os

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


class LoRAModelFabric:
    MODEL_ALIASES = {
        "meta-llama/llama-3.2-1b-instruct": "meta-llama/Llama-3.2-1B-Instruct",
        "llama-3.2-1b-instruct": "meta-llama/Llama-3.2-1B-Instruct",
        "meta-llama/llama-3.1-8b": "meta-llama/Llama-3.1-8B",
        "llama-3.1-8b": "meta-llama/Llama-3.1-8B",
        "qwen/qwen3-32b": "Qwen/Qwen3-32B",
        "qwen3-32b": "Qwen/Qwen3-32B",
    }

    @staticmethod
    def apply_basic_config(model, tokenizer):
        if tokenizer.eos_token is None:
            tokenizer.add_special_tokens({"eos_token": "<|endoftext|>"})
            model.resize_token_embeddings(len(tokenizer))
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model.config.pad_token_id = tokenizer.pad_token_id
        model.config.eos_token_id = tokenizer.eos_token_id
        model.config.bos_token_id = tokenizer.bos_token_id
        model.generation_config.pad_token_id = tokenizer.pad_token_id
        model.generation_config.eos_token_id = tokenizer.eos_token_id
        model.generation_config.bos_token_id = tokenizer.bos_token_id
        return model, tokenizer

    @classmethod
    def get_model(cls, model_name):
        canonical_name = cls.MODEL_ALIASES.get(model_name.strip().lower())
        if canonical_name is None:
            supported = ", ".join(sorted(set(cls.MODEL_ALIASES.values())))
            raise ValueError(f"Unsupported model '{model_name}'. Supported: {supported}")

        token = os.environ.get("HF_TOKEN")
        tokenizer = AutoTokenizer.from_pretrained(
            canonical_name,
            trust_remote_code=True,
            token=token,
        )
        tokenizer.padding_side = "left"

        load_kwargs = {
            "torch_dtype": torch.bfloat16,
            "low_cpu_mem_usage": True,
            "use_safetensors": True,
            "device_map": {"": 0},
            "trust_remote_code": True,
            "token": token,
        }

        if canonical_name == "Qwen/Qwen3-32B":
            quantization = os.environ.get("QUANT", "none").strip().lower()
            if quantization in {"fp8", "float8"}:
                try:
                    from torchao.quantization import Float8WeightOnlyConfig
                    from transformers import TorchAoConfig
                except ImportError as error:
                    raise ImportError(
                        "Qwen3-32B FP8 loading requires torchao==0.16.0 with "
                        "torch==2.8.*."
                    ) from error
                load_kwargs["quantization_config"] = TorchAoConfig(
                    quant_type=Float8WeightOnlyConfig(),
                )
            elif quantization not in {"", "none", "bf16", "bfloat16"}:
                raise ValueError(
                    f"Unsupported QUANT={quantization!r}. Use 'none' or 'fp8'."
                )

            max_gpu_memory = os.environ.get("MAX_GPU_MEM")
            if max_gpu_memory:
                load_kwargs["max_memory"] = {0: max_gpu_memory}
            load_kwargs["attn_implementation"] = os.environ.get(
                "ATTN_IMPL", "sdpa"
            )

        model = AutoModelForCausalLM.from_pretrained(
            canonical_name,
            **load_kwargs,
        )
        return cls.apply_basic_config(model, tokenizer)
