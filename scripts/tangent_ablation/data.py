"""Pinned BoolQ/SIQA parquet with the existing benchmark prompts and evaluator."""

import hashlib
import json
import os
from pathlib import Path
from typing import Callable

import pyarrow.parquet as pq
import torch
from datasets import Dataset
from huggingface_hub import get_hf_file_metadata, hf_hub_download, hf_hub_url
from torch.utils.data import DataLoader

from data.boolq import BoolQEval, _example_to_prompt_and_gold as boolq_example
from data.data_common import GenericSFTCollator
from data.siqa import SIQAEval, _example_to_prompt_and_gold as siqa_example


# Copied from the established Cloud benchmark_data_manifest.json.
SOURCES = {
    "boolq": {
        "repo": "google/boolq",
        "revision": "d89fc6ef4da46f056176dcf941ba670524ab1c87",
        "train": ("default/train/0000.parquet", 3685146, 9427),
        "validation": ("default/validation/0000.parquet", 1257630, 3270),
    },
    "siqa": {
        "repo": "allenai/social_i_qa",
        "revision": "537a2ec8ec565adc0b70b70752893e59e024df26",
        "train": ("default/train/0000.parquet", 3475355, 33410),
        "validation": ("default/validation/0000.parquet", 203280, 1954),
    },
}


def load_split(dataset: str, split: str, root: Path, guard: Callable) -> tuple:
    source = SOURCES[dataset]
    filename, expected_bytes, expected_rows = source[split]
    cache = root / dataset / source["revision"]
    guard(pending_write_bytes=expected_bytes + 1_000_000,
          reason=f"before_{dataset}_{split}_download", write_path=root)
    path = Path(hf_hub_download(
        source["repo"], filename, repo_type="dataset", revision=source["revision"],
        local_dir=cache, token=os.environ.get("HF_TOKEN"),
    ))
    if path.stat().st_size != expected_bytes:
        raise RuntimeError(f"Pinned {dataset}/{split} file size mismatch")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    metadata = get_hf_file_metadata(hf_hub_url(
        source["repo"], filename, repo_type="dataset", revision=source["revision"],
    ), token=os.environ.get("HF_TOKEN"))
    checksum = metadata.etag.strip('"') if metadata.etag else None
    if checksum is not None and len(checksum) == 64 and checksum != digest:
        raise RuntimeError(f"Pinned {dataset}/{split} SHA256 mismatch")
    table = pq.read_table(path)
    if table.num_rows != expected_rows:
        raise RuntimeError(f"Pinned {dataset}/{split} row count mismatch")
    result = Dataset(table)
    record = {
        "repo": source["repo"], "revision": source["revision"], "filename": filename,
        "bytes": expected_bytes, "rows": expected_rows, "sha256": digest,
        "upstream_etag": checksum, "fingerprint": result._fingerprint,
    }
    return result, record


def build_data(tokenizer, dataset: str, seed: int, microbatch: int,
               root: Path, guard: Callable, eval_max_samples: int | None = None) -> tuple:
    root.mkdir(parents=True, exist_ok=True)
    train, train_record = load_split(dataset, "train", root, guard)
    validation, validation_record = load_split(dataset, "validation", root, guard)
    if eval_max_samples is not None:
        validation = validation.select(range(min(eval_max_samples, len(validation))))
    example = boolq_example if dataset == "boolq" else siqa_example
    collator = GenericSFTCollator(tokenizer, example, max_length=1024)
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(train, batch_size=microbatch, shuffle=True,
                        generator=generator, collate_fn=collator,
                        pin_memory=torch.cuda.is_available(), drop_last=True)
    evaluator_class = BoolQEval if dataset == "boolq" else SIQAEval

    class PinnedEvaluator(evaluator_class):
        def load_eval_dataset(self):
            return validation

    manifest = {"dataset": dataset, "train": train_record,
                "validation": validation_record, "evaluated_rows": len(validation),
                "max_train_length": 1024, "drop_last": True,
                "prompt_and_eval": "existing repository chat prompt and mean choice-token logprob"}
    manifest["manifest_sha256"] = hashlib.sha256(
        json.dumps(manifest, sort_keys=True).encode()).hexdigest()
    return loader, PinnedEvaluator(), manifest
