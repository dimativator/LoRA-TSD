"""Reproducible generation data and assistant-only chat supervision."""

import csv
import fcntl
import hashlib
import json
from pathlib import Path
from typing import Callable
import urllib.request

import torch
from datasets import Dataset, load_dataset
from torch.utils.data import DataLoader


IFEVAL_REVISION = "e6890f85757dd84e27ca6df2dd30651dafad28e0"
SMOL_REVISION = "f73fe857d519ff6ac5af2ea67c4d3834da7b8bcc"
E2E_SYSTEM = "Generate a fluent restaurant description from the given meaning representation."


def digest(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def encode_messages(tokenizer, messages: list[dict], max_length: int) -> dict:
    """Supervise every assistant body and its end token, never chat headers."""
    full_ids = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=False)
    labels = [-100] * len(full_ids)
    for index, message in enumerate(messages):
        if message["role"] != "assistant":
            continue
        prefix = tokenizer.apply_chat_template(
            messages[:index], tokenize=True, add_generation_prompt=True,
        )
        completed = tokenizer.apply_chat_template(
            messages[:index + 1], tokenize=True, add_generation_prompt=False,
        )
        if completed[:len(prefix)] != prefix or full_ids[:len(completed)] != completed:
            raise ValueError("Chat template does not preserve assistant prefix boundaries")
        labels[len(prefix):len(completed)] = full_ids[len(prefix):len(completed)]
    ids = full_ids[:max_length]
    labels = labels[:max_length]
    return {"input_ids": ids, "labels": labels, "attention_mask": [1] * len(ids)}


class ChatCollator:
    """Right padding for supervised training, with padding excluded from loss."""

    def __init__(self, tokenizer) -> None:
        self.pad_token_id = tokenizer.pad_token_id

    def __call__(self, batch: list[dict]) -> dict[str, torch.Tensor]:
        return {
            key: torch.nn.utils.rnn.pad_sequence(
                [torch.tensor(row[key], dtype=torch.long) for row in batch],
                batch_first=True, padding_value=-100 if key == "labels" else (
                    self.pad_token_id if key == "input_ids" else 0
                ),
            )
            for key in ["input_ids", "labels", "attention_mask"]
        }


def _download(url: str, path: Path, guard: Callable, estimate: int) -> None:
    if path.exists():
        return
    guard(pending_write_bytes=estimate, reason="download:" + path.name)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    try:
        with urllib.request.urlopen(url, timeout=60) as source, temporary.open("wb") as target:
            while chunk := source.read(1024 * 1024):
                guard(pending_write_bytes=len(chunk), reason="download_chunk:" + path.name)
                target.write(chunk)
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def e2e_messages(mr: str, reference: str | None = None) -> list[dict]:
    messages = [
        {"role": "system", "content": E2E_SYSTEM},
        {"role": "user", "content": "Meaning Representation:\n" + mr},
    ]
    if reference is not None:
        messages.append({"role": "assistant", "content": reference})
    return messages


def _e2e_data(data_dir: Path, guard: Callable) -> tuple[Dataset, Dataset, dict, dict]:
    splits = {}
    hashes = {}
    for split, filename in [("train", "trainset.csv"), ("validation", "devset.csv"),
                            ("test", "testset_w_refs.csv")]:
        path = data_dir / "e2e" / filename
        _download("https://raw.githubusercontent.com/tuetschek/e2e-dataset/master/" + filename,
                  path, guard, 30_000_000)
        hashes[split] = hashlib.sha256(path.read_bytes()).hexdigest()
        with path.open(newline="") as stream:
            splits[split] = list(csv.DictReader(stream))
    def as_dataset(rows: list[dict]) -> Dataset:
        return Dataset.from_list([
            {"messages": e2e_messages(row["mr"], row["ref"])} for row in rows
        ])
    grouped = {}
    for row in splits["validation"]:
        grouped.setdefault(row["mr"], []).append(row["ref"])
    examples = [{"messages": e2e_messages(mr), "mr": mr, "references": refs}
                for mr, refs in grouped.items()]
    return as_dataset(splits["train"]), as_dataset(splits["validation"]), {
        "task": "e2e_nlg", "examples": examples,
    }, {"source": "tuetschek/e2e-dataset", "source_sha256": hashes,
        "raw_split_sizes": {key: len(value) for key, value in splits.items()},
        "generation_validation_unique_mrs": len(examples),
        "final_test": "Downloaded separately, never used for monitoring or selection"}


def _prepare_ifeval(data_dir: Path, guard: Callable) -> tuple[Path, list[dict]]:
    root = data_dir / "ifeval" / IFEVAL_REVISION
    package = root / "instruction_following_eval"
    files = ["__init__.py", "evaluation_lib.py", "instructions.py", "instructions_util.py",
             "instructions_registry.py", "data/input_data.jsonl"]
    for name in files:
        path = package / name
        if name == "__init__.py":
            guard(pending_write_bytes=0, reason="ifeval_package")
            path.parent.mkdir(parents=True, exist_ok=True)
            path.touch(exist_ok=True)
        else:
            _download("https://raw.githubusercontent.com/google-research/google-research/"
                      + IFEVAL_REVISION + "/instruction_following_eval/" + name,
                      path, guard, 1_000_000)
    examples = [json.loads(line) for line in (package / "data/input_data.jsonl").read_text().splitlines()]
    return root, examples


def _smol_data(data_dir: Path, guard: Callable) -> tuple[Dataset, Dataset, dict, dict]:
    revision_path = data_dir / "smol_revision.json"
    if revision_path.exists():
        revision = json.loads(revision_path.read_text())["revision"]
    else:
        revision = SMOL_REVISION
        guard(pending_write_bytes=1000, reason="dataset_revision")
        revision_path.write_text(json.dumps({"revision": revision}) + "\n")
    guard(pending_write_bytes=5_000_000_000, reason="smol_download_and_arrow_cache")
    data = load_dataset("HuggingFaceTB/smol-smoltalk", revision=revision,
                        cache_dir=str(data_dir / "datasets"))
    train = data["train"].shuffle(seed=42).select(range(100000))
    validation = data["test"].shuffle(seed=42).select(range(1000))
    root, prompts = _prepare_ifeval(data_dir, guard)
    generation = {"task": "ifeval", "ifeval_root": str(root), "examples": [
        {**example, "messages": [{"role": "user", "content": example["prompt"]}]}
        for example in prompts
    ]}
    prompt_strings = {row["prompt"].strip() for row in prompts}
    overlaps = []
    for index, row in enumerate(train):
        if any(message["role"] == "user" and message["content"].strip() in prompt_strings
               for message in row["messages"]):
            overlaps.append(index)
    if overlaps:
        raise ValueError(f"Exact IFEval prompt overlap in selected training data: {overlaps[:10]}")
    return train, validation, generation, {
        "source": "HuggingFaceTB/smol-smoltalk", "revision": revision,
        "selection_seed": 42, "selected_train_examples": len(train),
        "selected_validation_examples": len(validation), "validation_split": "official test",
        "selection_fingerprint": train._fingerprint,
        "train_source_indices_digest": digest(train._indices.column("indices").to_pylist()),
        "validation_fingerprint": validation._fingerprint,
        "validation_source_indices_digest": digest(validation._indices.column("indices").to_pylist()),
        "ifeval_revision": IFEVAL_REVISION, "ifeval_prompts": len(prompts),
        "ifeval_exact_prompt_train_overlap_count": len(overlaps),
        "overlap_check_limitation": "Exact user-message matching only, not semantic deduplication",
    }


def build_data(tokenizer, dataset: str, seed: int, max_length: int, microbatch: int,
               data_dir, guard: Callable) -> tuple[DataLoader, DataLoader, dict, dict]:
    """Return identical fixed datasets across optimizers with seeded shuffle."""
    data_dir = Path(data_dir)
    guard(pending_write_bytes=0, reason="data_directory")
    data_dir.mkdir(parents=True, exist_ok=True)
    with (data_dir / "prepare.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if dataset == "e2e_nlg":
            train, validation, generation, manifest = _e2e_data(data_dir, guard)
        elif dataset == "smol_smoltalk_100k":
            train, validation, generation, manifest = _smol_data(data_dir, guard)
        elif dataset in {"webnlg", "viggo"}:
            from structured_data import build_structured_data
            train, validation, generation, manifest = build_structured_data(dataset, data_dir, guard)
        else:
            raise ValueError("Unsupported generation dataset: " + dataset)
        manifest["raw_train_digest"] = digest({"fingerprint": train._fingerprint,
                                               "count": len(train)})
        processed = []
        for split_name, split in [("train", train), ("validation", validation)]:
            guard(pending_write_bytes=len(split) * max_length * 24,
                  reason="tokenization_cache:" + split_name)
            encoded = split.map(
                lambda row: encode_messages(tokenizer, row["messages"], max_length),
                remove_columns=split.column_names, keep_in_memory=False,
                desc="Tokenizing " + split_name,
            )
            valid = encoded.filter(lambda row: any(label != -100 for label in row["labels"][1:]),
                                   desc="Removing samples without supervised tokens")
            manifest[split_name + "_truncated_or_empty_removed"] = len(encoded) - len(valid)
            manifest[split_name + "_examples"] = len(valid)
            processed.append(valid)
        manifest.update({"max_length": max_length, "assistant_only_all_turns": True,
                         "seed": seed, "train_tokenized_fingerprint": processed[0]._fingerprint,
                         "validation_tokenized_fingerprint": processed[1]._fingerprint})
        manifest["manifest_digest"] = digest(manifest)
        guard(pending_write_bytes=20000, reason="data_manifest")
        (data_dir / f"{dataset}_length{max_length}_manifest.json").write_text(
            json.dumps(manifest, indent=2) + "\n")
    generator = torch.Generator().manual_seed(seed)
    collator = ChatCollator(tokenizer)
    train_loader = DataLoader(processed[0], batch_size=microbatch, shuffle=True,
                              generator=generator, collate_fn=collator, num_workers=0,
                              drop_last=True, pin_memory=torch.cuda.is_available())
    val_loader = DataLoader(processed[1], batch_size=microbatch, shuffle=False,
                            collate_fn=collator, num_workers=0)
    return train_loader, val_loader, generation, manifest


def build_test_loader(tokenizer, generation: dict, max_length: int, microbatch: int,
                      guard: Callable) -> tuple[DataLoader, dict]:
    """Prepare test targets for evaluation only, without adding them to training."""
    examples = generation["test_examples"]
    records = [{"messages": row["messages"] + [{"role": "assistant", "content": reference}]}
               for row in examples for reference in row["references"]]
    guard(pending_write_bytes=len(records) * max_length * 24, reason="test_tokenization")
    encoded = Dataset.from_list(records).map(
        lambda row: encode_messages(tokenizer, row["messages"], max_length),
        remove_columns=["messages"], keep_in_memory=True, desc="Tokenizing test targets")
    if any(not any(label != -100 for label in row["labels"][1:]) for row in encoded):
        raise ValueError("Test target lost all supervision at the configured sequence length")
    loader = DataLoader(encoded, batch_size=microbatch, shuffle=False,
                        collate_fn=ChatCollator(tokenizer), num_workers=0)
    return loader, {"test_target_examples": len(encoded), "test_generation_inputs": len(examples),
                    "test_tokenized_fingerprint": encoded._fingerprint,
                    "test_use": "Final evaluation only, never optimization or model selection"}
