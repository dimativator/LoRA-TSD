"""Pinned WebNLG2017 English and ViGGO official splits for generation."""

import csv
from collections import Counter
import hashlib
import json
from pathlib import Path
from typing import Callable

from datasets import Dataset


WEBNLG_REVISION = "c18b175ee040ae1254c37e540a244f75a06dc0a0"
VIGGO_REVISION = "c851cd5ff2ee92f0137fcf24014e37427a2d30b7"
SOURCES = {
    "webnlg": {
        "repository": "microsoft/LoRA", "revision": WEBNLG_REVISION,
        "base_url": f"https://raw.githubusercontent.com/microsoft/LoRA/{WEBNLG_REVISION}/examples/NLG/data/webnlg_challenge_2017/",
        "files": {
            "train": ("train.json", 20771939, "180edb6011360608c8548d4551366a9d55f140cbf9bc9397fbf4d97f6a89c639"),
            "validation": ("dev.json", 2596106, "50f5a2c24d760bc84ba3e852707cd3b43b0cf2da035b425f87945a2440880488"),
            "test": ("test.json", 5610377, "2574896b23b5959d1a0cd4bb501ffbeff1a11790a9996c4cb8efed289149ff4f"),
        },
    },
    "viggo": {
        "repository": "GEM/viggo", "revision": VIGGO_REVISION,
        "base_url": f"https://huggingface.co/datasets/GEM/viggo/resolve/{VIGGO_REVISION}/",
        "files": {
            "train": ("train.csv", 1370770, "39d4bf7ef5b7b78c1c137d05d461e779884ad08cf53e0ee6cf63c3653c5f8aae"),
            "validation": ("validation.csv", 187448, "c4d4b3b5075d84c1645cccd678e665e0d2a40226e3e9a383270a5c666006adbc"),
            "test": ("test.csv", 268969, "020f24199655b4c6f9246123263d2c6d3ea01189046150153dcc6cf8b914037a"),
        },
    },
}


def structured_messages(task: str, mr: str, reference: str | None = None) -> list[dict]:
    system = ("Generate a fluent English description of the facts in the given RDF triples."
              if task == "webnlg" else
              "Generate a conversational video game utterance from the given meaning representation, preserving its dialogue act and information.")
    prefix = "RDF triples:\n" if task == "webnlg" else "Meaning Representation:\n"
    messages = [{"role": "system", "content": system}, {"role": "user", "content": prefix + mr}]
    if reference is not None:
        messages.append({"role": "assistant", "content": reference})
    return messages


def _webnlg_rows(path: Path, split: str) -> tuple[list[dict], dict]:
    entries = json.loads(path.read_text(encoding="utf-8"))["entries"]
    rows = []
    comments = Counter()
    for index, wrapped in enumerate(entries):
        if len(wrapped) != 1:
            raise ValueError("Unexpected WebNLG entry wrapper")
        entry = next(iter(wrapped.values()))
        triples = [(triple["subject"], triple["property"], triple["object"])
                   for triple in entry["modifiedtripleset"]]
        if not triples or any(not isinstance(value, str) or not value.strip()
                              for triple in triples for value in triple):
            raise ValueError("Empty or malformed WebNLG triples")
        # Match Microsoft's converter's surface form. The grouping key treats
        # triple order as irrelevant, while prompts keep the first source order.
        mr = " | ".join(" : ".join(triple) for triple in triples)
        key = json.dumps(sorted(triples), ensure_ascii=False, separators=(",", ":"))
        for lexical_index, lexicalisation in enumerate(entry["lexicalisations"]):
            comments[lexicalisation["comment"]] += 1
            if lexicalisation.get("lang", "") not in {"", "en"}:
                raise ValueError("Non-English WebNLG lexicalisation")
            if lexicalisation["comment"] != "good":
                continue
            reference = lexicalisation["lex"]
            if not isinstance(reference, str) or not reference.strip():
                raise ValueError("Empty WebNLG reference")
            rows.append({"mr": mr, "key": key, "reference": reference,
                         "source_id": f"{split}:{index}:{lexical_index}",
                         "category": entry["category"]})
    return rows, {"raw_entries": len(entries), "raw_reference_rows": sum(comments.values()),
                  "reference_quality_counts": dict(comments),
                  "quality_filter": "comment == good, matching microsoft/LoRA format_converting_webnlg.py"}


def _viggo_rows(path: Path, split: str) -> tuple[list[dict], dict]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        raw = list(csv.DictReader(stream))
    rows = []
    for row in raw:
        if not row["mr"].strip() or not row["ref"].strip():
            raise ValueError("Empty ViGGO meaning representation or reference")
        rows.append({"mr": row["mr"], "key": row["mr"], "reference": row["ref"],
                     "source_id": row["gem_id"]})
    if len({row["source_id"] for row in rows}) != len(rows):
        raise ValueError("Duplicate ViGGO source IDs")
    return rows, {"raw_entries": len(raw), "raw_reference_rows": len(raw),
                  "quality_filter": "all official rows"}


def _group_examples(task: str, rows: list[dict], train_keys: set, validation_keys: set) -> list[dict]:
    groups = {}
    for row in rows:
        if row["key"] not in groups:
            groups[row["key"]] = {
                "messages": structured_messages(task, row["mr"]), "mr": row["mr"],
                "input_key": row["key"], "references": [], "source_ids": [],
                "train_overlap": row["key"] in train_keys,
                "validation_overlap": row["key"] in validation_keys,
            }
            if "category" in row:
                groups[row["key"]]["categories"] = []
        group = groups[row["key"]]
        if row["reference"] not in group["references"]:
            group["references"].append(row["reference"])
        group["source_ids"].append(row["source_id"])
        if "category" in row and row["category"] not in group["categories"]:
            group["categories"].append(row["category"])
    return list(groups.values())


def _lengths(values: list[str]) -> dict:
    def summary(numbers: list[int]) -> dict:
        numbers = sorted(numbers)
        return {"min": numbers[0], "median": numbers[len(numbers) // 2],
                "p95": numbers[min(len(numbers) - 1, int(len(numbers) * 0.95))],
                "max": numbers[-1]}
    return {"characters": summary([len(value) for value in values]),
            "whitespace_words": summary([len(value.split()) for value in values])}


def build_structured_data(dataset: str, data_dir: Path, guard: Callable) -> tuple[Dataset, Dataset, dict, dict]:
    """Guard must be bound to data_dir's filesystem; no tokenization or truncation here.

    Train/dev retain official rows. Test references are returned only in the
    separate final-evaluation payload and never appended to train/dev references.
    """
    # Lazy import permits generation_data.build_data to dispatch here without a cycle.
    from generation_data import _download, digest

    task = "webnlg" if dataset in {"webnlg", "webnlg2017"} else dataset
    if task not in SOURCES:
        raise ValueError("Unsupported structured generation dataset: " + dataset)
    source = SOURCES[task]
    directory = Path(data_dir) / task / source["revision"]
    splits = {}
    split_metadata = {}
    file_metadata = {}
    for split, (filename, expected_bytes, expected_sha) in source["files"].items():
        path = directory / filename
        _download(source["base_url"] + filename, path, guard, expected_bytes)
        actual_bytes = path.stat().st_size
        actual_sha = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual_bytes != expected_bytes or actual_sha != expected_sha:
            raise ValueError(f"Pinned {task}/{filename} checksum or size mismatch")
        reader = _webnlg_rows if task == "webnlg" else _viggo_rows
        rows, metadata = reader(path, split)
        if not rows:
            raise ValueError(f"Empty {task}/{split} split")
        splits[split] = rows
        split_metadata[split] = {
            **metadata, "selected_reference_rows": len(rows),
            "unique_inputs": len({row["key"] for row in rows}),
            "duplicate_input_reference_rows": len(rows) - len({(row["key"], row["reference"]) for row in rows}),
            "input_lengths": _lengths([row["mr"] for row in rows]),
            "reference_lengths": _lengths([row["reference"] for row in rows]),
            "rows_digest": digest(rows),
        }
        file_metadata[split] = {"url": source["base_url"] + filename,
                                "bytes": actual_bytes, "sha256": actual_sha}
    keys = {split: {row["key"] for row in rows} for split, rows in splits.items()}
    pairs = {split: {(row["key"], row["reference"]) for row in rows}
             for split, rows in splits.items()}
    overlap = {}
    for left, right in [("train", "validation"), ("train", "test"), ("validation", "test")]:
        overlap[f"{left}__{right}"] = {"exact_inputs": len(keys[left] & keys[right]),
                                       "exact_input_reference_pairs": len(pairs[left] & pairs[right])}
    dev = _group_examples(task, splits["validation"], keys["train"], keys["validation"])
    test = _group_examples(task, splits["test"], keys["train"], keys["validation"])
    for split, examples in [("validation", dev), ("test", test)]:
        split_metadata[split].update({
            "generation_examples": len(examples),
            "unique_references_per_input": dict(Counter(len(example["references"]) for example in examples)),
            "generation_examples_digest": digest(examples),
        })
    def supervised(rows: list[dict]) -> Dataset:
        return Dataset.from_list([{"messages": structured_messages(task, row["mr"], row["reference"])}
                                  for row in rows])
    manifest = {
        "dataset": task, "source": source["repository"], "revision": source["revision"],
        "source_files": file_metadata, "splits": split_metadata,
        "raw_split_sizes": {split: metadata["raw_reference_rows"] for split, metadata in split_metadata.items()},
        "split_overlaps": overlap,
        "split_policy": "Official splits preserved without decontamination; test is not strictly input-disjoint.",
        "overlap_flags": "train_overlap means an input occurs in train; validation_overlap means it occurs in dev (therefore true for dev examples themselves).",
        "grouping": ("Exact RDF triple-set key ignoring triple order, first source order for generation prompt"
                     if task == "webnlg" else "Exact original meaning representation string"),
        "reference_policy": "All selected rows train/dev loss; grouped generation refs deduplicated within their own split only.",
        "final_test": "test_examples reserved for final evaluation, excluded from monitoring and model selection",
        "length_policy": "No truncation or subsampling in this loader; character/word lengths are not token lengths.",
        "max_new_tokens": 128,
    }
    manifest["manifest_digest"] = digest(manifest)
    return supervised(splits["train"]), supervised(splits["validation"]), {
        "task": task, "examples": dev, "test_examples": test, "max_new_tokens": 128,
    }, manifest
