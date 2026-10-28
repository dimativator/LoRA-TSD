"""CPU preflight for pinned WebNLG/ViGGO data and requested optimizers."""

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import signal
import sys
import threading
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "cloud"))

from disk_guard import DiskGuard
from generation_preflight import inventory


METHODS = ("lora_tsd", "adamw", "lora_muon")
DATASETS = ("webnlg", "viggo")


class StorageWatchdog:
    """Protect CPU downloads/tokenization independently of their main thread."""

    def __init__(self, root: Path, telemetry: Path):
        self.root = root
        self.guard = DiskGuard(root, "structured_cpu_smoke_watchdog", telemetry)
        self.finished = threading.Event()
        self.probe = {"active": False, "started": 0.0, "error": None}
        self.thread = threading.Thread(target=self.monitor, daemon=True)

    def check(self) -> None:
        try:
            self.guard()
        except BaseException as error:
            self.probe["error"] = str(error)
        finally:
            self.probe["active"] = False

    def abort(self, error: str) -> None:
        record = {"timestamp": time.time(), "run": "structured_cpu_smoke",
                  "pid": os.getpid(), "status": "stop", "error": error}

        def publish() -> None:
            print("CRITICAL CLOUD.RU STORAGE: " + json.dumps(record), flush=True)
            try:
                with (self.root / "STOP_STORAGE.json").open("x") as stream:
                    json.dump(record, stream)
            except FileExistsError:
                pass
            except OSError as failure:
                print("STORAGE STOP SIGNAL WRITE FAILED: " + str(failure), flush=True)

        # Neither unavailable NFS nor blocked log streaming may delay the stop.
        threading.Thread(target=publish, daemon=True).start()
        os.kill(os.getpid(), signal.SIGTERM)

    def monitor(self) -> None:
        next_probe = 0.0
        while not self.finished.wait(0.5):
            now = time.monotonic()
            if self.probe["error"] is not None:
                self.abort(self.probe["error"])
                return
            if self.probe["active"] and now - self.probe["started"] > 15:
                self.abort("Storage telemetry unavailable: CPU probe exceeded 15 seconds")
                return
            if not self.probe["active"] and now >= next_probe:
                self.probe.update(active=True, started=now)
                threading.Thread(target=self.check, daemon=True).start()
                next_probe = now + 5

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, exc_type, exc, traceback):
        self.finished.set()
        self.thread.join(timeout=1)


def token_lengths(tokenizer, dataset: str, data_dir: Path, max_length: int,
                  guard) -> dict:
    """Scan full native chat sequences from already downloaded official rows."""
    from structured_data import SOURCES, _viggo_rows, _webnlg_rows, structured_messages

    source = SOURCES[dataset]
    reader = _webnlg_rows if dataset == "webnlg" else _viggo_rows
    summary = {}
    for split, (filename, _, _) in source["files"].items():
        guard(reason="native_token_length_scan:" + split)
        rows, _ = reader(data_dir / dataset / source["revision"] / filename, split)
        lengths = []
        longest_source_id = None
        longest = -1
        for index, row in enumerate(rows):
            if index % 128 == 0:
                guard()
            messages = structured_messages(dataset, row["mr"], row["reference"])
            length = len(tokenizer.apply_chat_template(
                messages, tokenize=True, add_generation_prompt=False))
            lengths.append(length)
            if length > longest:
                longest = length
                longest_source_id = row["source_id"]
        ordered = sorted(lengths)
        summary[split] = {
            "examples": len(lengths), "min": ordered[0],
            "median": ordered[len(ordered) // 2],
            "p95": ordered[min(len(ordered) - 1, int(len(ordered) * 0.95))],
            "max": ordered[-1], "truncated_count": sum(length > max_length for length in lengths),
            "tokens_beyond_limit": sum(max(0, length - max_length) for length in lengths),
            "longest_source_id": longest_source_id,
        }
    return {"max_length": max_length,
            "unit": "Official selected reference rows, including duplicate input/reference rows",
            "encoding": "Full native chat with assistant reference and end-of-turn token, before truncation",
            "splits": summary}


def scorer_smoke() -> dict:
    """Real sacreBLEU checks for perfect predictions and uneven multireferences."""
    import sacrebleu
    from generation_eval import _reference_streams, check_evaluation_dependencies

    examples = [
        {"references": ["Alpha is a French town beside a wide river.",
                         "A French town named Alpha lies beside a river."]},
        {"references": ["Beta is a multiplayer video game for four players."]},
    ]
    references = _reference_streams(examples)
    assert references == [[row["references"][0] for row in examples],
                          [examples[0]["references"][1], examples[1]["references"][0]]]
    predictions = [row["references"][0] for row in examples]
    scores = {}
    for dataset in DATASETS:
        check_evaluation_dependencies({"task": dataset}, lambda **kwargs: None)
        bleu = sacrebleu.metrics.BLEU()
        chrf = sacrebleu.metrics.CHRF(word_order=2)
        bleu_score = bleu.corpus_score(predictions, references).score
        chrf_score = chrf.corpus_score(predictions, references).score
        assert bleu_score > 99.9 and chrf_score > 99.9, (bleu_score, chrf_score)
        scores[dataset] = {"bleu": bleu_score, "chrfpp": chrf_score,
                           "bleu_signature": str(bleu.get_signature()),
                           "chrfpp_signature": str(chrf.get_signature())}
    return scores


def save_result(directory: Path, result: dict, guard) -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    path = directory / ("structured-smoke-" + stamp + ".json")
    payload = json.dumps(result, indent=2, allow_nan=False) + "\n"
    guard(pending_write_bytes=len(payload.encode()) + 20000,
          reason="cpu_smoke_result", write_path=directory)
    with path.open("x") as stream:
        stream.write(payload)
    return path


def main() -> None:
    from optimizer_presets import METHODS as available_methods

    parser = argparse.ArgumentParser()
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--methods", nargs="+", choices=available_methods, default=list(METHODS))
    args = parser.parse_args()
    if args.max_length <= 0:
        parser.error("max-length must be positive")
    if args.steps < 2:
        parser.error("steps must be at least two for the two-update smoke")
    if len(args.methods) != len(set(args.methods)):
        parser.error("Optimizer methods must occur only once")
    methods = tuple(args.methods)
    root = Path(os.environ["GENERATION_ROOT"])
    if "nfs2" in str(root).lower() or not root.is_relative_to("/home/jovyan/shares/SR006.nfs3"):
        raise ValueError("CPU smoke outputs must use the campaign root on NFS3")
    print("GEN_PREFLIGHT " + json.dumps(inventory()), flush=True)
    guard = DiskGuard(root, "structured_cpu_smoke")
    guard(pending_write_bytes=1_000_000, reason="cpu_smoke_log_directory")
    directory = root / "cpu-smoke"
    directory.mkdir(parents=True, exist_ok=True)
    guard.telemetry = directory / "disk-trainer.jsonl"
    data_dir = Path(os.environ.get("GENERATION_DATA_DIR", "/tmp/generation-data"))
    if not data_dir.resolve().is_relative_to(Path("/tmp").resolve()):
        raise ValueError("CPU smoke data and tokenization caches must use /tmp")

    def data_guard(pending_write_bytes: int = 0, reason: str = "periodic") -> dict:
        return guard(pending_write_bytes=pending_write_bytes, reason=reason, write_path=data_dir)

    data_guard(pending_write_bytes=300_000_000, reason="cpu_tokenizer_and_data_directories")
    data_dir.mkdir(parents=True, exist_ok=True)
    os.environ["HF_DATASETS_CACHE"] = str(data_dir / "datasets-cache")
    tokenizer_cache = data_dir / "tokenizer-cache"
    result = {"status": "running", "timestamp": time.time(), "datasets": {},
              "methods": list(methods), "max_length": args.max_length,
              "production_steps": args.steps,
              "checkpoints": False, "nfs2_accessed": False,
              "optimizer_losses": {}, "optimizer_smoke_rsvd_q": 1,
              "production_rsvd_q": 15, "optimizer_smoke_steps": 2}
    with StorageWatchdog(root, directory / "disk-watchdog.jsonl"):
        try:
            from transformers import AutoTokenizer
            from cpu_smoke import optimizer_smoke
            from generation_data import build_data, build_test_loader

            data_guard(pending_write_bytes=100_000_000, reason="native_tokenizer_cache")
            tokenizer = AutoTokenizer.from_pretrained(
                "meta-llama/Llama-3.2-1B-Instruct", token=os.environ.get("HF_TOKEN"),
                cache_dir=str(tokenizer_cache))
            if tokenizer.pad_token_id is None:
                tokenizer.pad_token = tokenizer.eos_token
            result["tokenizer"] = {"model": "meta-llama/Llama-3.2-1B-Instruct",
                                   "native_chat_template": True, "padding_side": tokenizer.padding_side}
            result["scorer_smoke"] = scorer_smoke()
            violations = []
            for dataset in DATASETS:
                train, validation, generation, manifest = build_data(
                    tokenizer, dataset, 42, args.max_length, 4, data_dir, data_guard)
                assert manifest["seed"] == 42 and train.generator.initial_seed() == 42
                lengths = token_lengths(tokenizer, dataset, data_dir, args.max_length, data_guard)
                entry = {"manifest": manifest, "token_lengths": lengths}
                result["datasets"][dataset] = entry
                oversized = {split: stats for split, stats in lengths["splits"].items()
                             if stats["truncated_count"]}
                if oversized:
                    violations.append({"dataset": dataset, "splits": oversized})
                else:
                    test, test_manifest = build_test_loader(
                        tokenizer, generation, args.max_length, 4, data_guard)
                    entry["test_manifest"] = test_manifest
                    assert len(train.dataset) and len(validation.dataset) and len(test.dataset)
                    batch = next(iter(train))
                    assert ((batch["labels"] != -100) & (batch["attention_mask"] == 0)).sum() == 0
                    assert int((batch["labels"][:, 1:] != -100).sum()) > 0
                print("GEN_STRUCTURED_DATA " + json.dumps({"dataset": dataset, **entry}), flush=True)
                del train, validation, generation
            if violations:
                result["length_violations"] = violations
                raise ValueError("Native chat exceeds configured sequence length; no GPU launch: " +
                                 json.dumps(violations, separators=(",", ":")))
            checks = optimizer_smoke(methods, steps=args.steps)
            if set(checks) != set(methods):
                raise AssertionError("Optimizer smoke results must match the requested methods")
            result["optimizer_preset_steps"] = {
                method: checks[method]["preset"]["steps"] for method in methods}
            if any(steps != args.steps for steps in result["optimizer_preset_steps"].values()):
                raise AssertionError("Optimizer presets must use the requested production steps")
            result["optimizer_losses"] = {method: checks[method]["losses"] for method in methods}
            result["optimizer_checks"] = {
                method: {key: value for key, value in checks[method].items() if key not in {"preset", "losses"}}
                for method in methods}
            result["status"] = "passed"
        except Exception as error:
            result.update(status="failed", error=str(error))
            path = save_result(directory, result, guard)
            print("GEN_CPU_SMOKE " + json.dumps({"status": "failed", "methods": list(methods),
                                                 "production_steps": args.steps,
                                                 "error": str(error),
                                                 "result_path": str(path),
                                                 "token_lengths": {key: value["token_lengths"]
                                                                   for key, value in result["datasets"].items()}}), flush=True)
            raise
        path = save_result(directory, result, guard)
        summary = {key: value for key, value in result.items() if key != "datasets"}
        summary["result_path"] = str(path)
        summary["data_manifests"] = {
            dataset: {key: entry["manifest"][key] for key in
                      ("source", "revision", "manifest_digest", "raw_train_digest",
                       "train_tokenized_fingerprint", "validation_tokenized_fingerprint")}
            for dataset, entry in result["datasets"].items()}
        summary["token_lengths"] = {dataset: entry["token_lengths"]
                                    for dataset, entry in result["datasets"].items()}
        print("GEN_CPU_SMOKE " + json.dumps(summary, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
