"""Print compact summaries from the persistent SIQA log directory."""

import json
import fcntl
import os
import math
import shutil
import subprocess
from collections import deque
from pathlib import Path

root = Path(os.environ.get("LORA_SIQA_LOG_ROOT", "/home/jovyan/lora-siqa-retractions-20261011"))
root.mkdir(parents=True, exist_ok=True)
for path in sorted(root.glob("*/run.json")):
    if (path.parent / "metrics.json").exists():
        continue
    state = json.loads(path.read_text())
    history_path = path.parent / "history.jsonl"
    recent = []
    if history_path.exists():
        with history_path.open() as stream:
            recent = [json.loads(line) for line in deque(stream, maxlen=2)]
    validation = []
    nonfinite_steps = []
    if history_path.exists():
        with history_path.open() as stream:
            for line in stream:
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue  # The worker may still be writing the last record.
                if "validation_score" in row:
                    validation.append(row)
                if "train/loss" in row and not math.isfinite(row["train/loss"]):
                    nonfinite_steps.append(row.get("step"))
    log_path = path.parent / "train.log"
    tail = []
    if log_path.exists():
        with log_path.open() as stream:
            tail = list(deque(stream, maxlen=3))
    print("BENCHMARK_PROGRESS " + json.dumps({"name": state["name"], "returncode": state.get("returncode"),
                                         "eval_every": state["command"][state["command"].index("--eval-every") + 1],
                                         "recent_metrics": recent, "validation": validation,
                                         "nonfinite_steps": nonfinite_steps, "log_tail": tail}), flush=True)
for path in sorted(root.glob("*/metrics.json")):
    metrics = json.loads(path.read_text())
    history_path = path.parent / "history.jsonl"
    rows = [json.loads(line) for line in history_path.read_text().splitlines()] if history_path.exists() else []
    metrics["cuda_peak_allocated_bytes"] = max((r.get("cuda_peak_allocated_bytes", 0) for r in rows), default=0)
    metrics["last_losses"] = [r["train/loss"] for r in rows if "train/loss" in r][-5:]
    state = json.loads((path.parent / "run.json").read_text())
    losses = [r for r in rows if "train/loss" in r]
    fallback_rows = [r for r in losses if r.get("train/retraction_fallback_layers", 0)]
    lr_rows = [r for r in losses if "lr_A" in r]
    train_steps = [r["step"] for r in losses]
    command = state["command"]
    lr_check = None
    if "--lr-scheduler" in command and command[command.index("--lr-scheduler") + 1] == "linear":
        total_steps = int(command[command.index("--steps") + 1])
        base_lr = float(command[command.index("--lr-a") + 1])
        min_lr = float(command[command.index("--min-lr") + 1]) if "--min-lr" in command else 0.0
        warmup_steps = max(1, round(0.1 * total_steps))
        mismatches = []
        for row in lr_rows:
            next_step = min(row["step"] + 1, total_steps)
            expected = base_lr * next_step / warmup_steps if next_step <= warmup_steps else base_lr + (min_lr - base_lr) * (next_step - warmup_steps) / max(1, total_steps - warmup_steps)
            if not math.isclose(row["lr_A"], expected, rel_tol=1e-9, abs_tol=1e-12):
                mismatches.append(row["step"])
        lr_check = {"record_count": len(lr_rows), "mismatch_count": len(mismatches), "first_mismatch_steps": mismatches[:10],
                    "warmup_steps": warmup_steps, "logged_lr_is_next_update": True,
                    "samples": [{"step": r["step"], "lr_A": r["lr_A"]} for r in lr_rows if r["step"] in {1, warmup_steps - 1, warmup_steps, total_steps - 1, total_steps}]}
    metrics["verification"] = {
        "returncode": state.get("returncode"),
        "command": state["command"],
        "train_record_count": len(losses),
        "train_steps_contiguous": train_steps == list(range(1, len(losses) + 1)),
        "last_train_step": max((r["step"] for r in losses), default=None),
        "nonfinite_train_steps": [r["step"] for r in losses if not math.isfinite(r["train/loss"])],
        "validation": [r for r in rows if "validation_score" in r],
        "retraction_fallbacks": {
            "record_count": sum("train/retraction_fallback_layers" in r for r in losses),
            "total_layer_events": sum(r["train/retraction_fallback_layers"] for r in fallback_rows),
            "steps": [{"step": r["step"], "layers": r["train/retraction_fallback_layers"]} for r in fallback_rows],
        },
        "lr_check": lr_check,
    }
    print("BENCHMARK_METRICS " + json.dumps(metrics), flush=True)
print("BENCHMARK_CHECKPOINT_COUNT " + str(len(list(root.rglob("*.pt")))))

# Only caches generated from this campaign's own private parquet paths qualify.
data_root = Path('/home/jovyan/lora-1b-benchmark-data-20261014')
cache_root = Path(os.environ.get('HF_HOME', '/home/jovyan/.cache/huggingface')) / 'datasets/parquet'
removed_cache_bytes = 0
if os.environ.get('LORA_BENCHMARK_CLEANUP_ARROW') == '1':
    for info_path in cache_root.glob('**/dataset_info.json'):
        metadata = json.loads(info_path.read_text())
        sources = list((metadata.get('download_checksums') or {}).keys())
        if not sources or not all(source.startswith(str(data_root) + '/') and source.endswith('.parquet') for source in sources):
            continue
        directory = info_path.parent
        if not directory.resolve().is_relative_to(cache_root.resolve()):
            raise ValueError('Unexpected cache path')
        removed_cache_bytes += sum(path.stat().st_size for path in directory.rglob('*') if path.is_file())
        shutil.rmtree(directory)
    print('BENCHMARK_OWN_ARROW_CACHE_REMOVED_BYTES ' + str(removed_cache_bytes), flush=True)

disk = shutil.disk_usage(root)
print("BENCHMARK_DISK " + json.dumps({"free_bytes": disk.free, "total_bytes": disk.total}))

for directory in [root, Path("/home/jovyan/lora-1b-benchmark-data-20261014"), Path("/home/jovyan/.cache/huggingface")]:
    result = subprocess.run(["du", "-sk", str(directory)], capture_output=True, text=True, timeout=30)
    print("BENCHMARK_STORAGE " + json.dumps({"path": str(directory), "du": result.stdout.strip(), "returncode": result.returncode}))
