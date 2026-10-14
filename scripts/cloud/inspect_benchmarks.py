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
    metrics["verification"] = {
        "returncode": state.get("returncode"),
        "command": state["command"],
        "train_record_count": len(losses),
        "last_train_step": max((r["step"] for r in losses), default=None),
        "nonfinite_train_steps": [r["step"] for r in losses if not math.isfinite(r["train/loss"])],
        "validation": [r for r in rows if "validation_score" in r],
    }
    print("BENCHMARK_METRICS " + json.dumps(metrics), flush=True)
print("BENCHMARK_CHECKPOINT_COUNT " + str(len(list(root.rglob("*.pt")))))

disk = shutil.disk_usage(root)
print("BENCHMARK_DISK " + json.dumps({"free_bytes": disk.free, "total_bytes": disk.total}))

for directory in [root, Path("/home/jovyan/lora-1b-benchmark-data-20261014"), Path("/home/jovyan/.cache/huggingface")]:
    result = subprocess.run(["du", "-sk", str(directory)], capture_output=True, text=True, timeout=30)
    print("BENCHMARK_STORAGE " + json.dumps({"path": str(directory), "du": result.stdout.strip(), "returncode": result.returncode}))
