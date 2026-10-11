"""Print compact summaries from the persistent SIQA log directory."""

import json
from pathlib import Path

root = Path("/home/jovyan/lora-siqa-retractions-20261011")
for path in sorted(root.glob("*/metrics.json")):
    metrics = json.loads(path.read_text())
    history_path = path.parent / "history.jsonl"
    rows = [json.loads(line) for line in history_path.read_text().splitlines()] if history_path.exists() else []
    metrics["cuda_peak_allocated_bytes"] = max((r.get("cuda_peak_allocated_bytes", 0) for r in rows), default=0)
    metrics["last_losses"] = [r["train/loss"] for r in rows if "train/loss" in r][-5:]
    print("SIQA_METRICS " + json.dumps(metrics), flush=True)
print("SIQA_CHECKPOINT_COUNT " + str(len(list(root.rglob("*.pt")))))
