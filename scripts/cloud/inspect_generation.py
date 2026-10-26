"""Emit persistent campaign logs and metrics without loading model weights."""

import json
import os
from pathlib import Path
import time


root = Path(os.environ.get("GENERATION_ROOT", "/home/jovyan/shares/SR006.nfs3/dimativator/lora-generation-1b-20261026"))
for path in ["/", "/tmp", "/home/jovyan", "/home/jovyan/shares/SR006.nfs3"]:
    stat = os.statvfs(path)
    print("GEN_STORAGE " + json.dumps({"timestamp": time.time(), "path": path,
                                      "available_bytes": stat.f_bavail * stat.f_frsize}), flush=True)
if (root / "STOP_STORAGE.json").exists():
    print("GEN_STORAGE_STOP " + (root / "STOP_STORAGE.json").read_text(), flush=True)
for run in sorted(root.glob("*_seed42")):
    for name in ["run.json", "config.json", "data-manifest.json", "completed.json"]:
        path = run / name
        if path.is_file():
            print("GEN_FILE " + json.dumps({"run": run.name, "file": name, "data": json.loads(path.read_text())}), flush=True)
    history = run / "history.jsonl"
    if history.exists():
        for line in history.read_text().splitlines():
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            print("GEN_HISTORY " + json.dumps({"run": run.name, "data": row}), flush=True)
    for name in ["disk.jsonl", "disk-trainer.jsonl"]:
        path = run / name
        if path.exists():
            lines = path.read_text().splitlines()
            for line in lines[-2:]:
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                print("GEN_DISK_LAST " + json.dumps({"run": run.name, "file": name, "data": row}), flush=True)
    log = run / "train.log"
    if log.exists():
        print("GEN_LOG_TAIL " + json.dumps({"run": run.name, "tail": log.read_text().splitlines()[-35:]}), flush=True)
    checkpoints = [str(p) for p in run.rglob("*") if p.suffix in {".pt", ".pth", ".safetensors", ".bin"}]
    print("GEN_CHECKPOINTS " + json.dumps({"run": run.name, "paths": checkpoints}), flush=True)
