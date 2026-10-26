"""Emit persistent campaign logs and metrics without loading model weights."""

import base64
import gzip
import hashlib
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
for path in sorted(root.glob("queue-*.json")):
    if path.is_file() and not path.is_symlink():
        try:
            record = json.loads(path.read_text())
        except json.JSONDecodeError as error:
            print("GEN_QUEUE " + json.dumps({"file": path.name, "error": str(error)}), flush=True)
        else:
            print("GEN_QUEUE " + json.dumps({"file": path.name, "data": record}), flush=True)
history_records = []
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
            history_records.append({"run": run.name, "data": row})
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
        lines = log.read_text().splitlines()
        for line in lines:
            if line.startswith("GEN_GPU_BEFORE_TRAIN "):
                print("GEN_GPU_INVENTORY " + json.dumps({"run": run.name, "data": json.loads(line.split(" ", 1)[1])}), flush=True)
        # Keep each emitted JSON record below the platform's stream chunk size.
        print("GEN_LOG_TAIL " + json.dumps({"run": run.name, "tail": [line[:220] for line in lines[-10:]]}), flush=True)
    checkpoints = [str(p) for p in run.rglob("*") if p.suffix in {".pt", ".pth", ".safetensors", ".bin"}]
    print("GEN_CHECKPOINTS " + json.dumps({"run": run.name, "paths": checkpoints}), flush=True)

# Transfer complete histories in small, checksummed records through Cloud logs.
payload = json.dumps(history_records, separators=(",", ":")).encode()
encoded = base64.b64encode(gzip.compress(payload, mtime=0)).decode()
digest = hashlib.sha256(payload).hexdigest()
chunks = [encoded[index:index + 1600] for index in range(0, len(encoded), 1600)]
for index, chunk in enumerate(chunks):
    print("GEN_HISTORY_CHUNK " + json.dumps({"index": index, "total": len(chunks),
                                             "sha256": digest, "records": len(history_records),
                                             "payload": chunk}), flush=True)
