"""Read-only collection of queue state, partial logs and complete results."""

import base64
import gzip
import hashlib
import json
import os
from pathlib import Path

root = Path(os.environ["SPEED_ROOT"])
records = []
for path in sorted(root.glob("*/result.json")):
    records.append(json.loads(path.read_text()))
queues = [json.loads(p.read_text()) for p in sorted(root.glob("queue-*.json"))]
cleanup = [json.loads(p.read_text()) for p in sorted(root.glob("cleanup-*.json"))]
capacity_queues = [json.loads(p.read_text()) for p in sorted((root / "capacity-probe").glob("queue-*.json"))]
model_staging = [json.loads(p.read_text()) for p in sorted(root.glob("model-stage-*.json"))]
capacity_staging = [json.loads(p.read_text()) for p in sorted((root / "capacity-probe").glob("model-stage-*.json"))]
tails = {}
failure_tails = {}
for record in records:
    if record.get("status") == "failed":
        name = f"{record['method']}_r{record['rank']}_b{record['batch']}"
        path = root / name / "train.log"
        if path.exists():
            with path.open("rb") as stream:
                stream.seek(0, 2)
                stream.seek(max(0, stream.tell() - 12000))
                failure_tails[name] = stream.read().decode(errors="replace")
worker_disk = []
for path in sorted(root.glob("bootstrap-disk-*.jsonl")):
    with path.open("rb") as stream:
        stream.seek(0, 2)
        stream.seek(max(0, stream.tell() - 32000))
        lines = stream.read().splitlines()
    for line in reversed(lines):
        try:
            record = json.loads(line)
        except (ValueError, UnicodeDecodeError):
            continue
        worker_disk.append({"file": path.name, "record": record})
        break
for queue in queues:
    if queue.get("current"):
        path = root / queue["current"]["name"] / "train.log"
        if path.exists():
            tails[str(queue["shard"])] = path.read_text()[-5000:]
storage = []
for path in ["/", "/tmp", "/home/jovyan", "/home/jovyan/shares/SR006.nfs3"]:
    stat = os.statvfs(path)
    storage.append({"path": path, "available_bytes": stat.f_bavail * stat.f_frsize})
payload = json.dumps({"results": records, "queues": queues, "tails": tails,
                      "failure_tails": failure_tails, "worker_disk": worker_disk,
                      "cleanup": cleanup, "model_staging": model_staging,
                      "capacity_queues": capacity_queues, "capacity_staging": capacity_staging,
                      "storage": storage,
                      "storage_stop": (root / "STOP_STORAGE.json").exists()}).encode()
encoded = base64.b64encode(gzip.compress(payload, mtime=0)).decode()
chunks = [encoded[i:i+10000] for i in range(0, len(encoded), 10000)]
for i, value in enumerate(chunks):
    print("SPEED_CHUNK " + json.dumps({"index": i, "total": len(chunks),
                                        "sha256": hashlib.sha256(payload).hexdigest(), "data": value}), flush=True)
