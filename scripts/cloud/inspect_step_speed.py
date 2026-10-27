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
tails = {}
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
                      "storage": storage, "storage_stop": (root / "STOP_STORAGE.json").exists()}).encode()
encoded = base64.b64encode(gzip.compress(payload, mtime=0)).decode()
chunks = [encoded[i:i+10000] for i in range(0, len(encoded), 10000)]
for i, value in enumerate(chunks):
    print("SPEED_CHUNK " + json.dumps({"index": i, "total": len(chunks),
                                        "sha256": hashlib.sha256(payload).hexdigest(), "data": value}), flush=True)
