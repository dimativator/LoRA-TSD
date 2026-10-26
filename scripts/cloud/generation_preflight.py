"""Read-only storage and dependency inventory for the generation pilot."""

import importlib.util
import json
import os
from pathlib import Path
import subprocess
import time


def inventory() -> dict:
    records = []
    for path in ["/", "/tmp", "/home/jovyan", "/home/jovyan/shares/SR006.nfs3"]:
        try:
            stat = os.statvfs(path)
            records.append({"path": path, "device": os.stat(path).st_dev,
                            "available_bytes": stat.f_bavail * stat.f_frsize})
        except OSError as error:
            records.append({"path": path, "error": str(error)})
    modules = ["torch", "transformers", "peft", "datasets", "pyarrow", "numpy",
               "sacrebleu", "nltk", "langdetect", "absl", "emoji", "immutabledict"]
    dependencies = {name: importlib.util.find_spec(name) is not None for name in modules}
    caches = []
    for root in [Path("/home/jovyan/.cache/huggingface"),
                 Path("/home/jovyan/shares/SR006.nfs3/dimativator")]:
        if root.is_dir():
            caches.append({"path": str(root), "entries": [p.name for p in root.iterdir()][:50]})
    result = {"timestamp": time.time(), "storage": records,
              "dependencies": dependencies, "cache_inventory": caches,
              "nfs2_accessed": False}
    command = ["nvidia-smi", "--query-gpu=index,name,memory.used,memory.free", "--format=csv,noheader"]
    try:
        proc = subprocess.run(command, capture_output=True, text=True, timeout=15)
        result["gpus"] = proc.stdout.strip()
    except (OSError, subprocess.TimeoutExpired):
        result["gpus"] = "not available in CPU preflight"
    return result


if __name__ == "__main__":
    print("GEN_PREFLIGHT " + json.dumps(inventory()), flush=True)
