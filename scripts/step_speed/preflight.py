"""Fresh exact-byte storage inventory, model-cache check and CPU smoke."""

import json
import os
from pathlib import Path
import subprocess
import sys
import time

from benchmark import smoke
from disk_guard import DiskGuard
from huggingface_hub import snapshot_download


def main() -> None:
    root = Path(os.environ["SPEED_ROOT"])
    guard = DiskGuard(root, "speed-preflight")
    disk = guard(pending_write_bytes=100_000_000, reason="preflight")
    # The gateway policy is independently checked before every submission.
    # NFS2 is inspected read-only, but is not used for bootstrap or any output.
    script = ("import os,json; p='/home/jovyan/shares/SR006.nfs2'; s=os.statvfs(p); "
              "print(json.dumps({'path':p,'available_bytes':s.f_bavail*s.f_frsize,'used_by_campaign':False}))")
    try:
        probe = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, timeout=10)
        nfs2 = json.loads(probe.stdout) if probe.returncode == 0 else {
            "path": "/home/jovyan/shares/SR006.nfs2", "status": "telemetry_unavailable",
            "error": probe.stderr[-1000:], "used_by_campaign": False}
    except subprocess.TimeoutExpired:
        nfs2 = {"path": "/home/jovyan/shares/SR006.nfs2", "status": "telemetry_unavailable",
                "error": "Read-only probe exceeded10seconds", "used_by_campaign": False}
    snapshot = Path(snapshot_download("meta-llama/Llama-3.2-1B-Instruct", local_files_only=True))
    weights = list(snapshot.glob("*.safetensors"))
    if not weights:
        raise RuntimeError("Cached model weights are missing")
    config = json.loads((snapshot / "config.json").read_text())
    if config["hidden_size"] != 2048 or config["num_hidden_layers"] != 16:
        raise RuntimeError("Unexpected cached model configuration")
    result = {"timestamp": time.time(), "disk": disk, "nfs2": nfs2,
              "model_snapshot": str(snapshot), "weight_bytes": sum(p.stat().st_size for p in weights),
              "smoke": smoke(), "status": "passed", "checkpoints": False}
    root.mkdir(parents=True, exist_ok=True)
    guard(pending_write_bytes=1_000_000, reason="before_preflight_result")
    (root / "preflight.json").write_text(json.dumps(result, indent=2) + "\n")
    print("SPEED_PREFLIGHT " + json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
