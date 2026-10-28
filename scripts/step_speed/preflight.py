"""Fresh exact-byte storage inventory, model-cache check and CPU smoke."""

import json
import os
from pathlib import Path
import subprocess
import sys
import time

from benchmark import smoke
from disk_guard import DiskGuard
from huggingface_hub import HfApi, get_hf_file_metadata, hf_hub_url
from model_storage import SPEC


def main() -> None:
    root = Path(os.environ["SPEED_ROOT"])
    guard = DiskGuard(root, "speed-preflight")
    disk = guard(pending_write_bytes=SPEC["download_bytes"] + 100_000_000,
                 reason="reserve_private_8b_download", write_path=Path("/tmp"))
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
    token = os.environ.get("HF_TOKEN")
    info = HfApi().model_info(SPEC["model"], revision=SPEC["revision"], files_metadata=True, token=token)
    inventory = {file.rfilename: file.size for file in info.siblings}
    if any(inventory.get(file["path"]) != file["bytes"] for file in SPEC["files"]):
        raise RuntimeError("Pinned model file inventory changed")
    get_hf_file_metadata(hf_hub_url(SPEC["model"], "config.json", revision=SPEC["revision"]), token=token)
    if SPEC["config"]["hidden_size"] != 4096 or SPEC["config"]["num_hidden_layers"] != 32:
        raise RuntimeError("Unexpected 8B model configuration")
    result = {"timestamp": time.time(), "disk": disk, "nfs2": nfs2,
              "model_revision": info.sha, "weight_bytes": SPEC["weight_bytes"],
              "private_download_bytes": SPEC["download_bytes"], "model_weights_downloaded": False,
              "smoke": smoke(), "status": "passed", "checkpoints": False}
    root.mkdir(parents=True, exist_ok=True)
    guard(pending_write_bytes=1_000_000, reason="before_preflight_result")
    (root / "preflight.json").write_text(json.dumps(result, indent=2) + "\n")
    print("SPEED_PREFLIGHT " + json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
