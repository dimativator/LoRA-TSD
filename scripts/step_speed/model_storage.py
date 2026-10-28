"""Private model storage, guarded bootstrap and verified end-of-job cleanup."""

import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/generation"))
from disk_guard import DiskGuard

PREFIX = "lora-step-speed-8b-20261028-"
SPEC = json.loads(Path(__file__).with_name("model-spec.json").read_text())


def create_owned(campaign: Path, base: Path = Path("/tmp")) -> Path:
    owned = Path(tempfile.mkdtemp(prefix=PREFIX, dir=base))
    (owned / "owner.json").write_text(json.dumps({"campaign": str(campaign),
                                                "created_at": time.time(), "pid": os.getpid()}))
    return owned


def cleanup_owned(owned: Path, campaign: Path, base: Path = Path("/tmp")) -> dict:
    if owned.is_symlink() or owned.parent.resolve() != base.resolve() or not owned.name.startswith(PREFIX):
        raise ValueError("Refusing cleanup outside the owned temporary model directory")
    owner = json.loads((owned / "owner.json").read_text())
    if owner["campaign"] != str(campaign):
        raise ValueError("Refusing cleanup for another campaign")
    before = os.statvfs(base)
    size = sum(path.stat().st_size for directory, _, files in os.walk(owned, followlinks=False)
               for filename in files if not (path := Path(directory) / filename).is_symlink())
    shutil.rmtree(owned)
    after = os.statvfs(base)
    return {"path": str(owned), "removed": not owned.exists(), "removed_file_bytes": size,
            "available_bytes_before": before.f_bavail * before.f_frsize,
            "available_bytes_after": after.f_bavail * after.f_frsize,
            "timestamp": time.time()}


def stage_model(campaign: Path) -> Path:
    from huggingface_hub import hf_hub_download

    owned = Path(os.environ["SPEED_MODEL_OWNED_ROOT"])
    owner = json.loads((owned / "owner.json").read_text())
    if (owner["campaign"] != str(campaign) or owned.is_symlink()
            or owned.parent.resolve() != Path("/tmp").resolve() or not owned.name.startswith(PREFIX)):
        raise ValueError("Invalid private model directory")
    guard = DiskGuard(campaign, "before_model_download")
    guard(pending_write_bytes=SPEC["download_bytes"] + 100_000_000,
          reason="reserve_for_complete_8b_model", write_path=owned)
    snapshot = None
    completed_bytes = 0
    for file in SPEC["files"]:
        guard(pending_write_bytes=file["bytes"] + 10_000_000,
              reason="before_model_file_" + file["path"], write_path=owned)
        path = Path(hf_hub_download(SPEC["model"], file["path"], revision=SPEC["revision"],
                                   cache_dir=owned / "hf-home/hub", token=os.environ.get("HF_TOKEN")))
        if path.stat().st_size != file["bytes"]:
            raise RuntimeError("Model file size does not match pinned inventory: " + file["path"])
        snapshot = path.parent
        completed_bytes += file["bytes"]
        guard(pending_write_bytes=1_000_000, reason="before_download_progress", write_path=campaign)
        progress = {"timestamp": time.time(), "job": os.environ.get("HOSTNAME"),
                    "file": file["path"], "completed_bytes": completed_bytes,
                    "expected_bytes": SPEC["download_bytes"], "private_root": str(owned)}
        progress_path = campaign / ("model-stage-" + os.environ.get("HOSTNAME", str(os.getpid())) + ".json")
        temporary = progress_path.with_suffix(".tmp")
        temporary.write_text(json.dumps(progress))
        temporary.replace(progress_path)
    print("SPEED_MODEL_STAGED " + json.dumps({"snapshot": str(snapshot),
                                               "bytes": SPEC["download_bytes"],
                                               "revision": SPEC["revision"]}), flush=True)
    return snapshot


def memory_before_load() -> dict:
    import torch

    available = None
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemAvailable:"):
            available = int(line.split()[1]) * 1024
    for limit_path, used_path in [
        ("/sys/fs/cgroup/memory.max", "/sys/fs/cgroup/memory.current"),
        ("/sys/fs/cgroup/memory/memory.limit_in_bytes", "/sys/fs/cgroup/memory/memory.usage_in_bytes"),
    ]:
        if Path(limit_path).exists():
            value = Path(limit_path).read_text().strip()
            if value != "max":
                remaining = int(value) - int(Path(used_path).read_text())
                available = remaining if available is None else min(available, remaining)
    free, total = torch.cuda.mem_get_info()
    result = {"timestamp": time.time(), "weight_bytes": SPEC["weight_bytes"],
              "host_available_bytes": available, "gpu_free_bytes": free, "gpu_total_bytes": total,
              "host_required_before_load": SPEC["weight_bytes"] + 4_000_000_000,
              "gpu_required_for_weights_and_workspace": SPEC["weight_bytes"] + 2_000_000_000}
    print("SPEED_MEMORY_BEFORE_LOAD " + json.dumps(result), flush=True)
    if available is None or available < result["host_required_before_load"]:
        raise MemoryError("Insufficient host RAM to load the 8B model")
    if free < result["gpu_required_for_weights_and_workspace"]:
        raise MemoryError("Insufficient VRAM for the 8B weights and loading workspace")
    return result


def stop_group(child: subprocess.Popen) -> None:
    if child.poll() is not None:
        return
    os.killpg(child.pid, signal.SIGTERM)
    try:
        child.wait(timeout=15)
    except subprocess.TimeoutExpired:
        os.killpg(child.pid, signal.SIGKILL)
        child.wait(timeout=15)
    print("SPEED_BOOTSTRAP_STOPPED " + json.dumps({"pid": child.pid, "returncode": child.returncode}), flush=True)


def run(arguments: list[str]) -> None:
    campaign = Path(os.environ["SPEED_ROOT"])
    guard = DiskGuard(campaign, "bootstrap_watchdog")
    guard(pending_write_bytes=100_000_000, reason="before_private_runtime", write_path=Path("/tmp"))
    guard(pending_write_bytes=1_000_000, reason="before_log_directory", write_path=campaign)
    campaign.mkdir(parents=True, exist_ok=True)
    owned = create_owned(campaign)
    env = os.environ.copy()
    env.update(SPEED_MODEL_OWNED_ROOT=str(owned), HF_HOME=str(owned / "hf-home"),
               HF_HUB_CACHE=str(owned / "hf-home/hub"), HF_XET_CACHE=str(owned / "hf-home/xet"),
               HF_HUB_DISABLE_XET="1", HF_HUB_DISABLE_PROGRESS_BARS="1",
               HF_HUB_ENABLE_HF_TRANSFER="0", HF_HUB_OFFLINE="0", TRANSFORMERS_OFFLINE="0",
               HF_HUB_DOWNLOAD_TIMEOUT="60", HF_HUB_ETAG_TIMEOUT="30")
    guard.telemetry = campaign / ("bootstrap-disk-" + os.environ.get("HOSTNAME", str(os.getpid())) + ".jsonl")
    child = None
    exit_code = 1
    try:
        child = subprocess.Popen(["bash", str(ROOT / "scripts/cloud/run_step_speed_runtime.sh"), *arguments],
                                 env=env, start_new_session=True)

        def terminate(signum, frame):
            stop_group(child)
            raise SystemExit(128 + signum)

        signal.signal(signal.SIGTERM, terminate)
        signal.signal(signal.SIGINT, terminate)
        probe = {"active": False, "started": 0., "error": None}

        def check_storage():
            try:
                guard()
            except BaseException as error:
                probe["error"] = error
            finally:
                probe["active"] = False

        next_probe = 0.
        while child.poll() is None or probe["active"]:
            now = time.monotonic()
            error = probe["error"]
            if probe["active"] and now - probe["started"] > 15:
                error = RuntimeError("Storage telemetry unavailable for15seconds")
            if error is not None:
                stop_group(child)
                print("CRITICAL CLOUD.RU STORAGE: " + str(error), flush=True)
                raise RuntimeError("Storage guard stopped the exact owned process group") from error
            if not probe["active"] and now >= next_probe and child.poll() is None:
                probe.update(active=True, started=now)
                threading.Thread(target=check_storage, daemon=True).start()
                next_probe = now + 5
            time.sleep(.5)
        exit_code = child.returncode
    finally:
        if child is not None:
            stop_group(child)
        cleanup = cleanup_owned(owned, campaign)
        cleanup.update(job=os.environ.get("HOSTNAME"), child_returncode=exit_code)
        print("SPEED_MODEL_CLEANUP " + json.dumps(cleanup), flush=True)
        try:
            guard(pending_write_bytes=1_000_000, reason="before_cleanup_record", write_path=campaign)
            path = campaign / ("cleanup-" + os.environ.get("HOSTNAME", str(os.getpid())) + ".json")
            temporary = path.with_suffix(".tmp")
            temporary.write_text(json.dumps(cleanup))
            temporary.replace(path)
        except RuntimeError as error:
            print("CLEANUP_RECORD_NOT_PERSISTED " + str(error), flush=True)
        print("SPEED_ENTRY_EXIT=" + str(exit_code), flush=True)


if __name__ == "__main__":
    run(sys.argv[1:])
