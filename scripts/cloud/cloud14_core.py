"""CPU-only campaign bookkeeping shared by the Cloud launcher and its tests."""

from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys

CAMPAIGN = "paper8b-three-seeds-20261113"
RESERVE = 5_000_000_000
ASSETS = Path("/home/jovyan/dimativator-paper8b-three-seeds-20261113/assets")
STATE = Path("/home/jovyan/shares/SR006.nfs3/dimativator-paper8b-three-seeds-20261113")
REVISION = "d04e592bb4f6aa9cfee91e2e20afa771667e1d4b"
TELEMETRY_PYTHON = os.readlink("/proc/self/exe") if Path("/proc/self/exe").exists() else sys.executable


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def save(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def digest(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def _raw_disk_records(paths: list[Path]) -> list[dict]:
    result = []
    for requested in paths:
        path = requested
        while not path.exists():
            if path == path.parent:
                raise OSError(f"Cannot resolve filesystem for {requested}")
            path = path.parent
        stat = os.statvfs(path)
        available = stat.f_bavail * stat.f_frsize
        result.append({"timestamp": now(), "requested_path": str(requested),
                       "path": str(path), "available_bytes": available,
                       "device": path.stat().st_dev,
                       "status": "ok" if available > RESERVE else "stop"})
    return result


def disk_records(paths: list[Path]) -> list[dict]:
    """A hung network filesystem must never hang the process guard itself."""
    process = subprocess.Popen([TELEMETRY_PYTHON, str(Path(__file__).resolve()), "disk"],
                               stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                               text=True)
    try:
        output, error = process.communicate(json.dumps([str(path) for path in paths]), timeout=15)
    except subprocess.TimeoutExpired as error:
        process.kill()
        # Do not wait again: a kernel-blocked NFS stat may not reap until the mount recovers.
        raise OSError("Disk telemetry timed out after15seconds") from error
    if process.returncode:
        raise OSError("Disk telemetry failed: " + error[-1500:])
    return json.loads(output)


def require_space(paths: list[Path], planned: dict[Path, int] | None = None) -> list[dict]:
    records = disk_records(paths)
    additional = {}
    for path, size in (planned or {}).items():
        record = disk_records([path])[0]
        device = record["device"]
        additional[device] = additional.get(device, 0) + size
    for record in records:
        if record["available_bytes"] <= RESERVE + additional.get(record["device"], 0):
            raise OSError(f"Disk reserve or planned-write headroom unavailable: {record}")
    return records


@contextmanager
def locked(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX)
        yield


def task_key(task: dict) -> str:
    return f"{task['method']}_{task['dataset']}_seed{task['seed']}"


def validate_plan(plan: dict) -> None:
    queue = plan["queue"]
    keys = [task_key(task) for task in queue]
    reused = [task_key(task) for method in plan["methods"] for task in method["existing"]]
    assert len(keys) == len(set(keys)) == 138
    assert len(reused) == len(set(reused)) == 42
    assert not set(keys) & set(reused)
    assert len(set(keys + reused)) == 180
    assert set(plan["seeds"]) == {0, 1, 42}


def claim(plan: dict, worker: int, state: Path = STATE) -> dict | None:
    """Claim once under the shared NFS lock; failures require an explicit audited retry."""
    with locked(state / "queue.lock"):
        for task in plan["queue"]:
            path = state / "tasks" / (task_key(task) + ".json")
            if path.exists():
                continue
            save(path, {"task": task, "worker": worker, "phase": "claimed",
                        "claimed_at": now(), "pid": os.getpid(),
                        "hostname": os.uname().nodename})
            return task
    return None


def wandb_row(row: dict) -> dict:
    """Use a custom x axis: train and validation can share an optimizer step."""
    return {"optimizer_step": row["step"],
            **{key: value for key, value in row.items() if key != "step"}}


def verify_rows(rows: list[dict], result: dict, task: dict) -> dict:
    train = [row for row in rows if "train/loss" in row]
    validation = [row for row in rows if "validation_score" in row]
    assert [row["step"] for row in train] == list(range(1, 3001))
    assert [row["step"] for row in validation] == list(range(250, 3001, 250))
    assert all(math.isfinite(row["train/loss"]) for row in train)
    assert all(math.isfinite(row["validation_score"]) for row in validation)
    assert result["optimizer"] == task["optimizer"]
    assert result["dataset"] == task["dataset"] and result["seed"] == task["seed"]
    assert result["model"] == "meta-llama/Meta-Llama-3.1-8B" and result["steps"] == 3000
    best = max(validation, key=lambda row: row["validation_score"])
    assert math.isclose(result["reported_accuracy"], best["validation_score"])
    assert result["reported_accuracy_step"] == best["step"]
    return {"train_records": len(train), "validation_points": len(validation),
            "best_accuracy": best["validation_score"], "best_step": best["step"]}


def cleanup_owned_assets(root: Path, state: Path, confirmation: dict) -> int:
    """Refuse cleanup until a final audit and all owners report synchronized completion."""
    assert confirmation["verified_runs"] == 180
    assert confirmation["wandb_sync_verified"] is True
    assert confirmation["results_export_verified"] is True
    assert confirmation["all_owned_jobs_terminal"] is True
    assert not list((state / "pending-wandb-sync").glob("*")), "Unverified recovery logs remain"
    assert not root.is_symlink()
    marker = json.loads((root / "ownership.json").read_text())
    assert marker["campaign"] == CAMPAIGN and marker["root"] == str(root.resolve())
    assert root.resolve().parent.name == "dimativator-paper8b-three-seeds-20261113"
    workers = list((state / "workers").glob("*.json"))
    assert len(workers) == 14
    assert all(json.loads(path.read_text())["phase"] == "complete" for path in workers)
    assert len(list((state / "tasks").glob("*.json"))) == 138
    assert all(json.loads(path.read_text())["phase"] == "complete"
               for path in (state / "tasks").glob("*.json"))
    files = [path for path in root.rglob("*") if path.is_file() and not path.is_symlink()]
    size = sum(path.stat().st_size for path in files)
    shutil.rmtree(root)
    save(state / "cleanup.json", {"completed_at": now(), "root": str(root),
                                  "removed_bytes": size, "confirmation": confirmation})
    return size


if __name__ == "__main__":
    assert sys.argv[1:] == ["disk"]
    print(json.dumps(_raw_disk_records([Path(path) for path in json.loads(sys.stdin.read())])))
