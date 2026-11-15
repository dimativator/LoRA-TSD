"""Prepare one pinned asset copy without modifying shared environments or caches."""

import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time

from cloud_repair_common import (ASSETS, CAMPAIGN, HERE, PYTHON, REVISION, STATE,
                                 TASKS, affected_paths, digest, locked, now,
                                 require_space, save, task_key)


def manifests() -> dict:
    model = json.loads((HERE / "cloud14_model_files.json").read_text())
    data = json.loads((HERE / "cloud14_data_files.json").read_text())
    required = {task["dataset"] for task in TASKS}
    data = {name: row for name, row in data.items() if name.split("/", 1)[0] in required}
    return {**{"model/" + name: {**row, "url":
        f"https://huggingface.co/meta-llama/Meta-Llama-3.1-8B/resolve/{REVISION}/{name}"}
        for name, row in model.items()},
        **{"data/" + name: row for name, row in data.items()}}


def canonical_paths() -> None:
    for path in (ASSETS.parent, ASSETS, STATE, STATE / "prepare.lock"):
        if path.is_symlink() or path.resolve() != path:
            raise ValueError("Repair namespace contains a symlink")


def asset_path(name: str) -> Path:
    path = ASSETS / name
    if not path.is_relative_to(ASSETS) or path.is_symlink() or path.resolve() != path:
        raise ValueError("Pinned asset path escapes its owned namespace")
    return path


def verify_file(path: Path, record: dict) -> bool:
    if path.is_symlink() or path.resolve() != path:
        raise ValueError("Pinned asset is a symlink")
    return path.is_file() and path.stat().st_size == record["bytes"] and digest(path) == record["sha256"]


def prepare() -> None:
    canonical_paths()
    if Path(sys.prefix).resolve() != PYTHON.parent.parent.resolve():
        os.execv(str(PYTHON), [str(PYTHON), str(Path(__file__).with_name("cloud_repair_entry.py")),
                                "prepare-child"])
    import requests
    import wandb
    assert wandb.Api(timeout=45).viewer, "W&B authentication required before asset preparation"
    records = manifests()
    missing = [(name, record) for name, record in records.items()
               if not verify_file(asset_path(name), record)]
    remaining = sum(record["bytes"] for _, record in missing)
    require_space(affected_paths(), {ASSETS: remaining + 2_000_000, STATE: 10_000_000})
    with locked(STATE / "prepare.lock"):
        canonical_paths()
        ownership = {"campaign": CAMPAIGN, "root": str(ASSETS.resolve())}
        ASSETS.mkdir(parents=True, exist_ok=True)
        marker = ASSETS / "ownership.json"
        if marker.exists():
            assert not marker.is_symlink() and json.loads(marker.read_text()) == ownership
        else:
            assert not list(ASSETS.iterdir()), "Refusing to adopt an unowned directory"
            save(marker, ownership)
        missing = [(name, record) for name, record in records.items()
                   if not verify_file(asset_path(name), record)]
        remaining = sum(record["bytes"] for _, record in missing)
        require_space(affected_paths(), {ASSETS: remaining + 2_000_000, STATE: 10_000_000})
        save(STATE / "preparation.json", {"phase": "downloading", "started_at": now(),
                                         "remaining_bytes": remaining})
        for name, record in missing:
            path = asset_path(name)
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary = path.with_name(path.name + ".part")
            assert not temporary.is_symlink()
            # Only this campaign's own incomplete download is replaced.
            if temporary.exists():
                temporary.unlink()
            require_space(affected_paths(), {ASSETS: remaining})
            headers = {"Authorization": "Bearer " + os.environ["HF_TOKEN"]}
            last_check = time.monotonic()
            hashed = hashlib.sha256()
            with requests.get(record["url"], headers=headers, stream=True, timeout=(25, 180)) as response:
                if response.status_code != 200:
                    raise RuntimeError(f"Pinned asset request failed with HTTP{response.status_code}: {name}")
                with temporary.open("wb") as stream:
                    for chunk in response.iter_content(chunk_size=8 * 1024 * 1024):
                        if stream.tell() + len(chunk) > record["bytes"]:
                            raise ValueError("Asset download exceeds pinned size")
                        if time.monotonic() - last_check >= 30:
                            space = require_space(affected_paths(), {ASSETS: remaining})
                            save(STATE / "preparation.json", {"phase": "downloading", "timestamp": now(),
                                 "file": name, "remaining_bytes": remaining, "filesystems": space})
                            print("REPAIR_ASSET_PROGRESS=" + json.dumps({"file": name,
                                  "written_bytes": stream.tell(), "remaining_bytes": remaining}), flush=True)
                            last_check = time.monotonic()
                        stream.write(chunk)
                        hashed.update(chunk)
                        remaining -= len(chunk)
            assert temporary.stat().st_size == record["bytes"] and hashed.hexdigest() == record["sha256"]
            temporary.replace(path)
        assert all(verify_file(asset_path(name), record) for name, record in records.items())
        for worker, task in enumerate(TASKS):
            path = STATE / "tasks" / (task_key(task) + ".json")
            if not path.exists():
                save(path, {"task": task, "worker": worker, "phase": "reserved"})
        save(ASSETS / "ready.json", {"campaign": CAMPAIGN, "revision": REVISION,
                                    "files": records, "verified_at": now()})
        save(STATE / "preparation.json", {"phase": "ready", "timestamp": now(),
                                         "filesystems": require_space(affected_paths())})
        print("REPAIR_ASSETS_READY", flush=True)


def prepare_guarded() -> None:
    from cloud_repair_runtime import owned_process
    canonical_paths()
    required = sum(row["bytes"] for name, row in manifests().items()
                   if not asset_path(name).is_file()
                   or asset_path(name).stat().st_size != row["bytes"])
    require_space(affected_paths(), {ASSETS: required + 2_000_000, STATE: 10_000_000})
    with tempfile.TemporaryDirectory(prefix="repair-prepare-") as temporary:
        directory = Path(temporary)
        command = [str(PYTHON), str(Path(__file__).with_name("cloud_repair_entry.py")), "prepare-child"]
        with owned_process(command, directory, os.environ.copy()) as process:
            process.wait()
        print((directory / "process.log").read_text()[-12000:], flush=True)
        if process.returncode:
            raise RuntimeError("Pinned asset preparation failed; preserve partial assets and diagnose")


def cleanup(confirmation: dict) -> dict:
    canonical_paths()
    assert confirmation["campaign"] == CAMPAIGN and confirmation["verified_runs"] == len(TASKS)
    assert confirmation["all_owned_jobs_terminal"] is True
    assert confirmation["all_diagnostics_independently_verified"] is True
    assert confirmation["exports_verified"] is True
    assert not ASSETS.is_symlink() and ASSETS.parent.name == "dimativator-paper8b-pro-rite-repair-20261115"
    assert json.loads((ASSETS / "ownership.json").read_text()) == {
        "campaign": CAMPAIGN, "root": str(ASSETS.resolve())}
    for task in TASKS:
        assert json.loads((STATE / "tasks" / (task_key(task) + ".json")).read_text())["phase"] == "complete"
    workers = list((STATE / "workers").glob("*.json"))
    assert len(workers) == len(TASKS)
    assert all(json.loads(path.read_text())["phase"] == "complete" for path in workers)
    assert not list((STATE / "pending-wandb-sync").glob("*"))
    before = require_space(affected_paths())
    size = sum(path.stat().st_size for path in ASSETS.rglob("*")
               if path.is_file() and not path.is_symlink())
    shutil.rmtree(ASSETS)
    assert not ASSETS.exists()
    result = {"completed_at": now(), "root": str(ASSETS), "removed_bytes": size,
              "before": before, "after": require_space(affected_paths()), "confirmation": confirmation}
    save(STATE / "cleanup.json", result)
    return result
