"""CPU-only read-only inventory required before the owned asset cleanup."""

import argparse
import json
import os
from pathlib import Path
import stat
from typing import Callable

from cloud14_assets import affected_paths
from cloud14_core import ASSETS, CAMPAIGN, STATE, now, require_space, snapshot_lines


def read_record(path: Path) -> dict | None:
    if path.is_symlink():
        raise ValueError("Audit record is a symlink")
    if not path.exists():
        return None
    if not stat.S_ISREG(path.lstat().st_mode):
        raise ValueError("Audit record is not a regular file")
    return json.loads(path.read_text())


def record_phases(directory: Path) -> dict:
    if directory.is_symlink() or not directory.is_dir():
        raise ValueError("Audit record directory is missing or a symlink")
    return {path.stem: read_record(path)["phase"] for path in directory.iterdir()
            if path.name.endswith(".json")}


def recovery_inventory(state: Path, check_space: Callable[[], object] | None = None) -> list[dict]:
    root = state / "pending-wandb-sync"
    if root.is_symlink():
        raise ValueError("Recovery root is a symlink")
    if not root.exists():
        return []
    if not root.is_dir():
        raise ValueError("Recovery root is not a directory")
    def fail_walk(error: OSError) -> None:
        raise error
    records = []
    for directory, directories, files in os.walk(root, followlinks=False, onerror=fail_walk):
        for name in sorted(directories + files):
            path = Path(directory) / name
            info = path.lstat()
            kind = "symlink" if stat.S_ISLNK(info.st_mode) else (
                "directory" if stat.S_ISDIR(info.st_mode) else "file")
            records.append({"path": str(path.relative_to(root)), "kind": kind,
                            "bytes": info.st_size if kind == "file" else 0})
            if len(records) > 20000:
                raise ValueError("Recovery inventory exceeds bounded size")
            if check_space is not None and len(records) % 100 == 0:
                check_space()
    return sorted(records, key=lambda row: row["path"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["inventory", "upload", "archive", "upload-child", "archive-child"],
                        nargs="?", default="inventory")
    parser.add_argument("--work-dir", type=Path)
    args = parser.parse_args()
    filesystems = require_space(affected_paths())
    if STATE.is_symlink() or ASSETS.is_symlink():
        raise ValueError("Campaign state or assets is a symlink")
    if args.mode != "inventory":
        from cloud14_recovery_sync import dispatch
        if not dispatch(args.mode, args.work_dir):
            return
    inventory = recovery_inventory(STATE, lambda: require_space(affected_paths()))
    snapshot = {"timestamp": now(), "campaign": CAMPAIGN, "audit_type": "final_read_only",
                "filesystems": filesystems, "pending_wandb_sync": inventory,
                "pending_root_exists": (STATE / "pending-wandb-sync").exists(),
                "assets_exist": ASSETS.exists(), "assets_symlink": ASSETS.is_symlink(),
                "ownership": read_record(ASSETS / "ownership.json"),
                "cleanup": read_record(STATE / "cleanup.json"),
                "recovery_upload": read_record(STATE / "recovery-upload.json"),
                "recovery_reconciled": read_record(STATE / "recovery-reconciled.json"),
                "worker_phases": record_phases(STATE / "workers"),
                "task_phases": record_phases(STATE / "tasks")}
    snapshot["filesystems_after"] = require_space(affected_paths())
    for line in snapshot_lines(snapshot):
        print(line, flush=True)


if __name__ == "__main__":
    main()
