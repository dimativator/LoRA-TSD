"""Delete only the two exact directories explicitly selected by the user."""

import hashlib
import json
import os
from pathlib import Path
import shutil
import stat
import time

from cloud14_core import STATE, decode_audit_environment, digest, locked, now, require_space, save, snapshot_lines


HOME_TARGET = Path("/home/jovyan/dimativator")
ARCHIVE = STATE / "reconciled-wandb-sync-be845f0bd3823bb9"
TARGETS = (HOME_TARGET, ARCHIVE)
RECEIPT = STATE / "user-selected-cleanup-20261115.json"
PATHS = [Path("/"), Path("/tmp"), Path("/home/jovyan"), STATE]
ACTIVE_GRID = Path("/home/jovyan/shares/SR006.nfs3/dimativator-polar-grid-124m-20261114-b16")


def verify_active_grid() -> list[dict]:
    def check_references(value: object) -> None:
        if isinstance(value, dict):
            for item in value.values():
                check_references(item)
        elif isinstance(value, list):
            for item in value:
                check_references(item)
        elif isinstance(value, str):
            for target in TARGETS:
                if value == str(target) or str(target) + "/" in value:
                    raise ValueError("Active experiment references a selected deletion target")

    if ACTIVE_GRID.resolve() != ACTIVE_GRID:
        raise ValueError("Unexpected active-grid path alias")
    records = []
    for lane in range(10):
        directory = ACTIVE_GRID / f"lane{lane}"
        plan = json.loads((directory / "plan.json").read_text())
        status = json.loads((directory / "status.json").read_text())
        check_references(plan)
        check_references(status)
        record = {"lane": lane, "state": status["state"], "source_revision": plan["revision"]}
        if status["state"] == "running":
            run = status["run"]["name"]
            if Path(run).name != run:
                raise ValueError("Unsafe active run name")
            output = ACTIVE_GRID / "runs" / run
            config = json.loads((output / "config.json").read_text())
            check_references(config)
            command = config["command"]
            if command[command.index("--results_base_folder") + 1] != str(ACTIVE_GRID / "runs"):
                raise ValueError("Active experiment output root differs from reviewed source")
            if command[command.index("--datasets_dir") + 1] != "/tmp/polar-grid-tokens":
                raise ValueError("Active experiment data root differs from reviewed source")
            age = time.time() - (output / "train.log").stat().st_mtime
            if age < -60 or age > 600:
                raise ValueError("Active experiment path evidence is stale")
            record.update(run=run, command=command, train_log_age_seconds=age)
        elif status["state"] not in {"completed", "completed_with_failures"}:
            raise ValueError("Unknown active-grid state")
        records.append(record)
    return records


def scan(root: Path, max_mtime_ns: int | None = None) -> dict:
    if root.is_symlink() or root.parent.resolve() != root.parent:
        raise ValueError("Selected root or an ancestor is a symlink")
    mountinfo = Path("/proc/self/mountinfo")
    if mountinfo.exists():
        for line in mountinfo.read_text().splitlines():
            mount = Path(line.split()[4])
            if mount == root or root in mount.parents:
                raise ValueError("Selected root contains a mountpoint")
    identity = root.lstat()
    if not stat.S_ISDIR(identity.st_mode):
        raise ValueError("Selected root is not a directory")
    rows = []
    last_space_check = time.monotonic()

    def failed(error: OSError) -> None:
        raise error

    for directory, directories, files in os.walk(root, followlinks=False, onerror=failed):
        for name in sorted(directories + files):
            path = Path(directory) / name
            info = path.lstat()
            if info.st_dev != identity.st_dev:
                raise ValueError("Nested filesystem within selected directory")
            if not (stat.S_ISREG(info.st_mode) or stat.S_ISDIR(info.st_mode)
                    or stat.S_ISLNK(info.st_mode)):
                raise ValueError("Special file within selected directory")
            if max_mtime_ns is not None and info.st_mtime_ns > max_mtime_ns:
                raise ValueError("Selected directory changed after the user-reviewed inventory")
            rows.append({"path": str(path.relative_to(root)), "mode": info.st_mode,
                         "inode": info.st_ino, "size": info.st_size,
                         "allocated": info.st_blocks * 512, "mtime_ns": info.st_mtime_ns})
            if len(rows) > 100000:
                raise ValueError("Selected directory unexpectedly large")
            if time.monotonic() - last_space_check > 30:
                require_space(PATHS)
                last_space_check = time.monotonic()
    unique_allocations = {row["inode"]: row["allocated"] for row in rows}
    return {"path": str(root), "device": identity.st_dev, "inode": identity.st_ino,
            "entries": len(rows), "allocated_bytes": identity.st_blocks * 512
            + sum(unique_allocations.values()),
            "metadata_sha256": hashlib.sha256(json.dumps(sorted(rows, key=lambda r: r["path"]),
                                                         sort_keys=True).encode()).hexdigest()}


def verify_archive(audit: dict) -> None:
    manifest_path = STATE / "recovery-manifest.json"
    if digest(manifest_path) != audit["manifest_sha256"]:
        raise ValueError("Recovery manifest differs from independently verified backup")
    manifest = json.loads(manifest_path.read_text())
    expected = {row["path"]: row for row in manifest["files"]}
    actual = set()
    last_check = time.monotonic()
    for directory, directories, files in os.walk(ARCHIVE, followlinks=False):
        if any((Path(directory) / name).is_symlink() for name in directories):
            raise ValueError("Unexpected archive link")
        for name in files:
            path = Path(directory) / name
            relative = str(path.relative_to(ARCHIVE))
            if path.is_symlink() or relative not in expected:
                raise ValueError("Unexpected archive file")
            row = expected[relative]
            if path.stat().st_size != row["bytes"] or digest(path) != row["sha256"]:
                raise ValueError("Archive file differs from independently verified backup")
            actual.add(relative)
            if time.monotonic() - last_check > 30:
                require_space(PATHS)
                last_check = time.monotonic()
    if actual != set(expected):
        raise ValueError("Archive inventory differs from verified manifest")


def remove_exact(root: Path, identity: dict) -> None:
    if root not in TARGETS or not shutil.rmtree.avoids_symlink_attacks:
        raise ValueError("Unsafe or unapproved deletion target")
    current = root.lstat()
    if stat.S_ISLNK(current.st_mode) or (current.st_dev, current.st_ino) != (
            identity["device"], identity["inode"]):
        raise ValueError("Selected root identity changed")
    parent = os.open(root.parent, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        shutil.rmtree(root.name, dir_fd=parent)
    finally:
        os.close(parent)
    if root.exists() or root.is_symlink():
        raise ValueError("Selected root remains after deletion")


def main() -> None:
    audit = decode_audit_environment("SELECTED_CLEANUP")
    if audit.get("authorization") != "user-remove-two-annotated-directories-20261115":
        raise ValueError("Explicit selected-root authorization required")
    if audit.get("targets") != [str(path) for path in TARGETS]:
        raise ValueError("Authorization targets differ from fixed deletion allowlist")
    if audit.get("all_unique_files_downloaded_sha256_verified") is not True:
        raise ValueError("Independent recovery-backup verification missing")
    if audit.get("active_jobs_checked") is not True:
        raise ValueError("Fresh active-job audit missing")
    audit_sha = hashlib.sha256(json.dumps(audit, sort_keys=True).encode()).hexdigest()
    require_space(PATHS, {STATE: 1_000_000})
    active_grid = verify_active_grid()
    with locked(STATE / "user-selected-cleanup-20261115.lock"):
        if RECEIPT.is_symlink() or STATE.resolve() != STATE:
            raise ValueError("Cleanup state path is a symlink")
        receipt = json.loads(RECEIPT.read_text()) if RECEIPT.exists() else {
            "authorization_sha256": audit_sha, "started_at": now(),
            "before": require_space(PATHS), "active_grid": active_grid, "targets": []}
        if receipt["authorization_sha256"] != audit_sha:
            raise ValueError("Existing cleanup belongs to a different authorization")
        for row in receipt["targets"]:
            root = Path(row["path"])
            if root not in TARGETS:
                raise ValueError("Unapproved root in cleanup journal")
            if row.get("phase") == "deleting" and not root.exists() and not root.is_symlink():
                row.update(deleted=True, phase="deleted", reconciled_at=now())
                save(RECEIPT, receipt)
        completed = {row["path"] for row in receipt["targets"] if row.get("deleted")}
        inspections = {}
        for root in TARGETS:
            if str(root) in completed:
                if root.exists() or root.is_symlink():
                    raise ValueError("Previously deleted target reappeared")
                continue
            identity = scan(root, audit["inventory_cutoff_ns"] if root == HOME_TARGET else None)
            if root == HOME_TARGET:
                directories = sorted(path.name for path in root.iterdir()
                                     if stat.S_ISDIR(path.lstat().st_mode))
                if directories != sorted(audit["home_children"]):
                    raise ValueError("Personal-root directories differ from reviewed inventory")
                if identity["allocated_bytes"] > 2_100_000_000:
                    raise ValueError("Personal-root size grew beyond reviewed scope")
            else:
                verify_archive(audit)
            inspections[root] = identity
        # Validate both targets before starting any deletion.
        save(RECEIPT, receipt)
        for root, identity in inspections.items():
            require_space(PATHS)
            if scan(root)["metadata_sha256"] != identity["metadata_sha256"]:
                raise ValueError("Selected directory changed during verification")
            row = next((item for item in receipt["targets"] if item["path"] == str(root)), None)
            if row is None:
                row = {**identity, "phase": "deleting", "deletion_started_at": now(), "deleted": False}
                receipt["targets"].append(row)
            elif (row["device"], row["inode"]) != (identity["device"], identity["inode"]):
                raise ValueError("Unfinished cleanup target identity changed")
            save(RECEIPT, receipt)
            remove_exact(root, identity)
            row.update(deleted=True, phase="deleted", deleted_at=now())
            save(RECEIPT, receipt)
            receipt["after"] = require_space(PATHS)
            save(RECEIPT, receipt)
        receipt.update(completed_at=now(), phase="complete",
                       after=require_space(PATHS))
        save(RECEIPT, receipt)
        for line in snapshot_lines(receipt):
            print(line, flush=True)


if __name__ == "__main__":
    main()
