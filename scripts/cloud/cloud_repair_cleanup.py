"""Attested diagnostic reconciliation and deletion of this campaign's assets only.

Independent W&B verification is performed by the local monitor. This CPU entry
checks its exact manifest attestations against the retained Cloud source bytes.
Reconciliation preserves those bytes; only a separate, fully gated cleanup can
remove model/data assets. Run long hashing operations under the owned CPU guard.
"""

import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import stat
import uuid

from cloud_repair_assets import manifests
from cloud_repair_common import (ASSETS, BUNDLE_SHA, CAMPAIGN, HOME, REVISION,
                                 STATE, TASKS, affected_paths, locked, now,
                                 require_space, run_name, task_key)


NAMESPACE = "dimativator-paper8b-pro-rite-repair-20261115"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def canonical(path: Path) -> None:
    require(path.is_absolute() and path.resolve() == path and not path.is_symlink(),
            "Cleanup namespace contains a link or noncanonical path")


def namespace() -> None:
    require(ASSETS == HOME / "assets" and HOME.name == NAMESPACE
            and STATE.name == NAMESPACE and STATE != HOME,
            "Refusing foreign asset/state namespace")
    for path in (HOME, ASSETS, STATE, STATE / "cleanup.lock"):
        canonical(path)


def read(path: Path) -> dict:
    canonical(path)
    require(path.is_file(), "Missing required evidence: " + str(path))
    return json.loads(path.read_text())


def durable_save(path: Path, value: dict) -> None:
    canonical(path)
    canonical(path.parent)
    temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    data = (json.dumps(value, indent=2, allow_nan=False) + "\n").encode()
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        parent = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        try:
            os.fsync(parent)
        finally:
            os.close(parent)
    finally:
        if temporary.exists():
            temporary.unlink()


def sha(value: dict) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def proof(confirmation: dict) -> str:
    require(confirmation.get("campaign") == CAMPAIGN
            and type(confirmation.get("verified_runs")) is int
            and confirmation["verified_runs"] == len(TASKS), "Missing eight verified results")
    for name in ("all_diagnostics_independently_verified", "exports_verified"):
        require(confirmation.get(name) is True, "Missing final attestation: " + name)
    if confirmation.get("asset_isolation_only") is True:
        from cloud_repair_closure import reviewed_proof
        closure = reviewed_proof(confirmation.get("asset_isolation_request"))
        require(confirmation.get("all_owned_jobs_terminal") is False
                and confirmation.get("all_known_jobs_terminal") is True,
                "Unknown allocation status must remain explicit")
        require(confirmation.get("runtime_bundle_sha256") == closure["runtime_bundle_sha256"]
                and confirmation.get("export_manifest_sha256") == closure["exports"]["manifest_sha256"]
                and confirmation.get("owned_terminal_jobs") == closure["allocation_proof"]["known_allocation_ids"],
                "Asset isolation export/runtime/allocation binding differs")
        expected = {key: row["diagnostic_manifest_sha256"]
                    for key, row in closure["scientific_verification"].items()}
        require(confirmation.get("diagnostic_manifests") == expected
                and all(row["independent_verification_sha256"] == confirmation.get("independent_verification_sha256")
                        for row in closure["scientific_verification"].values()),
                "Asset isolation independent diagnostic binding differs")
    else:
        require(confirmation.get("all_owned_jobs_terminal") is True,
                "Missing final attestation: all_owned_jobs_terminal")
    records = confirmation.get("diagnostic_manifests", {})
    require(set(records) == {task_key(task) for task in TASKS}, "Incomplete diagnostic attestations")
    require(all(isinstance(value, str) and len(value) == 64
                and all(char in "0123456789abcdef" for char in value) for value in records.values()),
            "Invalid diagnostic SHA attestation")
    return sha(confirmation)


def inventory(root: Path) -> tuple[dict, set]:
    """Reject links, special files and nested devices before reading any payload."""
    canonical(root)
    require(root.is_dir(), "Missing evidence directory: " + str(root))
    device = root.stat().st_dev
    files, directories = {}, set()
    for directory, children, names in os.walk(root, followlinks=False):
        for name in children + names:
            path = Path(directory) / name
            record = path.lstat()
            require(not stat.S_ISLNK(record.st_mode) and record.st_dev == device,
                    "Link or nested device in protected namespace")
            relative = str(path.relative_to(root))
            if stat.S_ISDIR(record.st_mode):
                directories.add(relative)
            else:
                require(stat.S_ISREG(record.st_mode), "Special file in protected namespace")
                files[relative] = record
    return files, directories


def verified_file(path: Path, expected: dict) -> str:
    canonical(path)
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    try:
        before = os.fstat(descriptor)
        require(stat.S_ISREG(before.st_mode) and before.st_size == expected["bytes"],
                "Evidence file size mismatch: " + str(path))
        result = hashlib.sha256()
        with os.fdopen(descriptor, "rb", closefd=False) as stream:
            for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                result.update(chunk)
        after = os.fstat(descriptor)
        require((before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns, before.st_ctime_ns)
                == (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns),
                "Evidence changed during hashing")
        require(result.hexdigest() == expected["sha256"], "Evidence SHA mismatch: " + str(path))
        require(path.stat().st_ino == before.st_ino and path.stat().st_dev == before.st_dev,
                "Evidence path changed during hashing")
        return result.hexdigest()
    finally:
        os.close(descriptor)


def remove_tree(parent: int, name: str) -> None:
    """Descriptor-relative removal also works with the retained Python3.10 env."""
    before = os.stat(name, dir_fd=parent, follow_symlinks=False)
    require(stat.S_ISDIR(before.st_mode), "Deletion target is no longer a directory")
    descriptor = os.open(name, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=parent)
    try:
        opened = os.fstat(descriptor)
        require((opened.st_dev, opened.st_ino) == (before.st_dev, before.st_ino),
                "Deletion directory changed while opening")
        with os.scandir(descriptor) as entries:
            for entry in entries:
                record = os.stat(entry.name, dir_fd=descriptor, follow_symlinks=False)
                require(record.st_dev == opened.st_dev, "Nested device during asset deletion")
                if stat.S_ISDIR(record.st_mode):
                    remove_tree(descriptor, entry.name)
                else:
                    require(stat.S_ISREG(record.st_mode), "Link or special file during asset deletion")
                    os.unlink(entry.name, dir_fd=descriptor)
        current = os.stat(name, dir_fd=parent, follow_symlinks=False)
        require((current.st_dev, current.st_ino) == (opened.st_dev, opened.st_ino),
                "Deletion directory changed before removal")
        os.rmdir(name, dir_fd=parent)
    finally:
        os.close(descriptor)


def completed_records(confirmation: dict) -> list[tuple[dict, dict]]:
    task_files, _ = inventory(STATE / "tasks")
    worker_files, _ = inventory(STATE / "workers")
    require(set(task_files) == {task_key(task) + ".json" for task in TASKS},
            "Unexpected or missing task records")
    require({name for name in worker_files if name.endswith(".json")}
            == {str(index) + ".json" for index in range(len(TASKS))},
            "Unexpected or missing worker records")
    require(all(name.endswith(".json") or name in {str(index) + ".lock"
                for index in range(len(TASKS))} for name in worker_files), "Unknown worker evidence")
    records = []
    for index, task in enumerate(TASKS):
        key = task_key(task)
        pair = (read(STATE / "tasks" / (key + ".json")), read(STATE / "workers" / f"{index}.json"))
        for record in pair:
            require(record.get("task") == task and record.get("worker") == index
                    and record.get("phase") == "complete" and record.get("returncode") == 0
                    and record.get("child_pid") is None and record.get("wandb_sync_complete") is True,
                    "Worker/task has not completed and synchronized")
            require(record.get("verification", {}).get("train_records") == 3000
                    and record.get("verification", {}).get("validation_points") == 12,
                    "Missing complete training/final-validation evidence")
            require(record.get("recovery_manifest_sha256") == confirmation["diagnostic_manifests"][key],
                    "Worker/task manifest differs from independent attestation")
        require(pair[0].get("wandb_id") == pair[1].get("wandb_id"), "Worker/task W&B identity differs")
        records.append(pair)
    return records


def verify_spool(directory: Path, task: dict, status: dict, attested: str) -> None:
    files, _ = inventory(directory)
    manifest_path = directory / "diagnostic-manifest.json"
    require("diagnostic-manifest.json" in files, "Missing diagnostic manifest")
    verified_file(manifest_path, {"bytes": files["diagnostic-manifest.json"].st_size, "sha256": attested})
    manifest = read(manifest_path)
    require(manifest.get("campaign") == CAMPAIGN and manifest.get("model_revision") == REVISION
            and manifest.get("runtime_bundle_sha256") == BUNDLE_SHA.read_text().strip()
            and manifest.get("run_name") == run_name(task)
            and manifest.get("wandb_id") == status.get("wandb_id")
            and bool(manifest.get("wandb_id")) and manifest.get("command") == status.get("command"),
            "Diagnostic manifest identity differs")
    listed = {}
    names = set()
    for row in manifest.get("files", []):
        path = PurePosixPath(row["path"])
        require(not path.is_absolute() and all(part not in (".", "..") for part in path.parts)
                and path.parts and path.parts[0] != "upload"
                and str(path) != "diagnostic-manifest.json" and str(path) not in listed,
                "Invalid or duplicate diagnostic source path")
        require(Path(row["wandb_name"]).name == row["wandb_name"]
                and row["wandb_name"] not in ("", ".", "..", "diagnostic-manifest.json")
                and row["wandb_name"] not in names, "Invalid or duplicate W&B diagnostic name")
        listed[str(path)] = row
        names.add(row["wandb_name"])
    actual = {name for name in files if name != "diagnostic-manifest.json"
              and PurePosixPath(name).parts[0] != "upload"}
    require(actual == set(listed) and bool(actual), "Diagnostic source coverage differs from attestation")
    for name, row in listed.items():
        verified_file(directory / name, row)
    # Upload hardlinks are redundant copies, but unknown entries or links still fail closed.
    expected_upload = {"upload/" + name for name in names | {"diagnostic-manifest.json"}}
    actual_upload = {name for name in files if name.startswith("upload/")}
    require(actual_upload <= expected_upload,
            "Unknown upload evidence")
    by_upload_name = {row["wandb_name"]: name for name, row in listed.items()}
    by_upload_name["diagnostic-manifest.json"] = "diagnostic-manifest.json"
    for name in actual_upload:
        source = files[by_upload_name[PurePosixPath(name).name]]
        require((files[name].st_dev, files[name].st_ino) == (source.st_dev, source.st_ino),
                "Upload copy is not a redundant source hardlink; preserve unique evidence")


def reconcile(confirmation: dict) -> dict:
    namespace()
    confirmation_sha = proof(confirmation)
    with locked(STATE / "cleanup.lock"):
        namespace()
        if confirmation.get("asset_isolation_only") is True:
            from cloud_repair_closure import close_campaign
            close_campaign(confirmation["asset_isolation_request"])
        records = completed_records(confirmation)
        before = require_space(affected_paths(), {STATE: 1_000_000})
        pending, verified = STATE / "pending-wandb-sync", STATE / "verified-raw-logs"
        for directory in (pending, verified):
            canonical(directory)
            directory.mkdir(exist_ok=True)
            inventory(directory)
            require({path.name for path in directory.iterdir()} <= {task_key(task) for task in TASKS},
                    "Unrecognized recovery evidence must be preserved")
        reconciled = []
        for index, task in enumerate(TASKS):
            key = task_key(task)
            source, destination = pending / key, verified / key
            require(source.exists() != destination.exists(), "Missing or duplicate diagnostic archive")
            directory = source if source.exists() else destination
            verify_spool(directory, task, records[index][0], confirmation["diagnostic_manifests"][key])
            if source.exists():
                require(source.stat().st_dev == verified.stat().st_dev, "Archive rename must stay on one filesystem")
                source.rename(destination)
                for parent in (pending, verified):
                    descriptor = os.open(parent, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
                    try:
                        os.fsync(descriptor)
                    finally:
                        os.close(descriptor)
            for path, status in zip((STATE / "tasks" / (key + ".json"), STATE / "workers" / f"{index}.json"), records[index]):
                status.update(syncphase="independently_verified", recovery_logs=str(destination),
                              independently_verified_at=now())
                durable_save(path, status)
            reconciled.append(key)
        result = {"campaign": CAMPAIGN, "completed_at": now(), "confirmation_sha256": confirmation_sha,
                  "reconciled": reconciled, "before": before}
        durable_save(STATE / "reconciliation.json", result)
        return result


def cleanup(confirmation: dict) -> dict:
    namespace()
    confirmation_sha = proof(confirmation)
    with locked(STATE / "cleanup.lock"):
        namespace()
        if confirmation.get("asset_isolation_only") is True:
            from cloud_repair_closure import validate_closure
            validate_closure(confirmation["asset_isolation_request"])
        records = completed_records(confirmation)
        require(all(status.get("syncphase") == "independently_verified" for pair in records for status in pair),
                "Independent diagnostic reconciliation is incomplete")
        reconciliation = read(STATE / "reconciliation.json")
        require(reconciliation.get("confirmation_sha256") == confirmation_sha
                and set(reconciliation.get("reconciled", [])) == {task_key(task) for task in TASKS},
                "Final attestation differs from reconciliation")
        pending = STATE / "pending-wandb-sync"
        canonical(pending)
        require(pending.is_dir() and not list(pending.iterdir()), "Unresolved pending W&B evidence remains")
        for index, task in enumerate(TASKS):
            verify_spool(STATE / "verified-raw-logs" / task_key(task), task, records[index][0],
                         confirmation["diagnostic_manifests"][task_key(task)])
        before = require_space(affected_paths(), {STATE: 1_000_000})
        intent_path, result_path = STATE / "cleanup-intent.json", STATE / "cleanup.json"
        if not ASSETS.exists():
            intent = read(intent_path)
            require(intent.get("campaign") == CAMPAIGN and intent.get("root") == str(ASSETS)
                    and intent.get("confirmation_sha256") == confirmation_sha
                    and intent.get("assets_verified") is True, "Absent assets lack the matching durable deletion intent")
            result = {**intent, "phase": "deleted", "completed_at": now(), "reconciled_absence": True}
        else:
            files, directories = inventory(ASSETS)
            expected = manifests()
            ready = read(ASSETS / "ready.json")
            require(ready.get("campaign") == CAMPAIGN and ready.get("revision") == REVISION
                    and ready.get("files") == expected, "Asset manifest differs from pinned files")
            require(read(ASSETS / "ownership.json") == {"campaign": CAMPAIGN, "root": str(ASSETS)},
                    "Asset ownership differs")
            require(set(files) == set(expected) | {"ownership.json", "ready.json"},
                    "Unknown or missing assets refuse deletion")
            expected_directories = {str(parent) for name in expected for parent in Path(name).parents
                                    if str(parent) != "."}
            require(directories == expected_directories, "Unknown asset directories refuse deletion")
            for name, row in expected.items():
                verified_file(ASSETS / name, row)
            # Hashing the full model can be lengthy. Refresh the launch/write
            # reserve immediately before recording intent and removing assets.
            before = require_space(affected_paths(), {STATE: 1_000_000})
            root = ASSETS.stat()
            intent = {"campaign": CAMPAIGN, "root": str(ASSETS), "phase": "deleting", "started_at": now(),
                      "confirmation_sha256": confirmation_sha, "assets_verified": True,
                      "asset_manifest_sha256": sha(expected), "root_device": root.st_dev,
                      "root_inode": root.st_ino, "removed_bytes": sum(row.st_size for row in files.values()),
                      "before": before}
            if intent_path.exists():
                previous = read(intent_path)
                require(previous.get("confirmation_sha256") == confirmation_sha
                        and previous.get("root_device") == root.st_dev and previous.get("root_inode") == root.st_ino,
                        "Existing deletion intent belongs to a different asset tree")
            durable_save(intent_path, intent)
            parent = os.open(HOME, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
            try:
                current = os.stat("assets", dir_fd=parent, follow_symlinks=False)
                require(current.st_dev == root.st_dev and current.st_ino == root.st_ino
                        and stat.S_ISDIR(current.st_mode), "Asset tree changed before deletion")
                remove_tree(parent, "assets")
                os.fsync(parent)
            finally:
                os.close(parent)
            require(not ASSETS.exists(), "Asset deletion did not complete")
            result = {**intent, "phase": "deleted", "completed_at": now()}
        # Record causation before post-delete telemetry, which can fail or hang.
        durable_save(result_path, result)
        after = require_space(affected_paths())
        result.update(after=after, telemetry_verified_at=now())
        durable_save(result_path, result)
        return result
