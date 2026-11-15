"""Read-only disk-guard evidence for one exact repair child, without GPU access.

This validates recorded protection, not process liveness. It never signals a
process, changes state, or reads model/data/configuration credentials.
"""

from datetime import datetime
import json
import math
import os
from pathlib import Path
import stat

from cloud_repair_common import HERE, PYTHON, STATE, TASKS, affected_paths, task_key


TAIL_BYTES = 64 * 1024
OWNER_BYTES = 64 * 1024
FRESH_SECONDS = 90
FUTURE_TOLERANCE_SECONDS = 5
INACTIVE = {"complete", "reserved", "queued", "failed", "waiting", "resource_waiting"}
TRANSITION = {"claimed", "syncing"}
ACTIVE = {"training", "validation", "profiling", "initializing"}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def canonical(path: Path) -> None:
    require(path.is_absolute() and not path.is_symlink() and path.resolve() == path,
            "noncanonical_or_linked_guard_path")


def seconds(value) -> float:
    if isinstance(value, datetime):
        moment = value
    else:
        require(isinstance(value, str), "invalid_guard_timestamp")
        moment = datetime.fromisoformat(value.replace("Z", "+00:00"))
    require(moment.tzinfo is not None, "timezone_missing_from_guard_timestamp")
    result = moment.timestamp()
    require(math.isfinite(result), "nonfinite_guard_timestamp")
    return result


def bounded_bytes(path: Path, limit: int, tail: bool = False) -> tuple[bytes, bool]:
    canonical(path)
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    try:
        record = os.fstat(descriptor)
        require(stat.S_ISREG(record.st_mode), "guard_evidence_is_not_a_regular_file")
        require(tail or record.st_size <= limit, "owned_process_record_is_oversized")
        start = max(0, record.st_size - limit) if tail else 0
        os.lseek(descriptor, start, os.SEEK_SET)
        data = os.read(descriptor, limit + (0 if tail else 1))
        require(len(data) <= limit, "owned_process_record_is_oversized")
        current = path.lstat()
        require((current.st_dev, current.st_ino) == (record.st_dev, record.st_ino)
                and not stat.S_ISLNK(current.st_mode), "guard_evidence_path_changed")
        return data, start > 0
    finally:
        os.close(descriptor)


def telemetry_rows(path: Path) -> list[dict]:
    data, cropped = bounded_bytes(path, TAIL_BYTES, tail=True)
    if cropped:
        # The first bytes can be the suffix of a record outside this byte bound.
        boundary = data.find(b"\n")
        data = data[boundary + 1:] if boundary >= 0 else b""
    rows = []
    for line in data.splitlines(keepends=True):
        if not line.endswith(b"\n"):
            break  # A writer may currently be appending its final JSONL record.
        if not line.strip():
            continue
        row = json.loads(line)
        require(isinstance(row, dict), "invalid_complete_guard_record")
        rows.append(row)
    return rows


def inspect_guard(status: dict, snapshot_time) -> dict:
    result = {"status": "unavailable", "ok": False, "reason": "invalid_task_status"}
    try:
        require(isinstance(status, dict), "invalid_task_status")
        worker = status.get("worker")
        require(type(worker) is int and worker in range(len(TASKS)), "invalid_guard_worker")
        task = TASKS[worker]
        require(status.get("task") == task, "guard_task_identity_mismatch")
        phase = status.get("phase")
        key = task_key(task)
        result.update(worker=worker, task_key=key,
                      phase=phase if isinstance(phase, str) else "unknown",
                      child_pid=status.get("child_pid") if type(status.get("child_pid")) is int else None)
        if phase in INACTIVE:
            return {**result, "status": "inactive", "reason": "no_active_child_expected_in_recorded_phase"}
        if phase in TRANSITION:
            return {**result, "status": "transition", "reason": "worker_child_handoff_in_progress"}
        require(phase in ACTIVE, "unknown_guard_phase")
        snapshot = seconds(snapshot_time)
        spool = STATE / "pending-wandb-sync" / key
        canonical(STATE)
        canonical(spool)
        require(status.get("recovery_logs") == str(spool), "active_recovery_log_namespace_mismatch")
        child = status.get("child_pid")
        if child is None:
            return {**result, "status": "transition", "reason": "active_child_pid_not_yet_recorded"}
        require(type(child) is int and child > 0, "invalid_active_child_pid")
        if phase == "profiling":
            batch = status.get("profiling_microbatch")
            require(type(batch) is int and batch in (1, 2, 4, 8, 16, 32), "invalid_profile_microbatch")
            directory = spool / f"cloud{worker}-{task['method']}-micro{batch}"
            expected = [str(PYTHON), str(HERE / "cloud_repair_entry.py"), "profile",
                        "--worker-id", str(worker), "--task", str(directory / "task.json"),
                        "--microbatch", str(batch)]
        else:
            directory = spool
            expected = status.get("command")
            require(isinstance(expected, list) and bool(expected)
                    and all(isinstance(value, str) for value in expected), "main_command_not_yet_recorded")
        canonical(directory)
        result["guard_directory"] = str(directory)
        owner_bytes, _ = bounded_bytes(directory / "owned-process.json", OWNER_BYTES)
        owner = json.loads(owner_bytes)
        require(isinstance(owner, dict), "invalid_owned_process_record")
        owner_pid = owner.get("pid")
        result["owner_pid"] = owner_pid if type(owner_pid) is int else None
        if type(owner_pid) is not int or owner_pid != child:
            return {**result, "status": "transition", "reason": "owner_pid_differs_from_active_status"}
        require(owner.get("command") == expected, "owned_child_command_mismatch")
        result["owner_command_verified"] = True
        rows = telemetry_rows(directory / "disk-guard.jsonl")
        matching = [row for row in rows if row.get("kind") == "telemetry"
                    and type(row.get("pid")) is int and row["pid"] == child]
        if not matching:
            return {**result, "status": "transition", "reason": "matching_child_telemetry_not_yet_complete"}
        latest = matching[-1]
        # A later record for another child indicates a concurrent handoff, not
        # evidence that the old child's last successful guard is still active.
        index = max(index for index, row in enumerate(rows) if row is latest)
        if any(row.get("pid") != child for row in rows[index + 1:]):
            return {**result, "status": "transition", "reason": "guard_child_changed_after_matching_telemetry"}
        if any(row.get("kind") != "telemetry" for row in rows[index + 1:]):
            return {**result, "status": "unavailable", "reason": "later_nontelemetry_guard_event"}
        result.update(telemetry={key: latest[key] for key in ("kind", "timestamp", "pid")},
                      guard_age_seconds=snapshot - seconds(latest.get("timestamp")))
        if not -FUTURE_TOLERANCE_SECONDS <= result["guard_age_seconds"] < FRESH_SECONDS:
            return {**result, "status": "stale", "reason": "guard_telemetry_not_fresh_relative_to_snapshot"}
        filesystems = latest.get("filesystems")
        require(isinstance(filesystems, list) and len(filesystems) == len(affected_paths())
                and all(isinstance(row, dict) for row in filesystems),
                "guard_filesystem_coverage_mismatch")
        paths = [row.get("requested_path") for row in filesystems]
        require(len(set(paths)) == len(paths) and set(paths) == {str(path) for path in affected_paths()},
                "guard_filesystem_coverage_mismatch")
        for row in filesystems:
            require(isinstance(row.get("path"), str) and Path(row["path"]).is_absolute()
                    and type(row.get("available_bytes")) is int
                    and type(row.get("device")) is int and row["device"] >= 0,
                    "invalid_exact_guard_filesystem_bytes")
            require(row.get("status") in ("ok", "stop"), "invalid_guard_filesystem_status")
            age = snapshot - seconds(row.get("timestamp"))
            if not -FUTURE_TOLERANCE_SECONDS <= age < FRESH_SECONDS:
                return {**result, "status": "stale", "reason": "guard_filesystem_telemetry_not_fresh"}
        result["telemetry"]["filesystems"] = [{key: row.get(key) for key in (
            "timestamp", "requested_path", "path", "available_bytes", "device", "status")}
            for row in filesystems]
        low = [row["requested_path"] for row in filesystems if row["available_bytes"] <= 5_000_000_000]
        if low:
            return {**result, "status": "unavailable", "reason": "disk_reserve_violation", "low_space_paths": low}
        require(all(row.get("status") == "ok" for row in filesystems), "guard_filesystem_status_not_ok")
        return {**result, "status": "ok", "ok": True, "reason": "matching_child_guard_and_all_filesystems_verified"}
    except (OSError, ValueError, TypeError, KeyError, OverflowError) as error:
        # Never serialize commands, arbitrary file content or OS exception text.
        reason = str(error) if isinstance(error, ValueError) and str(error).replace("_", "").isalnum() else "guard_evidence_unavailable"
        return {**result, "status": "unavailable", "reason": reason, "error_type": type(error).__name__}
