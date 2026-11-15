"""Read-only exact-child guard inspection with miniature JSONL fixtures."""

from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts/cloud"))
import cloud_repair_guard_inspect as guard


NOW = datetime(2026, 11, 15, 15, tzinfo=timezone.utc)


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


@pytest.fixture
def recorded(tmp_path, monkeypatch):
    state = tmp_path / "nfs3/state"
    monkeypatch.setattr(guard, "STATE", state)
    monkeypatch.setattr(guard, "HERE", tmp_path / "code/scripts/cloud")
    monkeypatch.setattr(guard, "PYTHON", tmp_path / "retained-venv/bin/python")
    paths = [tmp_path / name for name in ("root", "tmp", "home", "assets", "nfs3/state", "nfs3/mlsub-logs")]
    monkeypatch.setattr(guard, "affected_paths", lambda: paths)
    task = guard.TASKS[0]
    spool = state / "pending-wandb-sync" / guard.task_key(task)
    spool.mkdir(parents=True)
    command = [str(guard.PYTHON), "-u", "/tmp/frozen/src/main.py", "--seed", str(task["seed"])]
    status = {"task": task, "worker": 0, "phase": "training", "child_pid": 1234,
              "command": command, "recovery_logs": str(spool)}
    owner = {"pid": 1234, "command": command, "started_at": (NOW - timedelta(hours=1)).isoformat()}
    telemetry = {"kind": "telemetry", "pid": 1234, "timestamp": (NOW - timedelta(seconds=10)).isoformat(),
                 "filesystems": [{"timestamp": (NOW - timedelta(seconds=12)).isoformat(),
                     "requested_path": str(path), "path": str(path), "available_bytes": 9_000_000_000,
                     "device": 44, "status": "ok"} for path in paths]}
    write(spool / "owned-process.json", owner)
    (spool / "disk-guard.jsonl").write_text(json.dumps(telemetry) + "\n")
    return status, spool, owner, telemetry


def replace_guard(recorded, rows, partial=b""):
    path = recorded[1] / "disk-guard.jsonl"
    path.write_bytes(b"".join((json.dumps(row) + "\n").encode() for row in rows) + partial)


def test_exact_main_child_and_all_six_filesystems_are_verified_read_only(recorded):
    status, spool, _, _ = recorded
    before = {path.name: path.read_bytes() for path in spool.iterdir()}
    result = guard.inspect_guard(status, NOW.isoformat())
    assert result["status"] == "ok" and result["ok"] is True
    assert result["owner_pid"] == 1234 and result["owner_command_verified"] is True
    assert result["guard_age_seconds"] == 10
    assert len(result["telemetry"]["filesystems"]) == 6
    assert {path.name: path.read_bytes() for path in spool.iterdir()} == before


def test_incomplete_last_jsonl_line_is_ignored_without_mutating_it(recorded):
    status, spool, _, telemetry = recorded
    partial = b'{"kind":"telemetry","pid":1234,"timestamp":"unfinished'
    replace_guard(recorded, [telemetry], partial)
    before = (spool / "disk-guard.jsonl").read_bytes()
    assert guard.inspect_guard(status, NOW)["status"] == "ok"
    assert (spool / "disk-guard.jsonl").read_bytes() == before


def test_large_tail_discards_only_first_cropped_record(recorded):
    status, spool, _, telemetry = recorded
    path = spool / "disk-guard.jsonl"
    path.write_bytes(b"x" * (guard.TAIL_BYTES * 2) + b"\n" + (json.dumps(telemetry) + "\n").encode())
    assert guard.inspect_guard(status, NOW)["status"] == "ok"


def test_wrong_owned_pid_is_a_transition_not_protection_proof(recorded):
    status, spool, owner, _ = recorded
    write(spool / "owned-process.json", {**owner, "pid": 5678})
    result = guard.inspect_guard(status, NOW)
    assert result["status"] == "transition" and result["ok"] is False
    assert result["reason"] == "owner_pid_differs_from_active_status"


def test_matching_pid_with_wrong_command_fails_closed(recorded):
    status, spool, owner, _ = recorded
    write(spool / "owned-process.json", {**owner, "command": ["another", "experiment"]})
    result = guard.inspect_guard(status, NOW)
    assert result["status"] == "unavailable" and result["reason"] == "owned_child_command_mismatch"
    assert "another" not in json.dumps(result)


@pytest.mark.parametrize("seconds", [90, 200, -10])
def test_guard_timestamp_must_be_fresh_relative_to_snapshot(recorded, seconds):
    status, _, _, telemetry = recorded
    telemetry["timestamp"] = (NOW - timedelta(seconds=seconds)).isoformat()
    replace_guard(recorded, [telemetry])
    assert guard.inspect_guard(status, NOW)["status"] == "stale"


def test_fresh_guard_with_stale_filesystem_telemetry_is_stale(recorded):
    status, _, _, telemetry = recorded
    telemetry["filesystems"][0]["timestamp"] = (NOW - timedelta(seconds=91)).isoformat()
    replace_guard(recorded, [telemetry])
    assert guard.inspect_guard(status, NOW)["reason"] == "guard_filesystem_telemetry_not_fresh"


def test_completed_record_is_inactive_without_reading_historical_guard_files(recorded, monkeypatch):
    status = {**recorded[0], "phase": "complete", "recovery_logs": "/unavailable/historical/path"}
    monkeypatch.setattr(guard, "bounded_bytes", lambda *args, **kwargs: pytest.fail("Completed worker read guard files"))
    result = guard.inspect_guard(status, NOW)
    assert result["status"] == "inactive" and result["ok"] is False


@pytest.mark.parametrize("phase", ["claimed", "syncing"])
def test_natural_handoffs_are_transitions(recorded, phase):
    assert guard.inspect_guard({**recorded[0], "phase": phase}, NOW)["status"] == "transition"


def test_profile_owner_matches_exact_worker_task_and_microbatch_arguments(recorded):
    status, spool, _, telemetry = recorded
    status.update(phase="profiling", profiling_microbatch=8)
    directory = spool / "cloud0-lora_rite-micro8"
    command = [str(guard.PYTHON), str(guard.HERE / "cloud_repair_entry.py"), "profile", "--worker-id", "0",
               "--task", str(directory / "task.json"), "--microbatch", "8"]
    write(directory / "owned-process.json", {"pid": 1234, "command": command})
    (directory / "disk-guard.jsonl").write_text(json.dumps(telemetry) + "\n")
    assert guard.inspect_guard(status, NOW)["status"] == "ok"
    command[-1] = "4"
    write(directory / "owned-process.json", {"pid": 1234, "command": command})
    assert guard.inspect_guard(status, NOW)["reason"] == "owned_child_command_mismatch"


@pytest.mark.parametrize("target", ["owned-process.json", "disk-guard.jsonl", "directory"])
def test_guard_links_are_rejected_without_reading_foreign_contents(recorded, tmp_path, target):
    status, spool, _, _ = recorded
    foreign = tmp_path / "foreign"
    foreign.write_text("private foreign content")
    if target == "directory":
        relocated = spool.with_name("real-evidence")
        spool.rename(relocated)
        spool.symlink_to(relocated, target_is_directory=True)
    else:
        path = spool / target
        path.unlink()
        path.symlink_to(foreign)
    result = guard.inspect_guard(status, NOW)
    assert result["status"] == "unavailable" and foreign.read_text() == "private foreign content"
    assert "private foreign content" not in json.dumps(result)


def test_foreign_task_or_recovery_namespace_is_rejected(recorded):
    status = recorded[0]
    assert guard.inspect_guard({**status, "task": guard.TASKS[1]}, NOW)["reason"] == "guard_task_identity_mismatch"
    assert guard.inspect_guard({**status, "recovery_logs": str(recorded[1] / "..")}, NOW)["reason"] == "active_recovery_log_namespace_mismatch"


@pytest.mark.parametrize("bytes_left", [0, 4_999_999_999, 5_000_000_000])
def test_reserve_violation_is_reported_without_deletion_or_stop(recorded, bytes_left):
    status, spool, _, telemetry = recorded
    telemetry["filesystems"][0]["available_bytes"] = bytes_left
    replace_guard(recorded, [telemetry])
    before = (spool / "disk-guard.jsonl").read_bytes()
    result = guard.inspect_guard(status, NOW)
    assert result["status"] == "unavailable" and result["reason"] == "disk_reserve_violation"
    assert result["low_space_paths"] == [telemetry["filesystems"][0]["requested_path"]]
    assert (spool / "disk-guard.jsonl").read_bytes() == before


def test_missing_duplicate_or_noninteger_filesystem_values_fail_closed(recorded):
    status, _, _, telemetry = recorded
    telemetry["filesystems"][-1] = telemetry["filesystems"][0]
    replace_guard(recorded, [telemetry])
    assert guard.inspect_guard(status, NOW)["reason"] == "guard_filesystem_coverage_mismatch"
    telemetry["filesystems"][0]["available_bytes"] = float("nan")
    telemetry["filesystems"][-1] = {**telemetry["filesystems"][0], "requested_path": str(guard.affected_paths()[-1])}
    replace_guard(recorded, [telemetry])
    assert guard.inspect_guard(status, NOW)["reason"] == "invalid_exact_guard_filesystem_bytes"


def test_incomplete_first_telemetry_and_child_change_are_transitions(recorded):
    status, _, _, telemetry = recorded
    replace_guard(recorded, [], b'{"kind":"telemetry"')
    assert guard.inspect_guard(status, NOW)["status"] == "transition"
    replace_guard(recorded, [telemetry, {**telemetry, "pid": 5678}])
    assert guard.inspect_guard(status, NOW)["reason"] == "guard_child_changed_after_matching_telemetry"


@pytest.mark.parametrize("kind", ["stop", "unknown_event"])
def test_later_same_child_nontelemetry_event_never_blesses_old_success(recorded, kind):
    status, _, _, telemetry = recorded
    replace_guard(recorded, [telemetry, {"kind": kind, "pid": 1234, "timestamp": NOW.isoformat()}])
    result = guard.inspect_guard(status, NOW)
    assert result["status"] == "unavailable" and result["ok"] is False
    assert result["reason"] == "later_nontelemetry_guard_event"


def test_malformed_complete_line_is_unavailable_but_never_echoed(recorded):
    status, spool, _, _ = recorded
    (spool / "disk-guard.jsonl").write_text("arbitrary private bytes\n")
    result = guard.inspect_guard(status, NOW)
    assert result["status"] == "unavailable" and "arbitrary private bytes" not in json.dumps(result)


def test_only_two_diagnostic_files_are_read_and_unknown_payload_fields_are_not_exported(recorded, monkeypatch):
    status, _, _, telemetry = recorded
    telemetry["never_export"] = "arbitrary content"
    telemetry["filesystems"][0]["never_export"] = "another arbitrary field"
    replace_guard(recorded, [telemetry])
    paths = []
    original = guard.bounded_bytes
    def observed(path, *args, **kwargs):
        paths.append(path.name)
        return original(path, *args, **kwargs)
    monkeypatch.setattr(guard, "bounded_bytes", observed)
    result = guard.inspect_guard(status, NOW)
    assert result["status"] == "ok" and paths == ["owned-process.json", "disk-guard.jsonl"]
    assert "never_export" not in json.dumps(result)
