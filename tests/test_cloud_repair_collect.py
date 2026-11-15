"""CPU control collection preserves intact evidence without duplicate submissions."""

from datetime import datetime, timedelta, timezone
import fcntl
import json
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts/cloud"))
import cloud_repair_collect as collect
from cloud14_core import snapshot_lines

JOB = "lm-mpi-job-12345678-1234-1234-1234-123456789abc"
SECOND = "lm-mpi-job-abcdef12-1234-1234-1234-123456789abc"


@pytest.fixture
def ledger(tmp_path, monkeypatch):
    monkeypatch.setattr(collect, "ARTIFACTS", tmp_path)
    value = {"campaign": collect.CAMPAIGN, "jobs": [], "auxiliary_jobs": [
        {"job_id": JOB, "mode": "inspect", "created_at": collect.now()}],
        "ambiguous_submissions": [], "assets_ready_verified": False}
    collect.save(tmp_path / "campaign.json", value)
    return tmp_path, value


def snapshot(mode="inspect"):
    records = [{"requested_path": str(path), "available_bytes": 1_000_000_000_000,
                "status": "ok"} for path in collect.affected_paths()]
    result = {"campaign": collect.CAMPAIGN, "timestamp": collect.now(), "filesystems": records,
              "assets_ready": True, "preparation": {"phase": "ready"}, "tasks": [], "workers": []}
    if mode == "preflight":
        from cloud_repair_assets import manifests
        result.update(venv_import_returncode=0, asset_bytes=sum(r["bytes"] for r in manifests().values()),
                      space_for_fresh_assets=True, nfs2_read_only=[{"available_bytes": 0, "status": "stop"}])
    return result


def transport(value, returncode=0, timed_out=False):
    return {"stdout": "\n".join(snapshot_lines(value)), "stderr": "", "returncode": returncode,
            "ssh_returncode": 0, "timed_out": timed_out, "ssh_timed_out": False}


@pytest.mark.parametrize("returncode,timed_out", [(0, False), (1, False), (None, True)])
def test_intact_snapshot_credited_even_nonzero_cli_or_timeout(ledger, monkeypatch, returncode, timed_out):
    root, _ = ledger
    value = snapshot()
    calls = []
    def remote(command, deadline):
        calls.append(command)
        return transport(value, returncode, timed_out)
    monkeypatch.setattr(collect, "remote_cli", remote)
    result = collect.collect({"job_id": JOB, "mode": "inspect"}, "inspect", time.monotonic() + 100)
    assert calls == [["mlsub", "logs", JOB, "--tail", "80"]]
    assert result["collector_cli_returncode"] == returncode
    assert result["collector_timed_out"] is timed_out
    assert result["collector_source_job_id"] == JOB and result["collector_source_mode"] == "inspect"
    assert json.loads((root / "campaign.json").read_text())["assets_ready_verified"] is True


def test_partial_gzip_fails_closed(ledger, monkeypatch):
    root, _ = ledger
    value = transport(snapshot())
    value["stdout"] = value["stdout"][:-20]
    monkeypatch.setattr(collect, "remote_cli", lambda *a: value)
    with pytest.raises((AssertionError, ValueError, EOFError)):
        collect.collect({"job_id": JOB, "mode": "inspect"}, "inspect", time.monotonic() + 100)
    assert not (root / "inspection-latest.json").exists()
    assert not json.loads((root / "campaign.json").read_text())["assets_ready_verified"]


@pytest.mark.parametrize("damage", ["campaign", "stale", "future", "coverage"])
def test_wrong_or_stale_snapshot_cannot_enable_launch(ledger, monkeypatch, damage):
    value = snapshot()
    if damage == "campaign": value["campaign"] = "old180"
    elif damage == "stale": value["timestamp"] = (datetime.now(timezone.utc) - timedelta(hours=1)).isoformat()
    elif damage == "future": value["timestamp"] = (datetime.now(timezone.utc) + timedelta(hours=1)).isoformat()
    else: value["filesystems"].pop()
    monkeypatch.setattr(collect, "remote_cli", lambda *a: transport(value))
    with pytest.raises(ValueError):
        collect.collect({"job_id": JOB, "mode": "inspect"}, "inspect", time.monotonic() + 100)


@pytest.mark.parametrize("damage", [None, "venv", "reserve", "headroom", "missing_assets", "not_ready"])
def test_preflight_and_inspect_flags_require_actual_proofs(ledger, damage):
    value = snapshot("preflight")
    if damage == "venv": value["venv_import_returncode"] = 1
    elif damage == "reserve": value["filesystems"][0]["available_bytes"] = 5_000_000_000
    elif damage == "headroom":
        next(r for r in value["filesystems"] if r["requested_path"] == str(collect.ASSETS))["available_bytes"] = 6_000_000_000
    assert collect.validate_snapshot(value, "preflight") is (damage not in {"venv", "reserve", "headroom"})
    if damage == "missing_assets": value["assets_ready"] = False
    elif damage == "not_ready": value["preparation"] = {"phase": "downloading"}
    expected = damage not in {"reserve", "missing_assets", "not_ready"}
    assert collect.validate_snapshot(value, "inspect") is expected


def test_pending_latest_inspection_is_reused_without_cpu_flood(ledger, monkeypatch):
    _, value = ledger
    monkeypatch.setattr(collect.subprocess, "Popen", lambda *a, **k: pytest.fail("Duplicate CPU submission"))
    assert collect.select_job("inspect", None, True, time.monotonic() + 100)["job_id"] == JOB
    value["auxiliary_jobs"][0]["created_at"] = (datetime.now(timezone.utc) - timedelta(hours=1)).isoformat()
    collect.save(collect.ARTIFACTS / "campaign.json", value)
    monkeypatch.setattr(collect, "scheduler", lambda *a: {"jobs": [{"job_id": JOB, "status": "Running"}]})
    assert collect.select_job("inspect", None, True, time.monotonic() + 100)["job_id"] == JOB


def test_ambiguous_submission_blocks_new_cpu_but_explicit_logs_are_readable(ledger):
    root, value = ledger
    value["ambiguous_submissions"] = [{"note": "unresolved"}]
    collect.save(root / "campaign.json", value)
    with pytest.raises(RuntimeError, match="pending submission"):
        collect.select_job("inspect", None, True, time.monotonic() + 100)
    assert collect.select_job("inspect", JOB, True, time.monotonic() + 100)["job_id"] == JOB


def test_explicit_foreign_job_rejected(ledger):
    with pytest.raises(ValueError, match="owned auxiliary"):
        collect.select_job("inspect", SECOND, False, time.monotonic() + 100)


def test_new_cpu_snapshot_only_after_stale_terminal_previous(ledger, monkeypatch):
    root, value = ledger
    value["auxiliary_jobs"][0].update(scheduler_status="Completed", created_at=(
        datetime.now(timezone.utc) - timedelta(hours=1)).isoformat())
    collect.save(root / "campaign.json", value)
    class Submission:
        returncode = 0
        def __init__(self, command, **kwargs):
            assert command[-1] == "inspect" and "worker" not in command
            latest = json.loads((root / "campaign.json").read_text())
            latest["auxiliary_jobs"].append({"job_id": SECOND, "mode": "inspect", "created_at": collect.now()})
            collect.save(root / "campaign.json", latest)
        def communicate(self, timeout): return "recorded", ""
    monkeypatch.setattr(collect.subprocess, "Popen", Submission)
    assert collect.select_job("inspect", None, True, time.monotonic() + 100)["job_id"] == SECOND
    monkeypatch.setattr(collect.subprocess, "Popen", lambda *a, **k: pytest.fail("Duplicate CPU submission"))
    assert collect.select_job("inspect", None, True, time.monotonic() + 100)["job_id"] == SECOND


def test_scheduler_parses_plaintext_exact_ids_and_retains_history(ledger, monkeypatch):
    root, value = ledger
    value["jobs"] = [{"job_id": SECOND, "scheduler_status": "Completed"}]
    collect.save(root / "campaign.json", value)
    line = f"{JOB} Running repo:dimativator/LoRA-TSD|entry:scripts/cloud/cloud_repair_entry.py|note:owned"
    monkeypatch.setattr(collect, "remote_cli", lambda *a: {
        "stdout": line + "\n" + f"foreign Running note:{SECOND}", "returncode": 0,
        "ssh_returncode": 0, "timed_out": False, "ssh_timed_out": False})
    result = collect.scheduler(time.monotonic() + 100)
    records = {r["job_id"]: r for r in result["jobs"]}
    assert records[JOB]["status"] == "Running"
    assert records[SECOND]["status"] is None and records[SECOND]["historical_status"] == "Completed"
    assert collect.digest(Path(result["raw_proof"])) == result["raw_proof_sha256"]
    campaign = json.loads((root / "campaign.json").read_text())
    assert campaign["jobs"][0]["scheduler_status"] == "Completed"


def test_remote_timeout_wrapper_preserves_complete_partial_output(monkeypatch):
    value = snapshot()
    raw = "\n".join(snapshot_lines(value))
    def run(command, **kwargs):
        script = kwargs["input"]
        assert "start_new_session=True" in script and "os.killpg(p.pid,signal.SIGKILL)" in script
        assert "convert(e.output)" in script and "p.wait(" not in script
        return SimpleNamespace(returncode=0, stderr="", stdout=json.dumps({
            "stdout": raw, "stderr": "", "returncode": None, "timed_out": True}))
    monkeypatch.setattr(collect.subprocess, "run", run)
    result = collect.remote_cli(["mlsub", "logs", JOB, "--tail", "80"], time.monotonic() + 100)
    assert decode(result["stdout"])["campaign"] == collect.CAMPAIGN
    assert result["timed_out"] and result["returncode"] is None


def test_ssh_timeout_preserves_complete_payload_for_crc_verification(monkeypatch):
    raw = "\n".join(snapshot_lines(snapshot()))
    def timeout(*args, **kwargs):
        raise subprocess.TimeoutExpired("ssh", 1, output=raw.encode(), stderr=b"transport timeout")
    monkeypatch.setattr(collect.subprocess, "run", timeout)
    result = collect.remote_cli(["mlsub", "logs", JOB, "--tail", "80"], time.monotonic() + 100)
    assert result["ssh_timed_out"] and result["ssh_returncode"] is None
    assert decode(result["stdout"])["campaign"] == collect.CAMPAIGN


def test_ledger_wait_honors_collection_deadline(ledger):
    root, _ = ledger
    with (root / "submission.lock").open("a") as held:
        fcntl.flock(held, fcntl.LOCK_EX | fcntl.LOCK_NB)
        started = time.monotonic()
        with pytest.raises(RuntimeError, match="deadline"):
            with collect.ledger_lock(started + .03):
                pytest.fail("Acquired another controller's ledger lock")
        assert time.monotonic() - started < 1


def decode(raw):
    from cloud14_core import decode_snapshot
    return decode_snapshot(raw)
