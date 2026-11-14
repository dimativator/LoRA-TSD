"""The final inventory never follows recovery links or mutates evidence."""

from pathlib import Path
import sys
import ast
import json
import subprocess

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts/cloud"))
from cloud14_final_audit import recovery_inventory
import cloud14_final_audit as audit


def test_recovery_inventory_is_complete_and_read_only(tmp_path):
    root = tmp_path / "pending-wandb-sync"
    directory = root / "attempt" / "wandb" / "nested"
    directory.mkdir(parents=True)
    file = directory / "process.log"
    file.write_bytes(b"unique evidence")
    result = recovery_inventory(tmp_path)
    row = next(row for row in result if row["path"] == "attempt/wandb/nested/process.log")
    assert row == {"path": "attempt/wandb/nested/process.log", "kind": "file", "bytes": 15}
    assert file.read_bytes() == b"unique evidence"
    assert recovery_inventory(tmp_path) == result


def test_recovery_inventory_does_not_follow_symlinks(tmp_path):
    foreign = tmp_path / "foreign"
    foreign.mkdir()
    (foreign / "must-not-read.log").write_text("foreign")
    root = tmp_path / "pending-wandb-sync"
    root.mkdir()
    (root / "link").symlink_to(foreign, target_is_directory=True)
    assert recovery_inventory(tmp_path) == [{"path": "link", "kind": "symlink", "bytes": 0}]
    (root / "link").unlink()
    root.rmdir()
    root.symlink_to(foreign, target_is_directory=True)
    with pytest.raises(ValueError, match="symlink"):
        recovery_inventory(tmp_path)


def test_missing_recovery_root_is_empty_without_creation(tmp_path):
    assert recovery_inventory(tmp_path) == []
    assert not list(tmp_path.iterdir())


def test_inventory_does_not_treat_unreadable_directory_as_empty(tmp_path, monkeypatch):
    (tmp_path / "pending-wandb-sync").mkdir()
    def denied(*args, **kwargs):
        kwargs["onerror"](PermissionError("unreadable"))
    monkeypatch.setattr(audit.os, "walk", denied)
    with pytest.raises(PermissionError, match="unreadable"):
        recovery_inventory(tmp_path)


def test_inventory_rejects_regular_file_as_root(tmp_path):
    (tmp_path / "pending-wandb-sync").write_text("not a directory")
    with pytest.raises(ValueError, match="directory"):
        recovery_inventory(tmp_path)


def test_final_audit_submission_requests_only_cpu_without_credentials(tmp_path, monkeypatch):
    import cloud14_submit as submit
    campaign = {"jobs": [{"job_id": "preserve-owned-gpu"}], "preparation_jobs": [],
                "runtime_bundle_sha256": "a" * 64, "source_revision": "reviewed-dev",
                "ambiguous_submissions": []}
    (tmp_path / "campaign.json").write_text(json.dumps(campaign))
    monkeypatch.setattr(submit, "ARTIFACTS", tmp_path)
    monkeypatch.setattr(sys, "argv", ["cloud14_submit.py", "final-audit"])
    def run(command, **kwargs):
        source = next(line for line in kwargs["input"].splitlines() if line.startswith("command="))
        invocation = ast.literal_eval(source.split("=", 1)[1])
        assert invocation[invocation.index("--gpus") + 1] == "cpu"
        assert invocation[invocation.index("--entry") + 1] == "scripts/cloud/cloud14_final_audit.py"
        assert invocation[invocation.index("--args") + 1] == "inventory"
        assert not any("WANDB_API_KEY=" in argument for argument in invocation)
        assert "SR006.nfs3/mlsub-logs" in kwargs["input"]
        return subprocess.CompletedProcess(command, 0, "lm-mpi-job-abcdef12-1234", "")
    monkeypatch.setattr(submit.subprocess, "run", run)
    submit.main()
    result = json.loads((tmp_path / "campaign.json").read_text())
    assert result["jobs"] == campaign["jobs"]
    assert result["preparation_jobs"][-1]["mode"] == "final-audit"
    assert not result["ambiguous_submissions"]


def test_audit_refuses_assets_link_without_reading_foreign_record(tmp_path, monkeypatch, capsys):
    state = tmp_path / "state"
    state.mkdir()
    foreign = tmp_path / "foreign"
    foreign.mkdir()
    (foreign / "ownership.json").write_text('{"secret":"must-not-print"}')
    assets = tmp_path / "assets"
    assets.symlink_to(foreign, target_is_directory=True)
    monkeypatch.setattr(audit, "STATE", state)
    monkeypatch.setattr(audit, "ASSETS", assets)
    monkeypatch.setattr(audit, "require_space", lambda *args: [])
    monkeypatch.setattr(sys, "argv", ["audit", "inventory"])
    with pytest.raises(ValueError, match="symlink"):
        audit.main()
    assert "must-not-print" not in capsys.readouterr().out


def test_record_reader_and_phase_listing_refuse_links_and_nonfiles(tmp_path):
    foreign = tmp_path / "foreign.json"
    foreign.write_text('{"phase":"foreign"}')
    link = tmp_path / "linked.json"
    link.symlink_to(foreign)
    with pytest.raises(ValueError, match="symlink"):
        audit.read_record(link)
    directory = tmp_path / "directory.json"
    directory.mkdir()
    with pytest.raises(ValueError, match="regular"):
        audit.read_record(directory)
    records = tmp_path / "records"
    records.mkdir()
    (records / "linked.json").symlink_to(foreign)
    with pytest.raises(ValueError, match="symlink"):
        audit.record_phases(records)
    linked_directory = tmp_path / "tasks"
    linked_directory.symlink_to(tmp_path, target_is_directory=True)
    with pytest.raises(ValueError, match="symlink"):
        audit.record_phases(linked_directory)


def test_missing_optional_asset_record_after_cleanup_is_allowed(tmp_path):
    assert audit.read_record(tmp_path / "assets" / "ownership.json") is None


def test_phase_listing_propagates_directory_read_failure(tmp_path, monkeypatch):
    records = tmp_path / "workers"
    records.mkdir()
    (records / "worker.json").write_text('{"phase":"complete"}')
    original = Path.iterdir
    def denied(path):
        if path == records:
            raise PermissionError("unreadable records")
        return original(path)
    monkeypatch.setattr(Path, "iterdir", denied)
    with pytest.raises(PermissionError, match="unreadable records"):
        audit.record_phases(records)
