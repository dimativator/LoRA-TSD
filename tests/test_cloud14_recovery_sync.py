"""Recovery deduplication and archival require independently verified evidence."""

import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts/cloud"))
import cloud14_recovery_sync as sync
from cloud14_core import CAMPAIGN, digest, save


def completed_state(path):
    for directory, count in [("workers", 14), ("tasks", 138)]:
        for index in range(count):
            save(path / directory / (str(index) + ".json"), {"phase": "complete"})
    return {"campaign": CAMPAIGN, "verified_runs": 180, "all_owned_jobs_terminal": True}


def test_manifest_deduplicates_content_including_recursive_wandb_copies(tmp_path):
    root = tmp_path / "pending-wandb-sync"
    (root / "attempt/wandb/files/wandb/files").mkdir(parents=True)
    (root / "attempt/process.log").write_text("abc")
    (root / "attempt/wandb/files/wandb/files/process.log").write_text("abc")
    (root / "attempt/other.log").write_text("xyz")
    value, unique = sync.manifest(tmp_path)
    assert len(value["files"]) == 3 and len(unique) == 2
    assert value["original_bytes"] == 9
    assert all(row["name"] == "blob-" + row["sha256"] + ".bin" for row in value["unique_files"])


def test_manifest_refuses_symlinks_and_unexpected_payloads(tmp_path):
    root = tmp_path / "pending-wandb-sync"
    root.mkdir()
    (root / "link.log").symlink_to(tmp_path / "foreign.log")
    with pytest.raises(AssertionError, match="links"):
        sync.manifest(tmp_path)
    (root / "link.log").unlink()
    (root / "model.pt").write_text("not a diagnostic")
    with pytest.raises(AssertionError, match="type"):
        sync.manifest(tmp_path)


def test_completion_gate_refuses_live_worker_and_incomplete_grid(tmp_path):
    audit = completed_state(tmp_path)
    sync.complete(tmp_path, audit)
    save(tmp_path / "workers/0.json", {"phase": "active"})
    with pytest.raises(AssertionError):
        sync.complete(tmp_path, audit)
    save(tmp_path / "workers/0.json", {"phase": "complete"})
    (tmp_path / "tasks/0.json").unlink()
    with pytest.raises(AssertionError):
        sync.complete(tmp_path, audit)


def test_upload_preserves_original_until_independent_verification(tmp_path, monkeypatch):
    state = tmp_path / "state"
    audit = completed_state(state)
    source = state / "pending-wandb-sync/attempt/process.log"
    source.parent.mkdir(parents=True)
    source.write_text("unique evidence")
    work = tmp_path / "working"
    work.mkdir()
    saved = []
    class Run:
        def __init__(self, run_id):
            self.id = run_id
            self.path = "andrey/LoRA-Muon/" + run_id
        def save(self, path, **kwargs):
            saved.append(Path(path))
        def log(self, value):
            assert value["recovery/unique_files"] == 1
        def finish(self, exit_code):
            assert exit_code == 0
    def initialize(**kwargs):
        receipt = json.loads((state / "recovery-upload.json").read_text())
        assert receipt["phase"] == "initializing" and receipt["run_id"] == kwargs["id"]
        assert kwargs["resume"] == "allow"
        return Run(kwargs["id"])
    monkeypatch.setitem(sys.modules, "wandb", SimpleNamespace(init=initialize, Settings=lambda **kwargs: kwargs))
    monkeypatch.setattr(sync, "require_space", lambda *args: [])
    monkeypatch.setattr(sync, "decode_audit_environment", lambda prefix: audit)
    sync.upload(work, state)
    receipt = json.loads((state / "recovery-upload.json").read_text())
    assert receipt["phase"] == "uploaded_pending_independent_verification"
    assert receipt["manifest_sha256"] == digest(state / "recovery-manifest.json")
    assert source.read_text() == "unique evidence"
    assert len(saved) == 2
    with pytest.raises(AssertionError, match="adopted"):
        sync.upload(work, state)


def test_archive_moves_only_unchanged_independently_verified_spool(tmp_path, monkeypatch):
    audit = completed_state(tmp_path)
    source = tmp_path / "pending-wandb-sync/attempt/process.log"
    source.parent.mkdir(parents=True)
    source.write_text("abc")
    value, unique = sync.manifest(tmp_path)
    save(tmp_path / "recovery-manifest.json", value)
    sha256 = digest(tmp_path / "recovery-manifest.json")
    receipt = {"phase": "uploaded_pending_independent_verification", "manifest_sha256": sha256,
               "run_path": "andrey/LoRA-Muon/recovery123", "unique_files": 1}
    save(tmp_path / "recovery-upload.json", receipt)
    audit.update(all_unique_files_downloaded_sha256_verified=False, manifest_sha256=sha256,
                 run_path=receipt["run_path"], unique_files=1)
    monkeypatch.setattr(sync, "decode_audit_environment", lambda prefix: audit)
    monkeypatch.setattr(sync, "require_space", lambda *args, **kwargs: [])
    with pytest.raises(AssertionError):
        sync.archive(tmp_path)
    assert source.exists()
    audit["all_unique_files_downloaded_sha256_verified"] = True
    source.write_text("xyz")
    with pytest.raises(AssertionError, match="changed"):
        sync.archive(tmp_path)
    source.write_text("abc")
    result = sync.archive(tmp_path)
    assert not (tmp_path / "pending-wandb-sync").exists()
    assert (Path(result["archive"]) / "attempt/process.log").read_text() == "abc"
    assert json.loads((tmp_path / "recovery-upload.json").read_text())["phase"] == "independently_verified_archived"


def test_initialization_ambiguity_retains_and_reuses_durable_run_id(tmp_path, monkeypatch):
    state = tmp_path / "state"
    audit = completed_state(state)
    source = state / "pending-wandb-sync/attempt/process.log"
    source.parent.mkdir(parents=True)
    source.write_text("original")
    first = tmp_path / "first"
    first.mkdir()
    seen = []
    def ambiguous(**kwargs):
        seen.append(kwargs["id"])
        receipt = json.loads((state / "recovery-upload.json").read_text())
        assert receipt["run_id"] == kwargs["id"]
        raise RuntimeError("lost initialization response")
    fake = SimpleNamespace(init=ambiguous, Settings=lambda **kwargs: kwargs)
    monkeypatch.setitem(sys.modules, "wandb", fake)
    monkeypatch.setattr(sync, "require_space", lambda *args: [])
    monkeypatch.setattr(sync, "decode_audit_environment", lambda prefix: audit)
    with pytest.raises(RuntimeError, match="lost initialization"):
        sync.upload(first, state)
    second = tmp_path / "second"
    second.mkdir()
    with pytest.raises(RuntimeError, match="lost initialization"):
        sync.upload(second, state)
    assert len(seen) == 2 and seen[0] == seen[1]
    assert source.read_text() == "original"
    original_manifest = (state / "recovery-manifest.json").read_bytes()
    original_receipt = (state / "recovery-upload.json").read_bytes()
    source.write_text("changed!")
    third = tmp_path / "third"
    third.mkdir()
    with pytest.raises(AssertionError):
        sync.upload(third, state)
    assert (state / "recovery-manifest.json").read_bytes() == original_manifest
    assert (state / "recovery-upload.json").read_bytes() == original_receipt
