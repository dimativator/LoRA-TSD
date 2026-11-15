"""CPU-only cleanup gates using complete, miniature campaign-owned fixtures."""

import hashlib
from contextlib import contextmanager
import json
import os
from pathlib import Path
import sys

import pytest

CLOUD = Path(__file__).resolve().parents[1] / "scripts/cloud"
sys.path.insert(0, str(CLOUD))
import cloud_repair_cleanup as cleanup


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def record(data):
    return {"bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}


@pytest.fixture
def campaign(tmp_path, monkeypatch):
    home = tmp_path / "home" / cleanup.NAMESPACE
    assets = home / "assets"
    state = tmp_path / "nfs3" / cleanup.NAMESPACE
    assets.mkdir(parents=True)
    state.mkdir(parents=True)
    bundle_sha = tmp_path / "bundle.sha256"
    bundle_sha.write_text("c" * 64)
    monkeypatch.setattr(cleanup, "HOME", home)
    monkeypatch.setattr(cleanup, "ASSETS", assets)
    monkeypatch.setattr(cleanup, "STATE", state)
    monkeypatch.setattr(cleanup, "BUNDLE_SHA", bundle_sha)
    monkeypatch.setattr(cleanup, "affected_paths", lambda: [home, state, assets])
    monkeypatch.setattr(cleanup, "require_space", lambda *args: [{"available_bytes": 9_000_000_000}])
    expected = {"model/weights.bin": record(b"model"), "data/boolq/train.parquet": record(b"dataset")}
    monkeypatch.setattr(cleanup, "manifests", lambda: expected)
    for name, data in (("model/weights.bin", b"model"), ("data/boolq/train.parquet", b"dataset")):
        path = assets / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    write(assets / "ownership.json", {"campaign": cleanup.CAMPAIGN, "root": str(assets)})
    write(assets / "ready.json", {"campaign": cleanup.CAMPAIGN, "revision": cleanup.REVISION,
                                "files": expected})
    confirmation = {"campaign": cleanup.CAMPAIGN, "verified_runs": len(cleanup.TASKS),
                    "all_owned_jobs_terminal": True, "all_diagnostics_independently_verified": True,
                    "exports_verified": True, "diagnostic_manifests": {}}
    for index, task in enumerate(cleanup.TASKS):
        key = cleanup.task_key(task)
        directory = state / "pending-wandb-sync" / key
        directory.mkdir(parents=True)
        source = directory / "history.jsonl"
        source.write_bytes(b'{"step":3000}\n')
        manifest = {"campaign": cleanup.CAMPAIGN, "model_revision": cleanup.REVISION,
                    "runtime_bundle_sha256": "c" * 64, "run_name": cleanup.run_name(task),
                    "wandb_id": f"run{index}", "command": ["python", "main.py"],
                    "files": [{"path": "history.jsonl", "wandb_name": "history.jsonl",
                               **record(source.read_bytes())}]}
        manifest_path = directory / "diagnostic-manifest.json"
        write(manifest_path, manifest)
        manifest_sha = record(manifest_path.read_bytes())["sha256"]
        upload = directory / "upload"
        upload.mkdir()
        os.link(source, upload / source.name)
        os.link(manifest_path, upload / manifest_path.name)
        status = {"task": task, "worker": index, "phase": "complete", "returncode": 0,
                  "child_pid": None, "wandb_sync_complete": True, "wandb_id": f"run{index}",
                  "verification": {"train_records": 3000, "validation_points": 12},
                  "command": ["python", "main.py"], "recovery_logs": str(directory),
                  "recovery_manifest_sha256": manifest_sha,
                  "syncphase": "uploaded_awaiting_independent_verification"}
        write(state / "tasks" / (key + ".json"), status)
        write(state / "workers" / f"{index}.json", status)
        confirmation["diagnostic_manifests"][key] = manifest_sha
    return confirmation


def first_pending():
    return cleanup.STATE / "pending-wandb-sync" / cleanup.task_key(cleanup.TASKS[0])


def test_reconciliation_preserves_every_source_and_is_repeatable(campaign):
    source_bytes = (first_pending() / "history.jsonl").read_bytes()
    result = cleanup.reconcile(campaign)
    assert len(result["reconciled"]) == 8 and not list((cleanup.STATE / "pending-wandb-sync").iterdir())
    archived = cleanup.STATE / "verified-raw-logs" / cleanup.task_key(cleanup.TASKS[0])
    assert (archived / "history.jsonl").read_bytes() == source_bytes
    assert (archived / "upload/history.jsonl").stat().st_ino == (archived / "history.jsonl").stat().st_ino
    status = json.loads((cleanup.STATE / "workers/0.json").read_text())
    assert status["syncphase"] == "independently_verified" and status["recovery_logs"] == str(archived)
    assert cleanup.reconcile(campaign)["reconciled"] == result["reconciled"]


def test_deletes_only_new_owned_assets_preserves_logs_venv_and_other_projects(campaign):
    other = cleanup.HOME.parent / "dimativator-paper8b-three-seeds-20261113/venv/keep"
    other.parent.mkdir(parents=True)
    other.write_text("old environment")
    unrelated = cleanup.HOME / "unrelated-data"
    unrelated.write_text("keep")
    cleanup.reconcile(campaign)
    result = cleanup.cleanup(campaign)
    assert result["phase"] == "deleted" and not cleanup.ASSETS.exists()
    assert other.read_text() == "old environment" and unrelated.read_text() == "keep"
    assert (cleanup.STATE / "verified-raw-logs").is_dir()
    assert len(list((cleanup.STATE / "tasks").glob("*.json"))) == 8
    assert cleanup.cleanup(campaign)["reconciled_absence"] is True


@pytest.mark.parametrize("field", ["all_owned_jobs_terminal", "all_diagnostics_independently_verified", "exports_verified"])
def test_missing_independent_final_proof_refuses_all_mutation(campaign, field):
    campaign[field] = False
    with pytest.raises(ValueError, match="attestation"):
        cleanup.reconcile(campaign)
    assert first_pending().exists() and cleanup.ASSETS.exists()


def test_manifest_sha_or_source_corruption_is_not_reconciled(campaign):
    source = first_pending() / "history.jsonl"
    source.write_bytes(b'{"step":2000}\n')
    with pytest.raises(ValueError, match="SHA mismatch"):
        cleanup.reconcile(campaign)
    assert first_pending().exists() and not (cleanup.STATE / "verified-raw-logs" / first_pending().name).exists()


def test_manifest_bytes_must_match_exact_independent_hash(campaign):
    path = first_pending() / "diagnostic-manifest.json"
    path.write_text(path.read_text() + "\n")
    with pytest.raises(ValueError, match="SHA mismatch"):
        cleanup.reconcile(campaign)


def test_unknown_spool_file_is_preserved_and_blocks_reconciliation(campaign):
    (first_pending() / "unique-unsynced.bin").write_bytes(b"evidence")
    with pytest.raises(ValueError, match="coverage"):
        cleanup.reconcile(campaign)
    assert (first_pending() / "unique-unsynced.bin").read_bytes() == b"evidence"


def test_only_proven_redundant_upload_hardlinks_are_ignored(campaign):
    unique = first_pending() / "upload/history.jsonl"
    unique.unlink()
    unique.write_bytes(b"unique diagnostic evidence")
    with pytest.raises(ValueError, match="redundant source hardlink"):
        cleanup.reconcile(campaign)
    assert unique.read_bytes() == b"unique diagnostic evidence" and first_pending().exists()


@pytest.mark.parametrize("location", ["spool", "upload", "assets"])
def test_links_never_followed_or_deleted(campaign, tmp_path, location):
    target = tmp_path / "foreign"
    target.write_text("must survive")
    path = {"spool": first_pending() / "linked.json", "upload": first_pending() / "upload/linked.json",
            "assets": cleanup.ASSETS / "data/link"}[location]
    path.symlink_to(target)
    with pytest.raises(ValueError, match="Link"):
        cleanup.reconcile(campaign) if location != "assets" else (
            cleanup.reconcile(campaign), cleanup.cleanup(campaign))
    assert target.read_text() == "must survive" and path.is_symlink()


def test_incomplete_worker_and_missing_final_eval_block(campaign):
    path = cleanup.STATE / "workers/0.json"
    status = json.loads(path.read_text())
    status["verification"]["validation_points"] = 11
    write(path, status)
    with pytest.raises(ValueError, match="final-validation"):
        cleanup.reconcile(campaign)


def test_unknown_asset_or_pin_mismatch_blocks_deletion(campaign):
    cleanup.reconcile(campaign)
    unknown = cleanup.ASSETS / "foreign.txt"
    unknown.write_text("preserve")
    with pytest.raises(ValueError, match="Unknown or missing assets"):
        cleanup.cleanup(campaign)
    assert unknown.exists() and not (cleanup.STATE / "cleanup-intent.json").exists()
    unknown.unlink()
    (cleanup.ASSETS / "model/weights.bin").write_bytes(b"wrong")
    with pytest.raises(ValueError, match="SHA mismatch"):
        cleanup.cleanup(campaign)
    assert cleanup.ASSETS.exists()


def test_wrong_asset_namespace_cannot_be_adopted(campaign, tmp_path, monkeypatch):
    foreign = tmp_path / "other-project/assets"
    monkeypatch.setattr(cleanup, "ASSETS", foreign)
    with pytest.raises(ValueError, match="foreign"):
        cleanup.cleanup(campaign)


def test_pending_evidence_blocks_cleanup_even_after_prior_reconciliation(campaign):
    cleanup.reconcile(campaign)
    pending = cleanup.STATE / "pending-wandb-sync/new-evidence"
    pending.mkdir()
    (pending / "unsynced.log").write_text("unique")
    with pytest.raises(ValueError, match="pending"):
        cleanup.cleanup(campaign)
    assert cleanup.ASSETS.exists() and (pending / "unsynced.log").exists()


def test_post_telemetry_failure_preserves_deletion_receipt_and_matching_resume(campaign, monkeypatch):
    cleanup.reconcile(campaign)
    def unavailable_after_delete(*args):
        if not cleanup.ASSETS.exists():
            raise OSError("post telemetry failed")
        return []
    monkeypatch.setattr(cleanup, "require_space", unavailable_after_delete)
    with pytest.raises(OSError, match="post telemetry"):
        cleanup.cleanup(campaign)
    receipt = json.loads((cleanup.STATE / "cleanup.json").read_text())
    assert receipt["phase"] == "deleted" and receipt["assets_verified"] is True
    monkeypatch.setattr(cleanup, "require_space", lambda *args: [])
    assert cleanup.cleanup(campaign)["reconciled_absence"] is True


def test_absence_without_prior_intent_is_not_claimed_as_deletion(campaign):
    cleanup.reconcile(campaign)
    import shutil
    shutil.rmtree(cleanup.ASSETS)
    with pytest.raises(ValueError, match="Missing required evidence"):
        cleanup.cleanup(campaign)


def test_independent_archive_is_reverified_before_deletion(campaign):
    cleanup.reconcile(campaign)
    archived = cleanup.STATE / "verified-raw-logs" / cleanup.task_key(cleanup.TASKS[0]) / "history.jsonl"
    archived.write_bytes(b'{"step":2000}\n')
    with pytest.raises(ValueError, match="SHA mismatch"):
        cleanup.cleanup(campaign)
    assert cleanup.ASSETS.exists()


def test_ownership_marker_must_match_exact_campaign(campaign):
    cleanup.reconcile(campaign)
    write(cleanup.ASSETS / "ownership.json", {"campaign": "other", "root": str(cleanup.ASSETS)})
    with pytest.raises(ValueError, match="ownership"):
        cleanup.cleanup(campaign)
    assert cleanup.ASSETS.exists()


@pytest.mark.parametrize("mode", ["reconcile", "cleanup"])
def test_final_parent_modes_run_only_guarded_cpu_children(monkeypatch, capsys, mode):
    import cloud_repair_entry as entry
    import cloud_repair_runtime as runtime
    calls = []
    class Process:
        returncode = 0
        def wait(self):
            calls.append("wait")
    @contextmanager
    def owned(command, directory, environment):
        assert command == [str(entry.PYTHON), str(Path(entry.__file__).resolve()), mode + "-child"]
        assert directory.is_dir() and environment["REPAIR_TEST_GUARD"] == "present"
        (directory / "process.log").write_text("guarded final audit\n")
        calls.append("guard-start")
        yield Process()
        calls.append("guard-end")
    monkeypatch.setattr(runtime, "owned_process", owned)
    monkeypatch.setenv("REPAIR_TEST_GUARD", "present")
    monkeypatch.setattr(sys, "argv", ["cloud_repair_entry.py", mode])
    entry.main()
    assert calls == ["guard-start", "wait", "guard-end"]
    assert "guarded final audit" in capsys.readouterr().out


@pytest.mark.parametrize("mode", ["reconcile", "cleanup"])
def test_final_child_modes_decode_exact_attestation_and_emit_snapshot(campaign, monkeypatch, capsys, mode):
    import cloud_repair_entry as entry
    from cloud14_core import decode_snapshot, encode_audit_environment
    seen = []
    def action(confirmation):
        seen.append(confirmation)
        return {"campaign": cleanup.CAMPAIGN, "phase": mode + "-proof"}
    monkeypatch.setattr(cleanup, mode, action)
    for key, value in encode_audit_environment("REPAIR_FINAL_AUDIT", campaign).items():
        monkeypatch.setenv(key, value)
    monkeypatch.setattr(sys, "argv", ["cloud_repair_entry.py", mode + "-child"])
    entry.main()
    assert seen == [campaign]
    assert decode_snapshot(capsys.readouterr().out)["phase"] == mode + "-proof"


def test_assets_compatibility_cleanup_cannot_bypass_strict_gate(monkeypatch):
    import cloud_repair_assets as assets
    def refuse(confirmation):
        raise ValueError("strict final proof required")
    monkeypatch.setattr(cleanup, "cleanup", refuse)
    with pytest.raises(ValueError, match="strict final proof"):
        assets.cleanup({})
