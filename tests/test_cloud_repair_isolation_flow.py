"""Reviewed asset-only authority closes claims before reconciling or deleting."""

from copy import deepcopy
import fcntl
import json

import pytest

from test_cloud_repair_cleanup import campaign
from test_cloud_repair_closure import completed
from test_cloud_repair_finalize import complete
from test_cloud_repair_isolation import isolated
import cloud_repair_cleanup as cleanup
import cloud_repair_closure as closure
import cloud_repair_finalize as final
import cloud_repair_entry as entry
import cloud_repair_isolation as isolation
import cloud_repair_submit as submit


@pytest.fixture
def asset_authority(campaign, completed, monkeypatch):
    cleanup.BUNDLE_SHA.write_text("c" * 64)
    monkeypatch.setattr(closure, "STATE", cleanup.STATE)
    monkeypatch.setattr(closure, "BUNDLE_SHA", cleanup.BUNDLE_SHA)
    monkeypatch.setattr(closure, "require_space", lambda *args: [])
    (cleanup.STATE / "claims.lock").write_text("permanent claims lock")
    for worker in range(8):
        (cleanup.STATE / "workers" / f"{worker}.lock").touch()
    request = deepcopy(completed)
    proof = request["proof"]
    proof["runtime_bundle_sha256"] = "c" * 64
    proof["allocation_proof"].update(all_owned_jobs_terminal=False, all_known_jobs_terminal=True)
    for key, row in proof["scientific_verification"].items():
        row["diagnostic_manifest_sha256"] = campaign["diagnostic_manifests"][key]
    request["review"]["proof_sha256"] = closure.canonical_sha(proof)
    authority = {**campaign, "asset_isolation_only": True, "all_owned_jobs_terminal": False,
                 "all_known_jobs_terminal": True, "runtime_bundle_sha256": "c" * 64,
                 "independent_verification_sha256": "b" * 64,
                 "export_manifest_sha256": proof["exports"]["manifest_sha256"],
                 "owned_terminal_jobs": proof["allocation_proof"]["known_allocation_ids"],
                 "asset_isolation_request": request}
    return authority


def test_asset_only_reconciliation_closes_before_cleanup_and_preserves_claims(asset_authority):
    authority_bytes = json.dumps(asset_authority, sort_keys=True)
    cleanup.reconcile(asset_authority)
    marker = cleanup.STATE / "campaign-closed.json"
    assert marker.is_file() and cleanup.ASSETS.is_dir()
    closure.validate_closure(asset_authority["asset_isolation_request"])
    claims = {path.name: path.read_bytes() for path in (cleanup.STATE / "tasks").glob("*.json")}
    marker_bytes = marker.read_bytes()
    result = cleanup.cleanup(asset_authority)
    assert result["phase"] == "deleted" and not cleanup.ASSETS.exists()
    assert marker.read_bytes() == marker_bytes
    assert all((cleanup.STATE / "tasks" / name).read_bytes() == value for name, value in claims.items())
    assert len(list((cleanup.STATE / "verified-raw-logs").iterdir())) == 8
    assert asset_authority["all_owned_jobs_terminal"] is False
    assert json.dumps(asset_authority, sort_keys=True) == authority_bytes
    with pytest.raises(RuntimeError, match="permanently closed"):
        closure.assert_campaign_open()


def test_asset_only_cleanup_requires_actual_permanent_closure(asset_authority):
    with pytest.raises((ValueError, OSError)):
        cleanup.cleanup(asset_authority)
    assert cleanup.ASSETS.exists()


def test_active_worker_lock_blocks_asset_only_reconciliation(asset_authority):
    with (cleanup.STATE / "workers/0.lock").open("r+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(BlockingIOError):
            cleanup.reconcile(asset_authority)
    assert not (cleanup.STATE / "campaign-closed.json").exists()
    assert cleanup.ASSETS.exists()


@pytest.mark.parametrize("damage", ["review", "export", "diagnostic", "false_terminal"])
def test_incomplete_or_mismatched_asset_authority_refuses_all_deletion(asset_authority, damage):
    if damage == "review":
        asset_authority["asset_isolation_request"]["review"]["proof_sha256"] = "f" * 64
    elif damage == "export":
        asset_authority["export_manifest_sha256"] = "f" * 64
    elif damage == "diagnostic":
        key = next(iter(asset_authority["diagnostic_manifests"]))
        asset_authority["diagnostic_manifests"][key] = "f" * 64
    else:
        asset_authority["all_owned_jobs_terminal"] = True
    with pytest.raises(ValueError):
        cleanup.reconcile(asset_authority)
    assert not (cleanup.STATE / "campaign-closed.json").exists()
    assert cleanup.ASSETS.exists()


@pytest.fixture
def prepared(isolated, monkeypatch):
    artifacts, source = isolated
    checksum = artifacts / "fixture-bundle.sha256"
    checksum.write_text(json.loads((artifacts / "campaign.json").read_text())["runtime_bundle_sha256"])
    monkeypatch.setattr(closure, "BUNDLE_SHA", checksum)
    monkeypatch.setattr(submit, "ARTIFACTS", artifacts)
    original = artifacts.parent / "original"
    result = final.finalize(artifacts, original, source_review=source)
    assert result["authorization_created"] is False
    assert not (artifacts / "cleanup-authorization.json").exists()
    proof = json.loads((artifacts / "asset-isolation-proof.json").read_text())
    review = artifacts / "closure-review.json"
    review.write_text(json.dumps({"decision": isolation.CLOSURE_REVIEW_DECISION,
        "proof_sha256": isolation.canonical_sha(proof), "review_sha256": "f" * 64}))
    return artifacts, original, source, review


def test_preparation_exports_without_authority_then_requires_exact_independent_review(prepared):
    artifacts, original, source, review = prepared
    before = json.loads((artifacts / "campaign.json").read_text())
    authority = final.finalize(artifacts, original, source_review=source, closure_review=review)
    assert authority["all_owned_jobs_terminal"] is False and authority["all_known_jobs_terminal"] is True
    assert authority["asset_isolation_only"] is True and authority["verified_runs"] == 8
    assert json.loads((artifacts / "campaign.json").read_text()) == before
    assert submit.final_authorization(before) == authority
    assert final.finalize(artifacts, original, source_review=source, closure_review=review) == authority


def test_wrong_closure_review_cannot_create_cleanup_authority(prepared):
    artifacts, original, source, review = prepared
    value = json.loads(review.read_text())
    value["proof_sha256"] = "0" * 64
    review.write_text(json.dumps(value))
    with pytest.raises(ValueError):
        final.finalize(artifacts, original, source_review=source, closure_review=review)
    assert not (artifacts / "cleanup-authorization.json").exists()


@pytest.mark.parametrize("damage", ["unknown", "source", "new_gpu", "wrong_cpu_authority"])
def test_submission_revalidates_retained_requests_source_and_post_review_allocations(prepared, damage):
    artifacts, original, source, review = prepared
    final.finalize(artifacts, original, source_review=source, closure_review=review)
    current = json.loads((artifacts / "campaign.json").read_text())
    if damage == "unknown":
        current["superseded_unreceived_gpu_intents"][0]["retained_note"] = "changed"
    elif damage == "source":
        current["source_revision"] = "d" * 40
    else:
        current["auxiliary_jobs"].append({"job_id": "lm-mpi-job-ffffffff", "note": "later",
            "source_revision": current["source_revision"],
            "mode": "worker" if damage == "new_gpu" else "reconcile",
            "worker": 0 if damage == "new_gpu" else None,
            "gpus": 1 if damage == "new_gpu" else 0,
            "final_authorization_sha256": "0" * 64})
    with pytest.raises((ValueError, AssertionError)):
        submit.final_authorization(current)


@pytest.mark.parametrize("mode", ["inspect", "reconcile", "cleanup"])
def test_only_reviewed_zero_gpu_bookkeeping_can_follow_stable_authority(prepared, mode):
    artifacts, original, source, review = prepared
    authority = final.finalize(artifacts, original, source_review=source, closure_review=review)
    current = json.loads((artifacts / "campaign.json").read_text())
    current["auxiliary_jobs"].append({"job_id": "lm-mpi-job-fefefefe", "note": "later-" + mode,
        "source_revision": current["source_revision"], "mode": mode, "worker": None, "gpus": 0,
        "final_authorization_sha256": cleanup.sha(authority)})
    assert submit.final_authorization(current) == authority


@pytest.mark.parametrize("asset_state", ["absent", "directory", "broken_link"])
def test_read_only_inspection_distinguishes_absence_from_missing_ready_file(tmp_path, monkeypatch, asset_state):
    state, assets = tmp_path / "state", tmp_path / "assets"
    state.mkdir()
    if asset_state == "directory":
        assets.mkdir()
    elif asset_state == "broken_link":
        assets.symlink_to(tmp_path / "missing")
    monkeypatch.setattr(entry, "STATE", state)
    monkeypatch.setattr(entry, "ASSETS", assets)
    monkeypatch.setattr(entry, "affected_paths", lambda: [state, assets])
    monkeypatch.setattr(entry, "disk_records", lambda paths: [])
    monkeypatch.delenv("REPAIR_INSPECT_UPLOAD", raising=False)
    observed = {}
    def snapshot(value):
        observed.update(value)
        return []
    monkeypatch.setattr(entry, "snapshot_lines", snapshot)
    entry.inspect()
    assert observed["assets_root"] == str(assets)
    assert observed["assets_ready"] is False
    assert observed["assets_exist"] is (asset_state != "absent")
