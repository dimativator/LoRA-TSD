"""CPU-only permanent claim closure; assets, GPU APIs and claims stay untouched."""

import fcntl
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts/cloud"))
import cloud_repair_closure as closure


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def reviewed(request):
    request["review"]["proof_sha256"] = closure.canonical_sha(request["proof"])
    return request


@pytest.fixture
def completed(tmp_path, monkeypatch):
    state = tmp_path / "state"
    (state / "workers").mkdir(parents=True)
    monkeypatch.setattr(closure, "STATE", state)
    monkeypatch.setattr(closure, "require_space", lambda *args: [])
    checksum = tmp_path / "bundle.sha256"
    checksum.write_text("a" * 64)
    monkeypatch.setattr(closure, "BUNDLE_SHA", checksum)
    (state / "claims.lock").write_text("keep immutable lock content")
    scientific = {}
    for worker, task in enumerate(closure.TASKS):
        key = closure.task_key(task)
        (state / "workers" / f"{worker}.lock").touch()
        proof = {"worker": worker, "wandb_id": "run" + str(worker),
                 "wandb_path": "andrey/LoRA-Muon/run" + str(worker),
                 "returncode": 0, "train_records": 3000, "validation_points": 12,
                 **{key: "b" * 64 for key in ("diagnostic_manifest_sha256", "independent_verification_sha256",
                                            "history_sha256", "result_sha256", "diagnostic_inventory_sha256")}}
        scientific[key] = proof
        status = {"task": task, "worker": worker, "phase": "complete", "returncode": 0,
                  "child_pid": None, "wandb_sync_complete": True, "wandb_id": proof["wandb_id"],
                  "recovery_manifest_sha256": proof["diagnostic_manifest_sha256"],
                  "verification": {"train_records": 3000, "validation_points": 12},
                  "recovery_logs": str(state / "pending-wandb-sync" / key),
                  "syncphase": "uploaded_awaiting_independent_verification", "command": ["python", "main.py"]}
        write(state / "tasks" / (key + ".json"), status)
        write(state / "workers" / f"{worker}.json", status)
    original, current = "1" * 40, "2" * 40
    proof = {"campaign": closure.CAMPAIGN, "runtime_bundle_sha256": "a" * 64,
             "scientific_verification": scientific, "exports": {"manifest_sha256": "c" * 64},
             "allocation_proof": {"ledger_sha256": "c" * 64,
                 "known_allocation_ids": ["lm-mpi-job-abcd"],
                 "known_terminal": [{"job_id": "lm-mpi-job-abcd", "note": "known-worker7", "mode": "worker",
                                     "worker": 7, "gpus": 1, "status": "Completed", "evidence_sha256": "d" * 64}],
                 "unknown_requests": [{"note": "original-unreceived-worker0", "mode": "worker", "worker": 0,
                    "gpus": 1, "source_revision": original, "evidence_sha256": "d" * 64}]},
             "source_gate_proof": {"review_sha256": "e" * 64, "published_history_sha256": "e" * 64,
                 "current_revision": current, "sources": [
                     {"source_revision": original, "path": "scripts/cloud/cloud_repair_runtime.py",
                      "source_sha256": "f" * 64, "review_sha256": "e" * 64,
                      "gates": ["complete_claim_before_assets_cuda"]},
                     {"source_revision": current, "path": "scripts/cloud/cloud_repair_runtime.py",
                      "source_sha256": "f" * 64, "review_sha256": "e" * 64,
                      "gates": ["worker_closed_before_assets_cuda", "profile_closed_before_assets_cuda", "read_only_metadata"]}]}}
    request = {"proof": proof, "review": {"decision": "approved_for_campaign_closure_only", "review_sha256": "e" * 64}}
    return reviewed(request)


def first_task():
    return closure.STATE / "tasks" / (closure.task_key(closure.TASKS[0]) + ".json")


def test_closure_persists_exact_record_hashes_and_never_authorizes_cleanup(completed):
    before = {str(path.relative_to(closure.STATE)): path.read_bytes() for path in closure.STATE.rglob("*") if path.is_file()}
    marker = closure.close_campaign(completed)
    assert marker["closure_only"] is True and marker["cleanup_authorization"] is False
    assert marker["all_owned_jobs_terminal"] is False
    assert marker["reviewed_request_sha256"] == closure.canonical_sha(completed)
    key = closure.task_key(closure.TASKS[0])
    assert marker["records"][key]["task"]["file_sha256"] == closure.hashlib.sha256(before[str(first_task().relative_to(closure.STATE))]).hexdigest()
    assert all((closure.STATE / name).read_bytes() == value for name, value in before.items())
    assert closure.validate_closure(completed) == marker
    with pytest.raises(RuntimeError, match="permanently closed"):
        closure.assert_campaign_open()


def test_idempotent_same_proof_preserves_marker_bytes(completed):
    first = closure.close_campaign(completed)
    marker = closure.STATE / "campaign-closed.json"
    before = marker.read_bytes()
    assert closure.close_campaign(completed) == first and marker.read_bytes() == before


@pytest.mark.parametrize("lock_name", ["workers/0.lock", "workers/5.lock", "claims.lock"])
def test_active_parent_or_claim_lock_blocks_closure_without_task_mutation(completed, lock_name):
    before = first_task().read_bytes()
    with (closure.STATE / lock_name).open("r+") as held:
        fcntl.flock(held, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(BlockingIOError):
            closure.close_campaign(completed)
    assert not (closure.STATE / "campaign-closed.json").exists() and first_task().read_bytes() == before
    assert closure.close_campaign(completed)["closure_only"] is True


@pytest.mark.parametrize("damage", ["phase", "returncode", "manifest", "wandb_id", "final", "missing_verification", "worker"])
def test_incomplete_or_mismatched_records_cannot_close(completed, damage):
    path = first_task()
    record = json.loads(path.read_text())
    if damage == "phase": record["phase"] = "training"
    elif damage == "returncode": record["returncode"] = None
    elif damage == "manifest": record["recovery_manifest_sha256"] = "c" * 64
    elif damage == "wandb_id": record["wandb_id"] = "wrongrun"
    elif damage == "final": record["verification"]["validation_points"] = 11
    elif damage == "missing_verification": record.pop("verification")
    else: record["worker"] = 7
    write(path, record)
    with pytest.raises(ValueError): closure.close_campaign(completed)
    assert not (closure.STATE / "campaign-closed.json").exists()


def test_bare_boolean_or_unreviewed_digest_cannot_close(completed):
    with pytest.raises(ValueError): closure.close_campaign({"all_verified": True, "all_owned_jobs_terminal": True})
    completed["proof"]["exports"]["manifest_sha256"] = "f" * 64
    with pytest.raises(ValueError, match="Exact reviewed"): closure.close_campaign(completed)


@pytest.mark.parametrize("content", ["{}", "not complete JSON", "[]"])
def test_every_malformed_marker_fails_closed_and_is_never_repaired(completed, content):
    path = closure.STATE / "campaign-closed.json"
    path.write_text(content)
    with pytest.raises(ValueError): closure.assert_campaign_open()
    with pytest.raises(ValueError): closure.close_campaign(completed)
    assert path.read_text() == content


def test_open_guard_does_not_write_before_closure(completed):
    before = sorted(str(path) for path in closure.STATE.rglob("*"))
    assert closure.assert_campaign_open() is None
    assert sorted(str(path) for path in closure.STATE.rglob("*")) == before


@pytest.mark.parametrize("target", ["marker", "task", "lock", "state"])
def test_linked_marker_state_record_or_lock_is_never_followed(completed, tmp_path, target, monkeypatch):
    foreign = tmp_path / "foreign.json"
    foreign.write_text("foreign evidence")
    path = {"marker": closure.STATE / "campaign-closed.json", "task": first_task(),
            "lock": closure.STATE / "workers/0.lock", "state": closure.STATE}[target]
    if target == "state":
        real = closure.STATE.with_name("real-state")
        closure.STATE.rename(real)
        path.symlink_to(real, target_is_directory=True)
    else:
        if path.exists(): path.unlink()
        path.symlink_to(foreign)
    with pytest.raises(ValueError): closure.close_campaign(completed)
    assert foreign.read_text() == "foreign evidence"


def test_only_expected_reconciliation_metadata_can_change_after_closure(completed):
    marker = closure.close_campaign(completed)
    for path in list((closure.STATE / "tasks").glob("*.json")) + list((closure.STATE / "workers").glob("*.json")):
        record = json.loads(path.read_text())
        record.update(recovery_logs=str(closure.STATE / "verified-raw-logs" / closure.task_key(record["task"])),
                      syncphase="independently_verified", independently_verified_at="2026-11-15T18:00:00+00:00")
        write(path, record)
    assert closure.validate_closure(completed) == marker
    assert closure.close_campaign(completed) == marker
    record = json.loads(first_task().read_text())
    record["command"] = ["a", "different", "training", "command"]
    write(first_task(), record)
    with pytest.raises(ValueError, match="outside reconciliation"): closure.validate_closure(completed)


def test_wrong_reconciliation_paths_or_pairs_are_not_accepted(completed):
    closure.close_campaign(completed)
    record = json.loads(first_task().read_text())
    record.update(recovery_logs="/other-project/logs", syncphase="independently_verified",
                  independently_verified_at="2026-11-15T18:00:00+00:00")
    write(first_task(), record)
    with pytest.raises(ValueError, match="reconciliation metadata"): closure.validate_closure(completed)


def test_different_reviewed_request_cannot_replace_permanent_winner(completed):
    closure.close_campaign(completed)
    path = closure.STATE / "campaign-closed.json"
    before = path.read_bytes()
    completed["proof"]["exports"]["manifest_sha256"] = "d" * 64
    with pytest.raises(ValueError, match="Existing closure request differs"):
        closure.close_campaign(reviewed(completed))
    assert path.read_bytes() == before


def test_permanent_marker_timestamp_is_digest_bound(completed):
    closure.close_campaign(completed)
    path = closure.STATE / "campaign-closed.json"
    marker = json.loads(path.read_text())
    marker["created_at"] = "2020-01-01T00:00:00+00:00"
    write(path, marker)
    with pytest.raises(ValueError, match="marker digest"):
        closure.validate_closure(completed)
    with pytest.raises(ValueError, match="marker digest"):
        closure.assert_campaign_open()


@pytest.mark.parametrize("available", [5_000_000_000, 5_000_000_100])
def test_marker_requires_strict_reserve_plus_its_planned_bytes(completed, monkeypatch, available):
    import cloud14_core
    monkeypatch.setattr(cloud14_core, "disk_records", lambda paths: [{
        "requested_path": str(path), "device": 44, "available_bytes": available} for path in paths])
    monkeypatch.setattr(closure, "require_space", cloud14_core.require_space)
    before = first_task().read_bytes()
    with pytest.raises(OSError, match="reserve or planned-write"):
        closure.close_campaign(completed)
    assert not (closure.STATE / "campaign-closed.json").exists() and first_task().read_bytes() == before


def test_immediate_prewrite_headroom_matches_exact_marker_size(completed, monkeypatch):
    planned = []
    def space(paths, writes):
        assert paths == [closure.STATE]
        planned.append(writes)
    monkeypatch.setattr(closure, "require_space", space)
    closure.close_campaign(completed)
    assert planned == [{closure.STATE: (closure.STATE / "campaign-closed.json").stat().st_size}]


@pytest.mark.parametrize("damage", ["unknown_source", "unknown_gpu", "false_terminal", "false_all_terminal", "missing_current_gate", "missing_digest"])
def test_unknown_source_gates_and_scientific_digests_are_mandatory(completed, damage):
    proof = completed["proof"]
    if damage == "unknown_source": proof["source_gate_proof"]["sources"][0]["gates"] = []
    elif damage == "unknown_gpu": proof["allocation_proof"]["unknown_requests"][0]["worker"] = 3
    elif damage == "false_terminal": proof["allocation_proof"]["known_terminal"][0]["status"] = "Running"
    elif damage == "false_all_terminal": proof["allocation_proof"]["all_owned_jobs_terminal"] = True
    elif damage == "missing_current_gate": proof["source_gate_proof"]["sources"][1]["gates"] = ["worker_closed_before_assets_cuda"]
    else: proof["scientific_verification"][closure.task_key(closure.TASKS[0])].pop("history_sha256")
    with pytest.raises(ValueError): closure.close_campaign(reviewed(completed))


def test_entry_and_control_blobs_at_same_revision_are_distinct_reviewed_sources(completed):
    proof = completed["proof"]
    revision = proof["source_gate_proof"]["current_revision"]
    for name in ("cloud_repair_entry.py", "cloud_repair_control.py"):
        proof["source_gate_proof"]["sources"].append({"source_revision": revision,
            "path": "scripts/cloud/" + name, "source_sha256": "f" * 64,
            "review_sha256": "e" * 64, "gates": ["read_only_metadata"]})
    proof["allocation_proof"]["unknown_requests"].append({"note": "deferred-inspect4", "mode": "inspect",
        "gpus": 0, "source_revision": revision, "evidence_sha256": "d" * 64})
    assert closure.reviewed_proof(reviewed(completed)) == proof
    proof["source_gate_proof"]["sources"].append(dict(proof["source_gate_proof"]["sources"][-1]))
    with pytest.raises(ValueError, match="source gate entry"):
        closure.reviewed_proof(reviewed(completed))


def test_unknown_cpu_inspection_cannot_claim_a_gpu_worker_identity(completed):
    proof = completed["proof"]
    unknown = proof["allocation_proof"]["unknown_requests"][0]
    unknown.update(mode="inspect", worker=7, gpus=0)
    for name in ("cloud_repair_entry.py", "cloud_repair_control.py"):
        proof["source_gate_proof"]["sources"].append({"source_revision": unknown["source_revision"],
            "path": "scripts/cloud/" + name, "source_sha256": "f" * 64,
            "review_sha256": "e" * 64, "gates": ["read_only_metadata"]})
    with pytest.raises(ValueError, match="Unknown CPU request"):
        closure.reviewed_proof(reviewed(completed))


def test_no_gpu_or_assets_are_read_or_changed(completed, monkeypatch):
    paths = []
    torch_was_loaded = "torch" in sys.modules
    original = closure.os.open
    def tracked(path, *args, **kwargs):
        path = Path(path)
        assert path.is_relative_to(closure.STATE)
        paths.append(str(path))
        return original(path, *args, **kwargs)
    monkeypatch.setattr(closure.os, "open", tracked)
    closure.close_campaign(completed)
    assert paths and all("assets" not in Path(path).relative_to(closure.STATE).parts for path in paths)
    assert ("torch" in sys.modules) == torch_was_loaded
