"""Recover lost submission responses from both scheduler views without resubmitting."""

import ast
import json
from pathlib import Path
import sys
import time
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts/cloud"))
import cloud_repair_collect as collector
import cloud_repair_submit as submit

JOB = "lm-mpi-job-12345678-1234-1234-1234-123456789abc"
SECOND = "lm-mpi-job-abcdef12-1234-1234-1234-123456789abc"
NOTE = "repair8b_inspect-3_20261115"


@pytest.fixture
def ledger(tmp_path, monkeypatch):
    monkeypatch.setattr(submit, "ARTIFACTS", tmp_path)
    value = {"campaign": submit.CAMPAIGN, "jobs": [], "auxiliary_jobs": [],
             "ambiguous_submissions": [{"intent": "inspect-3", "mode": "inspect", "worker": None,
                                         "gpus": 0, "note": NOTE}]}
    submit.save(tmp_path / "campaign.json", value)
    return tmp_path, value


def line(job=JOB, status="Pending", note=NOTE, repo="dimativator/LoRA-TSD"):
    return f"{job} {status} repo:{repo}|entry:scripts/cloud/cloud_repair_entry.py|note:{note}"


@pytest.mark.parametrize("case", ["active_only", "consistent", "partial_full"])
def test_unique_pending_recovery_accepts_active_view_and_never_submits(ledger, monkeypatch, case):
    root, _ = ledger
    calls = []
    def remote(command, *args):
        calls.append(command)
        assert command in (["mlsub", "list"], ["mlsub", "list", "--active"])
        active = command[-1] == "--active"
        raw = line() if active or case == "consistent" else ""
        return {"stdout": raw, "returncode": None if case == "partial_full" and not active else 0,
                "ssh_returncode": 0, "timed_out": case == "partial_full" and not active,
                "ssh_timed_out": False}
    monkeypatch.setattr(collector, "remote_cli", remote)
    submit.recover()
    result = json.loads((root / "campaign.json").read_text())
    assert not result["ambiguous_submissions"] and len(result["auxiliary_jobs"]) == 1
    record = result["auxiliary_jobs"][0]
    assert record["job_id"] == JOB and record["scheduler_status"] == "Pending"
    assert record["listings_complete"] is (case != "partial_full")
    assert record["absence_proof"] is False
    assert submit.digest(Path(record["unique_note_evidence"])) == record["unique_note_evidence_sha256"]
    assert calls == [["mlsub", "list"], ["mlsub", "list", "--active"]]


@pytest.mark.parametrize("case", ["two_ids", "conflict", "wrong_repo", "wrong_note"])
def test_ambiguous_or_conflicting_recovery_keeps_pending_intent(ledger, monkeypatch, case):
    root, _ = ledger
    full = line()
    active = line(job=SECOND) if case == "two_ids" else line(status="Running")
    if case == "wrong_repo": full = active = line(repo="other/repo")
    if case == "wrong_note": full = active = line(note=NOTE + "-other")
    def remote(command, *args):
        return {"stdout": active if command[-1] == "--active" else full,
                "returncode": 0, "ssh_returncode": 0, "timed_out": False, "ssh_timed_out": False}
    monkeypatch.setattr(collector, "remote_cli", remote)
    with pytest.raises(RuntimeError):
        submit.recover()
    result = json.loads((root / "campaign.json").read_text())
    assert result["ambiguous_submissions"] and not result["auxiliary_jobs"] and not result["jobs"]


def test_recovery_cannot_adopt_duplicate_worker_or_exceed_eight(ledger, monkeypatch):
    root, value = ledger
    value["ambiguous_submissions"][0].update(mode="worker", worker=0, gpus=1)
    value["jobs"] = [{"job_id": SECOND, "worker": 0}]
    submit.save(root / "campaign.json", value)
    monkeypatch.setattr(collector, "remote_cli", lambda *args: pytest.fail("Duplicate worker reached scheduler"))
    with pytest.raises(AssertionError): submit.recover()
    assert json.loads((root / "campaign.json").read_text())["ambiguous_submissions"]


@pytest.mark.parametrize("bad", [f"{JOB} Pending foreign metadata", f"{JOB} Pending", JOB])
def test_recovery_same_id_missing_note_metadata_is_not_silently_dropped(ledger, monkeypatch, bad):
    root, _ = ledger
    def remote(command, *args):
        return {"stdout": line() if command[-1] == "--active" else bad,
                "returncode": 0, "ssh_returncode": 0, "timed_out": False, "ssh_timed_out": False}
    monkeypatch.setattr(collector, "remote_cli", remote)
    with pytest.raises(RuntimeError, match="conflicts"):
        submit.recover()
    result = json.loads((root / "campaign.json").read_text())
    assert result["ambiguous_submissions"] and not result["auxiliary_jobs"]


@pytest.mark.parametrize("mode", ["preflight", "inspect", "worker"])
def test_submission_waits_for_240_second_create_without_changing_exact_command(ledger, monkeypatch, capsys, mode):
    root, value = ledger
    value.update(ambiguous_submissions=[], source_revision="reviewed-dev-revision",
                 review_passed=True, assets_ready_verified=True,
                 runtime_bundle_sha256="a" * 64, gpu_limit=10)
    submit.save(root / "campaign.json", value)
    dummy_key = "test-only-wandb-key"
    monkeypatch.setattr(submit.netrc, "netrc", lambda: SimpleNamespace(
        authenticators=lambda host: ("api", None, dummy_key)))
    monkeypatch.setattr(sys, "argv", ["cloud_repair_submit.py", mode] + (
        ["--worker-id", "0"] if mode == "worker" else []))
    calls = []
    def run(command, **kwargs):
        calls.append(command)
        assert kwargs["timeout"] == 305
        tree = ast.parse(kwargs["input"])
        remote_command = ast.literal_eval(next(node.value for node in tree.body
            if isinstance(node, ast.Assign) and any(isinstance(target, ast.Name)
                and target.id == "command" for target in node.targets)))
        run_calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)
                     and isinstance(node.func, ast.Attribute) and node.func.attr == "run"]
        assert [next(k.value.value for k in node.keywords if k.arg == "timeout") for node in run_calls] == [50, 240]
        assert remote_command[:2] == ["mlsub", "run"]
        expected = {"--repo": "https://github.com/dimativator/LoRA-TSD", "--branch": "dev",
                    "--entry": "scripts/cloud/cloud_repair_entry.py", "--image": "torch28",
                    "--gpus": "1" if mode == "worker" else "cpu",
                    "--args": "worker --worker-id 0" if mode == "worker" else mode}
        for flag, setting in expected.items():
            assert remote_command[remote_command.index(flag) + 1] == setting
        assert "--no-pip" in remote_command
        pending = json.loads((root / "campaign.json").read_text())["ambiguous_submissions"]
        assert len(pending) == 1 and dummy_key not in json.dumps(pending)
        output = JOB + (" " + dummy_key if mode in ("inspect", "worker") else "")
        return SimpleNamespace(returncode=0, stdout=output, stderr="")
    monkeypatch.setattr(submit.subprocess, "run", run)
    submit.main()
    current = json.loads((root / "campaign.json").read_text())
    assert len(calls) == 1 and not current["ambiguous_submissions"]
    assert len(current["jobs"] if mode == "worker" else current["auxiliary_jobs"]) == 1
    assert dummy_key not in capsys.readouterr().out
    assert all(dummy_key not in path.read_text() for path in root.iterdir() if path.is_file())


def test_extended_submission_timeout_preserves_intent_and_blocks_duplicate(ledger, monkeypatch):
    root, value = ledger
    value.update(ambiguous_submissions=[], source_revision="reviewed-dev-revision")
    submit.save(root / "campaign.json", value)
    monkeypatch.setattr(sys, "argv", ["cloud_repair_submit.py", "preflight"])
    def timeout(command, **kwargs):
        assert kwargs["timeout"] == 305
        assert "timeout=50" in kwargs["input"] and "timeout=240" in kwargs["input"]
        raise submit.subprocess.TimeoutExpired(command, 305)
    monkeypatch.setattr(submit.subprocess, "run", timeout)
    with pytest.raises(RuntimeError, match="response lost"):
        submit.main()
    current = json.loads((root / "campaign.json").read_text())
    assert len(current["ambiguous_submissions"]) == 1 and not current["jobs"] and not current["auxiliary_jobs"]
    monkeypatch.setattr(submit.subprocess, "run", lambda *a, **k: pytest.fail("Duplicate submission attempted"))
    with pytest.raises(RuntimeError, match="pending submission"):
        submit.main()


def test_defer_only_cpu_observation_preserves_record_and_limits_one(ledger):
    _, value = ledger
    original = dict(value["ambiguous_submissions"][0])
    submit.defer_inspection(value)
    assert not value["ambiguous_submissions"]
    retained = value["unresolved_auxiliary_submissions"][0]
    assert all(retained[k] == v for k, v in original.items())
    assert retained["deferred_at"] and retained["reason"]
    value["ambiguous_submissions"] = [original]
    with pytest.raises(RuntimeError):
        submit.defer_inspection(value)


@pytest.mark.parametrize("change", [{"mode":"worker","worker":0,"gpus":1}, {"mode":"prepare"}, {"mode":"cleanup"}, {"mode":"reconcile"}, {"gpus":1}, {"gpus":False}, {"worker":0}])
def test_defer_never_waives_gpu_or_mutating_ambiguity(ledger, change):
    _, value = ledger
    value["ambiguous_submissions"][0].update(change)
    with pytest.raises(RuntimeError):
        submit.defer_inspection(value)
    assert value["ambiguous_submissions"] and not value.get("unresolved_auxiliary_submissions")


def test_deferred_cpu_receipt_recovery_retains_main_pending(ledger, monkeypatch):
    root, value = ledger
    submit.defer_inspection(value)
    current = {**value["unresolved_auxiliary_submissions"][0], "note":"new_inspection"}
    value["ambiguous_submissions"] = [current]
    submit.save(root / "campaign.json", value)
    monkeypatch.setattr(collector, "remote_cli", lambda *a: {"stdout":line(), "returncode":0,"ssh_returncode":0,"timed_out":False})
    submit.recover(auxiliary=True)
    result = json.loads((root/"campaign.json").read_text())
    assert result["ambiguous_submissions"] == [current]
    assert not result["unresolved_auxiliary_submissions"]
    assert result["auxiliary_jobs"][0]["job_id"] == JOB


@pytest.mark.parametrize("mode", ["worker", "preflight"])
def test_deferred_gpu_metadata_cannot_bypass_allocation_guard(ledger, monkeypatch, mode):
    root, value = ledger
    value.update(ambiguous_submissions=[], unresolved_auxiliary_submissions=[{"mode":"worker","gpus":1,"worker":0}])
    submit.save(root / "campaign.json", value)
    monkeypatch.setattr(sys, "argv", ["submit", mode] + (["--worker-id","0"] if mode == "worker" else []))
    monkeypatch.setattr(submit.subprocess, "run", lambda *a, **k: pytest.fail("Unproven GPU allocation reached submit"))
    with pytest.raises(RuntimeError, match="proven CPU-only"):
        submit.main()


def test_deferred_cpu_does_not_waive_fresh_ready_or_cleanup(ledger, monkeypatch):
    root, value = ledger
    submit.defer_inspection(value)
    value.update(review_passed=True, assets_ready_verified=False)
    submit.save(root / "campaign.json", value)
    monkeypatch.setattr(submit.subprocess, "run", lambda *a, **k: pytest.fail("Readiness/cleanup guard bypassed"))
    monkeypatch.setattr(sys, "argv", ["submit","worker","--worker-id","0"])
    with pytest.raises(AssertionError): submit.main()
    monkeypatch.setattr(sys, "argv", ["submit","cleanup"])
    with pytest.raises(AssertionError, match="audited before cleanup"): submit.main()


def test_refresh_is_one_distinct_cpu_request_with_old_intent_retained(ledger, monkeypatch):
    root, value = ledger
    value["source_revision"] = "reviewed-dev"
    submit.save(root / "campaign.json", value)
    monkeypatch.setattr(submit.netrc, "netrc", lambda: SimpleNamespace(authenticators=lambda host:("api",None,"testkey")))
    monkeypatch.setattr(sys, "argv", ["submit", "refresh-inspection"])
    def run(command, **kwargs):
        tree = ast.parse(kwargs["input"])
        remote = ast.literal_eval(next(n.value for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=="command" for t in n.targets)))
        assert remote[remote.index("--gpus")+1] == "cpu"
        assert remote[remote.index("--args")+1] == "inspect"
        current = json.loads((root/"campaign.json").read_text())
        assert current["unresolved_auxiliary_submissions"][0]["note"] == NOTE
        assert len(current["ambiguous_submissions"]) == 1
        assert current["ambiguous_submissions"][0]["note"] != NOTE
        return SimpleNamespace(returncode=0,stdout=SECOND,stderr="")
    monkeypatch.setattr(submit.subprocess,"run",run)
    submit.main()
    current=json.loads((root/"campaign.json").read_text())
    assert not current["ambiguous_submissions"] and not current["jobs"]
    assert current["unresolved_auxiliary_submissions"][0]["note"] == NOTE
    assert current["auxiliary_jobs"][0]["job_id"] == SECOND


def gpu_pending(worker):
    return {"intent": f"worker-{worker}", "note": f"repair8b_worker-{worker}_unique",
            "mode": "worker", "worker": worker, "gpus": 1}


def ready(value):
    value.update(source_revision="reviewed-dev", review_passed=True, assets_ready_verified=True,
                 runtime_bundle_sha256="a" * 64, gpu_limit=10)


def mock_create(monkeypatch, root, mode, worker=None, job=SECOND, timeout=False):
    monkeypatch.setattr(sys, "argv", ["submit", mode] + (["--worker-id", str(worker)] if worker is not None else []))
    monkeypatch.setattr(submit.netrc, "netrc", lambda: SimpleNamespace(
        authenticators=lambda host: ("api", None, "test-key-not-real")))
    calls = []
    def create(*args, **kwargs):
        calls.append(1)
        current = json.loads((root / "campaign.json").read_text())
        assert len({r["note"] for r in current["ambiguous_submissions"]}) == len(current["ambiguous_submissions"])
        if timeout:
            raise submit.subprocess.TimeoutExpired("ssh", 305)
        return SimpleNamespace(returncode=0, stdout=job, stderr="")
    monkeypatch.setattr(submit.subprocess, "run", create)
    return calls


def test_first_distinct_worker_is_allowed_while_unknown_worker_still_reserves_slot(ledger, monkeypatch):
    root, value = ledger
    ready(value)
    original = gpu_pending(0)
    value["ambiguous_submissions"] = [original]
    submit.save(root / "campaign.json", value)
    calls = mock_create(monkeypatch, root, "worker", 7)
    submit.main()
    current = json.loads((root / "campaign.json").read_text())
    assert len(calls) == 1 and current["ambiguous_submissions"] == [original]
    assert current["jobs"][0]["worker"] == 7 and current["reserved_gpu_slots"] == 2


def test_same_pending_worker_can_never_be_created_twice(ledger, monkeypatch):
    root, value = ledger
    ready(value)
    value["ambiguous_submissions"] = [gpu_pending(0)]
    submit.save(root / "campaign.json", value)
    calls = mock_create(monkeypatch, root, "worker", 0)
    with pytest.raises(AssertionError): submit.main()
    assert not calls


def test_all_eight_received_or_unknown_intents_consume_first_create_budget(ledger, monkeypatch):
    root, value = ledger
    ready(value)
    value["jobs"] = [{**gpu_pending(worker), "job_id": f"lm-mpi-job-0000000{worker}"} for worker in range(4)]
    value["ambiguous_submissions"] = [gpu_pending(worker) for worker in range(4, 8)]
    assert submit.reserved_gpu_slots(value) == 8
    submit.save(root / "campaign.json", value)
    calls = mock_create(monkeypatch, root, "worker", 7)
    with pytest.raises(AssertionError): submit.main()
    assert not calls


def test_one_normal_cpu_inspection_appends_without_erasing_gpu_ambiguity(ledger, monkeypatch):
    root, value = ledger
    ready(value)
    original = gpu_pending(0)
    value["ambiguous_submissions"] = [original]
    submit.save(root / "campaign.json", value)
    calls = mock_create(monkeypatch, root, "inspect", timeout=True)
    with pytest.raises(RuntimeError, match="response lost"): submit.main()
    current = json.loads((root / "campaign.json").read_text())
    assert len(current["ambiguous_submissions"]) == 2 and original in current["ambiguous_submissions"]
    assert sum(submit.cpu_inspection(row) for row in current["ambiguous_submissions"]) == 1
    assert current["reserved_gpu_slots"] == 1
    with pytest.raises(RuntimeError, match="pending submission"): submit.main()
    assert len(calls) == 1
    mock_create(monkeypatch, root, "worker", 7)
    with pytest.raises(RuntimeError, match="pending submission"): submit.main()


def test_successful_cpu_adoption_only_removes_its_own_pending_note(ledger, monkeypatch):
    root, value = ledger
    ready(value)
    original = gpu_pending(0)
    value["ambiguous_submissions"] = [original]
    submit.save(root / "campaign.json", value)
    mock_create(monkeypatch, root, "inspect")
    submit.main()
    current = json.loads((root / "campaign.json").read_text())
    assert current["ambiguous_submissions"] == [original]
    assert len(current["auxiliary_jobs"]) == 1 and current["reserved_gpu_slots"] == 1


def test_multiple_missing_gpu_intents_survive_next_distinct_timeout(ledger, monkeypatch):
    root, value = ledger
    ready(value)
    originals = [gpu_pending(0), gpu_pending(7)]
    value["ambiguous_submissions"] = originals
    submit.save(root / "campaign.json", value)
    mock_create(monkeypatch, root, "worker", 1, timeout=True)
    with pytest.raises(RuntimeError, match="response lost"): submit.main()
    current = json.loads((root / "campaign.json").read_text())
    assert all(row in current["ambiguous_submissions"] for row in originals)
    assert {row["worker"] for row in current["ambiguous_submissions"]} == {0, 7, 1}
    assert current["reserved_gpu_slots"] == 3


def test_multi_recovery_adopts_only_positive_notes_and_keeps_missing_others(ledger, monkeypatch):
    root, value = ledger
    pending = [gpu_pending(0), gpu_pending(7), gpu_pending(1)]
    value["ambiguous_submissions"] = pending
    submit.save(root / "campaign.json", value)
    calls = []
    def query(command, *args):
        calls.append(command)
        raw = line(job=JOB, note=pending[1]["note"])
        return {"stdout": raw, "returncode": 0, "ssh_returncode": 0, "timed_out": False, "ssh_timed_out": False}
    monkeypatch.setattr(collector, "remote_cli", query)
    submit.recover()
    current = json.loads((root / "campaign.json").read_text())
    assert current["ambiguous_submissions"] == [pending[0], pending[2]]
    assert current["jobs"][0]["worker"] == 7 and current["reserved_gpu_slots"] == 3
    assert calls == [["mlsub", "list"], ["mlsub", "list", "--active"]]


def test_observed_multiple_job_ids_persist_conflict_and_block_new_gpu(ledger, monkeypatch):
    root, value = ledger
    ready(value)
    value["ambiguous_submissions"] = [gpu_pending(0)]
    submit.save(root / "campaign.json", value)
    def query(command, *args):
        raw = line(job=SECOND if command[-1] == "--active" else JOB,
                   note=value["ambiguous_submissions"][0]["note"])
        return {"stdout": raw, "returncode": 0, "ssh_returncode": 0, "timed_out": False, "ssh_timed_out": False}
    monkeypatch.setattr(collector, "remote_cli", query)
    with pytest.raises(RuntimeError, match="Multiple job IDs"): submit.recover()
    current = json.loads((root / "campaign.json").read_text())
    intent = current["ambiguous_submissions"][0]
    assert intent["observed_job_ids"] == sorted([JOB, SECOND]) and intent["recovery_conflict"]
    assert current["reserved_gpu_slots"] == 2 and not current["jobs"]
    calls = mock_create(monkeypatch, root, "worker", 7)
    with pytest.raises(RuntimeError, match="conflicts"): submit.main()
    assert not calls
    # A later missing row never erases the already observed second allocation.
    monkeypatch.setattr(collector, "remote_cli", lambda *a: {
        "stdout": line(note=intent["note"]), "returncode": 0, "ssh_returncode": 0,
        "timed_out": False, "ssh_timed_out": False})
    with pytest.raises(RuntimeError, match="Multiple job IDs"): submit.recover()
    assert len(json.loads((root / "campaign.json").read_text())["ambiguous_submissions"][0]["observed_job_ids"]) == 2


def test_multiple_receipt_ids_persist_before_error_and_stop_more_gpu_creates(ledger, monkeypatch):
    root, value = ledger
    ready(value)
    value["ambiguous_submissions"] = []
    submit.save(root / "campaign.json", value)
    mock_create(monkeypatch, root, "worker", 0, job=JOB + " " + SECOND)
    with pytest.raises(RuntimeError, match="unique submission ID"): submit.main()
    current = json.loads((root / "campaign.json").read_text())
    assert current["ambiguous_submissions"][0]["observed_job_ids"] == sorted([JOB, SECOND])
    assert current["ambiguous_submissions"][0]["recovery_conflict"] and current["reserved_gpu_slots"] == 2
    calls = mock_create(monkeypatch, root, "worker", 7)
    with pytest.raises(RuntimeError, match="conflicts"): submit.main()
    assert not calls


def test_receipt_cannot_append_an_already_received_job_id(ledger, monkeypatch):
    root, value = ledger
    ready(value)
    value["jobs"] = [{**gpu_pending(0), "job_id": JOB}]
    value["ambiguous_submissions"] = []
    submit.save(root / "campaign.json", value)
    mock_create(monkeypatch, root, "worker", 7, job=JOB)
    with pytest.raises(RuntimeError, match="Receipt identity conflicts"): submit.main()
    current = json.loads((root / "campaign.json").read_text())
    assert len(current["jobs"]) == 1 and current["jobs"][0]["worker"] == 0
    assert current["ambiguous_submissions"][0]["worker"] == 7
    assert current["ambiguous_submissions"][0]["recovery_conflict"]


@pytest.mark.parametrize("multiple", [False, True])
def test_partial_timeout_ids_are_recorded_and_redacted_without_adoption(ledger, monkeypatch, multiple):
    root, value = ledger
    ready(value)
    value["ambiguous_submissions"] = []
    submit.save(root / "campaign.json", value)
    mock_create(monkeypatch, root, "worker", 0)
    def timeout(*args, **kwargs):
        raw = (JOB + (" " + SECOND if multiple else "") + " test-key-not-real").encode()
        raise submit.subprocess.TimeoutExpired("ssh", 305, output=raw, stderr=b"partial transport")
    monkeypatch.setattr(submit.subprocess, "run", timeout)
    with pytest.raises(RuntimeError, match="response lost"): submit.main()
    current = json.loads((root / "campaign.json").read_text())
    pending = current["ambiguous_submissions"][0]
    assert pending["partial_receipt"] and not current["jobs"]
    assert len(pending["observed_job_ids"]) == (2 if multiple else 1)
    assert current["reserved_gpu_slots"] == (2 if multiple else 1)
    assert bool(pending.get("recovery_conflict")) is multiple
    assert "test-key-not-real" not in (root / "submission-worker-0.log").read_text()
