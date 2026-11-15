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
def test_submission_waits_for_900_second_create_without_changing_exact_command(ledger, monkeypatch, capsys, mode):
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
        assert kwargs["timeout"] == 965
        tree = ast.parse(kwargs["input"])
        remote_command = ast.literal_eval(next(node.value for node in tree.body
            if isinstance(node, ast.Assign) and any(isinstance(target, ast.Name)
                and target.id == "command" for target in node.targets)))
        run_calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)
                     and isinstance(node.func, ast.Attribute) and node.func.attr == "run"]
        assert [next(k.value.value for k in node.keywords if k.arg == "timeout") for node in run_calls] == [50, 900]
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
        assert kwargs["timeout"] == 965
        assert "timeout=50" in kwargs["input"] and "timeout=900" in kwargs["input"]
        raise submit.subprocess.TimeoutExpired(command, 965)
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
            raise submit.subprocess.TimeoutExpired("ssh", 965)
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
        raise submit.subprocess.TimeoutExpired("ssh", 965, output=raw, stderr=b"partial transport")
    monkeypatch.setattr(submit.subprocess, "run", timeout)
    with pytest.raises(RuntimeError, match="response lost"): submit.main()
    current = json.loads((root / "campaign.json").read_text())
    pending = current["ambiguous_submissions"][0]
    assert pending["partial_receipt"] and not current["jobs"]
    assert len(pending["observed_job_ids"]) == (2 if multiple else 1)
    assert current["reserved_gpu_slots"] == (2 if multiple else 1)
    assert bool(pending.get("recovery_conflict")) is multiple
    assert "test-key-not-real" not in (root / "submission-worker-0.log").read_text()


def test_worker_verbose_only_changes_logger_flag_and_discards_all_raw_stderr(monkeypatch, capsys):
    command = ['mlsub', 'run', '--gpus', '1', '--env', 'WANDB_API_KEY=known-secret']
    calls = []
    arbitrary = 'arbitrary-admin-credential-not-in-secrets'
    def run(actual, **kwargs):
        calls.append(actual)
        assert kwargs == dict(capture_output=True, text=True, timeout=900)
        return SimpleNamespace(returncode=0, stdout=JOB + ' known-secret', stderr=(
            'Authorization: Bearer ' + arbitrary + '\n'
            'HTTP Request: POST https://api.example/jobs?token=' + arbitrary + ' "HTTP/1.1 503 Service Unavailable"\n'
            'https://api.example:443 "GET /jobs/list?key=' + arbitrary + ' HTTP/1.1" 200 12\n'
            'secret payload ' + arbitrary))
    monkeypatch.setattr(submit.subprocess, 'run', run)
    with pytest.raises(SystemExit) as error:
        submit.worker_submission(command, ['known-secret'])
    assert error.value.code == 0
    assert calls == [['mlsub', '-v'] + command[1:]]
    assert command == ['mlsub', 'run', '--gpus', '1', '--env', 'WANDB_API_KEY=known-secret']
    output = capsys.readouterr().out
    assert arbitrary not in output and 'known-secret' not in output
    metadata = json.loads(output.split('REPAIR_HTTP_DIAGNOSTICS ')[1])
    assert metadata == {'http_events': [{'method':'POST','path':'/jobs','status':503},
        {'method':'GET','path':'/jobs/list','status':200}], 'timed_out':False, 'returncode':0}
    assert submit.receipt_output(output).strip() == JOB + ' <redacted>'


def test_remote_worker_timeout_preserves_only_sanitized_metadata_and_partial_stdout(monkeypatch, capsys):
    arbitrary = 'unknown-admin-secret'
    def run(*args, **kwargs):
        raise submit.subprocess.TimeoutExpired(args[0], 900, output=(JOB+' known-key').encode(),
            stderr=('Authorization: '+arbitrary+'\nHTTP Request: POST https://host/jobs?key='+arbitrary+
                    ' "HTTP/1.1 500 Internal Server Error"').encode())
    monkeypatch.setattr(submit.subprocess, 'run', run)
    with pytest.raises(SystemExit) as error:
        submit.worker_submission(['mlsub','run'], ['known-key'])
    assert error.value.code == 124
    output = capsys.readouterr().out
    assert arbitrary not in output and 'known-key' not in output
    metadata = json.loads(output.split('REPAIR_HTTP_DIAGNOSTICS ')[1])
    assert metadata == {'http_events':[{'method':'POST','path':'/jobs','status':500}],
        'timed_out':True,'returncode':124}
    assert submit.receipt_output(output).strip() == JOB+' <redacted>'


def test_remote_timeout_never_mints_receipt_even_with_partial_job_id(ledger, monkeypatch):
    root, value = ledger
    ready(value)
    value['ambiguous_submissions'] = []
    submit.save(root/'campaign.json',value)
    mock_create(monkeypatch, root, 'worker',7)
    output = JOB+'\nREPAIR_HTTP_DIAGNOSTICS '+json.dumps({'http_events':[], 'timed_out':True, 'returncode':124})
    monkeypatch.setattr(submit.subprocess,'run',lambda *a,**k: SimpleNamespace(returncode=124,stdout=output,stderr=''))
    with pytest.raises(RuntimeError,match='response lost'):
        submit.main()
    current=json.loads((root/'campaign.json').read_text())
    assert not current['jobs']
    assert current['ambiguous_submissions'][0]['partial_receipt']
    assert current['ambiguous_submissions'][0]['observed_job_ids']==[JOB]


def test_http_metadata_job_path_is_not_submission_receipt(ledger):
    root,value=ledger
    ready(value)
    value['ambiguous_submissions']=[gpu_pending(7)]
    proof=root/'diagnostics.log'
    output='REPAIR_HTTP_DIAGNOSTICS '+json.dumps({'http_events':[{'method':'GET','path':'/jobs/'+JOB,'status':200}]})
    proof.write_text(output)
    assert not submit.remember_receipt(value,value['ambiguous_submissions'][0],output,proof)
    assert 'observed_job_ids' not in value['ambiguous_submissions'][0]


def test_distinct_network_creates_run_outside_lock_and_merge_without_lost_updates(ledger, monkeypatch):
    from concurrent.futures import ThreadPoolExecutor
    import fcntl
    import threading
    root, value = ledger
    ready(value)
    value['ambiguous_submissions'] = [gpu_pending(0)]
    submit.save(root/'campaign.json',value)
    monkeypatch.setattr(submit.netrc,'netrc',lambda: SimpleNamespace(
        authenticators=lambda host: ('api',None,'fake-key')))
    barrier=threading.Barrier(2)
    checked=threading.Event()
    def network(*args,**kwargs):
        assert kwargs['timeout']==965
        tree=ast.parse(kwargs['input'])
        command=ast.literal_eval(next(n.value for n in tree.body if isinstance(n,ast.Assign)
            and any(isinstance(t,ast.Name) and t.id=='command' for t in n.targets)))
        worker=int(command[command.index('--args')+1].split()[-1])
        barrier.wait(timeout=3)
        # Both calls are now in network transport; neither may retain the ledger lock.
        if worker==2:
            with (root/'submission.lock').open('a') as lock:
                fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
                fcntl.flock(lock, fcntl.LOCK_UN)
            checked.set()
        else:
            assert checked.wait(timeout=3)
        return SimpleNamespace(returncode=0,stdout=JOB if worker==2 else SECOND,stderr='')
    monkeypatch.setattr(submit.subprocess,'run',network)
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures=[executor.submit(submit.main,['worker','--worker-id',str(worker)]) for worker in (2,3)]
        for future in futures: future.result(timeout=5)
    current=json.loads((root/'campaign.json').read_text())
    assert {row['worker'] for row in current['jobs']}=={2,3}
    assert current['ambiguous_submissions']==[gpu_pending(0)]
    assert current['reserved_gpu_slots']==3


@pytest.mark.parametrize('late', ['same','none','timeout','different'])
def test_recovery_before_response_is_idempotent_or_preserves_late_conflict(ledger, monkeypatch, late):
    root,value=ledger
    ready(value)
    value['ambiguous_submissions']=[]
    submit.save(root/'campaign.json',value)
    monkeypatch.setattr(submit.netrc,'netrc',lambda:SimpleNamespace(
        authenticators=lambda host:('api',None,'fake-key')))
    def network(*args,**kwargs):
        # Simulate independent recover() adoption while the original SSH is in flight.
        with submit.locked(root/'submission.lock'):
            current=json.loads((root/'campaign.json').read_text())
            pending=current['ambiguous_submissions'].pop()
            current['jobs'].append({**pending,'job_id':JOB,'adopted_at':submit.now()})
            current['ambiguous_submissions'].append(gpu_pending(0))
            submit.save(root/'campaign.json',current)
        if late=='timeout': raise submit.subprocess.TimeoutExpired('ssh',965)
        return SimpleNamespace(returncode=0,stdout=JOB if late=='same' else SECOND if late=='different' else '',stderr='')
    monkeypatch.setattr(submit.subprocess,'run',network)
    if late=='different':
        with pytest.raises(RuntimeError,match='Late receipt identity conflicts'):
            submit.main(['worker','--worker-id','7'])
    else:
        submit.main(['worker','--worker-id','7'])
    current=json.loads((root/'campaign.json').read_text())
    assert len(current['jobs'])==1 and current['jobs'][0]['job_id']==JOB
    assert current['ambiguous_submissions']==[gpu_pending(0)]
    assert current['reserved_gpu_slots']==(3 if late=='different' else 2)
    row=current['jobs'][0]
    assert bool(row.get('transport_warning')) is (late in {'none','timeout'})
    if late=='different':
        assert row['recovery_conflict'] and set(row['observed_job_ids'])=={JOB,SECOND}
        monkeypatch.setattr(submit.subprocess,'run',lambda *a,**k:pytest.fail('Conflict permitted another create'))
        with pytest.raises(RuntimeError,match='allocation conflicts'):
            submit.main(['worker','--worker-id','2'])


def test_received_conflict_blocks_cleanup_before_network(ledger, monkeypatch):
    root,value=ledger
    ready(value)
    value['ambiguous_submissions']=[]
    value['jobs']=[{**gpu_pending(7),'job_id':JOB,'observed_job_ids':[JOB,SECOND],
                    'recovery_conflict':{'reason':'two actual IDs'}}]
    submit.save(root/'campaign.json',value)
    monkeypatch.setattr(submit.subprocess,'run',lambda *a,**k:pytest.fail('Conflict permitted cleanup'))
    with pytest.raises(AssertionError,match='Received allocation conflict'):
        submit.main(['cleanup'])


def replacement_ready(root,value,worker=0):
    from cloud_repair_common import affected_paths
    ready(value)
    original={**gpu_pending(worker),'note':submit.REPLACEABLE_NOTES[worker],'created_at':'original-time'}
    value['ambiguous_submissions']=[original]
    submit.save(root/'campaign.json',value)
    (root/f'submission-worker-{worker}.log').write_text('original timeout evidence')
    snapshot={'campaign':submit.CAMPAIGN,'timestamp':submit.now(),'collector_source_mode':'inspect',
        'filesystems':[{'requested_path':str(path),'available_bytes':6_000_000_000,'status':'ok'} for path in affected_paths()],
        'assets_ready':True,'preparation':{'phase':'ready'},'workers':[],
        'tasks':[{'worker':worker,'task':submit.TASKS[worker],'phase':'reserved'}],
        'owned_guards':[{'worker':worker,'phase':'reserved','status':'inactive','child_pid':None}]}
    submit.save(root/'inspection-latest.json',snapshot)
    return original,snapshot


@pytest.mark.parametrize('worker',[0,1])
def test_exact_authorized_unreceived_replacement_preserves_original_and_charges_both(ledger,monkeypatch,worker):
    root,value=ledger
    original,_=replacement_ready(root,value,worker)
    mock_create(monkeypatch,root,'worker',worker)
    submit.main(['worker','--worker-id',str(worker),'--replace-unreceived'])
    current=json.loads((root/'campaign.json').read_text())
    assert current['reserved_gpu_slots']==2 and not current['ambiguous_submissions']
    archived=current['superseded_unreceived_gpu_intents'][0]
    assert all(archived[k]==v for k,v in original.items())
    assert archived['original_still_reserves_gpu'] and Path(archived['submission_evidence']).read_text()=='original timeout evidence'
    assert submit.digest(Path(archived['submission_evidence']))==archived['submission_evidence_sha256']
    assert (root/f'submission-worker-{worker}.log').read_text()=='original timeout evidence'
    assert current['jobs'][0]['replacement_of']==original['note']
    assert current['jobs'][0]['user_override']['date']=='2026-11-15'
    monkeypatch.setattr(submit.subprocess,'run',lambda *a,**k:pytest.fail('Second replacement attempted'))
    with pytest.raises(AssertionError,match='already replaced once'):
        submit.main(['worker','--worker-id',str(worker),'--replace-unreceived'])


@pytest.mark.parametrize('damage',['otherworker','wrongnote','observed','conflict','replacement','stale','claimed','child','wandb','workerrecord','lowspace','missingguard','notready'])
def test_unreceived_replacement_rejects_scope_or_fresh_state_damage_before_network(ledger,monkeypatch,damage):
    from datetime import datetime,timedelta,timezone
    root,value=ledger
    _,snapshot=replacement_ready(root,value)
    if damage=='wrongnote':value['ambiguous_submissions'][0]['note']='another worker0 note'
    if damage=='observed':value['ambiguous_submissions'][0]['observed_job_ids']=[JOB]
    if damage=='conflict':value['ambiguous_submissions'][0]['recovery_conflict']={'reason':'known'}
    if damage=='replacement':value['ambiguous_submissions'][0]['replacement_of']='previous'
    if damage=='stale':snapshot['timestamp']=(datetime.now(timezone.utc)-timedelta(seconds=301)).isoformat()
    if damage=='claimed':snapshot['tasks'][0]['phase']='claimed'
    if damage=='child':snapshot['tasks'][0]['child_pid']=123
    if damage=='wandb':snapshot['tasks'][0]['wandb_id']='already-started'
    if damage=='workerrecord':snapshot['workers']=[{'worker':0,'pid':123}]
    if damage=='lowspace':snapshot['filesystems'][0]['available_bytes']=5_000_000_000
    if damage=='missingguard':snapshot['owned_guards']=[]
    if damage=='notready':value['assets_ready_verified']=False
    submit.save(root/'campaign.json',value)
    submit.save(root/'inspection-latest.json',snapshot)
    monkeypatch.setattr(submit.subprocess,'run',lambda *a,**k:pytest.fail('Unsafe replacement reached network'))
    with pytest.raises((AssertionError,ValueError)):
        submit.main(['worker','--worker-id','2' if damage=='otherworker' else '0','--replace-unreceived'])
    current=json.loads((root/'campaign.json').read_text())
    assert not current.get('superseded_unreceived_gpu_intents') and current['ambiguous_submissions']==value['ambiguous_submissions']


def test_two_replacements_keep_ten_slot_budget_and_block_early_cleanup(ledger,monkeypatch):
    root,value=ledger
    original,snapshot=replacement_ready(root,value)
    value['jobs']=[{**gpu_pending(worker),'job_id':f'lm-mpi-job-{worker:08x}-1234-1234-1234-123456789abc'} for worker in range(2,8)]
    other={**gpu_pending(1),'note':submit.REPLACEABLE_NOTES[1]}
    value['ambiguous_submissions'].append(other)
    (root/'submission-worker-1.log').write_text('second original timeout')
    snapshot['tasks'].append({'worker':1,'task':submit.TASKS[1],'phase':'reserved'})
    snapshot['owned_guards'].append({'worker':1,'phase':'reserved','status':'inactive','child_pid':None})
    submit.save(root/'campaign.json',value);submit.save(root/'inspection-latest.json',snapshot)
    mock_create(monkeypatch,root,'worker',0,job=JOB)
    submit.main(['worker','--worker-id','0','--replace-unreceived'])
    mock_create(monkeypatch,root,'worker',1,job=SECOND)
    submit.main(['worker','--worker-id','1','--replace-unreceived'])
    current=json.loads((root/'campaign.json').read_text())
    assert current['reserved_gpu_slots']==10 and len(current['jobs'])==8
    assert len(current['superseded_unreceived_gpu_intents'])==2
    monkeypatch.setattr(submit.subprocess,'run',lambda *a,**k:pytest.fail('Early cleanup attempted'))
    with pytest.raises(AssertionError,match='Superseded unanswered'):
        submit.main(['cleanup'])


def test_known_extra_allocation_blocks_replacement_at_cap(ledger,monkeypatch):
    root,value=ledger
    replacement_ready(root,value)
    value['jobs']=[{**gpu_pending(worker),'job_id':f'lm-mpi-job-{worker:08x}-1234-1234-1234-123456789abc',
        'observed_job_ids':[f'lm-mpi-job-{worker:08x}-1234-1234-1234-123456789abc', JOB,SECOND]} for worker in range(2,6)]
    submit.save(root/'campaign.json',value)
    monkeypatch.setattr(submit.subprocess,'run',lambda *a,**k:pytest.fail('Over-cap replacement attempted'))
    with pytest.raises(AssertionError,match='exceeds ten'):
        submit.main(['worker','--worker-id','0','--replace-unreceived'])
    assert not json.loads((root/'campaign.json').read_text()).get('superseded_unreceived_gpu_intents')
