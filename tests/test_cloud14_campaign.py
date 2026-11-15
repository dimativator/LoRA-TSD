"""CPU checks for ownership, logging, storage stops and the frozen argument protocol."""

import importlib.util
import json
from pathlib import Path
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

import pytest

CLOUD = Path(__file__).resolve().parents[1] / "scripts/cloud"
sys.path.insert(0, str(CLOUD))
import cloud14_core as core
import cloud14_runtime as runtime


def load_plan():
    return json.loads((CLOUD / "cloud14_plan.json").read_text())


def test_full_grid_and_parallel_claims(tmp_path):
    plan = load_plan()
    core.validate_plan(plan)
    core.reserve_for_generation_two(plan, tmp_path)
    with ThreadPoolExecutor(max_workers=14) as pool:
        claimed = list(pool.map(lambda worker: core.claim(plan, worker, tmp_path), range(138)))
    assert len({core.task_key(task) for task in claimed}) == 138
    assert core.claim(plan, 0, tmp_path) is None


def test_duplicate_plan_is_rejected():
    plan = load_plan()
    plan["queue"][-1] = plan["queue"][0]
    with pytest.raises(AssertionError):
        core.validate_plan(plan)


def test_frozen_source_and_all_arguments(tmp_path):
    frozen = runtime.unpack()
    hashes = json.loads((CLOUD / "cloud14_source_hashes.json").read_text())
    for path, expected in hashes.items():
        assert core.digest(frozen / path) == expected
    sys.path.insert(0, str(frozen / "src"))
    from args_parser import parse_config
    for task in load_plan()["queue"]:
        for batch in [1, 2, 4, 8, 16, 32]:
            config = parse_config(runtime.arguments(task, frozen, tmp_path, batch))
            assert config["train_batch_size"] * config["grad_accum_steps"] == 32
            assert config["steps"] == 3000 and config["eval_every"] == 250
            assert config["checkpoint_every"] == 0
            assert config["checkpoint_dir"] is None and config["resume_from"] is None


def test_same_step_rows_and_final_are_all_logged(tmp_path, monkeypatch):
    rows = [{"step": 250, "validation_score": .8}, {"step": 250, "train/loss": .3},
            {"step": 3000, "train/loss": .1}, {"step": 3000, "validation_score": .9}]
    (tmp_path / "history.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    class Finished:
        pid = 123
        def poll(self):
            return 0
    class FakeRun:
        def __init__(self):
            self.rows = []
        def log(self, value, **kwargs):
            assert "step" not in kwargs
            self.rows.append(value)
    monkeypatch.setattr(runtime, "require_space", lambda paths: [])
    run = FakeRun()
    result = runtime.watch(Finished(), tmp_path, {}, tmp_path / "status.json", run)
    assert result == rows
    assert run.rows == [core.wandb_row(row) for row in rows]


def test_partial_final_jsonl_line_is_read_later(tmp_path):
    path = tmp_path / "history.jsonl"
    path.write_text('{"step":1}\n{"step":2')
    rows, offset = runtime.read_rows(path, 0)
    assert rows == [{"step": 1}]
    with path.open("a") as stream:
        stream.write('}\n')
    assert runtime.read_rows(path, offset)[0] == [{"step": 2}]


def test_disk_guard_stops_only_owned_process(tmp_path, monkeypatch):
    calls = []
    def space(paths):
        calls.append(1)
        if len(calls) > 1:
            raise OSError("Simulated unavailable disk telemetry")
        return []
    monkeypatch.setattr(runtime, "require_space", space)
    other = subprocess.Popen([sys.executable, "-c", "import time;time.sleep(30)"])
    try:
        with pytest.raises(OSError):
            with runtime.owned_process([sys.executable, "-c", "import time;time.sleep(30)"], tmp_path, {}) as owned:
                owned.wait(timeout=10)
        assert owned.poll() is not None and other.poll() is None
    finally:
        other.terminate()
        other.wait(timeout=10)


def test_hung_disk_telemetry_is_bounded_without_second_wait(monkeypatch):
    class HungTelemetry:
        killed = False
        def communicate(self, *args, **kwargs):
            assert kwargs["timeout"] == 15
            raise subprocess.TimeoutExpired("disk", 15)
        def kill(self):
            self.killed = True
    process = HungTelemetry()
    monkeypatch.setattr(core.subprocess, "Popen", lambda *args, **kwargs: process)
    with pytest.raises(OSError, match="timed out"):
        core.disk_records([Path("/some/network/mount")])
    assert process.killed


def test_cleanup_refuses_missing_final_verification(tmp_path):
    root = tmp_path / "assets"
    root.mkdir()
    with pytest.raises(AssertionError):
        core.cleanup_owned_assets(root, tmp_path / "state", {"verified_runs": 179})
    assert root.exists()


def test_cleanup_removes_only_owned_assets_after_terminal_audit(tmp_path):
    root = tmp_path / "dimativator-paper8b-three-seeds-20261113/assets"
    root.mkdir(parents=True)
    state = tmp_path / "state"
    core.save(root / "ownership.json", {"campaign": core.CAMPAIGN, "root": str(root.resolve())})
    (root / "test-model").write_bytes(b"model")
    borrowed = tmp_path / "borrowed-model"
    borrowed.write_bytes(b"do not delete")
    for i in range(14):
        core.save(state / "workers" / f"{i}.json", {"phase": "complete"})
    for i in range(138):
        core.save(state / "tasks" / f"{i}.json", {"phase": "complete"})
    confirmation = {"verified_runs": 180, "wandb_sync_verified": True,
                    "results_export_verified": True, "all_owned_jobs_terminal": True}
    assert core.cleanup_owned_assets(root, state, confirmation) > 5
    assert not root.exists() and borrowed.read_bytes() == b"do not delete"


def test_final_evaluation_is_required():
    rows = [{"step": step, "train/loss": .1} for step in range(1, 3001)]
    rows += [{"step": step, "validation_score": .9} for step in range(250, 3000, 250)]
    with pytest.raises(AssertionError):
        core.verify_rows(rows, {}, {})


@pytest.mark.parametrize("rank", [0, 1])
def test_one_worker_per_cloud_mpi_rank(rank, monkeypatch):
    monkeypatch.setenv("OMPI_COMM_WORLD_SIZE", "2")
    monkeypatch.setenv("OMPI_COMM_WORLD_LOCAL_RANK", str(rank))
    monkeypatch.setattr(runtime, "require_space", lambda paths: [])
    monkeypatch.setattr(runtime, "unpack", lambda: None)
    monkeypatch.setattr(runtime, "configured_environment", lambda: {})
    def query(command, **kwargs):
        if "--query-gpu=index,uuid,memory.total,memory.free" in command:
            return subprocess.CompletedProcess(command, 0, "0, GPU-first, 81559, 81110\n1, GPU-second, 81559, 81110\n", "")
        return subprocess.CompletedProcess(command, 0, "", "")
    monkeypatch.setattr(runtime.subprocess, "run", query)
    captured = []
    monkeypatch.setattr(runtime.os, "execve", lambda executable, args, env: captured.append((args, env)))
    runtime.pair(3)
    assert len(captured) == 1
    args, env = captured[0]
    assert args[-2:] == ["--worker-id", str(6 + rank)]
    assert env["CUDA_VISIBLE_DEVICES"] == ["GPU-first", "GPU-second"][rank]


def test_reset_unstarted_refuses_any_training_progress(tmp_path):
    core.save(tmp_path / "tasks" / "task.json", {"phase": "training", "step": 1})
    audit = {"campaign": core.CAMPAIGN, "new_wandb_runs": 0, "timestamp": core.now(),
             "jobs": [{"job_id": str(i), "status": "failed"} for i in range(7)]}
    with pytest.raises(AssertionError):
        core.reset_unstarted(tmp_path, audit, {"task"})
    assert (tmp_path / "tasks/task.json").exists()


def test_reset_unstarted_preserves_claim_evidence(tmp_path):
    core.save(tmp_path / "tasks" / "task.json", {"phase": "claimed", "step": 0})
    core.save(tmp_path / "workers" / "0.json", {"phase": "active"})
    audit = {"campaign": core.CAMPAIGN, "new_wandb_runs": 0, "timestamp": core.now(),
             "jobs": [{"job_id": str(i), "status": "failed"} for i in range(7)]}
    result = core.reset_unstarted(tmp_path, audit, {"task"})
    assert result["archived_tasks"] == 1 and result["archived_workers"] == 1
    assert not (tmp_path / "tasks/task.json").exists()
    assert (Path(result["archive"]) / "tasks/task.json").exists()


def test_audit_transport_respects_gateway_environment_limits(monkeypatch):
    import re
    audit = {"jobs": [{"job_id": "lm-mpi-job-" + str(i) * 36, "status": "failed"} for i in range(7)],
             "timestamp": core.now(), "campaign": core.CAMPAIGN}
    values = core.encode_audit_environment("PAPER8B_STARTUP_RECOVERY_AUDIT", audit)
    for key, value in values.items():
        assert re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]{0,63}", key)
        assert re.fullmatch(r"[A-Za-z0-9._:/=+-]{0,255}", value)
        monkeypatch.setenv(key, value)
    assert core.decode_audit_environment("PAPER8B_STARTUP_RECOVERY_AUDIT") == audit


def test_snapshot_survives_mpi_prefixes_and_rejects_missing_chunks():
    import random
    randomizer = random.Random(42)
    payload = {"data": "".join(chr(randomizer.randint(32, 126)) for _ in range(50000))}
    lines = core.snapshot_lines(payload)
    assert all(len(line) < 1700 for line in lines)
    assert len(lines) > 1
    text = "\n".join("2026-11-13T16:00:00Z [1,0]<stdout>:" + line for line in lines)
    assert core.decode_snapshot(text) == payload
    corrupted = "\n".join("2026-11-13T16:00:00Z [1,0]<stdout>:" + line[:111]
                            + "[1,0]<stdout>:" + line[111:] for line in lines)
    assert core.decode_snapshot(corrupted) == payload
    with pytest.raises(AssertionError):
        core.decode_snapshot("\n".join(lines[:-1]))


def test_generation_reservation_preserves_active_failed_and_complete(tmp_path):
    plan = load_plan()
    records = []
    for task, phase in zip(plan['queue'][:3], ['training', 'failed', 'complete']):
        record = {'task': task, 'phase': phase, 'step': 123, 'wandb_id': 'preserved'}
        path = tmp_path / 'tasks' / (core.task_key(task) + '.json')
        core.save(path, record)
        records.append((path, path.read_bytes()))
    result = core.reserve_for_generation_two(plan, tmp_path)
    assert result['count'] == 135
    assert all(path.read_bytes() == before for path, before in records)
    # Generation-one launchers skip every existing task file.
    assert all((tmp_path / 'tasks' / (core.task_key(task) + '.json')).exists() for task in plan['queue'])
    with ThreadPoolExecutor(max_workers=14) as pool:
        tasks = list(pool.map(lambda worker: core.claim(plan, worker, tmp_path), range(135)))
    assert len({core.task_key(task) for task in tasks}) == 135
    assert core.claim(plan, 0, tmp_path) is None
    assert all(path.read_bytes() == before for path, before in records)


def test_retry_refuses_live_allocation_and_preserves_failed_attempt(tmp_path):
    plan = load_plan()
    task = plan['queue'][0]
    key = core.task_key(task)
    path = tmp_path / 'tasks' / (key + '.json')
    record = {'task': task, 'phase': 'failed', 'returncode': 1, 'step': 0,
              'worker': 5, 'hostname': 'owned-job-mpimaster-0', 'wandb_id': 'failed-id'}
    core.save(path, record)
    audit = {'campaign': core.CAMPAIGN, 'timestamp': core.now(), 'key': key,
             'diagnosis': 'RSVD initialization OOM; padded probe repair', 'job_status': 'Running',
             'job_id': 'owned-job', 'workers': [4, 5], 'wandb_id': 'failed-id', 'last_step': 0,
             'raw_partial_history_preserved': True}
    with pytest.raises(AssertionError):
        core.reserve_failed_attempt(plan, tmp_path, audit)
    assert json.loads(path.read_text()) == record
    audit['job_status'] = 'Completed'
    result = core.reserve_failed_attempt(plan, tmp_path, audit)
    assert json.loads(Path(result['archive']).read_text())['status'] == record
    assert json.loads(path.read_text())['phase'] == 'reserved_v2'
    assert core.claim(plan, 0, tmp_path) == task
    with pytest.raises(AssertionError):
        core.reserve_failed_attempt(plan, tmp_path, audit)


def test_diagnostic_upload_snapshot_excludes_wandb_and_symlinks(tmp_path):
    (tmp_path / 'process.log').write_text('diagnostic')
    (tmp_path / 'wandb/files/wandb').mkdir(parents=True)
    (tmp_path / 'wandb/files/debug.log').write_text('must not upload recursively')
    (tmp_path / 'foreign.log').symlink_to(tmp_path / 'process.log')
    assert runtime.diagnostic_files(tmp_path) == [tmp_path / 'process.log']


def test_padded_profile_matches_collator_masks_and_ignores_padding_labels():
    import torch
    from cloud14_profile import synthetic_batch
    for microbatch in [1, 2, 32]:
        batch = synthetic_batch(torch, microbatch, 0, 'cpu')
        mask = batch['attention_mask']
        assert mask.shape == (microbatch, 1024)
        assert (mask == 0).any()
        if microbatch > 1:
            assert mask[0].sum().item() == 1024
        assert (mask[:, 1:] <= mask[:, :-1]).all()
        assert (batch['labels'][mask == 0] == -100).all()
        assert (batch['labels'] != -100).sum().item() == 8 * microbatch


def test_generation_two_claim_never_takes_missing_files(tmp_path):
    assert core.claim(load_plan(), 0, tmp_path) is None


def test_rollover_archive_rejects_live_pair_and_keeps_durable_pending_intent(tmp_path, monkeypatch):
    import cloud14_rollover as rollover
    monkeypatch.setattr(rollover, 'ARTIFACTS', tmp_path)
    pair = {'pair_id': 2, 'job_id': 'owned-old', 'gpus': 2}
    core.save(tmp_path / 'campaign.json', {'jobs': [pair], 'allocation_history': []})
    job = {'job_id': 'owned-old', 'status': 'Running', 'audited_at': core.now()}
    with pytest.raises(AssertionError):
        rollover.archive_terminal_pair(pair, job)
    assert json.loads((tmp_path / 'campaign.json').read_text())['jobs'] == [pair]
    job['status'] = 'Failed'
    rollover.archive_terminal_pair(pair, job)
    result = json.loads((tmp_path / 'campaign.json').read_text())
    assert not result['jobs']
    assert result['pending_pair_replacements'] == {'2': 'owned-old'}
    assert result['allocation_history'][0]['jobs'][0]['terminal_audit'] == job


def test_collector_accepts_complete_crc_payload_despite_cli_exit_one(tmp_path, monkeypatch):
    import cloud14_collect as collect
    monkeypatch.setattr(collect, 'ARTIFACTS', tmp_path)
    campaign = {'jobs': [{'job_id': 'must-preserve'}], 'preparation_jobs': []}
    core.save(tmp_path / 'campaign.json', campaign)
    payload = {'timestamp': core.now(), 'assets_ready': True, 'filesystems': [], 'tasks': [], 'workers': []}
    text = '\n'.join(reversed(core.snapshot_lines(payload)))
    monkeypatch.setattr(collect.subprocess, 'run', lambda *a, **kw: subprocess.CompletedProcess(a, 1, text, ''))
    monkeypatch.setattr(sys, 'argv', ['cloud14_collect.py', '--job-id', 'exact-cpu-job'])
    collect.main()
    result = json.loads((tmp_path / 'inspection-latest.json').read_text())
    assert result['collector_transport_returncode'] == 1
    assert json.loads((tmp_path / 'campaign.json').read_text())['jobs'] == campaign['jobs']
    monkeypatch.setattr(collect.subprocess, 'run', lambda *a, **kw: subprocess.CompletedProcess(a, 1, 'no payload', ''))
    with pytest.raises(AssertionError, match='Incomplete'):
        collect.main()


def test_large_audit_environment_round_trip_uses_bounded_gateway_safe_chunks(monkeypatch):
    audit = {'campaign': core.CAMPAIGN, 'reviewed_authority': 'x' * 34000}
    values = core.encode_audit_environment('REPAIR_FINAL_AUDIT', audit)
    assert 32 < int(values['REPAIR_FINAL_AUDIT_COUNT']) <= 256
    assert all(len(value) <= 200 for key, value in values.items() if not key.endswith('_COUNT'))
    assert sum(len(value) for key, value in values.items() if not key.endswith('_COUNT')) <= 51200
    for key, value in values.items(): monkeypatch.setenv(key, value)
    assert core.decode_audit_environment('REPAIR_FINAL_AUDIT') == audit


def test_audit_environment_rejects_oversized_encode_before_any_environment_change(monkeypatch):
    monkeypatch.setenv('AUDIT_UNTOUCHED', 'retained')
    with pytest.raises(ValueError, match='bounded transport capacity'):
        core.encode_audit_environment('AUDIT', {'payload': 'x' * 38400})
    assert core.os.environ['AUDIT_UNTOUCHED'] == 'retained'


@pytest.mark.parametrize('count',['0','257','999','1000','-1','1.0',' 1','01',''])
def test_audit_environment_rejects_invalid_or_excessive_counts(monkeypatch,count):
    monkeypatch.setenv('AUDIT_COUNT',count)
    monkeypatch.setenv('AUDIT_0','e30=')
    with pytest.raises(ValueError):core.decode_audit_environment('AUDIT')


@pytest.mark.parametrize('raw',[b'[]',b'true',b'{"x":1,"x":2}',b'{"nested":{"x":1,"x":2}}',
    b'{"x":NaN}',b'{"x":Infinity}',b'{"x":-Infinity}',b'{"x":1e999}',b'{"x":-1e999}',b'{} trailing',b'\xff'])
def test_audit_environment_requires_strict_utf8_json_object(monkeypatch,raw):
    import base64
    encoded=base64.urlsafe_b64encode(raw).decode()
    monkeypatch.setenv('AUDIT_COUNT','1');monkeypatch.setenv('AUDIT_0',encoded)
    with pytest.raises(ValueError):core.decode_audit_environment('AUDIT')


@pytest.mark.parametrize('chunk',['','e30=garbage','*secret*','e30===','e30=\n','a'*201])
def test_audit_environment_rejects_malformed_or_oversized_chunks(monkeypatch,chunk):
    monkeypatch.setenv('AUDIT_COUNT','1');monkeypatch.setenv('AUDIT_0',chunk)
    with pytest.raises(ValueError):core.decode_audit_environment('AUDIT')


def test_audit_environment_requires_all_declared_chunks(monkeypatch):
    monkeypatch.setenv('AUDIT_COUNT','2');monkeypatch.setenv('AUDIT_0','e30=')
    monkeypatch.delenv('AUDIT_1',raising=False)
    with pytest.raises(KeyError):core.decode_audit_environment('AUDIT')


@pytest.mark.parametrize('audit',[[],True,{'x':float('nan')},{'x':float('inf')}])
def test_audit_environment_encoder_rejects_nonobject_or_nonfinite_json(audit):
    with pytest.raises(ValueError):core.encode_audit_environment('AUDIT',audit)
