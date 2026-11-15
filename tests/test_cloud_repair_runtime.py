"""CPU checks for one-task ownership, process guards and repair evidence preservation."""

from contextlib import contextmanager
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
from types import SimpleNamespace
import zipfile

import pytest

CLOUD = Path(__file__).resolve().parents[1] / "scripts/cloud"
sys.path.insert(0, str(CLOUD))
import cloud_repair_runtime as runtime
from cloud_repair_profile import synthetic_batch


@pytest.fixture
def frozen(tmp_path):
    target = tmp_path / "frozen"
    with zipfile.ZipFile(CLOUD / "cloud14_frozen_runtime.zip") as archive:
        archive.extractall(target)
    return target


def test_eight_task_protocol_keeps_all_48_cli_combinations(frozen, tmp_path):
    for task in runtime.TASKS:
        for batch in (1, 2, 4, 8, 16, 32):
            args = runtime.arguments(task, frozen, tmp_path, batch)
            assert args[args.index("--run-name") + 1] == runtime.run_name(task)
            assert args[args.index("--optimizer") + 1] == task["optimizer"]
            assert int(args[args.index("--train-batch-size") + 1]) * int(
                args[args.index("--grad-accum-steps") + 1]) == 32
            assert args[args.index("--steps") + 1] == "3000"
            assert args[args.index("--eval-every") + 1] == "250"
            assert "--checkpoint-dir" not in args and "--resume-from" not in args
            assert args[args.index("--checkpoint-every") + 1] == "0"


def test_task_outside_authorized_eight_is_rejected(frozen):
    with pytest.raises(ValueError, match="eight authorized"):
        runtime.materialize({**runtime.TASKS[0], "seed": 123}, frozen)


def test_duplicate_claim_fails_before_gpu_discovery(tmp_path, monkeypatch):
    monkeypatch.setattr(runtime, "STATE", tmp_path)
    monkeypatch.setattr(runtime, "require_space", lambda *args: [])
    path = tmp_path / "tasks" / (runtime.task_key(runtime.TASKS[0]) + ".json")
    path.parent.mkdir()
    path.write_text(json.dumps({"task": runtime.TASKS[0], "phase": "complete"}))
    monkeypatch.setattr(runtime, "single_gpu", lambda: pytest.fail("Duplicate reached GPU query"))
    with pytest.raises(ValueError, match="already claimed"):
        runtime.worker(0)


def test_duplicate_worker_lock_is_nonblocking(tmp_path, monkeypatch):
    monkeypatch.setattr(runtime, "STATE", tmp_path)
    monkeypatch.setattr(runtime, "require_space", lambda *args: [])
    (tmp_path / "workers").mkdir()
    with (tmp_path / "workers/0.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(BlockingIOError):
            runtime.worker(0)


def test_paired_mpi_is_rejected_before_query(monkeypatch):
    monkeypatch.setenv("OMPI_COMM_WORLD_SIZE", "2")
    monkeypatch.setattr(runtime.subprocess, "run", lambda *a, **k: pytest.fail("Reached GPU query"))
    with pytest.raises(ValueError, match="one MPI rank"):
        runtime.single_gpu()


def test_exactly_one_allocated_uuid_is_required(monkeypatch):
    for key in ("OMPI_COMM_WORLD_SIZE", "OMPI_COMM_WORLD_LOCAL_SIZE",
                "OMPI_COMM_WORLD_RANK", "OMPI_COMM_WORLD_LOCAL_RANK"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setattr(runtime.subprocess, "run", lambda *a, **k:
                        SimpleNamespace(stdout="GPU-123, 81920, 80000\n"))
    assert runtime.single_gpu()["cuda_visible_devices"] == "GPU-123"
    monkeypatch.setattr(runtime.subprocess, "run", lambda *a, **k:
                        SimpleNamespace(stdout="GPU-123, 81920, 80000\nGPU-456,81920,80000\n"))
    with pytest.raises(ValueError, match="exactly one allocated"):
        runtime.single_gpu()


def test_guard_stops_owned_process_only(tmp_path, monkeypatch):
    calls = []
    def space(paths):
        calls.append(1)
        if len(calls) > 1:
            raise OSError("Unavailable storage telemetry")
        return []
    monkeypatch.setattr(runtime, "require_space", space)
    other = subprocess.Popen([sys.executable, "-c", "import time;time.sleep(30)"])
    try:
        with pytest.raises(OSError, match="disk guard"):
            with runtime.owned_process([sys.executable, "-c", "import time;time.sleep(30)"],
                                       tmp_path, dict(os.environ)) as owned:
                owned.wait(timeout=10)
        assert owned.poll() is not None and other.poll() is None
    finally:
        other.terminate()
        other.wait(timeout=10)


def test_failed_pid_record_does_not_orphan_a_child(tmp_path, monkeypatch):
    monkeypatch.setattr(runtime, "require_space", lambda paths: [])
    original = subprocess.Popen
    children = []
    def create(*args, **kwargs):
        child = original(*args, **kwargs)
        children.append(child)
        return child
    monkeypatch.setattr(runtime.subprocess, "Popen", create)
    def unavailable(*args):
        raise OSError("Simulated failed PID record write")
    monkeypatch.setattr(runtime, "save", unavailable)
    with pytest.raises(OSError, match="PID record"):
        with runtime.owned_process([sys.executable, "-c", "import time;time.sleep(30)"],
                                   tmp_path, dict(os.environ)):
            pytest.fail("A failed owner record yielded a live child")
    assert len(children) == 1 and children[0].poll() is not None


def test_hung_guard_writer_cannot_block_owned_stop(tmp_path, monkeypatch):
    monkeypatch.setattr(runtime, "GUARD_INTERVAL", .02)
    monkeypatch.setattr(runtime, "WRITER_STALL_SECONDS", .03)
    monkeypatch.setattr(runtime, "require_space", lambda paths: [])
    entered, release = threading.Event(), threading.Event()
    def blocked_writer(*args):
        entered.set()
        release.wait(10)
    monkeypatch.setattr(runtime, "write_guard_event", blocked_writer)
    other = subprocess.Popen([sys.executable, "-c", "import time;time.sleep(30)"])
    try:
        with pytest.raises(OSError, match="could not record protection"):
            with runtime.owned_process([sys.executable, "-c", "import time;time.sleep(30)"],
                                       tmp_path, dict(os.environ)) as owned:
                assert entered.wait(timeout=2)
                owned.wait(timeout=3)
        assert owned.poll() is not None and other.poll() is None
    finally:
        release.set()
        other.terminate()
        other.wait(timeout=10)


def test_hung_pid_save_does_not_disable_watchdog(tmp_path, monkeypatch):
    monkeypatch.setattr(runtime, "GUARD_INTERVAL", .02)
    entered, release = threading.Event(), threading.Event()
    children, errors = [], []
    original = subprocess.Popen
    def create(*args, **kwargs):
        child = original(*args, **kwargs)
        children.append(child)
        return child
    def space(paths):
        if entered.is_set():
            raise OSError("Storage failed while PID save was blocked")
        return []
    def blocked_save(*args):
        entered.set()
        release.wait(10)
        raise OSError("PID save failed after unblock")
    monkeypatch.setattr(runtime, "require_space", space)
    monkeypatch.setattr(runtime.subprocess, "Popen", create)
    monkeypatch.setattr(runtime, "save", blocked_save)
    def owner():
        try:
            with runtime.owned_process([sys.executable, "-c", "import time;time.sleep(30)"],
                                       tmp_path, dict(os.environ)):
                pytest.fail("Blocked PID save must not yield")
        except OSError as error:
            errors.append(error)
    thread = threading.Thread(target=owner, daemon=True)
    thread.start()
    try:
        assert entered.wait(timeout=2)
        deadline = time.monotonic() + 3
        while children[0].poll() is None and time.monotonic() < deadline:
            time.sleep(.01)
        assert children[0].poll() is not None
        assert thread.is_alive(), "Watchdog stopped child while owner was still blocked"
    finally:
        release.set()
        thread.join(timeout=5)
        if children and children[0].poll() is None:
            runtime.stop_owned(children[0])
    assert not thread.is_alive() and errors


def test_partial_history_tail_is_preserved(tmp_path):
    path = tmp_path / "history.jsonl"
    path.write_text('{"step":1}\n{"step":2')
    rows, offset = runtime.read_rows(path, 0)
    assert rows == [{"step": 1}]
    with path.open("a") as stream:
        stream.write('}\n')
    assert runtime.read_rows(path, offset)[0] == [{"step": 2}]


def test_same_step_and_final_validation_rows_are_logged(tmp_path, monkeypatch):
    rows = [{"step": 250, "validation_score": .8}, {"step": 250, "train/loss": .3},
            {"step": 3000, "train/loss": .1}, {"step": 3000, "validation_score": .9}]
    (tmp_path / "history.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    monkeypatch.setattr(runtime, "require_space", lambda paths: [])
    logged = []
    run = SimpleNamespace(log=logged.append)
    process = SimpleNamespace(pid=123, poll=lambda: 0)
    assert runtime.watch(process, tmp_path, {}, tmp_path / "status.json", run) == rows
    assert logged == [runtime.wandb_row(r) for r in rows]


def test_flat_manifest_maps_unique_profile_evidence(tmp_path, monkeypatch):
    sha = tmp_path / "bundle.sha256"
    sha.write_text("known-bundle")
    monkeypatch.setattr(runtime, "BUNDLE_SHA", sha)
    for directory in (tmp_path, tmp_path / "cloud0-lora_rite-micro32",
                      tmp_path / "cloud0-lora_rite-micro16"):
        directory.mkdir(exist_ok=True)
        (directory / "process.log").write_text(directory.name)
    report = runtime.diagnostic_manifest(tmp_path, {"run_name": "repair", "wandb_id": "run123"})
    assert {r["wandb_name"] for r in report["files"]} == {
        "process.log", "profile-micro32-process.log", "profile-micro16-process.log"}
    for row in report["files"]:
        assert runtime.digest(tmp_path / row["path"]) == row["sha256"]


def test_diagnostic_symlink_is_rejected(tmp_path):
    (tmp_path / "external.json").symlink_to("/does-not-exist")
    with pytest.raises(ValueError, match="Unexpected link"):
        runtime.diagnostic_manifest(tmp_path, {})


def test_profile_has_real_padded_attention_and_eight_supervised_tokens():
    import torch
    for micro in (1, 16, 32):
        batch = synthetic_batch(torch, micro, 0, "cpu")
        assert batch["input_ids"].shape == (micro, 1024)
        assert (batch["attention_mask"] == 0).any()
        assert torch.all((batch["labels"] != -100).sum(1) == 8)
        assert torch.all(batch["labels"][batch["attention_mask"] == 0] == -100)


@pytest.mark.parametrize("returncode", [0, 1])
def test_raw_evidence_survives_success_and_failure(tmp_path, monkeypatch, frozen, returncode):
    monkeypatch.setattr(runtime, "STATE", tmp_path / "state")
    sha = tmp_path / "bundle.sha256"
    sha.write_text("known-bundle")
    monkeypatch.setattr(runtime, "BUNDLE_SHA", sha)
    monkeypatch.setattr(runtime, "require_space", lambda paths: [])
    monkeypatch.setattr(runtime, "choose_microbatch", lambda *a: 16)
    uploaded = []
    class Run:
        url = "https://wandb-radfan.ru/andrey/LoRA-Muon/runs/run123"
        config = {}
        summary = {}
        def define_metric(self, *args, **kwargs): pass
        def log(self, *args): pass
        def save(self, path, **kwargs): uploaded.append(Path(path).name)
        def finish(self, **kwargs): pass
    monkeypatch.setitem(sys.modules, "wandb", SimpleNamespace(
        init=lambda **k: Run(), util=SimpleNamespace(generate_id=lambda: "run123"),
        Settings=lambda **k: k))
    @contextmanager
    def process(command, directory, env):
        task = runtime.TASKS[0]
        rows = [{"step": i, "train/loss": .1} for i in range(1, 3001)]
        rows += [{"step": i, "validation_score": .8} for i in range(250, 3001, 250)]
        (directory / "history.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
        result = {"optimizer": task["optimizer"], "dataset": task["dataset"],
                  "seed": task["seed"], "model": "meta-llama/Meta-Llama-3.1-8B",
                  "steps": 3000, "reported_accuracy": .8, "reported_accuracy_step": 250}
        (directory / (runtime.run_name(task) + ".json")).write_text(json.dumps(result))
        (directory / "process.log").write_text("Training diagnostic")
        yield SimpleNamespace(pid=123, returncode=returncode, poll=lambda: 0)
    monkeypatch.setattr(runtime, "owned_process", process)
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    task = runtime.TASKS[0]
    status = {"phase": "claimed", "worker": 0}
    state = runtime.STATE / "tasks" / "task.json"
    if returncode:
        with pytest.raises(RuntimeError, match="Training exited"):
            runtime.run_task(task, 0, frozen, scratch, {}, status, state)
        assert status["phase"] == "failed" and not uploaded
    else:
        runtime.run_task(task, 0, frozen, scratch, {}, status, state)
        assert status["phase"] == "complete"
        assert status["syncphase"] == "uploaded_awaiting_independent_verification"
        assert "history.jsonl" in uploaded and "diagnostic-manifest.json" in uploaded
        assert status["recovery_manifest_sha256"]
    spool = Path(status["recovery_logs"])
    assert (spool / "history.jsonl").exists() and (spool / "process.log").exists()
    assert not list(spool.rglob("wandb"))
