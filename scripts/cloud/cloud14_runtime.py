"""Schedule independent GPU runs while leaving the frozen training source untouched."""

from contextlib import contextmanager
import io
import json
import math
import os
from pathlib import Path
import runpy
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
import zipfile

from cloud14_core import ASSETS, CAMPAIGN, STATE, claim, digest, disk_records, now, require_space, save, task_key, validate_plan, verify_rows, wandb_row
from cloud14_assets import PYTHON, affected_paths, configured_environment, prepare

HERE = Path(__file__).resolve().parent


def unpack() -> Path:
    bundle = HERE / "cloud14_frozen_runtime.zip"
    expected = (HERE / "cloud14_frozen_runtime.sha256").read_text().strip()
    if os.environ.get("PAPER8B_BUNDLE_SHA256"):
        assert expected == os.environ["PAPER8B_BUNDLE_SHA256"], "Submitted runtime hash differs"
    assert digest(bundle) == expected, "Frozen runtime bundle mismatch"
    target = Path(tempfile.gettempdir()) / ("paper8b-frozen-" + expected[:16])
    from cloud14_core import locked
    with locked(target.with_suffix(".lock")):
        if not (target / "ready").exists():
            target.mkdir(parents=True, exist_ok=True)
            with zipfile.ZipFile(io.BytesIO(bundle.read_bytes())) as archive:
                for info in archive.infolist():
                    assert (target / info.filename).resolve().is_relative_to(target.resolve())
                    assert not (info.external_attr >> 16) & 0o170000 == 0o120000
                archive.extractall(target)
            (target / "ready").write_text(expected)
    return target


def plan() -> dict:
    value = json.loads((HERE / "cloud14_plan.json").read_text())
    validate_plan(value)
    return value


def materialize(task: dict, frozen: Path, run_dir: Path) -> dict:
    spec = json.loads((frozen / "scripts/brain_lab/paper_8b_methods.json").read_text())
    method = next(method for method in spec["methods"] if method["id"] == task["method"])
    return {"method": method, "dataset": task["dataset"], "seed": task["seed"],
            "run_name": "paper8b_" + task_key(task)}


def arguments(task: dict, frozen: Path, run_dir: Path, microbatch: int) -> list[str]:
    sys.path.insert(0, str(frozen / "scripts/brain_lab"))
    import paper_8b_campaign
    args = paper_8b_campaign.arguments(materialize(task, frozen, run_dir), microbatch, run_dir)
    assert "--checkpoint-dir" not in args and "--resume-from" not in args
    assert args[args.index("--checkpoint-every") + 1] == "0"
    return args


def profile_main(args) -> None:
    frozen = unpack()
    sys.path.insert(0, str(frozen / "scripts/brain_lab"))
    import paper_8b_campaign
    permitted = {"paper8b_" + task_key(task) for task in plan()["queue"]}
    # The old queue's ownership map intentionally excludes these new Cloud workers.
    # Preserve the numerical profiler and replace only its pre-CUDA ownership predicate.
    def owns_task(worker: str, task: dict) -> bool:
        name = task["run_name"]
        if name not in permitted or not worker.startswith("cloud"):
            return False
        record = json.loads((STATE / "tasks" / (name.removeprefix("paper8b_") + ".json")).read_text())
        return (record["worker"] == int(worker.removeprefix("cloud"))
                and record["pid"] == os.getppid()
                and record["hostname"] == os.uname().nodename
                and record["phase"] == "profiling")
    paper_8b_campaign.task_is_enabled = owns_task
    sys.argv = ["paper_8b_profile.py", "--task", str(args.task), "--microbatch", str(args.microbatch)]
    runpy.run_path(str(frozen / "scripts/brain_lab/paper_8b_profile.py"), run_name="__main__")


@contextmanager
def owned_process(command: list[str], directory: Path, env: dict):
    require_space(affected_paths())
    log = (directory / "process.log").open("w")
    process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                               cwd=directory, env=env, start_new_session=True)
    stop_guard = threading.Event()
    guard_errors = []

    def guard() -> None:
        while not stop_guard.is_set() and process.poll() is None:
            try:
                records = require_space(affected_paths())
                with (directory / "disk-guard.jsonl").open("a") as stream:
                    stream.write(json.dumps({"timestamp": now(), "pid": process.pid,
                                             "filesystems": records}) + "\n")
            except Exception as error:
                guard_errors.append(error)
                print("PAPER8B_OWNED_DISK_STOP=" + json.dumps({"pid": process.pid,
                      "timestamp": now(), "error_type": type(error).__name__, "error": str(error)}), flush=True)
                try:
                    os.killpg(process.pid, signal.SIGTERM)
                    try:
                        process.wait(timeout=20)
                    except subprocess.TimeoutExpired:
                        os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                return
            stop_guard.wait(45)

    guard_thread = threading.Thread(target=guard, daemon=True)
    guard_thread.start()
    try:
        yield process
    finally:
        stop_guard.set()
        if process.poll() is None:
            try:
                os.killpg(process.pid, signal.SIGTERM)
                process.wait(timeout=20)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
            except ProcessLookupError:
                pass
        log.close()
        guard_thread.join(timeout=2)
    if guard_errors:
        raise OSError("Owned disk guard stopped the subprocess") from guard_errors[0]


def read_rows(path: Path, offset: int) -> tuple[list[dict], int]:
    if not path.exists():
        return [], offset
    rows = []
    with path.open() as stream:
        stream.seek(offset)
        while True:
            position = stream.tell()
            line = stream.readline()
            if not line or not line.endswith("\n"):
                return rows, position
            rows.append(json.loads(line))


def watch(process, directory: Path, status: dict, status_path: Path, run=None) -> list[dict]:
    offset = 0
    history = []
    checked = 0.0
    while True:
        rows, offset = read_rows(directory / "history.jsonl", offset)
        for row in rows:
            history.append(row)
            if "train/loss" in row:
                if not math.isfinite(row["train/loss"]):
                    raise RuntimeError("Nonfinite training loss")
                status["step"] = row["step"]
                status["last_training_update_at"] = now()
            if "validation_score" in row:
                if not math.isfinite(row["validation_score"]):
                    raise RuntimeError("Nonfinite validation accuracy")
                status["evaluation_count"] = status.get("evaluation_count", 0) + 1
                status["last_validation_at"] = now()
                if row["validation_score"] > status.get("best_accuracy", -1):
                    status["best_accuracy"] = row["validation_score"]
                    status["best_step"] = row["step"]
            if run is not None:
                run.log(wandb_row(row))
        if time.monotonic() - checked >= 45:
            status.update(timestamp=now(), child_pid=process.pid,
                          filesystems=require_space(affected_paths()))
            save(status_path, status)
            print("PAPER8B_STATUS=" + json.dumps(status), flush=True)
            checked = time.monotonic()
        if process.poll() is not None:
            # A process can finish while the last history line is being drained.
            rows, offset = read_rows(directory / "history.jsonl", offset)
            if rows:
                history.extend(rows)
                if run is not None:
                    for row in rows:
                        run.log(wandb_row(row))
            return history
        time.sleep(1)


def choose_microbatch(task: dict, frozen: Path, directory: Path, env: dict,
                      status: dict, status_path: Path) -> int:
    for batch in [32, 16, 8, 4, 2, 1]:
        attempt = directory / f"cloud{status['worker']}-{task['method']}-micro{batch}"
        attempt.mkdir()
        task_path = attempt / "task.json"
        save(task_path, materialize(task, frozen, directory))
        status.update(phase="profiling", profiling_microbatch=batch)
        save(status_path, status)
        command = [str(PYTHON), str(HERE / "cloud14_entry.py"), "profile",
                   "--task", str(task_path), "--microbatch", str(batch)]
        with owned_process(command, attempt, env) as process:
            watch(process, attempt, status, status_path)
        status.setdefault("profiles", []).append({"microbatch": batch, "returncode": process.returncode})
        output = (attempt / "process.log").read_text()
        print("PAPER8B_PROFILE=" + json.dumps({"worker": status["worker"], "task": task,
                                               "microbatch": batch, "returncode": process.returncode,
                                               "tail": output[-5000:]}), flush=True)
        if process.returncode == 0:
            return batch
        if process.returncode not in [42, 43]:
            raise RuntimeError(f"Unexpected profile exit {process.returncode}; diagnosis required")
    raise RuntimeError("No safe microbatch at current free VRAM; resource wait required")


def run_task(task: dict, worker: int, frozen: Path, env: dict) -> None:
    import wandb
    key = task_key(task)
    status_path = STATE / "tasks" / (key + ".json")
    status = json.loads(status_path.read_text())
    run = None
    with tempfile.TemporaryDirectory(prefix="paper8b-" + key + "-") as temporary:
        directory = Path(temporary)
        try:
            run = wandb.init(entity="andrey", project="LoRA-Muon", group=CAMPAIGN,
                             name="paper8b_" + key, mode="online", dir=str(directory),
                             config={"campaign": CAMPAIGN, "method": task["method"],
                                     "optimizer": task["optimizer"], "dataset": task["dataset"],
                                     "seed": task["seed"], "worker": worker, "steps": 3000,
                                     "eval_every": 250, "effective_batch": 32,
                                     "model_revision": env["MODEL_REVISION"],
                                     "frozen_bundle_sha256": (HERE / "cloud14_frozen_runtime.sha256").read_text().strip()},
                             settings=wandb.Settings(save_code=False, disable_git=True))
            run.define_metric("optimizer_step")
            run.define_metric("*", step_metric="optimizer_step")
            status.update(wandb_id=run.id, wandb_url=run.url, phase="profiling")
            batch = choose_microbatch(task, frozen, directory, env, status, status_path)
            status.pop("profiling_microbatch", None)
            status.update(phase="initializing", microbatch=batch, accumulation=32 // batch,
                          started_at=now(), step=0, evaluation_count=0)
            command = [str(PYTHON), "-u", str(frozen / "src/main.py"),
                       *arguments(task, frozen, directory, batch)]
            status["command"] = command
            run.config.update({"microbatch": batch, "command": command})
            status["phase"] = "training"
            save(status_path, status)
            with owned_process(command, directory, env) as process:
                rows = watch(process, directory, status, status_path, run)
            status["returncode"] = process.returncode
            if process.returncode != 0:
                raise RuntimeError(f"Training exited {process.returncode}")
            result = json.loads((directory / ("paper8b_" + key + ".json")).read_text())
            assert not list(directory.rglob("*.pt")) and not list(directory.rglob("*.safetensors"))
            verification = verify_rows(rows, result, task)
            status.update(verification=verification, phase="syncing", ended_at=now())
            save(status_path, status)
            run.summary.update({**verification, "verified": True, "returncode": 0,
                                "checkpoint_count": 0, "reported_accuracy": result["reported_accuracy"],
                                "reported_accuracy_step": result["reported_accuracy_step"]})
            for file in [directory / "history.jsonl", directory / ("paper8b_" + key + ".json"),
                         directory / "process.log"]:
                run.save(str(file), base_path=str(directory), policy="now")
            for file in list(directory.glob("cloud*/*.log")) + list(directory.glob("cloud*/*.jsonl")):
                run.save(str(file), base_path=str(directory), policy="now")
            run.save(str(directory / "disk-guard.jsonl"), base_path=str(directory), policy="now")
            run.finish(exit_code=0)
            run = None
            status.update(phase="complete", wandb_sync_complete=True, completed_at=now(),
                          best_accuracy=result["reported_accuracy"], best_step=result["reported_accuracy_step"])
            save(status_path, status)
        except Exception as error:
            status.update(phase="failed", failed_at=now(), error_type=type(error).__name__,
                          error=str(error), returncode=status.get("returncode"))
            save(status_path, status)
            try:
                if run is not None:
                    for file in directory.rglob("*.log"):
                        run.save(str(file), base_path=str(directory), policy="now")
                    for file in directory.rglob("*.jsonl"):
                        run.save(str(file), base_path=str(directory), policy="now")
                    run.finish(exit_code=1)
            finally:
                # An upload exception must not discard the only raw histories. This
                # temporary log spool is removed after independently verified sync.
                assert not list(directory.rglob("*.pt")) and not list(directory.rglob("*.safetensors"))
                recovery = STATE / "pending-wandb-sync" / (key + f"-{os.getpid()}")
                require_space(affected_paths(), {recovery: sum(p.stat().st_size for p in directory.rglob("*") if p.is_file())})
                recovery.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
                shutil.copytree(directory, recovery, symlinks=False)
                status["recovery_logs"] = str(recovery)
                save(status_path, status)
            print("PAPER8B_FAILURE=" + json.dumps(status), flush=True)
            raise


def worker(worker_id: int) -> None:
    assert worker_id in range(14)
    frozen = unpack()
    value = plan()
    env = configured_environment()
    scratch = Path(tempfile.mkdtemp(prefix=f"paper8b-worker{worker_id}-"))
    env.update(HF_HOME=str(scratch / "hf"), HF_HUB_CACHE=str(scratch / "hf/hub"),
               TORCHINDUCTOR_CACHE_DIR=str(scratch / "torchinductor"),
               TRITON_CACHE_DIR=str(scratch / "triton"), TMPDIR=str(scratch))
    os.environ.update(env)
    worker_path = STATE / "workers" / f"{worker_id}.json"
    status = {"worker": worker_id, "pid": os.getpid(), "hostname": os.uname().nodename,
              "started_at": now(), "cuda_visible_devices": env["CUDA_VISIBLE_DEVICES"]}
    # Hold one worker lock for the process lifetime; duplicate workers fail before CUDA.
    from cloud14_core import locked
    with locked(STATE / "workers" / f"{worker_id}.lock"):
        try:
            while True:
                require_space(affected_paths())
                task = claim(value, worker_id)
                if task is None:
                    break
                status.update(phase="active", task=task, timestamp=now())
                save(worker_path, status)
                run_task(task, worker_id, frozen, env)
            status.update(phase="complete", timestamp=now())
            status.pop("task", None)
            save(worker_path, status)
        except Exception as error:
            status.update(phase="failed", timestamp=now(), error_type=type(error).__name__)
            save(worker_path, status)
            raise


def pair(pair_id: int) -> None:
    assert pair_id in range(7)
    rank = int(os.environ.get("OMPI_COMM_WORLD_LOCAL_RANK", os.environ.get("OMPI_COMM_WORLD_RANK", "0")))
    assert int(os.environ["OMPI_COMM_WORLD_SIZE"]) == 2 and rank in [0, 1]
    require_space(affected_paths())
    unpack()
    query = subprocess.run(["nvidia-smi", "--query-gpu=index,uuid,memory.total,memory.free",
                            "--format=csv,noheader,nounits"], capture_output=True, text=True, check=True)
    devices = [line.split(",")[1].strip() for line in query.stdout.splitlines() if line.strip()]
    assert len(devices) == 2, f"Expected exactly2allocated GPUs, got{devices}"
    print("ALLOCATED_GPU_STATUS=" + query.stdout, flush=True)
    process_query = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,gpu_uuid,used_memory",
                                    "--format=csv,noheader,nounits"], capture_output=True, text=True)
    print("ALLOCATED_GPU_PROCESSES=" + process_query.stdout, flush=True)
    env = {**configured_environment(), "CUDA_VISIBLE_DEVICES": devices[rank],
           "CUDA_DEVICE_ORDER": "PCI_BUS_ID"}
    # Cloud starts this entry point once per GPU, not once per allocation.
    # Replace each MPI rank with exactly one corresponding independent worker.
    os.execve(str(PYTHON), [str(PYTHON), str(HERE / "cloud14_entry.py"), "worker",
                           "--worker-id", str(2 * pair_id + rank)], env)


def inspect() -> None:
    from cloud14_core import snapshot_lines
    ready = ASSETS / "ready.json"
    preparation = STATE / "prepare-status.json"
    snapshot = {"timestamp": now(),
          "filesystems": disk_records(affected_paths()),
          "assets_ready": ready.exists(),
          "assets_ready_record": json.loads(ready.read_text()) if ready.exists() else None,
          "asset_files": [{"path": str(path.relative_to(ASSETS)), "bytes": path.stat().st_size}
                          for path in ASSETS.rglob("*") if path.is_file()] if ASSETS.exists() else [],
          "preparation": json.loads(preparation.read_text()) if preparation.exists() else None,
          "tasks": [json.loads(path.read_text()) for path in sorted((STATE / "tasks").glob("*.json"))],
          "workers": [json.loads(path.read_text()) for path in sorted((STATE / "workers").glob("*.json"))]}
    for line in snapshot_lines(snapshot):
        print(line, flush=True)


def dispatch(args) -> None:
    if args.mode == "prepare":
        with tempfile.TemporaryDirectory(prefix="paper8b-prepare-") as temporary:
            directory = Path(temporary)
            status = {"phase": "preparing_assets", "started_at": now()}
            command = [sys.executable, str(HERE / "cloud14_entry.py"), "prepare-child"]
            with owned_process(command, directory, os.environ.copy()) as process:
                watch(process, directory, status, STATE / "prepare-status.json")
            print((directory / "process.log").read_text()[-20000:], flush=True)
            if process.returncode:
                raise RuntimeError(f"Asset preparation exited{process.returncode}; inspect before retry")
    elif args.mode == "prepare-child":
        prepare()
    elif args.mode == "profile":
        profile_main(args)
    elif args.mode == "worker":
        worker(args.worker_id)
    elif args.mode == "pair":
        pair(args.pair_id)
    elif args.mode == "inspect":
        inspect()
    elif args.mode == "reset-unstarted":
        from cloud14_core import decode_audit_environment, reset_unstarted
        require_space(affected_paths())
        audit = decode_audit_environment("PAPER8B_STARTUP_RECOVERY_AUDIT")
        result = reset_unstarted(STATE, audit, {task_key(task) for task in plan()["queue"]})
        print("PAPER8B_RESET_UNSTARTED=" + json.dumps(result), flush=True)
    elif args.mode == "cleanup":
        from cloud14_core import cleanup_owned_assets, decode_audit_environment
        confirmation = decode_audit_environment("PAPER8B_FINAL_CLEANUP_AUDIT")
        assert confirmation["campaign"] == CAMPAIGN
        assert len(confirmation["audit_sha256"]) == 64
        save(STATE / "final-cleanup-authorization.json", confirmation)
        cleanup_owned_assets(ASSETS, STATE, confirmation)
        print("PAPER8B_CLEANUP_COMPLETE=" + json.dumps(json.loads((STATE / "cleanup.json").read_text())), flush=True)
