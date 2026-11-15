"""One isolated corrected-optimizer task per single-GPU allocation."""

from contextlib import contextmanager
import fcntl
import importlib.util
import json
import math
import os
from pathlib import Path
import queue
import signal
import subprocess
import sys
import tempfile
import threading
import time
import zipfile

from cloud_repair_common import (ASSETS, BUNDLE, BUNDLE_SHA, CAMPAIGN, HERE,
                                 PYTHON, REVISION, STATE, TASKS, affected_paths,
                                 digest, locked, now, require_space, run_name,
                                 save, task_key, verify_rows, wandb_row)

GUARD_INTERVAL = 45
WRITER_STALL_SECONDS = 30


def unpack() -> Path:
    expected = BUNDLE_SHA.read_text().strip()
    if os.environ.get("PAPER8B_REPAIR_BUNDLE_SHA256") != expected:
        raise ValueError("Submitted repair bundle differs from recorded source")
    if digest(BUNDLE) != expected:
        raise ValueError("Repair runtime bundle checksum mismatch")
    target = Path(tempfile.gettempdir()) / ("paper8b-repair-" + expected[:16])
    with locked(target.with_suffix(".lock")):
        marker = target / "ready"
        if not marker.exists():
            target.mkdir(parents=True, exist_ok=True)
            with zipfile.ZipFile(BUNDLE) as archive:
                for item in archive.infolist():
                    if (not (target / item.filename).resolve().is_relative_to(target.resolve())
                            or (item.external_attr >> 16) & 0o170000 == 0o120000):
                        raise ValueError("Unsafe repair bundle entry")
                archive.extractall(target)
            marker.write_text(expected)
        elif marker.read_text() != expected:
            raise ValueError("Repair unpack marker differs")
    return target


def materialize(task: dict, frozen: Path) -> dict:
    if task not in TASKS:
        raise ValueError("Task outside the eight authorized reruns")
    spec = json.loads((frozen / "scripts/brain_lab/paper_8b_methods.json").read_text())
    method = next(row for row in spec["methods"] if row["id"] == task["method"])
    if method["optimizer"] != task["optimizer"]:
        raise ValueError("Method/optimizer identity mismatch")
    return {"method": method, "dataset": task["dataset"], "seed": task["seed"],
            "run_name": run_name(task)}


def arguments(task: dict, frozen: Path, directory: Path, microbatch: int) -> list[str]:
    path = frozen / "scripts/brain_lab/paper_8b_campaign.py"
    spec = importlib.util.spec_from_file_location("repair_argument_protocol", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    result = module.arguments(materialize(task, frozen), microbatch, directory)
    if ("--checkpoint-dir" in result or "--resume-from" in result
            or result[result.index("--checkpoint-every") + 1] != "0"):
        raise ValueError("Checkpointing or resume is forbidden")
    return result


def configured_environment(scratch: Path) -> dict:
    ready = json.loads((ASSETS / "ready.json").read_text())
    ownership = json.loads((ASSETS / "ownership.json").read_text())
    if (ASSETS.is_symlink() or ready["campaign"] != CAMPAIGN
            or ready["revision"] != REVISION
            or ownership != {"campaign": CAMPAIGN, "root": str(ASSETS.resolve())}):
        raise ValueError("Repair asset ownership/revision mismatch")
    return {**os.environ, "MODEL_LOCAL_PATH": str(ASSETS / "model"),
            "MODEL_REVISION": REVISION, "LORA_BENCHMARK_DATA_DIR": str(ASSETS / "data"),
            "HF_HOME": str(scratch / "hf"), "HF_HUB_CACHE": str(scratch / "hf/hub"),
            "HF_HUB_OFFLINE": "1", "HF_DATASETS_OFFLINE": "1",
            "TORCHINDUCTOR_CACHE_DIR": str(scratch / "torchinductor"),
            "TRITON_CACHE_DIR": str(scratch / "triton"), "TMPDIR": str(scratch),
            "PYTHONDONTWRITEBYTECODE": "1", "TOKENIZERS_PARALLELISM": "false",
            "WANDB_BASE_URL": "https://wandb-radfan.ru", "WANDB_MODE": "online",
            "WANDB_ENTITY": "andrey", "WANDB_PROJECT": "LoRA-Muon",
            "WANDB_LOG_MODEL": "false"}


def single_gpu() -> dict:
    for key in ("OMPI_COMM_WORLD_SIZE", "OMPI_COMM_WORLD_LOCAL_SIZE"):
        if int(os.environ.get(key, "1")) != 1:
            raise ValueError("Repair allocations must contain exactly one MPI rank")
    for key in ("OMPI_COMM_WORLD_RANK", "OMPI_COMM_WORLD_LOCAL_RANK"):
        if int(os.environ.get(key, "0")) != 0:
            raise ValueError("Repair worker must be the allocation's only rank")
    query = subprocess.run(["nvidia-smi", "--query-gpu=uuid,memory.total,memory.free",
                            "--format=csv,noheader,nounits"], capture_output=True,
                           text=True, check=True, timeout=20)
    rows = [line.split(",") for line in query.stdout.splitlines() if line.strip()]
    if len(rows) != 1 or len(rows[0]) != 3 or not rows[0][0].strip().startswith("GPU-"):
        raise ValueError("Expected exactly one allocated full-GPU UUID")
    return {"cuda_visible_devices": rows[0][0].strip(),
            "gpu_memory_total_mib": int(rows[0][1]), "gpu_memory_free_mib": int(rows[0][2])}


def stop_owned(process) -> None:
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
        try:
            process.wait(timeout=20)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass


def write_guard_event(directory: Path, event: dict) -> None:
    if event["kind"] == "telemetry":
        with (directory / "disk-guard.jsonl").open("a") as stream:
            stream.write(json.dumps(event) + "\n")
    else:
        print("REPAIR_OWNED_DISK_STOP=" + json.dumps(event), flush=True)


@contextmanager
def owned_process(command: list[str], directory: Path, env: dict):
    require_space(affected_paths())
    stopped = threading.Event()
    wake = threading.Event()
    writing = threading.Event()
    events = queue.Queue(maxsize=8)
    writer_started = [0.0]
    writer_errors = []
    failures = []
    with (directory / "process.log").open("w") as log:
        process = subprocess.Popen(command, cwd=directory, env=env, stdout=log,
                                   stderr=subprocess.STDOUT, start_new_session=True)

        def writer() -> None:
            while not stopped.is_set() or not events.empty():
                try:
                    event = events.get(timeout=.2)
                except queue.Empty:
                    continue
                writer_started[0] = time.monotonic()
                writing.set()
                try:
                    write_guard_event(directory, event)
                except Exception as error:
                    writer_errors.append(error)
                    wake.set()
                    return
                finally:
                    writing.clear()

        def guard() -> None:
            while not stopped.is_set() and process.poll() is None:
                try:
                    if writer_errors:
                        raise OSError("Disk-guard logging failed") from writer_errors[0]
                    if writing.is_set() and time.monotonic() - writer_started[0] >= WRITER_STALL_SECONDS:
                        raise OSError("Disk-guard logging stalled")
                    records = require_space(affected_paths())
                    events.put_nowait({"kind": "telemetry", "timestamp": now(),
                                       "pid": process.pid, "filesystems": records})
                except Exception as error:
                    failures.append(error)
                    stop_owned(process)
                    try:
                        events.put_nowait({"kind": "stop", "pid": process.pid,
                                           "timestamp": now(), "error": str(error)})
                    except queue.Full:
                        pass
                    return
                wake.wait(GUARD_INTERVAL)
                wake.clear()

        thread = threading.Thread(target=guard, daemon=True)
        writer_thread = threading.Thread(target=writer, daemon=True)
        try:
            thread.start()
            writer_thread.start()
            save(directory / "owned-process.json", {"pid": process.pid, "command": command,
                                                   "started_at": now()})
            yield process
        finally:
            stopped.set()
            wake.set()
            stop_owned(process)
            if thread.ident is not None:
                thread.join(timeout=16)
            if writer_thread.ident is not None:
                writer_thread.join(timeout=2)
        if failures or writer_errors or writer_thread.is_alive() or thread.is_alive():
            cause = (failures or writer_errors or [None])[0]
            raise OSError("Owned disk guard stopped or could not record protection") from cause


def read_rows(path: Path, offset: int) -> tuple[list[dict], int]:
    if not path.exists():
        return [], offset
    rows = []
    with path.open() as stream:
        stream.seek(offset)
        while True:
            start = stream.tell()
            line = stream.readline()
            if not line or not line.endswith("\n"):
                return rows, start
            rows.append(json.loads(line))


def watch(process, directory: Path, status: dict, status_path: Path, run=None) -> list[dict]:
    offset, checked = 0, 0.0
    history = []
    while True:
        rows, offset = read_rows(directory / "history.jsonl", offset)
        for row in rows:
            history.append(row)
            if "train/loss" in row:
                if not math.isfinite(row["train/loss"]):
                    raise RuntimeError("Nonfinite training loss")
                status.update(step=row["step"], last_training_update_at=now())
            if "validation_score" in row:
                if not math.isfinite(row["validation_score"]):
                    raise RuntimeError("Nonfinite validation accuracy")
                status.update(evaluation_count=status.get("evaluation_count", 0) + 1,
                              last_validation_at=now())
                if row["validation_score"] > status.get("best_accuracy", -1):
                    status.update(best_accuracy=row["validation_score"], best_step=row["step"])
            if status.get("phase") != "profiling" and "step" in status:
                pending_final = status["step"] == 3000 and status.get("evaluation_count", 0) < 12
                status["phase"] = "validation" if status["step"] % 250 == 249 or pending_final else "training"
            if run is not None:
                run.log(wandb_row(row))
        if time.monotonic() - checked >= 45:
            status.update(timestamp=now(), child_pid=process.pid,
                          filesystems=require_space(affected_paths()))
            save(status_path, status)
            checked = time.monotonic()
        if process.poll() is not None:
            remaining, _ = read_rows(directory / "history.jsonl", offset)
            if not remaining:
                return history
            # Drain a final complete line through the same finite-value checks.
            continue
        time.sleep(1)


def profile_main(args) -> None:
    from cloud_repair_closure import assert_campaign_open
    assert_campaign_open()
    frozen = unpack()
    materialized = json.loads(args.task.read_text())
    task = TASKS[args.worker_id]
    if materialized != materialize(task, frozen):
        raise ValueError("Profile task differs from the assigned repair task")
    import torch
    from cloud_repair_profile import main
    try:
        main(args, frozen, materialized)
    except torch.OutOfMemoryError:
        print("PROFILE_OOM", flush=True)
        raise SystemExit(42)


def choose_microbatch(task: dict, frozen: Path, directory: Path,
                      env: dict, status: dict, status_path: Path) -> int:
    for batch in (32, 16, 8, 4, 2, 1):
        attempt = directory / f"cloud{status['worker']}-{task['method']}-micro{batch}"
        attempt.mkdir()
        save(attempt / "task.json", materialize(task, frozen))
        status.update(phase="profiling", profiling_microbatch=batch)
        save(status_path, status)
        command = [str(PYTHON), str(HERE / "cloud_repair_entry.py"), "profile",
                   "--worker-id", str(status["worker"]), "--task", str(attempt / "task.json"),
                   "--microbatch", str(batch)]
        with owned_process(command, attempt, env) as process:
            watch(process, attempt, status, status_path)
        status.setdefault("profiles", []).append({"microbatch": batch,
                                                  "returncode": process.returncode})
        save(status_path, status)
        if process.returncode == 0:
            return batch
        if process.returncode not in (42, 43):
            raise RuntimeError(f"Unexpected profile exit {process.returncode}; diagnosis required")
    raise RuntimeError("No microbatch fits; resource diagnosis required, no automatic restart")


def diagnostic_manifest(directory: Path, status: dict) -> dict:
    rows = []
    for path in sorted(directory.rglob("*")):
        relative = path.relative_to(directory)
        if relative.parts[0] == "upload":
            continue
        if path.is_symlink():
            raise ValueError("Unexpected link in repair diagnostic spool")
        if path.is_file() and path.suffix in (".json", ".jsonl", ".log"):
            if path.name != "diagnostic-manifest.json":
                name = path.name if len(relative.parts) == 1 else (
                    "profile-micro" + relative.parts[0].rsplit("-micro", 1)[1] + "-" + path.name)
                rows.append({"path": str(relative), "wandb_name": name,
                             "bytes": path.stat().st_size, "sha256": digest(path)})
    if len({row["wandb_name"] for row in rows}) != len(rows):
        raise ValueError("Diagnostic upload names are not unique")
    return {"campaign": CAMPAIGN, "run_name": status["run_name"],
            "wandb_id": status["wandb_id"], "runtime_bundle_sha256": BUNDLE_SHA.read_text().strip(),
            "model_revision": REVISION, "command": status.get("command"), "files": rows}


def run_task(task: dict, worker_id: int, frozen: Path, scratch: Path, env: dict,
             status: dict, status_path: Path) -> None:
    import wandb
    directory = STATE / "pending-wandb-sync" / task_key(task)
    directory.mkdir(parents=True, exist_ok=False)
    status.update(recovery_logs=str(directory), run_name=run_name(task),
                  wandb_id=wandb.util.generate_id())
    save(status_path, status)
    run = None
    try:
        run = wandb.init(entity="andrey", project="LoRA-Muon", group=CAMPAIGN,
                         name=run_name(task), id=status["wandb_id"], resume="never",
                         mode="online", dir=str(scratch),
                         config={"campaign": CAMPAIGN, **task, "worker": worker_id,
                                 "steps": 3000, "eval_every": 250, "effective_batch": 32,
                                 "model_revision": REVISION,
                                 "runtime_bundle_sha256": BUNDLE_SHA.read_text().strip()},
                         settings=wandb.Settings(save_code=False, disable_git=True))
        run.define_metric("optimizer_step")
        run.define_metric("*", step_metric="optimizer_step")
        status["wandb_url"] = run.url
        batch = choose_microbatch(task, frozen, directory, env, status, status_path)
        status.pop("profiling_microbatch", None)
        command = [str(PYTHON), "-u", str(frozen / "src/main.py"),
                   *arguments(task, frozen, directory, batch)]
        status.update(phase="training", command=command, microbatch=batch,
                      accumulation=32 // batch, step=0, evaluation_count=0, started_at=now())
        save(status_path, status)
        run.config.update({"microbatch": batch, "command": command})
        with owned_process(command, directory, env) as process:
            rows = watch(process, directory, status, status_path, run)
        status["returncode"] = process.returncode
        if process.returncode != 0:
            raise RuntimeError(f"Training exited {process.returncode}")
        result = json.loads((directory / (run_name(task) + ".json")).read_text())
        if list(directory.rglob("*.pt")) or list(directory.rglob("*.safetensors")):
            raise ValueError("Unexpected model/checkpoint artifact")
        verification = verify_rows(rows, result, task)
        status.update(verification=verification, phase="syncing", ended_at=now())
        save(status_path, status)
        manifest = diagnostic_manifest(directory, status)
        save(directory / "diagnostic-manifest.json", manifest)
        status.update(recovery_manifest_sha256=digest(directory / "diagnostic-manifest.json"),
                      syncphase="uploading")
        save(status_path, status)
        upload = directory / "upload"
        upload.mkdir()
        for entry in manifest["files"] + [{"path": "diagnostic-manifest.json",
                                           "wandb_name": "diagnostic-manifest.json"}]:
            destination = upload / entry["wandb_name"]
            os.link(directory / entry["path"], destination)
            run.save(str(destination), base_path=str(upload), policy="now")
        run.summary.update({**verification, "verified": True, "returncode": 0,
                            "checkpoint_count": 0, "reported_accuracy": result["reported_accuracy"],
                            "reported_accuracy_step": result["reported_accuracy_step"]})
        run.finish(exit_code=0)
        run = None
        status.update(phase="complete", wandb_sync_complete=True, completed_at=now(),
                      best_accuracy=result["reported_accuracy"], best_step=result["reported_accuracy_step"],
                      child_pid=None,
                      syncphase="uploaded_awaiting_independent_verification",
                      recovery_manifest_sha256=digest(directory / "diagnostic-manifest.json"))
        save(status_path, status)
    except Exception as error:
        status.update(phase="failed", failed_at=now(), error_type=type(error).__name__,
                      error=str(error), syncphase="pending_recovery")
        save(status_path, status)
        if run is not None:
            try:
                run.summary.update({"verified": False, "failure": str(error)})
                run.finish(exit_code=1)
            except Exception:
                pass
        raise


def worker(worker_id: int) -> None:
    if worker_id not in range(len(TASKS)):
        raise ValueError("Worker outside the eight authorized reruns")
    from cloud_repair_closure import assert_campaign_open
    assert_campaign_open()
    require_space(affected_paths(), {STATE: 1_000_000})
    task = TASKS[worker_id]
    worker_path = STATE / "workers" / f"{worker_id}.json"
    worker_path.parent.mkdir(parents=True, exist_ok=True)
    with (worker_path.with_suffix(".lock")).open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        status_path = STATE / "tasks" / (task_key(task) + ".json")
        with locked(STATE / "claims.lock"):
            prior = json.loads(status_path.read_text())
            if prior.get("phase") != "reserved" or prior.get("task") != task:
                raise ValueError("Task already claimed or not explicitly reserved")
            status = {"task": task, "worker": worker_id, "phase": "claimed", "pid": os.getpid(),
                      "hostname": os.uname().nodename, "claimed_at": now()}
            save(status_path, status)
        try:
            require_space(affected_paths())
            gpu = single_gpu()
            status.update(gpu)
            frozen = unpack()
            scratch = Path(tempfile.mkdtemp(prefix=f"paper8b-repair-worker{worker_id}-"))
            env = configured_environment(scratch)
            env.update(CUDA_VISIBLE_DEVICES=gpu["cuda_visible_devices"], CUDA_DEVICE_ORDER="PCI_BUS_ID")
            os.environ.update(env)
            save(status_path, status)
            save(worker_path, status)
            run_task(task, worker_id, frozen, scratch, env, status, status_path)
        except Exception as error:
            status.update(phase="failed", error_type=type(error).__name__, error=str(error), timestamp=now())
            save(status_path, status)
            raise
        finally:
            save(worker_path, status)
