"""Run sequential datasets on one GPU with an independent storage watchdog."""

import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time

from disk_guard import DiskGuard


METHODS = ("lora_sgd", "adamw", "lora_muon", "riemannian_sgd", "lora_rite",
           "riemannion_original", "lora_pro", "lora_muon_manifold", "lora_tsd", "muon_full")


class StorageFailure(RuntimeError):
    pass


def check_guard(guard: DiskGuard, **kwargs) -> None:
    try:
        guard(**kwargs)
    except BaseException as error:
        raise StorageFailure(str(error)) from error


def stop_owned(child: subprocess.Popen) -> None:
    if child.poll() is not None:
        return
    os.killpg(child.pid, signal.SIGTERM)
    try:
        child.wait(timeout=10)
    except subprocess.TimeoutExpired:
        os.killpg(child.pid, signal.SIGKILL)
        child.wait(timeout=10)
    print("GEN_STOPPED " + json.dumps({"pid": child.pid, "returncode": child.returncode}), flush=True)


def write_record(path: Path, record: dict) -> None:
    path.write_text(json.dumps(record, indent=2) + "\n")


def publish_storage_stop(root: Path, method: str, error: BaseException) -> None:
    record = {"timestamp": time.time(), "method": method, "pid": os.getpid(),
              "status": "stop", "error": str(error)}
    print("CRITICAL CLOUD.RU STORAGE: " + json.dumps(record), flush=True)
    try:
        with (root / "STOP_STORAGE.json").open("x") as stream:
            json.dump(record, stream)
    except FileExistsError:
        pass
    except OSError as failure:
        print("STORAGE STOP SIGNAL WRITE FAILED: " + str(failure), flush=True)


def run_dataset(args: argparse.Namespace, dataset: str, root: Path,
                queue: dict, queue_path: Path) -> dict:
    name = f"{dataset}_{args.method}_seed42"
    guard = DiskGuard(root, name)
    check_guard(guard, pending_write_bytes=100_000_000, reason="before_log_directory")
    run = root / name
    run.mkdir(parents=True, exist_ok=False)
    guard.telemetry = run / "disk.jsonl"
    command = [sys.executable, "-u", str(Path(__file__).with_name("train_generation.py")),
               "--dataset", dataset, "--method", args.method, "--steps", str(args.steps),
               "--output-dir", str(run)]
    with (run / "train.log").open("w", buffering=1) as log:
        child = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        record = {"name": name, "dataset": dataset, "method": args.method,
                  "pid": child.pid, "supervisor_pid": os.getpid(),
                  "started_at": time.time(), "command": command, "status": "running",
                  "job_id": os.environ.get("HOSTNAME"), "checkpoints": False}
        try:
            write_record(run / "run.json", record)
            queue.update(current_dataset=dataset, pid=child.pid, status="running")
            write_record(queue_path, queue)
        except BaseException:
            stop_owned(child)
            raise
        print("GEN_STARTED " + json.dumps(record), flush=True)

        def stop_signal(signum, frame):
            stop_owned(child)
            raise SystemExit(128 + signum)

        signal.signal(signal.SIGTERM, stop_signal)
        signal.signal(signal.SIGINT, stop_signal)
        probe = {"active": False, "started": 0.0, "error": None}

        def check_storage():
            try:
                guard()
            except BaseException as error:
                probe["error"] = error
            finally:
                probe["active"] = False

        next_guard = 0.0
        try:
            while child.poll() is None or probe["active"]:
                now = time.monotonic()
                if probe["error"] is not None:
                    raise StorageFailure("Storage guard failed") from probe["error"]
                if probe["active"] and now - probe["started"] > 15:
                    raise StorageFailure("Storage telemetry unavailable: probe exceeded 15 seconds")
                if not probe["active"] and now >= next_guard and child.poll() is None:
                    probe.update(active=True, started=now)
                    threading.Thread(target=check_storage, daemon=True).start()
                    next_guard = now + 5
                time.sleep(0.5)
            if probe["error"] is not None:
                raise StorageFailure("Storage guard failed") from probe["error"]
            probe.update(active=True, started=time.monotonic())
            threading.Thread(target=check_storage, daemon=True).start()
            while probe["active"]:
                if time.monotonic() - probe["started"] > 15:
                    raise StorageFailure("Storage telemetry unavailable: exit probe exceeded 15 seconds")
                time.sleep(0.5)
            if probe["error"] is not None:
                raise StorageFailure("Storage guard failed before exit record") from probe["error"]
        except (KeyboardInterrupt, SystemExit):
            stop_owned(child)
            record.update(status="stopped", returncode=child.returncode, ended_at=time.time())
            write_record(run / "run.json", record)
            queue["results"].append(record)
            raise
        except StorageFailure as error:
            stop_owned(child)
            publish_storage_stop(root, args.method, error)
            record.update(status="storage_stopped", returncode=child.returncode,
                          ended_at=time.time(), error=str(error))
            write_record(run / "run.json", record)
            queue["results"].append(record)
            raise
        record.update(status="completed" if child.returncode == 0 else "failed",
                      returncode=child.returncode, ended_at=time.time())
        write_record(run / "run.json", record)
        print("GEN_FINISHED " + json.dumps(record), flush=True)
        print("\n".join((run / "train.log").read_text().splitlines()[-30:]), flush=True)
        return record


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", nargs="+", choices=["e2e_nlg", "smol_smoltalk_100k"], required=True)
    parser.add_argument("--method", choices=METHODS, required=True)
    parser.add_argument("--steps", type=int, default=3000)
    args = parser.parse_args()
    if len(args.dataset) != len(set(args.dataset)):
        parser.error("Datasets must occur only once in a queue")
    root = Path(os.environ["GENERATION_ROOT"])
    guard = DiskGuard(root, f"queue-{args.method}")
    check_guard(guard, pending_write_bytes=100_000_000, reason="before_queue")
    for dataset in args.dataset:
        run = root / f"{dataset}_{args.method}_seed42"
        if run.exists():
            raise FileExistsError(f"Refusing duplicate run: {run}")
    queue_path = root / f"queue-{args.method}.json"
    queue = {"method": args.method, "datasets": args.dataset, "steps": args.steps,
             "supervisor_pid": os.getpid(), "job_id": os.environ.get("HOSTNAME"),
             "started_at": time.time(), "current_dataset": None, "pid": None,
             "status": "starting", "checkpoints": False, "results": []}
    with queue_path.open("x") as stream:
        json.dump(queue, stream, indent=2)
        stream.write("\n")
    try:
        for dataset in args.dataset:
            result = run_dataset(args, dataset, root, queue, queue_path)
            queue["results"].append(result)
            queue.update(current_dataset=None, pid=None, status="between_datasets")
            write_record(queue_path, queue)
    except (KeyboardInterrupt, SystemExit):
        queue.update(status="interrupted", pid=None, ended_at=time.time())
        write_record(queue_path, queue)
        raise
    except StorageFailure as error:
        publish_storage_stop(root, args.method, error)
        queue.update(status="storage_stopped", pid=None, ended_at=time.time(), error=str(error))
        write_record(queue_path, queue)
        raise
    except BaseException as error:
        queue.update(status="failed", pid=None, ended_at=time.time(), error=str(error))
        write_record(queue_path, queue)
        raise
    failed = any(result["returncode"] != 0 for result in queue["results"])
    queue.update(status="completed_with_failures" if failed else "completed",
                 current_dataset=None, pid=None, ended_at=time.time())
    write_record(queue_path, queue)
    print("GEN_QUEUE_FINISHED " + json.dumps(queue), flush=True)
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
