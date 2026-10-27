"""One-GPU queue with process-isolated cases and an independent disk guard."""

import argparse
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import threading
import time

from benchmark import ROOT, case_name, grid
from disk_guard import DiskGuard
from huggingface_hub import snapshot_download
from supervise import stop_owned


def write(path: Path, value: dict, guard: DiskGuard) -> None:
    guard(pending_write_bytes=1_000_000, reason="before_status", write_path=path.parent)
    path.write_text(json.dumps(value, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shard", type=int, choices=range(4), required=True)
    parser.add_argument("--single", action="store_true")
    args = parser.parse_args()
    root = Path(os.environ["SPEED_ROOT"])
    guard = DiskGuard(root, f"speed-shard-{args.shard}")
    guard(pending_write_bytes=100_000_000, reason="before_queue")
    root.mkdir(parents=True, exist_ok=True)
    queue_path = root / f"queue-{args.shard}.json"
    if queue_path.exists():
        raise FileExistsError("Refusing duplicate queue: " + str(queue_path))
    guard.telemetry = root / f"disk-shard-{args.shard}.jsonl"
    source = Path(snapshot_download("meta-llama/Llama-3.2-1B-Instruct", local_files_only=True))
    model_path = Path("/tmp/step-speed-model")
    expected = sum(path.stat().st_size for path in source.rglob("*") if path.is_file())
    guard(pending_write_bytes=expected + 100_000_000, reason="before_model_stage", write_path=Path("/tmp"))
    shutil.copytree(source, model_path)
    cases = grid(args.shard)
    if args.single:
        cases = [{"method": "tsd_tau5", "rank": 16, "batch": 1}]
    state = {"shard": args.shard, "started_at": time.time(), "pid": os.getpid(),
             "hostname": os.environ.get("HOSTNAME"), "status": "running",
             "expected_cases": len(cases), "results": [], "current": None,
             "model_snapshot": source.name, "checkpoints": False}
    write(queue_path, state, guard)
    for case in cases:
        name = case_name(case)
        case_dir = root / name
        guard(pending_write_bytes=10_000_000, reason="before_case", write_path=root)
        case_dir.mkdir(exist_ok=False)
        command = [sys.executable, "-u", str(Path(__file__).with_name("benchmark.py")),
                   "--method", case["method"], "--rank", str(case["rank"]),
                   "--batch", str(case["batch"]), "--model-path", str(model_path),
                   "--output", str(case_dir / "result.json")]
        with (case_dir / "train.log").open("w", buffering=1) as log:
            child = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            state["current"] = {**case, "name": name, "pid": child.pid, "command": command}
            write(queue_path, state, guard)
            print("SPEED_STARTED " + json.dumps(state["current"]), flush=True)

            def terminate(signum, frame):
                stop_owned(child)
                raise SystemExit(128 + signum)

            signal.signal(signal.SIGTERM, terminate)
            signal.signal(signal.SIGINT, terminate)
            probe = {"active": False, "started": 0., "error": None}

            def check_storage():
                try:
                    guard()
                except BaseException as error:
                    probe["error"] = error
                finally:
                    probe["active"] = False

            next_probe = 0.
            while child.poll() is None or probe["active"]:
                now = time.monotonic()
                error = probe["error"]
                if probe["active"] and now - probe["started"] > 15:
                    error = RuntimeError("Disk telemetry unavailable for15seconds")
                if error is not None:
                    stop_owned(child)
                    incident = {"status": "storage_stopped", "error": str(error),
                                "stopped_pid": child.pid, "returncode": child.returncode,
                                "timestamp": time.time(), "shard": args.shard}
                    print("CRITICAL CLOUD.RU STORAGE: " + json.dumps(incident), flush=True)
                    try:
                        with (root / "STOP_STORAGE.json").open("x") as stream:
                            json.dump(incident, stream)
                    except FileExistsError:
                        pass
                    raise RuntimeError("Storage reserve violation; no checkpoints exist to remove") from error
                if not probe["active"] and now >= next_probe and child.poll() is None:
                    probe.update(active=True, started=now)
                    threading.Thread(target=check_storage, daemon=True).start()
                    next_probe = now + 5
                time.sleep(.5)
        result_path = case_dir / "result.json"
        if result_path.exists():
            result = json.loads(result_path.read_text())
        else:
            result = {**case, "status": "failed", "returncode": child.returncode,
                      "error": "Process exited without result; inspect train.log"}
            write(result_path, result, guard)
        state["results"].append({**case, "status": result["status"], "returncode": child.returncode})
        state["current"] = None
        write(queue_path, state, guard)
        print("SPEED_RESULT " + json.dumps(result), flush=True)
    state.update(status="completed", ended_at=time.time())
    write(queue_path, state, guard)
    print("SPEED_QUEUE_DONE " + json.dumps(state), flush=True)


if __name__ == "__main__":
    main()
