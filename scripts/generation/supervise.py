"""Run one owned GPU process with an independent storage watchdog."""

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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["e2e_nlg", "smol_smoltalk_100k"], required=True)
    parser.add_argument("--method", choices=["lora_tsd", "lora_rite"], required=True)
    parser.add_argument("--steps", type=int, default=3000)
    args = parser.parse_args()
    root = Path(os.environ["GENERATION_ROOT"])
    name = f"{args.dataset}_{args.method}_seed42"
    guard = DiskGuard(root, name)
    guard(pending_write_bytes=100_000_000, reason="before_log_directory")
    run = root / name
    run.mkdir(parents=True, exist_ok=False)
    guard.telemetry = run / "disk.jsonl"
    command = [sys.executable, "-u", str(Path(__file__).with_name("train_generation.py")),
               "--dataset", args.dataset, "--method", args.method, "--steps", str(args.steps),
               "--output-dir", str(run)]
    with (run / "train.log").open("w", buffering=1) as log:
        child = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        record = {"name": name, "pid": child.pid, "supervisor_pid": os.getpid(),
                  "started_at": time.time(), "command": command, "status": "starting",
                  "job_id": os.environ.get("HOSTNAME"), "checkpoints": False}
        (run / "run.json").write_text(json.dumps(record, indent=2) + "\n")
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
            while child.poll() is None:
                now = time.monotonic()
                if probe["error"] is not None:
                    raise RuntimeError("Storage guard failed") from probe["error"]
                if probe["active"] and now - probe["started"] > 15:
                    raise RuntimeError("Storage telemetry unavailable: probe exceeded 15 seconds")
                if not probe["active"] and now >= next_guard:
                    probe.update(active=True, started=now)
                    threading.Thread(target=check_storage, daemon=True).start()
                    next_guard = now + 5
                time.sleep(0.5)
        except BaseException:
            stop_owned(child)
            print("CRITICAL CLOUD.RU STORAGE: WATCHDOG STOPPED OWNED TRAINING PID " + str(child.pid), flush=True)
            record.update(status="stopped", returncode=child.returncode, ended_at=time.time())
            (run / "run.json").write_text(json.dumps(record, indent=2) + "\n")
            raise
        record.update(status="completed" if child.returncode == 0 else "failed",
                      returncode=child.returncode, ended_at=time.time())
        guard(reason="before_exit_record")
        (run / "run.json").write_text(json.dumps(record, indent=2) + "\n")
        print("GEN_FINISHED " + json.dumps(record), flush=True)
        print("\n".join((run / "train.log").read_text().splitlines()[-30:]), flush=True)
        if child.returncode:
            raise SystemExit(child.returncode)


if __name__ == "__main__":
    main()
