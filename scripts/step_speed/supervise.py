"""GPU capacity probe and guarded queues using a private, disposable 8B model."""

import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from benchmark import case_name, grid, METHODS
from disk_guard import DiskGuard
from model_storage import SPEC, stage_model
from supervise import stop_owned


def write(path: Path, value: dict, guard: DiskGuard) -> None:
    guard(pending_write_bytes=1_000_000, reason="before_status", write_path=path.parent)
    temporary = path.with_name(path.name + ".tmp-" + str(os.getpid()))
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def execute(case: dict, args, model_path: Path, root: Path, state: dict,
            queue_path: Path, guard: DiskGuard) -> dict:
    name = case_name(case)
    case_dir = root / name
    guard(pending_write_bytes=10_000_000, reason="before_case", write_path=root)
    case_dir.mkdir(exist_ok=False)
    microbatch = case["batch"] if args.capacity_probe or case["batch"] <= 32 else args.microbatch_cap
    command = [sys.executable, "-u", str(Path(__file__).with_name("benchmark.py")),
               "--method", case["method"], "--rank", str(case["rank"]),
               "--batch", str(case["batch"]), "--microbatch", str(microbatch),
               "--model-path", str(model_path), "--output", str(case_dir / "result.json")]
    if args.capacity_probe:
        command += ["--warmup", "1", "--windows", "1", "--steps-per-window", "1",
                    "--optimizer-diagnostic-steps", "0"]
    with (case_dir / "train.log").open("w", buffering=1) as log:
        child = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            def terminate(signum, frame):
                stop_owned(child)
                raise SystemExit(128 + signum)

            signal.signal(signal.SIGTERM, terminate)
            signal.signal(signal.SIGINT, terminate)
            state["current"] = {**case, "name": name, "pid": child.pid, "command": command,
                                "microbatch": microbatch}
            write(queue_path, state, guard)
            print("SPEED_STARTED " + json.dumps(state["current"]), flush=True)
            child.wait()
        finally:
            stop_owned(child)
    result_path = case_dir / "result.json"
    if result_path.exists():
        result = json.loads(result_path.read_text())
    else:
        result = {**case, "status": "failed", "returncode": child.returncode,
                  "error": "Process exited without result; inspect train.log"}
        write(result_path, result, guard)
    state["results"].append({**case, "status": result["status"], "returncode": child.returncode,
                             "microbatch": microbatch})
    state["current"] = None
    write(queue_path, state, guard)
    print("SPEED_RESULT " + json.dumps(result), flush=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shard", type=int, choices=range(4), default=0)
    parser.add_argument("--capacity-probe", action="store_true")
    parser.add_argument("--microbatch-cap", type=int, choices=[1, 2, 4, 8, 16, 32], default=32)
    args = parser.parse_args()
    root = Path(os.environ["SPEED_ROOT"])
    guard = DiskGuard(root, f"speed-shard-{args.shard}")
    guard(pending_write_bytes=100_000_000, reason="before_queue")
    root.mkdir(parents=True, exist_ok=True)
    queue_path = root / f"queue-{args.shard}.json"
    if queue_path.exists():
        raise FileExistsError("Refusing duplicate queue: " + str(queue_path))
    cases = grid(args.shard)
    state = {"shard": args.shard, "started_at": time.time(), "pid": os.getpid(),
             "hostname": os.environ.get("HOSTNAME"), "status": "staging_model",
             "expected_cases": None if args.capacity_probe else len(cases), "results": [],
             "current": None, "model_snapshot": SPEC["revision"], "checkpoints": False,
             "capacity_probe": args.capacity_probe, "microbatch_cap": args.microbatch_cap}
    write(queue_path, state, guard)
    model_path = stage_model(root)
    state["status"] = "running"
    write(queue_path, state, guard)
    if args.capacity_probe:
        state["selected_microbatch"] = None
        for batch in [32, 16, 8, 4, 2, 1]:
            results = []
            for method in METHODS:
                result = execute({"method": method, "rank": 32, "batch": batch}, args,
                                 model_path, root, state, queue_path, guard)
                results.append(result)
                if result["status"] == "failed":
                    raise RuntimeError("Capacity probe failed; inspect before retrying")
                if result["status"] == "oom":
                    break
            if len(results) == len(METHODS) and all(r["status"] == "completed" for r in results):
                safe = all(r["peak_reserved_bytes"] + 1_000_000_000 < r["memory_before_load"]["gpu_free_bytes"]
                           for r in results)
                if safe:
                    state["selected_microbatch"] = batch
                    break
        state.update(status="completed", ended_at=time.time())
        write(queue_path, state, guard)
        print("SPEED_CAPACITY " + json.dumps(state), flush=True)
        if state["selected_microbatch"] is None:
            raise RuntimeError("No common microbatch fits the available GPU memory")
        return
    for case in cases:
        execute(case, args, model_path, root, state, queue_path, guard)
    state.update(status="completed", ended_at=time.time())
    write(queue_path, state, guard)
    print("SPEED_QUEUE_DONE " + json.dumps(state), flush=True)


if __name__ == "__main__":
    main()
