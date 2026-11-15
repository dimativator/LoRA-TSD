"""CPU preparation/inspection and independent single-GPU corrected reruns."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

from cloud14_core import decode_audit_environment
from cloud_repair_common import (ASSETS, CAMPAIGN, PYTHON, STATE, TASKS,
                                 affected_paths, disk_records, now, snapshot_lines)


def inspect(preflight: bool = False) -> None:
    records = disk_records(affected_paths())
    result = {"campaign": CAMPAIGN, "timestamp": now(), "filesystems": records,
              "assets_ready": (ASSETS / "ready.json").exists(), "target_runs": len(TASKS)}
    try:
        result["nfs2_read_only"] = disk_records([Path("/home/jovyan/shares/SR006.nfs2")])
    except OSError as error:
        result["nfs2_read_only"] = {"unavailable": True, "error_type": type(error).__name__}
    if preflight:
        probe = subprocess.run([str(PYTHON), "-c", "import torch,wandb,transformers,peft,pyarrow,json,importlib.metadata; "
                                "assert torch.__version__.startswith('2.8.'); "
                                "print(json.dumps({p:importlib.metadata.version(p) for p in "
                                "['torch','wandb','transformers','peft','pyarrow','requests']}))"],
                               capture_output=True, text=True, timeout=90)
        result["venv_import_returncode"] = probe.returncode
        result["venv_import_output"] = probe.stdout[-500:]
        from cloud_repair_assets import manifests
        total = sum(row["bytes"] for row in manifests().values())
        home = next(row for row in records if row["requested_path"] == str(ASSETS))
        result["asset_bytes"] = total
        result["space_for_fresh_assets"] = home["available_bytes"] > 5_000_000_000 + total + 100_000_000
    else:
        result["tasks"] = [json.loads(path.read_text()) for path in sorted((STATE / "tasks").glob("*.json"))]
        result["workers"] = [json.loads(path.read_text()) for path in sorted((STATE / "workers").glob("*.json"))]
        result["pending_spools"] = [str(path) for path in sorted((STATE / "pending-wandb-sync").glob("*"))]
        preparation = STATE / "preparation.json"
        result["preparation"] = json.loads(preparation.read_text()) if preparation.exists() else None
        cleanup = STATE / "cleanup.json"
        result["cleanup"] = json.loads(cleanup.read_text()) if cleanup.exists() else None
    for line in snapshot_lines(result):
        print(line, flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["preflight", "prepare", "prepare-child", "worker", "profile", "inspect", "cleanup"])
    parser.add_argument("--worker-id", type=int, choices=range(len(TASKS)))
    parser.add_argument("--task", type=Path)
    parser.add_argument("--microbatch", type=int)
    args = parser.parse_args()
    if args.mode == "worker" and Path(sys.prefix).resolve() != PYTHON.parent.parent.resolve():
        os.execv(str(PYTHON), [str(PYTHON), str(Path(__file__).resolve()), *sys.argv[1:]])
    if args.mode in {"preflight", "inspect"}:
        inspect(args.mode == "preflight")
    elif args.mode in {"prepare", "prepare-child"}:
        from cloud_repair_assets import prepare, prepare_guarded
        prepare_guarded() if args.mode == "prepare" else prepare()
    elif args.mode == "cleanup":
        from cloud_repair_assets import cleanup
        result = cleanup(decode_audit_environment("REPAIR_FINAL_AUDIT"))
        for line in snapshot_lines(result):
            print(line, flush=True)
    else:
        from cloud_repair_runtime import profile_main, worker
        if args.mode == "worker":
            if args.worker_id is None:
                parser.error("Worker mode requires --worker-id")
            worker(args.worker_id)
        else:
            if args.task is None or args.microbatch is None:
                parser.error("Profile mode requires --task and --microbatch")
            profile_main(args)


if __name__ == "__main__":
    main()
