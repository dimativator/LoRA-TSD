"""Read W&B progress and independently verify finished seed histories."""

import argparse
from datetime import datetime, timezone
import json
import math
import netrc
import os
from pathlib import Path

from cloud14_core import CAMPAIGN, now, save, task_key, verify_rows
from cloud14_submit import ARTIFACTS, WORKSPACE


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--verify-finished", action="store_true")
    args = parser.parse_args()
    import wandb
    os.environ["WANDB_BASE_URL"] = "https://wandb-radfan.ru"
    os.environ["WANDB_API_KEY"] = netrc.netrc().authenticators("wandb-radfan.ru")[2]
    api = wandb.Api(timeout=45)
    plan = json.loads((WORKSPACE / "artifacts/three-seed-all-optimizers-8b-plan-20261113.json").read_text())
    tasks = {"paper8b_" + task_key(task): task for task in plan["queue"]}
    for method in plan["methods"]:
        for task in method["existing"]:
            tasks[task["run_name"]] = task
    verified_path = ARTIFACTS / "wandb-independent-verification.json"
    verified = json.loads(verified_path.read_text()) if verified_path.exists() else {}
    interrupted_path = ARTIFACTS / "interrupted-attempts.json"
    interrupted = json.loads(interrupted_path.read_text()) if interrupted_path.exists() else {}
    inspection_path = ARTIFACTS / "inspection-latest.json"
    if inspection_path.exists():
        for record in json.loads(inspection_path.read_text())["tasks"]:
            if record["phase"] == "failed" and record.get("wandb_id"):
                interrupted.setdefault(record["wandb_id"], {"diagnosis": record.get("error")})
    active = []
    failed = []
    pending_verification = []
    preflight = []
    for run in api.runs("andrey/LoRA-Muon", filters={"group": CAMPAIGN}, per_page=100):
        if run.name not in tasks:
            preflight.append({"name": run.name, "state": run.state, "url": run.url})
            continue
        task = tasks[run.name]
        summary = dict(run.summary)
        entry = {"name": run.name, "path": run.path, "url": run.url,
                 "method": task["method"], "dataset": task["dataset"], "seed": task["seed"],
                 "state": run.state, "step": summary.get("optimizer_step", 0),
                 "worker": run.config.get("worker"), "microbatch": run.config.get("microbatch"),
                 "best_accuracy": summary.get("best_accuracy"),
                 "best_step": summary.get("best_step"), "reused": bool(run.config.get("reused"))}
        if run.id in interrupted:
            entry.update(state="failed_or_interrupted", diagnosis=interrupted[run.id].get("diagnosis"),
                         wandb_reported_state=run.state)
            failed.append(entry)
            continue
        if run.state == "finished" and summary.get("verified") is True:
            if run.name in verified:
                assert verified[run.name]["path"] == run.path, "Duplicate completed seed requires audit"
                continue
            if args.verify_finished:
                directory = ARTIFACTS / "raw-wandb" / run.name
                directory.mkdir(parents=True, exist_ok=True)
                result_name = "metrics.json" if entry["reused"] else run.name + ".json"
                for name in ["history.jsonl", result_name]:
                    file = run.file(name)
                    assert file.name == name
                    file.download(root=str(directory), replace=True)
                rows = [json.loads(line) for line in (directory / "history.jsonl").read_text().splitlines()]
                result = json.loads((directory / result_name).read_text())
                checks = verify_rows(rows, result, task)
                assert summary["returncode"] == 0 and summary["checkpoint_count"] == 0
                verified[run.name] = {**entry, **checks, "verified_at": now(),
                                      "history_file": str(directory / "history.jsonl")}
                save(verified_path, verified)
            else:
                pending_verification.append(entry)
        elif run.state in ["failed", "crashed", "killed"]:
            failed.append(entry)
        else:
            evaluations = list(run.scan_history(keys=["optimizer_step", "validation_score"]))
            if evaluations:
                best = max(evaluations, key=lambda row: row["validation_score"])
                entry.update(best_accuracy=best["validation_score"], best_step=best["optimizer_step"],
                             evaluation_count=len(evaluations))
            else:
                entry["evaluation_count"] = 0
            if entry["microbatch"] is None:
                entry["phase"] = "profiling"
            elif entry["step"] == 0:
                entry["phase"] = "initializing"
            elif entry["step"] % 250 == 249 or (entry["step"] == 3000 and entry["evaluation_count"] < 12):
                entry["phase"] = "validation"
            else:
                entry["phase"] = "training"
            entry["created_at"] = run.created_at
            active.append(entry)
    reused_count = sum(entry["reused"] for entry in verified.values())
    new_count = len(verified) - reused_count
    snapshot = {"timestamp": now(), "campaign": CAMPAIGN, "verified_completed": len(verified),
                "target": 180, "reused_verified": reused_count, "new_verified": new_count,
                "new_required": 138, "active": active, "failed": failed,
                "pending_independent_verification": pending_verification,
                "preflight_runs": preflight}
    if len(verified) == 180:
        assert len({(row["method"], row["dataset"], row["seed"]) for row in verified.values()}) == 180
        assert not active and not pending_verification
        snapshot["all_histories_verified"] = True
    filename = "monitor-check-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + ".json"
    save(ARTIFACTS / filename, snapshot)
    save(ARTIFACTS / "latest-monitor-check.json", snapshot)
    print(json.dumps(snapshot, indent=2))


if __name__ == "__main__":
    main()
