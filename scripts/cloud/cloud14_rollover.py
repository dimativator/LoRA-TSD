"""Replace only naturally terminal allocations, retaining at most14allocatedGPUs."""

import argparse
from datetime import datetime, timezone
import fcntl
import json
import netrc
import os
from pathlib import Path
import subprocess
import sys
import time

from cloud14_core import CAMPAIGN, locked, now, save, task_key
from cloud14_scheduler import TERMINAL, snapshot
from cloud14_submit import ARTIFACTS

HERE = Path(__file__).resolve().parent


def run_helper(name: str, *args: str) -> None:
    result = subprocess.run([sys.executable, str(HERE / name), *args],
                            capture_output=True, text=True, timeout=260)
    with (ARTIFACTS / "rollover-actions.log").open("a") as stream:
        stream.write(now() + " " + name + " " + " ".join(args) + "\n" + result.stdout + result.stderr)
    result.check_returncode()


def inspection(mode: str = "inspect") -> dict:
    run_helper("cloud14_submit.py", mode)
    campaign = json.loads((ARTIFACTS / "campaign.json").read_text())
    job = campaign["preparation_jobs"][-1]["job_id"]
    deadline = time.monotonic() + 600
    while time.monotonic() < deadline:
        try:
            run_helper("cloud14_collect.py", "--job-id", job)
            value = json.loads((ARTIFACTS / "inspection-latest.json").read_text())
            age = (datetime.now(timezone.utc) - datetime.fromisoformat(value["timestamp"])).total_seconds()
            assert 0 <= age < 300
            return value
        except subprocess.CalledProcessError:
            time.sleep(10)
    raise RuntimeError("CPU inspection not available; no GPU replacement submitted")


def preserve_partial(record: dict, job: dict, diagnosis: str) -> dict:
    import wandb
    os.environ["WANDB_BASE_URL"] = "https://wandb-radfan.ru"
    os.environ["WANDB_API_KEY"] = netrc.netrc().authenticators("wandb-radfan.ru")[2]
    api = wandb.Api(timeout=45)
    run = api.run("andrey/LoRA-Muon/" + record["wandb_id"])
    assert run.name == "paper8b_" + task_key(record["task"])
    directory = ARTIFACTS / "partial-wandb" / (run.name + "-" + run.id)
    directory.mkdir(parents=True, exist_ok=True)
    rows = list(run.scan_history())
    save(directory / "attempt.json", {"task_status": record, "wandb_id": run.id,
                                      "wandb_state": run.state, "scheduler_job": job})
    (directory / "history.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    try:
        run.file("process.log").download(root=str(directory), replace=True)
    except wandb.errors.CommError:
        pass  # MPI can kill the healthy rank before its final raw-log upload.
    audit = {"campaign": CAMPAIGN, "timestamp": now(), "key": task_key(record["task"]),
             "diagnosis": diagnosis, "job_status": job["status"], "job_id": job["job_id"],
             "workers": [record["worker"]], "wandb_id": record["wandb_id"],
             "last_step": record.get("step", 0), "raw_partial_history_preserved": True,
             "partial_history_records": len(rows), "partial_artifact": str(directory),
             "interrupted_by_allocation_failure": record["phase"] != "failed"}
    archive = ARTIFACTS / "interrupted-attempts.json"
    old = json.loads(archive.read_text()) if archive.exists() else {}
    old[run.id] = audit
    save(archive, old)
    return audit


def archive_terminal_pair(pair: dict, scheduler: dict) -> None:
    assert scheduler["job_id"] == pair["job_id"] and scheduler["status"].lower() in TERMINAL
    age = (datetime.now(timezone.utc) - datetime.fromisoformat(scheduler["audited_at"])).total_seconds()
    assert 0 <= age < 300
    with (ARTIFACTS / "submission.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        path = ARTIFACTS / "campaign.json"
        campaign = json.loads(path.read_text())
        current = next(row for row in campaign["jobs"] if row["pair_id"] == pair["pair_id"])
        assert current["job_id"] == pair["job_id"]
        campaign.setdefault("allocation_history", []).append({"reason": "natural generation-two rollover",
                   "timestamp": now(), "jobs": [{**current, "terminal_audit": scheduler}]})
        campaign["jobs"].remove(current)
        campaign.setdefault("pending_pair_replacements", {})[str(pair["pair_id"])] = current["job_id"]
        save(path, campaign)


def once() -> bool:
    path = ARTIFACTS / "campaign.json"
    campaign = json.loads(path.read_text())
    assert campaign.get("generation_two_reserved") is True
    for pair_id in campaign.get("pending_pair_replacements", {}):
        if not any(row["pair_id"] == int(pair_id) for row in campaign["jobs"]):
            assert not campaign.get("ambiguous_submissions"), "Resolve submission ambiguity before another allocation"
            run_helper("cloud14_submit.py", "pair", "--pair-id", pair_id)
            return True
    scheduler = snapshot()
    by_id = {row["job_id"]: {**row, "audited_at": scheduler["timestamp"]} for row in scheduler["jobs"]}
    # Prove one corrected profile and real initialization before rolling more pairs.
    new_jobs = [row for row in campaign["jobs"] if row.get("launcher_generation") == 2]
    if new_jobs and not campaign.get("generation_two_runtime_verified"):
        state = inspection()
        passing = [row for row in state["tasks"] if row.get("launcher_generation") == 2
                   and row.get("step", 0) > 0 and row.get("memory_profile_version") == 2
                   and any(profile["returncode"] == 0 for profile in row.get("profiles", []))]
        if not passing:
            save(ARTIFACTS / "rollover-status.json", {"timestamp": now(), "phase": "waiting_real_initialization_proof"})
            return False
        save(ARTIFACTS / "generation-two-runtime-proof.json", {"timestamp": now(), "passing_tasks": passing})
        with locked(ARTIFACTS / "submission.lock"):
            campaign = json.loads(path.read_text())
            campaign["generation_two_runtime_verified"] = True
            save(path, campaign)
    for pair in campaign["jobs"]:
        job = by_id[pair["job_id"]]
        if job["status"].lower() not in TERMINAL or pair.get("launcher_generation") == 2:
            continue
        state = inspection()
        records = [record for record in state["tasks"]
                   if record.get("hostname", "").startswith(pair["job_id"] + "-")]
        incomplete = [record for record in records if record["phase"] != "complete"]
        if incomplete:
            audits = [preserve_partial(record, job,
                "RSVD OOM or MPI counterpart interruption; padded initialization memory probe added")
                for record in incomplete]
            # Refresh terminal state after downloading all partial evidence.
            fresh = snapshot()
            fresh_job = next(row for row in fresh["jobs"] if row["job_id"] == pair["job_id"])
            assert fresh_job["status"].lower() in TERMINAL
            for audit in audits:
                audit.update(timestamp=fresh["timestamp"], job_status=fresh_job["status"])
            save(ARTIFACTS / "retry-audit.json", {"attempts": audits})
            state = inspection("retry-failed")
        available = [row for row in state["tasks"] if row["phase"] == "reserved_v2"]
        if not available:
            continue
        fresh = snapshot()
        fresh_job = next(row for row in fresh["jobs"] if row["job_id"] == pair["job_id"])
        fresh_job["audited_at"] = fresh["timestamp"]
        archive_terminal_pair(pair, fresh_job)
        run_helper("cloud14_submit.py", "pair", "--pair-id", str(pair["pair_id"]))
        save(ARTIFACTS / "rollover-status.json", {"timestamp": now(), "phase": "replacement_submitted",
                                                "pair_id": pair["pair_id"], "interrupted_attempts": len(incomplete)})
        return True
    return False


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--watch", action="store_true")
    args = parser.parse_args()
    with (ARTIFACTS / "rollover.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            print("Existing rollover coordinator owns the lock")
            return
        save(ARTIFACTS / "rollover-coordinator.json", {"pid": os.getpid(), "started_at": now(), "watch": args.watch})
        while True:
            try:
                once()
            except Exception as error:
                save(ARTIFACTS / "rollover-error.json", {"timestamp": now(), "type": type(error).__name__,
                                                        "error": str(error)})
                print(now(), type(error).__name__, str(error), flush=True)
                if not args.watch:
                    raise
            if not args.watch:
                return
            campaign = json.loads((ARTIFACTS / "campaign.json").read_text())
            if len(campaign["jobs"]) == 7 and all(row.get("launcher_generation") == 2 for row in campaign["jobs"]):
                return
            time.sleep(60)


if __name__ == "__main__":
    main()
