"""Bounded local collection of repair CPU snapshots and exact scheduler evidence."""

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import json
from pathlib import Path
import re
import subprocess
import sys
import time
import zlib

from cloud14_core import decode_snapshot
from cloud_repair_common import ASSETS, CAMPAIGN, affected_paths, digest, now, save
from cloud_repair_submit import ARTIFACTS, SSH


JOB_ID = re.compile(r"^lm-mpi-job-[a-f0-9-]+$")
TERMINAL = {"completed", "failed", "stopped", "cancelled", "canceled"}
FRESH_SECONDS = 300


@contextmanager
def ledger_lock(deadline: float):
    with (ARTIFACTS / "submission.lock").open("a") as lock:
        while True:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                if time.monotonic() >= deadline:
                    raise RuntimeError("Collection deadline reached while another control action held the ledger")
                time.sleep(.05)
        yield


def text(value) -> str:
    return value.decode(errors="replace") if isinstance(value, bytes) else (value or "")


def age(timestamp: str) -> float:
    observed = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
    return (datetime.now(timezone.utc) - observed).total_seconds()


def remote_cli(command: list[str], deadline: float, limit: int = 65) -> dict:
    remaining = deadline - time.monotonic()
    if remaining < 15:
        raise RuntimeError("Collection deadline reached; reuse the recorded CPU job")
    timeout = min(limit, int(remaining - 10))
    remote = "import subprocess,os,signal,json\np=subprocess.Popen(" + repr(command)
    remote += ",stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True,start_new_session=True)\n"
    remote += "try:\n out,err=p.communicate(timeout=" + str(timeout) + ")\n"
    remote += " print(json.dumps({'returncode':p.returncode,'stdout':out,'stderr':err,'timed_out':False}))\n"
    remote += "except subprocess.TimeoutExpired as e:\n"
    remote += " try:os.killpg(p.pid,signal.SIGKILL)\n except ProcessLookupError:pass\n"
    remote += " convert=lambda x:x.decode(errors='replace') if isinstance(x,bytes) else (x or '')\n"
    remote += " print(json.dumps({'returncode':None,'stdout':convert(e.output),'stderr':convert(e.stderr),'timed_out':True}))\n"
    try:
        response = subprocess.run(SSH + ["python3", "-"], input=remote, capture_output=True,
                                  text=True, timeout=min(remaining, timeout + 10))
        output, error, transport, timed_out = response.stdout, response.stderr, response.returncode, False
    except subprocess.TimeoutExpired as failure:
        output, error, transport, timed_out = text(failure.output), text(failure.stderr), None, True
    try:
        payload = json.loads(output)
    except (ValueError, TypeError):
        payload = {"returncode": None, "stdout": output, "stderr": error, "timed_out": timed_out,
                   "envelope_incomplete": True}
    return {**payload, "ssh_returncode": transport, "ssh_timed_out": timed_out,
            "ssh_stderr": error, "raw_transport_stdout": output}


def owned_records(campaign: dict) -> list[dict]:
    rows = []
    def visit(value) -> None:
        if isinstance(value, list):
            for item in value:
                visit(item)
        elif isinstance(value, dict):
            if value.get("job_id"):
                if not JOB_ID.fullmatch(value["job_id"]):
                    raise ValueError("Invalid owned repair job ID")
                rows.append(value)
            for key in ("jobs", "auxiliary_jobs", "allocation_history", "archived_jobs"):
                if key in value:
                    visit(value[key])
    visit(campaign)
    return rows


def scheduler(deadline: float) -> dict:
    with ledger_lock(deadline):
        campaign = json.loads((ARTIFACTS / "campaign.json").read_text())
        if campaign.get("campaign") != CAMPAIGN:
            raise ValueError("Scheduler ledger belongs to another campaign")
        owned = {row["job_id"] for row in owned_records(campaign)}
    transport = remote_cli(["mlsub", "list"], deadline, 45)
    raw = text(transport.get("stdout"))
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    proof = ARTIFACTS / ("scheduler-list-" + stamp + ".log")
    proof.write_text(raw)
    save(ARTIFACTS / ("scheduler-transport-" + stamp + ".json"), transport)
    matches = {}
    for line in raw.splitlines():
        fields = line.strip().split(None, 2)
        if len(fields) == 3 and fields[0] in owned:
            matches.setdefault(fields[0], []).append((fields[1], line))
    results = []
    with ledger_lock(deadline):
        campaign = json.loads((ARTIFACTS / "campaign.json").read_text())
        grouped = {}
        for row in owned_records(campaign):
            grouped.setdefault(row["job_id"], []).append(row)
        for job, records in grouped.items():
            candidates = matches.get(job, [])
            valid = len(candidates) == 1 and (
                "repo:dimativator/LoRA-TSD|entry:scripts/cloud/cloud_repair_entry.py|" in candidates[0][1])
            result = {"job_id": job, "status": None, "source": "not_uniquely_listed",
                      "historical_status": records[0].get("scheduler_status")}
            if valid:
                status, line = candidates[0]
                result.update(status=status, source="fresh_exact_mlsub_list_row", evidence=line)
                for row in records:
                    row.update(scheduler_status=status, scheduler_observed_at=now(),
                               scheduler_evidence=str(proof))
            results.append(result)
        save(ARTIFACTS / "campaign.json", campaign)
    snapshot = {"campaign": CAMPAIGN, "timestamp": now(), "jobs": results,
                "raw_proof": str(proof), "raw_proof_sha256": digest(proof),
                "cli_returncode": transport.get("returncode"),
                "ssh_returncode": transport["ssh_returncode"],
                "timed_out": bool(transport.get("timed_out") or transport["ssh_timed_out"])}
    save(ARTIFACTS / "scheduler-snapshot.json", snapshot)
    return snapshot


def validate_snapshot(snapshot: dict, mode: str) -> bool:
    if snapshot.get("campaign") != CAMPAIGN or not -60 <= age(snapshot["timestamp"]) <= FRESH_SECONDS:
        raise ValueError("Snapshot campaign or timestamp is not current")
    records = snapshot["filesystems"]
    paths = [row["requested_path"] for row in records]
    if len(paths) != len(set(paths)) or set(paths) != {str(path) for path in affected_paths()}:
        raise ValueError("Snapshot lacks exact affected filesystem coverage")
    enough = all(type(row.get("available_bytes")) is int and row["available_bytes"] > 5_000_000_000
                 and row.get("status") == "ok" for row in records)
    if mode == "preflight":
        from cloud_repair_assets import manifests
        required = sum(row["bytes"] for row in manifests().values())
        home = next(row for row in records if row["requested_path"] == str(ASSETS))
        return bool(enough and snapshot.get("venv_import_returncode") == 0
                    and snapshot.get("asset_bytes") == required
                    and home["available_bytes"] > 5_000_000_000 + required + 100_000_000
                    and snapshot.get("space_for_fresh_assets") is True)
    return bool(enough and snapshot.get("assets_ready") is True
                and (snapshot.get("preparation") or {}).get("phase") == "ready")


def collect(job: dict, mode: str, deadline: float) -> dict:
    job_id = job["job_id"]
    if not JOB_ID.fullmatch(job_id) or job.get("mode") != mode:
        raise ValueError("Snapshot job is not the selected owned CPU mode")
    transport = remote_cli(["mlsub", "logs", job_id, "--tail", "80"], deadline)
    raw = text(transport.get("stdout")) + "\n" + text(transport.get("stderr"))
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    prefix = f"{mode}-{stamp}-{job_id}"
    (ARTIFACTS / (prefix + ".log")).write_text(raw)
    save(ARTIFACTS / (prefix + "-transport.json"), transport)
    snapshot = decode_snapshot(raw)
    flag = validate_snapshot(snapshot, mode)
    snapshot.update(collector_source_job_id=job_id, collector_source_mode=mode, collector_received_at=now(),
                    collector_cli_returncode=transport.get("returncode"),
                    collector_transport_returncode=transport.get("returncode"),
                    collector_ssh_returncode=transport["ssh_returncode"],
                    collector_timed_out=bool(transport.get("timed_out") or transport["ssh_timed_out"]))
    filename = "preflight.json" if mode == "preflight" else "inspection-latest.json"
    save(ARTIFACTS / filename, snapshot)
    with ledger_lock(deadline):
        campaign = json.loads((ARTIFACTS / "campaign.json").read_text())
        if campaign.get("campaign") != CAMPAIGN:
            raise ValueError("Snapshot ledger belongs to another campaign")
        for row in campaign["auxiliary_jobs"]:
            if row["job_id"] == job_id:
                row.update(snapshot_collected_at=now(), snapshot_timestamp=snapshot["timestamp"])
        campaign.update(storage_latest=snapshot["filesystems"], latest_snapshot=snapshot["timestamp"])
        if mode == "preflight":
            campaign["space_preflight_verified"] = flag
        else:
            campaign.update(assets_ready_verified=flag, latest_inspection=filename)
        save(ARTIFACTS / "campaign.json", campaign)
    return snapshot


def select_job(mode: str, job_id: str | None, submit: bool, deadline: float) -> dict:
    with ledger_lock(deadline):
        campaign = json.loads((ARTIFACTS / "campaign.json").read_text())
        if campaign.get("campaign") != CAMPAIGN:
            raise ValueError("Collector ledger belongs to another campaign")
        jobs = [row for row in campaign["auxiliary_jobs"] if row["mode"] == mode]
        if job_id:
            matches = [row for row in jobs if row["job_id"] == job_id]
            if len(matches) != 1:
                raise ValueError("Job ID is not an existing owned auxiliary job of this mode")
            return matches[0]
        latest = jobs[-1] if jobs else None
        if not submit:
            if latest is None:
                raise ValueError("No existing CPU snapshot job; use --submit once")
            return latest
        if campaign.get("ambiguous_submissions"):
            raise RuntimeError("Resolve the pending submission intent before another CPU job")
        if latest:
            timestamp = latest.get("snapshot_timestamp") or latest.get("created_at")
            if timestamp is None or age(timestamp) <= FRESH_SECONDS:
                return latest
    if latest and latest.get("scheduler_status", "").lower() not in TERMINAL:
        refreshed = scheduler(deadline)
        current = next(row for row in refreshed["jobs"] if row["job_id"] == latest["job_id"])
        if not current["status"] or current["status"].lower() not in TERMINAL:
            return latest
    remaining = deadline - time.monotonic()
    if remaining < 30:
        raise RuntimeError("Insufficient collection deadline for another CPU submission")
    command = [sys.executable, str(Path(__file__).with_name("cloud_repair_submit.py")), mode]
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                               text=True, start_new_session=True)
    try:
        output, error = process.communicate(timeout=min(175, remaining))
    except subprocess.TimeoutExpired:
        import os
        import signal
        os.killpg(process.pid, signal.SIGKILL)
        raise RuntimeError("CPU submission response is ambiguous; recover its existing intent") from None
    (ARTIFACTS / f"collector-{mode}-submission.log").write_text(output + error)
    with ledger_lock(deadline):
        campaign = json.loads((ARTIFACTS / "campaign.json").read_text())
        if process.returncode or campaign.get("ambiguous_submissions"):
            raise RuntimeError("CPU submission failed or is ambiguous; do not duplicate it")
        jobs = [row for row in campaign["auxiliary_jobs"] if row["mode"] == mode]
        if not jobs or (latest and jobs[-1]["job_id"] == latest["job_id"]):
            raise RuntimeError("CPU submission did not record a new auxiliary job")
        return jobs[-1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("preflight", "inspect", "scheduler"), default="inspect")
    parser.add_argument("--job-id")
    parser.add_argument("--submit", action="store_true")
    parser.add_argument("--deadline-seconds", type=int, default=240, choices=range(30, 301))
    args = parser.parse_args()
    if args.mode == "scheduler" and (args.job_id or args.submit):
        parser.error("Scheduler collection never submits jobs or selects a log job")
    deadline = time.monotonic() + args.deadline_seconds
    with (ARTIFACTS / "collection.lock").open("a") as control:
        fcntl.flock(control, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if args.mode == "scheduler":
            result = scheduler(deadline)
            print(json.dumps(result, indent=2))
        else:
            job = select_job(args.mode, args.job_id, args.submit, deadline)
            try:
                result = collect(job, args.mode, deadline)
            except (AssertionError, ValueError, OSError, EOFError, KeyError, TypeError, zlib.error) as error:
                with ledger_lock(deadline):
                    campaign = json.loads((ARTIFACTS / "campaign.json").read_text())
                    if campaign.get("campaign") != CAMPAIGN:
                        raise ValueError("Failed collection ledger belongs to another campaign")
                    campaign["space_preflight_verified" if args.mode == "preflight" else "assets_ready_verified"] = False
                    campaign["last_collection_error"] = {"job_id": job["job_id"],
                        "timestamp": now(), "error_type": type(error).__name__}
                    save(ARTIFACTS / "campaign.json", campaign)
                raise
            print(json.dumps({"timestamp": result["timestamp"], "job_id": job["job_id"],
                              "cli_returncode": result["collector_cli_returncode"],
                              "filesystems": result["filesystems"],
                              "assets_ready": result.get("assets_ready")}, indent=2))


if __name__ == "__main__":
    main()
