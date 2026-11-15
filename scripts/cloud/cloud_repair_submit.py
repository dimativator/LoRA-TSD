"""Submit journaled repair jobs through the existing gateway, without exposing secrets."""

import argparse
from datetime import datetime, timezone
import json
import netrc
from pathlib import Path
import re
import subprocess
import time

from cloud_repair_common import CAMPAIGN, TASKS, digest, locked, now, save
from cloud14_core import encode_audit_environment


ARTIFACTS = Path(__file__).resolve().parents[4] / "artifacts/paper_8b_pro_rite_repair_20261115"
SSH = ["ssh", "-o", "ClearAllForwardings=yes", "-o", "BatchMode=yes", "-o", "ConnectTimeout=25",
       "-o", "ControlMaster=no", "-o", "ControlPath=none", "-o", "ServerAliveInterval=10",
       "-o", "ServerAliveCountMax=2", "brain_lab"]


def cpu_inspection(record: dict) -> bool:
    return (record.get("mode") == "inspect" and type(record.get("gpus")) is int
            and record["gpus"] == 0 and record.get("worker") is None)


def gpu_intent(record: dict) -> bool:
    return (record.get("mode") == "worker" and type(record.get("gpus")) is int
            and record["gpus"] == 1 and type(record.get("worker")) is int
            and record["worker"] in range(len(TASKS)))


def reserved_gpu_slots(campaign: dict) -> int:
    """Every unanswered GPU intent consumes its slot; none is a retry opportunity."""
    pending = campaign.get("ambiguous_submissions", [])
    if any(row.get("gpus") != 0 and not gpu_intent(row) for row in pending):
        raise RuntimeError("Unclassified pending allocation must be resolved")
    records = campaign["jobs"] + [row for row in pending if gpu_intent(row)]
    workers = [row["worker"] for row in records]
    assert all(type(worker) is int and worker in range(len(TASKS)) for worker in workers)
    assert all(row.get("gpus", 1) == 1 for row in campaign["jobs"])
    assert len(workers) == len(set(workers)), "Worker already has an explicit create intent"
    ids = [row["job_id"] for row in campaign["jobs"]]
    assert len(ids) == len(set(ids)), "Duplicate received GPU job ID"
    assert len(records) <= len(TASKS), "Received jobs and unknown GPU intents exceed the eight tasks"
    return len(campaign["jobs"]) + sum(max(1, len(set(row.get("observed_job_ids", []))))
        for row in pending if gpu_intent(row))


def remember_receipt(campaign: dict, pending: dict, output: str, proof: Path, partial: bool = False) -> set:
    ids = set(re.findall(r"lm-mpi-job-[a-f0-9-]+", output))
    if ids:
        observed = ids | set(pending.get("observed_job_ids", []))
        pending["observed_job_ids"] = sorted(observed)
        known = {row["job_id"] for row in campaign["jobs"] + campaign["auxiliary_jobs"]}
        known.update(job for row in campaign["ambiguous_submissions"] if row is not pending
                     for job in row.get("observed_job_ids", []))
        reason = "Multiple job IDs in submission response" if len(observed) > 1 else (
            "Receipt job ID is already recorded or observed" if observed & known else None)
        if reason:
            pending["recovery_conflict"] = {"reason": reason, "observed_at": now(),
                "evidence": str(proof), "evidence_sha256": digest(proof)}
    if partial:
        pending["partial_receipt"] = True
    campaign["reserved_gpu_slots"] = reserved_gpu_slots(campaign)
    save(ARTIFACTS / "campaign.json", campaign)
    return ids


def defer_inspection(campaign: dict) -> None:
    """Retain one uncertain CPU observation without blocking independent training."""
    pending = campaign.get("ambiguous_submissions", [])
    if len(pending) != 1 or campaign.get("unresolved_auxiliary_submissions"):
        raise RuntimeError("Only one unresolved CPU observation may be deferred")
    record = pending[0]
    if not cpu_inspection(record):
        raise RuntimeError("GPU or state-changing submissions must be resolved before proceeding")
    campaign["unresolved_auxiliary_submissions"] = [{**record, "deferred_at": now(),
        "reason": "User resumed launch after CPU-only ambiguity was explained; retain for final audit"}]
    campaign["ambiguous_submissions"] = []


def recover(auxiliary: bool = False) -> None:
    with locked(ARTIFACTS / "submission.lock"):
        campaign = json.loads((ARTIFACTS / "campaign.json").read_text())
        deferred = campaign.get("unresolved_auxiliary_submissions", [])
        if len(deferred) > 1 or any(not cpu_inspection(row) for row in deferred):
            raise RuntimeError("Unresolved allocation must be proven CPU-only before proceeding")
        key = "unresolved_auxiliary_submissions" if auxiliary else "ambiguous_submissions"
        pending = campaign.get(key, [])
        if not pending:
            print("No ambiguous repair submission")
            return
        assert len({row["note"] for row in pending}) == len(pending), "Duplicate pending note"
        reserved_gpu_slots(campaign)
        if auxiliary:
            assert len(pending) == 1 and cpu_inspection(pending[0])
        from cloud_repair_collect import query_listings, listing_rows
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        prefix = "recovery-" + ("auxiliary" if auxiliary else "pending") + "-" + stamp
        queries = query_listings(time.monotonic() + 110, ARTIFACTS, prefix)
        text = "\n".join(f"# query={row['source']} observed_at={row['observed_at']}\n{row['stdout']}" for row in queries)
        save_path = ARTIFACTS / (prefix + ".log")
        save_path.write_text(text)
        rows = listing_rows(queries)
        adopted, unresolved = [], []
        for intent in list(pending):
            matches = [row for row in rows if row["note"] == intent["note"]]
            ids = {row["job_id"] for row in matches}
            observed = ids | set(intent.get("observed_job_ids", []))
            conflict = "Multiple job IDs share the pending note" if len(observed) > 1 else None
            if not ids and not conflict:
                unresolved.append({"note": intent["note"], "error": "No unique scheduler match yet; keep intent pending"})
                continue
            if not conflict and len(ids) != 1:
                conflict = "No unique scheduler match yet"
            job_id = next(iter(ids)) if ids else None
            same_id = [row for row in rows if row["job_id"] == job_id]
            if not conflict and (any(not row["owned_entry"] or row["note"] != intent["note"]
                    or not isinstance(row["status"], str) for row in same_id)
                    or len({row["status"].lower() for row in same_id}) != 1):
                conflict = "Scheduler identity/status evidence conflicts"
            if conflict:
                intent.update(observed_job_ids=sorted(observed), recovery_conflict={"reason": conflict,
                              "observed_at": now(), "evidence": str(save_path), "evidence_sha256": digest(save_path)})
                campaign["reserved_gpu_slots"] = reserved_gpu_slots(campaign)
                save(ARTIFACTS / "campaign.json", campaign)
                unresolved.append({"note": intent["note"], "error": conflict + "; keep intent pending"})
                continue
            collection = "jobs" if intent["mode"] == "worker" else "auxiliary_jobs"
            if any(row["job_id"] == job_id for row in campaign["jobs"] + campaign["auxiliary_jobs"]):
                unresolved.append({"note": intent["note"], "error": "Job ID is already recorded; preserve intent for audit"})
                continue
            latest = max(matches, key=lambda row: row["observed_at"])
            record = {**intent, "job_id": job_id, "scheduler_status": latest["status"],
                      "scheduler_observed_at": latest["observed_at"], "adopted_at": now(),
                      "unique_note_evidence": str(save_path), "absence_proof": False,
                      "unique_note_evidence_sha256": digest(save_path),
                      "listings_complete": all(row["complete"] for row in queries),
                      "listing_queries": [{name: value for name, value in row.items() if name != "stdout"} for row in queries]}
            campaign[collection].append(record)
            campaign[key] = [row for row in campaign[key] if row["note"] != intent["note"]]
            campaign["reserved_gpu_slots"] = reserved_gpu_slots(campaign)
            save(ARTIFACTS / "campaign.json", campaign)
            adopted.append(record)
        print(json.dumps({"adopted": adopted, "unresolved": unresolved}, indent=2))
        if not adopted and unresolved:
            raise RuntimeError(unresolved[0]["error"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["preflight", "prepare", "worker", "inspect", "cleanup", "reconcile", "recover", "recover-auxiliary", "refresh-inspection"])
    parser.add_argument("--worker-id", type=int, choices=range(len(TASKS)))
    args = parser.parse_args()
    if args.mode in {"recover", "recover-auxiliary"}:
        recover(args.mode == "recover-auxiliary")
        return
    if args.mode == "worker" and args.worker_id is None:
        parser.error("Worker mode requires --worker-id")
    with locked(ARTIFACTS / "submission.lock"):
        campaign = json.loads((ARTIFACTS / "campaign.json").read_text())
        if args.mode == "refresh-inspection":
            defer_inspection(campaign)
            save(ARTIFACTS / "campaign.json", campaign)
            args.mode = "inspect"
        deferred = campaign.get("unresolved_auxiliary_submissions", [])
        if len(deferred) > 1 or any(not cpu_inspection(row) for row in deferred):
            raise RuntimeError("Unresolved allocation must be proven CPU-only before proceeding")
        pending = campaign.get("ambiguous_submissions", [])
        reserved = reserved_gpu_slots(campaign)
        if pending and (args.mode not in {"worker", "inspect"} or any(not gpu_intent(row) for row in pending)):
            raise RuntimeError("Resolve CPU or state-changing pending submission before another create intent")
        if args.mode == "worker":
            if any(row.get("recovery_conflict") for row in pending):
                raise RuntimeError("Resolve observed pending allocation conflicts before another GPU create intent")
            assert campaign.get("review_passed") and campaign.get("assets_ready_verified")
            assert campaign.get("runtime_bundle_sha256")
            assert not any(row["worker"] == args.worker_id for row in campaign["jobs"])
            assert not any(row["worker"] == args.worker_id for row in pending if gpu_intent(row))
            assert reserved < len(TASKS) <= campaign["gpu_limit"] == 10
        if args.mode == "prepare":
            assert campaign.get("space_preflight_verified") is True
            assert campaign.get("preparation_guard_review_passed") is True
        environment = {"PYTHONDONTWRITEBYTECODE": "1", "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
                       "TMPDIR": "/tmp", "TEMP": "/tmp", "TMP": "/tmp",
                       "WANDB_BASE_URL": "https://wandb-radfan.ru", "WANDB_ENTITY": "andrey",
                       "WANDB_PROJECT": "LoRA-Muon"}
        secrets = []
        if args.mode in {"prepare", "worker", "inspect"}:
            key = netrc.netrc().authenticators("wandb-radfan.ru")[2]
            environment["WANDB_API_KEY"] = key
            secrets.append(key)
        if args.mode == "prepare":
            inventory = Path.home() / "Documents/Obsidian Vault/🧠 BRAIn Lab/general/servers.md"
            match = re.search(r"HF token \(llama 3\.1\):\s*`([^`]+)`", inventory.read_text())
            if match is None:
                raise RuntimeError("Configured model-access credential is unavailable")
            environment["HF_TOKEN"] = match.group(1)
            secrets.append(match.group(1))
        if args.mode == "worker":
            environment["PAPER8B_REPAIR_BUNDLE_SHA256"] = campaign["runtime_bundle_sha256"]
        if args.mode in {"cleanup", "reconcile"}:
            assert not campaign.get("unresolved_auxiliary_submissions"), "Deferred CPU inspection must be audited before cleanup"
            confirmation = json.loads((ARTIFACTS / "cleanup-authorization.json").read_text())
            assert confirmation["verified_runs"] == len(TASKS)
            environment.update(encode_audit_environment("REPAIR_FINAL_AUDIT", confirmation))
        number = len(campaign["jobs"]) + len(campaign["auxiliary_jobs"]) + len(campaign.get("unresolved_auxiliary_submissions", []))
        intent = f"{args.mode}-{args.worker_id if args.worker_id is not None else number}"
        note = "repair8b_" + intent + "_" + now().replace(":", "").replace("+", "_").replace(".", "_")
        if args.mode == "inspect":
            environment.update(REPAIR_INSPECT_UPLOAD="1", REPAIR_CONTROL_NOTE=note)
        arguments = args.mode + (f" --worker-id {args.worker_id}" if args.mode == "worker" else "")
        command = ["mlsub", "run", "--repo", "https://github.com/dimativator/LoRA-TSD", "--branch", "dev",
                   "--entry", "scripts/cloud/cloud_repair_entry.py", "--args", arguments,
                   "--gpus", "1" if args.mode == "worker" else "cpu", "--image", "torch28", "--no-pip",
                   "--note", note]
        for key, value in environment.items():
            command.extend(["--env", key + "=" + value])
        pending = {"intent": intent, "note": note, "mode": args.mode, "worker": args.worker_id,
                   "created_at": now(), "source_revision": campaign["source_revision"],
                   "environment_names": sorted(environment), "gpus": 1 if args.mode == "worker" else 0}
        campaign.setdefault("ambiguous_submissions", []).append(pending)
        campaign["reserved_gpu_slots"] = reserved_gpu_slots(campaign)
        save(ARTIFACTS / "campaign.json", campaign)
        remote = "import subprocess,json\ncommand=" + repr(command) + "\n"
        remote += "d=subprocess.run(command+['--dry-run'],capture_output=True,text=True,timeout=50)\n"
        remote += "assert d.returncode==0\n"
        remote += "assert json.loads(d.stdout)['logs_dir']=='/home/jovyan/shares/SR006.nfs3/mlsub-logs'\n"
        remote += "r=subprocess.run(command,capture_output=True,text=True,timeout=240)\n"
        remote += "out=r.stdout+r.stderr\n"
        remote += "for secret in " + repr(secrets) + ":out=out.replace(secret,'<redacted>')\n"
        remote += "print(out)\nraise SystemExit(r.returncode)\n"
        try:
            result = subprocess.run(SSH + ["python3", "-"], input=remote,
                                    capture_output=True, text=True, timeout=305)
        except subprocess.TimeoutExpired as error:
            output = "".join(value.decode(errors="replace") if isinstance(value, bytes) else (value or "")
                             for value in (error.output, error.stderr))
            for secret in secrets:
                output = output.replace(secret, "<redacted>")
            proof = ARTIFACTS / ("submission-" + intent + ".log")
            proof.write_text(output)
            remember_receipt(campaign, pending, output, proof, partial=True)
            raise RuntimeError("Submission response lost; adopt exact unique note, never retry blindly") from None
        output = result.stdout + result.stderr
        for secret in secrets:
            output = output.replace(secret, "<redacted>")
        proof = ARTIFACTS / ("submission-" + intent + ".log")
        proof.write_text(output)
        ids = remember_receipt(campaign, pending, output, proof)
        if len(ids) != 1:
            raise RuntimeError("No unique submission ID; recover exact note from scheduler")
        job_id = ids.pop()
        if pending.get("recovery_conflict"):
            raise RuntimeError("Receipt identity conflicts with an existing job; preserve pending intent")
        record = {**pending, "job_id": job_id, "transport_returncode": result.returncode,
                  "submitted_at": now()}
        campaign["ambiguous_submissions"] = [row for row in campaign["ambiguous_submissions"] if row["note"] != note]
        collection = "jobs" if args.mode == "worker" else "auxiliary_jobs"
        campaign[collection].append(record)
        campaign["reserved_gpu_slots"] = reserved_gpu_slots(campaign)
        save(ARTIFACTS / "campaign.json", campaign)
        print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
