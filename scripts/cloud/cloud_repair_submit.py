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


def recover() -> None:
    with locked(ARTIFACTS / "submission.lock"):
        campaign = json.loads((ARTIFACTS / "campaign.json").read_text())
        pending = campaign.get("ambiguous_submissions", [])
        if not pending:
            print("No ambiguous repair submission")
            return
        assert len(pending) == 1
        intent = pending[0]
        if intent["mode"] == "worker":
            assert intent["worker"] in range(len(TASKS)) and intent["gpus"] == 1
            assert len(campaign["jobs"]) < len(TASKS)
            assert not any(row["worker"] == intent["worker"] for row in campaign["jobs"])
        else:
            assert intent["gpus"] == 0
        from cloud_repair_collect import query_listings, listing_rows
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        prefix = "recovery-" + intent["intent"] + "-" + stamp
        queries = query_listings(time.monotonic() + 110, ARTIFACTS, prefix)
        text = "\n".join(f"# query={row['source']} observed_at={row['observed_at']}\n{row['stdout']}" for row in queries)
        save_path = ARTIFACTS / (prefix + ".log")
        save_path.write_text(text)
        rows = listing_rows(queries)
        matches = [row for row in rows if row["note"] == intent["note"]]
        ids = {row["job_id"] for row in matches}
        if len(ids) != 1:
            raise RuntimeError("No unique scheduler match yet; keep intent pending")
        job_id = next(iter(ids))
        same_id = [row for row in rows if row["job_id"] == job_id]
        if (any(not row["owned_entry"] or row["note"] != intent["note"] for row in same_id)
                or len({row["status"].lower() for row in same_id}) != 1):
            raise RuntimeError("Scheduler identity/status evidence conflicts; keep intent pending")
        latest = max(matches, key=lambda row: row["observed_at"])
        record = {**intent, "job_id": job_id, "scheduler_status": latest["status"],
                  "scheduler_observed_at": latest["observed_at"], "adopted_at": now(),
                  "unique_note_evidence": str(save_path), "absence_proof": False,
                  "unique_note_evidence_sha256": digest(save_path),
                  "listings_complete": all(row["complete"] for row in queries),
                  "listing_queries": [{key: value for key, value in row.items() if key != "stdout"} for row in queries]}
        collection = "jobs" if intent["mode"] == "worker" else "auxiliary_jobs"
        assert not any(row["job_id"] == record["job_id"] for row in campaign[collection])
        campaign[collection].append(record)
        campaign["ambiguous_submissions"] = []
        save(ARTIFACTS / "campaign.json", campaign)
        print(json.dumps(record, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["preflight", "prepare", "worker", "inspect", "cleanup", "reconcile", "recover"])
    parser.add_argument("--worker-id", type=int, choices=range(len(TASKS)))
    args = parser.parse_args()
    if args.mode == "recover":
        recover()
        return
    if args.mode == "worker" and args.worker_id is None:
        parser.error("Worker mode requires --worker-id")
    with locked(ARTIFACTS / "submission.lock"):
        campaign = json.loads((ARTIFACTS / "campaign.json").read_text())
        if campaign.get("ambiguous_submissions"):
            raise RuntimeError("Recover the unique pending submission note before creating another job")
        if args.mode == "worker":
            assert campaign.get("review_passed") and campaign.get("assets_ready_verified")
            assert campaign.get("runtime_bundle_sha256")
            assert not any(row["worker"] == args.worker_id for row in campaign["jobs"])
            assert len(campaign["jobs"]) < len(TASKS) <= campaign["gpu_limit"] == 10
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
            confirmation = json.loads((ARTIFACTS / "cleanup-authorization.json").read_text())
            assert confirmation["verified_runs"] == len(TASKS)
            environment.update(encode_audit_environment("REPAIR_FINAL_AUDIT", confirmation))
        number = len(campaign["jobs"]) + len(campaign["auxiliary_jobs"])
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
        campaign["ambiguous_submissions"] = [pending]
        save(ARTIFACTS / "campaign.json", campaign)
        remote = "import subprocess,json\ncommand=" + repr(command) + "\n"
        remote += "d=subprocess.run(command+['--dry-run'],capture_output=True,text=True,timeout=50)\n"
        remote += "assert d.returncode==0\n"
        remote += "assert json.loads(d.stdout)['logs_dir']=='/home/jovyan/shares/SR006.nfs3/mlsub-logs'\n"
        remote += "r=subprocess.run(command,capture_output=True,text=True,timeout=90)\n"
        remote += "out=r.stdout+r.stderr\n"
        remote += "for secret in " + repr(secrets) + ":out=out.replace(secret,'<redacted>')\n"
        remote += "print(out)\nraise SystemExit(r.returncode)\n"
        try:
            result = subprocess.run(SSH + ["python3", "-"], input=remote,
                                    capture_output=True, text=True, timeout=155)
        except subprocess.TimeoutExpired:
            raise RuntimeError("Submission response lost; adopt exact unique note, never retry blindly") from None
        output = result.stdout + result.stderr
        for secret in secrets:
            output = output.replace(secret, "<redacted>")
        (ARTIFACTS / ("submission-" + intent + ".log")).write_text(output)
        ids = set(re.findall(r"lm-mpi-job-[a-f0-9-]+", output))
        if len(ids) != 1:
            raise RuntimeError("No unique submission ID; recover exact note from scheduler")
        record = {**pending, "job_id": ids.pop(), "transport_returncode": result.returncode,
                  "submitted_at": now()}
        campaign["ambiguous_submissions"] = []
        collection = "jobs" if args.mode == "worker" else "auxiliary_jobs"
        campaign[collection].append(record)
        save(ARTIFACTS / "campaign.json", campaign)
        print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
