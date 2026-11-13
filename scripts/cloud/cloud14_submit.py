"""Submit one recorded campaign allocation without printing or persisting credentials."""

import argparse
import fcntl
import json
import netrc
from pathlib import Path
import re
import subprocess

from cloud14_core import now, save

WORKSPACE = Path(__file__).resolve().parents[4]
ARTIFACTS = WORKSPACE / "artifacts/paper_8b_three_seeds_cloud_20261113"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["prepare", "pair", "inspect", "preflight", "reset-unstarted", "cleanup"])
    parser.add_argument("--pair-id", type=int, choices=range(7))
    args = parser.parse_args()
    if args.mode == "pair" and args.pair_id is None:
        parser.error("Pair mode requires --pair-id")
    campaign_path = ARTIFACTS / "campaign.json"
    with (ARTIFACTS / "submission.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        campaign = json.loads(campaign_path.read_text())
        if args.mode == "pair":
            assert campaign["review_status"] == "passed"
            assert campaign.get("assets_ready_verified") is True
            assert not any(job["pair_id"] == args.pair_id for job in campaign["jobs"])
        intent = f"{args.mode}-{args.pair_id if args.pair_id is not None else len(campaign['preparation_jobs'])}"
        if args.mode == "pair":
            intent += "-attempt" + str(len(campaign.get("allocation_history", [])) + 1)
        assert intent not in campaign.get("ambiguous_submissions", [])
        environment = {"CUDA_DEVICE_ORDER": "PCI_BUS_ID", "WANDB_BASE_URL": "https://wandb-radfan.ru",
                       "WANDB_ENTITY": "andrey", "WANDB_PROJECT": "LoRA-Muon",
                       "PYTHONDONTWRITEBYTECODE": "1",
                       "PAPER8B_BUNDLE_SHA256": campaign["runtime_bundle_sha256"]}
        if args.mode == "reset-unstarted":
            audit = json.loads((ARTIFACTS / "startup-recovery-audit.json").read_text())
            assert audit["new_wandb_runs"] == 0
            environment["PAPER8B_STARTUP_RECOVERY_AUDIT"] = json.dumps(audit)
        if args.mode == "cleanup":
            audit = json.loads((ARTIFACTS / "final-cleanup-authorization.json").read_text())
            assert audit["verified_runs"] == 180 and audit["all_owned_jobs_terminal"] is True
            assert audit["wandb_sync_verified"] is True and audit["results_export_verified"] is True
            environment["PAPER8B_FINAL_CLEANUP_AUDIT"] = json.dumps(audit)
        secrets = []
        if args.mode in ["prepare", "pair"]:
            key = netrc.netrc().authenticators("wandb-radfan.ru")[2]
            environment["WANDB_API_KEY"] = key
            secrets.append(key)
        if args.mode == "prepare":
            inventory = Path.home() / "Documents/Obsidian Vault/🧠 BRAIn Lab/general/servers.md"
            key = re.search(r"HF token \(llama 3\.1\):\s*`([^`]+)`", inventory.read_text()).group(1)
            environment["HF_TOKEN"] = key
            secrets.append(key)
        run_args = args.mode + (f" --pair-id {args.pair_id}" if args.mode == "pair" else "")
        command = ["mlsub", "run", "--repo", "https://github.com/dimativator/LoRA-TSD", "--branch", "dev",
                   "--entry", "scripts/cloud/cloud14_entry.py", "--args", run_args,
                   "--gpus", "2" if args.mode == "pair" else "cpu", "--image", "torch28", "--no-pip",
                   "--note", f"paper8b_3seed_{intent}_20261113"]
        for key, value in environment.items():
            command.extend(["--env", key + "=" + value])
        remote = "import subprocess\nr=subprocess.run(" + repr(command) + ",capture_output=True,text=True)\nout=r.stdout+r.stderr\n"
        remote += "for secret in " + repr(secrets) + ":out=out.replace(secret,'<redacted>')\n"
        remote += "print(out)\nraise SystemExit(r.returncode)\n"
        campaign.setdefault("ambiguous_submissions", []).append(intent)
        save(campaign_path, campaign)
        try:
            result = subprocess.run(["ssh", "-o", "ClearAllForwardings=yes", "-o", "BatchMode=yes",
                                     "-o", "ConnectTimeout=25", "brain_lab", "python3", "-"],
                                    input=remote, capture_output=True, text=True, timeout=240)
        except subprocess.TimeoutExpired:
            raise RuntimeError(f"Submission{intent} is ambiguous. Inspect scheduler by unique note; do not retry blindly.") from None
        output = result.stdout + result.stderr
        for secret in secrets:
            output = output.replace(secret, "<redacted>")
        output = re.sub(r"hf_[A-Za-z0-9]+", "<redacted>", output)
        print(output)
        (ARTIFACTS / f"submission-{intent}.log").write_text(output)
        ids = re.findall(r"lm-mpi-job-[a-f0-9-]+", output)
        if not ids:
            if result.returncode:
                raise RuntimeError("Submission failed; review redacted log before clearing intent")
            raise RuntimeError("No job ID; inspect scheduler before retrying")
        record = {"mode": args.mode, "job_id": ids[-1], "submitted_at": now(),
                  "source_revision": campaign["source_revision"], "environment_names": sorted(environment)}
        if args.mode == "pair":
            record.update(pair_id=args.pair_id, workers=[2 * args.pair_id, 2 * args.pair_id + 1], gpus=2)
            campaign["jobs"].append(record)
        else:
            campaign["preparation_jobs"].append(record)
        campaign["ambiguous_submissions"].remove(intent)
        campaign["status"] = "gpu_allocations_submitted" if args.mode == "pair" else args.mode + "_submitted"
        save(campaign_path, campaign)


if __name__ == "__main__":
    main()
