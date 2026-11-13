"""Collect a bounded, chunked CPU inspection without printing full histories."""

import argparse
import json
from pathlib import Path
import subprocess
import sys

from cloud14_core import decode_snapshot, locked, save
from cloud14_submit import ARTIFACTS


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--submit", action="store_true")
    parser.add_argument("--job-id")
    args = parser.parse_args()
    if args.submit:
        subprocess.run([sys.executable, str(Path(__file__).with_name("cloud14_submit.py")), "inspect"], check=True)
    campaign = json.loads((ARTIFACTS / "campaign.json").read_text())
    job = args.job_id or next(row["job_id"] for row in reversed(campaign["preparation_jobs"])
                             if row["mode"] == "inspect")
    remote = "import subprocess\nr=subprocess.run(" + repr(["mlsub", "logs", job, "--tail", "80"])
    remote += ",capture_output=True,text=True,timeout=90)\nprint(r.stdout+r.stderr)\nraise SystemExit(r.returncode)\n"
    result = subprocess.run(["ssh", "-o", "ClearAllForwardings=yes", "-o", "BatchMode=yes",
                             "-o", "ConnectTimeout=25", "brain_lab", "python3", "-"],
                            input=remote, capture_output=True, text=True, timeout=120)
    text = result.stdout + result.stderr
    (ARTIFACTS / "inspection-latest.log").write_text(text)
    result.check_returncode()
    snapshot = decode_snapshot(text)
    save(ARTIFACTS / "inspection-latest.json", snapshot)
    with locked(ARTIFACTS / "submission.lock"):
        campaign = json.loads((ARTIFACTS / "campaign.json").read_text())
        campaign.update(assets_ready_verified=snapshot["assets_ready"], latest_inspection="inspection-latest.json",
                        storage_latest=snapshot["filesystems"])
        save(ARTIFACTS / "campaign.json", campaign)
    print(json.dumps({"timestamp": snapshot["timestamp"], "assets_ready": snapshot["assets_ready"],
          "filesystems": [(row["requested_path"], row["available_bytes"]) for row in snapshot["filesystems"]],
          "tasks": [{key: row.get(key) for key in ["worker", "phase", "step", "microbatch", "child_pid",
                                                     "evaluation_count", "best_accuracy", "timestamp"]}
                    for row in snapshot["tasks"]], "workers": snapshot["workers"]}, indent=2))


if __name__ == "__main__":
    main()
