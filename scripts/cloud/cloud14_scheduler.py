"""Bounded read-only scheduler snapshot of the campaign's current allocations."""

import json
import subprocess

from cloud14_core import now, save
from cloud14_submit import ARTIFACTS

TERMINAL = {"completed", "failed", "stopped", "cancelled", "canceled"}


def snapshot() -> dict:
    campaign = json.loads((ARTIFACTS / "campaign.json").read_text())
    jobs = [row["job_id"] for row in campaign["jobs"]]
    remote = """import concurrent.futures,json,subprocess
jobs=JOBS
def read(job):
 r=subprocess.run(['mlsub','status',job],capture_output=True,text=True,timeout=60)
 r.check_returncode()
 value=json.loads(r.stdout)
 assert value['job_name']==job
 return {'job_id':job,'status':value['status'],'details':value}
with concurrent.futures.ThreadPoolExecutor(max_workers=7) as pool:
 print(json.dumps(list(pool.map(read,jobs))))
""".replace("JOBS", repr(jobs))
    result = subprocess.run(["ssh", "-o", "ClearAllForwardings=yes", "-o", "BatchMode=yes",
                             "-o", "ConnectTimeout=25", "brain_lab", "python3", "-"],
                            input=remote, capture_output=True, text=True, timeout=100)
    result.check_returncode()
    rows = json.loads(result.stdout)
    assert {row["job_id"] for row in rows} == set(jobs)
    value = {"timestamp": now(), "jobs": rows}
    save(ARTIFACTS / "scheduler-snapshot.json", value)
    return value


if __name__ == "__main__":
    print(json.dumps(snapshot(), indent=2))
