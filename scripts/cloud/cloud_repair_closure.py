"""Permanently close completed claims without claiming unknown allocations ended.

CPU-only closure is a prerequisite for a separately reviewed asset-isolation
path. It is never cleanup authorization and never reads assets or CUDA state.
"""

from contextlib import contextmanager, ExitStack
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import stat

from cloud_repair_common import BUNDLE_SHA, CAMPAIGN, STATE, TASKS, now, require_space, task_key


SHA256 = re.compile(r"^[a-f0-9]{64}$")
REVISION = re.compile(r"^[a-f0-9]{40}$")
JOB = re.compile(r"^lm-mpi-job-[a-f0-9-]+$")
TERMINAL = {"completed", "failed", "stopped", "cancelled", "canceled"}
MAX_BYTES = 1_000_000
RECONCILIATION_FIELDS = {"recovery_logs", "syncphase", "independently_verified_at"}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def canonical_sha(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def canonical(path: Path) -> None:
    require(path.is_absolute() and not path.is_symlink() and path.resolve() == path,
            "Closure evidence path is linked or noncanonical")


def read(path: Path) -> tuple[dict, str]:
    canonical(path)
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    try:
        before = os.fstat(descriptor)
        require(stat.S_ISREG(before.st_mode) and before.st_size <= MAX_BYTES, "Invalid closure evidence file")
        with os.fdopen(descriptor, "rb", closefd=False) as stream:
            data = stream.read(MAX_BYTES + 1)
        after = os.fstat(descriptor)
        require(len(data) <= MAX_BYTES and (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
                == (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns), "Closure evidence changed while reading")
        current = path.lstat()
        require((current.st_dev, current.st_ino) == (before.st_dev, before.st_ino)
                and not stat.S_ISLNK(current.st_mode), "Closure evidence path changed")
        value = json.loads(data)
        require(isinstance(value, dict), "Closure evidence is not an object")
        return value, hashlib.sha256(data).hexdigest()
    finally:
        os.close(descriptor)


@contextmanager
def completed_claim_locks():
    """An existing parent holds its worker lock throughout profile and main."""
    canonical(STATE)
    with ExitStack() as stack:
        for path in [STATE / "workers" / f"{worker}.lock" for worker in range(len(TASKS))] + [STATE / "claims.lock"]:
            canonical(path)
            descriptor = os.open(path, os.O_RDWR | os.O_NOFOLLOW)
            stack.callback(os.close, descriptor)
            require(stat.S_ISREG(os.fstat(descriptor).st_mode), "Closure lock is not a regular file")
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield


def valid_sha(value) -> bool:
    return isinstance(value, str) and bool(SHA256.fullmatch(value))


def reviewed_proof(request: dict) -> dict:
    require(isinstance(request, dict) and isinstance(request.get("proof"), dict), "Structured closure proof required")
    proof, review = request["proof"], request.get("review", {})
    require(proof.get("campaign") == CAMPAIGN and proof.get("runtime_bundle_sha256") == BUNDLE_SHA.read_text().strip()
            and valid_sha(proof["runtime_bundle_sha256"]), "Closure campaign/bundle differs")
    require(review.get("decision") == "approved_for_campaign_closure_only"
            and review.get("proof_sha256") == canonical_sha(proof) and valid_sha(review.get("review_sha256")),
            "Exact reviewed closure request proof is missing")
    scientific = proof.get("scientific_verification", {})
    require(set(scientific) == {task_key(task) for task in TASKS}, "Exactly eight scientific proofs required")
    for worker, task in enumerate(TASKS):
        row = scientific[task_key(task)]
        require(isinstance(row, dict) and row.get("worker") == worker and type(row.get("worker")) is int
                and bool(re.fullmatch(r"[A-Za-z0-9_-]+", row.get("wandb_id", "")))
                and row.get("wandb_path") == "andrey/LoRA-Muon/" + row["wandb_id"], "Scientific W&B identity differs")
        require(type(row.get("returncode")) is int and row["returncode"] == 0
                and row.get("train_records") == 3000 and row.get("validation_points") == 12,
                "Scientific result is incomplete")
        require(all(valid_sha(row.get(key)) for key in ("diagnostic_manifest_sha256", "independent_verification_sha256",
                    "history_sha256", "result_sha256", "diagnostic_inventory_sha256")), "Scientific digest proof is incomplete")
    require(valid_sha(proof.get("exports", {}).get("manifest_sha256")), "Verified export manifest proof required")
    allocations = proof.get("allocation_proof", {})
    require(valid_sha(allocations.get("ledger_sha256")), "Owned allocation ledger proof required")
    require(allocations.get("all_owned_jobs_terminal") is not True
            and ("all_known_jobs_terminal" not in allocations or allocations["all_known_jobs_terminal"] is True),
            "Unknown requests cannot be relabeled all-terminal allocations")
    known, unknown = allocations.get("known_terminal"), allocations.get("unknown_requests")
    require(isinstance(known, list) and bool(known) and isinstance(unknown, list) and bool(unknown),
            "Distinct known-terminal and retained unknown requests required")
    known_ids = allocations.get("known_allocation_ids", [])
    require(isinstance(known_ids, list) and len(known_ids) == len(set(known_ids))
            and len(known) == len(known_ids) and {row.get("job_id") for row in known} == set(known_ids),
            "Known terminal allocation coverage differs")
    notes = set()
    for row in known:
        require(bool(JOB.fullmatch(row.get("job_id", ""))) and isinstance(row.get("note"), str) and bool(row["note"])
                and str(row.get("status")).lower() in TERMINAL and valid_sha(row.get("evidence_sha256")),
                "Known allocation lacks terminal identity/evidence")
        require(row["note"] not in notes, "Duplicate owned allocation note")
        notes.add(row["note"])
    unknown_workers, cpu_count = set(), 0
    gates = proof.get("source_gate_proof", {})
    require(valid_sha(gates.get("review_sha256")) and valid_sha(gates.get("published_history_sha256"))
            and bool(REVISION.fullmatch(gates.get("current_revision", ""))), "Reviewed published source gate proof required")
    sources = gates.get("sources", [])
    require(isinstance(sources, list) and bool(sources), "Source gate inventory required")
    by_revision, by_source = {}, {}
    for source in sources:
        revision = source.get("source_revision", "")
        path = source.get("path")
        identity = (revision, path)
        require(bool(REVISION.fullmatch(revision)) and identity not in by_source
                and path in {"scripts/cloud/cloud_repair_runtime.py", "scripts/cloud/cloud_repair_entry.py",
                             "scripts/cloud/cloud_repair_control.py"}
                and valid_sha(source.get("source_sha256")) and valid_sha(source.get("review_sha256"))
                and isinstance(source.get("gates"), list), "Invalid reviewed source gate entry")
        by_source[identity] = set(source["gates"])
        by_revision.setdefault(revision, set()).update(source["gates"])
    require({"worker_closed_before_assets_cuda", "profile_closed_before_assets_cuda"}
            <= by_source.get((gates["current_revision"], "scripts/cloud/cloud_repair_runtime.py"), set()),
            "Current worker/profile closure gates are not reviewed")
    for row in unknown:
        require(isinstance(row.get("note"), str) and bool(row["note"]) and row["note"] not in notes
                and valid_sha(row.get("evidence_sha256")) and bool(REVISION.fullmatch(row.get("source_revision", ""))),
                "Unknown request identity/evidence is incomplete")
        require(not row.get("job_id"), "Unknown request must not be relabeled a terminal allocation")
        notes.add(row["note"])
        reviewed = by_source.get((row["source_revision"], "scripts/cloud/cloud_repair_runtime.py"), set())
        if row.get("mode") == "worker":
            worker = row.get("worker")
            require(type(worker) is int and worker in (0, 1) and worker not in unknown_workers
                    and type(row.get("gpus")) is int and row["gpus"] == 1
                    and "complete_claim_before_assets_cuda" in reviewed, "Unknown worker source gate/scope differs")
            unknown_workers.add(worker)
        else:
            require(row.get("mode") == "inspect" and type(row.get("gpus")) is int and row["gpus"] == 0
                    and row.get("worker") is None
                    and all("read_only_metadata" in by_source.get((row["source_revision"], path), set())
                            for path in ("scripts/cloud/cloud_repair_entry.py", "scripts/cloud/cloud_repair_control.py")),
                    "Unknown CPU request is not reviewed read-only inspection")
            cpu_count += 1
    require(cpu_count <= 4, "Unknown inspection scope exceeds reviewed isolation limit")
    return proof


def reconciliation_metadata(record: dict, key: str) -> None:
    pending = str(STATE / "pending-wandb-sync" / key)
    archived = str(STATE / "verified-raw-logs" / key)
    if record.get("syncphase") == "uploaded_awaiting_independent_verification":
        require(record.get("recovery_logs") == pending and "independently_verified_at" not in record,
                "Unexpected pending reconciliation metadata")
    else:
        require(record.get("syncphase") == "independently_verified" and record.get("recovery_logs") == archived
                and isinstance(record.get("independently_verified_at"), str), "Unexpected completed reconciliation metadata")
        from datetime import datetime
        moment = datetime.fromisoformat(record["independently_verified_at"].replace("Z", "+00:00"))
        require(moment.tzinfo is not None, "Reconciliation timestamp lacks timezone")


def completion(record: dict, task: dict, worker: int, proof: dict) -> dict:
    require(record.get("task") == task and type(record.get("worker")) is int and record["worker"] == worker
            and record.get("phase") == "complete" and type(record.get("returncode")) is int and record["returncode"] == 0
            and record.get("child_pid") is None and record.get("wandb_sync_complete") is True,
            "Task/worker is not complete and synchronized")
    require(record.get("wandb_id") == proof["wandb_id"]
            and record.get("recovery_manifest_sha256") == proof["diagnostic_manifest_sha256"],
            "Task/worker independent W&B manifest identity differs")
    verification = record.get("verification")
    require(isinstance(verification, dict) and verification.get("train_records") == 3000
            and verification.get("validation_points") == 12, "Recorded training/FINAL proof is incomplete")
    reconciliation_metadata(record, task_key(task))
    return {"task": task, "worker": worker, "phase": "complete", "returncode": 0, "child_pid": None,
            "wandb_sync_complete": True, "wandb_id": record["wandb_id"], "wandb_path": proof["wandb_path"],
            "train_records": 3000, "validation_points": 12,
            "diagnostic_manifest_sha256": record["recovery_manifest_sha256"]}


def complete_records(proof: dict) -> dict:
    tasks, workers = STATE / "tasks", STATE / "workers"
    canonical(tasks)
    canonical(workers)
    require({path.name for path in tasks.glob("*.json")} == {task_key(task) + ".json" for task in TASKS}
            and {path.name for path in workers.glob("*.json")} == {str(worker) + ".json" for worker in range(len(TASKS))},
            "Closure task/worker record coverage differs")
    result = {}
    for worker, task in enumerate(TASKS):
        key = task_key(task)
        pair = {}
        for kind, path in (("task", tasks / (key + ".json")), ("worker", workers / f"{worker}.json")):
            record, file_sha = read(path)
            projection = completion(record, task, worker, proof["scientific_verification"][key])
            pair[kind] = {"file_sha256": file_sha,
                          "stable_sha256": canonical_sha({key: value for key, value in record.items()
                                                          if key not in RECONCILIATION_FIELDS}),
                          "completion": projection}
        require(pair["task"]["completion"] == pair["worker"]["completion"], "Task/worker completion projections differ")
        result[key] = pair
    return result


def marker_payload(request: dict, records: dict) -> dict:
    proof = request["proof"]
    return {"campaign": CAMPAIGN, "runtime_bundle_sha256": proof["runtime_bundle_sha256"],
            "closure_only": True, "cleanup_authorization": False, "all_owned_jobs_terminal": False,
            "reviewed_request_sha256": canonical_sha(request), "proof_sha256": canonical_sha(proof),
            "request": request, "records": records}


def read_marker() -> dict:
    marker, _ = read(STATE / "campaign-closed.json")
    require(marker.get("campaign") == CAMPAIGN and marker.get("closure_only") is True
            and marker.get("cleanup_authorization") is False and marker.get("all_owned_jobs_terminal") is False,
            "Invalid permanent closure marker")
    payload = {key: value for key, value in marker.items() if key != "payload_sha256"}
    require(marker.get("payload_sha256") == canonical_sha(payload), "Permanent closure marker digest differs")
    proof = reviewed_proof(marker["request"])
    require(marker.get("reviewed_request_sha256") == canonical_sha(marker["request"])
            and marker.get("proof_sha256") == canonical_sha(proof)
            and marker.get("runtime_bundle_sha256") == proof["runtime_bundle_sha256"], "Closure request binding differs")
    return marker


def verify_current(marker: dict, current: dict) -> None:
    require(set(marker["records"]) == set(current), "Permanent closure record coverage differs")
    for key, pair in current.items():
        for kind in ("task", "worker"):
            require(marker["records"][key][kind]["stable_sha256"] == pair[kind]["stable_sha256"]
                    and marker["records"][key][kind]["completion"] == pair[kind]["completion"],
                    "Permanent completed claim changed outside reconciliation metadata")


def write_marker(value: dict) -> None:
    path = STATE / "campaign-closed.json"
    canonical(path)
    data = (json.dumps(value, indent=2, allow_nan=False) + "\n").encode()
    require(len(data) <= MAX_BYTES, "Permanent closure marker is oversized")
    require_space([STATE], {STATE: len(data)})
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
    parent = os.open(STATE, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        os.fsync(parent)
    finally:
        os.close(parent)


def close_campaign(request: dict) -> dict:
    proof = reviewed_proof(request)
    with completed_claim_locks():
        records = complete_records(proof)
        path = STATE / "campaign-closed.json"
        if path.exists() or path.is_symlink():
            marker = read_marker()
            require(marker["reviewed_request_sha256"] == canonical_sha(request), "Existing closure request differs")
            verify_current(marker, records)
            return marker
        payload = marker_payload(request, records)
        marker = {**payload, "created_at": now()}
        marker["payload_sha256"] = canonical_sha(marker)
        write_marker(marker)
        return marker


def validate_closure(request: dict | None = None) -> dict:
    with completed_claim_locks():
        marker = read_marker()
        if request is not None:
            reviewed_proof(request)
            require(marker["reviewed_request_sha256"] == canonical_sha(request), "Closure request differs")
        verify_current(marker, complete_records(marker["request"]["proof"]))
        return marker


def assert_campaign_open() -> None:
    """Call before worker/profile assets or GPU discovery; every marker blocks."""
    canonical(STATE)
    path = STATE / "campaign-closed.json"
    if path.exists() or path.is_symlink():
        # Invalid/partial/link markers also fail closed; never repair or remove.
        read_marker()
        raise RuntimeError("Campaign namespace is permanently closed before assets/CUDA")
