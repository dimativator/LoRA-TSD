"""Local asset-isolation proof, retaining unanswered allocation uncertainty.

This module neither edits the campaign nor authorizes deletion. It revalidates
completed results and known allocations, and constructs a separately reviewed
closure request for the CPU closure helper.
"""

from copy import deepcopy
import hashlib
import json
from pathlib import Path, PurePosixPath
import re

from cloud_repair_common import CAMPAIGN, TASKS, digest, run_name, task_key
from cloud_repair_submit import REPLACEABLE_NOTES, USER_OVERRIDE, cpu_inspection, gpu_intent


REVISION = re.compile(r"^[a-f0-9]{40}$")
SHA256 = re.compile(r"^[a-f0-9]{64}$")
ORIGINAL_SOURCES = {0: "da3ac4a0debbb17373fe374c9113158fe5bee0ac",
                    1: "22ef29ac655b2a68bd6b65474219b34a199ec898"}
SOURCE_REVIEW_DECISION = "approved_source_gate_asset_isolation"
CLOSURE_REVIEW_DECISION = "approved_for_campaign_closure_only"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def canonical_sha(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def evidence_file(path: Path, artifacts: Path, expected: str | None = None) -> Path:
    require(path.is_absolute() and path.resolve() == path and not path.is_symlink()
            and path.is_file() and path.is_relative_to(artifacts.resolve()),
            "Evidence is missing, linked or outside repair artifacts")
    if expected is not None:
        require(isinstance(expected, str) and bool(SHA256.fullmatch(expected))
                and digest(path) == expected, "Evidence SHA differs")
    return path


def relative_file(value: str, artifacts: Path, expected: str) -> Path:
    path = PurePosixPath(value)
    require(not path.is_absolute() and path.parts and all(part not in (".", "..") for part in path.parts),
            "Evidence relative path escapes repair artifacts")
    return evidence_file(artifacts / str(path), artifacts, expected)


def read(path: Path, artifacts: Path) -> dict:
    return json.loads(evidence_file(path, artifacts).read_text())


def unknown_ledger(campaign: dict, artifacts: Path) -> dict:
    require(campaign.get("campaign") == CAMPAIGN, "Unknown ledger belongs to another campaign")
    require(not campaign.get("ambiguous_submissions"), "Resolve all current submission intents first")
    gpu = deepcopy(campaign.get("superseded_unreceived_gpu_intents", []))
    cpu = deepcopy(campaign.get("unresolved_auxiliary_submissions", []))
    require(len(gpu) <= 2 and len(cpu) <= 4, "Unknown request scope exceeds bounded allowance")
    requests = []
    for record in gpu + cpu:
        require(not any(record.get(key) for key in ("job_id", "observed_job_ids", "recovery_conflict")),
                "An unknown request has positive identity or conflict evidence")
        require(isinstance(record.get("source_revision"), str)
                and bool(REVISION.fullmatch(record["source_revision"])), "Unknown source revision is invalid")
        if record in gpu:
            worker = record.get("worker")
            require(gpu_intent(record) and worker in ORIGINAL_SOURCES
                    and record.get("note") == REPLACEABLE_NOTES[worker]
                    and record["source_revision"] == ORIGINAL_SOURCES[worker]
                    and record.get("original_still_reserves_gpu") is True
                    and record.get("user_override") == USER_OVERRIDE,
                    "Only the two exact original GPU requests may be isolated")
            path = evidence_file(Path(record.get("submission_evidence", "")), artifacts,
                                 record.get("submission_evidence_sha256"))
        else:
            require(cpu_inspection(record) and isinstance(record.get("intent"), str)
                    and bool(re.fullmatch(r"inspect-[0-9]+", record["intent"]))
                    and isinstance(record.get("note"), str)
                    and bool(re.fullmatch(r"repair8b_" + re.escape(record["intent"]) + r"_[A-Za-z0-9_.-]+", record["note"])),
                    "Only exact read-only zero-GPU inspection requests may be isolated")
            path = evidence_file(artifacts / ("submission-" + record["intent"] + ".log"), artifacts)
        requests.append({key: record[key] for key in ("note", "mode", "worker", "gpus", "source_revision")}
                        | {"evidence_sha256": digest(path)})
    notes = [row["note"] for row in requests]
    require(len(notes) == len(set(notes)) and len({row["worker"] for row in gpu}) == len(gpu),
            "Unknown request notes or GPU workers are duplicated")
    from cloud_repair_collect import owned_records
    require(not any(row.get("note") in notes for row in owned_records(campaign)),
            "A supposedly unknown request already has a known receipt")
    canonical = {"superseded_unreceived_gpu_intents": sorted(gpu, key=lambda row: row["note"]),
                 "unresolved_auxiliary_submissions": sorted(cpu, key=lambda row: row["note"])}
    return {"unknown_requests": sorted(requests, key=lambda row: row["note"]),
            "canonical_unknown_ledger": canonical, "unknown_ledger_sha256": canonical_sha(canonical)}


def verify_source_review(path: Path, artifacts: Path, campaign: dict, unknown: dict) -> dict:
    envelope = read(path, artifacts)
    proof, review = envelope["proof"], envelope["review"]
    require(review.get("decision") == SOURCE_REVIEW_DECISION
            and review.get("proof_sha256") == canonical_sha(proof)
            and review.get("independent") is True
            and isinstance(proof.get("prepared_by"), str) and bool(proof["prepared_by"].strip())
            and isinstance(review.get("reviewer"), str) and bool(review["reviewer"].strip())
            and review["reviewer"] != proof["prepared_by"],
            "Source gates lack an independent digest-bound approval")
    require(proof.get("campaign") == CAMPAIGN
            and proof.get("audit_kind") == "cpu_metadata_only_no_model_or_cuda"
            and proof.get("current_revision") == campaign.get("source_revision")
            and bool(REVISION.fullmatch(proof.get("current_revision", "")))
            and proof.get("unknown_requests") == unknown["unknown_requests"],
            "Source review is not bound to exact requests/current published revision")
    history = proof["published_history"]
    history_path = relative_file(history["path"], artifacts, history["sha256"])
    history_record = json.loads(history_path.read_text())
    revisions = history_record.get("published_revisions", [])
    require(history_record.get("campaign") == CAMPAIGN
            and history_record.get("complete_published_history") is True
            and history_record.get("ordering") == "oldest_to_newest"
            and isinstance(revisions, list) and bool(revisions) and revisions[-1] == proof["current_revision"]
            and len(revisions) == len(set(revisions))
            and all(isinstance(revision, str) and bool(REVISION.fullmatch(revision)) for revision in revisions)
            and {row["source_revision"] for row in unknown["unknown_requests"]} | {proof["current_revision"]} <= set(revisions),
            "Published gate history does not cover all unknown/current revisions")
    sources, indexed = [], {}
    for row in proof["sources"]:
        require(bool(REVISION.fullmatch(row.get("source_revision", "")))
                and row.get("path") in {"scripts/cloud/cloud_repair_runtime.py",
                    "scripts/cloud/cloud_repair_entry.py", "scripts/cloud/cloud_repair_control.py"}
                and isinstance(row.get("gates"), list), "Source blob identity is invalid")
        relative_file(row["source_path"], artifacts, row["sha256"])
        identity = (row["source_revision"], row["path"])
        require(identity not in indexed, "Source review repeats a blob identity")
        indexed[identity] = row
        sources.append({"source_revision": row["source_revision"], "path": row["path"],
                        "source_sha256": row["sha256"], "review_sha256": digest(path),
                        "gates": row["gates"]})
    for request in unknown["unknown_requests"]:
        possible_clones = revisions[revisions.index(request["source_revision"]):]
        for revision in possible_clones:
            if request["mode"] == "worker":
                gates = set(indexed.get((revision, "scripts/cloud/cloud_repair_runtime.py"), {}).get("gates", []))
                require(bool({"complete_claim_before_assets_cuda", "worker_closed_before_assets_cuda"} & gates),
                        "A possible late worker clone lacks its reviewed pre-asset/CUDA gate")
            else:
                require(all("read_only_metadata" in indexed.get((revision, path), {}).get("gates", [])
                            for path in ("scripts/cloud/cloud_repair_entry.py", "scripts/cloud/cloud_repair_control.py")),
                        "A possible late inspector clone lacks both reviewed read-only source gates")
    current = proof["current_revision"]
    current_gates = indexed.get((current, "scripts/cloud/cloud_repair_runtime.py"), {}).get("gates", [])
    require({"worker_closed_before_assets_cuda", "profile_closed_before_assets_cuda"} <= set(current_gates),
            "Current published source lacks both closure gates")
    return {"review_sha256": digest(path), "published_history_sha256": history["sha256"],
            "current_revision": current, "sources": sources}


def export_manifest(artifacts: Path) -> str:
    manifest_path = artifacts / "final/artifact-manifest.json"
    manifest = read(manifest_path, artifacts)
    required = {"paired-results.csv", "paired-results.json", "validation-curves.csv",
                "validation-curves.json", "validation-curves.svg", "raw-histories.json"}
    require(set(manifest) == required, "Exact final result/curve/history exports are required")
    for name, row in manifest.items():
        path = evidence_file(artifacts / "final" / name, artifacts, row["sha256"])
        require(type(row.get("bytes")) is int and path.stat().st_size == row["bytes"], "Export size differs")
    return digest(manifest_path)


def build_proof(artifacts: Path, source_review_path: Path) -> dict:
    """Read and revalidate every structured input, without changing any ledger/file."""
    from cloud_repair_collect import owned_records
    from cloud_repair_finalize import TERMINAL, verify_inspection, verify_local, verify_scheduler
    campaign = read(artifacts / "campaign.json", artifacts)
    require(campaign.get("tasks") == TASKS and campaign.get("target_runs") == 8
            and bool(SHA256.fullmatch(campaign.get("runtime_bundle_sha256", ""))), "Repair campaign scope differs")
    jobs = campaign.get("jobs", [])
    require(len(jobs) == 8 and {row.get("worker") for row in jobs} == set(range(8))
            and all(gpu_intent(row) for row in jobs), "Eight exact current worker receipts are required")
    unknown = unknown_ledger(campaign, artifacts)
    require(bool(unknown["unknown_requests"]), "Use ordinary finalization when all requests have receipts")
    source = verify_source_review(source_review_path, artifacts, campaign, unknown)
    verification_path = artifacts / "wandb-independent-verification.json"
    verified = read(verification_path, artifacts)
    verify_local(verified, campaign, artifacts)
    inspection = read(artifacts / "inspection-latest.json", artifacts)
    manifests = verify_inspection(inspection, verified, campaign, artifacts)
    scheduler = read(artifacts / "scheduler-snapshot.json", artifacts)
    known_campaign = deepcopy(campaign)
    for key in ("superseded_unreceived_gpu_intents", "unresolved_auxiliary_submissions"):
        known_campaign[key] = []
    verify_scheduler(known_campaign, scheduler, inspection, artifacts)
    require(all(str(row.get("status")).lower() in TERMINAL for row in scheduler["jobs"]),
            "Every known allocation, including current inspector, must be terminal")
    # Positive receipts discovered for an unknown note must be reconciled, never hidden.
    raw = evidence_file(Path(scheduler["raw_proof"]), artifacts, scheduler["raw_proof_sha256"]).read_text()
    require(not any("|note:" + row["note"] in raw for row in unknown["unknown_requests"]),
            "Fresh scheduler evidence identifies a supposedly unknown request")
    ledger = owned_records(campaign)
    known = []
    for row in scheduler["jobs"]:
        records = [item for item in ledger if item["job_id"] == row["job_id"]]
        require(records and all({key: item.get(key) for key in ("note", "mode", "worker", "gpus")}
                == {key: records[0].get(key) for key in ("note", "mode", "worker", "gpus")} for item in records),
                "Known allocation metadata differs across its ledger records")
        known.append({**{key: records[0].get(key) for key in ("note", "mode", "worker", "gpus")},
            "job_id": row["job_id"], "status": str(row["status"]).lower(),
            "evidence_sha256": scheduler["raw_proof_sha256"]})
    scientific = {}
    for worker, task in enumerate(TASKS):
        name, key = run_name(task), task_key(task)
        directory = artifacts / "raw-wandb" / name
        row = verified[name]
        scientific[key] = {"worker": worker, "wandb_id": row["id"], "wandb_path": row["path"],
            "train_records": row["train_records"], "validation_points": row["validation_points"], "returncode": 0,
            "diagnostic_manifest_sha256": manifests[key], "independent_verification_sha256": digest(verification_path),
            "history_sha256": digest(directory / "history.jsonl"), "result_sha256": digest(directory / (name + ".json")),
            "diagnostic_inventory_sha256": canonical_sha(read(directory / "diagnostic-manifest.json", artifacts)["files"])}
    return {"campaign": CAMPAIGN, "runtime_bundle_sha256": campaign["runtime_bundle_sha256"],
            "scientific_verification": scientific, "exports": {"manifest_sha256": export_manifest(artifacts)},
            "allocation_proof": {"ledger_sha256": canonical_sha(campaign),
                "known_allocation_ids": sorted(row["job_id"] for row in known), "known_terminal": sorted(known, key=lambda row: row["job_id"]),
                **unknown, "all_owned_jobs_terminal": False, "all_known_jobs_terminal": True},
            "source_gate_proof": source}


def closure_request(proof: dict, independent_review: dict) -> dict:
    require(isinstance(proof, dict) and proof.get("campaign") == CAMPAIGN
            and bool(SHA256.fullmatch(proof.get("runtime_bundle_sha256", "")))
            and set(proof.get("scientific_verification", {})) == {task_key(task) for task in TASKS},
            "A structured eight-result closure proof is required")
    for worker, task in enumerate(TASKS):
        row = proof["scientific_verification"][task_key(task)]
        require(type(row.get("worker")) is int and row["worker"] == worker
                and row.get("train_records") == 3000 and row.get("validation_points") == 12
                and type(row.get("returncode")) is int and row["returncode"] == 0
                and all(isinstance(row.get(key), str) and bool(SHA256.fullmatch(row[key]))
                        for key in ("diagnostic_manifest_sha256", "independent_verification_sha256",
                                    "history_sha256", "result_sha256", "diagnostic_inventory_sha256")),
                "Structured completed-result digest proof is incomplete")
    allocation = proof.get("allocation_proof", {})
    require(allocation.get("all_owned_jobs_terminal") is False
            and allocation.get("all_known_jobs_terminal") is True
            and bool(allocation.get("unknown_requests"))
            and bool(SHA256.fullmatch(allocation.get("unknown_ledger_sha256", "")))
            and bool(SHA256.fullmatch(proof.get("exports", {}).get("manifest_sha256", "")))
            and bool(SHA256.fullmatch(proof.get("source_gate_proof", {}).get("review_sha256", ""))),
            "Closure isolation/evidence proof is incomplete")
    require(independent_review.get("decision") == CLOSURE_REVIEW_DECISION
            and independent_review.get("proof_sha256") == canonical_sha(proof)
            and isinstance(independent_review.get("review_sha256"), str)
            and bool(SHA256.fullmatch(independent_review["review_sha256"])),
            "Closure request requires separate digest-bound independent review")
    return {"proof": deepcopy(proof), "review": deepcopy(independent_review)}
