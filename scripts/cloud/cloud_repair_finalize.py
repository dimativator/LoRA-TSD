"""Export eight paired repairs and authorize their strictly scoped final cleanup.

These selected reruns are paired comparisons, not a new matched three-seed grid.
This module only reads Cloud snapshots and writes local exports/authorization.
"""

import argparse
import csv
from datetime import datetime, timezone
from html import escape
import json
import math
from pathlib import Path

from cloud_repair_collect import owned_records
from cloud_repair_common import (CAMPAIGN, TASKS, affected_paths, digest, locked,
                                 now, run_name, save, task_key, verify_rows)
from cloud_repair_monitor import (ARTIFACTS, SHA256, cache_valid, expected_command,
                                  require, validate_manifest)


ORIGINAL = ARTIFACTS.parent / "paper_8b_three_seeds_cloud_20261113"
TERMINAL = {"completed", "failed", "stopped", "cancelled", "canceled"}
SCOPE = "Eight selected repaired cases paired with original same-method/dataset/seed results; no new three-seed aggregate."


def read(path: Path) -> dict:
    require(not path.is_symlink() and path.is_file(), "Missing or linked local finalization evidence")
    return json.loads(path.read_text())


def fresh(timestamp: str) -> None:
    age = (datetime.now(timezone.utc) - datetime.fromisoformat(timestamp.replace("Z", "+00:00"))).total_seconds()
    require(-60 <= age <= 300, "Refresh final inspection/scheduler evidence")


def verify_local(verified: dict, campaign: dict, artifacts: Path) -> tuple[list, list, dict]:
    require(set(verified) == {run_name(task) for task in TASKS}, "Exactly eight independently verified repairs required")
    repaired, curves, histories = [], [], {}
    for index, task in enumerate(TASKS):
        name = run_name(task)
        row = verified[name]
        require(all(row.get(key) == value for key, value in {"name": name, **task, "worker": index,
                "runtime_bundle_sha256": campaign["runtime_bundle_sha256"], "state": "finished",
                "train_records": 3000, "validation_points": 12,
                "all_diagnostics_independently_verified": True}.items()), "Independent verification identity differs")
        require(row.get("path") == "andrey/LoRA-Muon/" + row["id"], "Independent W&B identity differs")
        directory = artifacts / "raw-wandb" / name
        require(not directory.is_symlink() and directory.resolve().is_relative_to(artifacts.resolve())
                and cache_valid(row, directory), "Independent raw diagnostic evidence changed")
        manifest = read(directory / "diagnostic-manifest.json")
        command, lr = expected_command(task, row["microbatch"], campaign["runtime_bundle_sha256"])
        validate_manifest(manifest, row, campaign, command)
        history = [json.loads(line) for line in (directory / "history.jsonl").read_text().splitlines()]
        result = read(directory / (name + ".json"))
        require(result.get("run_name") == name, "Corrected result run name differs")
        checks = verify_rows(history, result, task)
        require(all(row.get(key) == value for key, value in checks.items()), "Cached corrected metrics differ from raw history")
        for point in history:
            if "train/loss" in point:
                require(math.isclose(point.get("lr_A_used", float("nan")), lr, rel_tol=1e-8, abs_tol=1e-12),
                        "Corrected actual learning rate differs")
                require(all(math.isfinite(value) for key, value in point.items()
                            if key.startswith("optimizer/") and isinstance(value, (int, float))),
                        "Nonfinite corrected optimizer diagnostic")
        repaired.append(row)
        histories[name] = history
        curves.extend({**task, "version": "corrected", "step": point["step"],
                       "accuracy_percent": point["validation_score"] * 100}
                      for point in history if "validation_score" in point)
    return repaired, curves, histories


def original_evidence(original: Path) -> tuple[dict, list, dict]:
    rows = read(original / "wandb-independent-verification.json")
    paired, curves, histories = {}, [], {}
    for task in TASKS:
        matches = [row for row in rows.values()
                   if all(row.get(key) == task[key] for key in ("method", "dataset", "seed"))]
        require(len(matches) == 1, "Original same-seed result is missing or ambiguous")
        row = matches[0]
        require(row.get("state") == "finished" and row.get("train_records") == 3000
                and row.get("validation_points") == 12, "Original result is incomplete")
        directory = original / "raw-wandb" / row["name"]
        require(not directory.is_symlink() and directory.resolve().is_relative_to(original.resolve()),
                "Original evidence escapes historical namespace")
        history_path = directory / "history.jsonl"
        require(not history_path.is_symlink() and Path(row["history_file"]).resolve() == history_path.resolve(),
                "Original canonical history path differs")
        history = [json.loads(line) for line in history_path.read_text().splitlines()]
        result = read(directory / ("metrics.json" if row.get("reused") is True else row["name"] + ".json"))
        require(result.get("run_name") == row["name"], "Original result run name differs")
        checks = verify_rows(history, result, task)
        require(all(row.get(key) == value for key, value in checks.items()), "Original cached metrics differ from raw history")
        paired[task_key(task)] = row
        histories[row["name"]] = history
        curves.extend({**task, "version": "original", "step": point["step"],
                       "accuracy_percent": point["validation_score"] * 100}
                      for point in history if "validation_score" in point)
    return paired, curves, histories


def verify_inspection(inspection: dict, verified: dict, campaign: dict | None = None,
                      artifacts: Path | None = None) -> dict:
    require(inspection.get("campaign") == CAMPAIGN and inspection.get("collector_source_mode") == "inspect",
            "Final inspection is not a repair inspection")
    fresh(inspection["timestamp"])
    require(inspection.get("collector_timed_out") is False, "Inspection transport timed out")
    if inspection.get("collector_transport_source") == "wandb_aux_snapshot":
        require(campaign is not None and artifacts is not None, "Missing local control proof context")
        from cloud_repair_control import verify_control_files
        verify_control_files(inspection, artifacts, campaign)
    else:
        require(inspection.get("collector_cli_returncode") == 0 and inspection.get("collector_ssh_returncode") == 0,
                "Inspection transport did not finish successfully")
    fs = inspection["filesystems"]
    require(len(fs) == len(affected_paths()) and {row["requested_path"] for row in fs}
            == {str(path) for path in affected_paths()}, "Affected filesystem coverage differs")
    require(all(type(row.get("available_bytes")) is int and row["available_bytes"] > 5_000_000_000
                and row.get("status") == "ok" for row in fs), "Storage reserve is not currently verified")
    require(len(inspection["tasks"]) == len(inspection["workers"]) == 8, "Exactly eight task/worker records required")
    manifests = {}
    for index, task in enumerate(TASKS):
        for collection in ("tasks", "workers"):
            matches = [row for row in inspection[collection] if row.get("task") == task and row.get("worker") == index]
            require(len(matches) == 1, "Task/worker completion is ambiguous")
            row, proof = matches[0], verified[run_name(task)]
            require(row.get("phase") == "complete" and row.get("returncode") == 0
                    and row.get("child_pid") is None and row.get("wandb_sync_complete") is True,
                    "Task/worker is not complete and synchronized")
            require(row.get("wandb_id") == proof["id"] and row.get("recovery_manifest_sha256")
                    == proof["diagnostic_manifest_sha256"], "Shared diagnostic manifest differs from independent proof")
            require(row.get("verification", {}).get("train_records") == 3000
                    and row.get("verification", {}).get("validation_points") == 12,
                    "Shared training/final-validation proof is incomplete")
        manifests[task_key(task)] = verified[run_name(task)]["diagnostic_manifest_sha256"]
    return manifests


def verify_scheduler(campaign: dict, scheduler: dict, inspection: dict, artifacts: Path) -> list[str]:
    require(scheduler.get("campaign") == CAMPAIGN, "Scheduler belongs to another campaign")
    fresh(scheduler["timestamp"])
    require(not campaign.get("ambiguous_submissions"), "Submission ambiguity remains")
    require(not campaign.get("superseded_unreceived_gpu_intents"), "Superseded unanswered GPU requests require final audit")
    require(not any(row.get("recovery_conflict") for row in campaign.get("jobs", [])
                    + campaign.get("auxiliary_jobs", [])), "Received allocation conflict remains")
    require(not campaign.get("unresolved_auxiliary_submissions"), "Deferred CPU observation remains unresolved")
    require(scheduler.get("cli_returncode") == 0 and scheduler.get("ssh_returncode") == 0
            and scheduler.get("timed_out") is False, "Scheduler transport is incomplete")
    proof = Path(scheduler["raw_proof"])
    require(not proof.is_symlink() and proof.resolve().is_relative_to(artifacts.resolve())
            and digest(proof) == scheduler.get("raw_proof_sha256"), "Scheduler raw proof changed")
    if "queries" in scheduler:
        queries = scheduler["queries"]
        require(len(queries) == 2 and {row.get("source") for row in queries} == {"full", "active"}
                and scheduler.get("listings_complete") is True and scheduler.get("absence_proof") is False,
                "Composite scheduler query coverage is incomplete")
        sources = []
        for query in queries:
            require(query.get("complete") is True and query.get("cli_returncode") == 0
                    and query.get("ssh_returncode") == 0 and query.get("timed_out") is False,
                    "Composite scheduler query transport is incomplete")
            require(query.get("command") == ["mlsub", "list"] + (["--active"] if query["source"] == "active" else []),
                    "Composite scheduler command differs")
            fresh(query["started_at"])
            fresh(query["observed_at"])
            source = Path(query["raw_proof"])
            require(not source.is_symlink() and source.resolve().is_relative_to(artifacts.resolve())
                    and digest(source) == query.get("raw_proof_sha256"), "Composite scheduler source SHA differs")
            sources.append(f"# query={query['source']} observed_at={query['observed_at']}\n{source.read_text()}")
        require(proof.read_text() == "\n".join(sources), "Composite scheduler proof differs from its query sources")
    ledger = owned_records(campaign)
    ids = {row["job_id"] for row in ledger}
    require(ids and len(scheduler["jobs"]) == len(ids)
            and {row["job_id"] for row in scheduler["jobs"]} == ids, "Scheduler lacks exact current/archived owned coverage")
    current = inspection.get("collector_source_job_id")
    current_records = [row for row in ledger if row["job_id"] == current]
    require(current_records and all(row.get("mode") == "inspect" and row.get("gpus") == 0
                                   for row in current_records), "Inspection exemption is not an owned CPU inspect job")
    source = proof.read_text().splitlines()
    parsed = []
    for line in source:
        fields = line.strip().split(None, 2)
        if fields and fields[0].startswith("lm-mpi-job-"):
            metadata = fields[2] if len(fields) == 3 else ""
            parsed.append({"job_id": fields[0], "status": fields[1].lower() if len(fields) >= 2 else None,
                           "line": line, "note": metadata.split("|note:", 1)[1].strip() if "|note:" in metadata else None,
                           "owned_entry": "repo:dimativator/LoRA-TSD|entry:scripts/cloud/cloud_repair_entry.py|" in metadata})
    terminal = []
    for row in scheduler["jobs"]:
        matches = [entry for entry in parsed if entry["job_id"] == row["job_id"]]
        notes = {entry["note"] for entry in ledger if entry["job_id"] == row["job_id"] and entry.get("note")}
        require(matches and row.get("source") == "fresh_exact_mlsub_list_row"
                and row.get("evidence") in {entry["line"] for entry in matches},
                "Scheduler job lacks unique fresh raw evidence")
        require(len({entry["status"] for entry in matches}) == 1
                and matches[0]["status"] == str(row.get("status")).lower(),
                "Scheduler status differs or conflicts across sources")
        require(len(notes) == 1 and {entry["note"] for entry in matches} == notes
                and all(entry["owned_entry"] for entry in matches), "Scheduler job entry point/note differs")
        require(not any(entry["job_id"] != row["job_id"] and entry["note"] in notes for entry in parsed),
                "Multiple scheduler job IDs share an owned submission note")
        if row["job_id"] != current:
            require(str(row.get("status")).lower() in TERMINAL, "An owned allocation is not terminal")
            terminal.append(row["job_id"])
    return sorted(terminal)


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot_curves(path: Path, curves: list[dict]) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        matplotlib.rcParams["svg.hashsalt"] = CAMPAIGN
        import matplotlib.pyplot as plt
    except ImportError:
        # Standalone deterministic SVG for environments without plotting packages.
        parts = ['<svg xmlns="http://www.w3.org/2000/svg" width="1000" height="900" viewBox="0 0 1000 900">',
                 '<rect width="100%" height="100%" fill="white"/>']
        for index, task in enumerate(TASKS):
            left, top = 60 + (index % 2) * 500, 35 + (index // 2) * 220
            parts.append(f'<text x="{left}" y="{top}" font-size="14">{escape(task_key(task))}</text>')
            parts.append(f'<path d="M{left},{top+10}v160h400" fill="none" stroke="black"/>')
            for version, color in (("original", "#b45309"), ("corrected", "#0369a1")):
                rows = [row for row in curves if all(row[key] == value for key, value in task.items())
                        and row["version"] == version]
                points = " ".join(f"{left+row['step']/3000*400:.3f},{top+170-row['accuracy_percent']*1.6:.3f}" for row in rows)
                parts.append(f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="2"/>')
            parts.append(f'<text x="{left}" y="{top+192}" font-size="11">Updates 0–3000; accuracy 0–100%; orange original, blue corrected</text>')
        path.write_text("\n".join(parts + ["</svg>"]))
        return
    figure, axes = plt.subplots(4, 2, figsize=(12, 13), constrained_layout=True)
    for axis, task in zip(axes.flat, TASKS):
        for version in ("original", "corrected"):
            rows = [row for row in curves if all(row[key] == value for key, value in task.items())
                    and row["version"] == version]
            axis.plot([row["step"] for row in rows], [row["accuracy_percent"] for row in rows], label=version)
        axis.set_title(task_key(task))
        axis.set_xlabel("Optimizer updates")
        axis.set_ylabel("Validation accuracy (%)")
        axis.set_ylim(0, 100)
        axis.grid(alpha=.2)
    axes.flat[0].legend()
    figure.savefig(path, metadata={"Date": None})
    plt.close(figure)


def finalize(artifacts: Path = ARTIFACTS, original: Path = ORIGINAL) -> dict:
    campaign = read(artifacts / "campaign.json")
    require(campaign.get("campaign") == CAMPAIGN and campaign.get("tasks") == TASKS
            and campaign.get("target_runs") == 8 and bool(SHA256.fullmatch(campaign.get("runtime_bundle_sha256", ""))),
            "Repair campaign scope/bundle differs")
    require(len(campaign.get("jobs", [])) == 8
            and {row.get("worker") for row in campaign["jobs"]} == set(range(8))
            and all(row.get("mode") == "worker" and row.get("gpus") == 1 for row in campaign["jobs"]),
            "Final current GPU ledger must contain exactly eight independent one-GPU workers")
    verification_path = artifacts / "wandb-independent-verification.json"
    verified = read(verification_path)
    repaired, curves, histories = verify_local(verified, campaign, artifacts)
    originals, old_curves, old_histories = original_evidence(original)
    curves = old_curves + curves
    inspection, scheduler = read(artifacts / "inspection-latest.json"), read(artifacts / "scheduler-snapshot.json")
    manifests = verify_inspection(inspection, verified, campaign, artifacts)
    terminal = verify_scheduler(campaign, scheduler, inspection, artifacts)
    paired = []
    for row in repaired:
        old = originals[task_key(row)]
        paired.append({**{key: row[key] for key in ("method", "optimizer", "dataset", "seed")},
                       "original_accuracy_percent": old["best_accuracy"] * 100, "original_best_step": old["best_step"],
                       "corrected_accuracy_percent": row["best_accuracy"] * 100, "corrected_best_step": row["best_step"],
                       "difference_percentage_points": (row["best_accuracy"] - old["best_accuracy"]) * 100,
                       "original_wandb_url": old["url"], "corrected_wandb_url": row["url"]})
    output = artifacts / "final"
    require(not output.is_symlink() and output.resolve().is_relative_to(artifacts.resolve()), "Export path escapes repair namespace")
    output.mkdir(exist_ok=True)
    require(not any(path.is_symlink() or not path.is_file() for path in output.iterdir()), "Unknown or linked export entry")
    write_csv(output / "paired-results.csv", paired)
    save(output / "paired-results.json", {"campaign": CAMPAIGN, "scope": SCOPE, "pairs": paired})
    write_csv(output / "validation-curves.csv", curves)
    save(output / "validation-curves.json", {"scope": SCOPE, "curves": curves})
    save(output / "raw-histories.json", {"scope": SCOPE, "original": old_histories, "corrected": histories})
    plot_curves(output / "validation-curves.svg", curves)
    exported = {path.name: {"bytes": path.stat().st_size, "sha256": digest(path)}
                for path in sorted(output.iterdir()) if path.name != "artifact-manifest.json"}
    save(output / "artifact-manifest.json", exported)
    require(all((output / name).stat().st_size == row["bytes"] and digest(output / name) == row["sha256"]
                for name, row in exported.items()), "Final export integrity check failed")
    # Recheck freshness after local hashing and plotting before issuing authority.
    verify_inspection(inspection, verified, campaign, artifacts)
    verify_scheduler(campaign, scheduler, inspection, artifacts)
    authorization = {"campaign": CAMPAIGN, "verified_runs": 8, "all_owned_jobs_terminal": True,
                     "all_diagnostics_independently_verified": True, "exports_verified": True,
                     "diagnostic_manifests": manifests, "runtime_bundle_sha256": campaign["runtime_bundle_sha256"],
                     "owned_terminal_jobs": terminal, "independent_verification_sha256": digest(verification_path),
                     "original_verification_sha256": digest(original / "wandb-independent-verification.json"),
                     "export_manifest_sha256": digest(output / "artifact-manifest.json")}
    authorization_path = artifacts / "cleanup-authorization.json"
    if authorization_path.exists():
        existing = read(authorization_path)
        require({key: value for key, value in existing.items() if key != "created_at"} == authorization,
                "Existing stable cleanup authorization differs; preserve it and investigate")
        return existing
    authorization["created_at"] = now()
    save(authorization_path, authorization)
    return authorization


def main() -> None:
    argparse.ArgumentParser(description=__doc__).parse_args()
    with locked(ARTIFACTS / "submission.lock"):
        print(json.dumps(finalize(), indent=2))


if __name__ == "__main__":
    main()
