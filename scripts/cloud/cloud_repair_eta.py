"""Local, provisional wall-time estimates for the eight corrected reruns only.

This never submits/stops jobs or edits training. Historical timings are priors
from the original implementations, not measured rates for the corrected code.
"""

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
from statistics import median

from cloud_repair_common import CAMPAIGN, TASKS, locked, run_name, save, task_key


ARTIFACTS = Path(__file__).resolve().parents[4] / "artifacts/paper_8b_pro_rite_repair_20261115"
PRIOR_TIMINGS = ARTIFACTS.parent / "paper_8b_three_seeds_cloud_20261113/cloud-own-completed-timings.json"
FINALIZATION_HOURS = [0.1, 0.5]


def timestamp(value) -> float | None:
    if not isinstance(value, str):
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        return parsed.replace(tzinfo=parsed.tzinfo or timezone.utc).timestamp()
    except ValueError:
        return None


def number(value) -> bool:
    return type(value) in (int, float) and math.isfinite(value)


def row_clock(row: dict) -> tuple[str, float] | None:
    for key in ("_runtime", "elapsed_seconds"):
        if number(row.get(key)):
            # Each relative clock has its own origin; they are not interchangeable.
            return key, float(row[key])
    if number(row.get("_timestamp")):
        return "epoch", float(row["_timestamp"])
    event_time = timestamp(row.get("timestamp"))
    return ("epoch", event_time) if event_time is not None else None


def history_observations(rows: list[dict]) -> dict:
    """Use optional event clocks without inventing clocks in canonical JSONL."""
    validations, step_costs, durations = set(), [], []
    last_train_step, last_event_time, train_at_step = None, None, {}
    for row in rows:
        step = row.get("optimizer_step", row.get("step"))
        if type(step) is not int:
            continue
        clock = row_clock(row)
        if "train/loss" in row and number(row["train/loss"]):
            if clock is not None:
                if (last_train_step is not None and step == last_train_step + 1
                        and last_event_time is not None and clock[0] == last_event_time[0]):
                    delta = clock[1] - last_event_time[1]
                    if delta > 0:
                        step_costs.append(delta)
                train_at_step[step] = clock
                last_event_time = clock
            else:
                last_event_time = None
            last_train_step = step
        if "validation_score" in row and number(row["validation_score"]) and step in range(250, 3001, 250):
            validations.add(step)
            duration = row.get("validation_duration_seconds", row.get("validation_seconds"))
            if (not number(duration) and clock is not None and step in train_at_step
                    and clock[0] == train_at_step[step][0]):
                duration = clock[1] - train_at_step[step][1]
            if number(duration) and duration > 0:
                durations.append(float(duration))
            if clock is not None:
                last_event_time = clock
            else:
                # The next training interval would include an unmeasured eval.
                last_event_time = None
    return {"completed_evaluations": len(validations),
            "full_validation_seconds": median(durations[-3:]) if durations else None,
            "recent_train_seconds_per_step": median(step_costs[-100:]) if step_costs else None}


def prior_seconds(task: dict, prior: dict) -> float | None:
    values = [row.get("wall_seconds_through_final_history") for row in prior.get("rows", [])
              if row.get("method") == task["method"] and row.get("dataset") == task["dataset"]]
    values = [float(value) for value in values if number(value) and value > 0]
    return median(values) if values else None


def estimate(campaign: dict, monitor: dict | None = None, inspection: dict | None = None,
             prior: dict | None = None, histories: dict | None = None,
             verified: dict | None = None, at: datetime | None = None) -> dict:
    if campaign.get("campaign") != CAMPAIGN or campaign.get("tasks") != TASKS:
        raise ValueError("ETA scope differs from the eight corrected reruns")
    monitor, inspection, prior = monitor or {}, inspection or {}, prior or {}
    histories, verified = histories or {}, verified or {}
    for evidence in (monitor, inspection):
        if evidence and evidence.get("campaign") != CAMPAIGN:
            raise ValueError("ETA evidence belongs to another campaign")
    at = at or datetime.now(timezone.utc)
    clock = at.timestamp()
    expected = {run_name(task) for task in TASKS}
    complete = campaign.get("status") == "complete" and campaign.get("assets_deleted_verified") is True
    verified_count = monitor.get("verified_completed", 0)
    verified_names = set(verified) & expected
    if len(verified_names) != verified_count:
        verified_names = set()
    if verified_count == len(TASKS) and monitor.get("all_histories_verified") is True:
        verified_names = expected
    if complete:
        if verified_names != expected:
            raise ValueError("Completed campaign lacks all eight independently verified results")
    shared = {}
    for status in inspection.get("tasks", []):
        if status.get("task") in TASKS:
            name = run_name(status["task"])
            if name in shared:
                raise ValueError("Duplicate ETA task record")
            shared[name] = status
    active = {row["name"]: row for row in monitor.get("active", [])}
    pending = {row["name"]: row for row in monitor.get("pending_independent_verification", [])}
    failed = {row["name"]: row for row in monitor.get("failed", [])}
    assumptions = [
        "Aggregate the maximum remaining independent worker, never sum/8.",
        "Original Cloud A10080GB same-method/dataset full wall times include initialization and FINAL; corrected-code/GPU rates can differ.",
        "With >=250 own updates, elapsed wall time since W&B creation (or shared claim/start fallback) includes observed initialization and full validations.",
        "Future resource/control outages can exceed conditional ranges. ETA never controls stopping or relaunching.",
    ]
    workers = []
    for index, task in enumerate(TASKS):
        name = run_name(task)
        status = shared.get(name, {})
        current = active.get(name, pending.get(name, failed.get(name, {})))
        phase = current.get("phase", status.get("phase", "queued"))
        if name in failed:
            phase = "failed"
        step = current.get("step", status.get("step", 0))
        step = step if type(step) is int and 0 <= step <= 3000 else 0
        observations = history_observations(histories.get(name, []))
        evals = max(observations["completed_evaluations"], current.get("evaluation_count", 0) or 0,
                    status.get("evaluation_count", 0) or 0)
        evals = min(12, max(0, int(evals)))
        base = {"worker": index, "name": name, "task": task, "phase": phase,
                "step": step, "evaluation_count": evals, "verified": name in verified_names,
                "assumptions": [], **observations}
        if base["verified"]:
            workers.append({**base, "phase": "verified_complete", "remaining_hours": 0.,
                            "range_hours": [0., 0.], "source": "independent_verification"})
            continue
        old_wall = prior_seconds(task, prior)
        own_start = next((timestamp(value) for value in (current.get("created_at"), status.get("created_at"),
                         status.get("claimed_at"), status.get("started_at")) if timestamp(value) is not None), None)
        elapsed = clock - own_start if own_start is not None and own_start <= clock else None
        remaining_evals = 12 - evals
        observed_validation = observations["full_validation_seconds"]
        validation_cost = observed_validation or (max(60., old_wall / 24) if old_wall else 300.)
        if observed_validation is None:
            base["assumptions"].append("Unmeasured full-validation floor: max(60s, original full wall/24), or300s without a prior.")
        validation_floor = max(60., remaining_evals * validation_cost)
        if name in pending or phase == "complete":
            central, low, high, source = .1, .03, .5, "independent_verification_pending"
            base["assumptions"].append("Training may be done; allow provisional0.03–0.5h for independent evidence publication/verification.")
        elif step >= 250 and elapsed is not None:
            residual = elapsed * (3000 - step) / step
            if observations["recent_train_seconds_per_step"] is not None:
                residual = max(residual, (3000 - step) * observations["recent_train_seconds_per_step"]
                               + remaining_evals * validation_cost)
            residual = max(residual, validation_floor) * 1.1
            central, low, high, source = residual / 3600, residual / 3600 * .75, residual / 3600 * 1.8, "own_partial_wall"
            base.update(own_elapsed_seconds=elapsed, remaining_validation_floor_seconds=validation_floor)
            base["assumptions"].append("Own elapsed/step projection plus10% margin; conditional range0.75–1.8x remains provisional.")
        elif old_wall is not None:
            # Corrected Pro/Rite add different numerical operations. Historical
            # low-score runs are timing priors only, never score comparators.
            setup = .25 if phase in ("queued", "reserved", "claimed", "profiling", "initializing") else .1
            central = old_wall / 3600 * 1.5 + setup
            low, high, source = old_wall / 3600 * .75, old_wall / 3600 * 4 + .5, "original_method_dataset_prior"
            base.update(original_full_wall_seconds=old_wall)
            base["assumptions"].append("Corrected-version/GPU timing unknown: original full wall multipliers0.75/1.5/4 plus setup allowance.")
        else:
            central, low, high, source = None, None, None, "unknown_no_own_rate_or_matching_prior"
        unknown_wait = phase in ("failed", "waiting", "resource_waiting", "blocked")
        conditional_work = central
        if unknown_wait:
            central, high = None, None
            base["assumptions"].append("Resource/failure wait has unknown duration; no finite upper ETA is justified.")
        if phase == "failed":
            conditional_work, low, source = None, 0., "failure_diagnosis_required"
            base["assumptions"].append("No lost updates are credited and no automatic retry is assumed.")
        workers.append({**base, "remaining_hours": central, "range_hours": [low, high],
                        "conditional_work_hours": conditional_work, "source": source, "unknown_wait": unknown_wait})
    all_verified = verified_names == expected
    preparing = campaign.get("status") in ("preparing_assets", "preparing", "preflight", "planned")
    preparation = [.25, 2.] if preparing else [0., 0.]
    numeric = [row["remaining_hours"] for row in workers if row["remaining_hours"] is not None]
    known = all(row["remaining_hours"] is not None for row in workers)
    bounded = known and all(row["range_hours"][1] is not None for row in workers)
    experiments = max(numeric) if known else None
    central = experiments + sum(preparation) / 2 + .2 if experiments is not None else None
    lower_known = all(row["range_hours"][0] is not None for row in workers)
    low = max(row["range_hours"][0] for row in workers) + preparation[0] + FINALIZATION_HOURS[0] if lower_known else None
    high = max(row["range_hours"][1] for row in workers) + preparation[1] + FINALIZATION_HOURS[1] if bounded else None
    if preparing and known:
        low, high = min(low, 2.), max(high, 12.) if high is not None else None
        assumptions.append("Prelaunch planning range kept at least as broad as2–12h, including asset preparation/profiling and unknown corrected-version timing.")
    if all_verified:
        experiments, central, low, high = 0., .2, *FINALIZATION_HOURS
        preparation = [0., 0.]
    if complete:
        experiments, central, low, high = 0., 0., 0., 0.
    conditional_range, conditional_central = [low, high], central
    global_wait_reasons = []
    if campaign.get("control_wait_unknown") is True:
        global_wait_reasons.append(campaign.get("control_wait_reason") or "Explicit control wait has unknown duration")
    if campaign.get("status") in ("blocked", "waiting", "control_wait", "control_blocked", "resource_waiting"):
        global_wait_reasons.append("Campaign control/resource phase has unknown waiting time")
    if monitor.get("verification_errors"):
        global_wait_reasons.append("Independent verification errors require resolution before completion")
    if global_wait_reasons and not complete:
        central, high = None, None
        assumptions.append("Conditional work estimates exclude the current unknown control/verification delay.")
    return {"timestamp": at.isoformat(), "campaign": CAMPAIGN, "scope_runs": len(TASKS),
            "phase": "complete" if complete else ("finalizing" if all_verified else campaign.get("status", "active")),
            "verified_completed": len(verified_names), "whole_campaign_remaining_hours": [low, high],
            "central_remaining_hours": central, "experiments_remaining_hours": experiments,
            "conditional_whole_campaign_remaining_hours": conditional_range,
            "conditional_central_remaining_hours": conditional_central,
            "preparation_overhead_hours": preparation, "finalization_overhead_hours": [0., 0.] if complete else FINALIZATION_HOURS,
            "provisional": not complete, "unknown_wait": not complete and (bool(global_wait_reasons)
                or any(row.get("unknown_wait") for row in workers)),
            "global_wait_reasons": global_wait_reasons,
            "all_worker_estimates_available": known, "aggregation": "max remaining worker residual plus preparation/finalization",
            "assumptions": assumptions, "per_worker": workers, "future_outages_can_exceed_range": True,
            "not_a_stop_budget": True}


def load(path: Path) -> dict:
    return json.loads(path.read_text()) if path.exists() else {}


def refresh(artifacts: Path = ARTIFACTS, prior_path: Path = PRIOR_TIMINGS) -> dict:
    with locked(artifacts / "submission.lock"):
        histories = {}
        for task in TASKS:
            name = run_name(task)
            path = artifacts / "raw-wandb" / name / "history.jsonl"
            if path.exists():
                rows = []
                for line in path.read_text().splitlines():
                    try:
                        rows.append(json.loads(line))
                    except json.JSONDecodeError:
                        break
                histories[name] = rows
        result = estimate(load(artifacts / "campaign.json"), load(artifacts / "latest-monitor-check.json"),
                          load(artifacts / "inspection-latest.json"), load(prior_path), histories,
                          load(artifacts / "wandb-independent-verification.json"))
        save(artifacts / "eta-estimate.json", result)
        return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.parse_args()
    print(json.dumps(refresh(), indent=2))


if __name__ == "__main__":
    main()
