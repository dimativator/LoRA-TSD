"""Logical ETA checks without Cloud/W&B calls or training changes."""

from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts/cloud"))
import cloud_repair_eta as eta


NOW = datetime(2026, 11, 15, 14, tzinfo=timezone.utc)


def campaign(status="active"):
    return {"campaign": eta.CAMPAIGN, "tasks": eta.TASKS, "status": status}


def prior():
    return {"rows": [{**task, "wall_seconds_through_final_history": 3600} for task in eta.TASKS]}


def monitor(rows=None, count=0):
    return {"campaign": eta.CAMPAIGN, "active": rows or [], "verified_completed": count}


def active(index, step, elapsed_hours, evals, phase="training"):
    return {"name": eta.run_name(eta.TASKS[index]), "worker": index, "phase": phase,
            "step": step, "evaluation_count": evals,
            "created_at": (NOW - timedelta(hours=elapsed_hours)).isoformat()}


def test_parallel_eta_uses_slowest_remaining_worker_not_average_or_sum():
    rows = [active(index, 1500, index + 1, 6) for index in range(8)]
    result = eta.estimate(campaign(), monitor(rows), prior=prior(), at=NOW)
    expected = max(row["remaining_hours"] for row in result["per_worker"])
    assert result["experiments_remaining_hours"] == expected
    assert expected == pytest.approx(8 * 1.1)
    assert result["central_remaining_hours"] == pytest.approx(expected + .2)
    assert result["not_a_stop_budget"] is True


@pytest.mark.parametrize("evals", [11, 12])
def test_final_pending_never_has_zero_experimental_eta(evals):
    rows = [active(index, 3000, 2, evals, "validation") for index in range(8)]
    result = eta.estimate(campaign(), monitor(rows), prior=prior(), at=NOW)
    assert result["experiments_remaining_hours"] > 0
    assert all(row["remaining_hours"] > 0 for row in result["per_worker"])


def test_optional_event_clocks_measure_full_validation_separately_from_updates():
    rows = [{"step": 249, "train/loss": 1., "_runtime": 100.},
            {"step": 250, "train/loss": 1., "_runtime": 102.},
            {"step": 250, "validation_score": .5, "_runtime": 162.},
            {"step": 251, "train/loss": 1., "_runtime": 164.}]
    observed = eta.history_observations(rows)
    assert observed == {"completed_evaluations": 1, "full_validation_seconds": 60.,
                        "recent_train_seconds_per_step": 2.}


def test_untimed_canonical_history_does_not_invent_validation_measurements():
    observed = eta.history_observations([{"step": 250, "train/loss": 1.},
                                         {"step": 250, "validation_score": .5}])
    assert observed["completed_evaluations"] == 1
    assert observed["full_validation_seconds"] is None
    assert observed["recent_train_seconds_per_step"] is None


def test_mixed_relative_and_epoch_clocks_are_never_subtracted():
    rows = [{"step": 249, "train/loss": 1., "_runtime": 100.},
            {"step": 250, "train/loss": 1., "_timestamp": NOW.timestamp()},
            {"step": 250, "validation_score": .5, "_runtime": 160.},
            {"step": 251, "train/loss": 1., "elapsed_seconds": 162.}]
    observed = eta.history_observations(rows)
    assert observed["completed_evaluations"] == 1
    assert observed["full_validation_seconds"] is None
    assert observed["recent_train_seconds_per_step"] is None


def test_numeric_and_iso_epoch_timestamps_share_a_clock_domain():
    rows = [{"step": 250, "train/loss": 1., "_timestamp": NOW.timestamp()},
            {"step": 250, "validation_score": .5,
             "timestamp": (NOW + timedelta(seconds=60)).isoformat()}]
    assert eta.history_observations(rows)["full_validation_seconds"] == 60


def test_missing_event_clock_breaks_update_interval_measurement():
    rows = [{"step": 1, "train/loss": 1., "_runtime": 1.},
            {"step": 2, "train/loss": 1.},
            {"step": 3, "train/loss": 1., "_runtime": 3.}]
    assert eta.history_observations(rows)["recent_train_seconds_per_step"] is None


def test_measured_future_full_validation_cost_is_a_residual_floor():
    rows = [active(index, 2750, .1, 11) for index in range(8)]
    histories = {eta.run_name(task): [{"step": 2750, "validation_score": .5,
                                      "validation_duration_seconds": 1800}] for task in eta.TASKS}
    result = eta.estimate(campaign(), monitor(rows), prior=prior(), histories=histories, at=NOW)
    assert result["experiments_remaining_hours"] >= .5 * 1.1
    assert all(row["full_validation_seconds"] == 1800 for row in result["per_worker"])


def test_verified_workers_contribute_zero_but_finalization_remains():
    state = monitor(count=8)
    state["all_histories_verified"] = True
    result = eta.estimate(campaign(), state, at=NOW)
    assert result["experiments_remaining_hours"] == 0
    assert result["whole_campaign_remaining_hours"] == [.1, .5]
    assert result["phase"] == "finalizing"
    assert all(row["remaining_hours"] == 0 for row in result["per_worker"])


def test_only_verified_completion_and_assets_deletion_produce_whole_eta_zero():
    state = monitor(count=8)
    state["all_histories_verified"] = True
    config = {**campaign("complete"), "assets_deleted_verified": True}
    result = eta.estimate(config, state, at=NOW)
    assert result["whole_campaign_remaining_hours"] == [0, 0]
    assert result["central_remaining_hours"] == 0 and result["provisional"] is False
    with pytest.raises(ValueError, match="eight independently verified"):
        eta.estimate(config, monitor(), at=NOW)


def test_no_rate_or_matching_prior_is_unknown_not_zero():
    result = eta.estimate(campaign(), monitor(), at=NOW)
    assert result["whole_campaign_remaining_hours"] == [None, None]
    assert result["central_remaining_hours"] is None and result["experiments_remaining_hours"] is None
    assert not result["all_worker_estimates_available"]


def test_prelaunch_uses_broad_corrected_version_planning_assumptions():
    result = eta.estimate(campaign("preparing_assets"), prior=prior(), at=NOW)
    low, high = result["whole_campaign_remaining_hours"]
    assert low <= 2 and high >= 12
    assert result["preparation_overhead_hours"][1] > 0
    assert all(row["source"] == "original_method_dataset_prior" for row in result["per_worker"])
    assert all("Corrected-version" in row["assumptions"][-1] for row in result["per_worker"])


def test_resource_wait_and_failed_jobs_have_no_finite_upper_bound():
    rows = [active(index, 1000, 1, 4) for index in range(8)]
    rows[0]["phase"] = "resource_waiting"
    result = eta.estimate(campaign(), monitor(rows), prior=prior(), at=NOW)
    assert result["whole_campaign_remaining_hours"][1] is None
    assert result["unknown_wait"] is True and result["central_remaining_hours"] is None


@pytest.mark.parametrize("status", ["blocked", "waiting", "control_wait"])
def test_global_control_wait_preserves_conditional_work_but_has_unknown_whole_eta(status):
    result = eta.estimate(campaign(status), monitor(), prior=prior(), at=NOW)
    assert result["conditional_whole_campaign_remaining_hours"][1] is not None
    assert result["conditional_central_remaining_hours"] > 0
    assert result["whole_campaign_remaining_hours"][1] is None
    assert result["central_remaining_hours"] is None and result["unknown_wait"] is True


def test_explicit_control_outage_keeps_prelaunch_work_range_conditional():
    config = {**campaign("preparing_assets"), "control_wait_unknown": True,
              "control_wait_reason": "Asset readiness inspection is unavailable"}
    result = eta.estimate(config, prior=prior(), at=NOW)
    assert result["conditional_whole_campaign_remaining_hours"][1] >= 12
    assert result["whole_campaign_remaining_hours"][1] is None
    assert result["global_wait_reasons"] == [config["control_wait_reason"]]


def test_verification_errors_do_not_leave_a_finite_whole_upper_eta():
    state = {**monitor(), "verification_errors": [{"name": eta.run_name(eta.TASKS[0]), "error": "Unavailable evidence"}]}
    result = eta.estimate(campaign(), state, prior=prior(), at=NOW)
    assert result["conditional_whole_campaign_remaining_hours"][1] is not None
    assert result["whole_campaign_remaining_hours"][1] is None and result["central_remaining_hours"] is None


def test_failed_partial_steps_are_not_credited_or_assumed_to_retry():
    state = monitor()
    state["failed"] = [active(index, 2999, 1., 11, "failed") for index in range(8)]
    result = eta.estimate(campaign(), state, prior=prior(), at=NOW)
    assert result["central_remaining_hours"] is None
    assert all(row["source"] == "failure_diagnosis_required" and row["conditional_work_hours"] is None
               for row in result["per_worker"])


def test_shared_completion_without_independent_verification_remains_positive():
    inspection = {"campaign": eta.CAMPAIGN, "tasks": [{"task": task, "phase": "complete", "step": 3000}
                                                      for task in eta.TASKS]}
    result = eta.estimate(campaign(), monitor(), inspection, prior=prior(), at=NOW)
    assert result["verified_completed"] == 0 and result["experiments_remaining_hours"] > 0
    assert all(row["source"] == "independent_verification_pending" for row in result["per_worker"])


def test_original_timing_file_is_read_only_and_new_output_uses_submission_lock(tmp_path):
    source = tmp_path / "old-timings.json"
    source.write_text(json.dumps(prior()))
    before = source.read_bytes()
    artifacts = tmp_path / "new"
    artifacts.mkdir()
    (artifacts / "campaign.json").write_text(json.dumps(campaign("preparing_assets")))
    result = eta.refresh(artifacts, source)
    assert source.read_bytes() == before
    assert (artifacts / "submission.lock").exists()
    assert json.loads((artifacts / "eta-estimate.json").read_text()) == result


def test_foreign_campaign_or_duplicate_shared_tasks_rejected():
    with pytest.raises(ValueError, match="scope"):
        eta.estimate({**campaign(), "campaign": "old180"})
    with pytest.raises(ValueError, match="another campaign"):
        eta.estimate(campaign(), {"campaign": "old180"})
    task = {"task": eta.TASKS[0]}
    with pytest.raises(ValueError, match="Duplicate"):
        eta.estimate(campaign(), inspection={"campaign": eta.CAMPAIGN, "tasks": [task, task]})
