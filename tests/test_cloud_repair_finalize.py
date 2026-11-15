"""Finalization fails closed and exports selected pairs without rewriting history."""

import copy
from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts/cloud"))
import cloud_repair_finalize as final


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value) + "\n")


def record(path, original):
    return {"path": original, "wandb_name": path.name, "bytes": path.stat().st_size,
            "sha256": final.digest(path), "local_file": str(path)}


@pytest.fixture
def complete(tmp_path, monkeypatch):
    artifacts, old = tmp_path / "repair", tmp_path / "original"
    artifacts.mkdir()
    old.mkdir()
    campaign = {"campaign": final.CAMPAIGN, "tasks": final.TASKS, "target_runs": 8,
                "runtime_bundle_sha256": "a" * 64, "jobs": [], "auxiliary_jobs": [],
                "archived_jobs": [], "ambiguous_submissions": []}
    verified, original, tasks = {}, {}, []
    timestamp = datetime.now(timezone.utc).isoformat()
    for index, task in enumerate(final.TASKS):
        name, old_name = final.run_name(task), "paper8b_" + final.task_key(task)
        command = ["python", "-u", "main.py", name]
        lr = .0001 if task["method"] == "lora_rite" else .00005
        monkeypatch.setattr(final, "expected_command", lambda task, batch, sha: (
            ["python", "-u", "main.py", final.run_name(task)],
            .0001 if task["method"] == "lora_rite" else .00005))
        for root, run_name, score in ((artifacts, name, .8), (old, old_name, .5)):
            directory = root / "raw-wandb" / run_name
            directory.mkdir(parents=True)
            rows = [{"step": step, "train/loss": 1., "lr_A_used": lr} for step in range(1, 3001)]
            rows += [{"step": step, "validation_score": score - .01 + step / 300000}
                     for step in range(250, 3001, 250)]
            history = directory / "history.jsonl"
            history.write_text("".join(json.dumps(row) + "\n" for row in rows))
            result = {"run_name": run_name, "optimizer": task["optimizer"], "dataset": task["dataset"],
                      "seed": task["seed"], "model": "meta-llama/Meta-Llama-3.1-8B", "steps": 3000,
                      "reported_accuracy": score, "reported_accuracy_step": 3000}
            write(directory / (run_name + ".json"), result)
            entry = {"name": run_name, **task, "worker": index, "state": "finished", "id": f"run{index}",
                     "path": f"andrey/LoRA-Muon/run{index}", "url": "https://example.invalid/" + run_name,
                     "train_records": 3000, "validation_points": 12, "best_accuracy": score, "best_step": 3000,
                     "history_file": str(history), "microbatch": 8,
                     "runtime_bundle_sha256": "a" * 64, "all_diagnostics_independently_verified": True}
            if root == old:
                original[run_name] = entry
            else:
                files = [record(history, "history.jsonl"), record(directory / (run_name + ".json"), run_name + ".json")]
                manifest = {"campaign": final.CAMPAIGN, "run_name": name, "wandb_id": entry["id"],
                            "runtime_bundle_sha256": "a" * 64, "model_revision": "d04e592bb4f6aa9cfee91e2e20afa771667e1d4b",
                            "command": command, "files": [{key: value for key, value in row.items()
                                                           if key != "local_file"} for row in files]}
                manifest_path = directory / "diagnostic-manifest.json"
                write(manifest_path, manifest)
                entry.update(diagnostic_manifest_sha256=final.digest(manifest_path),
                             verified_files=files + [record(manifest_path, "diagnostic-manifest.json")])
                verified[run_name] = entry
        tasks.append({"task": task, "worker": index, "phase": "complete", "returncode": 0,
                      "child_pid": None, "wandb_sync_complete": True, "wandb_id": f"run{index}",
                      "verification": {"train_records": 3000, "validation_points": 12},
                      "command": command, "recovery_manifest_sha256": verified[name]["diagnostic_manifest_sha256"]})
        campaign["jobs"].append({"job_id": f"lm-mpi-job-{index}abc", "mode": "worker", "worker": index, "gpus": 1})
    campaign["auxiliary_jobs"] = [{"job_id": "lm-mpi-job-abcd", "mode": "prepare", "gpus": 0},
                                  {"job_id": "lm-mpi-job-efab", "mode": "inspect", "gpus": 0}]
    campaign["archived_jobs"] = [{"job_id": "lm-mpi-job-fade", "mode": "worker", "gpus": 1}]
    paths = [Path("/"), Path("/tmp"), tmp_path / "home", tmp_path / "assets", tmp_path / "state", tmp_path / "logs"]
    monkeypatch.setattr(final, "affected_paths", lambda: paths)
    inspection = {"campaign": final.CAMPAIGN, "timestamp": timestamp, "tasks": tasks,
                  "workers": copy.deepcopy(tasks), "collector_source_mode": "inspect",
                  "collector_source_job_id": "lm-mpi-job-efab", "collector_cli_returncode": 0,
                  "collector_ssh_returncode": 0, "collector_timed_out": False,
                  "filesystems": [{"requested_path": str(path), "available_bytes": 9_000_000_000,
                                   "status": "ok"} for path in paths]}
    scheduler = {"campaign": final.CAMPAIGN, "timestamp": timestamp, "jobs": [],
                 "cli_returncode": 0, "ssh_returncode": 0, "timed_out": False}
    lines = []
    for job in final.owned_records(campaign):
        status = "Running" if job["mode"] == "inspect" else "Completed"
        line = job["job_id"] + " " + status + " repo:dimativator/LoRA-TSD|entry:scripts/cloud/cloud_repair_entry.py|note:test"
        lines.append(line)
        scheduler["jobs"].append({"job_id": job["job_id"], "status": status,
                                  "source": "fresh_exact_mlsub_list_row", "evidence": line})
    proof = artifacts / "scheduler.log"
    proof.write_text("\n".join(lines) + "\n")
    scheduler.update(raw_proof=str(proof), raw_proof_sha256=final.digest(proof))
    for filename, value in (("campaign.json", campaign), ("wandb-independent-verification.json", verified),
                            ("inspection-latest.json", inspection), ("scheduler-snapshot.json", scheduler)):
        write(artifacts / filename, value)
    write(old / "wandb-independent-verification.json", original)
    monkeypatch.setattr(final, "plot_curves", lambda path, curves: path.write_text('<svg xmlns="http://www.w3.org/2000/svg"/>'))
    return artifacts, old


def modify(complete, filename, change):
    path = complete[0] / filename
    value = json.loads(path.read_text())
    change(value)
    write(path, value)


def test_exports_exact_paired_cases_all_curves_and_stable_cleanup_authorization(complete):
    artifacts, old = complete
    old_bytes = (old / "wandb-independent-verification.json").read_bytes()
    result = final.finalize(artifacts, old)
    assert result["verified_runs"] == 8 and len(result["diagnostic_manifests"]) == 8
    assert len(result["owned_terminal_jobs"]) == 10
    pairs = json.loads((artifacts / "final/paired-results.json").read_text())
    assert len(pairs["pairs"]) == 8 and "no new three-seed" in pairs["scope"]
    assert all(abs(row["difference_percentage_points"] - 30) < 1e-9 for row in pairs["pairs"])
    curves = json.loads((artifacts / "final/validation-curves.json").read_text())["curves"]
    assert len(curves) == 8 * 2 * 12 and all(row["step"] % 250 == 0 for row in curves)
    authorization_bytes = (artifacts / "cleanup-authorization.json").read_bytes()
    assert final.finalize(artifacts, old) == result
    assert (artifacts / "cleanup-authorization.json").read_bytes() == authorization_bytes
    assert (old / "wandb-independent-verification.json").read_bytes() == old_bytes


@pytest.mark.parametrize("damage", ["one_missing", "wrong_seed", "partial", "manifest_sha", "unsynced", "child_alive"])
def test_incomplete_or_mismatched_inspection_refuses_authority(complete, damage):
    def change(value):
        row = value["tasks"][0]
        if damage == "one_missing": value["workers"].pop()
        elif damage == "wrong_seed": row["task"]["seed"] = 42
        elif damage == "partial": row["phase"] = "training"
        elif damage == "manifest_sha": row["recovery_manifest_sha256"] = "0" * 64
        elif damage == "unsynced": row["wandb_sync_complete"] = False
        else: row["child_pid"] = 123
    modify(complete, "inspection-latest.json", change)
    with pytest.raises(ValueError): final.finalize(*complete)
    assert not (complete[0] / "cleanup-authorization.json").exists()


@pytest.mark.parametrize("filename", ["inspection-latest.json", "scheduler-snapshot.json"])
def test_stale_final_snapshots_refuse_authority(complete, filename):
    modify(complete, filename, lambda value: value.update(timestamp=(datetime.now(timezone.utc) - timedelta(minutes=6)).isoformat()))
    with pytest.raises(ValueError, match="Refresh"): final.finalize(*complete)


@pytest.mark.parametrize("damage", ["missing_archived", "running_gpu", "running_prepare", "bad_source", "nonzero_transport", "proof_tamper"])
def test_scheduler_requires_all_terminal_fresh_owned_evidence(complete, damage):
    def change(value):
        if damage == "missing_archived": value["jobs"].pop()
        elif damage == "running_gpu": value["jobs"][0]["status"] = "Running"
        elif damage == "running_prepare": value["jobs"][8]["status"] = "Running"
        elif damage == "bad_source": value["jobs"][0]["source"] = "historical"
        elif damage == "nonzero_transport": value["cli_returncode"] = 1
        else: Path(value["raw_proof"]).write_text("tampered")
    modify(complete, "scheduler-snapshot.json", change)
    with pytest.raises(ValueError): final.finalize(*complete)


def test_current_inspection_exemption_cannot_be_a_gpu_job(complete):
    modify(complete, "inspection-latest.json", lambda value: value.update(collector_source_job_id="lm-mpi-job-0abc"))
    with pytest.raises(ValueError, match="exemption"): final.finalize(*complete)


def test_raw_evidence_tampering_and_scope_expansion_fail_closed(complete):
    path = complete[0] / "raw-wandb" / final.run_name(final.TASKS[0]) / "history.jsonl"
    path.write_text("{}\n")
    with pytest.raises(ValueError, match="raw diagnostic"): final.finalize(*complete)


def test_existing_stable_authorization_is_not_silently_replaced(complete):
    final.finalize(*complete)
    path = complete[0] / "cleanup-authorization.json"
    authorization = json.loads(path.read_text())
    authorization["diagnostic_manifests"][final.task_key(final.TASKS[0])] = "f" * 64
    write(path, authorization)
    before = path.read_bytes()
    with pytest.raises(ValueError, match="stable"): final.finalize(*complete)
    assert path.read_bytes() == before


def test_reused_original_uses_verified_metrics_json(complete):
    artifacts, old = complete
    path = old / "wandb-independent-verification.json"
    original = json.loads(path.read_text())
    first = next(iter(original.values()))
    first["reused"] = True
    directory = old / "raw-wandb" / first["name"]
    (directory / (first["name"] + ".json")).rename(directory / "metrics.json")
    write(path, original)
    assert final.finalize(artifacts, old)["verified_runs"] == 8


def test_verified_wandb_transport_uses_local_file_proof_without_claiming_cli_success(complete, monkeypatch):
    import cloud_repair_control as control
    called = []
    def verify_files(snapshot, artifacts, campaign):
        assert snapshot["collector_cli_returncode"] is None
        assert snapshot["collector_ssh_returncode"] is None
        called.append(artifacts)
    monkeypatch.setattr(control, "verify_control_files", verify_files)
    modify(complete, "inspection-latest.json", lambda value: value.update(
        collector_transport_source="wandb_aux_snapshot", collector_cli_returncode=None, collector_ssh_returncode=None))
    assert final.finalize(*complete)["verified_runs"] == 8 and len(called) == 2


def test_wandb_transport_label_without_local_file_proof_refuses_authority(complete):
    modify(complete, "inspection-latest.json", lambda value: value.update(
        collector_transport_source="wandb_aux_snapshot", collector_cli_returncode=None, collector_ssh_returncode=None))
    with pytest.raises(ValueError, match="W&B control file proof"): final.finalize(*complete)


def test_storage_at_reserve_refuses_authority(complete):
    modify(complete, "inspection-latest.json", lambda value: value["filesystems"][0].update(available_bytes=5_000_000_000))
    with pytest.raises(ValueError, match="Storage"): final.finalize(*complete)
