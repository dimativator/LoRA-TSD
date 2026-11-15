"""Independent repair evidence verification without network or campaign mutation."""

import base64
import copy
import hashlib
import io
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import zipfile

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts/cloud"))
import cloud_repair_monitor as monitor


def encoded(value):
    return (json.dumps(value) + "\n").encode()


@pytest.fixture
def evidence(tmp_path, monkeypatch):
    source = Path(__file__).resolve().parents[1]
    bundle = tmp_path / "runtime.zip"
    with zipfile.ZipFile(bundle, "w") as archive:
        for name in ("paper_8b_campaign.py", "paper_8b_methods.json"):
            archive.write(source / "scripts/brain_lab" / name, "scripts/brain_lab/" + name)
    sha = monitor.digest(bundle)
    checksum = tmp_path / "runtime.sha256"
    checksum.write_text(sha)
    monkeypatch.setattr(monitor, "BUNDLE", bundle)
    monkeypatch.setattr(monitor, "BUNDLE_SHA", checksum)
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    campaign = {"campaign": monitor.CAMPAIGN, "tasks": monitor.TASKS,
                "target_runs": 8, "runtime_bundle_sha256": sha}
    task = monitor.TASKS[0]
    name = monitor.run_name(task)
    command, lr = monitor.expected_command(task, 8, sha)
    rows = []
    for step in range(1, 3001):
        rows.append({"step": step, "train/loss": 1., "lr_A_used": lr,
                     "optimizer/update_norm": .01})
        if step % 250 == 0:
            rows.append({"step": step, "validation_score": .5 + step / 10000})
    result = {"run_name": name, "optimizer": task["optimizer"],
              "dataset": task["dataset"], "seed": task["seed"],
              "model": "meta-llama/Meta-Llama-3.1-8B", "steps": 3000,
              "reported_accuracy": .8, "reported_accuracy_step": 3000}
    run = SimpleNamespace(name=name, id="repair01", path=["andrey", "LoRA-Muon", "repair01"],
                          url="https://wandb-radfan.ru/andrey/LoRA-Muon/runs/repair01",
                          group=monitor.CAMPAIGN, state="finished", created_at="2026-11-15T00:00:00Z",
                          config={"campaign": monitor.CAMPAIGN, **task, "worker": 0,
                                  "steps": 3000, "eval_every": 250, "effective_batch": 32,
                                  "model_revision": monitor.REVISION, "runtime_bundle_sha256": sha,
                                  "microbatch": 8, "command": command},
                          summary={"verified": True, "returncode": 0, "checkpoint_count": 0,
                                   "optimizer_step": 3000})
    files = {"history.jsonl": b"".join(encoded(row) for row in rows),
             name + ".json": encoded(result), "profile-micro8-profile.json": b"{}\n"}
    manifest = {"campaign": monitor.CAMPAIGN, "run_name": name, "wandb_id": run.id,
                "runtime_bundle_sha256": sha, "model_revision": monitor.REVISION,
                "command": command, "files": []}

    def refresh_manifest():
        manifest["files"] = [{"path": ("cloud0-lora_rite-micro8/profile.json"
                              if filename.startswith("profile-") else filename),
                              "wandb_name": filename, "bytes": len(data),
                              "sha256": hashlib.sha256(data).hexdigest()}
                             for filename, data in files.items() if filename != "diagnostic-manifest.json"]
        files["diagnostic-manifest.json"] = encoded(manifest)

    refresh_manifest()

    def metadata(filename):
        data = files[filename]
        return SimpleNamespace(name=filename, size=len(data),
                               url=f"https://wandb-radfan.ru/files/andrey/LoRA-Muon/{run.id}/{filename}",
                               md5=base64.b64encode(hashlib.md5(data).digest()).decode())

    run.file = metadata
    run.files = lambda: [metadata(filename) for filename in files]
    run.scan_history = lambda **kwargs: [{"optimizer_step": row["step"],
                                          "validation_score": row["validation_score"]}
                                         for row in rows if "validation_score" in row]

    def download(url, destination, expected_size, run_id, api_key, md5):
        data = files[url.rsplit("/", 1)[1]]
        assert len(data) == expected_size
        destination.write_bytes(data)

    monkeypatch.setattr(monitor, "download_run_file", download)
    status = {"task": task, "phase": "complete", "wandb_id": run.id,
              "command": command, "returncode": 0,
              "recovery_manifest_sha256": hashlib.sha256(files["diagnostic-manifest.json"]).hexdigest()}
    return SimpleNamespace(campaign=campaign, task=task, run=run, rows=rows, result=result,
                           files=files, manifest=manifest, refresh=refresh_manifest,
                           artifacts=artifacts, status=status)


def verify(evidence):
    return monitor.verify_finished(evidence.run, evidence.task, evidence.campaign,
                                   evidence.artifacts, "not-a-real-key", evidence.status)


def test_independently_downloads_every_diagnostic_and_binds_manifest(evidence):
    evidence.status["recovery_manifest_sha256"] = hashlib.sha256(
        evidence.files["diagnostic-manifest.json"]).hexdigest()
    result = verify(evidence)
    assert result["train_records"] == 3000 and result["validation_points"] == 12
    assert result["best_step"] == 3000 and result["best_accuracy"] == .8
    assert len(result["verified_files"]) == 4
    assert monitor.cache_valid(result, evidence.artifacts / "raw-wandb" / evidence.run.name)


@pytest.mark.parametrize("field,value", [("runtime_bundle_sha256", "a" * 64),
    ("model_revision", "wrong"), ("seed", 42), ("worker", 1), ("steps", 2500)])
def test_rejects_unexpected_runtime_and_task_identity(evidence, field, value):
    evidence.run.config[field] = value
    with pytest.raises(monitor.VerificationError):
        verify(evidence)


def test_rejects_cli_hp_drift(evidence):
    command = evidence.run.config["command"][:]
    command[command.index("--lr-a") + 1] = "0.5"
    evidence.run.config["command"] = command
    evidence.status.pop("command")
    with pytest.raises(monitor.VerificationError, match="frozen protocol"):
        verify(evidence)


@pytest.mark.parametrize("damage", ["missing_train", "missing_final", "nan_loss", "wrong_lr", "wrong_best"])
def test_partial_nonfinite_or_mismatched_evidence_never_credited(evidence, damage):
    if damage == "missing_train":
        evidence.rows.pop(0)
    elif damage == "missing_final":
        evidence.rows.pop()
    elif damage == "nan_loss":
        evidence.rows[0]["train/loss"] = float("nan")
    elif damage == "wrong_lr":
        evidence.rows[0]["lr_A_used"] = .1
    else:
        evidence.result["reported_accuracy"] = .7
    evidence.files["history.jsonl"] = b"".join(encoded(row) for row in evidence.rows)
    evidence.files[evidence.run.name + ".json"] = encoded(evidence.result)
    evidence.refresh()
    with pytest.raises(monitor.VerificationError):
        verify(evidence)


@pytest.mark.parametrize("damage", ["sha", "path", "mapping", "duplicate", "identity", "shared_sha"])
def test_manifest_integrity_and_mapping_fail_closed(evidence, damage):
    if damage == "sha":
        evidence.manifest["files"][0]["sha256"] = "0" * 64
    elif damage == "path":
        evidence.manifest["files"][0]["path"] = "../history.jsonl"
    elif damage == "mapping":
        evidence.manifest["files"][-1]["path"] = "cloud7-lora_rite-micro8/profile.json"
    elif damage == "duplicate":
        evidence.manifest["files"].append(evidence.manifest["files"][0])
    elif damage == "identity":
        evidence.manifest["wandb_id"] = "another"
    else:
        evidence.status["recovery_manifest_sha256"] = "0" * 64
    evidence.files["diagnostic-manifest.json"] = encoded(evidence.manifest)
    with pytest.raises(monitor.VerificationError):
        verify(evidence)


def test_model_checkpoint_artifact_rejected(evidence):
    evidence.files["weights.pt"] = b"checkpoint"
    with pytest.raises(monitor.VerificationError, match="checkpoint"):
        verify(evidence)


def test_tampered_cached_diagnostic_fails_validation(evidence):
    result = verify(evidence)
    directory = evidence.artifacts / "raw-wandb" / evidence.run.name
    (directory / "profile-micro8-profile.json").write_text("changed")
    assert not monitor.cache_valid(result, directory)


def test_duplicate_run_rejected_and_counter_is_eight(evidence):
    api = SimpleNamespace(runs=lambda *args, **kwargs: [evidence.run, copy.copy(evidence.run)])
    result = monitor.monitor(api, evidence.campaign, evidence.artifacts, "key", True)
    assert result["target"] == 8 and result["verified_completed"] == 0
    assert len(result["verification_errors"]) == 1
    assert result["queued_count"] == 7
    assert not result["all_histories_verified"]


def test_interrupted_attempt_is_historical_and_not_credited(evidence):
    monitor.save(evidence.artifacts / "interrupted-attempts.json", {evidence.run.id: {}})
    api = SimpleNamespace(runs=lambda *args, **kwargs: [evidence.run])
    result = monitor.monitor(api, evidence.campaign, evidence.artifacts, "key", True)
    assert result["verified_completed"] == 0 and result["queued_count"] == 8
    assert result["interrupted_attempts"][0]["id"] == evidence.run.id


@pytest.mark.parametrize("shared_phase", [None, "training", "reserved"])
def test_final_evaluation_pending_is_active_validation(evidence, shared_phase):
    evidence.run.state = "running"
    evidence.run.summary["verified"] = False
    evidence.rows.pop()
    if shared_phase is not None:
        monitor.save(evidence.artifacts / "inspection-latest.json",
                     {"campaign": monitor.CAMPAIGN,
                      "tasks": [{**evidence.status, "phase": shared_phase, "step": 1500}]})
    api = SimpleNamespace(runs=lambda *args, **kwargs: [evidence.run])
    result = monitor.monitor(api, evidence.campaign, evidence.artifacts, "key", True)
    assert result["verified_completed"] == 0 and result["phase_counts"]["validation"] == 1
    assert result["active"][0]["evaluation_count"] == 11


def test_monitor_cache_reuses_complete_evidence_and_rechecks_new_artifacts(evidence):
    api = SimpleNamespace(runs=lambda *args, **kwargs: [evidence.run])
    first = monitor.monitor(api, evidence.campaign, evidence.artifacts, "key", True)
    assert first["verified_completed"] == 1 and first["queued_count"] == 7
    second = monitor.monitor(api, evidence.campaign, evidence.artifacts, "key", False)
    assert second["verified_completed"] == 1
    evidence.files["late-checkpoint.safetensors"] = b"wrong"
    rejected = monitor.monitor(api, evidence.campaign, evidence.artifacts, "key", False)
    assert rejected["verified_completed"] == 0 and rejected["verification_errors"]


def test_unexpected_run_blocks_final_completion(evidence):
    other = copy.copy(evidence.run)
    other.name = "old180_run"
    other.id = "unexpected"
    api = SimpleNamespace(runs=lambda *args, **kwargs: [evidence.run, other])
    result = monitor.monitor(api, evidence.campaign, evidence.artifacts, "key", True)
    assert result["verified_completed"] == 1 and result["unexpected_runs"]
    assert not result["all_histories_verified"]


def test_shared_complete_without_public_evidence_is_pending_not_active(evidence):
    monitor.save(evidence.artifacts / "inspection-latest.json",
                 {"campaign": monitor.CAMPAIGN, "tasks": [evidence.status]})
    api = SimpleNamespace(runs=lambda *args, **kwargs: [])
    result = monitor.monitor(api, evidence.campaign, evidence.artifacts, "key", True)
    assert not result["active"] and result["verified_completed"] == 0
    assert len(result["pending_independent_verification"]) == 1


@pytest.mark.parametrize("cached", [False, True])
def test_completed_shared_record_requires_manifest_sha_for_new_and_cached_credit(evidence, cached):
    api = SimpleNamespace(runs=lambda *args, **kwargs: [evidence.run])
    if cached:
        assert monitor.monitor(api, evidence.campaign, evidence.artifacts, "key", True)["verified_completed"] == 1
    evidence.status.pop("recovery_manifest_sha256")
    monitor.save(evidence.artifacts / "inspection-latest.json",
                 {"campaign": monitor.CAMPAIGN, "tasks": [evidence.status]})
    result = monitor.monitor(api, evidence.campaign, evidence.artifacts, "key", True)
    assert result["verified_completed"] == 0 and result["verification_errors"]


@pytest.mark.parametrize("phase", ["syncing", None])
def test_stale_or_missing_shared_completion_remains_pending(evidence, phase):
    record = {**evidence.status, "phase": phase}
    monitor.save(evidence.artifacts / "inspection-latest.json",
                 {"campaign": monitor.CAMPAIGN, "tasks": [record] if phase else []})
    api = SimpleNamespace(runs=lambda *args, **kwargs: [evidence.run])
    result = monitor.monitor(api, evidence.campaign, evidence.artifacts, "key", True)
    assert result["verified_completed"] == 0 and not result["verification_errors"]
    assert len(result["pending_independent_verification"]) == 1


def test_empty_diagnostic_verified_with_authenticated_bounded_download(evidence, monkeypatch):
    evidence.files["empty.log"] = b""
    response = io.BytesIO(b"")
    class Opener:
        def open(self, request, timeout):
            assert request.get_header("Authorization").startswith("Basic ")
            assert timeout == 45
            return response
    monkeypatch.setattr(monitor, "build_opener", lambda *args: Opener())
    path = monitor.fetch(evidence.run, "empty.log", evidence.artifacts, "key")
    assert path.read_bytes() == b""
