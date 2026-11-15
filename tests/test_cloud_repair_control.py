"""Guarded W&B inspection fallback, no submissions, credentials or network."""

import copy
from datetime import datetime, timedelta, timezone
import gzip
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace
from types import ModuleType

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts/cloud"))
import cloud_repair_control as control
import cloud_repair_collect as collector


@pytest.fixture
def fallback(tmp_path, monkeypatch):
    note = "repair8b_inspect-3_2026-11-15T120000_0000"
    auxiliary = {"job_id": "lm-mpi-job-abcd", "note": note, "mode": "inspect", "gpus": 0}
    campaign = {"campaign": control.CAMPAIGN, "runtime_bundle_sha256": "a" * 64,
                "auxiliary_jobs": [auxiliary], "jobs": [], "assets_ready_verified": False}
    control.save(tmp_path / "campaign.json", campaign)
    paths = [Path("/"), Path("/tmp"), tmp_path / "assets"]
    monkeypatch.setattr(collector, "affected_paths", lambda: paths)
    snapshot = {"campaign": control.CAMPAIGN, "timestamp": datetime.now(timezone.utc).isoformat(),
                "assets_ready": True, "preparation": {"phase": "ready"}, "tasks": [], "workers": [],
                "filesystems": [{"requested_path": str(path), "available_bytes": 9_000_000_000,
                                 "status": "ok"} for path in paths]}
    run = SimpleNamespace(name=note, id="control1", group=control.GROUP,
                          path=["andrey", "LoRA-Muon", "control1"], state="finished",
                          url="https://wandb-radfan.ru/andrey/LoRA-Muon/runs/control1",
                          config={"campaign": control.CAMPAIGN, "note": note, "mode": "inspect",
                                  "runtime_bundle_sha256": "a" * 64}, summary={})
    files = {}

    def refresh():
        data = control.snapshot_bytes(snapshot)
        compressed = gzip.compress(data, mtime=0)
        manifest = {"campaign": control.CAMPAIGN, "note": note, "mode": "inspect", "runtime_bundle_sha256": "a" * 64,
                    "snapshot_timestamp": snapshot["timestamp"], "file": "snapshot.json.gz", "bytes": len(compressed),
                    "sha256": hashlib.sha256(compressed).hexdigest(), "uncompressed_bytes": len(data),
                    "uncompressed_sha256": hashlib.sha256(data).hexdigest()}
        files.update({"snapshot.json.gz": compressed, "snapshot-manifest.json": json.dumps(manifest).encode()})
        run.summary.update(snapshot_verified=True, snapshot_sha256=manifest["sha256"], returncode=0, checkpoint_count=0,
                           snapshot_manifest_sha256=hashlib.sha256(files["snapshot-manifest.json"]).hexdigest())
    refresh()
    run.file = lambda name: SimpleNamespace(size=len(files[name]))
    def fetch(run, name, directory, key):
        assert key == "not-real"
        path = directory / name
        path.write_bytes(files[name])
        return path
    monkeypatch.setattr(control, "fetch", fetch)
    api = SimpleNamespace(runs=lambda *args, **kwargs: [run])
    return SimpleNamespace(artifacts=tmp_path, auxiliary=auxiliary, campaign=campaign,
                           snapshot=snapshot, run=run, files=files, refresh=refresh, api=api)


def collect(fallback):
    return control.collect_control(fallback.api, fallback.auxiliary, fallback.campaign,
                                   fallback.artifacts, "not-real")


def test_finished_exact_note_snapshot_is_independently_verified_and_truthful(fallback):
    result = collect(fallback)
    assert result["collector_transport_source"] == "wandb_aux_snapshot"
    assert result["collector_cli_returncode"] is None and result["collector_ssh_returncode"] is None
    assert result["collector_source_job_id"] == fallback.auxiliary["job_id"]
    assert result["collector_wandb_file_verified"] is True
    control.verify_control_files(result, fallback.artifacts, fallback.campaign)
    saved = json.loads((fallback.artifacts / "campaign.json").read_text())
    assert saved["assets_ready_verified"] is True and not saved["jobs"]


def test_nonready_progress_snapshot_never_credits_assets_early(fallback):
    fallback.snapshot.update(assets_ready=False, preparation={"phase": "downloading"})
    fallback.refresh()
    result = collect(fallback)
    assert result["collector_assets_ready_verified"] is False
    assert json.loads((fallback.artifacts / "campaign.json").read_text())["assets_ready_verified"] is False


@pytest.mark.parametrize("damage", ["duplicate", "running", "wrong_group", "wrong_note", "wrong_bundle", "bad_exit"])
def test_wrong_or_unfinished_control_identity_never_credits(fallback, damage):
    if damage == "duplicate": fallback.api.runs = lambda *args, **kwargs: [fallback.run, copy.copy(fallback.run)]
    elif damage == "running": fallback.run.state = "running"
    elif damage == "wrong_group": fallback.run.group = "other"
    elif damage == "wrong_note": fallback.run.config["note"] = "different"
    elif damage == "wrong_bundle": fallback.run.config["runtime_bundle_sha256"] = "b" * 64
    else: fallback.run.summary["returncode"] = 1
    with pytest.raises(ValueError): collect(fallback)
    assert not (fallback.artifacts / "inspection-latest.json").exists()


def test_stale_snapshot_refused_without_new_inspection_or_assets_credit(fallback):
    fallback.snapshot["timestamp"] = (datetime.now(timezone.utc) - timedelta(minutes=6)).isoformat()
    fallback.refresh()
    with pytest.raises(ValueError, match="timestamp"): collect(fallback)
    assert not (fallback.artifacts / "inspection-latest.json").exists()


@pytest.mark.parametrize("damage", ["sha", "manifest", "crc", "bomb_bound"])
def test_all_gzip_and_manifest_checks_fail_closed(fallback, damage):
    if damage == "sha":
        fallback.files["snapshot.json.gz"] += b"x"
    elif damage == "manifest":
        fallback.files["snapshot-manifest.json"] += b" "
    elif damage == "crc":
        payload = bytearray(fallback.files["snapshot.json.gz"])
        payload[-8] ^= 1
        fallback.files["snapshot.json.gz"] = bytes(payload)
        manifest = json.loads(fallback.files["snapshot-manifest.json"])
        manifest["sha256"] = hashlib.sha256(payload).hexdigest()
        fallback.files["snapshot-manifest.json"] = json.dumps(manifest).encode()
        fallback.run.summary.update(snapshot_sha256=manifest["sha256"],
            snapshot_manifest_sha256=hashlib.sha256(fallback.files["snapshot-manifest.json"]).hexdigest())
    else:
        manifest = json.loads(fallback.files["snapshot-manifest.json"])
        manifest["uncompressed_bytes"] = control.MAX_BYTES + 1
        fallback.files["snapshot-manifest.json"] = json.dumps(manifest).encode()
        fallback.run.summary["snapshot_manifest_sha256"] = hashlib.sha256(fallback.files["snapshot-manifest.json"]).hexdigest()
    with pytest.raises((ValueError, gzip.BadGzipFile)): collect(fallback)
    assert not (fallback.artifacts / "inspection-latest.json").exists()


def test_postcollection_tampering_is_rejected_for_final_authority(fallback):
    result = collect(fallback)
    result["tasks"] = [{"invented": "completion"}]
    with pytest.raises(ValueError, match="differs from collected"): control.verify_control_files(result, fallback.artifacts, fallback.campaign)


def test_select_only_existing_unique_owned_cpu_note(fallback):
    assert control.select_auxiliary(fallback.campaign, note=fallback.auxiliary["note"]) == fallback.auxiliary
    assert control.select_auxiliary(fallback.campaign, job_id=fallback.auxiliary["job_id"]) == fallback.auxiliary
    with pytest.raises(ValueError): control.select_auxiliary(fallback.campaign, note="unknown")
    fallback.campaign["auxiliary_jobs"].append(copy.copy(fallback.auxiliary))
    with pytest.raises(ValueError): control.select_auxiliary(fallback.campaign, note=fallback.auxiliary["note"])


def test_upload_parent_uses_only_owned_guarded_child_and_tmp(fallback, monkeypatch):
    from contextlib import contextmanager
    import cloud_repair_runtime as runtime
    observed = []
    monkeypatch.setenv("REPAIR_CONTROL_NOTE", fallback.auxiliary["note"])
    monkeypatch.setattr(control, "require_space", lambda paths, planned: observed.append(("space", planned)))
    @contextmanager
    def owned(command, directory, env):
        assert Path(command[0]) == control.PYTHON
        assert command[2] == "upload-child" and directory.parent == Path("/tmp")
        assert json.loads(Path(command[-1]).read_text())["campaign"] == control.CAMPAIGN
        observed.append(("guarded", command))
        yield SimpleNamespace(wait=lambda: None, returncode=0)
    monkeypatch.setattr(runtime, "owned_process", owned)
    control.upload_snapshot(fallback.snapshot)
    assert [row[0] for row in observed] == ["space", "guarded"]


def test_temp_wandb_paths_restore_environment_and_stay_owned(tmp_path, monkeypatch):
    monkeypatch.setenv("WANDB_DIR", "before")
    with control.temporary_wandb_paths(tmp_path):
        import os
        assert Path(os.environ["WANDB_DIR"]).parent == tmp_path
        assert Path(os.environ["WANDB_DATA_DIR"]).parent == tmp_path
    import os
    assert os.environ["WANDB_DIR"] == "before"


def test_upload_child_sdk_summary_api_uses_mapping_and_flat_temp_files(fallback, monkeypatch):
    import tempfile
    class Summary(dict):
        def update(self, mapping):
            super().update(mapping)
    summary, uploaded, finished = Summary(), [], []
    run = SimpleNamespace(summary=summary,
        save=lambda path, base_path, policy: uploaded.append((Path(path).name, Path(base_path), policy)),
        finish=lambda exit_code: finished.append(exit_code))
    fake = SimpleNamespace(init=lambda **kwargs: run, Settings=lambda **kwargs: None)
    monkeypatch.setitem(sys.modules, "wandb", fake)
    monkeypatch.setenv("REPAIR_CONTROL_NOTE", fallback.auxiliary["note"])
    monkeypatch.setenv("WANDB_BASE_URL", "https://wandb-radfan.ru")
    monkeypatch.setenv("WANDB_API_KEY", "not-real")
    checksum = fallback.artifacts / "bundle.sha256"
    checksum.write_text("a" * 64)
    monkeypatch.setattr(control, "BUNDLE_SHA", checksum)
    monkeypatch.setattr(control, "require_space", lambda *args: [])
    with tempfile.TemporaryDirectory(prefix="repair-control-test-", dir="/tmp") as temporary:
        source = Path(temporary) / "source.json"
        source.write_text(json.dumps(fallback.snapshot))
        control.upload_child(source)
        assert {name for name, directory, policy in uploaded} == {"snapshot.json.gz", "snapshot-manifest.json"}
        assert all(directory == source.parent and policy == "now" for name, directory, policy in uploaded)
    assert finished == [0] and summary["snapshot_verified"] is True


def test_remote_upload_module_import_does_not_need_local_monitor_hardening(monkeypatch):
    monkeypatch.setitem(sys.modules, "cloud_repair_monitor", ModuleType("cloud_repair_monitor"))
    monkeypatch.setitem(sys.modules, "cloud14_monitor", ModuleType("cloud14_monitor"))
    specification = importlib.util.spec_from_file_location("isolated_remote_control", Path(control.__file__))
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    assert callable(module.upload_snapshot) and callable(module.upload_child)
    assert module.SHA256.fullmatch("a" * 64)
