"""Preserve every unique recovery file, then archive only independently synced evidence."""

import os
from pathlib import Path
import secrets
import shutil
import stat
import tempfile
from typing import Callable

from cloud14_assets import PYTHON, affected_paths
from cloud14_core import CAMPAIGN, STATE, decode_audit_environment, digest, locked, now, require_space, save
from cloud14_final_audit import read_record, record_phases, recovery_inventory


def complete(state: Path, audit: dict) -> None:
    assert audit["campaign"] == CAMPAIGN and audit["verified_runs"] == 180
    assert audit["all_owned_jobs_terminal"] is True
    assert not state.is_symlink()
    workers = record_phases(state / "workers")
    tasks = record_phases(state / "tasks")
    assert len(workers) == 14 and set(workers.values()) == {"complete"}
    assert len(tasks) == 138 and set(tasks.values()) == {"complete"}


def manifest(state: Path, check_space: Callable[[], object] | None = None) -> tuple[dict, dict[str, Path]]:
    records = recovery_inventory(state, check_space)
    assert not any(row["kind"] == "symlink" for row in records), "Recovery links require separate diagnosis"
    root = state / "pending-wandb-sync"
    files = []
    unique = {}
    for row in records:
        if row["kind"] != "file":
            continue
        path = root / row["path"]
        assert stat.S_ISREG(path.lstat().st_mode) and path.resolve().is_relative_to(root.resolve())
        assert path.suffix in [".log", ".json", ".jsonl", ".txt", ".wandb"], "Unexpected recovery file type"
        sha256 = digest(path)
        assert path.stat().st_size == row["bytes"], "Recovery changed while hashing"
        files.append({"path": row["path"], "bytes": row["bytes"], "sha256": sha256})
        unique.setdefault(sha256, path)
        if check_space is not None:
            check_space()
    value = {"campaign": CAMPAIGN, "files": files,
             "original_bytes": sum(row["bytes"] for row in files),
             "unique_files": [{"name": "blob-" + sha256 + ".bin", "sha256": sha256,
                               "bytes": path.stat().st_size} for sha256, path in sorted(unique.items())]}
    return value, unique


def upload(directory: Path, state: Path = STATE) -> None:
    import wandb
    audit = decode_audit_environment("PAPER8B_RECOVERY_AUDIT")
    complete(state, audit)
    with locked(state / "recovery-sync.lock"):
        previous = read_record(state / "recovery-upload.json")
        assert previous is None or previous["phase"] in ["initializing", "uploading"], (
            "Existing recovery upload must be adopted or diagnosed, never duplicated")
        value, unique = manifest(state, lambda: require_space(affected_paths()))
        assert unique, "No recovery files to upload"
        planned = sum(path.stat().st_size for path in unique.values()) + 20_000_000
        require_space(affected_paths(), {directory: planned, Path("/home/jovyan"): 20_000_000,
                                         state: 2_000_000})
        payload = directory / "payload"
        payload.mkdir()
        for sha256, path in unique.items():
            target = payload / ("blob-" + sha256 + ".bin")
            shutil.copyfile(path, target)
            assert digest(target) == sha256
        save(payload / "recovery-manifest.json", value)
        require_space(affected_paths(), {state: (payload / "recovery-manifest.json").stat().st_size})
        manifest_sha256 = digest(payload / "recovery-manifest.json")
        run_id = previous["run_id"] if previous else secrets.token_hex(8)
        receipt = previous or {"timestamp": now(), "campaign": CAMPAIGN, "phase": "initializing",
                   "run_id": run_id, "run_path": "andrey/LoRA-Muon/" + run_id,
                   "run_url": "https://wandb-radfan.ru/andrey/LoRA-Muon/runs/" + run_id,
                   "manifest_sha256": manifest_sha256,
                   "unique_files": len(unique), "original_files": len(value["files"])}
        assert receipt["manifest_sha256"] == manifest_sha256
        save(state / "recovery-manifest.json", value)
        save(state / "recovery-upload.json", receipt)
        run = wandb.init(id=run_id, resume="allow", entity="andrey", project="LoRA-Muon", group=CAMPAIGN,
                         name=CAMPAIGN + "-recovery-evidence", job_type="recovery-sync",
                         mode="online", dir=str(directory),
                         settings=wandb.Settings(save_code=False, disable_git=True))
        assert run.id == run_id and run.path == receipt["run_path"]
        receipt.update(phase="uploading")
        save(state / "recovery-upload.json", receipt)
        for path in sorted(payload.iterdir()):
            require_space(affected_paths())
            run.save(str(path), base_path=str(payload), policy="now")
        run.log({"recovery/unique_files": len(unique), "recovery/original_files": len(value["files"]),
                 "recovery/original_bytes": value["original_bytes"]})
        run.finish(exit_code=0)
        receipt.update(phase="uploaded_pending_independent_verification", finished_at=now())
        save(state / "recovery-upload.json", receipt)


def archive(state: Path = STATE) -> dict:
    audit = decode_audit_environment("PAPER8B_RECOVERY_CONFIRM")
    complete(state, audit)
    assert audit["all_unique_files_downloaded_sha256_verified"] is True
    with locked(state / "recovery-sync.lock"):
        receipt = read_record(state / "recovery-upload.json")
        value = read_record(state / "recovery-manifest.json")
        assert receipt["phase"] == "uploaded_pending_independent_verification"
        assert receipt["manifest_sha256"] == audit["manifest_sha256"]
        assert receipt["run_path"] == audit["run_path"]
        assert receipt["unique_files"] == audit["unique_files"]
        assert digest(state / "recovery-manifest.json") == audit["manifest_sha256"]
        current, _ = manifest(state, lambda: require_space(affected_paths()))
        assert current == value, "Recovery files changed since the verified upload"
        destination = state / ("reconciled-wandb-sync-" + audit["manifest_sha256"][:16])
        assert not destination.exists() and not destination.is_symlink()
        require_space(affected_paths(), {state: 1_000_000})
        (state / "pending-wandb-sync").rename(destination)
        result = {"campaign": CAMPAIGN, "phase": "archived_verified", "timestamp": now(),
                  "archive": str(destination), "verification": audit,
                  "all_unique_files_uploaded_and_independently_verified": True}
        save(state / "recovery-reconciled.json", result)
        receipt.update(phase="independently_verified_archived", archive=str(destination))
        save(state / "recovery-upload.json", receipt)
        return result


def dispatch(mode: str, directory: Path | None) -> bool:
    if mode == "upload-child":
        upload(directory)
        return False
    if mode == "archive-child":
        archive()
        return False
    from cloud14_runtime import owned_process, watch
    with tempfile.TemporaryDirectory(prefix="paper8b-recovery-") as temporary:
        working = Path(temporary)
        command = [str(PYTHON), str(Path(__file__).with_name("cloud14_final_audit.py")),
                   mode + "-child", "--work-dir", str(working)]
        with owned_process(command, working, os.environ.copy()) as process:
            watch(process, working, {"phase": mode}, STATE / ("recovery-" + mode + "-watch.json"))
        if process.returncode:
            print((working / "process.log").read_text()[-5000:], flush=True)
            raise RuntimeError(f"Recovery {mode} failed; preserve spool and diagnose")
    return True
