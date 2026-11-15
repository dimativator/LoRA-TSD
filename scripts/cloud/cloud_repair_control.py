"""A single owned CPU inspection fallback through authenticated W&B files.

Remote uploads run only in an owned, disk-guarded child and temporary directory.
Local collection adopts an existing exact auxiliary note; it never submits jobs.
"""

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import gzip
import hashlib
import io
import json
import netrc
import os
from pathlib import Path
import re
import tempfile

from cloud_repair_collect import validate_snapshot
from cloud_repair_common import (BUNDLE_SHA, CAMPAIGN, PYTHON, affected_paths,
                                 locked, now, require_space, save)
from cloud_repair_submit import ARTIFACTS


GROUP = CAMPAIGN + "-control"
MAX_BYTES = 16_000_000
FAILURE_LOG_BYTES = 12_288
SHA256 = re.compile(r"^[a-f0-9]{64}$")


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def fetch(*args, **kwargs):
    # The downloader belongs to local collection. CPU upload must not import
    # local-only monitor hardening absent from the historical remote helper.
    from cloud_repair_monitor import fetch as implementation
    return implementation(*args, **kwargs)


def snapshot_bytes(snapshot: dict) -> bytes:
    require(snapshot.get("campaign") == CAMPAIGN, "Control snapshot belongs to another campaign")
    data = (json.dumps(snapshot, sort_keys=True, allow_nan=False) + "\n").encode()
    require(0 < len(data) <= MAX_BYTES, "Control snapshot is oversized")
    return data


def note_valid(note: str) -> bool:
    return isinstance(note, str) and note.startswith("repair8b_inspect-") and bool(re.fullmatch(r"[A-Za-z0-9_.-]{1,240}", note))


def failure_log_tail(path: Path) -> str:
    """Bounded diagnostics with real credentials removed before stdout export."""
    secrets = sorted({os.environ[name].encode() for name in ("WANDB_API_KEY", "HF_TOKEN")
                      if os.environ.get(name)}, key=len, reverse=True)
    padding = max((len(value) for value in secrets), default=0)
    require(not path.is_symlink(), "Owned control failure log is a link")
    with path.open("rb") as stream:
        stream.seek(0, 2)
        start = max(0, stream.tell() - FAILURE_LOG_BYTES - padding)
        stream.seek(start)
        data = stream.read(FAILURE_LOG_BYTES + padding)
    for value in secrets:
        data = data.replace(value, b"<redacted>")
    # If the read started inside a credential, its partial prefix cannot be
    # matched above. Discard the entire boundary padding after redaction.
    if start:
        data = data[padding:]
    decoded = data[-FAILURE_LOG_BYTES:].decode(errors="replace")
    return decoded.encode()[-FAILURE_LOG_BYTES:].decode(errors="ignore")


def upload_snapshot(snapshot: dict) -> None:
    """Called by the CPU inspector after printing its existing CRC snapshot."""
    note = os.environ.get("REPAIR_CONTROL_NOTE", "")
    require(note_valid(note), "Missing exact owned inspection note")
    data = snapshot_bytes(snapshot)
    require_space(affected_paths(), {Path("/tmp"): len(data) * 3 + 5_000_000})
    from cloud_repair_runtime import owned_process
    with tempfile.TemporaryDirectory(prefix="repair-control-", dir="/tmp") as temporary:
        directory = Path(temporary)
        path = directory / "source-snapshot.json"
        path.write_bytes(data)
        command = [str(PYTHON), str(Path(__file__).resolve()), "upload-child", "--snapshot", str(path)]
        succeeded = False
        try:
            with owned_process(command, directory, os.environ.copy()) as process:
                process.wait()
            require(process.returncode == 0, "Owned control upload failed; preserve its exact job/note and do not resubmit")
            succeeded = True
        finally:
            if not succeeded:
                try:
                    tail = failure_log_tail(directory / "process.log")
                    print("REPAIR_CONTROL_UPLOAD_FAILURE_LOG_TAIL_BEGIN\n" + tail
                          + "\nREPAIR_CONTROL_UPLOAD_FAILURE_LOG_TAIL_END", flush=True)
                except (OSError, ValueError) as error:
                    print("REPAIR_CONTROL_UPLOAD_FAILURE_LOG_UNAVAILABLE " + type(error).__name__, flush=True)


@contextmanager
def temporary_wandb_paths(directory: Path):
    names = ("WANDB_DIR", "WANDB_CACHE_DIR", "WANDB_CONFIG_DIR", "WANDB_DATA_DIR")
    previous = {name: os.environ.get(name) for name in names}
    try:
        for name in names:
            path = directory / name.lower()
            path.mkdir()
            os.environ[name] = str(path)
        yield
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def upload_child(snapshot_path: Path) -> None:
    note = os.environ.get("REPAIR_CONTROL_NOTE", "")
    require(note_valid(note), "Missing exact inspection note")
    require(os.environ.get("WANDB_BASE_URL") == "https://wandb-radfan.ru"
            and bool(os.environ.get("WANDB_API_KEY")), "Owned control upload credentials/host are unavailable")
    require(not snapshot_path.is_symlink() and snapshot_path.is_file()
            and snapshot_path.resolve().is_relative_to(Path("/tmp").resolve()), "Control upload source is not an owned temporary file")
    require(snapshot_path.stat().st_size <= MAX_BYTES, "Control source is oversized")
    snapshot = json.loads(snapshot_path.read_text())
    data = snapshot_bytes(snapshot)
    validate_snapshot(snapshot, "inspect")
    bundle = BUNDLE_SHA.read_text().strip()
    require(bool(SHA256.fullmatch(bundle)), "Repair bundle is not frozen")
    require_space(affected_paths(), {Path("/tmp"): len(data) * 3 + 5_000_000})
    directory = snapshot_path.parent
    payload = directory / "snapshot.json.gz"
    payload.write_bytes(gzip.compress(data, mtime=0))
    manifest = {"campaign": CAMPAIGN, "note": note, "mode": "inspect", "runtime_bundle_sha256": bundle,
                "snapshot_timestamp": snapshot["timestamp"], "file": payload.name,
                "bytes": payload.stat().st_size, "sha256": hashlib.sha256(payload.read_bytes()).hexdigest(),
                "uncompressed_bytes": len(data), "uncompressed_sha256": hashlib.sha256(data).hexdigest()}
    manifest_path = directory / "snapshot-manifest.json"
    save(manifest_path, manifest)
    import wandb
    with temporary_wandb_paths(directory):
        run = wandb.init(entity="andrey", project="LoRA-Muon", group=GROUP, name=note,
                         mode="online", dir=str(directory), resume="never",
                         config={"campaign": CAMPAIGN, "note": note, "mode": "inspect",
                                 "runtime_bundle_sha256": bundle},
                         settings=wandb.Settings(save_code=False, disable_git=True))
        try:
            run.save(str(payload), base_path=str(directory), policy="now")
            run.save(str(manifest_path), base_path=str(directory), policy="now")
            run.summary.update({"snapshot_verified": True, "snapshot_sha256": manifest["sha256"],
                                "snapshot_manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
                                "checkpoint_count": 0, "returncode": 0})
        except BaseException:
            run.finish(exit_code=1)
            raise
        run.finish(exit_code=0)


def select_auxiliary(campaign: dict, note: str | None = None, job_id: str | None = None) -> dict:
    require(campaign.get("campaign") == CAMPAIGN, "Control ledger belongs to another campaign")
    require(bool(note) != bool(job_id), "Select exactly one existing inspection note or job ID")
    matches = [row for row in campaign.get("auxiliary_jobs", []) if row.get("mode") == "inspect"
               and ((note and row.get("note") == note) or (job_id and row.get("job_id") == job_id))]
    require(len(matches) == 1 and matches[0].get("gpus") == 0 and note_valid(matches[0].get("note")),
            "No unique existing owned CPU inspection matches")
    require(bool(re.fullmatch(r"lm-mpi-job-[a-f0-9-]+", matches[0].get("job_id", ""))),
            "Inspection submission has no proven job identity")
    return matches[0]


def collect_control(api, auxiliary: dict, campaign: dict, artifacts: Path, api_key: str) -> dict:
    note = auxiliary["note"]
    selected = select_auxiliary(campaign, note=note)
    require(selected["job_id"] == auxiliary["job_id"]
            and bool(SHA256.fullmatch(campaign.get("runtime_bundle_sha256", ""))),
            "Control auxiliary identity or frozen bundle differs")
    runs = [run for run in api.runs("andrey/LoRA-Muon", filters={"group": GROUP}, per_page=100) if run.name == note]
    require(len(runs) == 1, "Control W&B run is absent or duplicated; reuse the same owned inspection note")
    run = runs[0]
    path = "/".join(run.path) if isinstance(run.path, (list, tuple)) else str(run.path)
    require(bool(re.fullmatch(r"[A-Za-z0-9_-]+", run.id)) and run.group == GROUP
            and path == "andrey/LoRA-Muon/" + run.id and run.state == "finished",
            "Control snapshot upload is not uniquely finished")
    require(all(run.config.get(key) == value for key, value in {"campaign": CAMPAIGN, "note": note,
                "mode": "inspect", "runtime_bundle_sha256": campaign["runtime_bundle_sha256"]}.items()),
            "Control run identity/bundle differs")
    require(run.summary.get("snapshot_verified") is True and run.summary.get("returncode") == 0
            and run.summary.get("checkpoint_count") == 0, "Control upload did not finish successfully")
    directory = artifacts / "control-snapshots" / note
    require(not directory.is_symlink() and directory.resolve().is_relative_to(artifacts.resolve()),
            "Control evidence path escapes repair artifacts")
    directory.mkdir(parents=True, exist_ok=True)
    for name in ("snapshot-manifest.json", "snapshot.json.gz"):
        require(0 < run.file(name).size <= MAX_BYTES, "Control upload file is oversized or empty")
    manifest_path = fetch(run, "snapshot-manifest.json", directory, api_key)
    require(hashlib.sha256(manifest_path.read_bytes()).hexdigest() == run.summary.get("snapshot_manifest_sha256"),
            "Control manifest summary SHA differs")
    manifest = json.loads(manifest_path.read_text())
    require(all(manifest.get(key) == value for key, value in {"campaign": CAMPAIGN, "note": note,
                "mode": "inspect", "runtime_bundle_sha256": campaign["runtime_bundle_sha256"],
                "file": "snapshot.json.gz"}.items()), "Control manifest identity differs")
    payload = fetch(run, "snapshot.json.gz", directory, api_key)
    compressed = payload.read_bytes()
    sha = hashlib.sha256(compressed).hexdigest()
    require(len(compressed) == manifest.get("bytes") and sha == manifest.get("sha256")
            and sha == run.summary.get("snapshot_sha256"), "Control gzip SHA/size differs")
    require(type(manifest.get("uncompressed_bytes")) is int and 0 < manifest["uncompressed_bytes"] <= MAX_BYTES,
            "Control uncompressed snapshot is oversized")
    # GzipFile verifies CRC and trailer while bounded reading prevents decompression bombs.
    with gzip.GzipFile(fileobj=io.BytesIO(compressed)) as stream:
        data = stream.read(MAX_BYTES + 1)
        require(len(data) <= MAX_BYTES and stream.read(1) == b"", "Control gzip exceeds its bound")
    require(len(data) == manifest["uncompressed_bytes"]
            and hashlib.sha256(data).hexdigest() == manifest.get("uncompressed_sha256"),
            "Control decoded snapshot SHA/size differs")
    snapshot = json.loads(data)
    require(snapshot.get("timestamp") == manifest.get("snapshot_timestamp"), "Control snapshot timestamp differs")
    ready = validate_snapshot(snapshot, "inspect")
    snapshot.update(collector_source_job_id=auxiliary["job_id"], collector_source_mode="inspect",
                    collector_received_at=now(), collector_transport_source="wandb_aux_snapshot",
                    collector_cli_returncode=None, collector_ssh_returncode=None, collector_timed_out=False,
                    collector_wandb_snapshot_verified=True, collector_control_run_id=run.id,
                    collector_wandb_file_verified=True,
                    collector_assets_ready_verified=ready,
                    collector_control_note=note, collector_snapshot_sha256=sha,
                    collector_manifest_sha256=run.summary["snapshot_manifest_sha256"],
                    collector_snapshot_file=str(payload.resolve()), collector_manifest_file=str(manifest_path.resolve()),
                    collector_control_url=run.url)
    with locked(artifacts / "submission.lock"):
        current = json.loads((artifacts / "campaign.json").read_text())
        selected = select_auxiliary(current, note=note)
        require(selected["job_id"] == auxiliary["job_id"] and current.get("runtime_bundle_sha256")
                == campaign["runtime_bundle_sha256"], "Control ledger changed during collection")
        ready = validate_snapshot(snapshot, "inspect")
        save(artifacts / "inspection-latest.json", snapshot)
        for row in current["auxiliary_jobs"]:
            if row["job_id"] == auxiliary["job_id"]:
                row.update(snapshot_collected_at=now(), snapshot_timestamp=snapshot["timestamp"],
                           snapshot_transport_source="wandb_aux_snapshot")
        current.update(assets_ready_verified=ready, latest_inspection="inspection-latest.json",
                       storage_latest=snapshot["filesystems"], latest_snapshot=snapshot["timestamp"])
        save(artifacts / "campaign.json", current)
    return snapshot


def verify_control_files(snapshot: dict, artifacts: Path, campaign: dict) -> None:
    """Rehash the exact locally collected W&B evidence before final authority."""
    require(snapshot.get("collector_transport_source") == "wandb_aux_snapshot"
            and snapshot.get("collector_wandb_file_verified") is True
            and snapshot.get("collector_wandb_snapshot_verified") is True,
            "Missing independently verified W&B control file proof")
    auxiliary = select_auxiliary(campaign, note=snapshot.get("collector_control_note"))
    require(auxiliary["job_id"] == snapshot.get("collector_source_job_id"), "Control snapshot owned note/job differs")
    directory = artifacts / "control-snapshots" / auxiliary["note"]
    require(not directory.is_symlink() and directory.resolve().is_relative_to(artifacts.resolve()),
            "Control evidence directory escapes repair namespace")
    payload, manifest_path = directory / "snapshot.json.gz", directory / "snapshot-manifest.json"
    require(not payload.is_symlink() and not manifest_path.is_symlink()
            and str(payload.resolve()) == snapshot.get("collector_snapshot_file")
            and str(manifest_path.resolve()) == snapshot.get("collector_manifest_file"), "Control evidence file identity differs")
    require(0 < payload.stat().st_size <= MAX_BYTES and 0 < manifest_path.stat().st_size <= MAX_BYTES,
            "Control proof file is oversized")
    compressed, manifest_bytes = payload.read_bytes(), manifest_path.read_bytes()
    manifest = json.loads(manifest_bytes)
    require(hashlib.sha256(manifest_bytes).hexdigest() == snapshot.get("collector_manifest_sha256")
            and hashlib.sha256(compressed).hexdigest() == snapshot.get("collector_snapshot_sha256")
            == manifest.get("sha256") and len(compressed) == manifest.get("bytes"), "Control local proof SHA/size differs")
    require(all(manifest.get(key) == value for key, value in {"campaign": CAMPAIGN, "note": auxiliary["note"],
                "mode": "inspect", "runtime_bundle_sha256": campaign["runtime_bundle_sha256"],
                "file": "snapshot.json.gz", "snapshot_timestamp": snapshot["timestamp"]}.items()),
            "Control local manifest identity differs")
    with gzip.GzipFile(fileobj=io.BytesIO(compressed)) as stream:
        data = stream.read(MAX_BYTES + 1)
        require(len(data) <= MAX_BYTES and stream.read(1) == b"", "Control proof decoded bound exceeded")
    require(len(data) == manifest.get("uncompressed_bytes") and hashlib.sha256(data).hexdigest()
            == manifest.get("uncompressed_sha256"), "Control decoded local proof differs")
    original = json.loads(data)
    require(all(snapshot.get(key) == value for key, value in original.items()), "Inspection differs from collected W&B payload")
    validate_snapshot(original, "inspect")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["collect", "upload-child"])
    parser.add_argument("--snapshot", type=Path)
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--note")
    selection.add_argument("--job-id")
    args = parser.parse_args()
    if args.mode == "upload-child":
        require(args.snapshot is not None, "Missing owned snapshot source")
        upload_child(args.snapshot)
        return
    with locked(ARTIFACTS / "submission.lock"):
        campaign = json.loads((ARTIFACTS / "campaign.json").read_text())
        auxiliary = select_auxiliary(campaign, args.note, args.job_id)
    credentials = netrc.netrc().authenticators("wandb-radfan.ru")
    require(credentials is not None, "W&B credentials unavailable")
    os.environ["WANDB_BASE_URL"] = "https://wandb-radfan.ru"
    os.environ["WANDB_API_KEY"] = credentials[2]
    import wandb
    with locked(ARTIFACTS / "collection.lock"):
        snapshot = collect_control(wandb.Api(timeout=45), auxiliary, campaign, ARTIFACTS, credentials[2])
    print(json.dumps({"campaign": CAMPAIGN, "snapshot_timestamp": snapshot["timestamp"],
                      "source": "wandb_aux_snapshot", "assets_ready_verified": snapshot["collector_assets_ready_verified"]}, indent=2))


if __name__ == "__main__":
    main()
