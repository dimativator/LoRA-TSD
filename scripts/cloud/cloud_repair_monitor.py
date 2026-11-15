"""Local-only independent W&B verification for the eight corrected reruns."""

import argparse
import base64
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import math
import netrc
import os
from pathlib import Path, PurePosixPath
import re
import tempfile
from urllib.parse import urlsplit
from urllib.request import Request, build_opener
import zipfile

from cloud14_monitor import SameOriginDownloadRedirect, download_run_file
from cloud_repair_common import (BUNDLE, BUNDLE_SHA, CAMPAIGN, PYTHON, REVISION,
                                 STATE, TASKS, digest, locked, now, run_name,
                                 save, task_key, verify_rows)


ARTIFACTS = Path(__file__).resolve().parents[4] / "artifacts/paper_8b_pro_rite_repair_20261115"
EXPECTED = {run_name(task): task for task in TASKS}
SHA256 = re.compile(r"^[a-f0-9]{64}$")


class VerificationError(ValueError):
    """A safe-to-report verification failure with no transport credentials."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise VerificationError(message)


def expected_command(task: dict, microbatch: int, bundle_sha: str) -> tuple[list[str], float]:
    require(type(microbatch) is int and microbatch in (1, 2, 4, 8, 16, 32), "Invalid microbatch")
    require(bool(SHA256.fullmatch(bundle_sha)), "Missing or invalid repair bundle checksum")
    require(digest(BUNDLE) == bundle_sha and BUNDLE_SHA.read_text().strip() == bundle_sha,
            "Local frozen repair bundle differs from campaign")
    with tempfile.TemporaryDirectory(prefix="repair-protocol-") as temporary:
        protocol = Path(temporary) / "scripts/brain_lab"
        protocol.mkdir(parents=True)
        with zipfile.ZipFile(BUNDLE) as archive:
            for name in ("paper_8b_campaign.py", "paper_8b_methods.json"):
                (protocol / name).write_bytes(archive.read("scripts/brain_lab/" + name))
        specification = json.loads((protocol / "paper_8b_methods.json").read_text())
        method = next(row for row in specification["methods"] if row["id"] == task["method"])
        require(method["optimizer"] == task["optimizer"], "Frozen optimizer identity differs")
        spec = importlib.util.spec_from_file_location("verified_repair_protocol", protocol / "paper_8b_campaign.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        directory = STATE / "pending-wandb-sync" / task_key(task)
        arguments = module.arguments({"method": method, "dataset": task["dataset"],
                                      "seed": task["seed"], "run_name": run_name(task)},
                                     microbatch, directory)
    main = f"/tmp/paper8b-repair-{bundle_sha[:16]}/src/main.py"
    return [str(PYTHON), "-u", main, *arguments], float(method["lr"])


def identity(run, task: dict, campaign: dict, status: dict | None = None) -> dict:
    config = dict(run.config)
    require(run.name == run_name(task), "Unexpected run name")
    require(getattr(run, "group", None) == CAMPAIGN, "Unexpected W&B group")
    path = "/".join(run.path) if isinstance(run.path, (list, tuple)) else str(run.path)
    require(path == "andrey/LoRA-Muon/" + run.id, "Unexpected W&B project/run identity")
    require(config.get("campaign") == CAMPAIGN and not config.get("reused"), "Wrong or reused campaign run")
    for key in ("method", "optimizer", "dataset", "seed"):
        require(config.get(key) == task[key], "W&B task identity differs")
    require(config.get("worker") == TASKS.index(task), "Wrong task worker")
    require(config.get("model_revision") == REVISION, "Wrong model revision")
    require(config.get("runtime_bundle_sha256") == campaign["runtime_bundle_sha256"], "Wrong runtime bundle")
    require((config.get("steps"), config.get("eval_every"), config.get("effective_batch")) == (3000, 250, 32),
            "Training protocol differs")
    if status is not None:
        require(status.get("task") == task, "Shared task record differs")
        require(not status.get("wandb_id") or status["wandb_id"] == run.id, "Shared task has a different run ID")
        if status.get("command") is not None:
            require(status["command"] == config.get("command"), "Shared and W&B commands differ")
    microbatch = config.get("microbatch")
    if microbatch is not None:
        command, learning_rate = expected_command(task, microbatch, campaign["runtime_bundle_sha256"])
        require(config.get("command") == command, "Actual command differs from exact frozen protocol")
    else:
        learning_rate = None
    return {"name": run.name, "id": run.id, "path": path, "url": run.url,
            **task, "worker": config["worker"], "microbatch": microbatch,
            "expected_learning_rate": learning_rate, "state": run.state}


def fetch(run, name: str, directory: Path, api_key: str) -> Path:
    require(PurePosixPath(name).name == name and name not in (".", ".."), "Invalid flat download name")
    file = run.file(name)
    require(file.name == name and type(file.size) is int and file.size >= 0, "Wrong W&B file metadata")
    destination = directory / name
    require(not destination.is_symlink(), "Local verified file is a link")
    if file.size:
        download_run_file(file.url, destination, file.size, run.id, api_key, file.md5 or "")
    else:
        parsed = urlsplit(file.url)
        require(parsed.scheme == "https" and parsed.netloc == "wandb-radfan.ru" and not parsed.fragment
                and parsed.path == f"/files/andrey/LoRA-Muon/{run.id}/{name}", "Unexpected empty-file URL")
        authorization = base64.b64encode(("api:" + api_key).encode()).decode()
        request = Request(file.url, headers={"Authorization": "Basic " + authorization,
                                            "Accept-Encoding": "identity"})
        with build_opener(SameOriginDownloadRedirect()).open(request, timeout=45) as response:
            require(response.read(1) == b"", "Nonempty content for a declared empty file")
        empty_md5 = hashlib.md5(b"").digest()
        require(not file.md5 or file.md5 in (empty_md5.hex(), base64.b64encode(empty_md5).decode()),
                "Empty-file checksum differs")
        with tempfile.NamedTemporaryFile(dir=directory, prefix=".download-", delete=False) as output:
            temporary = Path(output.name)
        temporary.replace(destination)
    return destination


def validate_manifest(manifest: dict, entry: dict, campaign: dict, command: list[str]) -> list[dict]:
    require((manifest.get("campaign"), manifest.get("run_name"), manifest.get("wandb_id"))
            == (CAMPAIGN, entry["name"], entry["id"]), "Diagnostic manifest identity differs")
    require(manifest.get("runtime_bundle_sha256") == campaign["runtime_bundle_sha256"]
            and manifest.get("model_revision") == REVISION and manifest.get("command") == command,
            "Diagnostic manifest runtime/command differs")
    rows = manifest.get("files")
    require(isinstance(rows, list) and 0 < len(rows) <= 10000, "Invalid diagnostic file inventory")
    paths, names = set(), set()
    for row in rows:
        path = PurePosixPath(row["path"])
        name = row["wandb_name"]
        require(not path.is_absolute() and ".." not in path.parts and "." not in path.parts
                and str(path) == row["path"] and path.suffix in (".json", ".jsonl", ".log"),
                "Unsafe diagnostic source path")
        require(PurePosixPath(name).name == name and name != "diagnostic-manifest.json", "Unsafe flat diagnostic name")
        expected_name = path.name
        if len(path.parts) > 1:
            match = re.fullmatch(f"cloud{entry['worker']}-{entry['method']}-micro(1|2|4|8|16|32)", path.parts[0])
            require(match is not None, "Unexpected profile diagnostic owner")
            expected_name = "profile-micro" + match.group(1) + "-" + path.name
        require(name == expected_name and name not in names and str(path) not in paths,
                "Duplicate or incorrect diagnostic upload mapping")
        require(type(row["bytes"]) is int and row["bytes"] >= 0
                and isinstance(row["sha256"], str) and bool(SHA256.fullmatch(row["sha256"])),
                "Invalid diagnostic size/checksum")
        names.add(name)
        paths.add(str(path))
    require({"history.jsonl", entry["name"] + ".json"}.issubset(names), "Canonical evidence missing from manifest")
    return rows


def verify_no_checkpoints(run) -> None:
    for file in run.files():
        require(PurePosixPath(file.name).suffix not in (".pt", ".pth", ".ckpt", ".safetensors", ".bin"),
                "Unexpected checkpoint/model file in W&B")


def complete_shared_record(status: dict) -> None:
    require(status.get("phase") == "complete" and status.get("returncode") == 0,
            "Shared record is not successfully complete")
    require(isinstance(status.get("recovery_manifest_sha256"), str)
            and bool(SHA256.fullmatch(status["recovery_manifest_sha256"])),
            "Completed shared record has no valid diagnostic manifest checksum")


def verify_finished(run, task: dict, campaign: dict, artifacts: Path, api_key: str,
                    status: dict | None = None) -> dict:
    entry = identity(run, task, campaign, status)
    summary = dict(run.summary)
    require(run.state == "finished" and summary.get("verified") is True, "Run is not finished and verified")
    require(summary.get("returncode") == 0 and summary.get("checkpoint_count") == 0, "Run exit/checkpoint proof differs")
    require(entry["microbatch"] is not None, "Finished run has no verified command")
    if status is not None:
        complete_shared_record(status)
    verify_no_checkpoints(run)
    directory = artifacts / "raw-wandb" / entry["name"]
    require(not directory.is_symlink() and directory.resolve().is_relative_to(artifacts.resolve()),
            "Local evidence directory escapes repair artifacts")
    directory.mkdir(parents=True, exist_ok=True)
    history = fetch(run, "history.jsonl", directory, api_key)
    result_path = fetch(run, entry["name"] + ".json", directory, api_key)
    manifest_path = fetch(run, "diagnostic-manifest.json", directory, api_key)
    rows = [json.loads(line) for line in history.read_text().splitlines()]
    result = json.loads(result_path.read_text())
    require(result.get("run_name") == entry["name"], "Result run name differs")
    try:
        checks = verify_rows(rows, result, task)
    except (AssertionError, KeyError, TypeError, ValueError):
        raise VerificationError("Canonical training/evaluation verification failed") from None
    for row in rows:
        if "train/loss" in row:
            require(math.isclose(row.get("lr_A_used", float("nan")), entry["expected_learning_rate"],
                                 rel_tol=1e-8, abs_tol=1e-12), "Actual learning rate differs")
            require(all(math.isfinite(value) for key, value in row.items()
                        if key.startswith("optimizer/") and isinstance(value, (int, float))),
                    "Nonfinite optimizer diagnostic")
        if "validation_score" in row:
            require(0 <= row["validation_score"] <= 1, "Validation accuracy is outside its range")
    manifest = json.loads(manifest_path.read_text())
    manifest_rows = validate_manifest(manifest, entry, campaign, list(run.config["command"]))
    manifest_sha = digest(manifest_path)
    if status is not None:
        require(status["recovery_manifest_sha256"] == manifest_sha, "Shared diagnostic manifest checksum differs")
    verified_files = []
    for row in manifest_rows:
        path = directory / row["wandb_name"]
        if row["wandb_name"] not in ("history.jsonl", entry["name"] + ".json"):
            path = fetch(run, row["wandb_name"], directory, api_key)
        require(path.stat().st_size == row["bytes"] and digest(path) == row["sha256"],
                "Diagnostic file checksum/size differs")
        verified_files.append({**row, "local_file": str(path.resolve())})
    verified_files.append({"path": "diagnostic-manifest.json", "wandb_name": "diagnostic-manifest.json",
                           "bytes": manifest_path.stat().st_size, "sha256": manifest_sha,
                           "local_file": str(manifest_path.resolve())})
    return {**entry, **checks, "verified_at": now(), "history_file": str(history.resolve()),
            "result_file": str(result_path.resolve()), "diagnostic_manifest_sha256": manifest_sha,
            "all_diagnostics_independently_verified": True, "verified_files": verified_files,
            "runtime_bundle_sha256": campaign["runtime_bundle_sha256"]}


def cache_valid(entry: dict, directory: Path) -> bool:
    try:
        files = entry["verified_files"]
        if not files or entry.get("all_diagnostics_independently_verified") is not True:
            return False
        expected_names = {"history.jsonl", entry["name"] + ".json", "diagnostic-manifest.json"}
        names = [row["wandb_name"] for row in files]
        if len(names) != len(set(names)) or not expected_names.issubset(names):
            return False
        for row in files:
            path = Path(row["local_file"])
            if path.is_symlink() or path.resolve().parent != directory.resolve():
                return False
            if path.name != row["wandb_name"] or path.stat().st_size != row["bytes"] or digest(path) != row["sha256"]:
                return False
        manifest_path = directory / "diagnostic-manifest.json"
        if digest(manifest_path) != entry["diagnostic_manifest_sha256"]:
            return False
        manifest = json.loads(manifest_path.read_text())
        if {row["wandb_name"] for row in manifest["files"]} != set(names) - {"diagnostic-manifest.json"}:
            return False
        by_name = {row["wandb_name"]: row for row in files}
        if any((by_name[row["wandb_name"]]["sha256"], by_name[row["wandb_name"]]["bytes"])
               != (row["sha256"], row["bytes"]) for row in manifest["files"]):
            return False
        return True
    except (OSError, KeyError, TypeError, ValueError):
        return False


def inspection_records(artifacts: Path) -> dict:
    path = artifacts / "inspection-latest.json"
    if not path.exists():
        return {}
    data = json.loads(path.read_text())
    require(data.get("campaign") == CAMPAIGN, "Inspection belongs to another campaign")
    records = {}
    for record in data.get("tasks", []):
        task = record.get("task")
        require(task in TASKS, "Inspection contains an unexpected task")
        name = run_name(task)
        require(name not in records, "Duplicate shared task record")
        records[name] = record
    return records


def monitor(api, campaign: dict, artifacts: Path, api_key: str, verify: bool = False) -> dict:
    require(campaign.get("campaign") == CAMPAIGN and campaign.get("tasks") == TASKS
            and campaign.get("target_runs") == len(TASKS), "Local campaign scope differs")
    require(bool(SHA256.fullmatch(campaign.get("runtime_bundle_sha256", ""))), "Repair bundle is not frozen")
    cache_path = artifacts / "wandb-independent-verification.json"
    verified = json.loads(cache_path.read_text()) if cache_path.exists() else {}
    require(set(verified).issubset(EXPECTED), "Verification cache contains an unexpected task")
    statuses = inspection_records(artifacts)
    has_inspection = (artifacts / "inspection-latest.json").exists()
    interrupted_path = artifacts / "interrupted-attempts.json"
    interrupted = json.loads(interrupted_path.read_text()) if interrupted_path.exists() else {}
    interrupted_ids = set(interrupted)
    interrupted_ids.update(record["wandb_id"] for record in statuses.values()
                           if record.get("phase") == "failed" and record.get("wandb_id"))
    grouped, unexpected, historical = {}, [], []
    for run in api.runs("andrey/LoRA-Muon", filters={"group": CAMPAIGN}, per_page=100):
        if run.name not in EXPECTED:
            unexpected.append({"name": run.name, "id": run.id, "state": run.state})
        elif run.id in interrupted_ids:
            historical.append({"name": run.name, "id": run.id, "state": run.state})
            if verified.get(run.name, {}).get("id") == run.id:
                verified.pop(run.name, None)
        else:
            grouped.setdefault(run.name, []).append(run)
    active, failed, pending, errors, queued = [], [], [], [], []
    for name, task in EXPECTED.items():
        runs = grouped.get(name, [])
        status = statuses.get(name)
        if len(runs) > 1:
            verified.pop(name, None)
            errors.append({"name": name, "error": "Duplicate non-interrupted W&B runs"})
            continue
        if not runs:
            verified.pop(name, None)
            if status and status.get("phase") == "failed":
                failed.append({"name": name, "phase": "failed", "error_type": status.get("error_type")})
            elif status and status.get("phase") not in ("reserved", None):
                record = {"name": name, **task, "phase": status["phase"],
                          "step": status.get("step", 0), "source": "shared_state"}
                if status["phase"] == "complete":
                    pending.append(record)
                else:
                    active.append(record)
            else:
                queued.append(name)
            continue
        run = runs[0]
        try:
            entry = identity(run, task, campaign, status)
            summary = dict(run.summary)
            if run.state == "finished" and summary.get("verified") is True:
                if has_inspection and (status is None or status.get("phase") != "complete"):
                    verified.pop(name, None)
                    pending.append({**entry, "reason": "Shared completion inspection pending"})
                    continue
                if status is not None:
                    complete_shared_record(status)
                require(summary.get("returncode") == 0 and summary.get("checkpoint_count") == 0,
                        "Finished run exit/checkpoint proof differs")
                require(entry["microbatch"] is not None, "Finished run has no verified command")
                verify_no_checkpoints(run)
                cached = verified.get(name)
                directory = artifacts / "raw-wandb" / name
                require(not directory.is_symlink() and directory.resolve().is_relative_to(artifacts.resolve()),
                        "Local evidence directory escapes repair artifacts")
                keys = ("name", "id", "path", "method", "optimizer", "dataset", "seed", "worker", "microbatch")
                if cached and (any(cached.get(key) != entry[key] for key in keys)
                               or cached.get("runtime_bundle_sha256") != campaign["runtime_bundle_sha256"]):
                    raise VerificationError("Cached run identity differs")
                if cached and cache_valid(cached, directory):
                    validate_manifest(json.loads((directory / "diagnostic-manifest.json").read_text()),
                                      entry, campaign, list(run.config["command"]))
                    if status is not None:
                        require(cached["diagnostic_manifest_sha256"] == status["recovery_manifest_sha256"],
                                "Cached manifest differs from shared task")
                    continue
                verified.pop(name, None)
                if verify:
                    verified[name] = verify_finished(run, task, campaign, artifacts, api_key, status)
                else:
                    pending.append(entry)
            elif run.state in ("failed", "crashed", "killed", "finished"):
                verified.pop(name, None)
                failed.append(entry)
            else:
                verified.pop(name, None)
                evaluations = list(run.scan_history(keys=["optimizer_step", "validation_score"]))
                evaluations = [row for row in evaluations if "validation_score" in row]
                require(all(math.isfinite(row["validation_score"]) for row in evaluations), "Nonfinite active validation")
                step = summary.get("optimizer_step", 0)
                phase = status.get("phase") if status else None
                if phase in (None, "claimed"):
                    phase = "profiling" if entry["microbatch"] is None else ("initializing" if step == 0 else "training")
                    if (step > 0 and step % 250 == 249) or (step == 3000 and len(evaluations) < 12):
                        phase = "validation"
                entry.update(step=step, phase=phase, evaluation_count=len(evaluations), created_at=run.created_at)
                if evaluations:
                    best = max(evaluations, key=lambda row: row["validation_score"])
                    entry.update(best_accuracy=best["validation_score"], best_step=best["optimizer_step"])
                active.append(entry)
        except Exception as error:
            verified.pop(name, None)
            errors.append({"name": name, "error_type": type(error).__name__,
                           "error": str(error) if isinstance(error, VerificationError) else "Verification/transport failed"})
    save(cache_path, verified)
    phases = {phase: sum(entry.get("phase") == phase for entry in active) for phase in (
        "training", "validation", "profiling", "initializing", "syncing", "waiting", "resource_waiting")}
    complete = len(verified) == len(TASKS) and not (active or failed or pending or errors or unexpected)
    snapshot = {"timestamp": now(), "campaign": CAMPAIGN, "target": len(TASKS),
                "verified_completed": len(verified), "active": active, "failed": failed,
                "pending_independent_verification": pending, "verification_errors": errors,
                "unexpected_runs": unexpected, "interrupted_attempts": historical,
                "queued": queued, "queued_count": len(queued), "phase_counts": phases,
                "all_histories_verified": complete,
                "all_diagnostics_independently_verified": complete
                    and all(row.get("all_diagnostics_independently_verified") for row in verified.values())}
    filename = "monitor-check-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + ".json"
    save(artifacts / filename, snapshot)
    save(artifacts / "latest-monitor-check.json", snapshot)
    return snapshot


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--verify-finished", action="store_true")
    args = parser.parse_args()
    import wandb
    credentials = netrc.netrc().authenticators("wandb-radfan.ru")
    require(credentials is not None, "W&B credentials unavailable")
    os.environ["WANDB_BASE_URL"] = "https://wandb-radfan.ru"
    os.environ["WANDB_API_KEY"] = credentials[2]
    api = wandb.Api(timeout=45)
    with locked(ARTIFACTS / "submission.lock"):
        campaign = json.loads((ARTIFACTS / "campaign.json").read_text())
        print(json.dumps(monitor(api, campaign, ARTIFACTS, credentials[2], args.verify_finished), indent=2))


if __name__ == "__main__":
    main()
