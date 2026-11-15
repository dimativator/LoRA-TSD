"""Isolated paths and task scope for the corrected Pro/Rite reruns."""

from pathlib import Path

from cloud14_core import (digest, disk_records, locked, now, require_space, save,
                          snapshot_lines, task_key, verify_rows, wandb_row)


CAMPAIGN = "paper8b-pro-rite-repair-20261115"
HOME = Path("/home/jovyan/dimativator-paper8b-pro-rite-repair-20261115")
ASSETS = HOME / "assets"
STATE = Path("/home/jovyan/shares/SR006.nfs3/dimativator-paper8b-pro-rite-repair-20261115")
PYTHON = Path("/home/jovyan/dimativator-paper8b-three-seeds-20261113/venv/bin/python")
REVISION = "d04e592bb4f6aa9cfee91e2e20afa771667e1d4b"
HERE = Path(__file__).resolve().parent
BUNDLE = HERE / "cloud_repair_frozen_runtime.zip"
BUNDLE_SHA = HERE / "cloud_repair_frozen_runtime.sha256"
TASKS = [
    {"method": "lora_rite", "optimizer": "lora_rite", "dataset": "boolq", "seed": 1},
    {"method": "lora_rite", "optimizer": "lora_rite", "dataset": "piqa", "seed": 0},
    {"method": "lora_rite", "optimizer": "lora_rite", "dataset": "siqa", "seed": 42},
    {"method": "lora_pro", "optimizer": "lora_pro", "dataset": "piqa", "seed": 1},
    {"method": "lora_pro", "optimizer": "lora_pro", "dataset": "siqa", "seed": 42},
    {"method": "lora_pro", "optimizer": "lora_pro", "dataset": "obqa", "seed": 1},
    {"method": "lora_pro", "optimizer": "lora_pro", "dataset": "obqa", "seed": 42},
    {"method": "lora_pro", "optimizer": "lora_pro", "dataset": "qnli", "seed": 1},
]


def affected_paths() -> list[Path]:
    return [Path("/"), Path("/tmp"), HOME, ASSETS, STATE,
            Path("/home/jovyan/shares/SR006.nfs3/mlsub-logs")]


def run_name(task: dict) -> str:
    return "repair8b_" + task_key(task)
