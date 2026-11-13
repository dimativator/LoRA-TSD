"""Prepare one owned copy of the pinned assets and one shared Python environment."""

import json
import os
from pathlib import Path
import subprocess
import sys
import time

from cloud14_core import ASSETS, CAMPAIGN, STATE, REVISION, digest, locked, now, require_space, save

HERE = Path(__file__).resolve().parent
VENV = ASSETS.parent / "venv"
PYTHON = VENV / "bin/python"
REQUIREMENTS = ["accelerate==1.10.1", "datasets==2.19.1", "numpy==2.0.2",
                "peft==0.17.1", "safetensors==0.7.0", "tokenizers==0.22.2",
                "tqdm==4.67.1", "transformers==4.57.3", "wandb==0.26.1"]


def affected_paths() -> list[Path]:
    return [Path("/"), Path("/tmp"), Path("/home/jovyan"), ASSETS, VENV,
            Path("/home/jovyan/shares/SR006.nfs3/mlsub-logs"), STATE]


def download(url: str, path: Path, size: int, sha256: str) -> None:
    import requests
    if path.exists():
        if path.stat().st_size == size and digest(path) == sha256:
            return
        raise ValueError(f"Existing asset has unexpected content: {path}")
    require_space(affected_paths(), {path: size})
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".part")
    if temporary.exists():
        # Only a verified campaign-owned incomplete download is replaced.
        assert not temporary.is_symlink() and temporary.resolve().is_relative_to(ASSETS.resolve())
        temporary.unlink()
    headers = {"Authorization": "Bearer " + os.environ["HF_TOKEN"]}
    last_report = time.monotonic()
    with requests.get(url, headers=headers, stream=True, timeout=(25, 180)) as response:
        response.raise_for_status()
        with temporary.open("wb") as stream:
            for chunk in response.iter_content(chunk_size=8 * 1024 * 1024):
                require_space(affected_paths(), {temporary: len(chunk)})
                stream.write(chunk)
                if stream.tell() > size:
                    raise ValueError(f"Download exceeds pinned size: {path.name}")
                if time.monotonic() - last_report >= 45:
                    print("ASSET_PROGRESS=" + json.dumps({"file": path.name, "bytes": stream.tell(),
                                                          "expected_bytes": size}), flush=True)
                    last_report = time.monotonic()
    assert temporary.stat().st_size == size and digest(temporary) == sha256, path.name
    temporary.replace(path)


def prepare() -> None:
    model = json.loads((HERE / "cloud14_model_files.json").read_text())
    data = json.loads((HERE / "cloud14_data_files.json").read_text())
    remaining = sum(entry["bytes"] for name, entry in model.items()
                    if not (ASSETS / "model" / name).exists())
    remaining += sum(entry["bytes"] for name, entry in data.items()
                     if not (ASSETS / "data" / name).exists())
    require_space(affected_paths(), {ASSETS: remaining,
                                    VENV: 0 if (VENV / "ready.json").exists() else 2_000_000_000})
    with locked(STATE / "prepare.lock"):
        if not PYTHON.exists():
            subprocess.run([sys.executable, "-m", "venv", "--system-site-packages", str(VENV)], check=True)
        if not (VENV / "ready.json").exists():
            subprocess.run([str(PYTHON), "-m", "pip", "install", "--no-cache-dir", *REQUIREMENTS], check=True)
            subprocess.run([str(PYTHON), "-c", "import torch,transformers,peft,datasets,wandb,pyarrow; "
                            "assert torch.__version__.startswith('2.8.0'); print('RUNTIME_IMPORTS_PASS')"], check=True)
            save(VENV / "ready.json", {"timestamp": now(), "requirements": REQUIREMENTS})
        if Path(sys.prefix) != VENV:
            os.execv(str(PYTHON), [str(PYTHON), str(HERE / "cloud14_entry.py"), "prepare"])
        # W&B must be usable before spending time or space downloading the model.
        subprocess.run([str(PYTHON), "-c", "import wandb; api=wandb.Api(timeout=45); "
                        "assert api.viewer; print('WANDB_AUTHENTICATED')"], check=True)
        ASSETS.mkdir(parents=True, exist_ok=True)
        ownership = ASSETS / "ownership.json"
        expected = {"campaign": CAMPAIGN, "root": str(ASSETS.resolve())}
        if ownership.exists():
            assert json.loads(ownership.read_text()) == expected
        else:
            assert not list(ASSETS.iterdir()), "Refusing to adopt an existing unowned asset directory"
            save(ownership, expected)
        for name, entry in model.items():
            url = f"https://huggingface.co/meta-llama/Meta-Llama-3.1-8B/resolve/{REVISION}/{name}"
            download(url, ASSETS / "model" / name, entry["bytes"], entry["sha256"])
        for name, entry in data.items():
            download(entry["url"], ASSETS / "data" / name, entry["bytes"], entry["sha256"])
        save(ASSETS / "ready.json", {"campaign": CAMPAIGN, "revision": REVISION, "timestamp": now(),
                                     "model_files": model, "dataset_files": data,
                                     "filesystems": require_space(affected_paths())})
        print("ASSETS_READY=" + json.dumps({"root": str(ASSETS), "timestamp": now()}), flush=True)


def configured_environment() -> dict:
    assert (ASSETS / "ready.json").exists(), "Assets not prepared"
    return {**os.environ, "MODEL_LOCAL_PATH": str(ASSETS / "model"), "MODEL_REVISION": REVISION,
            "LORA_BENCHMARK_DATA_DIR": str(ASSETS / "data"), "HF_HUB_OFFLINE": "1",
            "HF_DATASETS_OFFLINE": "1", "PYTHONDONTWRITEBYTECODE": "1",
            "TOKENIZERS_PARALLELISM": "false", "WANDB_MODE": "online",
            "WANDB_BASE_URL": "https://wandb-radfan.ru", "WANDB_ENTITY": "andrey",
            "WANDB_PROJECT": "LoRA-Muon", "WANDB_LOG_MODEL": "false"}
