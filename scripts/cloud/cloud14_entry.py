"""Cloud entry point. Preflight is CPU-only and never loads a model or modifies shared caches."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

from cloud14_core import ASSETS, CAMPAIGN, STATE, disk_records, now


def preflight() -> None:
    paths = [Path("/"), Path("/tmp"), Path("/home/jovyan"), ASSETS,
             Path("/home/jovyan/shares/SR006.nfs3/mlsub-logs"), STATE]
    snapshot = {"timestamp": now(), "campaign": CAMPAIGN, "filesystems": disk_records(paths),
                "python": sys.version, "nfs2_read_only": []}
    for path in [Path("/workspace-SR006.nfs2"), Path("/home/jovyan/shares/SR006.nfs2")]:
        try:
            snapshot["nfs2_read_only"].extend(disk_records([path]))
        except OSError as error:
            snapshot["nfs2_read_only"].append({"path": str(path), "error": str(error)})
    versions = subprocess.run([sys.executable, "-c",
        "import importlib.metadata,json; packages=['torch','transformers','peft','datasets','wandb','pyarrow']; "
        "print(json.dumps({p:importlib.metadata.version(p) for p in packages}))"],
        capture_output=True, text=True)
    snapshot["installed_packages"] = versions.stdout.strip()
    snapshot["package_probe_returncode"] = versions.returncode
    snapshot["assets_already_exist"] = ASSETS.exists()
    print("PAPER8B_PREFLIGHT=" + json.dumps(snapshot), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["preflight", "prepare", "pair", "worker", "profile", "inspect", "cleanup"])
    parser.add_argument("--pair-id", type=int)
    parser.add_argument("--worker-id", type=int)
    parser.add_argument("--task", type=Path)
    parser.add_argument("--microbatch", type=int)
    args = parser.parse_args()
    if args.mode == "preflight":
        preflight()
        return
    from cloud14_runtime import dispatch
    dispatch(args)


if __name__ == "__main__":
    main()
