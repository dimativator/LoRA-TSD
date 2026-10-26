"""Exact-byte storage guard shared by the supervisor and training process."""

import json
import os
from pathlib import Path
import time


class DiskGuard:
    def __init__(self, campaign_root: Path, run_name: str, telemetry: Path | None = None):
        self.root = campaign_root
        self.run_name = run_name
        self.telemetry = telemetry
        self.reserve = 5_000_000_000
        self.paths = [Path("/"), Path("/tmp"), Path("/home/jovyan"),
                      Path("/home/jovyan/shares/SR006.nfs3")]
        self.last_logged = 0.0

    def __call__(self, pending_write_bytes: int = 0, reason: str = "periodic",
                 write_path: Path | None = None) -> dict:
        abort = self.root / "STOP_STORAGE.json"
        if abort.exists():
            raise RuntimeError("Campaign storage stop signal is present: " + abort.read_text())
        records = []
        failed = []
        destination = Path(write_path) if write_path is not None else self.root
        while not destination.exists() and destination != destination.parent:
            destination = destination.parent
        write_device = os.stat(destination).st_dev
        for path in self.paths:
            try:
                stat = os.statvfs(path)
                available = stat.f_bavail * stat.f_frsize
                device = os.stat(path).st_dev
                planned = pending_write_bytes if device == write_device else 0
                record = {"path": str(path), "device": device,
                          "available_bytes": available,
                          "planned_write_bytes": planned,
                          "status": "ok" if available > self.reserve + planned else "stop"}
                if record["status"] != "ok":
                    failed.append(record)
            except OSError as error:
                record = {"path": str(path), "error": str(error), "status": "telemetry_unavailable"}
                failed.append(record)
            records.append(record)
        result = {"event": "disk_guard", "timestamp": time.time(), "run": self.run_name,
                  "pid": os.getpid(), "reason": reason, "pending_write_bytes": pending_write_bytes,
                  "write_path": str(write_path or self.root),
                  "reserve_bytes": self.reserve, "paths": records,
                  "status": "stop" if failed else "ok", "nfs2_accessed": False}
        if failed:
            print("CRITICAL CLOUD.RU STORAGE: " + json.dumps(result), flush=True)
            if self.root.is_dir():
                try:
                    with abort.open("x") as stream:
                        json.dump(result, stream)
                except FileExistsError:
                    pass
                except OSError as error:
                    print("STORAGE STOP SIGNAL WRITE FAILED: " + str(error), flush=True)
            raise RuntimeError("Storage reserve violated or telemetry unavailable")
        if pending_write_bytes or time.monotonic() - self.last_logged >= 45:
            print("GEN_DISK " + json.dumps(result), flush=True)
            if self.telemetry is not None:
                with self.telemetry.open("a") as stream:
                    stream.write(json.dumps(result) + "\n")
            self.last_logged = time.monotonic()
        return result
