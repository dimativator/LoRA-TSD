"""Read-only, CPU-only sizes and directory metadata for a cleanup assessment."""

import json
import os
from pathlib import Path
import stat
import subprocess
import sys

from cloud14_core import disk_records, now, snapshot_lines


HOME = Path("/home/jovyan")
STATE = HOME / "shares/SR006.nfs3/dimativator-paper8b-three-seeds-20261113"


def linked_path(root: Path) -> bool:
    boundary = STATE if root == STATE or STATE in root.parents else HOME
    path = root
    while path != boundary and path != path.parent:
        if path.is_symlink():
            return True
        path = path.parent
    return boundary != HOME and boundary.is_symlink()


def filesystems(paths: list[Path]) -> list[dict]:
    records = []
    for path in paths:
        try:
            records.extend(disk_records([path]))
        except (OSError, ValueError) as error:
            records.append({"requested_path": str(path), "status": "unavailable",
                            "timestamp": now(), "error": str(error)})
    return records


def bounded_probe(kind: str, root: Path, depth: int = 1) -> dict:
    command = [sys.executable, str(Path(__file__).resolve()), "--probe", kind,
               str(root), str(depth)]
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                               text=True)
    try:
        output, error = process.communicate(timeout=100 if kind == "sizes" else 15)
        if process.returncode:
            return {"path": str(root), "error": "Metadata subprocess failed",
                    "returncode": process.returncode, "stderr": error[-1500:]}
        return json.loads(output)
    except subprocess.TimeoutExpired:
        process.kill()
        # A kernel-blocked network filesystem operation may not reap immediately.
        return {"path": str(root), "error": "Metadata subprocess timed out"}
    except ValueError as error:
        return {"path": str(root), "error": type(error).__name__}


def entries(root: Path) -> dict:
    try:
        if linked_path(root):
            return {"path": str(root), "skipped": "symlink_or_linked_ancestor"}
        records = []
        with os.scandir(root) as iterator:
            for entry in iterator:
                info = entry.stat(follow_symlinks=False)
                kind = "symlink" if stat.S_ISLNK(info.st_mode) else (
                    "directory" if stat.S_ISDIR(info.st_mode) else "file")
                records.append({"name": entry.name, "kind": kind,
                                "uid": info.st_uid, "gid": info.st_gid,
                                "device": info.st_dev,
                                "bytes": info.st_size if kind == "file" else None})
                if len(records) > 1000:
                    raise ValueError("Top-level directory inventory exceeded limit")
        return {"path": str(root), "entries": sorted(records, key=lambda r: r["name"])}
    except (OSError, ValueError) as error:
        return {"path": str(root), "error": str(error)}


def sizes(root: Path, depth: int = 1) -> dict:
    if linked_path(root):
        return {"path": str(root), "skipped": "symlink_or_linked_ancestor"}
    if not root.exists():
        return {"path": str(root), "exists": False}
    command = ["du", "-x", "-B1", f"--max-depth={depth}", "--", str(root)]
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                               text=True)
    try:
        output, error = process.communicate(timeout=90)
    except subprocess.TimeoutExpired:
        process.kill()
        return {"path": str(root), "error": "du exceeded90seconds"}
    rows = []
    for line in output.splitlines():
        count, separator, path = line.partition("\t")
        if not separator or not count.isdigit():
            return {"path": str(root), "error": "Unexpected du output",
                    "returncode": process.returncode}
        rows.append({"path": path, "allocated_bytes": int(count)})
    rows.sort(key=lambda row: row["allocated_bytes"], reverse=True)
    return {"path": str(root), "returncode": process.returncode,
            "rows": rows[:500], "rows_omitted": max(0, len(rows) - 500),
            "errors": error[-2500:]}


def main() -> None:
    paths = [Path("/"), Path("/tmp"), HOME,
             HOME / "shares/SR006.nfs3", HOME / "shares/SR006.nfs2"]
    snapshot = {"timestamp": now(), "audit_type": "read_only_home_inventory",
                "filesystems_before": filesystems(paths),
                "home_entries": bounded_probe("entries", HOME),
                "cache_entries": bounded_probe("entries", HOME / ".cache"),
                "huggingface_entries": bounded_probe("entries", HOME / ".cache/huggingface"),
                "sizes": []}
    roots = [(HOME, 1), (HOME / ".cache", 2),
             (HOME / ".cache/huggingface/hub", 1),
             (HOME / "dimativator-paper8b-three-seeds-20261113", 2),
             (HOME / "lora-siqa-retractions-20261011", 1),
             (HOME / "lora-1b-benchmark-data-20261014", 1),
             (HOME / "lora-siqa-data", 1), (STATE, 2)]
    for root, depth in roots:
        snapshot["sizes"].append(bounded_probe("sizes", root, depth))
    snapshot["filesystems_after"] = filesystems(paths)
    snapshot["finished_at"] = now()
    for line in snapshot_lines(snapshot):
        print(line, flush=True)


if __name__ == "__main__":
    if len(sys.argv) == 5 and sys.argv[1] == "--probe":
        kind, root, depth = sys.argv[2], Path(sys.argv[3]), int(sys.argv[4])
        if kind not in {"entries", "sizes"}:
            raise ValueError("Unknown metadata probe")
        print(json.dumps(entries(root) if kind == "entries" else sizes(root, depth)))
    else:
        main()
