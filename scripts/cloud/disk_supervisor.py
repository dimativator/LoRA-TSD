"""Fail-closed storage supervisor for this campaign's owned subprocess group."""
import argparse
import datetime
import json
import os
from pathlib import Path
import signal
import subprocess
import time

RESERVE = 5_000_000_000
REQUIRED_PATHS = ('/', '/tmp', '/home/jovyan/shares/SR006.nfs3')

def snapshot(paths: list[str], pending_bytes: int = 0, write_path: str = '/tmp') -> dict:
    rows = []
    try:
        write_device = os.stat(write_path).st_dev
    except OSError:
        write_device = None
    for path in paths:
        try:
            stat = os.statvfs(path)
            available = stat.f_bavail * stat.f_frsize
            device = os.stat(path).st_dev
            planned = pending_bytes if device == write_device else 0
            rows.append({'path': path, 'resolved_path': os.path.realpath(path), 'device': device,
                         'available_bytes': available, 'pending_bytes': planned,
                         'status': 'ok' if available > RESERVE + planned else 'stop'})
        except OSError as error:
            rows.append({'path': path, 'status': 'telemetry_unavailable', 'error': str(error)})
    return {'timestamp': datetime.datetime.now(datetime.timezone.utc).isoformat(),
            'reserve_bytes': RESERVE, 'paths': rows,
            'status': 'ok' if all(r['status'] == 'ok' for r in rows) else 'stop'}

def stop_owned(process: subprocess.Popen) -> None:
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait()
    except ProcessLookupError:
        pass

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--campaign-root', required=True)
    parser.add_argument('--planned-temp-write-bytes', type=int, default=3_000_000_000)
    parser.add_argument('command', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ['--'] else args.command
    if not command:
        parser.error('A subprocess command is required')
    root = Path(args.campaign_root)
    marker = root / 'STOP_STORAGE.json'
    paths = list(REQUIRED_PATHS)
    paths += [str(root.parent), os.environ.get('HF_HOME', '/tmp/lora-siqa-hf')]
    # Check the existing containing filesystem before creating output directories.
    for i, path in enumerate(paths):
        if path in REQUIRED_PATHS:
            continue
        candidate = Path(path)
        while not candidate.exists() and candidate != candidate.parent:
            candidate = candidate.parent
        paths[i] = str(candidate)
    first = snapshot(paths, args.planned_temp_write_bytes)
    print('DISK_GUARD ' + json.dumps(first), flush=True)
    if first['status'] != 'ok' or marker.exists():
        return 74
    root.mkdir(parents=True, exist_ok=True)
    telemetry = root / ('disk-guard-worker' + os.environ.get('LORA_BENCHMARK_WORKER', 'unknown') + '.jsonl')
    def interrupted(signum, frame):
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    process = subprocess.Popen(command, start_new_session=True)
    try:
        while process.poll() is None:
            report = snapshot(paths)
            if marker.exists():
                report['status'] = 'campaign_stop'
            if report['status'] != 'ok':
                print('CRITICAL CLOUD.RU STORAGE: ' + json.dumps(report), flush=True)
                stop_owned(process)
                try:
                    with marker.open('x') as out:
                        json.dump(report, out)
                except (OSError, FileExistsError):
                    pass
                return 74
            print('DISK_GUARD ' + json.dumps(report), flush=True)
            with telemetry.open('a') as stream:
                stream.write(json.dumps(report) + '\n')
            try:
                process.wait(timeout=45)
            except subprocess.TimeoutExpired:
                pass
        return process.returncode
    finally:
        stop_owned(process)

if __name__ == '__main__':
    raise SystemExit(main())
