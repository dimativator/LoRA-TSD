"""Guard setup and all descendants, including blocked or long evaluations."""

import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts/generation'))
from disk_guard import DiskGuard


def main() -> None:
    root = Path(os.environ['ABLATION_ROOT'])
    guard = DiskGuard(root, 'bootstrap')
    guard(pending_write_bytes=1_000_000_000, reason='before_runtime_setup', write_path=Path('/tmp'))
    guard(pending_write_bytes=100_000_000, reason='before_logs', write_path=root)
    root.mkdir(parents=True, exist_ok=True)
    guard.telemetry = root / ('bootstrap-' + os.environ.get('HOSTNAME', str(os.getpid())) + '.jsonl')
    child = subprocess.Popen(['bash', str(ROOT / 'scripts/cloud/run_tangent_runtime.sh'), *sys.argv[1:]],
                             start_new_session=True)

    def stop() -> None:
        if child.poll() is not None:
            return
        os.killpg(child.pid, signal.SIGTERM)
        try:
            child.wait(timeout=10)
        except subprocess.TimeoutExpired:
            os.killpg(child.pid, signal.SIGKILL)
            child.wait(timeout=10)
        print('ABLATION_STOPPED ' + json.dumps({'process_group': child.pid, 'code': child.returncode}), flush=True)

    def stop_signal(signum, frame) -> None:
        stop()
        raise SystemExit(128 + signum)

    signal.signal(signal.SIGTERM, stop_signal)
    signal.signal(signal.SIGINT, stop_signal)
    probe = {'active': False, 'started': 0., 'error': None}

    def check() -> None:
        try:
            guard()
        except BaseException as error:
            probe['error'] = str(error)
        finally:
            probe['active'] = False

    next_probe = 0.
    try:
        while child.poll() is None:
            now = time.monotonic()
            if probe['error'] or (probe['active'] and now - probe['started'] > 15):
                stop()
                print('CRITICAL CLOUD.RU STORAGE: OWNED EXPERIMENT STOPPED: ' +
                      str(probe['error'] or 'TELEMETRY UNAVAILABLE'), flush=True)
                # No checkpoint artifacts are written by this campaign.
                raise RuntimeError('Storage watchdog stopped this worker')
            if not probe['active'] and now >= next_probe:
                probe.update(active=True, started=now)
                threading.Thread(target=check, daemon=True).start()
                next_probe = now + 5
            time.sleep(.5)
    finally:
        stop()
    print('ABLATION_ENTRY_EXIT=' + str(child.returncode), flush=True)
    raise SystemExit(child.returncode)


if __name__ == '__main__':
    main()
