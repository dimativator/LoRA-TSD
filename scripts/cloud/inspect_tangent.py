"""Read only this campaign's metrics and worker storage telemetry."""

import base64
import gzip
import hashlib
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts/generation'))
from disk_guard import DiskGuard


def last_record(path: Path) -> dict | None:
    if not path.exists():
        return None
    # An incomplete final line can belong to an active writer.
    with path.open('rb') as stream:
        stream.seek(max(0, path.stat().st_size - 32768))
        lines = stream.read().splitlines()
    for line in reversed(lines):
        try:
            return json.loads(line)
        except (json.JSONDecodeError, UnicodeDecodeError):
            continue
    return None


def main() -> None:
    root = Path(os.environ['ABLATION_ROOT'])
    disk = DiskGuard(root, 'inspector')()
    runs = []
    for directory in sorted(root.iterdir()):
        if not directory.is_dir() or directory.is_symlink() or directory.name == 'claims':
            continue
        process = directory / 'process.json'
        if not process.exists() and not (directory / 'config.json').exists():
            continue
        entry = {'run_name': directory.name}
        for key, filename in [('process', 'process.json'), ('result', 'result.json'),
                              ('config', 'config.json'), ('gpu_before_train', 'gpu-before-train.json')]:
            path = directory / filename
            if path.exists():
                entry[key] = json.loads(path.read_text())
        entry['last_metric'] = last_record(directory / 'history.jsonl')
        entry['evaluations'] = []
        history = directory / 'history.jsonl'
        if history.exists():
            for line in history.read_text().splitlines():
                try:
                    row = json.loads(line)
                    if row.get('event') == 'validation':
                        entry['evaluations'].append(row)
                except json.JSONDecodeError:
                    pass
        entry['last_disk'] = last_record(directory / 'disk-trainer.jsonl')
        log = directory / 'train.log'
        if log.exists():
            with log.open('rb') as stream:
                stream.seek(max(0, log.stat().st_size - 6000))
                entry['log_tail'] = stream.read().decode(errors='replace')
        spectrum = directory / 'spectra.jsonl'
        entry['spectra_bytes'] = spectrum.stat().st_size if spectrum.exists() else 0
        entry['spectra'] = []
        if spectrum.exists():
            for line in spectrum.read_text().splitlines():
                try:
                    entry['spectra'].append(json.loads(line))
                except json.JSONDecodeError:
                    pass
        runs.append(entry)
    guards = [last_record(path) for path in root.glob('bootstrap-*.jsonl')]
    result = {'timestamp': time.time(), 'disk': disk, 'bootstrap_guards': guards, 'runs': runs,
              'selections': [json.loads(p.read_text()) for p in sorted(root.glob('selected-*.json'))],
              'summary': json.loads((root / 'campaign-results.json').read_text()) if (root / 'campaign-results.json').exists() else None,
              'failures': [json.loads(p.read_text()) for p in root.glob('STOP*.json')]}
    payload = json.dumps(result).encode()
    compressed = base64.b64encode(gzip.compress(payload)).decode()
    parts = [compressed[i:i + 6000] for i in range(0, len(compressed), 6000)]
    for index, part in enumerate(parts):
        print('ABLATION_CHUNK ' + json.dumps({'index': index, 'total': len(parts), 'sha256': hashlib.sha256(payload).hexdigest(), 'data': part}), flush=True)
    print('ABLATION_INSPECT ' + json.dumps({'timestamp': result['timestamp'], 'disk': disk,
          'runs': [{'name': r['run_name'], 'last_metric': r['last_metric'], 'spectra_bytes': r['spectra_bytes']} for r in runs],
          'failures': result['failures']}), flush=True)


if __name__ == '__main__':
    main()
