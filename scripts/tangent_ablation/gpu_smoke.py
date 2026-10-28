"""Actual 1B smoke, exercising every direction and both benchmark loaders."""

import json
import os
from pathlib import Path
import subprocess
import sys


def main() -> None:
    root = Path(os.environ['ABLATION_ROOT'])
    records = []
    for variant in ('frobenius', 'one_step', 'factorwise', 'coupled'):
        dataset = 'siqa' if variant == 'factorwise' else 'boolq'
        run = root / ('smoke_' + variant)
        command = [sys.executable, '-u', str(Path(__file__).with_name('train.py')),
                   '--variant', variant, '--dataset', dataset, '--lr', '.005', '--seed', '42',
                   '--steps', '2', '--eval-every', '1', '--eval-max-samples', '4', '--output-dir', str(run)]
        subprocess.run(command, check=True)
        result = json.loads((run / 'result.json').read_text())
        records.append(result)
    result = {'status': 'passed', 'cases': records}
    (root / 'gpu-smoke.json').write_text(json.dumps(result, indent=2) + '\n')
    print('ABLATION_GPU_SMOKE ' + json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
