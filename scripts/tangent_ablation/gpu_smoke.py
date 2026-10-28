"""Actual 1B smoke, exercising every direction and both benchmark loaders."""

import json
import os
from pathlib import Path
import subprocess
import sys


def main() -> None:
    root = Path(os.environ['ABLATION_ROOT'])
    records = []
    cases = [(variant, 'boolq') for variant in ('frobenius', 'one_step', 'factorwise', 'coupled')] + [('coupled', 'siqa')]
    for variant, dataset in cases:
        run = root / ('smoke_paired_' + dataset + '_' + variant)
        command = [sys.executable, '-u', str(Path(__file__).with_name('train.py')),
                   '--variant', variant, '--dataset', dataset, '--lr', '.005', '--seed', '42',
                   '--steps', '2', '--eval-every', '1', '--eval-max-samples', '4', '--output-dir', str(run)]
        subprocess.run(command, check=True)
        result = json.loads((run / 'result.json').read_text())
        records.append(result)
    digests = {record['initial_adapter_sha256'] for record in records if record['dataset'] == 'boolq'}
    if len(digests) != 1:
        raise RuntimeError('The four BoolQ variants must start from identical factor tensors')
    result = {'status': 'passed', 'cases': records, 'paired_initialization_verified': True}
    (root / 'gpu-smoke.json').write_text(json.dumps(result, indent=2) + '\n')
    print('ABLATION_GPU_SMOKE ' + json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
