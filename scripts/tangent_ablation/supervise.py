"""Six independent GPU workers with a shared, claimed experiment queue."""

import argparse
import json
import os
from pathlib import Path
import signal
import statistics
import subprocess
import sys
import threading
import time
import uuid

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts/generation'))
from disk_guard import DiskGuard

VARIANTS = ('frobenius', 'one_step', 'factorwise', 'coupled')
LRS = (0.001, 0.003, 0.005, 0.01)
SEEDS = (42, 0, 1)


def name(case: dict) -> str:
    return f"{case['phase']}_{case['variant']}_{case['dataset']}_lr{case['lr']:g}_s{case['seed']}"


def atomic_json(path: Path, record: dict) -> None:
    temporary = path.with_name(path.name + f'.{uuid.uuid4().hex}.tmp')
    temporary.write_text(json.dumps(record, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def sweep_cases() -> list[dict]:
    return [dict(phase='sweep', variant=variant, dataset='boolq', lr=lr, seed=42)
            for lr in LRS for variant in VARIANTS]


def read_result(root: Path, case: dict) -> dict | None:
    path = root / name(case) / 'result.json'
    if not path.exists():
        return None
    result = json.loads(path.read_text())
    if result.get('status') != 'completed' or result.get('steps') != 3000:
        raise RuntimeError('Incomplete result cannot be selected: ' + str(path))
    return result


def pending(root: Path) -> tuple[list[dict], bool]:
    cases = sweep_cases()
    all_done = True
    for variant in VARIANTS:
        candidates = [case for case in cases if case['variant'] == variant]
        results = [(case, read_result(root, case)) for case in candidates]
        if any(result is None for _, result in results):
            all_done = False
            continue
        # Deterministic tie breaking: prefer the smaller LR.
        best, result = max(results, key=lambda pair: (pair[1]['best_score'], -pair[0]['lr']))
        selection = {'variant': variant, 'lr': best['lr'], 'selection_dataset': 'boolq',
                     'selection_seed': 42, 'metric': 'max accuracy over steps250..3000',
                     'candidates': [{'lr': c['lr'], 'best_score': r['best_score']} for c, r in results],
                     'reused_boolq_seed42': name(best)}
        atomic_json(root / f'selected-{variant}.json', selection)
        for dataset in ('boolq', 'siqa'):
            for seed in SEEDS:
                if dataset == 'boolq' and seed == 42:
                    continue  # Identical complete selected sweep run is reused.
                case = dict(phase='confirm', variant=variant, dataset=dataset, lr=best['lr'], seed=seed)
                cases.append(case)
                if read_result(root, case) is None:
                    all_done = False
    return cases, all_done


def run_case(root: Path, case: dict, worker: str) -> None:
    run = root / name(case)
    run.mkdir(exist_ok=False)
    command = [sys.executable, '-u', str(Path(__file__).with_name('train.py')),
               '--variant', case['variant'], '--dataset', case['dataset'], '--lr', str(case['lr']),
               '--seed', str(case['seed']), '--steps', '3000', '--eval-every', '250',
               '--output-dir', str(run)]
    record = {**case, 'run_name': run.name, 'worker': worker, 'job_id': os.environ.get('HOSTNAME'),
              'command': command, 'status': 'starting', 'started_at': time.time(), 'checkpoints': False}
    atomic_json(run / 'process.json', record)
    with (run / 'train.log').open('w', buffering=1) as log:
        child = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
        record.update(status='running', pid=child.pid)
        atomic_json(run / 'process.json', record)
        print('ABLATION_STARTED ' + json.dumps(record), flush=True)
        try:
            code = child.wait()
        finally:
            if child.poll() is None:
                child.terminate()
        record.update(returncode=code, ended_at=time.time(), status='completed' if code == 0 else 'failed')
        atomic_json(run / 'process.json', record)
    print('ABLATION_FINISHED ' + json.dumps(record), flush=True)
    print('\n'.join((run / 'train.log').read_text().splitlines()[-15:]), flush=True)
    if code != 0 or read_result(root, case) is None:
        atomic_json(root / 'STOP_FAILURE.json', record)
        raise RuntimeError('Training failed. Queue stopped for inspection: ' + run.name)


def summarize(root: Path) -> dict:
    rows = []
    for variant in VARIANTS:
        selected = json.loads((root / f'selected-{variant}.json').read_text())
        for dataset in ('boolq', 'siqa'):
            results = []
            for seed in SEEDS:
                case = dict(phase='sweep' if dataset == 'boolq' and seed == 42 else 'confirm',
                            variant=variant, dataset=dataset, lr=selected['lr'], seed=seed)
                result = read_result(root, case)
                if result is None:
                    raise RuntimeError('Cannot summarize an incomplete campaign')
                results.append({'seed': seed, 'best_score': result['best_score'],
                                'best_step': result.get('best_step'), 'run': name(case)})
            scores = [record['best_score'] for record in results]
            rows.append({'variant': variant, 'dataset': dataset, 'lr': selected['lr'],
                         'mean_best_accuracy': statistics.mean(scores),
                         'sample_std_best_accuracy': statistics.stdev(scores), 'seeds': results})
    return {'status': 'completed', 'timestamp': time.time(), 'rows': rows,
            'metric': 'best accuracy at scheduled evaluations250..3000; mean and sample std across3 seeds'}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--worker', required=True)
    args = parser.parse_args()
    root = Path(os.environ['ABLATION_ROOT'])
    guard = DiskGuard(root, 'queue-' + args.worker, root / f'disk-worker-{args.worker}.jsonl')
    guard(pending_write_bytes=100_000_000, reason='before_queue')
    root.mkdir(parents=True, exist_ok=True)
    claims = root / 'claims'
    claims.mkdir(exist_ok=True)
    while True:
        guard()
        if (root / 'STOP_FAILURE.json').exists():
            raise RuntimeError('Campaign training failure signal present')
        cases, complete = pending(root)
        if complete:
            atomic_json(root / 'campaign-results.json', summarize(root))
            print('ABLATION_QUEUE_COMPLETE', flush=True)
            return
        claimed = False
        for case in cases:
            if read_result(root, case) is not None:
                continue
            claim = claims / (name(case) + '.json')
            try:
                with claim.open('x') as stream:
                    json.dump({'worker': args.worker, 'job': os.environ.get('HOSTNAME'),
                               'claimed_at': time.time(), **case}, stream)
            except FileExistsError:
                continue
            claimed = True
            run_case(root, case, args.worker)
            break
        if not claimed:
            time.sleep(10)


if __name__ == '__main__':
    main()
