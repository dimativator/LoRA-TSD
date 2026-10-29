"""Read-only model inventory, exact-byte mount checks, and CPU integration smoke."""

import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from train import smoke, _data_module
from initialization import factor_digest
import torch
from disk_guard import DiskGuard
from huggingface_hub import HfApi, try_to_load_from_cache


def main() -> None:
    root = Path(os.environ['ABLATION_ROOT'])
    guard = DiskGuard(root, 'cpu-preflight')
    disk = guard(pending_write_bytes=3_000_000_000, reason='reserve_1b_model',
                 write_path=Path(os.environ['HF_HOME']))
    script = "import os,json; p='/home/jovyan/shares/SR006.nfs2'; s=os.statvfs(p); print(json.dumps({'path':p,'available_bytes':s.f_bavail*s.f_frsize,'used_by_campaign':False}))"
    try:
        probe = subprocess.run([sys.executable, '-c', script], capture_output=True, text=True, timeout=10)
        nfs2 = json.loads(probe.stdout) if probe.returncode == 0 else {'status': 'telemetry_unavailable', 'error': probe.stderr[-500:], 'used_by_campaign': False}
    except subprocess.TimeoutExpired:
        nfs2 = {'status': 'telemetry_unavailable', 'error': 'read-only statvfs timeout', 'used_by_campaign': False}
    model = 'meta-llama/Llama-3.2-1B-Instruct'
    info = HfApi().model_info(model, files_metadata=True, token=os.environ.get('HF_TOKEN'))
    files = [{'path': f.rfilename, 'bytes': f.size,
              'cached': isinstance(try_to_load_from_cache(model, f.rfilename, revision=info.sha), str)}
             for f in info.siblings if f.rfilename.endswith('.safetensors')]
    data = []
    for dataset in ('boolq', 'siqa'):
        for split in ('train', 'validation'):
            _, record = _data_module.load_split(dataset, split, Path('/tmp/tangent-ablation-data'), guard)
            data.append({'dataset': dataset, 'split': split, **record})
    tests = subprocess.run([sys.executable, str(Path(__file__).with_name('test_optimizer.py'))],
                           capture_output=True, text=True, timeout=180)
    print(tests.stdout + tests.stderr, flush=True)
    if tests.returncode:
        raise RuntimeError('Optimizer numerical tests failed')
    queue_tests = subprocess.run([sys.executable, str(Path(__file__).with_name('test_tau3_queue.py'))],
                                 capture_output=True, text=True, timeout=60)
    print(queue_tests.stdout + queue_tests.stderr, flush=True)
    if queue_tests.returncode:
        raise RuntimeError('Tau3 queue tests failed')
    expected = json.loads(Path(__file__).with_name('expected_initializations.json').read_text())
    source = Path(os.environ['ABLATION_INITIALIZATION_ROOT'])
    reused = []
    for case, digest in expected.items():
        dataset, seed = case.split(':')
        path = source / 'initializations' / f'init-{dataset}-s{seed}.pt'
        payload = torch.load(path, map_location='cpu', weights_only=True)
        assert factor_digest(payload['factors']) == payload['adapter_sha256'] == digest
        reused.append({'dataset':dataset,'seed':int(seed),'sha256':digest,'path':str(path)})
    result = {'status': 'passed', 'timestamp': time.time(), 'disk': disk, 'nfs2': nfs2,
              'model': model, 'model_revision': info.sha, 'model_files': files,
              'data': data, 'optimizer_tests': 'passed', 'smoke': smoke(coupled_tau=int(os.environ.get('ABLATION_COUPLED_TAU','5'))),
              'paired_initializations':reused, 'coupled_tau':int(os.environ.get('ABLATION_COUPLED_TAU','5')), 'checkpoints': False}
    guard(pending_write_bytes=1_000_000, reason='before_preflight_result')
    (root / 'preflight.json').write_text(json.dumps(result, indent=2) + '\n')
    print('ABLATION_PREFLIGHT ' + json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
