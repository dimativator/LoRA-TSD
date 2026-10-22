import json
import importlib.util
import os
from pathlib import Path
import subprocess
import sys
import types

import pytest
import torch

from args_parser import parse_config
from optimizers import build_optimizer
from optimizers.schedulers import make_torch_linear_warmup_decay

_spec = importlib.util.spec_from_file_location('checkpointing', Path(__file__).resolve().parents[1] / 'src/run/checkpointing.py')
_checkpointing = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_checkpointing)
load_training_checkpoint = _checkpointing.load_training_checkpoint
save_training_checkpoint = _checkpointing.save_training_checkpoint


@pytest.mark.parametrize('name', ['RiemannionOriginal', 'riemannion_original'])
def test_original_cli_alias_and_factory(name):
    config = parse_config(['--run-name', 'test', '--model', 'model', '--dataset', 'siqa',
                           '--optimizer', name, '--lr-a', '3e-5', '--retraction', 'factor'])
    a = torch.nn.Parameter(torch.randn(4, 9))
    b = torch.nn.Parameter(torch.randn(11, 4))
    optimizer = build_optimizer([{'params': [a, b], 'param_names': ['lora_A.weight', 'lora_B.weight']}], config)
    assert optimizer.__class__.__name__ == 'RiemannionOriginal'
    assert optimizer.param_groups[0]['retraction'] == 'factor'


def test_original_scheduler_boundaries_and_resume(tmp_path):
    model = torch.nn.Linear(1, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=3e-5)
    scheduler = make_torch_linear_warmup_decay(optimizer, 3000, min_lr=1e-5)
    used = []
    for _ in range(300):
        used.append(optimizer.param_groups[0]['lr'])
        optimizer.step()
        scheduler.step()
    assert used[0] == pytest.approx(1e-7)
    assert used[-1] == pytest.approx(3e-5)
    config = {'checkpoint_dir': str(tmp_path)}
    save_training_checkpoint(config, model, optimizer, 300, 0, 300, None, [], [], scheduler)
    restored = torch.optim.SGD(model.parameters(), lr=3e-5)
    restored_scheduler = make_torch_linear_warmup_decay(restored, 3000, min_lr=1e-5)
    load_training_checkpoint(config, model, restored, restored_scheduler)
    assert restored_scheduler.last_epoch == scheduler.last_epoch
    assert restored.param_groups[0]['lr'] == optimizer.param_groups[0]['lr']
    for _ in range(300, 3000):
        used.append(restored.param_groups[0]['lr'])
        restored.step()
        restored_scheduler.step()
    assert used[-1] == pytest.approx(1e-5)


def test_ablation_commands_cover_maps_seeds_and_fixed_settings(tmp_path):
    capture = tmp_path / 'commands.jsonl'
    fake = tmp_path / 'python'
    fake.write_text('#!' + sys.executable + '\nimport json,os,sys\n'
                    'with open(os.environ["CAPTURE"],"a") as stream: stream.write(json.dumps(sys.argv[1:])+"\\n")\n')
    fake.chmod(0o755)
    root = Path(__file__).resolve().parents[1]
    env = dict(os.environ, PYTHON=str(fake), CAPTURE=str(capture), METHODS='riemannion_original lora_tsd',
               DATASETS='siqa', SEEDS='42 0 1', RETRACTIONS='factor svd orthographic ksl')
    subprocess.run(['bash', str(root / 'scripts/retraction_ablation/run.sh')], env=env, check=True)
    configs = [parse_config(json.loads(line)[1:]) for line in capture.read_text().splitlines()]
    assert len(configs) == 24
    assert len({c['run_name'] for c in configs}) == 24
    for c in configs:
        assert c['steps'] == 3000 and c['eval_every'] == 300
        assert c['train_batch_size'] * c['grad_accum_steps'] == 32
        assert c['grad_checkpointing'] and c['checkpoint_every'] == 0 and c['checkpoint_dir'] is None
        assert c['lr_scheduler'] == 'linear' and c['min_lr'] == 1e-5
        if c['optimizer'] == 'riemannion_original':
            assert c['lr_A'] == 3e-5 and c['riemannian_momentum'] == .95
        else:
            assert c['lr_A'] == .005 and c['retraction_momentum'] == 'projected'


def test_training_loop_logs_exact_eval_boundary_and_used_lr(tmp_path, monkeypatch):
    class Progress:
        def __init__(self, **kwargs):
            pass

        def update(self, *args):
            pass

        def set_postfix(self, **kwargs):
            pass

        def close(self):
            pass

    class BatchValue:
        def to(self, device):
            return torch.ones(1)

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones(1))

        def forward(self, x):
            return types.SimpleNamespace(loss=(self.weight * x).square().mean())

    package = types.ModuleType('training_test')
    package.__path__ = [str(Path(__file__).resolve().parents[1] / 'src/run')]
    monkeypatch.setitem(sys.modules, 'training_test', package)
    monkeypatch.setitem(sys.modules, 'tqdm', types.SimpleNamespace(tqdm=Progress))
    import importlib
    train = importlib.import_module('training_test.train').train
    model = Model()
    optimizer = torch.optim.SGD(model.parameters(), lr=3e-5)
    scheduler = make_torch_linear_warmup_decay(optimizer, 6, min_lr=1e-5)
    config = dict(steps=6, run_name='cpu', grad_accum_steps=2, grad_clip_norm=0,
                  eval_every=3, train_batch_size=2, checkpoint_every=0,
                  metrics_jsonl=str(tmp_path / 'history.jsonl'))
    losses, validation = train(config, model, optimizer, [{'x': BatchValue()} for _ in range(12)],
                               None, lambda *args: .75, optimizer_lr_scheduler=scheduler)
    records = [json.loads(line) for line in (tmp_path / 'history.jsonl').read_text().splitlines()]
    assert len(losses) == 6
    assert validation == [{'step': 3, 'accuracy': .75}]
    assert [r['step'] for r in records if 'validation_score' in r] == [3]
    logged_lrs = [r['lr_A_used'] for r in records if 'train/loss' in r]
    assert logged_lrs[0] == pytest.approx(3e-5)
    assert logged_lrs[-1] == pytest.approx(1e-5)
