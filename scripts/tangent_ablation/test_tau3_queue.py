"""Check that the tau3 continuation contains only the nine requested runs."""
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
from unittest.mock import patch
import unittest


def load_queue():
    spec = importlib.util.spec_from_file_location('tau3_queue', Path(__file__).with_name('supervise.py'))
    queue = importlib.util.module_from_spec(spec)
    with patch.dict(os.environ, {'ABLATION_VARIANTS':'coupled', 'ABLATION_COUPLED_TAU':'3'}):
        spec.loader.exec_module(queue)
    return queue


class Tau3QueueTests(unittest.TestCase):
    def test_four_lrs_and_five_new_confirmation_runs(self):
        q = load_queue()
        self.assertEqual(q.VARIANTS, ('coupled',))
        self.assertEqual(len(q.sweep_cases()), 4)
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            for case in q.sweep_cases():
                run = root / q.name(case); run.mkdir()
                (run/'result.json').write_text(json.dumps({'status':'completed','steps':3000,
                    'best_score':1-abs(case['lr']-.003),'coupled_tau':3}))
            cases, done = q.pending(root)
            self.assertFalse(done)
            self.assertEqual(len(cases), 9)
            confirms = [c for c in cases if c['phase']=='confirm']
            self.assertEqual(len(confirms), 5)
            self.assertTrue(all(c['variant']=='coupled' and c['lr']==.003 for c in confirms))
            self.assertEqual({(c['dataset'],c['seed']) for c in confirms},
                             {('boolq',0),('boolq',1),('siqa',42),('siqa',0),('siqa',1)})
            for case in confirms:
                run = root / q.name(case); run.mkdir()
                (run/'result.json').write_text(json.dumps({'status':'completed','steps':3000,
                    'best_score':.8,'coupled_tau':3}))
            self.assertTrue(q.pending(root)[1])
            with patch.dict(os.environ, {'ABLATION_COUPLED_TAU':'3'}):
                summary=q.summarize(root)
            self.assertEqual(len(summary['rows']),2)
            self.assertTrue(all(r['coupled_tau']==3 for r in summary['rows']))

    def test_subprocess_explicitly_receives_tau3(self):
        q = load_queue()
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);case=q.sweep_cases()[0]
            class Child:
                pid=12345
                def wait(self): return 0
                def poll(self): return 0
            def launch(command, **kwargs):
                self.assertEqual(command[command.index('--coupled-tau')+1], '3')
                run=root/q.name(case)
                (run/'result.json').write_text(json.dumps({'status':'completed','steps':3000,
                                                          'best_score':.8,'coupled_tau':3}))
                return Child()
            with patch.dict(os.environ, {'ABLATION_COUPLED_TAU':'3'}), patch.object(q.subprocess,'Popen',launch):
                q.run_case(root,case,'0')
            self.assertEqual(json.loads((root/q.name(case)/'process.json').read_text())['coupled_tau'],3)


if __name__=='__main__': unittest.main(verbosity=2)
