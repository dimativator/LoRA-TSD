"""Queue selection, cross-worker claims and watchdog process-group checks."""

from concurrent.futures import ThreadPoolExecutor
from contextlib import redirect_stdout
import io
import json
import multiprocessing
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

import supervise


class NoopGuard:
    def __init__(self, *args, **kwargs):
        pass

    def __call__(self, *args, **kwargs):
        return {"status": "ok"}


def fake_run(root: Path, case: dict, worker: str) -> None:
    run = root / supervise.name(case)
    run.mkdir(exist_ok=False)
    time.sleep(.005)
    supervise.atomic_json(run / "result.json", {
        "status": "completed", "steps": 3000, "best_score": case["lr"],
        "worker": worker,
    })


def queue_worker(root: str, worker: int) -> None:
    os.environ["ABLATION_ROOT"] = root
    sys.argv = ["supervise.py", "--worker", str(worker)]
    with patch.object(supervise, "DiskGuard", NoopGuard), patch.object(supervise, "run_case", fake_run):
        # No training takes place, so shorten the idle poll in this test only.
        original_sleep = time.sleep
        with patch.object(supervise.time, "sleep", lambda seconds: original_sleep(min(seconds, .01))):
            with redirect_stdout(io.StringIO()):
                supervise.main()


class QueueTests(unittest.TestCase):
    def write_result(self, root: Path, case: dict, score: float = .7) -> None:
        run = root / supervise.name(case)
        run.mkdir(exist_ok=True)
        supervise.atomic_json(run / "result.json", {
            "status": "completed", "steps": 3000, "best_score": score,
        })

    def test_sweep_has_four_boolq_lrs_for_each_variant(self):
        cases = supervise.sweep_cases()
        self.assertEqual(len(cases), 16)
        self.assertEqual(len({supervise.name(case) for case in cases}), 16)
        for variant in supervise.VARIANTS:
            selected = [case for case in cases if case["variant"] == variant]
            self.assertEqual({case["lr"] for case in selected}, {.001, .003, .005, .01})
            self.assertTrue(all(case["dataset"] == "boolq" and case["seed"] == 42 for case in selected))

    def test_only_complete_variant_can_select_lr(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cases, complete = supervise.pending(root)
            self.assertEqual(len(cases), 16)
            self.assertFalse(complete)
            first_variant = [case for case in cases if case["variant"] == "frobenius"]
            for case in first_variant[:-1]:
                self.write_result(root, case)
            cases, complete = supervise.pending(root)
            self.assertEqual(len(cases), 16)
            self.assertFalse((root / "selected-frobenius.json").exists())
            self.write_result(root, first_variant[-1])
            cases, complete = supervise.pending(root)
            self.assertEqual(len(cases), 21)
            self.assertFalse(complete)
            selected = json.loads((root / "selected-frobenius.json").read_text())
            self.assertEqual(selected["lr"], .001)
            self.assertIn("sweep_frobenius_boolq_lr0.001_s42", selected["reused_boolq_seed42"])

    def test_twenty_confirmation_runs_and_all_36_required(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for case in supervise.sweep_cases():
                self.write_result(root, case, score=case["lr"])
            cases, complete = supervise.pending(root)
            self.assertEqual(len(cases), 36)
            self.assertFalse(complete)
            confirmation = [case for case in cases if case["phase"] == "confirm"]
            self.assertEqual(len(confirmation), 20)
            for variant in supervise.VARIANTS:
                selected = [case for case in confirmation if case["variant"] == variant]
                self.assertEqual({(case["dataset"], case["seed"]) for case in selected},
                                 {("boolq", 0), ("boolq", 1), ("siqa", 42), ("siqa", 0), ("siqa", 1)})
                self.assertTrue(all(case["lr"] == .01 for case in selected))
            for case in confirmation[:-1]:
                self.write_result(root, case)
            self.assertFalse(supervise.pending(root)[1])
            self.write_result(root, confirmation[-1])
            self.assertTrue(supervise.pending(root)[1])

    def test_rejects_incomplete_training_result(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            case = supervise.sweep_cases()[0]
            run = root / supervise.name(case)
            run.mkdir()
            (run / "result.json").write_text(json.dumps({"status": "completed", "steps": 250}))
            with self.assertRaisesRegex(RuntimeError, "Incomplete result"):
                supervise.pending(root)

    def test_same_pid_concurrent_atomic_writers_never_share_tempfile(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "selected.json"
            with ThreadPoolExecutor(max_workers=6) as pool:
                list(pool.map(lambda index: supervise.atomic_json(path, {"index": index}), range(60)))
            self.assertIn(json.loads(path.read_text())["index"], range(60))
            self.assertEqual(list(path.parent.glob("*.tmp")), [])

    def test_six_workers_claim_every_run_exactly_once(self):
        with tempfile.TemporaryDirectory() as directory:
            context = multiprocessing.get_context("fork")
            workers = [context.Process(target=queue_worker, args=(directory, worker)) for worker in range(6)]
            try:
                for worker in workers:
                    worker.start()
                for worker in workers:
                    worker.join(timeout=15)
                    self.assertEqual(worker.exitcode, 0)
                root = Path(directory)
                self.assertEqual(len(list((root / "claims").glob("*.json"))), 36)
                self.assertEqual(len(list(root.glob("*/result.json"))), 36)
                self.assertTrue(supervise.pending(root)[1])
            finally:
                for worker in workers:
                    if worker.is_alive():
                        worker.terminate()
                        worker.join(timeout=5)

    def test_storage_watchdog_stops_training_descendant(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "scripts/cloud").mkdir(parents=True)
            child_script = root / "training_child.py"
            child_script.write_text(
                "import signal,time\nfrom pathlib import Path\n"
                f"root=Path({str(root)!r})\n"
                "def stop(signum,frame):\n (root/'child_stopped').write_text(str(signum))\n raise SystemExit(0)\n"
                "signal.signal(signal.SIGTERM,stop)\n(root/'child_ready').write_text('ready')\n"
                "while True: time.sleep(.05)\n")
            supervisor_script = root / "fake_supervisor.py"
            supervisor_script.write_text(
                "import os,subprocess,sys,time\nfrom pathlib import Path\n"
                f"root=Path({str(root)!r})\n"
                "(root/'owned_group').write_text(str(os.getpgrp()))\n"
                f"child=subprocess.Popen([sys.executable,{str(child_script)!r}])\n"
                "child.wait()\n")
            (root / "scripts/cloud/run_tangent_runtime.sh").write_text(
                f'#!/bin/bash\nexec "{sys.executable}" "{supervisor_script}"\n')
            source = (
                "import sys\nfrom pathlib import Path\n"
                f"sys.path.insert(0,{str(Path(__file__).resolve().parent)!r})\n"
                "import bootstrap\n"
                f"bootstrap.ROOT=Path({str(root)!r})\n"
                "class Guard:\n"
                " def __init__(self,*args,**kwargs): pass\n"
                " def __call__(self,*args,**kwargs):\n"
                "  if (bootstrap.ROOT/'child_ready').exists(): raise RuntimeError('test_storage_stop')\n"
                "bootstrap.DiskGuard=Guard\nbootstrap.main()\n")
            env = {**os.environ, "ABLATION_ROOT": str(root / "campaign")}
            process = subprocess.Popen([sys.executable, "-c", source], env=env,
                                       stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            try:
                output, _ = process.communicate(timeout=15)
                self.assertNotEqual(process.returncode, 0)
                self.assertIn("ABLATION_STOPPED", output)
                self.assertTrue((root / "child_stopped").exists(), output)
            finally:
                if process.poll() is None:
                    process.kill()
                    process.wait(timeout=5)
                group = root / "owned_group"
                if group.exists():
                    try:
                        os.killpg(int(group.read_text()), signal.SIGKILL)
                    except ProcessLookupError:
                        pass


if __name__ == "__main__":
    unittest.main(verbosity=2)
