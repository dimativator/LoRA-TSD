"""Publish previously verified histories to the same W&B group without rerunning training."""

import hashlib
import json
import netrc
import os
from pathlib import Path
import tempfile

from cloud14_core import CAMPAIGN, now, save, verify_rows, wandb_row
from cloud14_submit import ARTIFACTS, WORKSPACE


def main() -> None:
    import wandb
    os.environ["WANDB_BASE_URL"] = "https://wandb-radfan.ru"
    os.environ["WANDB_API_KEY"] = netrc.netrc().authenticators("wandb-radfan.ru")[2]
    old = WORKSPACE / "artifacts/paper_8b_additional_seeds_20261025"
    records = json.loads((old / "completed-run-metrics.json").read_text())
    records += json.loads((old / "excluded-completed-run-metrics-20261026T063558.json").read_text())
    imported_path = ARTIFACTS / "reused-wandb-runs.json"
    imported = json.loads(imported_path.read_text()) if imported_path.exists() else {}
    for record in records:
        name = record["run_name"]
        if name in imported:
            continue
        history_path = old / "raw" / record["host"] / name / "history.jsonl"
        rows = [json.loads(line) for line in history_path.read_text().splitlines()]
        task = {"optimizer": record["raw_result"]["optimizer"], "dataset": record["dataset"],
                "seed": record["seed"]}
        verification = verify_rows(rows, record["raw_result"], task)
        run_id = hashlib.sha256((CAMPAIGN + ":reused:" + name).encode()).hexdigest()[:16]
        with tempfile.TemporaryDirectory(prefix="paper8b-import-") as temporary:
            run = wandb.init(entity="andrey", project="LoRA-Muon", group=CAMPAIGN,
                             id=run_id, resume="allow", name=name, mode="online", dir=temporary,
                             job_type="reused-verified-run",
                             config={"campaign": CAMPAIGN, "reused": True, "run_name": name,
                                     **task, "steps": 3000, "eval_every": 250, "effective_batch": 32,
                                     "original_started_at": record["started_at"],
                                     "original_ended_at": record["ended_at"]},
                             settings=wandb.Settings(save_code=False, disable_git=True))
            run.define_metric("optimizer_step")
            run.define_metric("*", step_metric="optimizer_step")
            # A previous interrupted import is diagnosed before retrying, preventing
            # duplicate histories from being treated as independent seed observations.
            assert run.step == 0, "Partial reused-run import requires explicit history audit"
            for row in rows:
                run.log(wandb_row(row))
            run.summary.update({**verification, "verified": True, "reused": True,
                                "returncode": 0, "checkpoint_count": 0})
            result_path = Path(temporary) / "metrics.json"
            result_path.write_text(json.dumps(record["raw_result"], indent=2) + "\n")
            run.save(str(history_path), base_path=str(history_path.parent), policy="now")
            run.save(str(result_path), base_path=temporary, policy="now")
            path, url = run.path, run.url
            run.finish(exit_code=0)
            remote = wandb.Api(timeout=45).run(path)
            assert remote.summary["verified"] is True
            imported[name] = {"path": path, "url": url, "imported_at": now(), **verification}
            save(imported_path, imported)
            print(json.dumps({"reused_imported": len(imported), "target": 42, "url": url}), flush=True)


if __name__ == "__main__":
    main()
