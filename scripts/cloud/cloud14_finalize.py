"""Export the 180 verified seed results and authorize only the owned asset cleanup."""

import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import statistics

from cloud14_core import CAMPAIGN, digest, now, save
from cloud14_submit import ARTIFACTS, WORKSPACE


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    verified_file = ARTIFACTS / "wandb-independent-verification.json"
    verified = json.loads(verified_file.read_text())
    assert len(verified) == 180
    grid = {(row["method"], row["dataset"], row["seed"]) for row in verified.values()}
    assert len(grid) == 180
    inspection = json.loads((ARTIFACTS / "inspection-latest.json").read_text())
    assert len(inspection["workers"]) == 14
    assert all(row["phase"] == "complete" for row in inspection["workers"])
    assert len(inspection["tasks"]) == 138
    assert all(row["phase"] == "complete" for row in inspection["tasks"])
    scheduler = json.loads((ARTIFACTS / "scheduler-snapshot.json").read_text())
    age = (datetime.now(timezone.utc) - datetime.fromisoformat(scheduler["timestamp"])).total_seconds()
    assert 0 <= age < 300, "Refresh scheduler snapshot before final cleanup"
    campaign = json.loads((ARTIFACTS / "campaign.json").read_text())
    assert {row["job_id"] for row in scheduler["jobs"]} == {row["job_id"] for row in campaign["jobs"]}
    assert all(row["status"].lower() in ["completed", "failed", "stopped", "cancelled", "canceled"]
               for row in scheduler["jobs"])
    plan = json.loads((WORKSPACE / "artifacts/three-seed-all-optimizers-8b-plan-20261113.json").read_text())
    output = ARTIFACTS / "final"
    output.mkdir(exist_ok=True)
    raw = [{"method": row["method"], "dataset": row["dataset"], "seed": row["seed"],
            "best_accuracy_percent": row["best_accuracy"] * 100, "best_step": row["best_step"],
            "wandb_url": row["url"]} for row in verified.values()]
    write_csv(output / "per-seed-results.csv", raw)
    summary = []
    curves = []
    for method in plan["methods"]:
        for dataset in plan["datasets"]:
            observations = [row for row in raw if row["method"] == method["id"] and row["dataset"] == dataset]
            assert sorted(row["seed"] for row in observations) == [0, 1, 42]
            values = [row["best_accuracy_percent"] for row in observations]
            summary.append({"method": method["id"], "dataset": dataset, "n": 3,
                            "mean_percent": statistics.mean(values),
                            "sample_sd_percent": statistics.stdev(values)})
            for step in range(250, 3001, 250):
                scores = []
                for row in verified.values():
                    if row["method"] == method["id"] and row["dataset"] == dataset:
                        history = [json.loads(line) for line in Path(row["history_file"]).read_text().splitlines()]
                        scores.append(next(item["validation_score"] * 100 for item in history
                                           if "validation_score" in item and item["step"] == step))
                curves.append({"method": method["id"], "dataset": dataset, "step": step,
                               "mean_percent": statistics.mean(scores),
                               "sample_sd_percent": statistics.stdev(scores)})
    write_csv(output / "summary-mean-sample-sd.csv", summary)
    write_csv(output / "validation-curves.csv", curves)
    save(output / "summary.json", {"campaign": CAMPAIGN, "completed_at": now(), "runs": 180,
         "metric": "Per-seed best validation including final3000; mean and sampleSD across0/1/42",
         "independent_test_evaluation": False, "summary": summary, "raw": raw})
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    figure, axes = plt.subplots(2, 3, figsize=(16, 9), constrained_layout=True)
    for axis, dataset in zip(axes.flat, plan["datasets"]):
        values = [next(row for row in summary if row["method"] == method["id"] and row["dataset"] == dataset)
                  for method in plan["methods"]]
        axis.errorbar(range(10), [row["mean_percent"] for row in values],
                      yerr=[row["sample_sd_percent"] for row in values], fmt="o", capsize=3)
        axis.set_xticks(range(10), [method["method"] for method in plan["methods"]], rotation=65, ha="right")
        axis.set_title(dataset)
        axis.set_ylabel("Best validation accuracy (%)")
        axis.grid(axis="y", alpha=.2)
    figure.savefig(output / "best-validation-comparison.png", dpi=180)
    figure.savefig(output / "best-validation-comparison.pdf")
    plt.close(figure)
    figure, axes = plt.subplots(2, 3, figsize=(16, 8), constrained_layout=True)
    for axis, dataset in zip(axes.flat, plan["datasets"]):
        for method in plan["methods"]:
            values = [row for row in curves if row["method"] == method["id"] and row["dataset"] == dataset]
            x = [row["step"] for row in values]
            y = [row["mean_percent"] for row in values]
            sd = [row["sample_sd_percent"] for row in values]
            axis.plot(x, y, label=method["method"])
            axis.fill_between(x, [a-b for a,b in zip(y, sd)], [a+b for a,b in zip(y, sd)], alpha=.08)
        axis.set_title(dataset)
        axis.set_xlabel("Optimizer updates")
        axis.set_ylabel("Validation accuracy (%)")
    axes.flat[0].legend(fontsize=7, ncol=2)
    figure.savefig(output / "validation-curves.png", dpi=180)
    figure.savefig(output / "validation-curves.pdf")
    plt.close(figure)
    save(output / "artifact-manifest.json", {path.name: digest(path) for path in output.iterdir() if path.is_file()})
    audit = {"campaign": CAMPAIGN, "created_at": now(), "verified_runs": 180,
             "wandb_sync_verified": True, "results_export_verified": True,
             "all_owned_jobs_terminal": True, "audit_sha256": digest(verified_file),
             "export_manifest_sha256": digest(output / "artifact-manifest.json")}
    save(ARTIFACTS / "final-cleanup-authorization.json", audit)
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
