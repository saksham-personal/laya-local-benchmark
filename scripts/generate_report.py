"""Generate an auditable report from measured suite outputs only."""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from laya_bench.reporting import render_html  # noqa: E402


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    keys = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def fmt(value: object) -> str:
    return "—" if value is None else f"{value:.4f}" if isinstance(value, float) else str(value)


def table(rows: list[dict], keys: list[str]) -> list[str]:
    return ["| " + " | ".join(keys) + " |",
            "|" + "|".join("---" for _ in keys) + "|"] + [
            "| " + " | ".join(fmt(row.get(key)) for key in keys) + " |" for row in rows
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    suite = json.loads(args.suite.read_text(encoding="utf-8"))
    machine = json.loads(Path(suite["machine_file"]).read_text(encoding="utf-8"))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    quality, calibration, selective, latency = [], [], [], []
    details = {}
    for variant, datasets in suite["quality"].items():
        details[variant] = {}
        for dataset, files in datasets.items():
            summary = json.loads(Path(files["summary"]).read_text(encoding="utf-8"))
            q = summary["quality"]
            details[variant][dataset] = q
            for task, metrics in [("overall", q["overall"]), *q["per_task"].items()]:
                quality.append({"variant": variant, "dataset": dataset, "task": task,
                                "rows": metrics["count"], "accuracy": metrics["accuracy"],
                                "macro_f1": metrics["macro_f1"], "weighted_f1": metrics["weighted_f1"],
                                "roc_auc": metrics["roc_auc"]})
                calibration.append({"variant": variant, "dataset": dataset, "task": task,
                                    "brier": metrics["brier"], "ece": metrics["ece"],
                                    "probability_count": metrics["probability_count"],
                                    "ece_count": metrics["ece_count"]})
            for split, entries in summary["selective"]["by_split"].items():
                for entry in entries:
                    selective.append({"variant": variant, "dataset": dataset, "split": split,
                                      "strategy": summary["selective"]["strategy"], **entry})
    for path in suite.get("latency", []):
        latency.extend(json.loads(Path(path).read_text(encoding="utf-8")))
    write_csv(args.output_dir / "quality.csv", quality)
    write_csv(args.output_dir / "calibration.csv", calibration)
    write_csv(args.output_dir / "selective_routing.csv", selective)
    write_csv(args.output_dir / "latency.csv", latency)
    processor = str(machine.get("processor") or "").lower()
    host_os = str(machine.get("os") or "").lower()
    target_host = (
        "6240r" in processor and "windows" in host_os
        and machine.get("logical_processors") == 8
        and machine.get("torch", {}).get("cuda_available") is False
    )
    recommendation = "No production recommendation: representative, human-judged company data and target VDI runs are required."
    if not target_host:
        recommendation += " This host is not identified as the target Xeon Gold 6240R VDI."
    overview = {
        "suite_status": suite.get("status"),
        "mode": suite.get("mode"),
        "checkpoint": suite.get("checkpoint"),
        "machine": machine,
        "target_vdi_identified": target_host,
        "artifact_gates": suite.get("gates", {}),
        "recommendation": recommendation,
        "quality_rows": quality,
        "calibration_rows": calibration,
        "selective_rows": selective,
        "latency_rows": latency,
        "per_task_metrics": details,
    }
    (args.output_dir / "report.json").write_text(
        json.dumps(overview, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    (args.output_dir / "report.html").write_text(
        render_html(overview, "Laya local benchmark") + "\n", encoding="utf-8")
    lines = ["# Laya local benchmark", "", recommendation, "",
             "## Scope", "",
             f"Suite: {suite.get('mode')}; checkpoint: {suite.get('checkpoint')}; "
             f"machine: {'target VDI' if target_host else 'development host or unverified VDI'}.",
             "Custom cases are synthetic development evidence. BFCL cases measure a nearby function-choice task, not company routing or BFCL leaderboard accuracy.",
             "Quality, runtime, and memory measurements are comparable only on matching hardware, dataset, thread count, batch size, and token bucket.",
             "", "## Quality", ""]
    lines += table(quality, ["variant", "dataset", "task", "rows", "accuracy", "macro_f1", "roc_auc"])
    lines += ["", "## Calibration", ""]
    lines += table(calibration, ["variant", "dataset", "task", "brier", "ece"])
    lines += ["", "## Selective routing", ""]
    lines += table([row for row in selective if row["split"] == "test"],
                   ["variant", "dataset", "target_coverage", "achieved_coverage",
                    "accepted_accuracy", "escalation_rate"])
    lines += ["", "## Artifact gates", ""]
    for variant, gate in suite.get("gates", {}).items():
        lines.append(f"- {variant}: {'PASS' if gate['passed'] else 'FAIL'}; "
                     f"agreement {fmt(gate['comparison']['agreement'])}; "
                     f"accuracy delta {fmt(gate['comparison']['accuracy_delta'])}")
    if not suite.get("gates"):
        lines.append("No exported artifact was available for comparison.")
    lines += ["", "## CPU performance", ""]
    lines += table(latency, ["variant", "threads", "length_bucket", "batch", "p50_ms",
                             "p95_ms", "p99_ms", "items_per_second", "sampled_peak_rss_bytes"])
    lines += ["", "## Provenance", "",
              f"Raw suite: {args.suite.resolve()}",
              "Raw predictions and machine metadata are retained beside the suite file.",
              "Missing values indicate unavailable measurements, not zero.", ""]
    (args.output_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")
    print(args.output_dir / "report.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
