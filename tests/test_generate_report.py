from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_report_preserves_missing_measurements(tmp_path: Path):
    machine = tmp_path / "machine.json"
    machine.write_text(json.dumps({"os": "Windows", "processor": "development CPU"}), encoding="utf-8")
    summary = tmp_path / "summary.json"
    metric = {"count": 2, "accuracy": 0.5, "macro_f1": 0.5, "weighted_f1": 0.5,
              "roc_auc": None, "brier": None, "ece": 0.1,
              "probability_count": 0, "ece_count": 2}
    summary.write_text(json.dumps({"quality": {"overall": metric, "per_task": {"intent": metric}},
                                   "selective": {"strategy": "ranked_per_split",
                                                 "by_split": {"test": []}}}), encoding="utf-8")
    suite = tmp_path / "suite.json"
    suite.write_text(json.dumps({"status": "complete", "mode": "quick", "checkpoint": "english",
                                 "machine_file": str(machine),
                                 "quality": {"torch_fp32": {"custom": {"summary": str(summary)}}},
                                 "gates": {}, "latency": []}), encoding="utf-8")
    output = tmp_path / "reports"
    subprocess.run([sys.executable, str(ROOT / "scripts/generate_report.py"),
                    "--suite", str(suite), "--output-dir", str(output)],
                   check=True, capture_output=True, text=True)
    markdown = (output / "report.md").read_text(encoding="utf-8")
    assert "No production recommendation" in markdown
    assert "not identified as the target" in markdown
    assert "No exported artifact" in markdown
    assert "0.5000" in markdown
    assert "## CPU performance" in markdown
    assert not (output / "latency.csv").exists()
    assert (output / "report.html").is_file()
