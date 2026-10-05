from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("run_suite", ROOT / "scripts" / "run_suite.py")
module = importlib.util.module_from_spec(spec)
assert spec and spec.loader
spec.loader.exec_module(module)


def row(i, gold, pred, probability):
    return {"id": str(i), "task": "external_research", "split": "test", "gold": gold,
            "pred": pred, "confidence": probability,
            "probabilities": {"no": 1 - probability, "yes": probability}}


def test_int8_gate_rejects_dropped_rare_recall():
    reference = [row(i, "yes" if i < 5 else "no", "yes" if i < 5 else "no", .9) for i in range(100)]
    candidate = [dict(r) for r in reference]
    candidate[0]["pred"] = "no"
    assert module.gate(reference, candidate, int8=True)["passed"] is False


def test_fp32_gate_accepts_matched_outputs():
    reference = [row(i, "yes" if i % 2 else "no", "yes" if i % 2 else "no", .9) for i in range(30)]
    assert module.gate(reference, [dict(r) for r in reference], int8=False)["passed"] is True
