from __future__ import annotations

import importlib.util
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "benchmark_latency.py"
spec = importlib.util.spec_from_file_location("benchmark_latency", SCRIPT)
module = importlib.util.module_from_spec(spec)
assert spec and spec.loader
spec.loader.exec_module(module)


def test_percentile_interpolates():
    assert module.percentile([1, 3, 5, 7, 9], .5) == 5
    assert module.percentile([1, 3, 5, 7], .5) == 4
    assert module.percentile([9], .95) == 9
