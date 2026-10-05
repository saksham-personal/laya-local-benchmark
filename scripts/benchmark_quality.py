"""Measure one local Laya variant on one frozen JSONL dataset."""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from laya_bench.metrics import evaluate_quality, selective_report  # noqa: E402
from laya_bench.runtime import load_variant, predict_row, read_jsonl, write_jsonl  # noqa: E402


def balanced_prefix(rows: list[dict], limit: int) -> list[dict]:
    if limit <= 0 or limit >= len(rows):
        return rows
    buckets: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        buckets[row["task"]].append(row)
    selected: list[dict] = []
    while len(selected) < limit and any(buckets.values()):
        for task in sorted(buckets):
            if buckets[task] and len(selected) < limit:
                selected.append(buckets[task].pop(0))
    return selected


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--variant", choices=["torch_fp32", "onnx_fp32", "onnx_int8"], required=True)
    parser.add_argument("--checkpoint", default="english")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--max-len", type=int, default=512)
    parser.add_argument("--limit", type=int, default=0, help="balanced smoke subset; 0 means all")
    parser.add_argument("--output", type=Path, required=True, help="prediction JSONL path")
    args = parser.parse_args()
    rows = balanced_prefix(read_jsonl(args.dataset), args.limit)
    start = time.perf_counter()
    agent = load_variant(args.variant, args.model_dir, args.threads)
    load_seconds = time.perf_counter() - start
    predictions: list[dict] = []
    start = time.perf_counter()
    for index, row in enumerate(rows, 1):
        predictions.append(predict_row(agent, row, max_len=args.max_len,
                                       variant=f"{args.checkpoint}/{args.variant}"))
        if index % 50 == 0:
            print(f"Scored {index}/{len(rows)}", flush=True)
    predict_seconds = time.perf_counter() - start
    write_jsonl(args.output, predictions)
    result = {
        "checkpoint": args.checkpoint,
        "variant": args.variant,
        "dataset": str(args.dataset),
        "prediction_file": str(args.output),
        "rows": len(predictions),
        "threads": args.threads,
        "max_len": args.max_len,
        "load_seconds": load_seconds,
        "predict_seconds": predict_seconds,
        "quality": evaluate_quality(predictions),
        "selective": selective_report(predictions),
    }
    summary = args.output.with_suffix(".summary.json")
    summary.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"rows": len(predictions), "load_seconds": load_seconds,
                      "predict_seconds": predict_seconds, "output": str(args.output),
                      "summary": str(summary)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
