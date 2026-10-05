"""Bounded CPU latency and throughput benchmark for one Laya runtime."""
from __future__ import annotations

import argparse
import json
import math
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from laya_bench.runtime import load_variant  # noqa: E402

QUESTION = {"decision": {"type": "choice", "instructions": "Select the first appropriate information path.",
                         "criteria": {
                             "structured_database": "Look up exact fields in stored rows.",
                             "semantic_retrieval": "Match company descriptions by meaning.",
                             "external_research": "Find current facts from outside sources.",
                         }}}
PHRASE = "The company offers software for financial institutions and provides evidence about products, customers, revenue, and geography. "


def percentile(values: list[float], q: float) -> float:
    if not values:
        return float("nan")
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    left = math.floor(position)
    right = math.ceil(position)
    return ordered[left] + (ordered[right] - ordered[left]) * (position - left)


def state_for_bucket(agent, bucket: int) -> str:
    # Leave room for Laya's question/criteria markers within max_len=512.
    target = {128: 55, 256: 190, 512: 420}[bucket]
    state = PHRASE
    while len(agent.tok.encode(state, add_special_tokens=False)) < target:
        state += PHRASE
    return state


class PeakRSS:
    def __init__(self):
        self.peak = 0
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def __enter__(self):
        import psutil
        process = psutil.Process()

        def sample():
            while not self._stop.is_set():
                try:
                    self.peak = max(self.peak, process.memory_info().rss)
                except psutil.Error:
                    return
                self._stop.wait(0.02)
        self._thread = threading.Thread(target=sample, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *_):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=1)


def call(agent, state: str, batch: int):
    if batch == 1:
        return agent.system_one(state, QUESTION, max_len=512)
    return agent.predict_batch([state] * batch, QUESTION, batch_size=batch, max_len=512)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--variant", choices=["torch_fp32", "onnx_fp32", "onnx_int8"], required=True)
    parser.add_argument("--checkpoint", default="english")
    parser.add_argument("--threads", type=int, choices=[1, 4, 8], required=True)
    parser.add_argument("--lengths", type=int, nargs="+", choices=[128, 256, 512], default=[128, 256, 512])
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    warmups, singles, batches = (3, 20, 10) if args.quick else (10, 100, 50)
    with PeakRSS() as memory:
        load_start = time.perf_counter()
        agent = load_variant(args.variant, args.model_dir, args.threads)
        cold_load = time.perf_counter() - load_start
        initial_state = state_for_bucket(agent, 128)
        start = time.perf_counter()
        first = call(agent, initial_state, 1)
        first_call = time.perf_counter() - start
        results = []
        for bucket in args.lengths:
            state = state_for_bucket(agent, bucket)
            for batch in (1, 8):
                for _ in range(warmups):
                    call(agent, state, batch)
                times = []
                for _ in range(singles if batch == 1 else batches):
                    start = time.perf_counter()
                    response = call(agent, state, batch)
                    times.append((time.perf_counter() - start) * 1000)
                total_seconds = sum(times) / 1000
                usage = response[0].get("usage", {}) if isinstance(response, list) else response.get("usage", {})
                results.append({
                    "checkpoint": args.checkpoint, "variant": args.variant,
                    "threads": args.threads, "inter_op_threads": 1,
                    "execution_mode": "ORT_SEQUENTIAL" if args.variant.startswith("onnx") else "torch_eager",
                    "length_bucket": bucket, "observed_input_tokens": usage.get("input_tokens"),
                    "batch": batch, "warmup": warmups, "timed_calls": len(times),
                    "p50_ms": percentile(times, .50), "p95_ms": percentile(times, .95),
                    "p99_ms": percentile(times, .99),
                    "items_per_second": batch * len(times) / total_seconds,
                    "cold_load_seconds": cold_load, "first_call_ms": first_call * 1000,
                })
                print(f"{args.variant} t={args.threads} len={bucket} batch={batch}: "
                      f"p50={results[-1]['p50_ms']:.1f}ms p95={results[-1]['p95_ms']:.1f}ms",
                      flush=True)
    for result in results:
        result["sampled_peak_rss_bytes"] = memory.peak
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
