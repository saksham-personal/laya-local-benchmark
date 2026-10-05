"""Staged, bounded Laya quality-first benchmark orchestration."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from laya_bench.metrics import compare_variants  # noqa: E402
from laya_bench.runtime import read_jsonl  # noqa: E402

CUSTOM = ROOT / "data/synthetic/company_routing.jsonl"
PUBLIC = ROOT / "data/public/bfcl_v4_live_tool_selection.jsonl"


def run(*parts: str) -> None:
    command = [sys.executable, *parts]
    print("+", " ".join(command), flush=True)
    subprocess.run(command, check=True, cwd=ROOT)


def gate(reference: list[dict], candidate: list[dict], *, int8: bool) -> dict:
    left = [row for row in reference if row["split"] == "test"]
    right = [row for row in candidate if row["split"] == "test"]
    comparison = compare_variants(left, right)
    tasks = sorted({row["task"] for row in left} | {row["task"] for row in right})
    per_task = {task: compare_variants(
        [row for row in left if row["task"] == task],
        [row for row in right if row["task"] == task]) for task in tasks}
    worst_recall = min(
        (delta for result in per_task.values()
         for delta in result["per_class_recall_delta"].values() if delta is not None),
        default=0)
    worst_accuracy = min(
        (result["accuracy_delta"] for result in per_task.values()
         if result["accuracy_delta"] is not None), default=0)
    if int8:
        passed = (comparison["matched"] == len(left) == len(right)
                  and comparison["agreement"] is not None and comparison["agreement"] >= .95
                  and comparison["accuracy_delta"] is not None and comparison["accuracy_delta"] >= -.02
                  and worst_accuracy >= -.05
                  and worst_recall >= -.05)
        criteria = {"min_agreement": .95, "max_accuracy_drop": .02,
                    "max_task_accuracy_drop": .05, "max_task_class_recall_drop": .05}
    else:
        passed = (comparison["matched"] == len(left) == len(right)
                  and comparison["agreement"] is not None and comparison["agreement"] >= .98
                  and comparison["accuracy_delta"] is not None and comparison["accuracy_delta"] >= -.01
                  and worst_accuracy >= -.03 and worst_recall >= -.05)
        criteria = {"min_agreement": .98, "max_accuracy_drop": .01,
                    "max_task_accuracy_drop": .03, "max_task_class_recall_drop": .05}
    return {"passed": passed, "criteria": criteria, "comparison": comparison,
            "per_task": per_task, "basis": "held-out application-specific test split"}


def main(*, quick: bool = False) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-dir", type=Path, required=True, help="verified model/runtime root")
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--checkpoint", choices=["english", "typed-decisions"], default="english")
    parser.add_argument("--force", action="store_true", help="replace previous results")
    parser.add_argument("--skip-latency", action="store_true")
    args = parser.parse_args()
    model = (args.work_dir / "models" / args.checkpoint).resolve()
    if not (model / "model.safetensors").is_file():
        parser.error(f"Verified local model missing: {model}")
    for dataset in (CUSTOM, PUBLIC):
        if not dataset.is_file():
            parser.error(f"Frozen dataset missing: {dataset}")
    raw = args.results / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    suite_file = args.results / f"{args.checkpoint}-{'quick' if quick else 'full'}.json"
    if suite_file.exists() and not args.force:
        parser.error(f"{suite_file} exists; pass --force to replace")
    machine_file = args.results / "machine.json"
    run("scripts/inspect_machine.py", "--output", str(machine_file))
    available = ["torch_fp32"]
    if (model / "laya.onnx").is_file():
        available.append("onnx_fp32")
    if (model / "laya.int8.onnx").is_file():
        available.append("onnx_int8")
    suite: dict = {"checkpoint": args.checkpoint, "mode": "quick" if quick else "full",
                   "model_dir": str(model), "machine_file": str(machine_file),
                   "quality": {}, "gates": {}, "latency": [], "status": "in_progress"}
    for variant in available:
        suite["quality"][variant] = {}
        for label, dataset in (("custom", CUSTOM), ("public", PUBLIC)):
            output = raw / f"{args.checkpoint}-{variant}-{label}-{'quick' if quick else 'full'}.jsonl"
            command = ["scripts/benchmark_quality.py", "--dataset", str(dataset),
                       "--model-dir", str(model), "--variant", variant,
                       "--checkpoint", args.checkpoint, "--threads", "4",
                       "--output", str(output)]
            if quick:
                command.extend(["--limit", "30"])
            run(*command)
            suite["quality"][variant][label] = {
                "predictions": str(output), "summary": str(output.with_suffix(".summary.json"))}
    reference = read_jsonl(suite["quality"]["torch_fp32"]["custom"]["predictions"])
    if "onnx_fp32" in available:
        suite["gates"]["onnx_fp32"] = gate(
            reference, read_jsonl(suite["quality"]["onnx_fp32"]["custom"]["predictions"]), int8=False)
    if "onnx_int8" in available:
        suite["gates"]["onnx_int8"] = gate(
            read_jsonl(suite["quality"].get("onnx_fp32", suite["quality"]["torch_fp32"])["custom"]["predictions"]),
            read_jsonl(suite["quality"]["onnx_int8"]["custom"]["predictions"]), int8=True)
    if not args.skip_latency:
        passed = ["torch_fp32"]
        if suite["gates"].get("onnx_fp32", {}).get("passed"):
            passed.append("onnx_fp32")
            if suite["gates"].get("onnx_int8", {}).get("passed"):
                passed.append("onnx_int8")
        for variant in passed:
            if quick:
                output = raw / f"{args.checkpoint}-{variant}-t4-quick-latency.json"
                run("scripts/benchmark_latency.py", "--model-dir", str(model),
                    "--variant", variant, "--checkpoint", args.checkpoint,
                    "--threads", "4", "--lengths", "128", "--quick", "--output", str(output))
                suite["latency"].append(str(output))
                continue
            sweep: list[tuple[int, float]] = []
            for threads in (1, 4, 8):
                output = raw / f"{args.checkpoint}-{variant}-t{threads}-sweep-latency.json"
                run("scripts/benchmark_latency.py", "--model-dir", str(model),
                    "--variant", variant, "--checkpoint", args.checkpoint,
                    "--threads", str(threads), "--lengths", "128", "--quick",
                    "--output", str(output))
                suite["latency"].append(str(output))
                rows = json.loads(output.read_text(encoding="utf-8"))
                sweep.append((threads, next(row["p95_ms"] for row in rows if row["batch"] == 1)))
            best = min(sweep, key=lambda item: (item[1], item[0]))[0]
            output = raw / f"{args.checkpoint}-{variant}-t{best}-full-latency.json"
            run("scripts/benchmark_latency.py", "--model-dir", str(model),
                "--variant", variant, "--checkpoint", args.checkpoint,
                "--threads", str(best), "--output", str(output))
            suite["latency"].append(str(output))
            suite.setdefault("selected_threads", {})[variant] = best
    suite["status"] = "complete"
    suite_file.write_text(json.dumps(suite, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    run("scripts/generate_report.py", "--suite", str(suite_file), "--output-dir", str(args.results / "reports"))
    print(suite_file)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
