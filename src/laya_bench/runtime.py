"""Load pinned local Laya artifacts and normalize its typed answers."""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
VENDOR = ROOT / "vendor"


def prepare_offline(threads: int) -> None:
    if threads < 1:
        raise ValueError("threads must be positive")
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_DATASETS_OFFLINE"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["USE_TF"] = "0"
    os.environ["OMP_NUM_THREADS"] = str(threads)
    os.environ["MKL_NUM_THREADS"] = str(threads)
    os.environ["LAYA_BENCH_ORT_THREADS"] = str(threads)
    if str(VENDOR) not in sys.path:
        sys.path.insert(0, str(VENDOR))


def load_variant(variant: str, model_dir: str | Path, threads: int) -> Any:
    """Load one runtime in one process; all paths must resolve locally."""
    prepare_offline(threads)
    model_dir = Path(model_dir).resolve()
    if not (model_dir / "rl_agent_config.json").is_file():
        raise FileNotFoundError(f"Local Laya checkpoint missing: {model_dir}")
    if variant == "torch_fp32":
        import torch
        torch.set_num_threads(threads)
        try:
            torch.set_num_interop_threads(1)
        except RuntimeError:
            pass
        from laya.agent import Agent
        return Agent(str(model_dir), device="cpu", compile=False)
    if variant in {"onnx_fp32", "onnx_int8"}:
        from laya.onnx_agent import ONNXAgent
        suffix = "laya.onnx" if variant == "onnx_fp32" else "laya.int8.onnx"
        path = model_dir / suffix
        if not path.is_file():
            raise FileNotFoundError(f"Local ONNX artifact missing: {path}")
        agent = ONNXAgent(str(model_dir), onnx_path=str(path))
        providers = agent.session.get_providers()
        if providers != ["CPUExecutionProvider"]:
            raise RuntimeError(f"Expected CPUExecutionProvider, got {providers}")
        return agent
    raise ValueError(f"Unknown variant: {variant}")


def decode_answer(answer: dict[str, Any]) -> tuple[str, float, dict[str, float]]:
    kind = answer.get("type")
    if kind == "choice":
        probs = {str(k): float(v) for k, v in answer["probabilities"].items()}
        prediction = str(answer["choice"])
        if prediction not in probs:
            raise ValueError("Choice label missing from probabilities")
        confidence = float(answer.get("answer_confidence", probs[prediction]))
        return prediction, confidence, probs
    if kind == "noul":
        yes = float(answer["noul"])
        if not 0 <= yes <= 1:
            raise ValueError("Invalid noul probability")
        prediction = "yes" if yes >= 0.5 else "no"
        confidence = float(answer.get("answer_confidence", max(yes, 1 - yes)))
        return prediction, confidence, {"no": 1 - yes, "yes": yes}
    raise ValueError(f"Unsupported Laya answer type: {kind!r}")


def predict_row(agent: Any, row: dict[str, Any], *, max_len: int | None = None, variant: str = "") -> dict[str, Any]:
    kwargs = {"max_len": max_len} if max_len else {}
    result = agent.system_one(row["state"], row["question"], **kwargs)
    answer = result["answers"]["decision"]
    prediction, confidence, probs = decode_answer(answer)
    return {
        "id": row["id"], "task": row["task"], "split": row["split"],
        "gold": row["gold"], "pred": prediction, "confidence": confidence,
        "probabilities": probs, "variant": variant,
        "input_tokens": result.get("usage", {}).get("input_tokens"),
    }


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def write_jsonl(path: str | Path, rows: list[dict[str, Any]]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
