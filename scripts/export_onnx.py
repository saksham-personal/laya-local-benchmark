"""Run the pinned official Laya ONNX exporter on a local checkpoint."""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["USE_TF"] = "0"
sys.path.insert(0, str(ROOT / "vendor"))
from export_onnx_upstream import export_to_onnx, int8_output_path, quantize_model  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--mode", choices=["fp32", "int8"], required=True)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    model_dir = args.model_dir.resolve()
    if not (model_dir / "model.safetensors").is_file():
        parser.error(f"Local PyTorch checkpoint missing: {model_dir}")
    fp32 = model_dir / "laya.onnx"
    int8 = Path(int8_output_path(str(fp32)))
    if args.mode == "fp32":
        if fp32.exists() and not args.force:
            parser.error(f"{fp32} exists; pass --force to replace")
        export_to_onnx(str(model_dir), str(fp32))
        print(fp32)
    else:
        if not fp32.is_file():
            parser.error("Export and validate ONNX FP32 before INT8")
        if int8.exists() and not args.force:
            parser.error(f"{int8} exists; pass --force to replace")
        quantize_model(str(fp32), str(int8), per_channel=False)
        print(int8)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
