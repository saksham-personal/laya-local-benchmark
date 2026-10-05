"""Record machine and runtime facts without treating this PC as the target VDI."""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
from pathlib import Path


def package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def inspect() -> dict:
    result = {
        "os": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor() or None,
        "logical_processors": os.cpu_count(),
        "python": sys.version,
        "packages": {name: package_version(name) for name in
                     ("torch", "onnxruntime", "transformers", "numpy", "psutil", "onnx", "onnxscript")},
        "environment": {key: os.environ.get(key) for key in
                        ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "ORT_NUM_THREADS",
                         "HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE")},
        "cpu_flags": {"avx2": None, "avx512": None, "vnni": None},
    }
    try:
        import psutil
        memory = psutil.virtual_memory()
        result["memory"] = {"total_bytes": memory.total, "available_bytes": memory.available}
        result["physical_cores"] = psutil.cpu_count(logical=False)
    except Exception as exc:
        result["memory_error"] = str(exc)
    try:
        import torch
        result["torch"] = {"cuda_available": torch.cuda.is_available(),
                           "cpu_capability": torch.backends.cpu.get_cpu_capability()}
        capability = result["torch"]["cpu_capability"].upper()
        result["cpu_flags"]["avx2"] = "AVX2" in capability or "AVX512" in capability
        result["cpu_flags"]["avx512"] = "AVX512" in capability
        result["cpu_flags"]["vnni"] = "VNNI" in capability
    except Exception as exc:
        result["torch_error"] = str(exc)
    try:
        import onnxruntime as ort
        result["onnx_providers"] = ort.get_available_providers()
    except Exception as exc:
        result["onnx_error"] = str(exc)
    if sys.platform == "win32":
        try:
            command = ["powershell", "-NoProfile", "-Command",
                       "Get-CimInstance Win32_Processor | Select-Object -First 1 -ExpandProperty Name"]
            completed = subprocess.run(command, capture_output=True, text=True, timeout=10)
            if completed.returncode == 0:
                result["processor"] = completed.stdout.strip() or result["processor"]
        except Exception:
            pass
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result = inspect()
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
