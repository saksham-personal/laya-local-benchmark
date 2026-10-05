"""Freeze the four ONNX export wheels into a checksum-pinned GitHub asset."""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PREFIXES = ("onnx-", "onnxscript-", "onnx_ir-", "ml_dtypes-")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wheel-dir", type=Path, required=True)
    parser.add_argument("--archive", type=Path, required=True)
    args = parser.parse_args()
    wheels = sorted(path for path in args.wheel_dir.glob("*.whl")
                    if path.name.startswith(PREFIXES))
    if len(wheels) != 4 or len({next(p for p in PREFIXES if w.name.startswith(p)) for w in wheels}) != 4:
        parser.error("Expected exactly one wheel each for onnx, onnxscript, onnx_ir, ml_dtypes")
    args.archive.parent.mkdir(parents=True, exist_ok=True)
    files = []
    with zipfile.ZipFile(args.archive, "w", compression=zipfile.ZIP_STORED, allowZip64=True) as bundle:
        for path in wheels:
            info = zipfile.ZipInfo(path.name, date_time=(2026, 10, 5, 0, 0, 0))
            info.compress_type = zipfile.ZIP_STORED
            with path.open("rb") as source, bundle.open(info, "w") as target:
                shutil.copyfileobj(source, target, length=8 * 1024 * 1024)
            files.append({"path": path.name, "size_bytes": path.stat().st_size,
                          "sha256": sha256(path)})
    manifest = {
        "id": "laya-export-wheels-windows-x64", "kind": "runtime",
        "asset": args.archive.name,
        "github": {"repository": "saksham-personal/laya-local-benchmark", "release": "assets-v1"},
        "archive_sha256": sha256(args.archive), "size_bytes": args.archive.stat().st_size,
        "install_subdir": "export-wheels", "files": files,
        "platform": "Windows x64 CPython 3.12",
        "purpose": "ONNX FP32 export and dynamic INT8 quantization; install after offline base runtime",
    }
    (ROOT / "configs/export-wheels.asset.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    requirements = []
    for file in files:
        name = file["path"].split("-")[0].replace("_", "-")
        version = file["path"].split("-")[1]
        requirements.append(f"{name}=={version} --hash=sha256:{file['sha256']}")
    (ROOT / "configs/export-requirements.txt").write_text(
        "\n".join(requirements) + "\n", encoding="utf-8")
    print(json.dumps({"archive": str(args.archive), "sha256": manifest["archive_sha256"],
                      "size_bytes": manifest["size_bytes"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
