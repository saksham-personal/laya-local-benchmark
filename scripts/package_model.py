"""Package a pinned local Laya checkpoint as a checksum-verifiable Release asset."""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HF_REPO = "convaiinnovations/laya"
HF_REVISION = "7b928d828b7b0e022f929d9bd2e44165aa270148"
GITHUB_REPO = "saksham-personal/laya-local-benchmark"
RELEASE = "assets-v1"
FILES = ("rl_agent_config.json", "model.safetensors",
         "encoder/config.json", "tokenizer/tokenizer.json",
         "tokenizer/tokenizer_config.json")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def package(model_dir: Path, archive: Path, checkpoint: str) -> dict:
    for name in FILES:
        if not (model_dir / name).is_file():
            raise FileNotFoundError(f"Missing pinned source file: {name}")
    archive.parent.mkdir(parents=True, exist_ok=True)
    records = []
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6,
                         allowZip64=True) as bundle:
        for name in sorted(FILES):
            source = model_dir / name
            info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            with source.open("rb") as input_stream, bundle.open(info, "w", force_zip64=True) as output_stream:
                shutil.copyfileobj(input_stream, output_stream, length=8 * 1024 * 1024)
            records.append({"path": name, "size_bytes": source.stat().st_size,
                            "sha256": sha256(source),
                            "hf_path": f"{checkpoint}/{name}" if checkpoint != "english" else name})
    return {
        "id": f"laya-{checkpoint}-pytorch",
        "kind": "model",
        "asset": archive.name,
        "github": {"repository": GITHUB_REPO, "release": RELEASE},
        "upstream": {"repository": HF_REPO, "revision": HF_REVISION,
                     "subfolder": "" if checkpoint == "english" else checkpoint,
                     "license": "Apache-2.0", "runtime": "PyTorch CPU"},
        "archive_sha256": sha256(archive),
        "size_bytes": archive.stat().st_size,
        "files": records,
        "install_subdir": f"models/{checkpoint}",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--checkpoint", choices=["english", "typed-decisions"], required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    item = package(args.model_dir, args.archive, args.checkpoint)
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(item, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"asset": str(args.archive), "archive_sha256": item["archive_sha256"],
                      "size_bytes": item["size_bytes"], "manifest": str(args.manifest)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
