from __future__ import annotations

import importlib.util
import json
import tempfile
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("download_assets", ROOT / "scripts" / "download_assets.py")
module = importlib.util.module_from_spec(spec)
assert spec and spec.loader
spec.loader.exec_module(module)


def test_verified_extract_rejects_escape_and_wrong_hash():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        archive = root / "asset.zip"
        with zipfile.ZipFile(archive, "w") as bundle:
            bundle.writestr("safe.txt", "ok")
        expected = [{"path": "safe.txt", "size_bytes": 2,
                     "sha256": module.sha256(_write(root / "expected.txt", "ok"))}]
        module.safe_extract(archive, root / "out", expected)
        assert (root / "out" / "safe.txt").read_text() == "ok"
        expected[0]["sha256"] = "0" * 64
        try:
            module.safe_extract(archive, root / "bad", expected)
        except ValueError as exc:
            assert "hash mismatch" in str(exc)
        else:
            raise AssertionError("wrong hash was accepted")
        with zipfile.ZipFile(archive, "w") as bundle:
            bundle.writestr("../escape.txt", "ok")
        try:
            module.safe_extract(archive, root / "out", [{"path": "../escape.txt", "size_bytes": 2,
                                                          "sha256": "0" * 64}])
        except ValueError as exc:
            assert "Unsafe" in str(exc)
        else:
            raise AssertionError("path traversal was accepted")


def _write(path: Path, text: str) -> Path:
    path.write_text(text, encoding="utf-8")
    return path
