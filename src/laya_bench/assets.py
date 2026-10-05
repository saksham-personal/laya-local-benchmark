"""Verify and install checksum-pinned Laya benchmark release assets."""
from __future__ import annotations

import hashlib
import json
import math
import shutil
import urllib.request
import zipfile
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Mapping

CHUNK_SIZE = 8 * 1024 * 1024


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(CHUNK_SIZE), b""):
            digest.update(block)
    return digest.hexdigest()


def _safe_relative(value: str, what: str) -> PurePosixPath:
    if not isinstance(value, str) or not value or "\\" in value or ":" in value:
        raise ValueError(f"Unsafe {what}: {value!r}")
    path = PurePosixPath(value)
    if path.is_absolute() or any(part in ("", ".", "..") for part in path.parts) or path.as_posix() != value:
        raise ValueError(f"Unsafe {what}: {value!r}")
    return path


def _under(root: Path, relative: str, what: str) -> Path:
    safe = _safe_relative(relative, what)
    root = root.resolve()
    result = (root.joinpath(*safe.parts)).resolve()
    try:
        result.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"{what} escapes work directory: {relative!r}") from exc
    return result


def safe_extract(archive_path: str | Path, target_dir: str | Path,
                 files: Iterable[Mapping[str, Any]]) -> Path:
    """Extract a ZIP only when member names, exact inventory, sizes and hashes match."""
    target = Path(target_dir).resolve()
    expected: dict[str, Mapping[str, Any]] = {}
    for record in files:
        name = record.get("path")
        _safe_relative(name, "manifest file path")
        if name in expected:
            raise ValueError(f"Duplicate manifest path: {name}")
        size = record.get("size_bytes")
        digest = record.get("sha256")
        if not isinstance(size, int) or isinstance(size, bool) or size < 0:
            raise ValueError(f"Invalid size for manifest file: {name}")
        if not isinstance(digest, str) or len(digest) != 64 or any(c not in "0123456789abcdefABCDEF" for c in digest):
            raise ValueError(f"Invalid SHA-256 for manifest file: {name}")
        expected[name] = record

    with zipfile.ZipFile(archive_path) as archive:
        infos = [entry for entry in archive.infolist() if not entry.is_dir()]
        names = [entry.filename for entry in infos]
        if len(names) != len(set(names)):
            raise ValueError("Archive contains duplicate file paths")
        if set(names) != set(expected):
            raise ValueError("Archive file list does not match signed manifest")
        for info in infos:
            name = info.filename
            _safe_relative(name, "archive member")
            mode = (info.external_attr >> 16) & 0o170000
            if mode == 0o120000:
                raise ValueError(f"Unsafe symbolic link in archive: {name}")
            destination = _under(target, name, "archive member")
            record = expected[name]
            if info.file_size != record["size_bytes"]:
                raise ValueError(f"Archive size does not match manifest: {name}")
            destination.parent.mkdir(parents=True, exist_ok=True)
            temporary = destination.with_name(destination.name + ".partial")
            try:
                with archive.open(info) as source, temporary.open("wb") as sink:
                    shutil.copyfileobj(source, sink, length=CHUNK_SIZE)
                if temporary.stat().st_size != record["size_bytes"] or sha256(temporary).lower() != record["sha256"].lower():
                    raise ValueError(f"Extracted file hash mismatch: {name}")
                temporary.replace(destination)
            finally:
                temporary.unlink(missing_ok=True)
    return target


def verify_installed(root: str | Path, files: Iterable[Mapping[str, Any]]) -> bool:
    base = Path(root).resolve()
    for record in files:
        path = _under(base, record["path"], "manifest file path")
        if not path.is_file() or path.stat().st_size != record["size_bytes"] or sha256(path).lower() != record["sha256"].lower():
            return False
    return True


def install_asset(item: Mapping[str, Any], work_dir: str | Path, offline: bool = False,
                  timeout: float = 300) -> Path:
    """Fetch (unless offline) and install one manifest entry below work_dir."""
    base = Path(work_dir).resolve()
    base.mkdir(parents=True, exist_ok=True)
    asset_name = item["asset"]
    _safe_relative(asset_name, "release asset filename")
    if len(PurePosixPath(asset_name).parts) != 1:
        raise ValueError("Release asset must be a filename, not a path")
    archive = _under(base / "downloads", asset_name, "release asset")
    archive.parent.mkdir(parents=True, exist_ok=True)
    expected_hash = item["archive_sha256"]
    if not isinstance(expected_hash, str) or len(expected_hash) != 64 or any(c not in "0123456789abcdefABCDEF" for c in expected_hash):
        raise ValueError(f"Invalid archive SHA-256 in manifest: {item.get('id')}")
    valid_archive = archive.is_file() and sha256(archive).lower() == expected_hash.lower()
    if not valid_archive:
        if offline:
            if archive.exists():
                raise ValueError(f"Local archive checksum mismatch: {archive}")
            raise FileNotFoundError(f"Verified local archive missing: {archive}")
        github = item["github"]
        repository, release = github["repository"], github["release"]
        if not repository or "/" not in repository or not release:
            raise ValueError("Invalid GitHub repository or release in manifest")
        url = f"https://github.com/{repository}/releases/download/{release}/{asset_name}"
        temporary = archive.with_name(archive.name + ".partial")
        try:
            with urllib.request.urlopen(url, timeout=timeout) as response, temporary.open("wb") as sink:
                shutil.copyfileobj(response, sink, length=CHUNK_SIZE)
            if sha256(temporary).lower() != expected_hash.lower():
                raise ValueError(f"GitHub archive hash mismatch: {item.get('id')}")
            temporary.replace(archive)
        finally:
            temporary.unlink(missing_ok=True)

    install_subdir = item["install_subdir"]
    root = _under(base, install_subdir, "install_subdir")
    if root == base:
        raise ValueError("install_subdir must name a child of work_dir")
    files = item["files"]
    if not isinstance(files, list) or not files:
        raise ValueError("Asset manifest must contain at least one file")
    if not verify_installed(root, files):
        safe_extract(archive, root, files)
    return root


def load_manifest(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as stream:
        data = json.load(stream)
    assets = data.get("assets") if isinstance(data, dict) else None
    if not isinstance(assets, list):
        raise ValueError("Manifest must contain an assets array")
    return assets


def select_assets(items: Iterable[dict[str, Any]], asset_ids: Iterable[str] | None = None,
                  kind: str | None = None) -> list[dict[str, Any]]:
    requested = set(asset_ids or ())
    items = list(items)
    selected = [item for item in items if (not requested or item.get("id") in requested)
                and (kind is None or item.get("kind") == kind)]
    if requested and requested != {item.get("id") for item in selected}:
        raise ValueError("Unknown or filtered asset id")
    return selected
