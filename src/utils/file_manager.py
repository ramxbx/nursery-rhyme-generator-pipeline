"""Standardized file and directory handling for pipeline stages (GPT-8)."""
from __future__ import annotations

import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def ensure_dirs(paths: dict[str, Path]) -> dict[str, Path]:
    """Create every directory in a {name: Path} mapping; returns the same mapping."""
    for p in paths.values():
        ensure_dir(p)
    return paths


def _atomic_write(path: Path, write_fn) -> Path:
    """Write via a temp file in the same directory, then atomically replace."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=str(path.parent), prefix=f".{path.name}.", suffix=".tmp")
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "wb") as f:
            write_fn(f)
        os.replace(tmp_path, path)
    except BaseException:
        tmp_path.unlink(missing_ok=True)
        raise
    return path


def safe_write_text(path: Path, content: str, encoding: str = "utf-8") -> Path:
    return _atomic_write(path, lambda f: f.write(content.encode(encoding)))


def safe_write_bytes(path: Path, content: bytes) -> Path:
    return _atomic_write(path, lambda f: f.write(content))


def safe_write_json(path: Path, data: Any, indent: int = 2) -> Path:
    return safe_write_text(path, json.dumps(data, indent=indent, ensure_ascii=False))


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def scene_path(base_dir: Path, scene_index: int, suffix: str) -> Path:
    """Consistent per-scene file naming, e.g. scene_path(images_dir, 3, '.png') -> scene_003.png"""
    return base_dir / f"scene_{scene_index:03d}{suffix}"


def clean_dir(path: Path, keep_dir: bool = True) -> None:
    """Remove all contents of a directory (used by `make clean`)."""
    if not path.exists():
        return
    for child in path.iterdir():
        if child.is_dir():
            shutil.rmtree(child)
        else:
            child.unlink()
    if not keep_dir:
        path.rmdir()
