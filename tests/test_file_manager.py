"""Unit tests for file_manager.py (GPT-16)."""
from src.utils.file_manager import ensure_dir, read_json, safe_write_json, scene_path


def test_ensure_dir_creates_nested_directories(tmp_path):
    target = tmp_path / "a" / "b" / "c"
    ensure_dir(target)
    assert target.is_dir()


def test_safe_write_json_then_read_json_roundtrips(tmp_path):
    path = tmp_path / "out.json"
    data = {"cast": ["Star"], "scenes": [1, 2, 3]}
    safe_write_json(path, data)
    assert read_json(path) == data


def test_safe_write_json_does_not_leave_temp_files_behind(tmp_path):
    path = tmp_path / "out.json"
    safe_write_json(path, {"a": 1})
    leftovers = [p for p in tmp_path.iterdir() if p != path]
    assert leftovers == []


def test_scene_path_naming_convention(tmp_path):
    assert scene_path(tmp_path, 3, ".png").name == "scene_003.png"
    assert scene_path(tmp_path, 42, ".wav").name == "scene_042.wav"
