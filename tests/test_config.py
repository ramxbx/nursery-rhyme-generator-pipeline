"""Unit tests for config.py (GPT-16)."""
import pytest

from src.config import ConfigError, load_config


def _write(path, content):
    path.write_text(content, encoding="utf-8")
    return path


def test_load_config_raises_on_missing_file(tmp_path):
    with pytest.raises(ConfigError, match="not found"):
        load_config(tmp_path)


def test_load_config_raises_clearly_on_invalid_yaml(tmp_path):
    _write(tmp_path / "pipeline.yaml", "paths: [unclosed")
    with pytest.raises(ConfigError, match="Invalid YAML"):
        load_config(tmp_path)


def test_load_config_raises_on_missing_required_keys(tmp_path):
    _write(tmp_path / "pipeline.yaml", "paths:\n  data_dir: data\n")
    with pytest.raises(ConfigError, match="missing required key"):
        load_config(tmp_path)


def test_load_config_loads_valid_project_config():
    # Exercises the real config/ directory shipped in the repo.
    config = load_config()
    assert config.llm["primary"]
    assert config.video["width"] == 1920
    assert "data_dir" in config.pipeline["paths"]
