"""Centralized YAML configuration loading and validation (GPT-6)."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_DIR = REPO_ROOT / "config"

REQUIRED_PIPELINE_KEYS = {"paths", "models", "video", "gpu"}
REQUIRED_PATHS_KEYS = {"data_dir", "scripts_dir", "images_dir", "audio_dir", "output_dir", "logs_dir"}
REQUIRED_LLM_KEYS = {"endpoint", "primary", "fallback", "context_length", "max_concurrent_requests"}


class ConfigError(Exception):
    """Raised when a config file is missing, malformed, or fails validation."""


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise ConfigError(f"Config file not found: {path}")
    try:
        with path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
    except yaml.YAMLError as e:
        raise ConfigError(f"Invalid YAML in {path}: {e}") from e
    if not isinstance(data, dict):
        raise ConfigError(f"Expected a YAML mapping at the top level of {path}, got {type(data).__name__}")
    return data


def _require_keys(data: dict, required: set[str], context: str) -> None:
    missing = required - data.keys()
    if missing:
        raise ConfigError(f"{context} is missing required key(s): {sorted(missing)}")


@dataclass
class PipelineConfig:
    pipeline: dict[str, Any]
    sd: dict[str, Any]
    tts: dict[str, Any]
    root: Path = field(default=REPO_ROOT)

    @property
    def paths(self) -> dict[str, Path]:
        return {k: (self.root / v) for k, v in self.pipeline["paths"].items()}

    @property
    def llm(self) -> dict[str, Any]:
        return self.pipeline["models"]["llm"]

    @property
    def video(self) -> dict[str, Any]:
        return self.pipeline["video"]

    @property
    def gpu(self) -> dict[str, Any]:
        return self.pipeline["gpu"]


def load_config(config_dir: Path | None = None) -> PipelineConfig:
    """Load and validate pipeline.yaml, sd_config.yml, and tts_config.yml."""
    config_dir = config_dir or DEFAULT_CONFIG_DIR

    pipeline = _load_yaml(config_dir / "pipeline.yaml")
    _require_keys(pipeline, REQUIRED_PIPELINE_KEYS, "pipeline.yaml")
    _require_keys(pipeline["paths"], REQUIRED_PATHS_KEYS, "pipeline.yaml:paths")
    _require_keys(pipeline["models"].get("llm", {}), REQUIRED_LLM_KEYS, "pipeline.yaml:models.llm")

    sd = _load_yaml(config_dir / "sd_config.yml")
    tts = _load_yaml(config_dir / "tts_config.yml")

    return PipelineConfig(pipeline=pipeline, sd=sd, tts=tts)
