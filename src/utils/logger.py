"""Structured, timestamped logging for pipeline stages (GPT-9)."""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_LOG_FILE = REPO_ROOT / "logs" / "pipeline.log"

_CONFIGURED = False


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "timestamp": self.formatTime(record, "%Y-%m-%dT%H:%M:%S%z"),
            "level": record.levelname,
            "stage": record.name,
            "message": record.getMessage(),
        }
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        extra = getattr(record, "extra_fields", None)
        if extra:
            payload.update(extra)
        return json.dumps(payload, ensure_ascii=False)


def _configure_root(log_file: Path) -> None:
    global _CONFIGURED
    if _CONFIGURED:
        return

    log_file.parent.mkdir(parents=True, exist_ok=True)
    root = logging.getLogger("pipeline")
    root.setLevel(logging.INFO)

    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setFormatter(JsonFormatter())
    root.addHandler(file_handler)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"))
    root.addHandler(console_handler)

    root.propagate = False
    _CONFIGURED = True


def get_logger(stage: str, log_file: Path | None = None) -> logging.Logger:
    """Return a logger for a pipeline stage, e.g. get_logger('script_agent')."""
    _configure_root(log_file or DEFAULT_LOG_FILE)
    return logging.getLogger(f"pipeline.{stage}")


def log_with_fields(logger: logging.Logger, level: int, message: str, **fields) -> None:
    """Log a message with extra structured fields captured by JsonFormatter."""
    logger.log(level, message, extra={"extra_fields": fields})
