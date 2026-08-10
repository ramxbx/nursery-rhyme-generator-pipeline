"""Shared CPU-hosted local-model client with primary/fallback retry policy.

Used by script_agent (GPT-10), and reusable as-is by visual_agent's prompt
drafting (GPT-11) and upload_agent's metadata drafting (GPT-14) — all three
talk to the same LM Studio endpoint under the same registry from GPT-18.
"""
from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any

from src.utils.logger import get_logger, log_with_fields

logger = get_logger("llm_client")


class LLMError(Exception):
    """Raised when both primary and fallback models fail to produce a usable response."""


@dataclass
class LLMResult:
    text: str
    model_used: str
    latency_s: float
    attempts: int


def _strip_code_fence(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.startswith("json"):
            text = text[4:]
    return text.strip()


def _call_once(endpoint: str, model: str, system_prompt: str, user_prompt: str,
                timeout_s: int, max_tokens: int = 400, temperature: float = 0.2) -> tuple[str, float]:
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    req = urllib.request.Request(
        f"{endpoint}/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    start = time.perf_counter()
    with urllib.request.urlopen(req, timeout=timeout_s) as resp:
        body = json.loads(resp.read().decode("utf-8"))
    elapsed = time.perf_counter() - start
    return body["choices"][0]["message"]["content"], elapsed


def call_with_fallback(system_prompt: str, user_prompt: str, llm_config: dict,
                        parse_json: bool = True, max_tokens: int = 400) -> Any:
    """Try the primary model, retry it, then fall back to the fallback model.

    Raises LLMError if every attempt (primary retries + fallback retries) fails.
    Never selects any model in llm_config['excluded'].
    """
    endpoint = llm_config["endpoint"]
    timeout_s = llm_config.get("request_timeout_s", 60)
    max_retries = llm_config.get("max_retries", 2)
    excluded = set(llm_config.get("excluded", []))

    candidates = [llm_config["primary"], llm_config["fallback"]]
    candidates = [c for c in candidates if c not in excluded]

    attempts = 0
    last_error: Exception | None = None

    for model in candidates:
        for attempt in range(1, max_retries + 1):
            attempts += 1
            try:
                raw, elapsed = _call_once(endpoint, model, system_prompt, user_prompt, timeout_s, max_tokens)
            except (urllib.error.URLError, TimeoutError, OSError) as e:
                last_error = e
                log_with_fields(logger, 30, "llm call failed (transport)", model=model, attempt=attempt, error=str(e))
                continue

            if not parse_json:
                log_with_fields(logger, 20, "llm call ok", model=model, attempt=attempt, latency_s=round(elapsed, 2))
                return LLMResult(text=raw, model_used=model, latency_s=elapsed, attempts=attempts)

            try:
                parsed = json.loads(_strip_code_fence(raw))
            except json.JSONDecodeError as e:
                last_error = e
                log_with_fields(logger, 30, "llm call failed (invalid json)", model=model, attempt=attempt, error=str(e))
                continue

            log_with_fields(logger, 20, "llm call ok", model=model, attempt=attempt, latency_s=round(elapsed, 2))
            return parsed

    raise LLMError(f"All models failed after {attempts} attempt(s): {last_error}")
