"""Script generation agent (GPT-10).

Codex (this module) owns parsing, schema validation, retries, and duration
estimation deterministically. The CPU-hosted local model (LFM2-1.2B primary,
Gemma-3-1B-it-QAT fallback, per GPT-18) is only asked to annotate ONE
already-isolated source line at a time with speaker + stage_direction -
never to split/structure the rhyme itself. That decomposition failure was
the exact reason gemma-3-1b was disqualified from whole-rhyme structuring
in the GPT-18 benchmark.
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.config import PipelineConfig, load_config
from src.utils.file_manager import ensure_dirs, safe_write_json
from src.utils.llm_client import LLMError, call_with_fallback
from src.utils.logger import get_logger, log_with_fields
from src.utils.prompt_builder import build_dialogue_prompt

logger = get_logger("script_agent")

REQUIRED_ANNOTATION_KEYS = {"speaker", "stage_direction"}

SECONDS_PER_SYLLABLE = 0.35
MIN_LINE_DURATION_S = 1.5
COMMA_PAUSE_S = 0.2
LINE_END_PAUSE_S = 0.5


def split_lines(rhyme_text: str) -> list[str]:
    """Deterministic line splitting - no model involved."""
    return [line.strip() for line in rhyme_text.splitlines() if line.strip()]


def count_syllables(word: str) -> int:
    word = re.sub(r"[^a-z]", "", word.lower())
    if not word:
        return 0
    vowel_groups = re.findall(r"[aeiouy]+", word)
    count = len(vowel_groups)
    if word.endswith("e") and count > 1:
        count -= 1
    return max(count, 1)


def estimate_duration(line: str) -> float:
    """Syllable-aware duration estimate, plain code, no model involved."""
    words = re.findall(r"[A-Za-z']+", line)
    syllables = sum(count_syllables(w) for w in words)
    pauses = line.count(",") * COMMA_PAUSE_S + LINE_END_PAUSE_S
    duration = syllables * SECONDS_PER_SYLLABLE + pauses
    return round(max(duration, MIN_LINE_DURATION_S), 2)


def _validate_annotation(annotation: Any) -> dict:
    if not isinstance(annotation, dict) or not REQUIRED_ANNOTATION_KEYS.issubset(annotation.keys()):
        raise ValueError(f"Annotation missing required keys {REQUIRED_ANNOTATION_KEYS}: {annotation!r}")
    if not isinstance(annotation["speaker"], str) or not isinstance(annotation["stage_direction"], str):
        raise ValueError(f"Annotation fields must be strings: {annotation!r}")
    return annotation


def annotate_line(line_text: str, line_index: int, line_total: int, cast_so_far: list[str],
                   llm_config: dict) -> dict:
    prompt = build_dialogue_prompt(line_text, line_index, line_total, cast_so_far)
    try:
        result = call_with_fallback(
            system_prompt=prompt,
            user_prompt=line_text,
            llm_config=llm_config,
            parse_json=True,
            max_tokens=200,
        )
        return _validate_annotation(result)
    except (LLMError, ValueError) as e:
        log_with_fields(logger, 40, "annotation failed, using deterministic fallback",
                         line_index=line_index, error=str(e))
        # Deterministic fallback so the pipeline never hard-fails on a single bad line.
        speaker = cast_so_far[0] if cast_so_far else "Narrator"
        return {"speaker": speaker, "stage_direction": "A gentle, child-friendly scene."}


def extract_cast(cast_so_far: list[str], speaker: str) -> list[str]:
    if speaker and speaker.lower() not in {c.lower() for c in cast_so_far}:
        return cast_so_far + [speaker]
    return cast_so_far


def generate_script(rhyme_text: str, config: PipelineConfig | None = None) -> dict:
    config = config or load_config()
    lines = split_lines(rhyme_text)
    if not lines:
        raise ValueError("Source rhyme has no non-empty lines")

    scenes = []
    cast: list[str] = []
    for i, line in enumerate(lines, start=1):
        annotation = annotate_line(line, i, len(lines), cast, config.llm)
        cast = extract_cast(cast, annotation["speaker"])
        scenes.append({
            "line": line,
            "speaker": annotation["speaker"],
            "stage_direction": annotation["stage_direction"],
            "duration_s": estimate_duration(line),
        })
        log_with_fields(logger, 20, "line processed", line_index=i, total=len(lines), speaker=annotation["speaker"])

    assert len(scenes) == len(lines), "All source lines must be represented"
    return {"cast": cast, "scenes": scenes}


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a scene script from a nursery rhyme.")
    parser.add_argument("input", type=Path, help="Path to a text file containing the rhyme")
    parser.add_argument("--out", type=Path, default=None, help="Output JSON path (default: data/scripts/<stem>.json)")
    args = parser.parse_args()

    config = load_config()
    ensure_dirs(config.paths)

    rhyme_text = args.input.read_text(encoding="utf-8")
    script = generate_script(rhyme_text, config)

    out_path = args.out or (config.paths["scripts_dir"] / f"{args.input.stem}.json")
    safe_write_json(out_path, script)
    print(f"Wrote {out_path} ({len(script['scenes'])} scenes, cast: {script['cast']})")


if __name__ == "__main__":
    main()
