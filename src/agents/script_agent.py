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
import difflib
import re
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.config import PipelineConfig, load_config
from src.utils.file_manager import ensure_dirs, safe_write_json
from src.utils.llm_client import LLMError, call_with_fallback
from src.utils.logger import get_logger, log_with_fields
from src.utils.prompt_builder import build_dialogue_prompt, build_elaborate_scene_prompt, build_rewrite_prompt

logger = get_logger("script_agent")

REQUIRED_ANNOTATION_KEYS = {"speaker", "stage_direction"}
LAST_WORD_RE = re.compile(r"^(.*\b)([A-Za-z']+)([.,!?;:]*)\s*$")
REWRITE_MAX_RETRIES = 2

# CPU-only model for elaborate, whole-poem-aware scene descriptions.
# Benchmarked bonsai-27b (too slow, 8+min timeout), gemma-4-e2b (works but
# burns a hidden "thinking" token tax), qwen2.5-7b-instruct (clean, ~130s/
# scene), lfm2-1.2b-bench (also clean, ~40s/scene - 3x faster, no quality
# loss once given a generous token budget and the consistency rules in
# elaborate_scene_template.txt). gemma-3-1b-bench, by contrast, drifted
# off-poem and re-introduced the color-contradiction bug even with the
# same prompt - not used here despite being similarly fast.
ELABORATE_MODEL = "lfm2-1.2b-bench"
ELABORATE_MAX_TOKENS = 500
ELABORATE_TIMEOUT_S = 90

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


def _split_last_word(line: str) -> tuple[str, str, str]:
    """Split a line into (body, last_word, trailing_punct). Empty last_word
    means the line has no recognizable trailing word (leave it untouched)."""
    m = LAST_WORD_RE.match(line.strip())
    if not m or not m.group(2):
        return line, "", ""
    return m.group(1), m.group(2), m.group(3)


def _candidate_is_valid(candidate: str, last_word: str, original_line: str) -> bool:
    first_line = candidate.strip().splitlines()[0] if candidate.strip() else ""
    cleaned = re.sub(r"[.,!?;:]+$", "", first_line.strip())
    if not cleaned.lower().endswith(last_word.lower()):
        return False
    # Catch the "tacks the target word on twice" failure mode (e.g. "...night, star, star.")
    word_occurrences = len(re.findall(rf"\b{re.escape(last_word)}\b", cleaned, flags=re.IGNORECASE))
    if word_occurrences > 1:
        return False
    # Catch run-on babbling: reject candidates far longer than the source line.
    if len(cleaned.split()) > 1.6 * len(original_line.split()):
        return False
    return True


def rewrite_line_creatively(line: str, llm_config: dict) -> str:
    """Rewrite a line's wording while keeping its exact original end-word,
    which guarantees the rhyme scheme is preserved by construction (see
    GPT-20: full free-form rewriting loses either rhyme or theme on these
    small models; fixing the end-word turns it back into a bounded,
    single-line task). Falls back to the original line on any failure -
    never breaks the pipeline over a creative-writing miss.
    """
    body, last_word, punct = _split_last_word(line)
    if not last_word:
        return line

    system_prompt = build_rewrite_prompt(last_word + punct)
    for attempt in range(1, REWRITE_MAX_RETRIES + 1):
        try:
            result = call_with_fallback(system_prompt=system_prompt, user_prompt=line,
                                         llm_config=llm_config, parse_json=False, max_tokens=60)
        except LLMError as e:
            log_with_fields(logger, 30, "rewrite call failed", attempt=attempt, error=str(e))
            break

        candidate = result.text.strip().splitlines()[0].strip() if result.text.strip() else ""
        if candidate and _candidate_is_valid(candidate, last_word, line):
            log_with_fields(logger, 20, "line rewritten", attempt=attempt, original=line, rewritten=candidate)
            return candidate
        log_with_fields(logger, 30, "rewrite failed validation, retrying", attempt=attempt, candidate=candidate)

    log_with_fields(logger, 30, "rewrite exhausted retries, keeping original line", line=line)
    return line


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
            max_tokens=280,
        )
        return _validate_annotation(result)
    except (LLMError, ValueError) as e:
        log_with_fields(logger, 40, "annotation failed, using deterministic fallback",
                         line_index=line_index, error=str(e))
        # Deterministic fallback so the pipeline never hard-fails on a single bad line.
        speaker = cast_so_far[0] if cast_so_far else "Narrator"
        return {"speaker": speaker, "stage_direction": "A gentle, child-friendly scene.",
                "scene_description": "A soft, colorful children's picture-book setting.",
                "mood": "gentle and playful"}


ELABORATE_RETRIES = 2
ELABORATE_SIMILARITY_THRESHOLD = 0.55  # above this, treat as a near-duplicate of the previous scene


def _elaborate_description_is_valid(text: str, previous_description: str) -> bool:
    if not text:
        return False
    if "\n" in text:
        return False  # verse/line-break formatting leaked in - not plain prose
    if previous_description:
        similarity = difflib.SequenceMatcher(None, text, previous_description).ratio()
        if similarity > ELABORATE_SIMILARITY_THRESHOLD:
            return False  # near-duplicate of the previous scene, not a fresh description
    return True


def draft_elaborate_scene_description(all_lines: list[str], line_index: int, speaker: str,
                                       base_llm_config: dict, fallback_description: str,
                                       previous_description: str = "") -> str:
    """Rich, whole-poem-aware scene description via a small CPU-only model,
    chained to the previous scene's description so appearance/setting stay
    visually consistent scene-to-scene instead of being reinvented each
    time (GPT-22 follow-up - this is what actually fixed the black/white
    wool contradiction the user caught). Validates against two failure
    modes discovered chaining full descriptions caused: drifting into
    verse instead of prose, and near-duplicating the previous scene
    instead of writing something new. Falls back to the short per-line
    description (already computed by annotate_line) if it can't produce a
    valid one - never blocks the pipeline."""
    system_prompt = build_elaborate_scene_prompt(all_lines, line_index, speaker, previous_description)
    elaborate_config = {
        "endpoint": base_llm_config["endpoint"],
        "primary": ELABORATE_MODEL,
        "fallback": ELABORATE_MODEL,
        "request_timeout_s": ELABORATE_TIMEOUT_S,
        "max_retries": 1,
        "excluded": base_llm_config.get("excluded", []),
    }
    for attempt in range(1, ELABORATE_RETRIES + 1):
        try:
            result = call_with_fallback(system_prompt=system_prompt, user_prompt="Write the scene description now.",
                                         llm_config=elaborate_config, parse_json=False, max_tokens=ELABORATE_MAX_TOKENS)
        except LLMError as e:
            log_with_fields(logger, 30, "elaborate scene description call failed", line_index=line_index,
                             attempt=attempt, error=str(e))
            continue
        text = result.text.strip()
        if _elaborate_description_is_valid(text, previous_description):
            return text
        log_with_fields(logger, 30, "elaborate scene description failed validation, retrying",
                         line_index=line_index, attempt=attempt)

    log_with_fields(logger, 30, "elaborate scene description exhausted retries, keeping short fallback",
                     line_index=line_index)
    return fallback_description


def extract_cast(cast_so_far: list[str], speaker: str) -> list[str]:
    if speaker and speaker.lower() not in {c.lower() for c in cast_so_far}:
        return cast_so_far + [speaker]
    return cast_so_far


def generate_script(rhyme_text: str, config: PipelineConfig | None = None) -> dict:
    config = config or load_config()
    lines = split_lines(rhyme_text)
    if not lines:
        raise ValueError("Source rhyme has no non-empty lines")

    creative_rewrite = config.pipeline.get("script", {}).get("creative_rewrite", True)
    if creative_rewrite:
        lines = [rewrite_line_creatively(line, config.llm) for line in lines]

    elaborate_enabled = config.pipeline.get("script", {}).get("elaborate_scene_description", True)

    scenes = []
    cast: list[str] = []
    previous_description = ""
    for i, line in enumerate(lines, start=1):
        annotation = annotate_line(line, i, len(lines), cast, config.llm)
        cast = extract_cast(cast, annotation["speaker"])
        scene_description = annotation.get("scene_description") or "A soft, colorful children's picture-book setting."

        if elaborate_enabled:
            scene_description = draft_elaborate_scene_description(
                lines, i, annotation["speaker"], config.llm, scene_description, previous_description)
            previous_description = scene_description
            log_with_fields(logger, 20, "elaborate scene description drafted", line_index=i)

        scenes.append({
            "line": line,
            "speaker": annotation["speaker"],
            "stage_direction": annotation["stage_direction"],
            "scene_description": scene_description,
            "mood": annotation.get("mood") or "gentle and playful",
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
