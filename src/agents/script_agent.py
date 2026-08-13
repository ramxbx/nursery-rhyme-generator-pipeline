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
from src.utils.prompt_builder import build_dialogue_prompt, build_elaborate_story_prompt, build_rewrite_prompt

logger = get_logger("script_agent")

REQUIRED_ANNOTATION_KEYS = {"speaker", "stage_direction"}
LAST_WORD_RE = re.compile(r"^(.*\b)([A-Za-z']+)([.,!?;:]*)\s*$")
REWRITE_MAX_RETRIES = 2

# CPU-only model for the scene descriptions - all scenes drafted together in ONE
# call (see draft_elaborate_scene_descriptions) so the model holds the whole story,
# and the one subject's appearance it establishes, in view the entire time.
#
# These are written as comma-separated visual descriptors ("white sheep, thick
# fluffy wool, green meadow, blue sky"), NOT narrative prose - they are fed
# straight to Stable Diffusion by visual_agent.py with no intermediate rewriting.
# Diffusion models key off short visual noun phrases; narrative prose with
# dialogue, character names and quoted speech produced markedly worse images
# (abstract blobs, human figures, no subject at all) when passed through
# verbatim. Writing them in SD's native format here removes the need for a
# separate LLM compression call in the visual stage entirely.
#
# Tried lfm2-1.2b-bench (1.2B) for the speed (~40s vs gemma-4-e2b's ~80s/scene):
# it followed FORMAT rules once the prompt was reframed away from "poem" language,
# but content reliability regressed - reproducing the exact black/white wool color
# contradiction this whole GPT-27 effort exists to prevent, plus depicting the
# Narrator as a physical character and giving the sheep dog behavior ("barks").
# Small models follow format instructions far more reliably than factual-consistency
# ones - a pattern seen repeatedly in this project. gemma-4-e2b-bench (4.6B) holds
# both, so it stays.
ELABORATE_MODEL = "gemma-4-e2b-bench"
# Token/timeout budgets scale with poem length. The descriptor lists themselves are
# short (20-35 words/scene, ~40 tokens), but gemma-4-e2b-bench is a reasoning model:
# it burns a large, variable hidden "reasoning_content" pass BEFORE the API's
# "content" field starts filling in, and llm_client only reads "content". Budgeting
# for the visible output alone (tried 90/scene + 150) starved that reasoning pass and
# silently truncated the response mid-list, so parsing rejected it and every scene
# fell back to its short annotation. The overhead here is deliberately generous -
# it is reasoning headroom, not output length.
ELABORATE_TOKENS_PER_SCENE = 120
ELABORATE_TOKEN_OVERHEAD = 600
ELABORATE_SECONDS_PER_SCENE = 60
ELABORATE_TIMEOUT_OVERHEAD_S = 120

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
                "scene_description": "soft colorful picture-book setting, gentle pastel colors, simple shapes",
                "mood": "gentle and playful"}


ELABORATE_RETRIES = 2
# Above this, treat two scenes as near-duplicates of each other. Raised from
# 0.55 when scene descriptions became comma-separated descriptor lists rather
# than prose: every scene now deliberately repeats the same subject descriptors
# ("white sheep, thick fluffy wool, ...") to hold appearance consistent, so
# baseline similarity between two perfectly good scenes is far higher than it
# was for prose. The check still catches genuinely identical scenes.
ELABORATE_SIMILARITY_THRESHOLD = 0.85

SCENE_HEADER_RE = re.compile(r"(?:^|\n)\s*SCENE\s+(\d+)\s*:\s*", re.IGNORECASE)


def _parse_scene_blocks(text: str, expected_count: int) -> list[str] | None:
    """Parses the "SCENE N:\n<paragraph>" format instead of asking the model
    to emit long paragraphs inside JSON string fields - small local models
    reliably mangle JSON escaping over multi-sentence values, whereas a
    plain delimited format only needs regex splitting. Returns None (never
    partial results) if the count is wrong, a scene is empty/missing, or a
    scene contains an internal line break (verse/formatting drift) - an
    all-or-nothing response is treated as invalid so callers fall back
    cleanly rather than mixing rich and short descriptions."""
    matches = list(SCENE_HEADER_RE.finditer(text))
    if len(matches) != expected_count:
        return None
    indices = [int(m.group(1)) for m in matches]
    if indices != list(range(1, expected_count + 1)):
        return None
    blocks = []
    for i, m in enumerate(matches):
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        block = text[start:end].strip()
        if not block or "\n" in block:
            return None
        blocks.append(block)
    return blocks


def _has_near_duplicate(blocks: list[str]) -> bool:
    for i in range(len(blocks)):
        for j in range(i + 1, len(blocks)):
            if difflib.SequenceMatcher(None, blocks[i], blocks[j]).ratio() > ELABORATE_SIMILARITY_THRESHOLD:
                return True  # model repeated itself instead of writing a fresh scene
    return False


def draft_elaborate_scene_descriptions(lines: list[str], speakers: list[str], base_llm_config: dict,
                                        fallback_descriptions: list[str]) -> list[str]:
    """Rich, whole-poem-aware scene descriptions for every line, drafted in
    ONE call so the model holds the whole story - and the single subject's
    appearance/setting it establishes - in view the entire time, rather
    than handing off a lossy per-scene summary between separate calls
    (which still let color/character facts drift even with chaining;
    GPT-27 follow-up). All-or-nothing: falls back to the short per-line
    descriptions (already computed by annotate_line) for every scene if it
    can't produce a fully valid response - never blocks the pipeline."""
    line_total = len(lines)
    system_prompt = build_elaborate_story_prompt(lines, speakers)
    max_tokens = ELABORATE_TOKENS_PER_SCENE * line_total + ELABORATE_TOKEN_OVERHEAD
    timeout_s = ELABORATE_SECONDS_PER_SCENE * line_total + ELABORATE_TIMEOUT_OVERHEAD_S
    elaborate_config = {
        "endpoint": base_llm_config["endpoint"],
        "primary": ELABORATE_MODEL,
        "fallback": ELABORATE_MODEL,
        "request_timeout_s": timeout_s,
        "max_retries": 1,
        "excluded": base_llm_config.get("excluded", []),
    }
    for attempt in range(1, ELABORATE_RETRIES + 1):
        try:
            result = call_with_fallback(system_prompt=system_prompt,
                                         user_prompt="Write all the scene descriptions now.",
                                         llm_config=elaborate_config, parse_json=False, max_tokens=max_tokens)
        except LLMError as e:
            log_with_fields(logger, 30, "elaborate scene descriptions call failed", attempt=attempt, error=str(e))
            continue
        blocks = _parse_scene_blocks(result.text.strip(), line_total)
        if blocks and not _has_near_duplicate(blocks):
            return blocks
        log_with_fields(logger, 30, "elaborate scene descriptions failed validation, retrying", attempt=attempt)

    log_with_fields(logger, 30, "elaborate scene descriptions exhausted retries, keeping short fallbacks")
    return fallback_descriptions


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

    annotations = []
    cast: list[str] = []
    for i, line in enumerate(lines, start=1):
        annotation = annotate_line(line, i, len(lines), cast, config.llm)
        cast = extract_cast(cast, annotation["speaker"])
        annotations.append(annotation)
        log_with_fields(logger, 20, "line processed", line_index=i, total=len(lines), speaker=annotation["speaker"])

    fallback_descriptions = [
        a.get("scene_description") or "soft colorful picture-book setting, gentle pastel colors, simple shapes"
        for a in annotations
    ]

    if elaborate_enabled:
        speakers = [a["speaker"] for a in annotations]
        scene_descriptions = draft_elaborate_scene_descriptions(lines, speakers, config.llm, fallback_descriptions)
        log_with_fields(logger, 20, "elaborate scene descriptions drafted", count=len(scene_descriptions))
    else:
        scene_descriptions = fallback_descriptions

    scenes = [
        {
            "line": line,
            "speaker": annotation["speaker"],
            "stage_direction": annotation["stage_direction"],
            "scene_description": scene_description,
            "mood": annotation.get("mood") or "gentle and playful",
            "duration_s": estimate_duration(line),
        }
        for line, annotation, scene_description in zip(lines, annotations, scene_descriptions)
    ]

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
