"""Reusable prompt template loading/rendering for pipeline agents (GPT-7)."""
from __future__ import annotations

from pathlib import Path
from string import Template

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_TEMPLATE_DIR = REPO_ROOT / "templates"


class PromptError(Exception):
    """Raised when a template is missing a required variable."""


def load_template(name: str, template_dir: Path | None = None) -> str:
    template_dir = template_dir or DEFAULT_TEMPLATE_DIR
    path = template_dir / name
    if not path.exists():
        raise PromptError(f"Prompt template not found: {path}")
    return path.read_text(encoding="utf-8")


def render(template_str: str, **kwargs) -> str:
    """Render a $-style template, raising PromptError on missing variables
    instead of silently leaving placeholders (which would corrupt the prompt)."""
    try:
        return Template(template_str).substitute(**kwargs)
    except KeyError as e:
        raise PromptError(f"Missing template variable: {e}") from e


def build_dialogue_prompt(line_text: str, line_index: int, line_total: int, cast_so_far: list[str]) -> str:
    """Bounded per-line annotation prompt (GPT-10 design: lines are pre-split
    in code; the model only labels one already-isolated line at a time)."""
    template = load_template("dialogue_prompt_template.txt")
    return render(
        template,
        line_text=line_text,
        line_index=str(line_index),
        line_total=str(line_total),
        cast_so_far=", ".join(cast_so_far) if cast_so_far else "(none yet)",
    )


def build_elaborate_story_prompt(all_lines: list[str], speakers: list[str]) -> str:
    """Whole-poem-aware scene description prompt: asks for every scene in a
    single call instead of one line at a time. A small local model chaining
    off just a short per-scene anchor still let appearance/character facts
    drift scene-to-scene (a color contradiction re-appeared even with
    chaining) - having the model hold the whole story in one generation
    pass, rather than handing off a lossy summary between separate calls,
    is what actually keeps facts consistent (GPT-27 follow-up)."""
    template = load_template("elaborate_scene_template.txt")
    # Deliberately NOT showing the poem as verse-formatted line breaks
    # (e.g. joined with "\n") anywhere in this prompt - smaller local
    # models (lfm2-1.2b-bench) reliably drifted into continuing in rhyme
    # themselves when shown the literal rhyming line breaks, regardless of
    # an explicit "plain prose" instruction (GPT-27 follow-up). This
    # semicolon-joined form carries the same content without the visual
    # shape of a poem.
    full_poem = "; ".join(all_lines)
    line_roles = "\n".join(
        f'Line {i}: "{line}" (speaker: {speaker})'
        for i, (line, speaker) in enumerate(zip(all_lines, speakers), start=1)
    )
    return render(
        template,
        full_poem=full_poem,
        line_roles=line_roles,
        line_total=str(len(all_lines)),
    )


def build_rewrite_prompt(target_word: str) -> str:
    template = load_template("rewrite_prompt_template.txt")
    return render(template, target_word=target_word)
