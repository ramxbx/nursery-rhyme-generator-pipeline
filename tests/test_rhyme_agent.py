"""Unit tests for rhyme generation. The model is stubbed - these cover the
deterministic validation that decides whether a generated poem is usable."""
import random

import pytest

from src.agents import rhyme_agent as ra


def _poem(lines):
    return "\n".join(lines)


# Deliberately not one of the shipped fallback poems - reusing one of those
# here would trip the originality check and mask what each test is measuring.
GOOD = [
    "A bright red kite with ribbon tail,", "Went up to catch the morning sail,",
    "It climbed above the apple tree,", "And waved at all the world to see.",
    "The sky was wide, the sun was bright,", "The clouds were soft, the wind was light.",
    "It spun around and round and round,", "And never once came near the ground.",
    "A sparrow flew up nice and slow,", "To watch the paper ribbons glow.",
    "They played together all the day,", "And chased the little clouds away.",
    "And when the sun began to set,", "The kite came down without regret.",
    "It folded up its paper wing,", "And slept beside a ball of string.",
]

SEED = {"title": "Five Little Ducks", "subject": "a duck", "setting": "a pond", "mood": "gentle"}


def test_rhyme_key_matches_across_different_spellings():
    assert ra._rhyme_key("star") == ra._rhyme_key("are")
    assert ra._rhyme_key("grew") == ra._rhyme_key("blew")
    assert ra._rhyme_key("bright") != ra._rhyme_key("pond")


def test_a_word_does_not_rhyme_with_itself():
    """Ending both lines on the same word is the cheapest way to fake a rhyme
    scheme, and the most obvious when sung."""
    assert not ra.lines_rhyme("we saw a cat,", "we found a cat,")


def test_every_shipped_fallback_poem_passes_validation():
    """The fallback poems are what a run degrades to, so they must clear the
    same bar a generated poem does - otherwise the safety net is the weak
    point."""
    for path in sorted(ra.FALLBACK_DIR.glob("*.txt")):
        # Originality is skipped: these poems ARE the corpus that check
        # compares against, so each would match itself.
        lines, reason = ra.validate_poem(path.read_text(encoding="utf-8"), SEED, check_originality=False)
        assert lines, f"{path.name} failed validation: {reason}"


def test_ten_fallback_poems_are_available():
    assert len(list(ra.FALLBACK_DIR.glob("*.txt"))) >= 10


def test_wrong_line_count_is_rejected():
    lines, reason = ra.validate_poem(_poem(GOOD[:12]), SEED)
    assert not lines and "12" in reason


def test_a_leading_title_line_is_stripped_rather_than_failing():
    """Models habitually announce the poem before writing it; that is a
    formatting slip, not a bad poem."""
    lines, reason = ra.validate_poem(_poem(["The Little Duck"] + GOOD), SEED)
    assert lines == GOOD, reason


def test_prose_with_line_breaks_is_rejected():
    flat = [f"this is line number {i} of plain writing," for i in range(ra.RHYME_LINES)]
    lines, reason = ra.validate_poem(_poem(flat), SEED)
    assert not lines
    assert "couplets rhyme" in reason or "repeats a line" in reason


def test_a_repeated_line_is_rejected():
    lines, reason = ra.validate_poem(_poem(GOOD[:15] + [GOOD[0]]), SEED)
    assert not lines and "repeats" in reason


def test_reciting_a_shipped_poem_is_rejected_as_unoriginal():
    """The whole point is a NEW poem in the style of a famous one - a model
    that recites something back has not done the task."""
    recited = (ra.FALLBACK_DIR / "little_duck.txt").read_text(encoding="utf-8")
    lines, reason = ra.validate_poem(recited, SEED)
    assert not lines and "reproduces" in reason


def test_generation_falls_back_to_a_written_poem_when_the_model_fails(monkeypatch):
    """An unloaded or failing model must degrade to a usable poem, never block
    the run."""
    def boom(**kwargs):
        raise ra.LLMError("model not loaded")

    monkeypatch.setattr(ra, "call_with_fallback", boom)

    class FakeConfig:
        llm = {"endpoint": "http://localhost:1234/v1"}

    name, text, generated = ra.generate_rhyme(FakeConfig(), rng=random.Random(0))
    assert not generated
    assert len(text.strip().splitlines()) == ra.RHYME_LINES


def test_generation_accepts_a_valid_poem_from_the_model(monkeypatch):
    class FakeResult:
        text = _poem(GOOD)

    monkeypatch.setattr(ra, "call_with_fallback", lambda **kwargs: FakeResult())

    class FakeConfig:
        llm = {"endpoint": "http://localhost:1234/v1"}

    name, text, generated = ra.generate_rhyme(FakeConfig(), rng=random.Random(0))
    assert generated
    assert text.strip().splitlines() == GOOD


@pytest.mark.parametrize("bad_line", ["No.", "a " * 20])
def test_lines_far_off_the_metre_are_rejected(bad_line):
    lines, reason = ra.validate_poem(_poem([bad_line] + GOOD[1:]), SEED)
    assert not lines


def test_a_shorter_poem_is_validated_against_its_own_length():
    """End-to-end runs are gated on poem length - every line costs a Bark take
    and an image - so a 6-line poem must not be judged as a failed 16-line one."""
    six = GOOD[:6]
    lines, reason = ra.validate_poem(_poem(six), SEED, line_total=6)
    assert lines == six, reason


def test_rhyme_requirement_scales_with_length():
    """A 6-line poem has three couplets; demanding five would reject every
    valid short poem."""
    four = GOOD[:4]
    lines, reason = ra.validate_poem(_poem(four), SEED, line_total=4)
    assert lines == four, reason


def test_default_length_is_unchanged():
    lines, reason = ra.validate_poem(_poem(GOOD), SEED)
    assert lines == GOOD, reason
