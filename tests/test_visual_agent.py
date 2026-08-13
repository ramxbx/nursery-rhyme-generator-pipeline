"""Unit tests for visual_agent (GPT-16). The SD pipeline is stubbed - these
test deterministic logic, not live generation."""
from src.agents import visual_agent as va


class FakeTokenizer:
    """Whitespace tokenizer standing in for CLIP's - enough to exercise the
    truncation branch without loading the real pipeline."""

    def __call__(self, text, add_special_tokens=True):
        return {"input_ids": text.split()}

    def decode(self, ids):
        return " ".join(ids)


class FakePipe:
    tokenizer = FakeTokenizer()


def test_seed_for_character_is_deterministic():
    assert va.seed_for_character("Star") == va.seed_for_character("Star")
    assert va.seed_for_character("Star") == va.seed_for_character("  star  ")  # normalized


def test_seed_for_character_differs_across_names():
    assert va.seed_for_character("Star") != va.seed_for_character("Narrator")


def test_build_prompt_includes_style_anchor_and_description():
    prompt = va.build_prompt(FakePipe(), {"scene_description": "a sheep in a field"})
    assert prompt.startswith(va.STYLE_ANCHOR)
    assert "a sheep in a field" in prompt


def test_build_prompt_falls_back_when_description_missing():
    prompt = va.build_prompt(FakePipe(), {})
    assert va.STYLE_ANCHOR in prompt
    assert "picture-book" in prompt


def test_build_prompt_truncates_to_clip_budget():
    long_description = " ".join(f"word{i}" for i in range(200))
    prompt = va.build_prompt(FakePipe(), {"scene_description": long_description})
    assert len(prompt.split()) == va.CLIP_TOKEN_BUDGET


def test_build_prompt_keeps_style_anchor_when_truncating():
    long_description = " ".join(f"word{i}" for i in range(200))
    prompt = va.build_prompt(FakePipe(), {"scene_description": long_description})
    assert prompt.startswith(va.STYLE_ANCHOR)  # style anchor survives truncation
