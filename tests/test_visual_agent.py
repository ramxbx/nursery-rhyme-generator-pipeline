"""Unit tests for visual_agent (GPT-16). LLM calls and the SD pipeline are
mocked/synthetic - these test deterministic logic, not live generation."""
import numpy as np
from PIL import Image

from src.agents import visual_agent as va
from src.utils.llm_client import LLMError, LLMResult


def test_seed_for_character_is_deterministic():
    assert va.seed_for_character("Star") == va.seed_for_character("Star")
    assert va.seed_for_character("Star") == va.seed_for_character("  star  ")  # normalized


def test_seed_for_character_differs_across_names():
    assert va.seed_for_character("Star") != va.seed_for_character("Narrator")


def test_is_low_quality_flags_near_blank_image():
    blank = Image.fromarray(np.full((64, 64, 3), 128, dtype=np.uint8))
    assert va._is_low_quality(blank)


def test_is_low_quality_accepts_varied_image():
    rng = np.random.default_rng(0)
    varied = Image.fromarray(rng.integers(0, 255, (64, 64, 3), dtype=np.uint8))
    assert not va._is_low_quality(varied)


def test_draft_subject_description_falls_back_on_llm_error(monkeypatch):
    monkeypatch.setattr(va, "call_with_fallback", lambda **kwargs: (_ for _ in ()).throw(LLMError("boom")))
    result = va.draft_subject_description("Star", {})
    assert "star" in result.lower()


def test_draft_image_prompt_falls_back_on_llm_error(monkeypatch):
    monkeypatch.setattr(va, "call_with_fallback", lambda **kwargs: (_ for _ in ()).throw(LLMError("boom")))
    result = va.draft_image_prompt("Star", "a friendly star", "shining brightly", {},
                                    scene_description="a night sky", mood="calm")
    assert "a friendly star" in result
    assert "shining brightly" in result


def test_draft_image_prompt_uses_llm_text_when_available(monkeypatch):
    monkeypatch.setattr(va, "call_with_fallback",
                         lambda **kwargs: LLMResult(text="a cool custom prompt", model_used="fake",
                                                     latency_s=0.0, attempts=1))
    result = va.draft_image_prompt("Star", "a friendly star", "shining brightly", {})
    assert result == "a cool custom prompt"
