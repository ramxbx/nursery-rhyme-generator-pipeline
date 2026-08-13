"""Unit tests for prompt_builder.py (GPT-16)."""
import pytest

from src.utils.prompt_builder import PromptError, build_dialogue_prompt, render


def test_render_substitutes_variables():
    result = render("Hello $name, you are $age.", name="Star", age="1")
    assert result == "Hello Star, you are 1."


def test_render_raises_promt_error_on_missing_variable():
    with pytest.raises(PromptError):
        render("Hello $name.")


def test_build_dialogue_prompt_includes_line_and_cast():
    prompt = build_dialogue_prompt("Twinkle, twinkle,", 1, 4, ["Star", "Narrator"])
    assert "Twinkle, twinkle," in prompt
    assert "Star, Narrator" in prompt
    assert "1" in prompt and "4" in prompt


def test_build_dialogue_prompt_handles_empty_cast():
    prompt = build_dialogue_prompt("Line one", 1, 1, [])
    assert "(none yet)" in prompt


