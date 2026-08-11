"""Unit tests for script_agent (GPT-16). LLM calls are mocked - these test
the deterministic logic (splitting, timing, validation, fallback), not
live model output."""
from src.agents import script_agent as sa
from src.utils.llm_client import LLMError, LLMResult


def test_split_lines_strips_empty_and_whitespace():
    text = "Line one,\n\n  Line two.  \n\n\nLine three."
    assert sa.split_lines(text) == ["Line one,", "Line two.", "Line three."]


def test_count_syllables_basic_words():
    assert sa.count_syllables("star") == 1
    assert sa.count_syllables("wonder") == 2
    assert sa.count_syllables("diamond") >= 2
    assert sa.count_syllables("") == 0


def test_estimate_duration_respects_minimum_and_scales_with_length():
    short = sa.estimate_duration("Hi.")
    long = sa.estimate_duration("Twinkle, twinkle, little star, how I wonder what you are.")
    assert short >= sa.MIN_LINE_DURATION_S
    assert long > short


def test_split_last_word_extracts_word_and_punctuation():
    body, word, punct = sa._split_last_word("Twinkle, twinkle, little star,")
    assert word == "star"
    assert punct == ","

    body, word, punct = sa._split_last_word("How I wonder what you are.")
    assert word == "are"
    assert punct == "."


def test_split_last_word_handles_no_trailing_word():
    body, word, punct = sa._split_last_word("...")
    assert word == ""


def test_candidate_is_valid_accepts_correct_ending():
    assert sa._candidate_is_valid("Glimmer, glimmer, shining star.", "star", "Twinkle, twinkle, little star,")


def test_candidate_is_valid_rejects_wrong_ending():
    assert not sa._candidate_is_valid("Glimmer, glimmer, shining sun.", "star", "Twinkle, twinkle, little star,")


def test_candidate_is_valid_rejects_duplicate_target_word():
    assert not sa._candidate_is_valid("Bright star, oh shining star.", "star", "Twinkle, twinkle, little star,")


def test_candidate_is_valid_rejects_run_on_length():
    long_candidate = "Glimmer glimmer shining bright sparkling twinkling wonderful amazing star."
    assert not sa._candidate_is_valid(long_candidate, "star", "Twinkle, twinkle, little star,")


def test_validate_annotation_requires_speaker_and_stage_direction():
    ok = sa._validate_annotation({"speaker": "Star", "stage_direction": "shines"})
    assert ok["speaker"] == "Star"
    try:
        sa._validate_annotation({"speaker": "Star"})
        assert False, "expected ValueError for missing stage_direction"
    except ValueError:
        pass


def test_extract_cast_deduplicates_case_insensitively():
    cast = sa.extract_cast([], "Star")
    cast = sa.extract_cast(cast, "star")
    cast = sa.extract_cast(cast, "Narrator")
    assert cast == ["Star", "Narrator"]


def test_annotate_line_falls_back_deterministically_on_llm_error(monkeypatch):
    def raise_llm_error(**kwargs):
        raise LLMError("simulated failure")

    monkeypatch.setattr(sa, "call_with_fallback", raise_llm_error)
    result = sa.annotate_line("Twinkle, twinkle, little star,", 1, 4, ["Star"], {})
    assert result["speaker"] == "Star"
    assert "scene_description" in result
    assert "mood" in result


def test_rewrite_line_creatively_falls_back_to_original_on_repeated_failure(monkeypatch):
    def bad_result(**kwargs):
        return LLMResult(text="completely wrong ending", model_used="fake", latency_s=0.0, attempts=1)

    monkeypatch.setattr(sa, "call_with_fallback", bad_result)
    line = "Twinkle, twinkle, little star,"
    result = sa.rewrite_line_creatively(line, {})
    assert result == line


def test_generate_script_covers_all_source_lines(monkeypatch):
    def fake_annotate(line_text, line_index, line_total, cast_so_far, llm_config):
        return {"speaker": "Narrator", "stage_direction": "does something",
                "scene_description": "a place", "mood": "calm"}

    monkeypatch.setattr(sa, "rewrite_line_creatively", lambda line, cfg: line)
    monkeypatch.setattr(sa, "annotate_line", fake_annotate)

    class FakeConfig:
        pipeline = {"script": {"creative_rewrite": True, "elaborate_scene_description": False}}
        llm = {}

    rhyme = "Line one,\nLine two.\nLine three."
    result = sa.generate_script(rhyme, FakeConfig())
    assert len(result["scenes"]) == 3
    assert [s["line"] for s in result["scenes"]] == ["Line one,", "Line two.", "Line three."]
    assert all("duration_s" in s and s["duration_s"] > 0 for s in result["scenes"])
