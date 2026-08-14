"""Unit tests for subtitle generation, including karaoke timing."""
import re

from src.utils import subtitles as sub


def test_karaoke_durations_sum_to_line_duration():
    """Highlight timings must add up to the line's length, or the sweep drifts
    out of step with the singing."""
    text = sub._karaoke_text("Once upon a time there is a happy cow,", 3.0)
    total_cs = sum(int(cs) for cs in re.findall(r"\\kf(\d+)", text))
    assert total_cs == 300


def test_karaoke_weights_longer_words_for_longer(): 
    """A two-syllable word should hold longer than a one-syllable word."""
    text = sub._karaoke_text("a happy", 2.0)
    durations = [int(cs) for cs in re.findall(r"\\kf(\d+)", text)]
    assert durations[1] > durations[0]


def test_karaoke_handles_single_word_and_empty_line():
    assert r"\kf" in sub._karaoke_text("cow", 1.0)
    assert sub._karaoke_text("   ", 1.0) is not None


def test_escape_neutralises_ass_override_braces():
    """Braces in a lyric would otherwise be parsed as styling tags."""
    assert "{" not in sub._escape("a {weird} line")
    assert "}" not in sub._escape("a {weird} line")


def test_write_ass_subtitles_rejects_length_mismatch(tmp_path):
    try:
        sub.write_ass_subtitles(["one", "two"], [(0.0, 1.0)], tmp_path / "x.ass", 1920, 1080)
        assert False, "expected ValueError on lines/timeline mismatch"
    except ValueError:
        pass


def test_karaoke_can_be_disabled(tmp_path):
    path = sub.write_ass_subtitles(["hello there"], [(0.0, 1.0)], tmp_path / "p.ass",
                                   1920, 1080, karaoke=False)
    assert r"\kf" not in path.read_text(encoding="utf-8")
