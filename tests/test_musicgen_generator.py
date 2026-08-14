"""Unit tests for MusicGen duration handling. The model itself is never loaded -
these cover the looping arithmetic that keeps long videos under MusicGen's hard
position-embedding ceiling."""
import numpy as np

from src.utils import musicgen_generator as mg


def test_loop_fills_a_duration_longer_than_the_clip():
    """The bug this guards: a 54.9s video asked MusicGen for 2745 tokens against
    a 2048 limit and crashed the whole animate stage."""
    sr = 32000
    clip = np.random.default_rng(0).uniform(-0.5, 0.5, sr * 10).astype(np.float32)
    out = mg._loop_to_length(clip, sr, 35.0)
    assert abs(len(out) / sr - 35.0) < 0.01


def test_loop_truncates_when_the_clip_is_already_long_enough():
    sr = 32000
    clip = np.ones(sr * 20, dtype=np.float32)
    out = mg._loop_to_length(clip, sr, 5.0)
    assert abs(len(out) / sr - 5.0) < 0.01


def test_loop_seam_is_crossfaded_not_abrupt():
    """A hard cut between repeats clicks audibly under quiet narration."""
    sr = 32000
    clip = np.ones(sr * 4, dtype=np.float32) * 0.5
    out = mg._loop_to_length(clip, sr, 10.0)
    jumps = np.abs(np.diff(out))
    assert jumps.max() < 0.05, "loop joint should not step abruptly"


def test_generation_request_is_capped_below_the_position_embedding_limit():
    """MusicGen's decoder maxes out at 2048 positions (40.96s); anything at or
    above that raises IndexError rather than degrading."""
    assert mg.MAX_GENERATION_S * mg.TOKENS_PER_SECOND < 2048


def test_short_clip_with_tiny_crossfade_still_reaches_target():
    sr = 100  # crossfade window collapses to near nothing at this rate
    clip = np.ones(8, dtype=np.float32)
    out = mg._loop_to_length(clip, sr, 1.0)
    assert len(out) == 100
