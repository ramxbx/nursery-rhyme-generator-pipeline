"""Lightweight 'sing-song' melodic pitch-contour shaping for narration.

Not true singing-voice synthesis - that needs a note-aligned score and a
dedicated model like DiffSinger (a much bigger lift, flagged as a stretch
goal in GPT-12). This applies a pitch contour across a line's natural
word-segments, giving spoken narration a melodic, sung-like cadence
without any new model dependency.

GPT-23 fix: the contour used to be an arbitrary wandering pattern with no
relationship to any real tune, which is why it didn't sound "in tune" -
melodically varied, but not actually music. Now defaults to the real
"Twinkle Twinkle Little Star" melody (semitone offsets from the tonic),
which is also the tune "Baa Baa Black Sheep" and the ABC Song share - the
most common melody in the classic-nursery-rhyme repertoire. A running
offset can be threaded across calls (see audio_agent.py) so the melody
continues progressing through a whole poem instead of restarting at note
one on every line.

GPT-30 fix: segments used to be pitch-shifted via librosa's phase-vocoder
(librosa.effects.pitch_shift), which introduced audible metallic/robotic
artifacts on speech, especially at this contour's larger shifts (up to 9
semitones). Switched to ffmpeg's rubberband filter (Rubber Band Library,
formant-preserving, designed for vocals) - already compiled into this
project's ffmpeg build, no new dependency.
"""
from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf

from src.utils.ffmpeg_helper import rubberband_pitch_shift

# "Twinkle Twinkle Little Star" / "Baa Baa Black Sheep" / ABC Song melody,
# semitone offsets from the tonic: C C G G A A G | F F E E D D C
DEFAULT_CONTOUR_SEMITONES = [0, 0, 7, 7, 9, 9, 7, 5, 5, 4, 4, 2, 2, 0]
CROSSFADE_S = 0.03
MIN_SEGMENT_S = 0.05


def _split_into_segments(audio: np.ndarray, n_segments: int) -> list[np.ndarray]:
    n_segments = max(1, n_segments)
    bounds = np.linspace(0, len(audio), n_segments + 1).astype(int)
    return [audio[bounds[i]:bounds[i + 1]] for i in range(n_segments)]


def _crossfade_concat(segments: list[np.ndarray], sr: int) -> np.ndarray:
    if len(segments) == 1:
        return segments[0]
    fade_len = max(1, int(sr * CROSSFADE_S))
    out = segments[0].copy()
    for seg in segments[1:]:
        if len(out) < fade_len or len(seg) < fade_len:
            out = np.concatenate([out, seg])
            continue
        fade_out = np.linspace(1.0, 0.0, fade_len, dtype=np.float32)
        fade_in = np.linspace(0.0, 1.0, fade_len, dtype=np.float32)
        overlap = out[-fade_len:] * fade_out + seg[:fade_len] * fade_in
        out = np.concatenate([out[:-fade_len], overlap, seg[fade_len:]])
    return out


def _pitch_shift_segment(seg: np.ndarray, sr: int, n_steps: int) -> np.ndarray:
    pitch_ratio = 2 ** (n_steps / 12)
    with tempfile.TemporaryDirectory() as td:
        in_path = Path(td) / "in.wav"
        out_path = Path(td) / "out.wav"
        sf.write(in_path, seg, sr, subtype="FLOAT")
        rubberband_pitch_shift(in_path, out_path, pitch_ratio)
        shifted, _ = sf.read(out_path, dtype="float32")
    return shifted


def apply_singsong_contour(audio: np.ndarray, sr: int, n_words: int,
                            contour: list[int] | None = None, start_offset: int = 0) -> tuple[np.ndarray, int]:
    """Pitch-shift a narration clip's word-segments along a melodic
    contour, starting at start_offset within the contour (so a full poem
    can progress continuously through the melody across lines instead of
    restarting at note one each time - see audio_agent.py). Operates on
    the natural (unpadded) speech only - call this before adding silence
    padding. Returns (shifted_audio, next_offset)."""
    if len(audio) < sr * MIN_SEGMENT_S:
        return audio, start_offset

    contour = contour or DEFAULT_CONTOUR_SEMITONES
    n_segments = max(1, min(n_words, 12))
    segments = _split_into_segments(audio, n_segments)

    shifted = []
    for i, seg in enumerate(segments):
        if len(seg) < sr * MIN_SEGMENT_S:
            shifted.append(seg)
            continue
        n_steps = contour[(start_offset + i) % len(contour)]
        if n_steps == 0:
            shifted.append(seg)
            continue
        shifted.append(_pitch_shift_segment(seg, sr, n_steps).astype(np.float32))

    return _crossfade_concat(shifted, sr), start_offset + n_segments
