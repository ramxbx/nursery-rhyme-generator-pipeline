"""Bark singing-voice synthesis backend.

Unlike Piper (which speaks, and is then pitch-shifted into a tune by
singing.py), Bark generates sung vocals directly from text when the lyrics are
wrapped in music-note markers. That makes it the only local option in this
project that actually sings rather than post-processing speech.

Three properties of Bark shape everything below and are worth knowing before
changing any of it:

* It is slow. Roughly 17x realtime on this CPU - ~4 minutes of compute for a
  ~15 second line. The audio stage goes from seconds to many minutes.
* It picks its own melody. There is no way to specify a tune, so the output
  will not be "Baa Baa Black Sheep" - it is a plausible children's-song melody
  Bark invented. This is a deliberate, user-accepted trade for real singing.
* It is non-deterministic and unreliable. Sampling is required, so the same
  input gives different output each run, including sometimes not singing at
  all, rambling past the lyrics, or emitting near-silence. Callers must be
  prepared to fall back.

Bark also ignores requested durations entirely, so scene timing has to follow
the generated audio rather than the script's estimate - see audio_agent.
"""
from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("HF_HOME", str(Path(__file__).resolve().parent.parent.parent / "models" / "hf_cache"))

import numpy as np

from src.utils.logger import get_logger, log_with_fields

logger = get_logger("bark_tts")

BARK_MODEL = "suno/bark-small"
# One fixed speaker preset across every line, so the "singer" stays the same
# voice through the whole video rather than changing character line to line.
DEFAULT_VOICE_PRESET = "v2/en_speaker_9"
# Bark emits at 24kHz regardless of what the rest of the pipeline uses.
BARK_SAMPLE_RATE = 24000
# Below this peak amplitude the take is treated as a failed generation (Bark
# occasionally returns near-silence) rather than usable audio.
MIN_PEAK_AMPLITUDE = 0.02
# Bark commonly pads a line with several seconds of trailing near-silence or
# breath noise; anything quieter than this fraction of the peak is trimmed.
TRIM_THRESHOLD = 0.02
# Bark has no duration control and will sometimes carry on well past the lyric
# - one run produced 12s for a line the script estimated at ~4s, while an
# identical earlier run produced 2.3s. Scene length follows audio length, so an
# over-long take leaves one image parked on screen while the rest of the poem
# flicks past. Takes longer than this multiple of the script's (syllable-based)
# estimate are rejected and regenerated; because generation is non-deterministic
# a retry usually lands in range.
MAX_DURATION_RATIO = 2.0
# Rejecting is only worth so much: each attempt is minutes of compute, so after
# this many the line falls back to Piper, which is fast and length-predictable.
MAX_TAKES = 3


def build_bark() -> tuple:
    """Load Bark once per process - the model load is far too slow to repeat
    per line."""
    import torch
    from transformers import AutoProcessor, BarkModel

    processor = AutoProcessor.from_pretrained(BARK_MODEL)
    model = BarkModel.from_pretrained(BARK_MODEL, torch_dtype=torch.float32)
    model.eval()
    log_with_fields(logger, 20, "bark loaded", model=BARK_MODEL)
    return model, processor


def _trim_silence(audio: np.ndarray) -> np.ndarray:
    """Drop leading/trailing near-silence. Bark pads generously, and that
    padding would otherwise stretch the scene it belongs to."""
    peak = float(np.abs(audio).max())
    if peak <= 0:
        return audio
    loud = np.where(np.abs(audio) > peak * TRIM_THRESHOLD)[0]
    if len(loud) == 0:
        return audio
    return audio[loud[0]:loud[-1] + 1]


# Every Bark generation is independent, so each line lands in whatever key it
# happens to pick - measured at 3.2 semitones of spread across four lines of one
# poem (B, A#, A#, G), which reads as four different singers rather than one
# song. Each line is therefore transposed onto a common tonic. ~220Hz (A3) sits
# in a comfortable register for a children's song and close to where this
# speaker preset already sings, keeping the corrections small.
TARGET_TONIC_HZ = 220.0
# Cap the correction: past a few semitones Rubber Band starts to colour the
# voice, and a wildly off-key take is better left slightly off than mangled.
MAX_TRANSPOSE_SEMITONES = 5.0
# Below this, leave the line alone - the shift would be inaudible and only
# costs quality.
MIN_TRANSPOSE_SEMITONES = 0.3


def estimate_f0(audio: np.ndarray, sr: int) -> float:
    """Median fundamental frequency of the voiced parts of a clip, via
    autocorrelation. Used as the line's perceived key centre."""
    frame, hop = 2048, 512
    f0s = []
    for i in range(0, max(0, len(audio) - frame), hop):
        seg = audio[i:i + frame].astype(np.float64)
        seg -= seg.mean()
        if np.abs(seg).max() < 0.02:  # unvoiced/silent frame
            continue
        corr = np.correlate(seg, seg, mode="full")[len(seg) - 1:]
        rising = np.where(np.diff(corr) > 0)[0]
        if len(rising) == 0:
            continue
        peak = int(np.argmax(corr[rising[0]:]) + rising[0])
        if peak == 0:
            continue
        f0 = sr / peak
        if 70 < f0 < 500:  # plausible singing range
            f0s.append(f0)
    return float(np.median(f0s)) if f0s else 0.0


def normalize_pitch(audio: np.ndarray, sr: int, target_hz: float = TARGET_TONIC_HZ) -> np.ndarray:
    """Transpose a line so its key centre sits at target_hz, so independently
    generated lines share a key instead of wandering between them."""
    import tempfile

    import soundfile as sf

    from src.utils.ffmpeg_helper import rubberband_pitch_shift

    f0 = estimate_f0(audio, sr)
    if f0 <= 0:
        return audio
    semitones = 12 * np.log2(target_hz / f0)
    if abs(semitones) < MIN_TRANSPOSE_SEMITONES:
        return audio
    semitones = float(np.clip(semitones, -MAX_TRANSPOSE_SEMITONES, MAX_TRANSPOSE_SEMITONES))

    with tempfile.TemporaryDirectory() as td:
        in_path, out_path = Path(td) / "in.wav", Path(td) / "out.wav"
        sf.write(in_path, audio, sr, subtype="FLOAT")
        rubberband_pitch_shift(in_path, out_path, pitch_ratio=2 ** (semitones / 12))
        shifted, _ = sf.read(out_path, dtype="float32")
    log_with_fields(logger, 20, "bark line transposed to common key",
                     from_hz=round(f0, 1), semitones=round(semitones, 2))
    return shifted.astype(np.float32)


def _generate_take(bark, text: str, voice_preset: str) -> np.ndarray | None:
    """One Bark generation. Returns None if the take is structurally unusable
    (empty, or near-silence, both of which Bark produces occasionally)."""
    import torch

    model, processor = bark
    # The music-note markers are what prompt Bark to sing rather than speak.
    prompt = f"♪ {text} ♪"
    inputs = processor(prompt, voice_preset=voice_preset)
    with torch.no_grad():
        # Sampling is required - Bark produces nothing usable greedily.
        generated = model.generate(**inputs, do_sample=True)

    audio = generated.cpu().numpy().squeeze().astype(np.float32)
    if audio.ndim != 1 or len(audio) == 0:
        return None
    if float(np.abs(audio).max()) < MIN_PEAK_AMPLITUDE:
        return None
    return _trim_silence(audio)


def sing_line(bark, text: str, voice_preset: str = DEFAULT_VOICE_PRESET,
              target_duration_s: float | None = None) -> np.ndarray | None:
    """Generate sung audio for one line, at BARK_SAMPLE_RATE.

    Retries when a take runs far longer than target_duration_s (the script's
    syllable-based estimate for the line). Bark has no duration control and
    will sometimes keep going past the lyric; since scene length follows audio
    length, an over-long take parks one image on screen while the rest of the
    poem races past. Generation is non-deterministic, so simply asking again
    usually produces something in range.

    The best over-long take is kept as a last resort rather than discarded, so
    an unlucky run degrades to "a bit long" instead of falling back to speech.
    Returns None only when no take was usable at all."""
    best = None
    for attempt in range(1, MAX_TAKES + 1):
        audio = _generate_take(bark, text, voice_preset)
        if audio is None:
            log_with_fields(logger, 30, "bark take unusable, retrying", attempt=attempt)
            continue

        duration = len(audio) / BARK_SAMPLE_RATE
        if target_duration_s is None or duration <= target_duration_s * MAX_DURATION_RATIO:
            return normalize_pitch(audio, BARK_SAMPLE_RATE)

        # Too long - keep it only if it is the shortest over-long take so far.
        if best is None or len(audio) < len(best):
            best = audio
        log_with_fields(logger, 30, "bark take too long, retrying", attempt=attempt,
                         duration_s=round(duration, 2), target_s=round(target_duration_s, 2),
                         limit_s=round(target_duration_s * MAX_DURATION_RATIO, 2))

    if best is None:
        return None
    log_with_fields(logger, 30, "bark takes all over-long, keeping shortest",
                     duration_s=round(len(best) / BARK_SAMPLE_RATE, 2))
    return normalize_pitch(best, BARK_SAMPLE_RATE)
