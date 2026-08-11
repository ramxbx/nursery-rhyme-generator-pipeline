"""Procedural background music (GPT-20 feedback #2) - no model, no download.

Synthesizes a soft, repeating major-pentatonic melody (music-box/lullaby
character - three stacked harmonics with a gentle attack/release envelope),
long enough to cover the full video and mixed at low volume under the
narration in animate_agent.
"""
from __future__ import annotations

import numpy as np

PENTATONIC_SEMITONES = [0, 2, 4, 7, 9]  # major pentatonic
BASE_FREQ_HZ = 261.63  # C4
DEFAULT_PATTERN = [0, 2, 4, 2, 0, 2, 4, 7, 4, 2, 0, 2]


def _note_freq(scale_index: int) -> float:
    octave, degree = divmod(scale_index, len(PENTATONIC_SEMITONES))
    semitone = PENTATONIC_SEMITONES[degree] + 12 * octave
    return BASE_FREQ_HZ * (2 ** (semitone / 12))


def _envelope(n_samples: int, attack: float = 0.15, release: float = 0.35) -> np.ndarray:
    env = np.ones(n_samples, dtype=np.float32)
    a = max(1, int(n_samples * attack))
    r = max(1, int(n_samples * release))
    env[:a] = np.linspace(0.0, 1.0, a, dtype=np.float32)
    env[-r:] *= np.linspace(1.0, 0.0, r, dtype=np.float32)
    return env


def _synth_note(freq: float, duration_s: float, sample_rate: int, amplitude: float) -> np.ndarray:
    t = np.linspace(0.0, duration_s, int(duration_s * sample_rate), endpoint=False, dtype=np.float32)
    wave = (np.sin(2 * np.pi * freq * t)
            + 0.3 * np.sin(2 * np.pi * 2 * freq * t)
            + 0.15 * np.sin(2 * np.pi * 3 * freq * t)) / 1.45
    return (wave * _envelope(len(t)) * amplitude).astype(np.float32)


def generate_background_music(duration_s: float, sample_rate: int = 48000, note_duration_s: float = 0.5,
                               amplitude: float = 0.22, pattern: list[int] | None = None) -> np.ndarray:
    pattern = pattern or DEFAULT_PATTERN
    notes = []
    total = 0.0
    i = 0
    while total < duration_s:
        freq = _note_freq(pattern[i % len(pattern)])
        notes.append(_synth_note(freq, note_duration_s, sample_rate, amplitude))
        total += note_duration_s
        i += 1

    music = np.concatenate(notes)
    target_len = int(duration_s * sample_rate)
    music = music[:target_len] if len(music) >= target_len else np.pad(music, (0, target_len - len(music)))

    fade_samples = min(len(music), int(sample_rate * 1.0))
    if fade_samples > 0:
        music[-fade_samples:] *= np.linspace(1.0, 0.0, fade_samples, dtype=np.float32)
    return music
