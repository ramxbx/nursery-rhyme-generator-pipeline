"""Unit tests for audio_agent (GPT-16). Piper synthesis is mocked with
synthetic audio chunks - these test the padding/resample logic, not
Piper's actual voice quality."""
from dataclasses import dataclass

import numpy as np

from src.agents import audio_agent as aa


@dataclass
class FakeChunk:
    sample_rate: int
    audio_float_array: np.ndarray


class FakeVoice:
    def __init__(self, sample_rate: int, natural_duration_s: float):
        self.sample_rate = sample_rate
        self.natural_duration_s = natural_duration_s

    def synthesize(self, text, syn_config=None):
        n = int(self.sample_rate * self.natural_duration_s)
        return [FakeChunk(sample_rate=self.sample_rate, audio_float_array=np.zeros(n, dtype=np.float32))]


def test_resample_linear_preserves_duration_and_changes_sample_count():
    audio = np.zeros(22050, dtype=np.float32)  # 1 second at 22050Hz
    resampled = aa.resample_linear(audio, orig_sr=22050, target_sr=48000)
    assert len(resampled) == 48000


def test_resample_linear_noop_when_rates_match():
    audio = np.arange(100, dtype=np.float32)
    result = aa.resample_linear(audio, orig_sr=48000, target_sr=48000)
    assert np.array_equal(result, audio)


def test_synthesize_line_pads_short_speech_to_target_duration(monkeypatch):
    monkeypatch.setattr(aa, "apply_singsong_contour", lambda audio, sr, n_words: audio)
    voice = FakeVoice(sample_rate=22050, natural_duration_s=1.0)
    tts_config = {"sample_rate": 48000, "singing_mode": False}
    audio = aa.synthesize_line(voice, "Hi.", target_duration_s=2.0, tts_config=tts_config)
    assert abs(len(audio) / 48000 - 2.0) < 0.01


def test_synthesize_line_does_not_truncate_speech_longer_than_target(monkeypatch):
    monkeypatch.setattr(aa, "apply_singsong_contour", lambda audio, sr, n_words: audio)
    voice = FakeVoice(sample_rate=22050, natural_duration_s=3.0)
    tts_config = {"sample_rate": 48000, "singing_mode": False}
    audio = aa.synthesize_line(voice, "A much longer line of speech.", target_duration_s=1.0, tts_config=tts_config)
    assert len(audio) / 48000 >= 2.9  # not cut down to 1.0s
