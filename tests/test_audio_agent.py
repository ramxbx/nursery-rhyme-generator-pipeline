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
    monkeypatch.setattr(aa, "apply_singsong_contour", lambda audio, sr, n_words, start_offset=0: (audio, start_offset))
    voice = FakeVoice(sample_rate=22050, natural_duration_s=1.0)
    tts_config = {"sample_rate": 48000, "singing_mode": False}
    audio, offset = aa.synthesize_line(voice, "Hi.", target_duration_s=2.0, tts_config=tts_config)
    assert abs(len(audio) / 48000 - 2.0) < 0.01
    assert offset == 0


def test_synthesize_line_does_not_truncate_speech_longer_than_target(monkeypatch):
    monkeypatch.setattr(aa, "apply_singsong_contour", lambda audio, sr, n_words, start_offset=0: (audio, start_offset))
    voice = FakeVoice(sample_rate=22050, natural_duration_s=3.0)
    tts_config = {"sample_rate": 48000, "singing_mode": False}
    audio, offset = aa.synthesize_line(voice, "A much longer line of speech.", target_duration_s=1.0, tts_config=tts_config)
    assert len(audio) / 48000 >= 2.9  # not cut down to 1.0s


def test_apply_singsong_contour_threads_offset_across_calls():
    from src.utils.singing import apply_singsong_contour
    audio = np.random.default_rng(0).uniform(-0.1, 0.1, 22050).astype(np.float32)
    _, offset1 = apply_singsong_contour(audio, sr=22050, n_words=4, start_offset=0)
    assert offset1 == 4
    _, offset2 = apply_singsong_contour(audio, sr=22050, n_words=3, start_offset=offset1)
    assert offset2 == 7


def test_bark_take_is_trimmed_of_padding():
    """Bark pads lines with trailing near-silence, which would otherwise
    stretch the scene that audio belongs to."""
    from src.utils.bark_tts import _trim_silence

    speech = np.concatenate([
        np.zeros(1000, dtype=np.float32),
        np.ones(500, dtype=np.float32) * 0.5,
        np.zeros(4000, dtype=np.float32),
    ])
    assert len(_trim_silence(speech)) == 500


def test_generate_audio_falls_back_to_piper_when_bark_take_unusable(monkeypatch, tmp_path):
    """A single bad Bark generation should degrade one line to Piper, not
    fail the stage or ship silence."""
    monkeypatch.setattr(aa, "load_voice", lambda cfg: FakeVoice(22050, 1.0))
    monkeypatch.setattr(aa, "build_bark", lambda: object(), raising=False)
    monkeypatch.setattr("src.utils.bark_tts.build_bark", lambda: object())
    monkeypatch.setattr(aa, "synthesize_line_bark", lambda bark, text, cfg, target=None: None)

    class FakeConfig:
        tts = {"backend": "bark", "sample_rate": 48000, "singing_mode": False}
        paths = {"audio_dir": tmp_path}

    monkeypatch.setattr(aa, "ensure_dirs", lambda paths: {"audio_dir": tmp_path})
    script = {"scenes": [{"line": "Baa baa black sheep,", "duration_s": 2.0}]}
    manifest = aa.generate_audio(script, FakeConfig())

    assert len(manifest) == 1
    assert manifest[0]["source"] == "piper"
    assert manifest[0]["actual_duration_s"] > 0


class FakeBarkTakes:
    """Returns a scripted sequence of take lengths (seconds) so retry logic
    can be tested without running the real model."""

    def __init__(self, durations_s):
        self.durations = list(durations_s)
        self.calls = 0

    def __call__(self, bark, text, preset):
        self.calls += 1
        n = int(24000 * self.durations[min(self.calls - 1, len(self.durations) - 1)])
        return np.ones(n, dtype=np.float32) * 0.5


def test_bark_retries_when_take_runs_far_past_the_lyric(monkeypatch):
    """An over-long take should be rejected and regenerated - Bark has no
    duration control and scene length follows audio length."""
    from src.utils import bark_tts

    takes = FakeBarkTakes([20.0, 3.0])  # first rambles, second is fine
    monkeypatch.setattr(bark_tts, "_generate_take", takes)
    monkeypatch.setattr(bark_tts, "normalize_pitch", lambda a, sr: a)

    audio = bark_tts.sing_line(object(), "a short line", target_duration_s=4.0)
    assert takes.calls == 2
    assert abs(len(audio) / 24000 - 3.0) < 0.01


def test_bark_keeps_shortest_take_when_all_are_over_long(monkeypatch):
    """Falling back to speech is worse than a slightly-long sung take, so the
    least-bad take is kept rather than discarded."""
    from src.utils import bark_tts

    takes = FakeBarkTakes([20.0, 12.0, 15.0])
    monkeypatch.setattr(bark_tts, "_generate_take", takes)
    monkeypatch.setattr(bark_tts, "normalize_pitch", lambda a, sr: a)

    audio = bark_tts.sing_line(object(), "a short line", target_duration_s=4.0)
    assert takes.calls == bark_tts.MAX_TAKES
    assert abs(len(audio) / 24000 - 12.0) < 0.01
