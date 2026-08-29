"""Unit tests for the motion stage. The AnimateDiff model is never loaded -
these cover the settings contract and how motion clips reach assembly."""
from pathlib import Path

from src.agents import motion_agent as ma
from src.utils import ffmpeg_helper as fh


def test_settings_match_the_configuration_chosen_by_the_sweep():
    """Nine configurations were compared by eye; these are the winners. A silent
    drift here would quietly change the look of every video."""
    assert (ma.MOTION_WIDTH, ma.MOTION_HEIGHT) == (384, 384)
    assert ma.MOTION_STEPS == 10, "12 steps was tested and judged worse"
    assert ma.MOTION_FRAMES == 16


def test_a_clip_covers_two_seconds():
    """Scene durations vary, but every generated clip must be the same length so
    the loop arithmetic in build_motion_scene_clip holds."""
    assert ma.MOTION_FRAMES / ma.MOTION_FPS == 2.0


def test_motion_clip_is_looped_enough_to_fill_the_scene(monkeypatch):
    """A 2s clip under a 5.4s scene must loop, or the video runs out of picture
    before the line finishes singing."""
    captured = {}
    monkeypatch.setattr(fh, "run", lambda args: captured.setdefault("args", args))
    fh.build_motion_scene_clip(Path("m.mp4"), Path("a.wav"), 5.4, 24, 1920, 1080, Path("o.mp4"))
    loops = int(captured["args"][captured["args"].index("-stream_loop") + 1])
    assert (loops + 1) * 2.0 >= 5.4


def test_interpolation_precedes_upscaling(monkeypatch):
    """minterpolate must see the clean 384x384 source; running it after the
    upscale would have it estimating motion from lanczos artefacts."""
    captured = {}
    monkeypatch.setattr(fh, "run", lambda args: captured.setdefault("args", args))
    fh.build_motion_scene_clip(Path("m.mp4"), Path("a.wav"), 3.0, 24, 1920, 1080, Path("o.mp4"))
    chain = captured["args"][captured["args"].index("-filter_complex") + 1]
    assert chain.index("minterpolate") < chain.index("scale=")


def test_interpolation_uses_motion_compensation_not_blending(monkeypatch):
    """Blending cross-fades between frames and ghosts; mci estimates where
    pixels actually moved."""
    captured = {}
    monkeypatch.setattr(fh, "run", lambda args: captured.setdefault("args", args))
    fh.build_motion_scene_clip(Path("m.mp4"), Path("a.wav"), 3.0, 24, 1920, 1080, Path("o.mp4"))
    chain = captured["args"][captured["args"].index("-filter_complex") + 1]
    assert "mi_mode=mci" in chain


def test_scenes_without_motion_fall_back_to_ken_burns(monkeypatch, tmp_path):
    """Motion is per-scene, not all-or-nothing: a scene whose generation OOMed
    still ships with its still image."""
    from src.agents import animate_agent as aa

    used = []
    monkeypatch.setattr(aa, "build_motion_scene_clip",
                        lambda *a, **k: used.append("motion") or a[-1].write_bytes(b"x"))
    monkeypatch.setattr(aa, "build_scene_clip",
                        lambda *a, **k: used.append("kenburns") or a[-2].write_bytes(b"x"))
    # must actually create the merged file - with music and subtitles off,
    # assembly moves it into place rather than re-encoding
    monkeypatch.setattr(aa, "crossfade_concat",
                        lambda clips, durations, fps, out, **k: out.write_bytes(b"x"))
    monkeypatch.setattr(aa, "probe", lambda p: {"format": {"duration": "4.0"}})
    monkeypatch.setattr(aa, "_validate_output", lambda *a, **k: None)
    monkeypatch.setattr(aa, "ensure_dirs", lambda p: {"output_dir": tmp_path, "images_dir": tmp_path})

    class Cfg:
        video = {"fps": 24, "width": 1920, "height": 1080}
        pipeline = {"music": {"enabled": False}, "subtitles": {"enabled": False}}
        tts = {"sample_rate": 48000}
        paths = {"output_dir": tmp_path}

    images = [{"scene_index": i, "image_path": str(tmp_path / f"i{i}.png")} for i in (1, 2)]
    audio = [{"scene_index": i, "audio_path": str(tmp_path / f"a{i}.wav"),
              "actual_duration_s": 3.0, "line": "x"} for i in (1, 2)]
    motion = [{"scene_index": 1, "motion_path": str(tmp_path / "m1.mp4")}]  # scene 2 has none

    aa.assemble_video(images, audio, Cfg(), tmp_path / "out.mp4", motion)
    assert used == ["motion", "kenburns"]


def test_composite_is_preferred_when_a_subject_was_segmented(monkeypatch, tmp_path):
    """The whole point of the composite path: the subject stays sharp because it
    comes from the still, and only the background is allowed to warp."""
    from src.agents import animate_agent as aa

    used = []
    monkeypatch.setattr(aa, "build_composite_scene_clip",
                        lambda *a, **k: used.append("composite") or a[-1].write_bytes(b"x"))
    monkeypatch.setattr(aa, "build_motion_scene_clip",
                        lambda *a, **k: used.append("motion") or a[-1].write_bytes(b"x"))
    monkeypatch.setattr(aa, "build_scene_clip",
                        lambda *a, **k: used.append("kenburns") or a[-2].write_bytes(b"x"))
    monkeypatch.setattr(aa, "crossfade_concat",
                        lambda clips, durations, fps, out, **k: out.write_bytes(b"x"))
    monkeypatch.setattr(aa, "probe", lambda p: {"format": {"duration": "4.0"}})
    monkeypatch.setattr(aa, "_validate_output", lambda *a, **k: None)
    monkeypatch.setattr(aa, "ensure_dirs", lambda p: {"output_dir": tmp_path, "images_dir": tmp_path})

    class Cfg:
        video = {"fps": 24, "width": 1920, "height": 1080}
        pipeline = {"music": {"enabled": False}, "subtitles": {"enabled": False}}
        tts = {"sample_rate": 48000}
        paths = {"output_dir": tmp_path}

    images = [{"scene_index": i, "image_path": str(tmp_path / f"i{i}.png")} for i in (1, 2, 3)]
    audio = [{"scene_index": i, "audio_path": str(tmp_path / f"a{i}.wav"),
              "actual_duration_s": 3.0, "line": "x"} for i in (1, 2, 3)]
    motion = [
        {"scene_index": 1, "motion_path": str(tmp_path / "m1.mp4"),
         "subject_path": str(tmp_path / "s1.png")},   # segmented -> composite
        {"scene_index": 2, "motion_path": str(tmp_path / "m2.mp4")},  # no subject -> motion
        # scene 3 has no motion at all -> ken burns
    ]

    aa.assemble_video(images, audio, Cfg(), tmp_path / "out.mp4", motion)
    assert used == ["composite", "motion", "kenburns"]
