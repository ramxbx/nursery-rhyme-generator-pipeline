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


def test_each_scene_picks_its_own_source(monkeypatch, tmp_path):
    """Per scene, not all-or-nothing: a scene whose motion generation OOMed
    still ships, falling back to a Ken Burns pan over its still."""
    from src.agents import animate_agent as aa

    used = []
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

    images = [{"scene_index": i, "image_path": str(tmp_path / f"i{i}.png")} for i in (1, 2)]
    audio = [{"scene_index": i, "audio_path": str(tmp_path / f"a{i}.wav"),
              "actual_duration_s": 3.0, "line": "x"} for i in (1, 2)]
    # Scene 1 has a clip; scene 2 does not.
    motion = [{"scene_index": 1, "motion_path": str(tmp_path / "m1.mp4")}]

    aa.assemble_video(images, audio, Cfg(), tmp_path / "out.mp4", motion)
    assert used == ["motion", "kenburns"]


def _pipeline_cfg(tmp_path, motion_enabled):
    class Cfg:
        paths = {"scripts_dir": tmp_path, "images_dir": tmp_path / "images",
                 "audio_dir": tmp_path / "audio", "output_dir": tmp_path,
                 "data_dir": tmp_path, "logs_dir": tmp_path}
        pipeline = {"motion": {"enabled": motion_enabled}}
    return Cfg()


def _stub_orchestration(monkeypatch, tmp_path, motion_enabled):
    """run_pipeline with every stage stubbed; returns the recorded stage calls."""
    from src import orchestration as orch

    calls = []
    monkeypatch.setattr(orch, "load_config", lambda: _pipeline_cfg(tmp_path, motion_enabled))
    monkeypatch.setattr(orch, "ensure_dirs", lambda p: None)
    monkeypatch.setattr(orch, "run_stage",
                        lambda module, args, name, **k: calls.append((name, args)))
    monkeypatch.setattr(orch, "validate_script", lambda p: {"scenes": [{}, {}]})
    monkeypatch.setattr(orch, "validate_manifest", lambda *a, **k: None)
    monkeypatch.setattr(orch, "manifest_is_current", lambda *a, **k: False)
    monkeypatch.setattr(orch.Path, "exists", lambda self: True)
    # Content-fingerprinting is covered in test_run_isolation.py; these tests
    # are about which stages run, so it is stubbed rather than fed real files.
    monkeypatch.setattr(orch, "script_fingerprint", lambda p: "deadbeef")
    monkeypatch.setattr(orch, "stamp_manifest", lambda *a, **k: None)
    return orch, calls


def test_no_motion_skips_the_stage_and_assembles_from_stills(monkeypatch, tmp_path):
    """Ken Burns fallback is the whole point of the switch."""
    orch, calls = _stub_orchestration(monkeypatch, tmp_path, motion_enabled=False)
    orch.run_pipeline(tmp_path / "poem.txt", name="t", force=True)

    stages = [name for name, _ in calls]
    assert "motion" not in stages, stages


def test_a_stale_motion_manifest_is_not_used_when_motion_is_off(monkeypatch, tmp_path):
    """The bug this fixes: orchestration passed --motion-manifest unconditionally,
    so after one animated run, turning motion off skipped the expensive stage but
    still assembled from the previous run's clips - the switch looked inert."""
    orch, calls = _stub_orchestration(monkeypatch, tmp_path, motion_enabled=False)
    orch.run_pipeline(tmp_path / "poem.txt", name="t", force=True)

    animate_args = next(args for name, args in calls if name == "animate")
    assert "--motion-manifest" not in animate_args, animate_args


def test_motion_manifest_is_passed_when_motion_is_on(monkeypatch, tmp_path):
    orch, calls = _stub_orchestration(monkeypatch, tmp_path, motion_enabled=True)
    orch.run_pipeline(tmp_path / "poem.txt", name="t", force=True)

    stages = [name for name, _ in calls]
    animate_args = next(args for name, args in calls if name == "animate")
    assert "motion" in stages
    assert "--motion-manifest" in animate_args


def test_the_cli_flag_overrides_the_config_file(monkeypatch, tmp_path):
    """--no-motion must win over motion.enabled: true, and --motion over false,
    so a run can be redirected without editing config."""
    orch, calls = _stub_orchestration(monkeypatch, tmp_path, motion_enabled=True)
    orch.run_pipeline(tmp_path / "poem.txt", name="t", force=True, motion=False)
    assert "motion" not in [name for name, _ in calls]

    orch, calls = _stub_orchestration(monkeypatch, tmp_path, motion_enabled=False)
    orch.run_pipeline(tmp_path / "poem.txt", name="t", force=True, motion=True)
    assert "motion" in [name for name, _ in calls]
