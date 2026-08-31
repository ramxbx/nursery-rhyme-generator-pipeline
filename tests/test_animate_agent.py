"""Unit tests for animate_agent / ffmpeg_helper (GPT-16). Pure timing math
and validation logic - no real ffmpeg invocation."""
import pytest

from src.agents import animate_agent as an
from src.utils.ffmpeg_helper import compute_scene_timeline


def test_compute_scene_timeline_matches_hand_computed_values():
    timeline = compute_scene_timeline([2.0, 3.0, 2.5], crossfade_s=0.5)
    assert timeline == [(0.0, 2.0), (1.5, 4.5), (4.0, 6.5)]


def test_compute_scene_timeline_clamps_crossfade_to_short_scenes():
    timeline = compute_scene_timeline([0.2, 3.0], crossfade_s=0.5)
    # xfade_dur clamped to min(0.5, 0.2, 3.0) = 0.2, not 0.5
    assert timeline[1][0] == pytest.approx(0.0)


def test_compute_scene_timeline_single_scene():
    assert compute_scene_timeline([4.0], crossfade_s=0.5) == [(0.0, 4.0)]


def test_compute_scene_timeline_empty():
    assert compute_scene_timeline([], crossfade_s=0.5) == []


def test_validate_output_raises_on_wrong_resolution(monkeypatch, tmp_path):
    monkeypatch.setattr(an, "probe", lambda path: {
        "streams": [
            {"codec_type": "video", "codec_name": "h264", "width": 1280, "height": 720, "r_frame_rate": "24/1"},
            {"codec_type": "audio", "codec_name": "aac"},
        ]
    })
    with pytest.raises(an.AssemblyError, match="1920x1080"):
        an._validate_output(tmp_path / "fake.mp4", 1920, 1080, 24)


def test_validate_output_raises_on_missing_audio_stream(monkeypatch, tmp_path):
    monkeypatch.setattr(an, "probe", lambda path: {
        "streams": [
            {"codec_type": "video", "codec_name": "h264", "width": 1920, "height": 1080, "r_frame_rate": "24/1"},
        ]
    })
    with pytest.raises(an.AssemblyError, match="audio"):
        an._validate_output(tmp_path / "fake.mp4", 1920, 1080, 24)


def test_validate_output_passes_for_correct_stream_info(monkeypatch, tmp_path):
    monkeypatch.setattr(an, "probe", lambda path: {
        "streams": [
            {"codec_type": "video", "codec_name": "h264", "width": 1920, "height": 1080, "r_frame_rate": "24/1"},
            {"codec_type": "audio", "codec_name": "aac"},
        ]
    })
    an._validate_output(tmp_path / "fake.mp4", 1920, 1080, 24)  # should not raise


def test_camera_motion_varies_across_scenes():
    """A single slow centre push on every scene reads as a stuck camera by the
    third one."""
    from src.utils.ffmpeg_helper import MOTIONS, motion_for_scene

    moves = [motion_for_scene(i) for i in range(1, 10)]
    assert len(set(moves)) == len(MOTIONS)
    assert moves[0] != moves[1]


def test_motion_cycle_does_not_align_with_the_shot_cycle():
    """Three moves against four framings means the pairing takes twelve scenes
    to repeat; equal cycles would pair the same move with the same shot forever."""
    from src.utils.ffmpeg_helper import MOTIONS
    from src.agents.visual_agent import SHOT_CYCLE

    assert len(MOTIONS) != len(SHOT_CYCLE)


def test_motion_assignment_is_deterministic():
    from src.utils.ffmpeg_helper import motion_for_scene

    assert [motion_for_scene(i) for i in range(1, 7)] == [motion_for_scene(i) for i in range(1, 7)]


def test_no_camera_move_relies_on_zoompans_accumulator():
    """The bug this catches: push_in and pull_out were written as
    z='min(zoom+step,MAX)', relying on zoompan's `zoom` variable carrying over
    between frames. It does not on this input - these clips are built from
    `-loop 1 -i image` with d=1, so every output frame comes from a fresh input
    frame and `zoom` resets each time. The expression evaluated to a constant,
    and 5 of 8 scenes in a finished video had no motion at all.

    The filter stayed valid ffmpeg and produced a correct-looking video, so
    nothing failed - only measuring first-vs-last frames revealed it."""
    from src.utils.ffmpeg_helper import MOTIONS, _zoompan

    for motion in MOTIONS:
        spec = _zoompan(motion, n_frames=96, width=1920, height=1080, fps=24)
        z = spec.split("z=", 1)[1].split(":", 1)[0]
        assert "zoom" not in z, (
            f"{motion} drives zoom from the accumulator ({z}), which does not "
            f"advance with d=1 on a looped still")


def test_every_camera_move_actually_changes_across_the_clip():
    """Each move must differ between its first and last frame - a filter that
    parses and renders is not evidence that anything moved."""
    from src.utils.ffmpeg_helper import MOTIONS, _zoompan

    for motion in MOTIONS:
        spec = _zoompan(motion, n_frames=96, width=1920, height=1080, fps=24)
        # `on` is the output frame index; without it nothing can vary over time.
        assert "on/" in spec, f"{motion} has no term that varies with frame index: {spec}"
