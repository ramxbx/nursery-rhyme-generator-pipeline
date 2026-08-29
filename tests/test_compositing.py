"""Unit tests for subject cutouts and background-only prompts."""
import numpy as np
import pytest

from src.utils import compositing as cp


def _fake_scene(tmp_path, name="scene.png"):
    """A bright square subject on a dark background - the easiest possible
    case for GrabCut, so a failure here is a real failure."""
    import cv2

    img = np.full((256, 256, 3), 20, np.uint8)
    img[70:190, 80:180] = (240, 240, 240)
    path = tmp_path / name
    cv2.imwrite(str(path), img)
    return path


def test_background_prompt_drops_the_subject():
    """The subject is composited back from the still, so leaving it in the
    background prompt yields two of them - and wastes the motion budget
    warping a face instead of moving grass."""
    p = cp.background_prompt("style", "wide shot",
                             "fluffy white lamb, sunny day, rural setting")
    assert "lamb" not in p
    assert "sunny day" in p and "rural setting" in p
    assert cp.BACKGROUND_SUFFIX in p


def test_background_prompt_survives_a_single_segment():
    p = cp.background_prompt("style", "wide shot", "a misty field")
    assert "a misty field" in p


def test_background_prompt_survives_an_empty_description():
    assert cp.background_prompt("style", "wide shot", "")


def test_subject_rect_varies_with_shot_type():
    """A close-up fills the frame; a wide shot has the subject small and
    centred. One rectangle cannot serve both."""
    close = cp.SUBJECT_RECTS["close"]
    wide = cp.SUBJECT_RECTS["wide"]
    assert close[2] * close[3] > wide[2] * wide[3]


def test_cutout_produces_an_alpha_channel(tmp_path):
    import cv2

    out = tmp_path / "cut.png"
    coverage = cp.cut_out_subject(_fake_scene(tmp_path), out, "close")
    assert coverage is not None, "clean subject on a plain background should segment"
    rgba = cv2.imread(str(out), cv2.IMREAD_UNCHANGED)
    assert rgba.shape[2] == 4
    assert rgba[:, :, 3].max() == 255 and rgba[:, :, 3].min() == 0


def test_implausible_mask_is_rejected_rather_than_shipped(monkeypatch, tmp_path):
    """GrabCut occasionally returns near-empty when subject and background
    share colours. Overlaying a fragment is worse than not compositing."""
    import cv2

    def all_background(img, mask, rect, bgd, fgd, iters, mode):
        mask[:] = cv2.GC_BGD

    monkeypatch.setattr(cv2, "grabCut", all_background)
    assert cp.cut_out_subject(_fake_scene(tmp_path), tmp_path / "c.png", "close") is None


def test_missing_image_returns_none(tmp_path):
    assert cp.cut_out_subject(tmp_path / "nope.png", tmp_path / "c.png") is None
