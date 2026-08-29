"""Subject cutouts for compositing a sharp still over an animated background.

AnimateDiff's failure mode is that it warps whatever it animates. A drifting
background reads as watercolour atmosphere; a drifting face reads as broken.
So the subject is not animated at all - it is cut out of the still the image
stage already produced, at full 768x768 quality, and composited over a
background-only clip.

Segmentation is OpenCV GrabCut rather than a learned matting model. It needs no
extra dependency (opencv-python is already present for video export), and for a
cartoon subject against a distinct background it is good enough: measured ~27%
foreground coverage with a clean silhouette on a test frame. Its weaknesses are
small holes in dark features like eyes, and whatever the initialisation
rectangle clips - which is why the rectangle varies with shot type.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from src.utils.logger import get_logger, log_with_fields

logger = get_logger("compositing")

# GrabCut is seeded with a rectangle assumed to contain the subject. How much of
# the frame that is depends on the framing: a close-up fills it, a wide shot has
# the subject small and centred. Values are (x, y, w, h) as fractions.
SUBJECT_RECTS = {
    "close": (0.06, 0.06, 0.88, 0.90),
    "medium": (0.14, 0.14, 0.72, 0.80),
    "wide": (0.28, 0.30, 0.44, 0.60),
}
GRABCUT_ITERATIONS = 5
# Feathering the alpha stops the composite showing a hard cut edge against a
# soft, low-resolution background.
FEATHER_PX = 7
# Below this the mask almost certainly failed - GrabCut occasionally returns
# near-empty when the subject's colours match the background closely. Better to
# skip compositing for that scene than to overlay a fragment.
MIN_COVERAGE = 0.04
MAX_COVERAGE = 0.92


def cut_out_subject(image_path: Path, out_path: Path, shot: str = "medium") -> float | None:
    """Write an RGBA cutout of the subject. Returns foreground coverage, or None
    if the mask looks like a failure and the scene should skip compositing."""
    import cv2

    img = cv2.imread(str(image_path))
    if img is None:
        return None
    h, w = img.shape[:2]
    fx, fy, fw, fh = SUBJECT_RECTS.get(shot, SUBJECT_RECTS["medium"])
    rect = (int(w * fx), int(h * fy), int(w * fw), int(h * fh))

    mask = np.zeros((h, w), np.uint8)
    bgd, fgd = np.zeros((1, 65), np.float64), np.zeros((1, 65), np.float64)
    try:
        cv2.grabCut(img, mask, rect, bgd, fgd, GRABCUT_ITERATIONS, cv2.GC_INIT_WITH_RECT)
    except cv2.error as e:  # noqa: PERF203 - failure here is non-fatal
        log_with_fields(logger, 30, "grabcut failed, scene keeps its full frame", error=str(e)[:120])
        return None

    fg = np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)
    coverage = float(fg.mean() / 255)
    if not (MIN_COVERAGE <= coverage <= MAX_COVERAGE):
        log_with_fields(logger, 30, "subject mask implausible, skipping composite",
                         coverage=round(coverage, 3), shot=shot)
        return None

    # Close small holes (eyes, nostrils) before feathering, or they show as
    # background-coloured specks in the middle of the subject.
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, kernel)
    fg = cv2.GaussianBlur(fg, (FEATHER_PX, FEATHER_PX), 0)

    rgba = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
    rgba[:, :, 3] = fg
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), rgba)
    log_with_fields(logger, 20, "subject cut out", shot=shot, coverage=round(coverage, 3),
                     path=str(out_path))
    return coverage


# Segments that describe where a scene is rather than who is in it. The scene
# description leads with the subject, so dropping the first segment and keeping
# the rest gives the setting - which is what the background clip should show.
BACKGROUND_SUFFIX = "empty scene, scenery only, no characters"
BACKGROUND_NEGATIVE = "animal, person, character, creature, face, figure"


def background_prompt(style: str, framing: str, scene_description: str) -> str:
    """Prompt for a background-only animation.

    Keeping the subject out matters twice over: the subject is composited from
    the still afterwards, so a second copy in the background would read as two
    animals, and AnimateDiff spends its motion budget warping whatever creature
    it drew instead of moving grass and light."""
    segments = [s.strip() for s in (scene_description or "").split(",") if s.strip()]
    setting = ", ".join(segments[1:]) if len(segments) > 1 else (segments[0] if segments else "")
    parts = [p for p in (style, framing, setting, BACKGROUND_SUFFIX) if p]
    return ", ".join(parts)
