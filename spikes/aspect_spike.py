"""Does generating 16:9 natively beat generating square and cropping?

The finished video is 1920x1080. A 384x384 clip gets scaled 5x to cover that
and then centre-cropped, so 168 of its 384 rows are generated at full diffusion
cost and then thrown away - 96% of the delivered pixels are lanczos invention.

512x288 is 147,456 pixels against 384x384's 147,456: the same compute, to
within noise. But it is already 16:9, so nothing is cropped, and the horizontal
resolution rises from 384 to SD1.5's native 512.

The risk is that AnimateDiff's motion module was trained on square-ish frames,
so 288 rows may be far enough off-distribution to break the motion. That is not
predictable from first principles, which is why this renders both and puts them
through the identical delivery path for a like-for-like look.

Run:  ./.venv/Scripts/python.exe -u spikes/aspect_spike.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from diffusers.utils import export_to_video

from src.agents.motion_agent import MOTION_FPS, animate_scene, build_motion_pipeline
from src.config import load_config
from src.utils.ffmpeg_helper import build_motion_scene_clip
from src.utils.file_manager import read_json

# Scene 2 of the lamb run: a medium shot, which is the framing animated scenes
# now use throughout. Its 384px source was already legible, so this measures the
# aspect change rather than rescuing an unreadable wide shot.
SCENE_INDEX = 2
SIZES = [(512, 288), (384, 384)]
OUT_DIR = Path("data/output")


def main() -> None:
    config = load_config()
    manifest = read_json(Path("data/images/manifest.json"))
    entry = next(e for e in manifest if e["scene_index"] == SCENE_INDEX)
    audio = next(a for a in read_json(Path("data/audio/manifest.json"))
                 if a["scene_index"] == SCENE_INDEX)
    audio_path, duration = Path(audio["audio_path"]), audio["actual_duration_s"]

    print(f"scene {SCENE_INDEX}  seed={entry['seed']}  shot={entry.get('shot')}")
    print(f"prompt: {entry['prompt']}\n")

    pipe = build_motion_pipeline(config.sd)
    for width, height in SIZES:
        tag = f"{width}x{height}"
        started = time.perf_counter()
        frames = animate_scene(pipe, entry["prompt"], entry["seed"], config.sd,
                                composite=False, size=(width, height))
        if frames is None:
            print(f"{tag}: OOM")
            continue
        elapsed = time.perf_counter() - started

        raw = OUT_DIR / f"aspect_{tag}_raw.mp4"
        export_to_video(frames, str(raw), fps=MOTION_FPS)
        # Through the real delivery path, so what is compared is what a viewer
        # would actually see rather than the generator's own output.
        build_motion_scene_clip(raw, audio_path, duration, config.video["fps"],
                                 config.video["width"], config.video["height"],
                                 OUT_DIR / f"aspect_{tag}_1080p.mp4")
        print(f"{tag}: {elapsed / 60:.1f} min  ({elapsed / len(frames):.0f} s/frame)  -> {raw.name}")


if __name__ == "__main__":
    main()
