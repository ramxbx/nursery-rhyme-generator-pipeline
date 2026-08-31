"""Motion generation agent (GPT-26).

Regenerates each scene as a short animated clip with AnimateDiff, so the video
moves rather than panning across a still.

Note what this does NOT do: it never opens the image stage's PNG. AnimateDiff
here is text-to-video, so a scene is rebuilt from its prompt and its seed. The
image stage's real contribution is choosing that seed - it renders several
candidates and keeps whichever one CLIP judged to contain the subject best.

This is a GPU stage and therefore its own subprocess, like visual_agent - one
model load, released when the process exits. It must not run concurrently with
the image stage; orchestration sequences them.

Settings are the outcome of a nine-configuration sweep (see STATUS.md), judged
by eye rather than by metric:

* 384x384, because 256x256 degrades past legibility and 512x512 costs 150 s/frame
  with no quality gain the eye could find. SD1.5 is native at 512, so everything
  here is off-distribution - the soft "watercolour" look is the model degrading
  gracefully, and it is the look this project wants.
* 10 steps. 12 was tested and judged worse: more denoising smooths away the
  texture that gives 10 its character.
* 16 frames exported at 8fps, so one clip covers 2 seconds, then ffmpeg
  interpolates to the video frame rate.

Wide aspect ratios were tried and rejected: 512x288 is the same pixel count and
the same cost, but SD1.5 draws a herd of duplicated subjects instead of one.

~36 minutes per scene on a GTX 1050, which is 81% of a run's wall clock - a
six-scene poem is about four hours. Enable deliberately.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import torch

from src.config import PipelineConfig, load_config
from src.utils.file_manager import ensure_dirs, read_json, safe_write_json, scene_path
from src.utils.logger import get_logger, log_with_fields

logger = get_logger("motion_agent")

MOTION_ADAPTER = "guoyww/animatediff-motion-adapter-v1-5-2"
# Chosen by eye over eight alternatives - see the grid in STATUS.md.
MOTION_WIDTH = MOTION_HEIGHT = 384
MOTION_STEPS = 10
MOTION_FRAMES = 16
# 16 frames at 8fps = 2 seconds of source motion per scene.
MOTION_FPS = 8


def build_motion_pipeline(sd_config: dict):
    """AnimateDiff on top of the project's own checkpoint.

    Loaded straight to CUDA with no CPU offload: offloading was measured at
    508 s/frame against 527 s resident, i.e. it shares storage rather than
    compute, and the GPU does all the work either way. Keeping weights resident
    avoids the transfer without costing speed."""
    from diffusers import AnimateDiffPipeline, DPMSolverMultistepScheduler, MotionAdapter

    adapter = MotionAdapter.from_pretrained(MOTION_ADAPTER, torch_dtype=torch.float16)
    pipe = AnimateDiffPipeline.from_pretrained(
        sd_config["base_model"], motion_adapter=adapter,
        torch_dtype=torch.float16, safety_checker=None)
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(
        pipe.scheduler.config, algorithm_type="dpmsolver++", use_karras_sigmas=True)
    pipe = pipe.to("cuda" if torch.cuda.is_available() else "cpu")
    # Decoding 16 frames at once is the one real VRAM spike in an otherwise
    # fitting run.
    pipe.vae.enable_slicing()
    return pipe


def motion_size(config: PipelineConfig) -> tuple[int, int]:
    """Generation size for a motion clip, as (width, height).

    Configurable because the square default wastes 44% of every frame: the
    finished video is 16:9, so a 384x384 clip is scaled 5x to cover 1920x1080
    and then centre-cropped, discarding 168 of its 384 rows. 512x288 is the
    same 147,456 pixels - the same diffusion cost, to within noise - but none
    of it is thrown away, and the horizontal resolution rises to SD1.5's native
    512. Both dimensions must stay divisible by 8 for the VAE."""
    motion = config.pipeline.get("motion", {})
    width = int(motion.get("width", MOTION_WIDTH))
    height = int(motion.get("height", MOTION_HEIGHT))
    for name, value in (("width", width), ("height", height)):
        if value % 8:
            raise ValueError(f"motion.{name}={value} must be divisible by 8")
    return width, height


def animate_scene(pipe, prompt: str, seed: int, sd_config: dict,
                   size: tuple[int, int] | None = None):
    """One scene's frames. Returns None on OOM so the caller can fall back to
    the still image rather than losing the whole stage."""
    width, height = size or (MOTION_WIDTH, MOTION_HEIGHT)
    try:
        result = pipe(
            prompt=prompt,
            negative_prompt=sd_config.get("negative_prompt"),
            num_frames=MOTION_FRAMES,
            num_inference_steps=MOTION_STEPS,
            width=width,
            height=height,
            guidance_scale=sd_config.get("guidance_scale", 7.0),
            generator=torch.Generator("cpu").manual_seed(seed),
        )
        return result.frames[0]
    except torch.cuda.OutOfMemoryError:
        log_with_fields(logger, 40, "motion generation OOM, scene keeps its still image")
        torch.cuda.empty_cache()
        return None


def generate_motion(images_manifest: list[dict], config: PipelineConfig) -> list[dict]:
    """A motion clip per scene, reusing the image stage's prompt and seed so the
    animation depicts the same moment the still did."""
    from diffusers.utils import export_to_video

    dirs = ensure_dirs(config.paths)
    motion_dir = Path(dirs["images_dir"]).parent / "motion"
    motion_dir.mkdir(parents=True, exist_ok=True)

    size = motion_size(config)
    pipe = build_motion_pipeline(config.sd)
    manifest = []
    for entry in images_manifest:
        idx = entry["scene_index"]
        started = time.perf_counter()
        # The image stage's own prompt, so the animation depicts the moment the
        # still did - and its seed, so it starts from the composition CLIP chose.
        prompt = entry["prompt"]
        frames = animate_scene(pipe, prompt, entry["seed"], config.sd, size)
        if frames is None:
            continue
        out_path = scene_path(motion_dir, idx, ".mp4")
        export_to_video(frames, str(out_path), fps=MOTION_FPS)

        elapsed = round(time.perf_counter() - started, 1)
        record = {"scene_index": idx, "motion_path": str(out_path),
                  "frames": MOTION_FRAMES, "fps": MOTION_FPS,
                  "width": size[0], "height": size[1],
                  "duration_s": MOTION_FRAMES / MOTION_FPS, "elapsed_s": elapsed,
                  "prompt": prompt}
        manifest.append(record)
        log_with_fields(logger, 20, "scene motion generated", scene_index=idx,
                         elapsed_s=elapsed, path=str(out_path))
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate animated clips for each scene.")
    # Unused - the prompts and seeds come from the images manifest - but kept
    # so every stage takes the same first argument.
    parser.add_argument("script", type=Path)
    parser.add_argument("--images-manifest", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    config = load_config()
    dirs = ensure_dirs(config.paths)
    images = read_json(args.images_manifest or (dirs["images_dir"] / "manifest.json"))

    manifest = generate_motion(images, config)
    out_path = args.out or (Path(dirs["images_dir"]).parent / "motion" / "manifest.json")
    safe_write_json(out_path, manifest)
    print(f"Wrote {len(manifest)} motion clips, manifest at {out_path}")


if __name__ == "__main__":
    main()
