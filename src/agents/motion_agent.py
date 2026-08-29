"""Motion generation agent (GPT-26).

Turns each scene's still image into a short animated clip with AnimateDiff,
so the video moves rather than panning across stills.

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

Roughly 40 minutes per scene on a GTX 1050, which dominates the pipeline - a
six-scene poem is about four hours. Enable deliberately.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import torch
from PIL import Image

from src.config import PipelineConfig, load_config
from src.utils.compositing import BACKGROUND_NEGATIVE, background_prompt, cut_out_subject
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

# Rebuilt here rather than imported from visual_agent: importing it would pull
# the whole SD image pipeline into this process for two string constants.
STYLE_PREFIX = "modern disney style, cute children's cartoon"
SHOT_PREFIXES = {
    "wide": "wide establishing shot, full scene visible, detailed background",
    "medium": "medium shot, surroundings in frame",
    "close": "close up, simple background",
}


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


def animate_scene(pipe, prompt: str, seed: int, sd_config: dict):
    """One scene's frames. Returns None on OOM so the caller can fall back to
    the still image rather than losing the whole stage."""
    try:
        result = pipe(
            prompt=prompt,
            # Characters are pushed out of the background pass explicitly - the
            # setting words alone are not enough to stop SD drawing an animal.
            negative_prompt=", ".join(filter(None, [sd_config.get("negative_prompt"),
                                                     BACKGROUND_NEGATIVE])),
            num_frames=MOTION_FRAMES,
            num_inference_steps=MOTION_STEPS,
            width=MOTION_WIDTH,
            height=MOTION_HEIGHT,
            guidance_scale=sd_config.get("guidance_scale", 7.0),
            generator=torch.Generator("cpu").manual_seed(seed),
        )
        return result.frames[0]
    except torch.cuda.OutOfMemoryError:
        log_with_fields(logger, 40, "motion generation OOM, scene keeps its still image")
        torch.cuda.empty_cache()
        return None


def generate_motion(script: dict, images_manifest: list[dict], config: PipelineConfig) -> list[dict]:
    """A motion clip per scene, reusing the image stage's prompt and seed so the
    animation depicts the same moment the still did."""
    from diffusers.utils import export_to_video

    dirs = ensure_dirs(config.paths)
    motion_dir = Path(dirs["images_dir"]).parent / "motion"
    motion_dir.mkdir(parents=True, exist_ok=True)

    pipe = build_motion_pipeline(config.sd)
    manifest = []
    for entry in images_manifest:
        idx = entry["scene_index"]
        started = time.perf_counter()
        scene = script["scenes"][idx - 1] if idx - 1 < len(script["scenes"]) else {}
        # The background is animated WITHOUT the subject: it is composited back
        # in afterwards from the still, sharp. Animating it here would both
        # duplicate it and spend the motion budget warping a face.
        prompt = background_prompt(STYLE_PREFIX, SHOT_PREFIXES.get(entry.get("shot"), ""),
                                    scene.get("scene_description", ""))
        frames = animate_scene(pipe, prompt, entry["seed"], config.sd)
        if frames is None:
            continue
        out_path = scene_path(motion_dir, idx, ".mp4")
        export_to_video(frames, str(out_path), fps=MOTION_FPS)

        subject_path = motion_dir / f"subject_{idx:03d}.png"
        coverage = cut_out_subject(Path(entry["image_path"]), subject_path,
                                    entry.get("shot", "medium"))
        elapsed = round(time.perf_counter() - started, 1)
        record = {"scene_index": idx, "motion_path": str(out_path),
                  "frames": MOTION_FRAMES, "fps": MOTION_FPS,
                  "duration_s": MOTION_FRAMES / MOTION_FPS, "elapsed_s": elapsed,
                  "background_prompt": prompt}
        # A scene whose subject could not be segmented still gets its animated
        # background; assembly falls back to using it whole.
        if coverage is not None:
            record["subject_path"] = str(subject_path)
            record["subject_coverage"] = round(coverage, 3)
        manifest.append(record)
        log_with_fields(logger, 20, "scene motion generated", scene_index=idx,
                         elapsed_s=elapsed, path=str(out_path))
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate animated clips for each scene.")
    parser.add_argument("script", type=Path)
    parser.add_argument("--images-manifest", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    config = load_config()
    dirs = ensure_dirs(config.paths)
    script = read_json(args.script)
    images = read_json(args.images_manifest or (dirs["images_dir"] / "manifest.json"))

    manifest = generate_motion(script, images, config)
    out_path = args.out or (Path(dirs["images_dir"]).parent / "motion" / "manifest.json")
    safe_write_json(out_path, manifest)
    print(f"Wrote {len(manifest)} motion clips, manifest at {out_path}")


if __name__ == "__main__":
    main()
