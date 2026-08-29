"""Feasibility spike: AnimateDiff on the existing lamb scenes.

Question this answers, in order:
  1. Does SD1.5 + motion module fit in 4.29GB at all?
  2. How slow is it per scene?
  3. Does IP-Adapter conditioning keep the lamb looking like our lamb?

Uses the real scene prompts from the image manifest, and conditions on the
already-generated scene image so the motion clip inherits its look rather than
inventing a new lamb.
"""
import os, sys, time
from pathlib import Path

ROOT = Path(r"C:\Users\abhiv\Documents\Projects\LM_CLAUDE_WS")
os.environ.setdefault("HF_HOME", str(ROOT / "models" / "hf_cache"))
sys.path.insert(0, str(ROOT))

import json
import torch
from PIL import Image
from diffusers import AnimateDiffPipeline, DPMSolverMultistepScheduler, MotionAdapter
from diffusers.utils import export_to_video

from src.config import load_config

OUT = Path(__file__).parent
MOTION_ADAPTER = "guoyww/animatediff-motion-adapter-v1-5-2"
FRAMES = 8           # halved: 16 frames killed the process outright
STEPS = 20
SCENES = [3]         # one close-up only


def vram():
    free, total = torch.cuda.mem_get_info()
    return f"{(total - free) / 1e9:.2f}GB used / {total / 1e9:.2f}GB"


def main():
    cfg = load_config()
    manifest = {e["scene_index"]: e for e in json.load(open(ROOT / "data/images/manifest.json"))}

    t0 = time.time()
    adapter = MotionAdapter.from_pretrained(MOTION_ADAPTER, torch_dtype=torch.float16)
    pipe = AnimateDiffPipeline.from_pretrained(
        cfg.sd["base_model"], motion_adapter=adapter, torch_dtype=torch.float16, safety_checker=None)
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(
        pipe.scheduler.config, algorithm_type="dpmsolver++", use_karras_sigmas=True)
    # Everything available for memory: 16 frames of latents is the whole problem.
    pipe.enable_vae_slicing()
    pipe.enable_vae_tiling()
    pipe.enable_model_cpu_offload()
    print(f"loaded in {time.time() - t0:.0f}s | {vram()}")

    # Condition on the existing scene image so the clip inherits our lamb.
    try:
        pipe.load_ip_adapter("h94/IP-Adapter", subfolder="models", weight_name="ip-adapter_sd15.bin")
        pipe.set_ip_adapter_scale(0.5)
        if getattr(pipe, "image_encoder", None) is not None:
            pipe.image_encoder = pipe.image_encoder.to(pipe._execution_device, dtype=torch.float16)
        ip_ok = True
    except Exception as e:
        print("ip-adapter unavailable:", str(e)[:120])
        ip_ok = False
    print(f"after ip-adapter | {vram()}")

    for idx in SCENES:
        entry = manifest[idx]
        prompt = entry["prompt"]
        ref = Image.open(ROOT / entry["image_path"]).convert("RGB").resize((512, 512))
        extra = {"ip_adapter_image": ref} if ip_ok else {}
        print(f"\n-- scene {idx} [{entry.get('shot')}] {FRAMES} frames")
        print(f"   {prompt[:110]}")
        t = time.time()
        try:
            out = pipe(
                prompt=prompt,
                negative_prompt=cfg.sd["negative_prompt"],
                num_frames=FRAMES,
                num_inference_steps=STEPS,
                guidance_scale=cfg.sd.get("guidance_scale", 7.0),
                generator=torch.Generator("cpu").manual_seed(entry["seed"]),
                **extra,
            )
        except torch.cuda.OutOfMemoryError as e:
            print(f"   OOM after {time.time() - t:.0f}s | {vram()}")
            torch.cuda.empty_cache()
            return
        elapsed = time.time() - t
        path = OUT / f"anim_scene_{idx:03d}.mp4"
        export_to_video(out.frames[0], str(path), fps=8)
        print(f"   OK {elapsed:.0f}s ({elapsed / FRAMES:.1f}s/frame) | {vram()} -> {path.name}")


if __name__ == "__main__":
    main()
