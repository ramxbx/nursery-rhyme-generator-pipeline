"""AnimateDiff attempt 2: use the idle VRAM instead of thrashing host RAM.

Attempt 1 enabled every "memory saving" option - model CPU offload, VAE
slicing, VAE tiling. All three trade VRAM for host RAM, and host RAM turned out
to be the scarce resource: 7.14GB resident on a 16GB machine, still loading
after 26 minutes, while only 2.16GB of 4.29GB VRAM was in use. The 16-frame run
was killed outright by the OS.

So this inverts it. No offload at all - load straight to CUDA and let the model
sit in VRAM, which is the resource actually going spare. Expect ~3.5GB of 4.29
resident, which is tight but is what the card is for.

VAE decode is the one place that genuinely needs slicing: decoding 8 frames at
once is a real VRAM spike at the end of an otherwise-fitting run, and slicing it
costs host RAM only briefly.
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

OUT = Path(__file__).resolve().parent.parent / "data" / "output"
MOTION_ADAPTER = "guoyww/animatediff-motion-adapter-v1-5-2"
# 2 seconds at 8fps. 16 frames previously appeared to kill the process, but that
# was the missing opencv import failing AFTER generation - worth retrying.
FRAMES = int(__import__('os').environ.get('AD_FRAMES', 8))
# Halved. Cost is linear in steps, and DPM++ 2M Karras is usable at 10.
STEPS = int(__import__('os').environ.get('AD_STEPS', 8))
# 384x384 is 0.5625x the latent area of 512x512 - the last lever with real
# headroom after no-offload gave nothing (527s/frame vs 508s offloaded).
WIDTH = HEIGHT = 256
SCENE = 3            # the lamb close-up
USE_IP_ADAPTER = False   # attempt 1 loaded it; leave it off until the base fits


def vram():
    free, total = torch.cuda.mem_get_info()
    return f"{(total - free) / 1e9:.2f}GB VRAM used / {total / 1e9:.2f}GB"


def main():
    cfg = load_config()
    entry = {e["scene_index"]: e for e in json.load(open(ROOT / "data/images/manifest.json"))}[SCENE]

    t0 = time.time()
    adapter = MotionAdapter.from_pretrained(MOTION_ADAPTER, torch_dtype=torch.float16)
    pipe = AnimateDiffPipeline.from_pretrained(
        cfg.sd["base_model"], motion_adapter=adapter, torch_dtype=torch.float16, safety_checker=None)
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(
        pipe.scheduler.config, algorithm_type="dpmsolver++", use_karras_sigmas=True)

    # The inversion: straight to CUDA, no offload hooks, nothing streaming
    # through host RAM during inference.
    pipe = pipe.to("cuda")
    # Decoding all frames at once is the one real VRAM spike; slice just that.
    pipe.vae.enable_slicing()
    print(f"loaded in {time.time() - t0:.0f}s | {vram()}", flush=True)

    extra = {}
    if USE_IP_ADAPTER:
        pipe.load_ip_adapter("h94/IP-Adapter", subfolder="models", weight_name="ip-adapter_sd15.bin")
        pipe.set_ip_adapter_scale(0.5)
        pipe.image_encoder = pipe.image_encoder.to("cuda", dtype=torch.float16)
        extra["ip_adapter_image"] = Image.open(ROOT / entry["image_path"]).convert("RGB").resize((512, 512))
        print(f"ip-adapter on | {vram()}", flush=True)

    print(f"scene {SCENE} [{entry.get('shot')}] {FRAMES} frames, {STEPS} steps, {WIDTH}x{HEIGHT}", flush=True)
    t = time.time()
    try:
        out = pipe(
            prompt=entry["prompt"],
            negative_prompt=cfg.sd["negative_prompt"],
            num_frames=FRAMES,
            num_inference_steps=STEPS,
            width=WIDTH,
            height=HEIGHT,
            guidance_scale=cfg.sd.get("guidance_scale", 7.0),
            generator=torch.Generator("cpu").manual_seed(entry["seed"]),
            **extra,
        )
    except torch.cuda.OutOfMemoryError:
        print(f"CUDA OOM after {time.time() - t:.0f}s | {vram()}", flush=True)
        return
    elapsed = time.time() - t
    # The motion arc AnimateDiff produces is fixed regardless of frame count, so
    # 8 diffused frames still describe the whole 2 seconds - just sparsely. Export
    # them at 4fps (8 frames = 2s) and let ffmpeg synthesise the in-betweens.
    raw = OUT / f"anim4_{WIDTH}px_{FRAMES}f_{STEPS}s_raw.mp4"
    # fps chosen so the clip is always 2 seconds regardless of frame count
    export_to_video(out.frames[0], str(raw), fps=FRAMES / 2)
    path = OUT / f"anim4_{WIDTH}px_{FRAMES}f_{STEPS}s.mp4"
    t_i = time.time()
    import subprocess
    from src.utils.ffmpeg_helper import ffmpeg_path
    # minterpolate estimates motion vectors between frames rather than blending,
    # so it fills genuine in-betweens instead of cross-fading. mci = motion
    # compensated interpolation; the alternative (blend) just ghosts.
    subprocess.run([ffmpeg_path(), "-y", "-v", "error", "-i", str(raw),
                    "-vf", "minterpolate=fps=24:mi_mode=mci:mc_mode=aobmc:me_mode=bidir:vsbmc=1",
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", str(path)], check=False)
    print(f"interpolated 4fps -> 24fps in {time.time() - t_i:.0f}s", flush=True)
    print(f"OK {elapsed:.0f}s ({elapsed / FRAMES:.1f}s/frame) | {vram()} -> {path.name}", flush=True)


if __name__ == "__main__":
    main()
