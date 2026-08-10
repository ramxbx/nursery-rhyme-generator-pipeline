"""One-time fetch of the local model weights for the visual/audio stages.

Downloads:
  - stable-diffusion-v1-5/stable-diffusion-v1-5 (fp16 variant)
  - latent-consistency/lcm-lora-sdv1-5
  - rhasspy/piper-voices: en_US-lessac-medium
"""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
os.environ.setdefault("HF_HOME", str(ROOT / "models" / "hf_cache"))

import torch
from diffusers import DiffusionPipeline
from huggingface_hub import hf_hub_download

print("== SD1.5 base (fp16) ==")
pipe = DiffusionPipeline.from_pretrained(
    "stable-diffusion-v1-5/stable-diffusion-v1-5",
    torch_dtype=torch.float16,
    variant="fp16",
    safety_checker=None,
)
print("SD1.5 loaded:", type(pipe).__name__)

print("== LCM-LoRA ==")
pipe.load_lora_weights("latent-consistency/lcm-lora-sdv1-5")
print("LCM-LoRA loaded")

print("== Piper voice: en_US-lessac-medium ==")
voice_dir = ROOT / "models" / "piper"
voice_dir.mkdir(parents=True, exist_ok=True)
for fname in ["en/en_US/lessac/medium/en_US-lessac-medium.onnx",
              "en/en_US/lessac/medium/en_US-lessac-medium.onnx.json"]:
    dest = hf_hub_download(repo_id="rhasspy/piper-voices", filename=fname, local_dir=str(voice_dir))
    print("fetched:", dest)

print("ALL DOWNLOADS COMPLETE")
