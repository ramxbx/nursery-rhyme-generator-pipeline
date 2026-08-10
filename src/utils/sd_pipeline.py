"""SD1.5 + LCM-LoRA pipeline construction, tuned for a 4GB GTX 1050.

Kept separate from visual_agent so the heavy load happens once per process
and the memory-safety settings (fp16, SDPA attention, VAE slicing/tiling,
model CPU offload) live in one place, matching config/sd_config.yml.
"""
from __future__ import annotations

import os
from pathlib import Path

# Must be set before the diffusers/huggingface_hub import below, so this
# process reuses the weights already fetched into the repo's models/hf_cache
# (GPT-19) instead of re-downloading into the default ~/.cache/huggingface.
os.environ.setdefault("HF_HOME", str(Path(__file__).resolve().parent.parent.parent / "models" / "hf_cache"))

import torch
from diffusers import DiffusionPipeline, LCMScheduler

from src.utils.logger import get_logger, log_with_fields

logger = get_logger("sd_pipeline")


def build_pipeline(sd_config: dict) -> DiffusionPipeline:
    device_available = torch.cuda.is_available()
    pipe = DiffusionPipeline.from_pretrained(
        sd_config["base_model"],
        torch_dtype=torch.float16,
        variant=sd_config.get("variant", "fp16"),
        safety_checker=None,
    )
    pipe.load_lora_weights(sd_config["lora"])
    pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)

    if sd_config.get("enable_vae_slicing"):
        pipe.enable_vae_slicing()
    if sd_config.get("enable_vae_tiling"):
        pipe.enable_vae_tiling()

    if device_available and sd_config.get("enable_model_cpu_offload", True):
        pipe.enable_model_cpu_offload()
        log_with_fields(logger, 20, "pipeline ready", device="cuda (offloaded)")
    elif device_available:
        pipe = pipe.to("cuda")
        log_with_fields(logger, 20, "pipeline ready", device="cuda")
    else:
        pipe = pipe.to("cpu")
        log_with_fields(logger, 30, "CUDA unavailable, running on CPU", device="cpu")

    return pipe


def to_cpu_fallback(pipe: DiffusionPipeline) -> DiffusionPipeline:
    """Called on CUDA OOM: release GPU memory and move the pipeline to CPU."""
    torch.cuda.empty_cache()
    pipe = pipe.to("cpu")
    log_with_fields(logger, 40, "switched pipeline to CPU fallback after OOM")
    return pipe
