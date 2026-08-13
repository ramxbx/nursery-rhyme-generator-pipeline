"""SD1.5 pipeline construction, tuned for a 4GB GTX 1050.

Kept separate from visual_agent so the heavy load happens once per process
and the memory-safety settings (fp16, SDPA attention, VAE slicing/tiling,
model CPU offload) live in one place, matching config/sd_config.yml.

Runs the standard scheduler at 25-50 steps rather than LCM-LoRA's few-step
distilled trajectory - LCM was tested extensively (4/8/14+ steps, several
LoRA stacking combinations) and never clearly beat the standard scheduler
on quality, while its narrow few-step convergence window turned out to be
a poor fit for later experiments like IP-Adapter (mismatched, ~3.5x slower
per step and less stable than IP-Adapter + standard scheduler should be).
"""
from __future__ import annotations

import os
from pathlib import Path

# Must be set before the diffusers/huggingface_hub import below, so this
# process reuses the weights already fetched into the repo's models/hf_cache
# (GPT-19) instead of re-downloading into the default ~/.cache/huggingface.
os.environ.setdefault("HF_HOME", str(Path(__file__).resolve().parent.parent.parent / "models" / "hf_cache"))

import torch
from diffusers import DiffusionPipeline, DPMSolverMultistepScheduler

from src.utils.logger import get_logger, log_with_fields

logger = get_logger("sd_pipeline")


def build_pipeline(sd_config: dict) -> DiffusionPipeline:
    device_available = torch.cuda.is_available()
    # `variant` is optional: community fine-tunes often publish fp32 weights
    # only, and passing variant="fp16" against those fails outright. Without
    # it, torch_dtype still casts to fp16 at load, which is what actually
    # matters for fitting in 4GB.
    load_kwargs = {"torch_dtype": torch.float16, "safety_checker": None}
    if sd_config.get("variant"):
        load_kwargs["variant"] = sd_config["variant"]
    pipe = DiffusionPipeline.from_pretrained(sd_config["base_model"], **load_kwargs)
    # DPM++ 2M Karras - the community-standard SD1.5 sampler, converges better
    # in the 20-30 step range this pipeline runs at than the PNDM default the
    # checkpoints ship with (GPT-33).
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(
        pipe.scheduler.config, algorithm_type="dpmsolver++", use_karras_sigmas=True)

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


CLIP_VERIFIER_MODEL = "openai/clip-vit-base-patch32"


def build_clip_verifier():
    """Small standalone CLIP model (image + text towers), used only to check
    that a generated image actually shows its intended subject (GPT-34).

    This is the only check in the pipeline that looks at image *content* -
    pixel-statistics checks can spot a blank or tiled frame but have no idea
    whether the picture is of a sheep or a toddler, which is a failure this
    project hits regularly. Deliberately the small ViT-B/32 (~600MB): a
    coarse similarity score is all that's needed, and it stays cheap enough
    to run on every candidate image."""
    from transformers import CLIPModel, CLIPProcessor

    model = CLIPModel.from_pretrained(CLIP_VERIFIER_MODEL, torch_dtype=torch.float16)
    processor = CLIPProcessor.from_pretrained(CLIP_VERIFIER_MODEL)
    model.eval()
    if torch.cuda.is_available():
        model = model.to("cuda")
    return model, processor


def clip_subject_similarity(verifier, image, text: str) -> float:
    """Cosine similarity between an image and a text description in CLIP's
    joint embedding space - higher means the image more plausibly shows what
    the text describes. Calibrated against real renders from this project:
    images that did contain the described subject scored 0.27-0.31, images of
    an entirely wrong subject scored 0.23-0.25."""
    model, processor = verifier
    device = next(model.parameters()).device
    inputs = processor(text=[text], images=[image], return_tensors="pt", padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        out = model(**inputs)
    img_emb = out.image_embeds / out.image_embeds.norm(dim=-1, keepdim=True)
    txt_emb = out.text_embeds / out.text_embeds.norm(dim=-1, keepdim=True)
    return float((img_emb @ txt_emb.T).item())
