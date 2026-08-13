"""Visual composition agent (GPT-11).

Reads each scene's description from the script and generates one image
with plain SD1.5. This is the only stage allowed to touch the GTX 1050's
GPU - one process, one load, released when this stage's process exits.

Deliberately minimal. Earlier versions layered on an LLM prompt-drafting
call, LCM-LoRA, a style LoRA, chunked long-prompt encoding, and three
quality-gate/retry checks (blank, tiled, CLIP wrong-subject). Each was
tested and none reliably improved final image quality - the LLM drafting
step in particular lost the actual subject on some scenes, and the retry
gates mostly burned attempts without converging on better output. Stripped
back to the simplest thing that works: scene description in, image out.

Native generation is low-res (~512x288); upscaling to 1920x1080 happens
later during FFmpeg assembly (GPT-13), not here - see GPT-18/GPT-11 for why
(4GB VRAM can't safely do native 1080p diffusion on this hardware).
"""
from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import torch

from src.config import PipelineConfig, load_config
from src.utils.file_manager import ensure_dirs, read_json, safe_write_json, scene_path
from src.utils.logger import get_logger, log_with_fields
from src.utils.sd_pipeline import (
    build_clip_verifier, build_pipeline, clip_subject_similarity, to_cpu_fallback,
)

logger = get_logger("visual_agent")

# Best-of-N over seeds, scored by CLIP subject similarity (GPT-34). A single
# generation lands a correct, coherent subject maybe half the time even with a
# good checkpoint, and "amazing SD1.5 images" seen online are heavily
# survivor-biased - people generate a batch and keep one. This is the automated
# equivalent. Rather than accept the first image clearing a threshold, generate
# a few and keep the best-scoring one, so a scene never ships worse output than
# it had to. Kept small: each attempt is a full ~40s generation.
CLIP_ATTEMPTS = 3
# Above this, a candidate is good enough to stop early rather than spend the
# remaining attempts. Set from observed scores on this checkpoint: a full run
# where every scene rendered a clear, correct subject scored 0.275-0.290, so an
# earlier 0.30 bar was never reached and every scene wasted its full attempt
# budget. 0.275 stops early on a genuinely good image while still rejecting the
# 0.23-0.25 band that wrong-subject renders fall into.
CLIP_GOOD_ENOUGH = 0.275

# Leads every prompt so the style stays consistent scene-to-scene (GPT-22).
# "modern disney style" is the trigger phrase mo-di-diffusion was fine-tuned on
# (config/sd_config.yml) - without it the checkpoint drifts toward generic
# photographic output. The framing terms are here because the checkpoint's
# failure mode is composing the subject small and distant in a wide landscape;
# stating the framing explicitly keeps it front and centre (GPT-31).
STYLE_ANCHOR = "modern disney style, cute children's cartoon, subject close up and centered, simple background"


def seed_for_character(name: str) -> int:
    """Deterministic seed so the same character keeps the same base seed
    across scenes/runs, supporting visual consistency."""
    digest = hashlib.sha256(name.strip().lower().encode("utf-8")).hexdigest()
    return int(digest[:8], 16)


def build_prompt(pipe, scene: dict) -> str:
    """Scene description + fixed style anchor, truncated to what SD1.5's
    CLIP text encoder can actually read (77 tokens per pass, including
    BOS/EOS - anything beyond that is silently ignored by the model). The
    style anchor is placed first so it always survives truncation."""
    description = scene.get("scene_description") or "soft colorful picture-book setting, gentle pastel colors"
    text = f"{STYLE_ANCHOR}, {description}"
    tokenizer = pipe.tokenizer
    ids = tokenizer(text, add_special_tokens=False)["input_ids"]
    if len(ids) <= CLIP_TOKEN_BUDGET:
        return text
    return tokenizer.decode(ids[:CLIP_TOKEN_BUDGET])


CLIP_TOKEN_BUDGET = 75  # CLIP's hard cap is 77 including BOS/EOS


def generate_image(pipe, prompt: str, seed: int, sd_config: dict):
    """Generate one image, with a CPU fallback on CUDA OOM."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    for _ in range(2):
        generator = torch.Generator(device=device if device == "cuda" else "cpu").manual_seed(seed)
        try:
            result = pipe(
                prompt=prompt,
                negative_prompt=sd_config.get("negative_prompt"),
                num_inference_steps=sd_config.get("steps", 30),
                guidance_scale=sd_config.get("guidance_scale", 7.0),
                width=sd_config.get("width", 512),
                height=sd_config.get("height", 288),
                generator=generator,
            )
            return result.images[0], pipe
        except torch.cuda.OutOfMemoryError:
            log_with_fields(logger, 40, "CUDA OOM during generation, falling back to CPU")
            pipe = to_cpu_fallback(pipe)
            device = "cpu"

    raise RuntimeError("Image generation failed on both GPU and CPU")


def generate_best_image(pipe, prompt: str, seed: int, sd_config: dict, clip_verifier, subject: str):
    """Generate up to CLIP_ATTEMPTS candidates from consecutive seeds and keep
    the one whose CLIP subject-similarity is highest, stopping early once a
    candidate is clearly good (GPT-34). Returns (image, pipe, score, attempts)."""
    best_image, best_score = None, -1.0
    for attempt in range(CLIP_ATTEMPTS):
        image, pipe = generate_image(pipe, prompt, seed + attempt, sd_config)
        score = clip_subject_similarity(clip_verifier, image, subject)
        if score > best_score:
            best_image, best_score = image, score
        if score >= CLIP_GOOD_ENOUGH:
            return best_image, pipe, best_score, attempt + 1
        log_with_fields(logger, 20, "candidate below quality bar, retrying",
                         attempt=attempt + 1, score=round(score, 4), subject=subject)
    return best_image, pipe, best_score, CLIP_ATTEMPTS


def _subject_of(scene: dict) -> str:
    """The scene description's first comma-separated segment is the subject -
    the template requires every prompt to lead with it (species + colour + a
    key trait). Used as the CLIP verification target."""
    description = scene.get("scene_description") or ""
    return description.split(",", 1)[0].strip() or "a children's cartoon character"


def generate_visuals(script: dict, config: PipelineConfig) -> list[dict]:
    dirs = ensure_dirs(config.paths)
    pipe = build_pipeline(config.sd)
    clip_verifier = build_clip_verifier()
    manifest = []

    for i, scene in enumerate(script["scenes"], start=1):
        speaker = scene["speaker"]
        prompt = build_prompt(pipe, scene)
        subject = _subject_of(scene)
        # Offset by scene index so a single-narrator poem doesn't generate
        # every scene from the identical seed (GPT-28).
        seed = seed_for_character(speaker) + i * 1000
        image, pipe, score, attempts = generate_best_image(
            pipe, prompt, seed, config.sd, clip_verifier, subject)
        log_with_fields(logger, 20, "scene subject verified", scene_index=i,
                         score=round(score, 4), attempts=attempts, subject=subject)

        out_path = scene_path(dirs["images_dir"], i, ".png")
        image.save(out_path)
        manifest.append({"scene_index": i, "speaker": speaker, "prompt": prompt,
                          "seed": seed, "image_path": str(out_path),
                          "clip_score": round(score, 4), "attempts": attempts})
        log_with_fields(logger, 20, "scene image generated", scene_index=i, image_path=str(out_path))

    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate scene images from a script JSON file.")
    parser.add_argument("script", type=Path, help="Path to a script JSON file (from script_agent)")
    parser.add_argument("--out", type=Path, default=None, help="Manifest output path")
    args = parser.parse_args()

    config = load_config()
    script = read_json(args.script)
    manifest = generate_visuals(script, config)

    out_path = args.out or (config.paths["images_dir"] / "manifest.json")
    safe_write_json(out_path, manifest)
    print(f"Wrote {len(manifest)} images, manifest at {out_path}")


if __name__ == "__main__":
    main()
