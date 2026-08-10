"""Visual composition agent (GPT-11).

Codex controls stage execution, quality gates, retries, and fallback
decisions. The CPU-hosted local model only drafts image prompts from
already-approved scene data (bounded task). SD1.5 + LCM-LoRA generates the
images and is the only stage allowed to touch the GTX 1050's GPU - one
process, one load, released when this stage's process exits.

Native generation is low-res (~512x288); upscaling to 1920x1080 happens
later during FFmpeg assembly (GPT-13), not here - see GPT-18/GPT-11 for why
(4GB VRAM can't safely do native 1080p diffusion on this hardware).
"""
from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import numpy as np
import torch

from src.config import PipelineConfig, load_config
from src.utils.file_manager import ensure_dirs, read_json, safe_write_json, scene_path
from src.utils.llm_client import LLMError, call_with_fallback
from src.utils.logger import get_logger, log_with_fields
from src.utils.prompt_builder import PromptError, build_scene_image_prompt
from src.utils.sd_pipeline import build_pipeline, to_cpu_fallback

logger = get_logger("visual_agent")

MAX_GEN_RETRIES = 2
LOW_QUALITY_STD_THRESHOLD = 8.0  # near-uniform (blank) image guard


def seed_for_character(name: str) -> int:
    """Deterministic seed so the same character keeps the same base seed
    across scenes/runs, supporting visual consistency."""
    digest = hashlib.sha256(name.strip().lower().encode("utf-8")).hexdigest()
    return int(digest[:8], 16)


def draft_subject_description(speaker: str, llm_config: dict) -> str:
    """One bounded LLM call per unique character to fix a consistent visual
    description, reused across every scene that character appears in."""
    system = (
        "Describe a children's picture-book character in ONE short phrase "
        "(under 15 words) covering species/type, key visual traits, and "
        "color palette. Output ONLY the phrase, no other text."
    )
    try:
        result = call_with_fallback(system_prompt=system, user_prompt=speaker,
                                     llm_config=llm_config, parse_json=False, max_tokens=60)
        return result.text.strip().strip('"')
    except LLMError as e:
        log_with_fields(logger, 30, "subject description drafting failed, using deterministic default",
                         speaker=speaker, error=str(e))
        return f"a friendly cartoon {speaker.lower()}, soft pastel colors"


def draft_image_prompt(speaker: str, subject_description: str, stage_direction: str, llm_config: dict) -> str:
    """Bounded prompt drafting: expand scene data into an SD-style prompt."""
    template_prompt = build_scene_image_prompt(speaker, subject_description, stage_direction)
    try:
        result = call_with_fallback(
            system_prompt=template_prompt,
            user_prompt="Write the image prompt now.",
            llm_config=llm_config,
            parse_json=False,
            max_tokens=120,
        )
        text = result.text.strip()
        return text if text else template_prompt
    except (LLMError, PromptError) as e:
        log_with_fields(logger, 30, "image prompt drafting failed, using template directly", error=str(e))
        return f"{subject_description}, {stage_direction}, picture-book illustration style"


def _is_low_quality(image) -> bool:
    arr = np.asarray(image.convert("L"), dtype=np.float32)
    return float(arr.std()) < LOW_QUALITY_STD_THRESHOLD


def generate_image(pipe, prompt: str, seed: int, sd_config: dict):
    """Generate one image, retrying on low-quality output, with a CPU
    fallback on CUDA OOM."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    last_image = None
    for attempt in range(MAX_GEN_RETRIES + 1):
        gen_seed = seed + attempt
        generator = torch.Generator(device=device if device == "cuda" else "cpu").manual_seed(gen_seed)
        try:
            result = pipe(
                prompt,
                num_inference_steps=sd_config.get("steps", 4),
                guidance_scale=sd_config.get("guidance_scale", 1.0),
                width=sd_config.get("width", 512),
                height=sd_config.get("height", 288),
                generator=generator,
            )
            image = result.images[0]
        except torch.cuda.OutOfMemoryError:
            log_with_fields(logger, 40, "CUDA OOM during generation, falling back to CPU", attempt=attempt)
            pipe = to_cpu_fallback(pipe)
            device = "cpu"
            continue

        if not _is_low_quality(image):
            return image, pipe
        last_image = image
        log_with_fields(logger, 30, "low-quality image detected, retrying", attempt=attempt, seed=gen_seed)

    return last_image, pipe


def generate_visuals(script: dict, config: PipelineConfig) -> list[dict]:
    dirs = ensure_dirs(config.paths)
    pipe = build_pipeline(config.sd)

    subject_descriptions: dict[str, str] = {}
    manifest = []

    for i, scene in enumerate(script["scenes"], start=1):
        speaker = scene["speaker"]
        if speaker not in subject_descriptions:
            subject_descriptions[speaker] = draft_subject_description(speaker, config.llm)
            log_with_fields(logger, 20, "subject description fixed", speaker=speaker,
                             description=subject_descriptions[speaker])

        prompt = draft_image_prompt(speaker, subject_descriptions[speaker], scene["stage_direction"], config.llm)
        seed = seed_for_character(speaker)
        image, pipe = generate_image(pipe, prompt, seed, config.sd)

        out_path = scene_path(dirs["images_dir"], i, ".png")
        image.save(out_path)
        manifest.append({"scene_index": i, "speaker": speaker, "prompt": prompt,
                          "seed": seed, "image_path": str(out_path)})
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
