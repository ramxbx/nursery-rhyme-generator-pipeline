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
from collections import Counter
import hashlib
import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import torch

from src.config import PipelineConfig, load_config
from src.utils.character_bank import (
    character_key, load_bank, reference_path, resolve_character, save_bank,
)
from src.utils.file_manager import ensure_dirs, read_json, safe_write_json, scene_path
from src.utils.logger import get_logger, log_with_fields
from src.utils.sd_pipeline import (
    build_clip_verifier,
    build_img2img, build_pipeline, clip_subject_similarity, enable_ip_adapter, to_cpu_fallback,
)

logger = get_logger("visual_agent")

# Best-of-N over seeds, scored by CLIP subject similarity (GPT-34). A single
# generation lands a correct, coherent subject maybe half the time even with a
# good checkpoint, and "amazing SD1.5 images" seen online are heavily
# survivor-biased - people generate a batch and keep one. This is the automated
# equivalent. Rather than accept the first image clearing a threshold, generate
# a few and keep the best-scoring one, so a scene never ships worse output than
# it had to. Kept small: each attempt is a full ~40s generation.
# Raised from 3 when framing began varying per scene: a wide shot has many more
# ways to compose badly than a portrait does (subject lost in the background,
# cropped, or duplicated across the frame), so it needs more chances to land
# one. Costs ~40s per extra attempt, and only scenes that miss their bar spend
# them - a good candidate still exits early.
CLIP_ATTEMPTS = 5
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
# photographic output.
#
# This used to also carry "subject close up and centered, simple background",
# added under GPT-31 when the checkpoint kept composing subjects tiny and
# distant. That worked and then overcorrected: every scene of every video became
# a portrait of the subject filling the frame, none of the objects the poem
# names ever appeared, and actions could not read at all because there was no
# room in frame for them. Framing now varies per scene (see SHOTS) and only the
# style is fixed here.
STYLE_ANCHOR = "modern disney style, cute children's cartoon"


@dataclass(frozen=True)
class Shot:
    """One camera framing, with the CLIP bar that suits it.

    The bar has to vary with the framing. CLIP subject-similarity measures how
    much of the image is the subject, so a wide shot scores lower than a close
    up of the same quality simply because the subject is smaller. Holding all
    shots to the close-up bar (0.275, calibrated on portraits) would make every
    wide scene burn its full attempt budget and then keep whichever candidate
    happened to be most zoomed-in - quietly undoing the variety this exists to
    create."""
    name: str
    prompt: str
    clip_good_enough: float


SHOTS = {
    "wide": Shot("wide", "wide establishing shot, full scene visible, detailed background", 0.235),
    "medium": Shot("medium", "medium shot, subject and surroundings both in frame", 0.260),
    "close": Shot("close", "close up, subject centered and filling the frame, simple background", 0.275),
}

# Cycled by scene position. Wide opens each group of four to re-establish the
# setting, medium does the work of showing the subject doing something to
# something, and close-ups land on every fourth scene as a reaction beat.
# Deterministic, so a re-run of the same poem frames identically.
SHOT_CYCLE = ["wide", "medium", "close", "medium"]

# When the motion stage is on, the scene is re-generated by AnimateDiff at
# 384px instead of the image stage's 768, and a wide shot does not survive that
# drop. Measured on the first full animated run: scene 1's "wide establishing
# shot" put the lamb in roughly 40 source pixels, which upscaled to an
# unreadable grey smudge, while scene 2's medium shot at the identical
# resolution kept the lamb, the cottage and its window frames all legible.
#
# So the subject has to be big enough in frame to survive - which is the same
# thing as saying: no wide shots when we are animating. Establishing shots come
# back for free if the motion stage is ever turned off.
ANIMATED_SHOT_CYCLE = ["medium", "close", "medium", "close"]


def shot_for_scene(scene_index: int, animated: bool = False) -> Shot:
    """Framing for a 1-based scene index.

    `animated` selects the cycle that omits wide shots - see
    ANIMATED_SHOT_CYCLE for why 384px cannot carry one."""
    cycle = ANIMATED_SHOT_CYCLE if animated else SHOT_CYCLE
    return SHOTS[cycle[(scene_index - 1) % len(cycle)]]


def seed_for_character(name: str) -> int:
    """Deterministic seed so the same character keeps the same base seed
    across scenes/runs, supporting visual consistency."""
    digest = hashlib.sha256(name.strip().lower().encode("utf-8")).hexdigest()
    return int(digest[:8], 16)


def build_prompt(pipe, scene: dict, shot: Shot | None = None, character: str | None = None) -> str:
    """Style anchor + this scene's framing + the character's canonical
    description + the scene description, truncated to what SD1.5's CLIP text
    encoder can actually read (77 tokens per pass, including BOS/EOS - anything
    beyond that is silently ignored by the model).

    Order is deliberate: style, framing and the character's fixed appearance all
    precede the scene's own wording, so the things that must stay constant
    across every scene and every episode are the things that survive
    truncation."""
    description = scene.get("scene_description") or "soft colorful picture-book setting, gentle pastel colors"
    framing = f"{shot.prompt}, " if shot else ""
    identity = ""
    if character:
        identity = f"{character}, "
        # The description's first segment IS the subject (that is what
        # _subject_of reads), so with a canonical descriptor in front of it the
        # prompt would name the subject twice. Beyond wasting scarce tokens,
        # the repeat re-weights the subject's own colour terms and bleeds them
        # across the frame - which is how a barnyard came back pink.
        # Only when that segment really is a bare subject label. The template
        # asks for "species + colour + one key trait" there, which is short; a
        # longer opening segment means the model packed action or setting into
        # it, and dropping that would silently lose what the scene is about.
        segments = [s.strip() for s in description.split(",") if s.strip()]
        if len(segments) > 1 and len(segments[0].split()) <= SUBJECT_SEGMENT_MAX_WORDS:
            description = ", ".join(segments[1:])
    text = f"{STYLE_ANCHOR}, {framing}{identity}{description}"
    tokenizer = pipe.tokenizer
    ids = tokenizer(text, add_special_tokens=False)["input_ids"]
    if len(ids) <= CLIP_TOKEN_BUDGET:
        return text
    return tokenizer.decode(ids[:CLIP_TOKEN_BUDGET])


CLIP_TOKEN_BUDGET = 75  # CLIP's hard cap is 77 including BOS/EOS

# A leading descriptor segment longer than this is carrying more than the
# subject's name, so it is kept rather than deduplicated away.
SUBJECT_SEGMENT_MAX_WORDS = 5


def generate_image(pipe, prompt: str, seed: int, sd_config: dict, reference=None):
    """Generate one image, with a CPU fallback on CUDA OOM.

    `reference` is a character portrait for IP-Adapter to condition on; passing
    None keeps the call text-only, which is what reference generation itself
    needs (there is nothing to condition on yet)."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    extra = {"ip_adapter_image": reference} if reference is not None else {}
    for _ in range(2):
        generator = torch.Generator(device=device if device == "cuda" else "cpu").manual_seed(seed)
        try:
            result = pipe(
                prompt=prompt,
                **extra,
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


def generate_best_image(pipe, prompt: str, seed: int, sd_config: dict, clip_verifier, subject: str,
                         good_enough: float = CLIP_GOOD_ENOUGH, reference=None):
    """Generate up to CLIP_ATTEMPTS candidates from consecutive seeds and keep
    the one whose CLIP subject-similarity is highest, stopping early once a
    candidate is clearly good (GPT-34).

    Returns (image, pipe, score, attempts, winning_seed).

    The winning seed is returned rather than left implicit because callers need
    the seed that actually produced this image, not the seed the search started
    from. The hires pass refines from it, and - when the motion stage is on -
    AnimateDiff regenerates the whole scene from it. Recording the base seed
    instead meant a scene that took three attempts was animated from `seed+0`,
    a composition CLIP had already rejected. It could not simply be recomputed
    downstream either: on the early-return path the winner is the last attempt,
    but when every candidate misses the bar the best one can be any of them.

    `good_enough` is per-shot rather than global - see Shot. A wide framing
    cannot reach a close-up's score, so judging it by that bar would spend
    every attempt and then pick the most zoomed-in candidate."""
    best_image, best_score, best_seed = None, -1.0, seed
    for attempt in range(CLIP_ATTEMPTS):
        candidate_seed = seed + attempt
        image, pipe = generate_image(pipe, prompt, candidate_seed, sd_config, reference)
        score = clip_subject_similarity(clip_verifier, image, subject)
        if score > best_score:
            best_image, best_score, best_seed = image, score, candidate_seed
        if score >= good_enough:
            return best_image, pipe, best_score, attempt + 1, best_seed
        log_with_fields(logger, 20, "candidate below quality bar, retrying",
                         attempt=attempt + 1, score=round(score, 4), subject=subject,
                         bar=good_enough)
    return best_image, pipe, best_score, CLIP_ATTEMPTS, best_seed


# Framing words for the reference portrait, applied AFTER the character's own
# descriptor rather than before it.
#
# The first version led with "character reference portrait, full body, standing,
# facing viewer". Every one of those is a human-portrait cue, and leading with
# them meant SD drew a human: asked for "tiny spider, dark brown" it produced a
# boy in a brown jacket. IP-Adapter then propagated that boy into all six
# scenes, faithfully - which is how we learned the conditioning works.
#
# So the subject leads, and the framing words that remain are species-neutral.
REFERENCE_FRAMING = ("full body, centered, plain white background, even lighting, "
                      "modern disney style, cute children's cartoon")
# A wrong reference poisons every scene that character ever appears in, which is
# far worse than one bad scene image. References are therefore held to a higher
# CLIP bar than scenes, and a character with no acceptable reference is left
# unconditioned rather than conditioned on something wrong.
REFERENCE_CLIP_MIN = 0.28


def ensure_reference(pipe, key: str, descriptor: str, seed: int, sd_config: dict,
                      clip_verifier) -> "Path | None":
    """The canonical portrait a character is conditioned on, generated once.

    A brand-new character has no reference, so its first appearance bootstraps
    one: a purpose-made full-body portrait on a plain background, best-of-N like
    any other image. Every scene afterwards - in this poem and in every later
    episode - is conditioned on that file, so this single image decides what the
    character looks like from then on.

    Generated with a plain background and neutral pose on purpose. Conditioning
    on a scene crop instead would drag that scene's setting, lighting and pose
    into everything the character ever appears in."""
    path = reference_path(key)
    if path.exists():
        return path
    prompt = f"{descriptor}, {REFERENCE_FRAMING}"
    image, pipe, score, attempts, _ = generate_best_image(
        pipe, prompt, seed, sd_config, clip_verifier, descriptor, REFERENCE_CLIP_MIN)
    if image is None or score < REFERENCE_CLIP_MIN:
        log_with_fields(logger, 30, "no acceptable reference, leaving character unconditioned",
                         character=key, score=round(score, 4) if image is not None else None,
                         bar=REFERENCE_CLIP_MIN)
        return None
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)
    log_with_fields(logger, 20, "character reference generated", character=key,
                     score=round(score, 4), attempts=attempts, path=str(path))
    return path


def hires_size(width: int, height: int, scale: float) -> tuple[int, int]:
    """Target size for the refinement pass, rounded to multiples of 8.

    SD's VAE downsamples by 8, so a dimension that is not a multiple of 8 is
    silently cropped - which would shift the image against the composition the
    first pass produced."""
    return (max(8, int(width * scale) // 8 * 8), max(8, int(height * scale) // 8 * 8))


def hires_fix(pipe, image, prompt: str, seed: int, sd_config: dict, reference=None):
    """Second pass at higher resolution, the standard SD "hires fix".

    SD1.5 was trained at 512x512 and generating directly above that produces
    duplicated, incoherent content - this project already hit exactly that as
    the tiling artifact in GPT-28, which is why GPT-32 moved generation back to
    native 512. So resolution cannot simply be raised.

    Instead: keep the coherent 512 composition, upscale it, and re-render at the
    higher size with a low denoise strength. Low denoise preserves the
    composition while the pass fills in detail the first one had no pixels for.
    That is what makes faces possible - a subject 60px tall in a wide shot
    cannot have eyes at 512, but at 768 the same subject has ~90px and the
    refinement pass can actually draw them.

    Applied only to the already-selected best candidate, never to every
    attempt - refining all of them would multiply the stage's cost by
    CLIP_ATTEMPTS for output that gets discarded."""
    from PIL import Image

    width, height = hires_size(image.width, image.height, sd_config.get("hires_scale", 1.5))
    upscaled = image.resize((width, height), Image.LANCZOS)

    img2img = build_img2img(pipe)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    # Once IP-Adapter is attached, the UNet expects image embeds on EVERY call
    # through it - including this one, which shares the same UNet. Omitting them
    # here raised "TypeError: argument of type 'NoneType' is not iterable" from
    # process_encoder_hidden_states and killed the stage after the reference had
    # already been generated. Conditioning the refinement pass on the character
    # is what we want anyway: it is the pass that draws the face.
    extra = {"ip_adapter_image": reference} if reference is not None else {}
    try:
        result = img2img(
            prompt=prompt,
            image=upscaled,
            **extra,
            strength=sd_config.get("hires_denoise", 0.4),
            num_inference_steps=sd_config.get("hires_steps", 24),
            guidance_scale=sd_config.get("guidance_scale", 7.0),
            negative_prompt=sd_config.get("negative_prompt"),
            generator=generator,
        )
        return result.images[0]
    except torch.cuda.OutOfMemoryError:
        # Better to ship the sharp-enough 512 image than fail the stage.
        log_with_fields(logger, 30, "hires fix OOM, keeping base image", width=width, height=height)
        torch.cuda.empty_cache()
        return image


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

    # Characters persist across poems, so the pig in this episode is the pig
    # from the last one (GPT-36). Loaded once and saved once, so a failed run
    # does not half-register a character.
    visual_config = config.pipeline.get("visual", {})
    use_bank = visual_config.get("character_bank", True)
    bank = load_bank() if use_bank else {}
    # The still and its animated version share a prompt and a seed, so the
    # framing decision has to be made here, once, for both.
    animated = config.pipeline.get("motion", {}).get("enabled", False)

    # Reference portraits are generated BEFORE the adapter is attached: a brand
    # new character has nothing to condition on yet, and conditioning a
    # reference on some other character's reference would be worse than useless.
    references, ip_ready = {}, False
    if use_bank and visual_config.get("ip_adapter", True):
        from PIL import Image

        # One character per poem, not one per scene. The scene-description
        # template establishes a single subject for the whole story, but the
        # model still varies its wording - one run called the same character
        # "tiny spider" in five scenes and "radiant arachnid" in the sixth,
        # which registered two characters, generated two references, and
        # conditioned the last scene on a different creature. The most common
        # key across scenes is the poem's actual subject.
        keys = [character_key(_subject_of(s)) for s in script["scenes"]]
        main_key = Counter(keys).most_common(1)[0][0]
        main_scene = next(s for s, k in zip(script["scenes"], keys) if k == main_key)
        subject = _subject_of(main_scene)
        canonical, base_seed, _ = resolve_character(
            bank, subject, seed_for_character(subject), main_scene.get("scene_description", ""))
        path = ensure_reference(pipe, main_key, canonical, base_seed, config.sd, clip_verifier)
        if path:
            # Every scene conditions on this one reference, whatever synonym its
            # own description happened to use.
            ref_image = Image.open(path).convert("RGB")
            references = {k: ref_image for k in set(keys)}
        if references:
            ip_ready = enable_ip_adapter(pipe, visual_config.get("ip_adapter_scale", 0.5))

    for i, scene in enumerate(script["scenes"], start=1):
        speaker = scene["speaker"]
        shot = shot_for_scene(i, animated)
        subject = _subject_of(scene)
        # Derived from the scene's own wording, BEFORE `subject` is replaced by
        # the canonical descriptor below. Keying off the descriptor instead
        # returns its last word - "flank" for a pig with a dark spot on one -
        # so the reference lookup missed and every scene silently generated
        # unconditioned.
        key = character_key(subject)
        # Offset by scene index so a single-narrator poem doesn't generate
        # every scene from the identical seed (GPT-28).
        seed = seed_for_character(speaker) + i * 1000
        canonical = None

        if use_bank:
            canonical, base_seed, is_new = resolve_character(
                bank, subject, seed_for_character(subject), scene.get("scene_description", ""))
            if is_new:
                log_with_fields(logger, 20, "character registered", key=key, descriptor=canonical)
            # The stored descriptor replaces this poem's wording for the subject,
            # which is what holds appearance steady between episodes.
            subject = canonical
            seed = base_seed + i * 1000

        prompt = build_prompt(pipe, scene, shot, canonical)
        reference = references.get(key) if ip_ready else None
        image, pipe, score, attempts, seed = generate_best_image(
            pipe, prompt, seed, config.sd, clip_verifier, subject, shot.clip_good_enough, reference)
        log_with_fields(logger, 20, "scene subject verified", scene_index=i, shot=shot.name,
                         score=round(score, 4), attempts=attempts, subject=subject,
                         seed=seed, conditioned=reference is not None)

        if config.sd.get("hires_fix", True):
            image = hires_fix(pipe, image, prompt, seed, config.sd, reference)
            log_with_fields(logger, 20, "hires pass applied", scene_index=i,
                             size=f"{image.width}x{image.height}")

        out_path = scene_path(dirs["images_dir"], i, ".png")
        image.save(out_path)
        manifest.append({"scene_index": i, "speaker": speaker, "prompt": prompt,
                          "seed": seed, "image_path": str(out_path), "shot": shot.name,
                          "clip_score": round(score, 4), "attempts": attempts})
        log_with_fields(logger, 20, "scene image generated", scene_index=i, image_path=str(out_path))

    # Saved once, after every scene succeeded. Writing per-scene would leave a
    # character half-registered if the stage died midway, and the next run would
    # then inherit a descriptor from a poem that never shipped.
    if use_bank:
        save_bank(bank)

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
