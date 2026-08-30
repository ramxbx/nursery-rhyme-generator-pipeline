"""Generate a single image from a prompt, outside the rhyme pipeline.

The pipeline only ever renders images as part of a full rhyme run, which needs
LM Studio up and produces a whole video. This is the one-off path: same
checkpoint, scheduler, negative prompt and memory settings from
config/sd_config.yml, but a prompt typed on the command line.

    python scripts/generate_image.py "a red barn at sunrise"

Style: the config's checkpoint is a Disney-style SD1.5 fine-tune whose trigger
phrase is prepended by default (same STYLE_ANCHOR the visual agent uses).
Pass --raw to send the prompt through untouched, e.g. when testing a prompt
meant for a different look.
"""
from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.agents.visual_agent import STYLE_ANCHOR, generate_image, hires_fix
from src.config import load_config
from src.utils.sd_pipeline import build_pipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate one image with the local SD1.5 setup.")
    parser.add_argument("prompt")
    parser.add_argument("--seed", type=int, default=0,
                        help="Same seed + same prompt reproduces the same image.")
    parser.add_argument("--steps", type=int, help="Overrides sd_config steps.")
    parser.add_argument("--out", type=Path, help="Output PNG path (default: data/images/manual/).")
    parser.add_argument("--raw", action="store_true", help="Skip the style anchor prefix.")
    parser.add_argument("--no-hires", action="store_true",
                        help="Skip the 768 refine pass (faster, less detail).")
    args = parser.parse_args()

    config = load_config()
    sd_config = dict(config.sd)
    if args.steps:
        sd_config["steps"] = args.steps

    prompt = args.prompt if args.raw else f"{STYLE_ANCHOR}, {args.prompt}"
    print(f"prompt: {prompt}")

    pipe = build_pipeline(sd_config)
    image, pipe = generate_image(pipe, prompt, args.seed, sd_config)

    if sd_config.get("hires_fix", True) and not args.no_hires:
        image = hires_fix(pipe, image, prompt, args.seed, sd_config)

    out = args.out
    if out is None:
        out = (config.paths["images_dir"] / "manual" /
               f"{datetime.now():%Y%m%d-%H%M%S}-seed{args.seed}.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    image.save(out)
    print(f"saved: {out}  ({image.width}x{image.height})")


if __name__ == "__main__":
    main()
