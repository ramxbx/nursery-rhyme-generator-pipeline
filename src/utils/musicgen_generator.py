"""MusicGen-based background music generation (GPT-24).

Real generated instrumental music instead of the procedural synth in
music_generator.py - meaningfully better quality, at the cost of real
latency on CPU (~10x realtime for musicgen-small). Runs CPU-only,
consistent with keeping the GPU reserved for the image stage; invoked
from within animate_agent's own orchestration subprocess, so the model is
released when that process exits.
"""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
os.environ.setdefault("HF_HOME", str(ROOT / "models" / "hf_cache"))

import numpy as np
import torch
from transformers import AutoProcessor, MusicgenForConditionalGeneration

DEFAULT_PROMPT = ("cheerful children's nursery rhyme background music, xylophone and ukulele, "
                   "playful, gentle, major key")
TOKENS_PER_SECOND = 50  # MusicGen/EnCodec frame rate

# MusicGen's decoder has max_position_embeddings=2048, i.e. a hard ceiling of
# 40.96s of audio. Asking for more does not degrade gracefully - it raises
# "IndexError: index out of range in self" from the positional embedding
# lookup, which killed the animate stage on the first 16-scene video (54.9s).
# Four-scene runs never hit it because they were only ~16s long.
MAX_GENERATION_S = 38.0     # under the 40.96s ceiling, with margin

# When a video outruns one generation, the clip is looped rather than
# generated in independent chunks. Separate MusicGen calls come back in
# unrelated keys and tempos, so concatenating them sounds like a broken radio;
# one clip looped stays musically coherent, and this is background bed under a
# nursery rhyme, where a repeat is unremarkable.
LOOP_CROSSFADE_S = 1.0


def _loop_to_length(clip: np.ndarray, sample_rate: int, duration_s: float) -> np.ndarray:
    """Repeat a clip up to duration_s, crossfading each joint so the seam is
    not audible as a click or an abrupt restart."""
    target = int(duration_s * sample_rate)
    if len(clip) >= target:
        return clip[:target]

    xf = min(int(LOOP_CROSSFADE_S * sample_rate), len(clip) // 4)
    if xf < 1:
        reps = int(np.ceil(target / len(clip)))
        return np.tile(clip, reps)[:target]

    ramp = np.linspace(0.0, 1.0, xf, dtype=np.float32)
    out = clip.copy()
    while len(out) < target:
        joined = out[-xf:] * (1.0 - ramp) + clip[:xf] * ramp
        out = np.concatenate([out[:-xf], joined, clip[xf:]])
    return out[:target]


def generate_background_music_musicgen(duration_s: float, prompt: str = DEFAULT_PROMPT,
                                         model_name: str = "facebook/musicgen-small") -> tuple[np.ndarray, int]:
    processor = AutoProcessor.from_pretrained(model_name)
    model = MusicgenForConditionalGeneration.from_pretrained(model_name)
    model.eval()

    generate_s = min(duration_s, MAX_GENERATION_S)
    inputs = processor(text=[prompt], padding=True, return_tensors="pt")
    max_new_tokens = max(1, int(generate_s * TOKENS_PER_SECOND))
    with torch.no_grad():
        audio_values = model.generate(**inputs, max_new_tokens=max_new_tokens)

    sample_rate = model.config.audio_encoder.sampling_rate
    audio = audio_values[0, 0].numpy().astype(np.float32)
    if duration_s > generate_s:
        audio = _loop_to_length(audio, sample_rate, duration_s)
    return audio, sample_rate
