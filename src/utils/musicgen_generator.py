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


def generate_background_music_musicgen(duration_s: float, prompt: str = DEFAULT_PROMPT,
                                         model_name: str = "facebook/musicgen-small") -> tuple[np.ndarray, int]:
    processor = AutoProcessor.from_pretrained(model_name)
    model = MusicgenForConditionalGeneration.from_pretrained(model_name)
    model.eval()

    inputs = processor(text=[prompt], padding=True, return_tensors="pt")
    max_new_tokens = max(1, int(duration_s * TOKENS_PER_SECOND))
    with torch.no_grad():
        audio_values = model.generate(**inputs, max_new_tokens=max_new_tokens)

    sample_rate = model.config.audio_encoder.sampling_rate
    audio = audio_values[0, 0].numpy().astype(np.float32)
    return audio, sample_rate
