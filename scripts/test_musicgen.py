import os
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
os.environ.setdefault("HF_HOME", str(ROOT / "models" / "hf_cache"))

import torch
import soundfile as sf
from transformers import MusicgenForConditionalGeneration, AutoProcessor

print("== Loading musicgen-small ==")
start = time.perf_counter()
processor = AutoProcessor.from_pretrained("facebook/musicgen-small")
model = MusicgenForConditionalGeneration.from_pretrained("facebook/musicgen-small")
print(f"Loaded in {time.perf_counter() - start:.1f}s")

PROMPT = "cheerful children's nursery rhyme background music, xylophone and ukulele, playful, major key, gentle"

inputs = processor(text=[PROMPT], padding=True, return_tensors="pt")

sample_rate = model.config.audio_encoder.sampling_rate
duration_s = 12
tokens_per_second = 50  # musicgen's EnCodec frame rate
max_new_tokens = duration_s * tokens_per_second

print(f"== Generating {duration_s}s of audio ({max_new_tokens} tokens) ==")
start = time.perf_counter()
with torch.no_grad():
    audio_values = model.generate(**inputs, max_new_tokens=max_new_tokens)
elapsed = time.perf_counter() - start
print(f"Generated in {elapsed:.1f}s")

out_path = ROOT / "data" / "output" / "musicgen_test.wav"
sf.write(out_path, audio_values[0, 0].numpy(), sample_rate)
print(f"Wrote {out_path}, sample_rate={sample_rate}")
