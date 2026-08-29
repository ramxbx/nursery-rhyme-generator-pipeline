"""Rebuild of the FIRST sung version - the one that sounded best.

Algorithm as it stood then:
  * whole-syllable time-stretch onto the note (consonants stretch too)
  * per-syllable F0 as the pitch reference
  * NO vowel looping, NO octave guard, NO polish chain

Everything added after this made the measurements better and the sound worse.
Writes to its own filename so nothing overwrites it again.
"""
import sys
from pathlib import Path
import numpy as np, soundfile as sf

sys.path.insert(0, r"C:\Users\abhiv\Documents\Projects\LM_CLAUDE_WS")
sys.path.insert(0, str(Path(__file__).parent))

from piper import PiperVoice
from src.config import load_config
from src.utils.bark_tts import estimate_f0
import sing_spike as S

OUT = Path(__file__).parent


def main():
    line = "Twinkle twinkle little star"
    cfg = load_config().tts
    voice = PiperVoice.load(S.ROOT / cfg["voice_onnx"], config_path=S.ROOT / cfg["voice_config"],
                             include_alignments=True)
    chunk = next(iter(voice.synthesize(line, include_alignments=True)))
    sr, audio, al = chunk.sample_rate, chunk.audio_float_array, chunk.phoneme_alignments

    spans = S.syllable_spans(al)                 # whole syllables, not vowel-split
    ref_f0 = estimate_f0(audio, sr)
    n = min(len(spans), len(S.MELODY_SEMITONES))
    print(f"syllables: {len(spans)}  ref F0 {ref_f0:.1f} Hz")

    pieces = []
    for i in range(n):
        s, e = spans[i]
        target_s = S.MELODY_BEATS[i] * S.SEC_PER_BEAT
        pieces.append(S.warp(audio[s:e], sr, S.MELODY_SEMITONES[i], target_s, ref_f0))

    xf = int(sr * 0.012)
    out = pieces[0]
    for p in pieces[1:]:
        if len(out) > xf and len(p) > xf:
            ramp = np.linspace(0, 1, xf, dtype=np.float32)
            tail = out[-xf:] * (1 - ramp) + p[:xf] * ramp
            out = np.concatenate([out[:-xf], tail, p[xf:]])
        else:
            out = np.concatenate([out, p])

    peak = np.abs(out).max()
    if peak > 0:
        out = (out / peak * 0.9).astype(np.float32)
    sf.write(OUT / "sung_v1_original.wav", out, sr, subtype="PCM_16")
    print(f"sung: {len(out)/sr:.2f}s -> {OUT/'sung_v1_original.wav'}")


if __name__ == "__main__":
    main()
