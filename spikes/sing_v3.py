"""v3: intelligible words, v1's natural feel.

v1 stretched whole syllables 3-5x, which smeared the plosives and made the
words hard to make out. v2 fixed that by looping the vowel, which nailed the
pitch and sounded like a machine.

v3 takes the useful half of each:
  * Piper synthesizes at length_scale=2.0, so the phonemes are GENERATED long
    rather than stretched long. Measured: vowels grow 2.1x, consonants 1.7x,
    and nothing is smeared because nothing was time-stretched.
  * Consonants then pass through with no time change at all - only pitch
    shifted. Plosives stay crisp, so the words stay intelligible.
  * Only the vowel is time-stretched, and now it only needs ~4x instead of
    ~10x, which Rubber Band does cleanly.
  * No looping anywhere, so each vowel keeps its own natural pitch movement -
    that variation is what made v1 sound human.
"""
import sys, tempfile
from pathlib import Path
import numpy as np, soundfile as sf

sys.path.insert(0, r"C:\Users\abhiv\Documents\Projects\LM_CLAUDE_WS")
sys.path.insert(0, str(Path(__file__).parent))

from piper import PiperVoice
from piper.config import SynthesisConfig
from src.config import load_config
from src.utils.bark_tts import estimate_f0
from src.utils.ffmpeg_helper import rubberband_pitch_shift
import sing_spike as S

OUT = Path(__file__).parent
LENGTH_SCALE = 1.4


def rb(seg, sr, pitch_ratio, tempo_ratio):
    if len(seg) < 64:
        return seg
    with tempfile.TemporaryDirectory() as td:
        a, b = Path(td) / "a.wav", Path(td) / "b.wav"
        sf.write(a, seg, sr, subtype="FLOAT")
        rubberband_pitch_shift(a, b, pitch_ratio=float(np.clip(pitch_ratio, 0.5, 2.0)),
                                tempo_ratio=float(np.clip(tempo_ratio, 0.15, 4.0)))
        out, _ = sf.read(b, dtype="float32")
    return out


def render_syllable(audio, sr, part, semitone, target_s, ref_f0):
    s, vs, ve, e = part
    onset, nucleus, coda = audio[s:vs], audio[vs:ve], audio[ve:e]
    own = S.robust_f0(nucleus if len(nucleus) > 256 else audio[s:e], sr, ref_f0)
    note_hz = S.TONIC_HZ * (2 ** (semitone / 12.0))
    pr = note_hz / own if own > 0 else 1.0

    fixed_s = (len(onset) + len(coda)) / sr
    nucleus_target = max(target_s - fixed_s, 0.10)
    cur = len(nucleus) / sr if len(nucleus) else 0.0

    parts = []
    if len(onset):
        parts.append(rb(onset, sr, pr, 1.0))          # crisp: pitch only
    if len(nucleus):
        parts.append(rb(nucleus, sr, pr, cur / nucleus_target if nucleus_target > 0 else 1.0))
    if len(coda):
        parts.append(rb(coda, sr, pr, 1.0))           # crisp: pitch only
    return np.concatenate(parts) if parts else audio[s:e]


def main():
    line = "Twinkle twinkle little star"
    cfg = load_config().tts
    voice = PiperVoice.load(S.ROOT / cfg["voice_onnx"], config_path=S.ROOT / cfg["voice_config"],
                             include_alignments=True)
    chunk = next(iter(voice.synthesize(line, syn_config=SynthesisConfig(length_scale=LENGTH_SCALE),
                                        include_alignments=True)))
    sr, audio, al = chunk.sample_rate, chunk.audio_float_array, chunk.phoneme_alignments
    parts = S.syllable_parts(al)
    ref = estimate_f0(audio, sr)
    n = min(len(parts), len(S.MELODY_SEMITONES))
    print(f"base (length_scale={LENGTH_SCALE}): {len(audio)/sr:.2f}s, {len(parts)} syllables, ref F0 {ref:.1f} Hz")

    pieces = []
    for i in range(n):
        s, vs, ve, e = parts[i]
        tgt = S.MELODY_BEATS[i] * S.SEC_PER_BEAT
        stretch = (tgt - (vs - s + e - ve) / sr) / max((ve - vs) / sr, 1e-6)
        print(f"  syl {i+1}: cons {(vs-s+e-ve)/sr*1000:5.0f}ms  vowel {(ve-vs)/sr*1000:5.0f}ms  -> vowel x{stretch:.1f}")
        pieces.append(render_syllable(audio, sr, parts[i], S.MELODY_SEMITONES[i], tgt, ref))

    xf = int(sr * 0.012)
    out = pieces[0]
    for p in pieces[1:]:
        if len(out) > xf and len(p) > xf:
            ramp = np.linspace(0, 1, xf, dtype=np.float32)
            out = np.concatenate([out[:-xf], out[-xf:] * (1 - ramp) + p[:xf] * ramp, p[xf:]])
        else:
            out = np.concatenate([out, p])
    peak = np.abs(out).max()
    if peak > 0:
        out = (out / peak * 0.9).astype(np.float32)

    sf.write(OUT / "sung_v3.wav", out, sr, subtype="PCM_16")
    S.polish(OUT / "sung_v3.wav", OUT / "sung_v3_cleaned.wav")
    print(f"sung: {len(out)/sr:.2f}s (score {sum(S.MELODY_BEATS[:n])*S.SEC_PER_BEAT:.2f}s)")


if __name__ == "__main__":
    main()
