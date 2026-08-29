"""Render the full 16-line poem three ways so the versions can be compared at
length rather than on one line.

v1  whole syllable time-stretched onto the note. Natural feel, smeared plosives.
v2  vowel looped by pitch period. Exact pitch, machine-like tone.
v3  consonants never stretched; vowels + sonorants absorb the note, no looping.

Each is written raw and polished. Nothing here imports the mutated spike module -
the three are reproduced independently so v2 is what was actually heard.
"""
import subprocess, sys, tempfile
from pathlib import Path
import numpy as np, soundfile as sf

sys.path.insert(0, r"C:\Users\abhiv\Documents\Projects\LM_CLAUDE_WS")

from piper import PiperVoice
from piper.config import SynthesisConfig
from src.config import load_config
from src.utils.bark_tts import estimate_f0
from src.utils.ffmpeg_helper import ffmpeg_path, rubberband_pitch_shift

ROOT = Path(r"C:\Users\abhiv\Documents\Projects\LM_CLAUDE_WS")
OUT = Path(__file__).parent
POEM = ROOT / "data" / "a_farm_animal.txt"

# Twinkle, full 14-note tune, cycled so longer lines keep progressing through it.
MELODY = [0, 0, 7, 7, 9, 9, 7, 5, 5, 4, 4, 2, 2, 0]
BEATS = [1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 2]
TONIC_HZ = 262.0
SEC_PER_BEAT = 60.0 / 108.0
GAP_S = 0.35

VOWELS = set("aeiouyæɑɐɒɔəɛɜɪʊʌœøɨʉɯ")
SONORANTS = set("lrɹmnŋwjʎɰ")
SKIP = {"^", "$", "ˈ", "ˌ", "ː", " "}


def rb(seg, sr, pitch_ratio, tempo_ratio, lo=0.15):
    if len(seg) < 64:
        return seg
    with tempfile.TemporaryDirectory() as td:
        a, b = Path(td) / "a.wav", Path(td) / "b.wav"
        sf.write(a, seg, sr, subtype="FLOAT")
        rubberband_pitch_shift(a, b, pitch_ratio=float(np.clip(pitch_ratio, 0.5, 2.0)),
                                tempo_ratio=float(np.clip(tempo_ratio, lo, 4.0)))
        out, _ = sf.read(b, dtype="float32")
    return out


def robust_f0(seg, sr, reference):
    if len(seg) < 256:
        return reference
    frame = int(min(2048, max(512, len(seg) // 2)))
    hop = max(frame // 4, 128)
    f0s = []
    for i in range(0, max(1, len(seg) - frame), hop):
        w = seg[i:i + frame].astype(np.float64)
        if len(w) < frame or np.abs(w).max() < 0.02:
            continue
        w = w - w.mean()
        corr = np.correlate(w, w, mode="full")[len(w) - 1:]
        rising = np.where(np.diff(corr) > 0)[0]
        if len(rising) == 0:
            continue
        peak = int(np.argmax(corr[rising[0]:]) + rising[0])
        if peak > 0 and 70 < sr / peak < 500:
            f0s.append(sr / peak)
    if not f0s:
        return reference
    est = float(np.median(f0s))
    if reference > 0 and abs(12 * np.log2(est / reference)) > 7.0:
        return reference
    return est


def marks_of(alignments, sustain_set):
    out, s = [], 0
    for a in alignments:
        n = int(a.num_samples)
        sustainable = a.phoneme not in SKIP and any(c in sustain_set for c in a.phoneme)
        out.append((s, s + n, sustainable))
        s += n
    return out, s


def spans_of(marks, total):
    idx = [i for i, m in enumerate(marks) if m[2]]
    if not idx:
        return [(0, total)]
    nuclei, cur = [], [idx[0]]
    for i in idx[1:]:
        if i == cur[-1] + 1:
            cur.append(i)
        else:
            nuclei.append(cur); cur = [i]
    nuclei.append(cur)
    bounds = [0]
    for a, b in zip(nuclei, nuclei[1:]):
        bounds.append((marks[a[-1]][1] + marks[b[0]][0]) // 2)
    bounds.append(total)
    return list(zip(bounds, bounds[1:])), nuclei


def parts_of(marks, spans):
    parts = []
    for s, e in spans:
        inner = [m for m in marks if m[0] >= s and m[1] <= e and m[2]]
        parts.append((s, inner[0][0], inner[-1][1], e) if inner else (s, s, e, e))
    return parts


def sustain_loop(nucleus, sr, note_hz, target_s, own_f0):
    if own_f0 <= 0 or len(nucleus) < 64:
        return None
    ratio = note_hz / own_f0
    n_out = max(int(len(nucleus) / ratio), 64)
    tuned = np.interp(np.linspace(0, len(nucleus) - 1, n_out),
                      np.linspace(0, len(nucleus) - 1, len(nucleus)), nucleus).astype(np.float32)
    period = max(int(sr / note_hz), 8)
    grain_len = period * 2
    if len(tuned) < grain_len + period:
        grain_len = max(len(tuned) - period, period)
    if grain_len < period:
        return None
    mid = len(tuned) // 2
    st = max(0, min(mid - grain_len // 2, len(tuned) - grain_len))
    grain = tuned[st:st + grain_len]
    target_n = int(target_s * sr)
    xf = period
    ramp = np.linspace(0, 1, xf, dtype=np.float32)
    out = tuned[:st + grain_len] if st else grain.copy()
    while len(out) < target_n:
        out = np.concatenate([out[:-xf], out[-xf:] * (1 - ramp) + grain[:xf] * ramp, grain[xf:]])
    return out[:target_n]


def synth(voice, text, length_scale):
    ch = next(iter(voice.synthesize(text, syn_config=SynthesisConfig(length_scale=length_scale),
                                     include_alignments=True)))
    return ch.sample_rate, ch.audio_float_array, ch.phoneme_alignments


def render_line(voice, text, note_offset, version):
    length_scale = 1.4 if version == "v3" else 1.0
    sustain_set = (VOWELS | SONORANTS) if version == "v3" else VOWELS
    sr, audio, al = synth(voice, text, length_scale)
    if not al:
        return sr, audio, note_offset
    marks, total = marks_of(al, sustain_set)
    spans, _ = spans_of(marks, total)
    parts = parts_of(marks, spans)
    ref = estimate_f0(audio, sr)

    pieces = []
    for i, (s, vs, ve, e) in enumerate(parts):
        semi = MELODY[(note_offset + i) % len(MELODY)]
        beat = BEATS[(note_offset + i) % len(BEATS)]
        target_s = beat * SEC_PER_BEAT
        note_hz = TONIC_HZ * (2 ** (semi / 12.0))

        if version == "v1":
            own = robust_f0(audio[s:e], sr, ref)
            pr = note_hz / own if own > 0 else 1.0
            cur = (e - s) / sr
            pieces.append(rb(audio[s:e], sr, pr, cur / target_s))
            continue

        onset, nucleus, coda = audio[s:vs], audio[vs:ve], audio[ve:e]
        own = robust_f0(nucleus if len(nucleus) > 256 else audio[s:e], sr, ref)
        pr = note_hz / own if own > 0 else 1.0
        nuc_target = max(target_s - (len(onset) + len(coda)) / sr, 0.10)
        seq = []
        if len(onset):
            seq.append(rb(onset, sr, pr, 1.0))
        if len(nucleus):
            if version == "v2":
                held = sustain_loop(nucleus, sr, note_hz, nuc_target, own)
                seq.append(held if held is not None else rb(nucleus, sr, pr, len(nucleus) / sr / nuc_target))
            else:
                seq.append(rb(nucleus, sr, pr, len(nucleus) / sr / nuc_target))
        if len(coda):
            seq.append(rb(coda, sr, pr, 1.0))
        pieces.append(np.concatenate(seq) if seq else audio[s:e])

    xf = int(sr * 0.012)
    out = pieces[0]
    for p in pieces[1:]:
        if len(out) > xf and len(p) > xf:
            ramp = np.linspace(0, 1, xf, dtype=np.float32)
            out = np.concatenate([out[:-xf], out[-xf:] * (1 - ramp) + p[:xf] * ramp, p[xf:]])
        else:
            out = np.concatenate([out, p])
    return sr, out, note_offset + len(parts)


def polish(src, dst):
    chain = "highpass=f=80,afftdn=nr=10:nf=-32,deesser=i=0.4,lowpass=f=10000,dynaudnorm=f=200:g=5"
    r = subprocess.run([ffmpeg_path(), "-y", "-v", "error", "-i", str(src), "-af", chain, str(dst)],
                       capture_output=True, text=True)
    if r.returncode != 0:
        chain = "highpass=f=80,afftdn=nr=10:nf=-32,lowpass=f=10000,dynaudnorm=f=200:g=5"
        subprocess.run([ffmpeg_path(), "-y", "-v", "error", "-i", str(src), "-af", chain, str(dst)],
                       capture_output=True, text=True)


def main():
    lines = [l.strip() for l in POEM.read_text(encoding="utf-8").splitlines() if l.strip()]
    cfg = load_config().tts
    voice = PiperVoice.load(ROOT / cfg["voice_onnx"], config_path=ROOT / cfg["voice_config"],
                             include_alignments=True)
    print(f"{len(lines)} lines\n")
    for version in ("v1", "v2", "v3"):
        offset, out, sr = 0, [], 22050
        for text in lines:
            sr, seg, offset = render_line(voice, text, offset, version)
            out.append(seg)
            out.append(np.zeros(int(sr * GAP_S), dtype=np.float32))
        full = np.concatenate(out)
        peak = np.abs(full).max()
        if peak > 0:
            full = (full / peak * 0.9).astype(np.float32)
        raw = OUT / f"poem_{version}.wav"
        sf.write(raw, full, sr, subtype="PCM_16")
        polish(raw, OUT / f"poem_{version}_cleaned.wav")
        print(f"{version}: {len(full)/sr:.1f}s -> poem_{version}_cleaned.wav")


if __name__ == "__main__":
    main()
