"""Spike: score-driven singing.

Renders a line with Piper, uses its phoneme alignments to find exact syllable
boundaries, then warps each syllable to a note of a KNOWN melody (pitch +
duration). This is the opposite of Bark: melody is an input, not an output.
"""
import sys, tempfile
from pathlib import Path
import numpy as np, soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[0]))
sys.path.insert(0, r"C:\Users\abhiv\Documents\Projects\LM_CLAUDE_WS")

from piper import PiperVoice
from src.config import load_config
from src.utils.ffmpeg_helper import rubberband_pitch_shift
from src.utils.bark_tts import estimate_f0

ROOT = Path(r"C:\Users\abhiv\Documents\Projects\LM_CLAUDE_WS")
OUT = Path(__file__).parent

# Twinkle Twinkle, first phrase: C C G G A A G, last note held.
MELODY_SEMITONES = [0, 0, 7, 7, 9, 9, 7]
MELODY_BEATS     = [1, 1, 1, 1, 1, 1, 2]
TONIC_HZ = 262.0          # C4 - comfortable children's register
BPM = 108.0
SEC_PER_BEAT = 60.0 / BPM

VOWELS = set("aeiouyæɑɐɒɔəɛɜɪʊʌœøyɨʉɯiu")
SKIP = {"^", "$", "ˈ", "ˌ", "ː", " "}

# Phonemes a singer can hold a note on. Vowels obviously, but also the
# sonorants - you can sustain the "l" in "twinkle" or the "r" in "star" as long
# as you like, and singers do. Treating them as unstretchable consonants (the
# first attempt) left only the tiny vowel to absorb the whole note, forcing 7-9x
# stretches on it while long consonant runs ate the note's time budget.
SONORANTS = set("lrɹmnŋwjʎɰ")


def is_sustainable(phoneme: str) -> bool:
    if phoneme in SKIP:
        return False
    return any(c in VOWELS or c in SONORANTS for c in phoneme)


def syllable_parts(alignments):
    """Like syllable_spans, but also reports where the vowel sits inside each
    syllable: (start, vowel_start, vowel_end, end) in samples.

    Needed because a sung note must be lengthened on the VOWEL only. Stretching
    a whole syllable stretches its consonants too, which turns the 's' in
    "star" into a 313ms hiss - measured, and the single most audible artifact
    in the first version of this spike."""
    marks, sample = [], 0
    for a in alignments:
        n = int(a.num_samples)
        marks.append((sample, sample + n, is_sustainable(a.phoneme)))
        sample += n
    spans = syllable_spans(alignments)
    parts = []
    for s, e in spans:
        inner = [m for m in marks if m[0] >= s and m[1] <= e and m[2]]
        if inner:
            parts.append((s, inner[0][0], inner[-1][1], e))
        else:
            parts.append((s, s, e, e))  # no vowel found - treat all as nucleus
    return parts


def syllable_spans(alignments):
    """Group phoneme alignments into syllables at midpoints between vowel runs."""
    spans, sample = [], 0
    marks = []          # (start_sample, end_sample, is_vowel)
    for a in alignments:
        n = int(a.num_samples)
        is_vowel = any(ch in VOWELS for ch in a.phoneme) and a.phoneme not in SKIP
        marks.append((sample, sample + n, is_vowel, a.phoneme))
        sample += n
    total = sample

    vowel_idx = [i for i, m in enumerate(marks) if m[2]]
    if not vowel_idx:
        return [(0, total)]
    # merge adjacent vowel phonemes into one nucleus
    nuclei, cur = [], [vowel_idx[0]]
    for i in vowel_idx[1:]:
        if i == cur[-1] + 1:
            cur.append(i)
        else:
            nuclei.append(cur); cur = [i]
    nuclei.append(cur)

    bounds = [0]
    for a, b in zip(nuclei, nuclei[1:]):
        gap_start = marks[a[-1]][1]
        gap_end = marks[b[0]][0]
        bounds.append((gap_start + gap_end) // 2)
    bounds.append(total)
    return list(zip(bounds, bounds[1:]))


def warp(seg, sr, semitone, target_s, ref_f0):
    """Pitch-shift a syllable onto its note and stretch it to the note length.

    The reference pitch is measured on THIS syllable, not on the whole line.
    Using a line-level reference leaves each syllable's own speech intonation
    riding on top of the melody, which put notes up to 5 semitones off."""
    cur_s = len(seg) / sr
    if cur_s < 0.01:
        return seg
    own_f0 = estimate_f0(seg, sr)
    # Short/unvoiced syllables give no usable estimate; fall back to the line.
    src_f0 = own_f0 if own_f0 > 0 else ref_f0
    note_hz = TONIC_HZ * (2 ** (semitone / 12.0))
    pitch_ratio = note_hz / src_f0 if src_f0 > 0 else 1.0
    pitch_ratio = float(np.clip(pitch_ratio, 0.5, 2.0))
    # Sung notes are long and spoken syllables are short, so stretches of 5x
    # are routine here - clipping at 4x silently left notes short.
    tempo_ratio = float(np.clip(cur_s / target_s, 0.1, 4.0))
    with tempfile.TemporaryDirectory() as td:
        a, b = Path(td) / "a.wav", Path(td) / "b.wav"
        sf.write(a, seg, sr, subtype="FLOAT")
        rubberband_pitch_shift(a, b, pitch_ratio=pitch_ratio, tempo_ratio=tempo_ratio)
        out, _ = sf.read(b, dtype="float32")
    return out


def robust_f0(seg, sr, reference):
    """F0 of a short segment, guarded against octave errors.

    bark_tts.estimate_f0 analyses in 2048-sample (93ms) frames, but spoken
    vowels here run 23-81ms - too short to fill even one frame, so it returns
    a subharmonic and the note lands an octave or a twelfth low. Measured: three
    of seven notes were 10-19 semitones out from exactly this.

    Two defences: analyse in frames small enough to fit, and reject any estimate
    implausibly far from the line's own median pitch, since speech intonation
    does not swing by an octave within one line."""
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
        if peak > 0:
            f = sr / peak
            if 70 < f < 500:
                f0s.append(f)
    if not f0s:
        return reference
    est = float(np.median(f0s))
    # Reject octave/subharmonic slips rather than singing them.
    if reference > 0 and abs(12 * np.log2(est / reference)) > 7.0:
        return reference
    return est


def sustain_vowel(nucleus, sr, note_hz, target_s, own_f0):
    """Hold a vowel on an exact pitch for an exact duration by looping its own
    pitch periods.

    Time-stretching cannot get here: spoken vowels run 23-81ms and a sung note
    wants ~500ms, a 10-20x stretch that Rubber Band smears into mush. Looping
    sidesteps it entirely and wins twice more - resampling the grain sets the
    pitch EXACTLY (no residual speech glide, which was the main source of the
    1.69 semitone error), and every repetition is identical, so the note is
    dead steady instead of wobbling.
    """
    if own_f0 <= 0 or len(nucleus) < 64:
        return None
    # Resample so one period equals the target note's period. This both
    # corrects the pitch and defines the loop length.
    ratio = note_hz / own_f0
    n_out = max(int(len(nucleus) / ratio), 64)
    src_t = np.linspace(0, len(nucleus) - 1, num=len(nucleus))
    dst_t = np.linspace(0, len(nucleus) - 1, num=n_out)
    tuned = np.interp(dst_t, src_t, nucleus).astype(np.float32)

    period = max(int(sr / note_hz), 8)
    # The loop advances by (grain_periods - 1) periods per repeat, so anything
    # above 2 makes the signal repeat at a SUBMULTIPLE of the note and adds an
    # audible low buzz under it - with 4 the measured pitch came back an octave
    # and a fifth low. At 2 the advance is exactly one period, so the only
    # periodicity present is the fundamental itself.
    grain_periods = 2
    grain_len = period * grain_periods
    if len(tuned) < grain_len + period:
        grain_len = max(len(tuned) - period, period)
    if grain_len < period:
        return None
    # Take the grain from the middle, where the vowel is most steady.
    mid = len(tuned) // 2
    start = max(0, min(mid - grain_len // 2, len(tuned) - grain_len))
    grain = tuned[start:start + grain_len]

    target_n = int(target_s * sr)
    xf = period                      # crossfade exactly one period: phase-aligned
    ramp = np.linspace(0, 1, xf, dtype=np.float32)
    out = tuned[:start + grain_len] if start else grain.copy()
    while len(out) < target_n:
        head = out[-xf:] * (1 - ramp) + grain[:xf] * ramp
        out = np.concatenate([out[:-xf], head, grain[xf:]])
    return out[:target_n]


def shift_only(seg, sr, pitch_ratio):
    """Pitch-shift with no time change - used for consonants, which keep their
    natural length."""
    if len(seg) < 64:
        return seg
    with tempfile.TemporaryDirectory() as td:
        a, b = Path(td) / "a.wav", Path(td) / "b.wav"
        sf.write(a, seg, sr, subtype="FLOAT")
        rubberband_pitch_shift(a, b, pitch_ratio=pitch_ratio, tempo_ratio=1.0)
        out, _ = sf.read(b, dtype="float32")
    return out


def warp_syllable(audio, sr, part, semitone, target_s, ref_f0):
    """Render one syllable onto its note, lengthening only the vowel.

    Consonants pass through at natural speed (pitch-shifted only) so sibilants
    and stops stay crisp; the vowel absorbs whatever time the note needs."""
    s, vs, ve, e = part
    onset, nucleus, coda = audio[s:vs], audio[vs:ve], audio[ve:e]
    own = robust_f0(nucleus, sr, ref_f0)
    note_hz = TONIC_HZ * (2 ** (semitone / 12.0))
    pitch_ratio = float(np.clip(note_hz / own if own > 0 else 1.0, 0.5, 2.0))

    fixed_s = (len(onset) + len(coda)) / sr
    # The vowel takes the note length minus the consonants, but never less than
    # a floor - otherwise a consonant-heavy syllable would squeeze it to nothing.
    nucleus_target = max(target_s - fixed_s, 0.08)
    cur = len(nucleus) / sr
    parts = []
    if len(onset):
        parts.append(shift_only(onset, sr, pitch_ratio))
    if len(nucleus):
        held = sustain_vowel(nucleus, sr, note_hz, nucleus_target, own)
        # Fall back to time-stretching when the vowel is too short or unvoiced
        # to find a loopable period in.
        parts.append(held if held is not None
                     else warp(nucleus, sr, semitone, nucleus_target, own or ref_f0))
    if len(coda):
        parts.append(shift_only(coda, sr, pitch_ratio))
    return np.concatenate(parts) if parts else audio[s:e]


def polish(in_path, out_path):
    """Final cleanup pass: everything here is time-invariant, so the rhythm the
    score just established is left exactly as it is.

    * highpass 80Hz - rumble the pitch shift can introduce below the voice
    * afftdn - mild spectral denoise for the smear large stretches leave behind
    * deesser - tames sibilants; vowel-only stretching already stops them being
      lengthened, this catches what remains
    * lowpass 10kHz - above the voice, nothing but stretch artifacts live here
    """
    import subprocess
    from src.utils.ffmpeg_helper import ffmpeg_path
    chain = "highpass=f=80,afftdn=nr=10:nf=-32,deesser=i=0.4,lowpass=f=10000,dynaudnorm=f=200:g=5"
    r = subprocess.run([ffmpeg_path(), "-y", "-v", "error", "-i", str(in_path),
                        "-af", chain, str(out_path)], capture_output=True, text=True)
    if r.returncode != 0:
        # deesser is not in every ffmpeg build - retry without it rather than fail
        chain = "highpass=f=80,afftdn=nr=10:nf=-32,lowpass=f=10000,dynaudnorm=f=200:g=5"
        r = subprocess.run([ffmpeg_path(), "-y", "-v", "error", "-i", str(in_path),
                            "-af", chain, str(out_path)], capture_output=True, text=True)
        print("  (deesser unavailable, used fallback chain)")
    if r.returncode != 0:
        print("  polish FAILED:", r.stderr[-300:])
    return out_path


def main():
    line = "Twinkle twinkle little star"
    cfg = load_config().tts
    voice = PiperVoice.load(ROOT / cfg["voice_onnx"], config_path=ROOT / cfg["voice_config"],
                             include_alignments=True)
    chunk = next(iter(voice.synthesize(line, include_alignments=True)))
    sr = chunk.sample_rate
    audio = chunk.audio_float_array
    al = chunk.phoneme_alignments
    assert al, "no alignments"

    parts = syllable_parts(al)
    print(f"line: {line!r}")
    print(f"syllables found: {len(parts)}  (melody wants {len(MELODY_SEMITONES)})")
    for i, (s, vs, ve, e) in enumerate(parts):
        print(f"  syl {i+1}: {(e-s)/sr*1000:6.1f} ms  (vowel {(ve-vs)/sr*1000:5.1f} ms)")

    n = min(len(parts), len(MELODY_SEMITONES))
    ref_f0 = estimate_f0(audio, sr)
    print(f"speech median F0: {ref_f0:.1f} Hz -> tonic {TONIC_HZ} Hz")

    pieces = []
    for i in range(n):
        target_s = MELODY_BEATS[i] * SEC_PER_BEAT
        pieces.append(warp_syllable(audio, sr, parts[i], MELODY_SEMITONES[i], target_s, ref_f0))

    # short crossfades so syllables join legato instead of butting together
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

    sf.write(OUT / "sung_score_driven.wav", out, sr, subtype="PCM_16")
    polish(OUT / "sung_score_driven.wav", OUT / "sung_cleaned.wav")
    sf.write(OUT / "spoken_original.wav", audio, sr, subtype="PCM_16")
    print(f"\nspoken : {len(audio)/sr:.2f}s")
    print(f"sung   : {len(out)/sr:.2f}s  (score says {sum(MELODY_BEATS[:n])*SEC_PER_BEAT:.2f}s)")
    print(f"wrote {OUT/'sung_score_driven.wav'}")


if __name__ == "__main__":
    main()
