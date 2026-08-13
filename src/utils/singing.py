"""Lightweight 'sing-song' melodic pitch-contour shaping for narration.

Not true singing-voice synthesis - that needs a note-aligned score and a
dedicated model like DiffSinger (a much bigger lift, flagged as a stretch
goal in GPT-12). This applies a pitch contour across a line's natural
word-segments, giving spoken narration a melodic, sung-like cadence
without any new model dependency.

GPT-23 fix: the contour used to be an arbitrary wandering pattern with no
relationship to any real tune, which is why it didn't sound "in tune" -
melodically varied, but not actually music. Now defaults to the real
"Twinkle Twinkle Little Star" melody (semitone offsets from the tonic),
which is also the tune "Baa Baa Black Sheep" and the ABC Song share - the
most common melody in the classic-nursery-rhyme repertoire. A running
offset can be threaded across calls (see audio_agent.py) so the melody
continues progressing through a whole poem instead of restarting at note
one on every line.

GPT-30 fix: segments used to be pitch-shifted via librosa's phase-vocoder
(librosa.effects.pitch_shift), which introduced audible metallic/robotic
artifacts on speech, especially at this contour's larger shifts (up to 9
semitones). Switched to ffmpeg's rubberband filter (Rubber Band Library,
formant-preserving, designed for vocals) - already compiled into this
project's ffmpeg build, no new dependency.
"""
from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf

from src.utils.ffmpeg_helper import rubberband_pitch_shift

# "Twinkle Twinkle Little Star" / "Baa Baa Black Sheep" / ABC Song melody,
# semitone offsets from the tonic: C C G G A A G | F F E E D D C
DEFAULT_CONTOUR_SEMITONES = [0, 0, 7, 7, 9, 9, 7, 5, 5, 4, 4, 2, 2, 0]
# Relative note lengths in beats, parallel to the semitones above. The tune is
# six quarter notes resolving onto a held half note, twice - "twin-kle twin-kle
# lit-tle STAAAR". Singing is pitch AND duration; applying the pitches while
# leaving every syllable at its original speech length is a large part of why
# the result read as speech-with-pitch-changes rather than singing, since the
# phrase never resolves onto a held note.
DEFAULT_NOTE_BEATS = [1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 2]
# Bounds on how far a syllable may be time-stretched to reach its note length.
# Rubber Band degrades audibly past roughly a factor of two, and over-stretching
# a short consonant-heavy syllable sounds worse than letting the rhythm be
# slightly loose.
MIN_STRETCH = 0.6
MAX_STRETCH = 2.2
CROSSFADE_S = 0.03
MIN_SEGMENT_S = 0.05


ENERGY_FRAME_S = 0.02
ENERGY_SMOOTH_FRAMES = 5


def _energy_envelope(audio: np.ndarray, sr: int) -> np.ndarray:
    """Smoothed short-time RMS energy. Syllables show up as energy peaks
    (vowels) separated by valleys (consonants, gaps)."""
    frame = max(1, int(sr * ENERGY_FRAME_S))
    n_frames = max(1, len(audio) // frame)
    trimmed = audio[: n_frames * frame].reshape(n_frames, frame)
    rms = np.sqrt((trimmed.astype(np.float64) ** 2).mean(axis=1))
    if len(rms) >= ENERGY_SMOOTH_FRAMES:
        kernel = np.ones(ENERGY_SMOOTH_FRAMES) / ENERGY_SMOOTH_FRAMES
        rms = np.convolve(rms, kernel, mode="same")
    return rms


def _segment_boundaries(audio: np.ndarray, n_segments: int, sr: int) -> list[int]:
    """Sample offsets splitting speech at syllable boundaries rather than at
    even time offsets. Returns n_segments + 1 offsets, starting at 0 and
    ending at len(audio).

    Cutting a line into equal slices means a melody note can start and end
    mid-vowel, which is a large part of why the result read as "pitched-up
    speech" rather than singing - the notes simply did not line up with the
    words. Here the energy envelope's lowest points (quiet moments between
    syllables) are used as cut points, so each note lands on a syllable.

    Falls back to an even split when the envelope has no usable structure
    (very short or unusually flat audio)."""
    n_segments = max(1, n_segments)
    even = list(np.linspace(0, len(audio), n_segments + 1).astype(int))
    if n_segments == 1:
        return even

    envelope = _energy_envelope(audio, sr)
    frame = max(1, int(sr * ENERGY_FRAME_S))
    min_frames = max(1, int(MIN_SEGMENT_S / ENERGY_FRAME_S))
    if len(envelope) < n_segments * min_frames:
        return even

    # Search for the quietest frame near each evenly-spaced ideal boundary,
    # rather than taking global minima - that keeps segments roughly evenly
    # paced (so the melody keeps time) while still snapping each cut to the
    # nearest natural gap between syllables.
    boundaries = [0]
    for i in range(1, n_segments):
        ideal = int(len(envelope) * i / n_segments)
        lo = max(boundaries[-1] // frame + min_frames, ideal - min_frames)
        hi = min(len(envelope) - min_frames, ideal + min_frames)
        if lo >= hi:
            return even
        boundaries.append(int(lo + np.argmin(envelope[lo:hi])) * frame)
    boundaries.append(len(audio))

    if any(boundaries[i] >= boundaries[i + 1] for i in range(n_segments)):
        return even
    return boundaries


def _pitch_shift_whole(audio: np.ndarray, sr: int, n_steps: int) -> np.ndarray:
    """Pitch-shift an entire clip by n_steps semitones in a single pass."""
    if n_steps == 0:
        return audio.astype(np.float32)
    pitch_ratio = 2 ** (n_steps / 12)
    with tempfile.TemporaryDirectory() as td:
        in_path = Path(td) / "in.wav"
        out_path = Path(td) / "out.wav"
        sf.write(in_path, audio, sr, subtype="FLOAT")
        rubberband_pitch_shift(in_path, out_path, pitch_ratio)
        shifted, _ = sf.read(out_path, dtype="float32")
    return shifted.astype(np.float32)


def _time_stretch(audio: np.ndarray, sr: int, factor: float) -> np.ndarray:
    """Stretch a clip to `factor` times its length (factor > 1 = longer)."""
    if abs(factor - 1.0) < 0.02:
        return audio.astype(np.float32)
    with tempfile.TemporaryDirectory() as td:
        in_path = Path(td) / "in.wav"
        out_path = Path(td) / "out.wav"
        sf.write(in_path, audio, sr, subtype="FLOAT")
        rubberband_pitch_shift(in_path, out_path, pitch_ratio=1.0, tempo_ratio=1.0 / factor)
        stretched, _ = sf.read(out_path, dtype="float32")
    return stretched.astype(np.float32)


def _apply_rhythm(audio: np.ndarray, sr: int, boundaries: list[int],
                   beats: list[float]) -> tuple[np.ndarray, list[int]]:
    """Re-time syllables so their relative lengths follow the melody's rhythm,
    keeping the clip's total duration unchanged.

    Each syllable is stretched or compressed toward the share of the line its
    note is entitled to. A syllable on a half note ends up roughly twice the
    length of one on a quarter note, which is what makes a phrase resolve onto
    a held note instead of running past it at conversational speed.

    Total duration is preserved deliberately: scene timing, subtitle timing and
    the padding in audio_agent are all derived from the script's duration
    estimate, so a line that silently grew or shrank here would drift out of
    sync with everything else.

    Returns the re-timed audio and its new segment boundaries."""
    n = len(beats)
    total_beats = float(sum(beats))
    original_lengths = [boundaries[i + 1] - boundaries[i] for i in range(n)]
    total_len = sum(original_lengths)

    pieces, new_lengths = [], []
    for i in range(n):
        target = total_len * (beats[i] / total_beats)
        current = original_lengths[i]
        if current < sr * MIN_SEGMENT_S:
            pieces.append(audio[boundaries[i]:boundaries[i + 1]])
            new_lengths.append(current)
            continue
        factor = float(np.clip(target / current, MIN_STRETCH, MAX_STRETCH))
        piece = _time_stretch(audio[boundaries[i]:boundaries[i + 1]], sr, factor)
        pieces.append(piece)
        new_lengths.append(len(piece))

    retimed = np.concatenate(pieces) if pieces else audio
    new_boundaries = [0]
    for length in new_lengths:
        new_boundaries.append(new_boundaries[-1] + length)
    new_boundaries[-1] = len(retimed)
    return retimed, new_boundaries


def apply_singsong_contour(audio: np.ndarray, sr: int, n_words: int,
                            contour: list[int] | None = None, start_offset: int = 0) -> tuple[np.ndarray, int]:
    """Pitch-shift a narration clip along a melodic contour, one note per
    syllable, starting at start_offset within the contour (so a full poem
    progresses continuously through the melody across lines rather than
    restarting at note one each time - see audio_agent.py). Operates on the
    natural (unpadded) speech only - call this before adding silence
    padding. Returns (shifted_audio, next_offset).

    Works by shifting the WHOLE clip once per distinct note in the contour,
    then cross-fading between those full-length versions at syllable
    boundaries - rather than cutting the clip up and pitch-shifting each
    fragment separately.

    The fragment approach produced audible clicks and dropouts mid-line:
    every fragment went through its own Rubber Band pass, which applies its
    own windowing and edge handling and returns a slightly different length
    than it was given, so the pieces simply did not line up when rejoined.
    Longer cross-fades only smeared the seams rather than fixing them.

    Shifting whole copies avoids that entirely: every version is internally
    continuous and time-aligned with the others, because they all come from
    one source processed end to end. Cross-fading between them is then just
    a smooth pitch glide, closer to how a voice actually moves between
    notes. Costs one Rubber Band pass per distinct semitone value (~5 for
    this melody) instead of one per syllable, which the audio stage's
    budget absorbs easily."""
    if len(audio) < sr * MIN_SEGMENT_S:
        return audio, start_offset

    contour = contour or DEFAULT_CONTOUR_SEMITONES
    n_segments = max(1, min(n_words, 12))
    notes = [contour[(start_offset + i) % len(contour)] for i in range(n_segments)]
    beats = [DEFAULT_NOTE_BEATS[(start_offset + i) % len(DEFAULT_NOTE_BEATS)]
             for i in range(n_segments)]
    # Always resolve a line onto a held note. The melody's own half notes fall
    # at fixed positions (7 and 14), so whether a line ended on one depended
    # entirely on how many syllables happened to precede it - most lines ran
    # straight past their last word at conversational speed. Nursery rhymes
    # phrase the other way round: the final syllable of a line is held ("baa
    # baa black sheeep"), which is what makes a line sound finished.
    beats[-1] = max(beats[-1], 2)

    # Rhythm first, then pitch. Re-timing changes where the syllables sit, so
    # boundaries have to be recomputed from the re-timed audio before the
    # pitch contour is laid over it.
    boundaries = _segment_boundaries(audio, n_segments, sr)
    audio, boundaries = _apply_rhythm(audio, sr, boundaries, beats)

    # One pass per distinct note, not per syllable - the same note reused
    # across syllables costs nothing extra.
    versions = {n: _pitch_shift_whole(audio, sr, n) for n in set(notes)}
    # Rubber Band can return a marginally different length; clamp every
    # version to the shortest so sample indices stay aligned across them.
    usable = min(len(v) for v in versions.values())
    versions = {n: v[:usable] for n, v in versions.items()}
    boundaries = [min(b, usable) for b in boundaries]

    fade = max(1, int(sr * CROSSFADE_S))

    out = np.zeros(usable, dtype=np.float32)
    for i, note in enumerate(notes):
        start, end = boundaries[i], boundaries[i + 1]
        window = np.ones(end - start, dtype=np.float32)
        # Ramp each note in and out across the boundary so consecutive notes
        # sum to unity gain - a glide between pitches rather than a hard cut.
        ramp = min(fade, (end - start) // 2)
        if ramp > 0:
            if i > 0:
                window[:ramp] = np.linspace(0.0, 1.0, ramp)
            if i < len(notes) - 1:
                window[-ramp:] = np.linspace(1.0, 0.0, ramp)
        out[start:end] += versions[note][start:end] * window

    return out, start_offset + n_segments
