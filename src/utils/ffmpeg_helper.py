"""FFmpeg/FFprobe invocation helpers (GPT-13's assembly backend)."""
from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

# Winget's per-user PATH registration isn't visible to processes spawned
# from a shell that started before the install - fall back to the known
# install location so this doesn't silently break in a stale shell.
_WINGET_FFMPEG_DIR = Path(
    "C:/Users/abhiv/AppData/Local/Microsoft/WinGet/Packages/"
    "Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe/ffmpeg-9.0-full_build/bin"
)


class FFmpegError(Exception):
    pass


def _find_binary(name: str) -> str:
    found = shutil.which(name)
    if found:
        return found
    fallback = _WINGET_FFMPEG_DIR / f"{name}.exe"
    if fallback.exists():
        return str(fallback)
    raise FFmpegError(f"{name} not found on PATH or at {fallback}")


def ffmpeg_path() -> str:
    return _find_binary("ffmpeg")


def ffprobe_path() -> str:
    return _find_binary("ffprobe")


def run(args: list[str]) -> str:
    result = subprocess.run(args, capture_output=True, text=True)
    if result.returncode != 0:
        raise FFmpegError(f"Command failed ({args[0]}): {result.stderr[-2000:]}")
    return result.stdout


def probe(path: Path) -> dict:
    out = run([ffprobe_path(), "-v", "quiet", "-print_format", "json",
               "-show_format", "-show_streams", str(path)])
    return json.loads(out)


# A finished video is encoded three times, not once: each scene clip, then the
# crossfade concat, then the subtitle burn-in. (The music mix copies the video
# stream, so it costs nothing.) Every one of those was previously running on
# libx264's default CRF 23, so the two intermediates were discarding detail
# before the final encode ever saw it - generation loss on top of an already
# soft 384px source.
#
# The intermediates are now near-lossless: CRF 12 is visually transparent and
# costs only temp-directory space, which is freed when the run ends. Only the
# last encode is asked to compress for real.
INTERMEDIATE_CRF = 12
FINAL_CRF = 18
# `slow` against x264's default `medium`. The encode is seconds against ~36
# minutes per scene of diffusion, so preset time is free here in a way it
# would not be in any normal video pipeline.
X264_PRESET = "slow"


def _x264(crf: int = INTERMEDIATE_CRF) -> list[str]:
    """Video encode flags, so quality settings live in one place."""
    return ["-c:v", "libx264", "-crf", str(crf), "-preset", X264_PRESET,
            "-pix_fmt", "yuv420p"]


# Camera moves cycled across scenes. A single slow centre push repeated on every
# scene reads as a stuck camera by the third one, and at zoom+0.0008 capped at
# 1.15 it was barely visible anyway - a 3s clip only reached 1.06x. Three moves
# against the four-shot framing cycle means the pairing does not repeat for
# twelve scenes.
#
# Expressions are ffmpeg zoompan: `on` is the output frame number, `iw/ih` the
# input size after scaling. Zoom is applied about a point that must be recomputed
# per frame, which is why the x/y expressions reference zoom rather than being
# constants.
MOTIONS = ("push_in", "pan_right", "pull_out")
ZOOM_MAX = 1.18          # far enough to read as movement, near enough to stay sharp
PAN_ZOOM = 1.12          # panning needs headroom cropped in to have somewhere to go


def motion_for_scene(scene_index: int) -> str:
    """Camera move for a 1-based scene index. Deterministic, so a re-run frames
    and moves identically."""
    return MOTIONS[(scene_index - 1) % len(MOTIONS)]


def _zoompan(motion: str, n_frames: int, width: int, height: int, fps: int) -> str:
    """A zoompan filter for one camera move.

    Every expression is a pure function of `on`, the output frame index, and
    never of zoompan's own `zoom` accumulator.

    That accumulator does not work on this input. `zoom` carries over only
    between the `d` output frames generated from a single input frame, and these
    clips are built from `-loop 1 -i image` with `d=1`, so every output frame
    comes from a fresh input frame and `zoom` resets to its initial value each
    time. `z='min(zoom+step,MAX)'` therefore evaluates to the same constant
    forever: measured on a finished video, push_in and pull_out scenes had a
    first-to-last-frame difference of ~1 (i.e. static) against ~45 for
    pan_right, which was only ever correct because it drives `x` from `on` and
    holds zoom fixed."""
    centre_x, centre_y = "iw/2-(iw/zoom/2)", "ih/2-(ih/zoom/2)"
    # Progress through the clip, 0.0 -> 1.0.
    t = f"on/{max(n_frames - 1, 1)}"
    if motion == "pull_out":
        # Starts wide and settles in: zoom decreases toward 1.0 rather than away
        # from it, so the frame opens up over the line.
        z = f"'{ZOOM_MAX}-{ZOOM_MAX - 1.0:.6f}*{t}'"
        return f"zoompan=z={z}:x='{centre_x}':y='{centre_y}':d=1:s={width}x{height}:fps={fps}"
    if motion == "pan_right":
        # Fixed zoom, horizontal drift across the cropped-in frame.
        x = f"'(iw-iw/zoom)*{t}'"
        return f"zoompan=z={PAN_ZOOM}:x={x}:y='ih/2-(ih/zoom/2)':d=1:s={width}x{height}:fps={fps}"
    z = f"'1+{ZOOM_MAX - 1.0:.6f}*{t}'"
    return f"zoompan=z={z}:x='{centre_x}':y='{centre_y}':d=1:s={width}x{height}:fps={fps}"


def build_motion_scene_clip(motion_path: Path, audio_path: Path, duration_s: float,
                             fps: int, width: int, height: int, out_path: Path) -> Path:
    """Build a scene from an AnimateDiff clip instead of a still.

    The generated clip is 2 seconds and scenes run 3-5, so it is looped to fill.
    Looping rather than slowing it down: at 8fps the source has no frames to
    spare, and stretching turns a walk into a crawl. The loop point is visible
    on close inspection but reads as a cycle, which suits a child's rhyme.

    minterpolate raises the source to the video frame rate. It estimates motion
    vectors rather than cross-fading, so it invents genuine in-between positions
    - blending would ghost. It is also what makes 16 diffused frames enough:
    without it the motion is visibly choppy at 8fps.

    Upscaling happens after interpolation, so the interpolator works on the
    clean 384x384 source rather than on upscaling artefacts."""
    n_loops = max(1, int(duration_s / 2.0) + 1)
    chain = (
        f"minterpolate=fps={fps}:mi_mode=mci:mc_mode=aobmc:me_mode=bidir:vsbmc=1,"
        f"scale={width}:{height}:force_original_aspect_ratio=increase:flags=lanczos,"
        f"crop={width}:{height}"
    )
    args = [
        ffmpeg_path(), "-y",
        "-stream_loop", str(n_loops), "-i", str(motion_path),
        "-i", str(audio_path),
        "-filter_complex", f"[0:v]{chain}[v]",
        "-map", "[v]", "-map", "1:a",
        "-t", f"{duration_s:.3f}",
        *_x264(), "-r", str(fps),
        "-c:a", "aac", "-shortest",
        str(out_path),
    ]
    run(args)
    return out_path


def build_scene_clip(image_path: Path, audio_path: Path, duration_s: float,
                      fps: int, width: int, height: int, out_path: Path,
                      motion: str = "push_in") -> Path:
    """Ken Burns pan/zoom on a static image, upscaled to (width, height),
    muxed with the scene's audio, trimmed to duration_s.

    Scales to COVER the frame and centre-crops the overflow, rather than
    stretching to fit. Scene images are generated square (512x512, SD1.5's
    native training resolution - see GPT-32), so a plain stretch to 16:9
    would visibly distort the subject. Prompts ask for a centred subject
    (STYLE_ANCHOR in visual_agent.py), which is what makes a centre crop
    safe here."""
    n_frames = max(1, int(round(duration_s * fps)))
    # Upscale generously before zoompan: it crops from the scaled input, so
    # zooming a frame-sized image would magnify its own pixels and soften.
    zoompan = (
        f"scale={width * 2}:{height * 2}:force_original_aspect_ratio=increase:flags=lanczos,"
        f"crop={width * 2}:{height * 2},"
        f"{_zoompan(motion, n_frames, width, height, fps)}"
    )
    args = [
        ffmpeg_path(), "-y",
        "-loop", "1", "-i", str(image_path),
        "-i", str(audio_path),
        "-filter_complex", f"[0:v]{zoompan}[v]",
        "-map", "[v]", "-map", "1:a",
        "-t", f"{duration_s:.3f}",
        *_x264(), "-r", str(fps),
        "-c:a", "aac", "-shortest",
        str(out_path),
    ]
    run(args)
    return out_path


# How far the music is pushed down while a word is being sung. 6dB is enough
# to open a clear space for the voice without the music audibly pumping.
MUSIC_DUCK_RATIO = 8
MUSIC_DUCK_THRESHOLD = 0.05
# Slow release so the music swells back between lines rather than fluttering
# under every syllable.
MUSIC_DUCK_ATTACK_MS = 20
MUSIC_DUCK_RELEASE_MS = 300
# Streaming-standard programme loudness. The pipeline used to hand back a
# ~-32 dBFS file that needed the volume slider at maximum to hear.
TARGET_LUFS = -16
TARGET_TRUE_PEAK = -1.5


def mix_background_music(video_path: Path, music_path: Path, out_path: Path, music_volume: float = 0.18,
                          duck: bool = True, sample_rate: int = 48000) -> Path:
    """Mix a background music track under an existing video's narration
    audio, matching the narration's duration exactly. Video stream is
    copied (not re-encoded) - only the audio is touched.

    Three things here exist to keep the words intelligible for a child, which
    a plain amix does not deliver:

    * `normalize=0` on amix. By default amix divides every input by the number
      of inputs, so simply having a music track present cost the voice 6dB -
      the single biggest reason the lyrics were hard to make out.
    * Sidechain ducking. The music is compressed by the voice, so it steps back
      whenever a word is sung and returns in the gaps. Constant background
      music at a level low enough never to cover the voice is too quiet to be
      worth having; ducking lets it be present AND out of the way.
    * loudnorm to a standard programme loudness, so the finished video plays at
      the same volume as anything else the child watches. It is followed by an
      explicit aresample because loudnorm runs its analysis at 192kHz and emits
      at that rate - without pinning it back, the output stream silently
      changes sample rate.
    """
    if duck:
        # The voice feeds both the mix and the compressor's sidechain, so it has
        # to be split - an ffmpeg input cannot be consumed twice.
        chain = (
            f"[1:a]volume={music_volume}[bg];"
            f"[0:a]asplit=2[voice][key];"
            f"[bg][key]sidechaincompress=threshold={MUSIC_DUCK_THRESHOLD}:ratio={MUSIC_DUCK_RATIO}"
            f":attack={MUSIC_DUCK_ATTACK_MS}:release={MUSIC_DUCK_RELEASE_MS}[bgducked];"
            f"[voice][bgducked]amix=inputs=2:duration=first:dropout_transition=0:normalize=0[mixed];"
        )
    else:
        chain = (
            f"[1:a]volume={music_volume}[bg];"
            f"[0:a][bg]amix=inputs=2:duration=first:dropout_transition=0:normalize=0[mixed];"
        )
    filter_complex = (chain +
                       f"[mixed]loudnorm=I={TARGET_LUFS}:TP={TARGET_TRUE_PEAK}:LRA=11,"
                       f"aresample={sample_rate}[a]")
    args = [
        ffmpeg_path(), "-y",
        "-i", str(video_path), "-i", str(music_path),
        "-filter_complex", filter_complex,
        "-map", "0:v", "-map", "[a]",
        "-c:v", "copy", "-c:a", "aac",
        str(out_path),
    ]
    run(args)
    return out_path


def compute_scene_timeline(durations: list[float], crossfade_s: float) -> list[tuple[float, float]]:
    """(start, end) time for each scene in the crossfaded output timeline,
    using the exact same per-pair clamped crossfade duration as
    crossfade_concat's xfade/acrossfade filter graph - shared so subtitle
    timing (GPT-21) can never drift from the actual video timing."""
    if not durations:
        return []
    starts, ends = [0.0], [durations[0]]
    running_duration = durations[0]
    for i in range(1, len(durations)):
        xfade_dur = min(crossfade_s, durations[i - 1], durations[i])
        start = max(0.0, running_duration - xfade_dur)
        starts.append(start)
        ends.append(start + durations[i])
        running_duration = running_duration + durations[i] - xfade_dur
    return list(zip(starts, ends))


def crossfade_concat(clips: list[Path], durations: list[float], fps: int,
                      out_path: Path, crossfade_s: float = 0.4) -> Path:
    """Chain-merge scene clips with an xfade/acrossfade transition between
    each consecutive pair."""
    if len(clips) == 1:
        args = [ffmpeg_path(), "-y", "-i", str(clips[0]), "-c", "copy", str(out_path)]
        run(args)
        return out_path

    inputs = []
    for c in clips:
        inputs += ["-i", str(c)]

    filter_parts = []
    running_duration = durations[0]
    prev_v, prev_a = "0:v", "0:a"
    for i in range(1, len(clips)):
        xfade_dur = min(crossfade_s, durations[i - 1], durations[i])
        offset = max(0.0, running_duration - xfade_dur)
        out_v, out_a = f"v{i}", f"a{i}"
        filter_parts.append(f"[{prev_v}][{i}:v]xfade=transition=fade:duration={xfade_dur:.3f}:offset={offset:.3f}[{out_v}]")
        filter_parts.append(f"[{prev_a}][{i}:a]acrossfade=d={xfade_dur:.3f}[{out_a}]")
        running_duration = running_duration + durations[i] - xfade_dur
        prev_v, prev_a = out_v, out_a

    filter_complex = ";".join(filter_parts)
    args = [
        ffmpeg_path(), "-y", *inputs,
        "-filter_complex", filter_complex,
        "-map", f"[{prev_v}]", "-map", f"[{prev_a}]",
        *_x264(), "-r", str(fps),
        "-c:a", "aac",
        str(out_path),
    ]
    run(args)
    return out_path


# Rubber Band's defaults are tuned for speed, not vocal quality:
#   formant=preserved - the default (shifted) moves the vocal formants along
#     with the pitch, which is precisely the "chipmunk" effect. At the 7-9
#     semitone shifts this project's melody uses, that alone accounts for much
#     of the artificial sound. Preserving formants keeps the voice sounding
#     like the same singer hitting a higher note.
#   pitchq=quality - default is `speed`. The audio stage costs ~20s of a ~16min
#     pipeline run, so trading time for quality here is essentially free.
#
# Settings deliberately NOT used, having been measured rather than assumed:
# `smoothing=on` sounds like it should suit sustained singing, but on Piper's
# output it was catastrophic - it produced the audible mid-line clicks and
# dropouts reported as "broken sounds", degrading a discontinuity metric
# (largest sample jump / signal RMS) from 1.27 to 16.98. `window=long` was
# neutral-to-slightly-worse and buys nothing. The combination below measures
# at 1.16, i.e. marginally *cleaner* than the untouched Piper speech it
# started from (1.28).
RUBBERBAND_VOCAL_OPTS = "formant=preserved:pitchq=quality"


def rubberband_pitch_shift(in_path: Path, out_path: Path, pitch_ratio: float,
                            tempo_ratio: float = 1.0) -> Path:
    """Pitch-shift and/or time-stretch a WAV file using ffmpeg's rubberband
    filter (Rubber Band Library, compiled into this project's ffmpeg build via
    --enable-librubberband). Used in place of librosa's phase-vocoder
    pitch_shift (singing.py, GPT-30), which introduced audible metallic
    artifacts on Piper TTS output.

    pitch_ratio is 2**(semitones/12). tempo_ratio > 1 makes the audio shorter,
    < 1 makes it longer - so to stretch a syllable to N times its length, pass
    tempo_ratio = 1/N. Both happen in a single pass when both are given, which
    is cheaper and cleaner than chaining two separate passes."""
    opts = f"pitch={pitch_ratio}"
    if tempo_ratio != 1.0:
        opts += f":tempo={tempo_ratio}"
    args = [
        ffmpeg_path(), "-y",
        "-i", str(in_path),
        "-af", f"rubberband={opts}:{RUBBERBAND_VOCAL_OPTS}",
        str(out_path),
    ]
    run(args)
    return out_path


def burn_subtitles(video_path: Path, ass_path: Path, out_path: Path) -> Path:
    """Burn an ASS subtitle file into the video. Re-encodes video (required
    for burn-in); audio is copied untouched."""
    # ffmpeg's filtergraph parser needs forward slashes and an escaped
    # colon for Windows paths passed to the ass= filter argument.
    escaped_path = str(ass_path).replace("\\", "/").replace(":", "\\:")
    args = [
        ffmpeg_path(), "-y",
        "-i", str(video_path),
        "-vf", f"ass='{escaped_path}'",
        *_x264(FINAL_CRF),
        "-c:a", "copy",
        str(out_path),
    ]
    run(args)
    return out_path
