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


def build_scene_clip(image_path: Path, audio_path: Path, duration_s: float,
                      fps: int, width: int, height: int, out_path: Path) -> Path:
    """Ken Burns pan/zoom on a static image, upscaled to (width, height),
    muxed with the scene's audio, trimmed to duration_s.

    Scales to COVER the frame and centre-crops the overflow, rather than
    stretching to fit. Scene images are generated square (512x512, SD1.5's
    native training resolution - see GPT-32), so a plain stretch to 16:9
    would visibly distort the subject. Prompts ask for a centred subject
    (STYLE_ANCHOR in visual_agent.py), which is what makes a centre crop
    safe here."""
    n_frames = max(1, int(round(duration_s * fps)))
    zoompan = (
        f"scale={width}:{height}:force_original_aspect_ratio=increase:flags=lanczos,"
        f"crop={width}:{height},"
        f"zoompan=z='min(zoom+0.0008,1.15)':d={n_frames}:s={width}x{height}:fps={fps}"
    )
    args = [
        ffmpeg_path(), "-y",
        "-loop", "1", "-i", str(image_path),
        "-i", str(audio_path),
        "-filter_complex", f"[0:v]{zoompan}[v]",
        "-map", "[v]", "-map", "1:a",
        "-t", f"{duration_s:.3f}",
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-r", str(fps),
        "-c:a", "aac", "-shortest",
        str(out_path),
    ]
    run(args)
    return out_path


def mix_background_music(video_path: Path, music_path: Path, out_path: Path, music_volume: float = 0.18) -> Path:
    """Mix a background music track under an existing video's narration
    audio, matching the narration's duration exactly. Video stream is
    copied (not re-encoded) - only the audio is touched."""
    filter_complex = (
        f"[1:a]volume={music_volume}[bg];"
        f"[0:a][bg]amix=inputs=2:duration=first:dropout_transition=0[a]"
    )
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
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-r", str(fps),
        "-c:a", "aac",
        str(out_path),
    ]
    run(args)
    return out_path


def rubberband_pitch_shift(in_path: Path, out_path: Path, pitch_ratio: float) -> Path:
    """Pitch-shift a WAV file by pitch_ratio (2**(semitones/12)) using
    ffmpeg's rubberband filter (Rubber Band Library, compiled into this
    project's ffmpeg build via --enable-librubberband). Formant-preserving,
    designed for vocals/speech - used in place of librosa's phase-vocoder
    pitch_shift (singing.py, GPT-30), which introduced audible metallic
    artifacts on Piper TTS output, especially at the 7-9 semitone shifts
    the singing contour uses."""
    args = [
        ffmpeg_path(), "-y",
        "-i", str(in_path),
        "-af", f"rubberband=pitch={pitch_ratio}",
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
        "-c:v", "libx264", "-pix_fmt", "yuv420p",
        "-c:a", "copy",
        str(out_path),
    ]
    run(args)
    return out_path
