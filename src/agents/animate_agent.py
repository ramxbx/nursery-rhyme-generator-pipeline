"""Video animation/assembly agent (GPT-13).

Default path only: FFmpeg zoompan Ken Burns pan/zoom (upscaling each scene
image to 1920x1080 via lanczos) + xfade/acrossfade crossfades between
scenes. AnimateDiff was dropped from scope entirely (GPT-13/GPT-18) - not
attempted here, not even behind a flag - the 4GB GTX 1050 can't safely hold
multi-frame video-diffusion latents.
"""
from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import soundfile as sf

from src.config import PipelineConfig, load_config
from src.utils.ffmpeg_helper import (
    build_scene_clip, burn_subtitles, compute_scene_timeline, crossfade_concat,
    mix_background_music, probe,
)
from src.utils.file_manager import ensure_dirs, read_json
from src.utils.logger import get_logger, log_with_fields
from src.utils.music_generator import generate_background_music
from src.utils.subtitles import write_ass_subtitles

logger = get_logger("animate_agent")

CROSSFADE_S = 0.4


class AssemblyError(Exception):
    pass


def _validate_output(out_path: Path, expected_width: int, expected_height: int, expected_fps: int) -> None:
    info = probe(out_path)
    video_streams = [s for s in info["streams"] if s["codec_type"] == "video"]
    audio_streams = [s for s in info["streams"] if s["codec_type"] == "audio"]

    if not video_streams:
        raise AssemblyError("Output has no video stream")
    if not audio_streams:
        raise AssemblyError("Output has no audio stream")

    v = video_streams[0]
    if v.get("codec_name") != "h264":
        raise AssemblyError(f"Expected h264, got {v.get('codec_name')}")
    if int(v.get("width", 0)) != expected_width or int(v.get("height", 0)) != expected_height:
        raise AssemblyError(f"Expected {expected_width}x{expected_height}, got {v.get('width')}x{v.get('height')}")

    num, den = (v.get("r_frame_rate", "0/1").split("/") + ["1"])[:2]
    actual_fps = round(int(num) / max(int(den), 1))
    if actual_fps != expected_fps:
        raise AssemblyError(f"Expected {expected_fps}fps, got {actual_fps}fps")


def assemble_video(images_manifest: list[dict], audio_manifest: list[dict],
                    config: PipelineConfig, out_path: Path) -> Path:
    dirs = ensure_dirs(config.paths)
    fps = config.video["fps"]
    width, height = config.video["width"], config.video["height"]

    images_by_scene = {m["scene_index"]: m for m in images_manifest}
    audio_by_scene = {m["scene_index"]: m for m in audio_manifest}
    scene_indices = sorted(set(images_by_scene) & set(audio_by_scene))
    if not scene_indices:
        raise AssemblyError("No matching scenes between image and audio manifests")

    with tempfile.TemporaryDirectory(dir=dirs["output_dir"]) as tmp:
        tmp_dir = Path(tmp)
        clips, durations, lines = [], [], []
        for idx in scene_indices:
            image_path = Path(images_by_scene[idx]["image_path"])
            audio_entry = audio_by_scene[idx]
            audio_path = Path(audio_entry["audio_path"])
            duration = audio_entry["actual_duration_s"]

            clip_path = tmp_dir / f"clip_{idx:03d}.mp4"
            build_scene_clip(image_path, audio_path, duration, fps, width, height, clip_path)
            clips.append(clip_path)
            durations.append(duration)
            lines.append(audio_entry.get("line", ""))
            log_with_fields(logger, 20, "scene clip built", scene_index=idx, duration_s=duration)

        merged_path = tmp_dir / "merged_no_music.mp4"
        crossfade_concat(clips, durations, fps, merged_path, crossfade_s=CROSSFADE_S)

        music_config = config.pipeline.get("music", {})
        if music_config.get("enabled", True):
            total_duration = probe(merged_path)["format"]["duration"]
            if music_config.get("backend", "procedural") == "musicgen":
                from src.utils.musicgen_generator import DEFAULT_PROMPT, generate_background_music_musicgen
                music, sample_rate = generate_background_music_musicgen(
                    float(total_duration), music_config.get("prompt", DEFAULT_PROMPT))
                log_with_fields(logger, 20, "musicgen background music generated",
                                 duration_s=round(float(total_duration), 2))
            else:
                sample_rate = 48000
                music = generate_background_music(float(total_duration), sample_rate=sample_rate)
            music_path = tmp_dir / "background_music.wav"
            sf.write(music_path, music, sample_rate, subtype="PCM_16")
            with_music_path = tmp_dir / "with_music.mp4"
            mix_background_music(merged_path, music_path, with_music_path,
                                  music_config.get("volume", 0.18), duck=music_config.get("duck", True),
                                  # the narration's rate, not the music's - MusicGen emits 32kHz
                                  sample_rate=config.tts.get("sample_rate", 48000))
            log_with_fields(logger, 20, "background music mixed", duration_s=round(float(total_duration), 2))
        else:
            with_music_path = merged_path

        subtitles_config = config.pipeline.get("subtitles", {})
        if subtitles_config.get("enabled", True):
            timeline = compute_scene_timeline(durations, CROSSFADE_S)
            ass_path = tmp_dir / "subtitles.ass"
            write_ass_subtitles(lines, timeline, ass_path, width, height,
                                 font_size=subtitles_config.get("font_size", 72))
            burn_subtitles(with_music_path, ass_path, out_path)
            log_with_fields(logger, 20, "subtitles burned in", num_lines=len(lines))
        else:
            with_music_path.replace(out_path)

    _validate_output(out_path, width, height, fps)
    log_with_fields(logger, 20, "assembly validated", out_path=str(out_path))
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Assemble scene images + audio into the final video.")
    parser.add_argument("script", type=Path, help="Path to the script JSON (used to name the output file)")
    parser.add_argument("--images-manifest", type=Path, default=None)
    parser.add_argument("--audio-manifest", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    config = load_config()
    images_manifest = read_json(args.images_manifest or (config.paths["images_dir"] / "manifest.json"))
    audio_manifest = read_json(args.audio_manifest or (config.paths["audio_dir"] / "manifest.json"))

    out_path = args.out or (config.paths["output_dir"] / f"{args.script.stem}.mp4")
    assemble_video(images_manifest, audio_manifest, config, out_path)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
