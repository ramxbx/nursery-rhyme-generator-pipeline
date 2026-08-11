"""Audio generation agent (GPT-12).

Piper TTS (CPU, ONNX) synthesizes narration per scene. Rather than fight
Piper's length_scale to hit an exact rhyme-meter duration (risks distorted,
unnatural speech), each line is synthesized at its natural rate and then
padded with trailing silence up to the scene's target duration from
script_agent - i.e. the "rhythmic phrasing" is pause-shaping after the
words, never truncating or warping speech to force a duration.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import numpy as np
import soundfile as sf
from piper import PiperVoice, SynthesisConfig

from src.config import PipelineConfig, load_config
from src.utils.file_manager import ensure_dirs, read_json, safe_write_json, scene_path
from src.utils.logger import get_logger, log_with_fields
from src.utils.singing import apply_singsong_contour

logger = get_logger("audio_agent")

REPO_ROOT = Path(__file__).resolve().parent.parent.parent


def load_voice(tts_config: dict) -> PiperVoice:
    model_path = REPO_ROOT / tts_config["voice_onnx"]
    config_path = REPO_ROOT / tts_config["voice_config"]
    return PiperVoice.load(model_path, config_path=config_path)


def resample_linear(audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    if orig_sr == target_sr:
        return audio
    duration = len(audio) / orig_sr
    n_target = int(round(duration * target_sr))
    orig_t = np.linspace(0.0, duration, num=len(audio), endpoint=False)
    target_t = np.linspace(0.0, duration, num=n_target, endpoint=False)
    return np.interp(target_t, orig_t, audio).astype(np.float32)


def synthesize_line(voice: PiperVoice, text: str, target_duration_s: float, tts_config: dict) -> np.ndarray:
    syn_config = SynthesisConfig(
        noise_scale=tts_config.get("noise_scale", 0.667),
        noise_w_scale=tts_config.get("noise_w", 0.8),
        length_scale=tts_config.get("length_scale", 1.0),
        normalize_audio=True,
    )
    chunks = list(voice.synthesize(text, syn_config=syn_config))
    if not chunks:
        raise ValueError(f"Piper produced no audio for line: {text!r}")

    orig_sr = chunks[0].sample_rate
    audio = np.concatenate([c.audio_float_array for c in chunks])

    if tts_config.get("singing_mode", True):
        audio = apply_singsong_contour(audio, orig_sr, n_words=max(1, len(text.split())))

    natural_duration = len(audio) / orig_sr
    if natural_duration < target_duration_s:
        pad_samples = int(round((target_duration_s - natural_duration) * orig_sr))
        audio = np.concatenate([audio, np.zeros(pad_samples, dtype=np.float32)])
    elif natural_duration > target_duration_s * 1.15:
        log_with_fields(logger, 30, "speech longer than target duration, not truncating",
                         natural_s=round(natural_duration, 2), target_s=target_duration_s)

    return resample_linear(audio, orig_sr, tts_config.get("sample_rate", 48000))


def generate_audio(script: dict, config: PipelineConfig) -> list[dict]:
    dirs = ensure_dirs(config.paths)
    voice = load_voice(config.tts)
    sample_rate = config.tts.get("sample_rate", 48000)

    manifest = []
    for i, scene in enumerate(script["scenes"], start=1):
        audio = synthesize_line(voice, scene["line"], scene["duration_s"], config.tts)
        out_path = scene_path(dirs["audio_dir"], i, ".wav")
        sf.write(out_path, audio, sample_rate, subtype="PCM_16")

        actual_duration = round(len(audio) / sample_rate, 2)
        manifest.append({"scene_index": i, "line": scene["line"], "audio_path": str(out_path),
                          "target_duration_s": scene["duration_s"], "actual_duration_s": actual_duration})
        log_with_fields(logger, 20, "scene audio generated", scene_index=i, audio_path=str(out_path),
                         actual_duration_s=actual_duration)

    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate scene narration audio from a script JSON file.")
    parser.add_argument("script", type=Path, help="Path to a script JSON file (from script_agent)")
    parser.add_argument("--out", type=Path, default=None, help="Manifest output path")
    args = parser.parse_args()

    config = load_config()
    script = read_json(args.script)
    manifest = generate_audio(script, config)

    out_path = args.out or (config.paths["audio_dir"] / "manifest.json")
    safe_write_json(out_path, manifest)
    print(f"Wrote {len(manifest)} audio files, manifest at {out_path}")


if __name__ == "__main__":
    main()
