"""End-to-end pipeline orchestration (GPT-15).

Codex is the orchestration authority: it sequences stages, validates each
stage's output before advancing, and supports resuming a failed run
without repeating already-completed stages. Each stage runs as its own
subprocess (not an in-process function call) so the visual stage's CUDA
context is fully torn down when it exits - the "one GPU stage at a time"
rule from GPT-18 is enforced by process boundaries, not convention.

YouTube upload (GPT-14) is intentionally NOT part of this pipeline - it
needs user-provided credentials and stays a separate, explicit step.

Resume/checkpoint is coarse-grained: if a stage's output file already
exists, that stage is skipped entirely (not resumed mid-stage) unless
--force is passed. Per-scene manifests (data/images, data/audio) are not
namespaced by run name, matching the existing agent CLIs - running a
different rhyme without --force may reuse stale outputs from a prior run.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import load_config
from src.utils.file_manager import ensure_dirs, read_json
from src.utils.logger import get_logger, log_with_fields

logger = get_logger("orchestration")


class StageError(Exception):
    pass


def run_stage(module: str, args: list[str], stage_name: str, max_retries: int = 1) -> None:
    """Run a pipeline stage as an isolated subprocess. Retries restart the
    whole stage from scratch (coarse-grained - no partial-stage resume)."""
    cmd = [sys.executable, "-m", module, *args]
    last_error = ""
    for attempt in range(1, max_retries + 2):
        start = time.perf_counter()
        result = subprocess.run(cmd, capture_output=True, text=True)
        elapsed = time.perf_counter() - start
        if result.returncode == 0:
            log_with_fields(logger, 20, f"{stage_name} stage complete", attempt=attempt, elapsed_s=round(elapsed, 2))
            return
        last_error = result.stderr[-3000:]
        log_with_fields(logger, 40, f"{stage_name} stage failed", attempt=attempt,
                         elapsed_s=round(elapsed, 2), error=last_error)
    raise StageError(f"{stage_name} stage failed after {max_retries + 1} attempt(s): {last_error}")


def validate_script(path: Path) -> dict:
    if not path.exists():
        raise StageError(f"script output missing: {path}")
    data = read_json(path)
    if not data.get("scenes"):
        raise StageError(f"script output has no scenes: {path}")
    return data


def manifest_is_current(path: Path, expected_scenes: int) -> bool:
    """Whether a cached manifest can be reused for the current script.

    Per-scene manifests are not namespaced by run name, so a manifest left by a
    previous rhyme with a different scene count would otherwise be silently
    reused - the resume check only asked whether the file existed. A 16-scene
    run followed by a 4-scene one then failed validation with "manifest has 16
    entries, expected 4" and no way forward short of deleting files by hand.
    Treating a count mismatch as stale makes resume safe across runs."""
    if not path.exists():
        return False
    try:
        return len(read_json(path)) == expected_scenes
    except (ValueError, OSError):
        return False  # unreadable/corrupt cache is stale by definition


def validate_manifest(path: Path, expected_scenes: int, stage_name: str) -> None:
    if not path.exists():
        raise StageError(f"{stage_name} manifest missing: {path}")
    data = read_json(path)
    if len(data) != expected_scenes:
        raise StageError(f"{stage_name} manifest has {len(data)} entries, expected {expected_scenes}: {path}")


def generate_input_rhyme(config, seed: str | None = None, lines: int | None = None) -> Path:
    """Stage 0: write the poem the rest of the run is built from.

    Kept in-process rather than a subprocess like the other stages - it touches
    no GPU, so the isolation those need does not apply, and the caller needs the
    chosen filename back to name every downstream artefact."""
    from src.agents.rhyme_agent import RHYME_LINES, generate_rhyme

    name, text, generated = generate_rhyme(config, seed, line_total=lines or RHYME_LINES)
    path = Path(config.paths["data_dir"]) / f"{name}.txt"
    path.write_text(text, encoding="utf-8")
    log_with_fields(logger, 20, "rhyme stage complete", path=str(path), generated=generated,
                     lines=len(text.strip().splitlines()))
    return path


def run_pipeline(input_path: Path, name: str | None = None, force: bool = False,
                  motion: bool | None = None) -> Path:
    """Run every stage. `motion` overrides config/pipeline.yaml when given:
    True forces AnimateDiff clips, False forces Ken Burns pans."""
    config = load_config()
    ensure_dirs(config.paths)
    name = name or input_path.stem

    script_path = config.paths["scripts_dir"] / f"{name}.json"
    images_manifest = config.paths["images_dir"] / "manifest.json"
    audio_manifest = config.paths["audio_dir"] / "manifest.json"
    output_path = config.paths["output_dir"] / f"{name}.mp4"

    pipeline_start = time.perf_counter()
    log_with_fields(logger, 20, "pipeline started", name=name, input=str(input_path))

    # Stage 1: script (CPU)
    if force or not script_path.exists():
        run_stage("src.agents.script_agent", [str(input_path), "--out", str(script_path)], "script")
    else:
        log_with_fields(logger, 20, "skipping script stage, output exists", path=str(script_path))
    script = validate_script(script_path)
    n_scenes = len(script["scenes"])

    # Stage 2: visual (GPU - only stage allowed to touch it, isolated subprocess)
    if force or not manifest_is_current(images_manifest, n_scenes):
        run_stage("src.agents.visual_agent", [str(script_path), "--out", str(images_manifest)], "visual")
    else:
        log_with_fields(logger, 20, "skipping visual stage, output is current", path=str(images_manifest))
    validate_manifest(images_manifest, n_scenes, "visual")

    # Stage 2b: motion (GPU) - optional, and by far the most expensive stage in
    # the pipeline at ~40 min per scene. Its own subprocess so the model is
    # released before the audio stage starts, and sequenced after the image
    # stage so the two never share the card.
    motion_manifest = Path(config.paths["images_dir"]).parent / "motion" / "manifest.json"
    use_motion = (config.pipeline.get("motion", {}).get("enabled", False)
                  if motion is None else motion)
    if use_motion:
        if force or not manifest_is_current(motion_manifest, n_scenes):
            run_stage("src.agents.motion_agent",
                      [str(script_path), "--images-manifest", str(images_manifest),
                       "--out", str(motion_manifest)], "motion")
        else:
            log_with_fields(logger, 20, "skipping motion stage, output is current",
                             path=str(motion_manifest))
    else:
        log_with_fields(logger, 20, "motion stage off, scenes will use Ken Burns pans")

    # Stage 3: audio (CPU)
    if force or not manifest_is_current(audio_manifest, n_scenes):
        run_stage("src.agents.audio_agent", [str(script_path), "--out", str(audio_manifest)], "audio")
    else:
        log_with_fields(logger, 20, "skipping audio stage, output is current", path=str(audio_manifest))
    validate_manifest(audio_manifest, n_scenes, "audio")

    # Stage 4: animate/assembly (CPU, ffmpeg) - always re-run, cheap relative to the others
    # and its output must reflect whatever images/audio currently exist.
    animate_args = [
        str(script_path),
        "--images-manifest", str(images_manifest),
        "--audio-manifest", str(audio_manifest),
        "--out", str(output_path),
    ]
    # Only point assembly at motion clips when motion is actually on. Passing
    # the path unconditionally meant a previous run's clips were still picked up
    # after motion was turned off - the expensive stage was correctly skipped,
    # but the video was assembled from stale animation anyway, so the switch
    # looked like it did nothing.
    if use_motion:
        animate_args += ["--motion-manifest", str(motion_manifest)]
    run_stage("src.agents.animate_agent", animate_args, "animate")
    if not output_path.exists():
        raise StageError(f"expected final video at {output_path} but it wasn't created")

    elapsed = time.perf_counter() - pipeline_start
    log_with_fields(logger, 20, "pipeline complete", output=str(output_path), total_elapsed_s=round(elapsed, 1))
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the full nursery-rhyme pipeline end-to-end (script -> visual -> audio -> animate). "
                    "YouTube upload is a separate manual step - see GPT-14.")
    parser.add_argument("input", type=Path, nargs="?", default=None,
                        help="Path to a text file containing the rhyme (omit with --generate)")
    parser.add_argument("--generate", action="store_true",
                        help="Write a new 16-line rhyme first instead of reading one from a file")
    # Load-bearing for runtime, not a nicety: with the motion stage on, every
    # extra line is another ~36 minutes of animation.
    parser.add_argument("--lines", type=int, default=None,
                        help="With --generate, how many lines to write (default 16)")
    parser.add_argument("--seed", default=None,
                        help="With --generate, take inspiration from this seed rhyme title")
    parser.add_argument("--name", default=None, help="Base name for output files (default: input filename stem)")
    parser.add_argument("--force", action="store_true", help="Re-run every stage even if outputs already exist")
    # Overrides motion.enabled in config/pipeline.yaml for this run only.
    motion_group = parser.add_mutually_exclusive_group()
    motion_group.add_argument("--motion", dest="motion", action="store_true", default=None,
                              help="Animate scenes with AnimateDiff (~39 min per scene)")
    motion_group.add_argument("--no-motion", dest="motion", action="store_false",
                              help="Ken Burns pan/zoom over the still images instead (minutes, not hours)")
    args = parser.parse_args()

    if not args.generate and args.input is None:
        parser.error("provide an input file, or pass --generate to write a new rhyme")

    try:
        input_path = args.input
        if args.generate:
            input_path = generate_input_rhyme(load_config(), args.seed, args.lines)
            print(f"Generated rhyme: {input_path}")
        # A generated run names its outputs after the poem, so a new rhyme never
        # silently reuses the previous run's cached scenes.
        output = run_pipeline(input_path, args.name, args.force or args.generate, args.motion)
    except StageError as e:
        logger.error(f"Pipeline failed: {e}")
        sys.exit(1)

    print(f"Pipeline complete: {output}")


if __name__ == "__main__":
    main()
