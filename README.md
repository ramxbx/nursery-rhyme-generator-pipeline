# Nursery Rhyme Generator Pipeline

Local-first pipeline that turns a nursery rhyme (or any short rhyming poem) into
a narrated, illustrated, subtitled video. Runs entirely on local models tuned
for a 4GB-VRAM GPU (GTX 1050) + 16GB RAM - no cloud APIs, no paid services.

## How it works

```
rhyme.txt --> script_agent --> visual_agent --> audio_agent --> animate_agent --> rhyme.mp4
              (CPU, LLM)       (GPU, SD1.5)     (CPU, Piper)    (CPU, ffmpeg)
```

1. **script_agent** splits the rhyme into lines, creatively rewrites each line
   (keeping the original end-word fixed so the rhyme scheme is preserved by
   construction), then annotates each line with a speaker, stage direction,
   scene description, and mood - all via a small local LLM (CPU-only).
2. **visual_agent** drafts an image prompt per scene from that same data (same
   local LLM, bounded prompt-drafting only) and generates the image with
   SD1.5 + LCM-LoRA - the only stage allowed to use the GPU.
3. **audio_agent** synthesizes narration per line with Piper TTS, pitch-shaped
   into a sing-song melodic contour, padded to match each scene's timing.
4. **animate_agent** assembles everything: Ken Burns pan/zoom per scene,
   crossfade transitions, a procedurally generated background music bed, and
   burned-in kid-readable subtitles - into the final H.264/1080p/24fps video.

**The script agent already produces both the script *and* the image prompts**
in the sense you're asking about - you don't need a separate step. Running the
pipeline on a new rhyme automatically creates a new script (with per-scene
image-prompt-ready descriptions) and new images from it.

## Prerequisites

- Python 3.12+, Git, FFmpeg (all already set up in this repo's environment)
- [LM Studio](https://lmstudio.ai) with two GGUF models downloaded:
  - `liquid/lfm2-1.2b` (primary)
  - `google/gemma-3-1b` (fallback)
- An NVIDIA GPU for the image stage (CPU fallback exists but is slow)

## One-time setup

```bash
make setup      # creates .venv and installs Python dependencies
```

Model weights (SD1.5, LCM-LoRA, Piper voice) are fetched separately -
see `scripts/fetch_models.py` if setting up on a new machine; they're
already downloaded in this environment under `models/`.

### LM Studio must be running CPU-only for the LLM stage

This is the one manual step that doesn't survive an LM Studio restart. The
pipeline calls LM Studio's local server for text generation, and that server
**must** be running the two worker models CPU-only (not GPU-offloaded) so the
GPU stays free for the image stage. Before running the pipeline:

```bash
lms runtime select llama.cpp-win-x86_64-avx2   # genuinely CPU-only engine -
                                                 # the default Vulkan engine
                                                 # reserves GPU memory even
                                                 # with --gpu off
lms load liquid/lfm2-1.2b   --gpu off -c 4096 --parallel 1 --identifier lfm2-1.2b-bench -y
lms load google/gemma-3-1b  --gpu off -c 4096 --parallel 1 --identifier gemma-3-1b-bench -y
```

Verify with `lms ps` (should show both models loaded, `DEVICE: Local`) and
`nvidia-smi` (LM Studio's `llama-server.exe` should show near-zero GPU memory,
not gigabytes - if it's using multiple GB, the runtime engine got reset back
to the Vulkan build and needs reselecting).

## Running it

```bash
make run                          # runs data/rhyme.txt (the bundled sample)
make run INPUT=data/my_rhyme.txt  # runs a different rhyme
```

Or directly:

```bash
.venv/Scripts/python.exe -m src.orchestration data/my_rhyme.txt
```

The pipeline resumes automatically - if a run is interrupted or you rerun the
same input, stages whose output already exists are skipped. Force a full
regeneration with `--force`:

```bash
.venv/Scripts/python.exe -m src.orchestration data/my_rhyme.txt --force
```

## Adding a new rhyme

Just create a plain text file, one line per line of the rhyme:

```
data/my_rhyme.txt
--------------------
Baa, baa, black sheep,
Have you any wool?
Yes sir, yes sir,
Three bags full.
```

Then `make run INPUT=data/my_rhyme.txt`. That's the whole workflow - the
script agent handles turning it into a full scene-by-scene script (with
creative rewrites and image-prompt-ready descriptions) automatically as the
first pipeline stage. No separate "generate a script" step needed.

Each run's outputs are named after the input file's stem (`my_rhyme.txt` ->
`my_rhyme.json` / `my_rhyme.mp4`, etc.), except the two shared manifest files
noted below.

## Where output files are saved

| Stage | Output |
|---|---|
| Script | `data/scripts/<name>.json` - scenes with line, speaker, stage_direction, scene_description, mood, duration |
| Images | `data/images/scene_NNN.png` + `data/images/manifest.json` |
| Audio | `data/audio/scene_NNN.wav` + `data/audio/manifest.json` |
| **Final video** | `data/output/<name>.mp4` |
| Logs | `logs/pipeline.log` (structured JSON, one line per event) |

**Note**: the per-scene image/audio files and their manifests are *not*
namespaced by rhyme name - running a second, different rhyme without
`--force` will reuse/overwrite the first run's scene images and audio if the
manifests already exist for that stage. Run `make clean` between different
rhymes, or use `--force`, to avoid stale cross-run reuse.

## Configuration

Toggles live in `config/pipeline.yaml`, `config/sd_config.yml`,
`config/tts_config.yml`:

- `script.creative_rewrite` - creative line rewriting vs. verbatim recital
- `music.enabled` / `music.volume` - procedural background music
- `singing_mode` (in tts_config.yml) - sing-song pitch contour vs. flat narration
- `subtitles.enabled` / `subtitles.font_size` - burned-in subtitles

## Other Makefile commands

```bash
make test    # run the test suite (pytest, 46 tests)
make clean   # remove generated scripts/images/audio/output/logs (keeps input files, models, venv)
```

## Not included: YouTube upload

The pipeline stops at the finished video. Publishing to YouTube needs your
own API credentials and is intentionally a separate manual step - see the
`GPT-14` issue in Linear for that design when you're ready for it.
