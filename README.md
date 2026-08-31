# Nursery Rhyme Generator Pipeline

Local-first pipeline that turns a nursery rhyme (or any short rhyming poem) into
a narrated, illustrated, subtitled video. Runs entirely on local models tuned
for a 4GB-VRAM GPU (GTX 1050) + 16GB RAM - no cloud APIs, no paid services.

## How it works

Everything runs locally, on a 4GB GTX 1050. No API calls, no cloud.

```
rhyme_agent -> script_agent -> visual_agent -> motion_agent -> audio_agent -> animate_agent -> <name>.mp4
 (CPU, LLM)    (CPU, LLM)      (GPU, SD1.5)    (GPU, AD)       (GPU, Bark)   (CPU, ffmpeg)
  optional                                      optional
```

1. **rhyme_agent** *(optional, `--generate`)* writes a new 16-line AABB poem in
   the style of a traditional rhyme, and validates it: right line count, real
   rhyming couplets, no repeated lines, on-metre, and not a recital of anything
   it was shown. Ten hand-written poems in `data/fallback_rhymes/` are the
   safety net when the model can't produce a passing one.
2. **script_agent** rewrites the lines to a shared syllable count so they sing
   to one melody (end-words held fixed, so the rhyme scheme survives by
   construction), groups them into evenly-timed scenes, and elaborates each
   into a scene description.
3. **visual_agent** builds an image prompt per scene, then renders several
   candidates from consecutive seeds and keeps whichever one CLIP judges to
   actually contain the subject. A hires pass re-renders the winner at 768.
   Framing cycles across scenes so a video isn't all close-ups.
4. **motion_agent** *(optional, OFF by default - `--motion`)* regenerates each
   scene as a 2-second AnimateDiff clip. It is text-to-video: it reuses the
   prompt and the winning seed, not the still itself. **This stage is ~83% of a
   run's wall clock**, which is why it is off - see below.
5. **audio_agent** sings each line with Bark, falling back to Piper per line
   when a take comes back unusable. Every downstream duration follows what Bark
   actually sang, not an estimate.
6. **animate_agent** assembles it: each scene's clip (or a Ken Burns pan over
   its still, when there's no motion clip), crossfades, a MusicGen backing track
   ducked under the voice, and burned-in subtitles - to H.264/1080p/24fps.

Stages are separate subprocesses. The GPU ones never run concurrently, and each
releases its model when it exits.

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

Model weights are fetched separately - see `scripts/fetch_models.py` if setting
up on a new machine; they're already downloaded in this environment under
`models/` and the HuggingFace cache. The pipeline pulls: the
`nitrosocke/mo-di-diffusion` SD1.5 fine-tune, the AnimateDiff motion adapter,
Bark, MusicGen, a Piper voice, and CLIP.

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

Verify with `nvidia-smi` - the server process should show near-zero GPU memory,
not gigabytes. If it's using multiple GB, the runtime engine got reset back to
the Vulkan build and needs reselecting.

> **Check the endpoint, not the app.** On this machine `Bionic.exe` - not LM
> Studio - is what actually serves `127.0.0.1:1234`, despite the `lms` CLI
> being installed and appearing to work. `lms ps` can look healthy while the
> pipeline gets HTTP 400 "model not loaded". Always verify with:
>
> ```bash
> curl -s http://127.0.0.1:1234/v1/models
> ```
>
> Every model named in `config/pipeline.yaml`, plus `google/gemma-4-e2b` (used
> by the rhyme and scene-description stages), must appear in that list under
> exactly that name.

## Running it

### Quick start

Write a new poem and make a video of it, start to finish:

```bash
.venv/Scripts/python.exe -m src.orchestration --generate --lines 8
```

That takes about an hour and needs nothing but the LLM endpoint running. The
finished video lands in `data/output/`.

Use your own poem instead:

```bash
.venv/Scripts/python.exe -m src.orchestration data/my_rhyme.txt
```

Or via make, which is equivalent:

```bash
make run                          # runs data/rhyme.txt (the bundled sample)
make run INPUT=data/my_rhyme.txt  # runs a different rhyme
```

### All options

| Flag | What it does |
|---|---|
| `<input>` | Path to a rhyme text file, one line per line. Omit when using `--generate`. |
| `--generate` | Write a new poem first instead of reading one from a file. |
| `--lines N` | With `--generate`, how many lines to write (default 16). Use an even number - poems are couplets, and an odd count leaves a line with no rhyme partner. |
| `--seed "Title"` | With `--generate`, take inspiration from a particular seed rhyme (see `data/seed_rhymes.json`). |
| `--name NAME` | Base name for output files. Defaults to the input file's stem. |
| `--force` | Re-run every stage even if its output already exists. |
| `--motion` | Animate each scene with AnimateDiff. **Hours, not minutes** - see below. |
| `--no-motion` | Ken Burns pan/zoom over the stills. This is the default. |

### Common recipes

```bash
# Fast: new 8-line poem, stills with pan/zoom               (~1 hour)
python -m src.orchestration --generate --lines 8

# Fastest useful test: 4 lines                              (~25 min)
python -m src.orchestration --generate --lines 4

# Your own poem, forcing a clean regeneration
python -m src.orchestration data/my_rhyme.txt --force

# Riff on a specific traditional rhyme
python -m src.orchestration --generate --seed "Five Little Ducks"

# Animated, on a GPU that can take it                       (~6 hours for 8 lines)
python -m src.orchestration --generate --lines 8 --motion
```

### Resuming

The pipeline resumes automatically: if a run is interrupted, or you re-run the
same input, any stage whose output already exists is skipped. That is what makes
a failed six-hour render recoverable rather than a total loss.

Resume is safe across different runs: each stage stamps its output with a hash
of the input it consumed, so a cached manifest is only reused when it provably
belongs to the poem being rendered. A different poem - even one of exactly the
same length - regenerates rather than inheriting the previous run's images. Pass
`--force` to re-run everything regardless, or `make clean` to wipe generated
artifacts entirely.

### Before you start a long run

1. **Check the LLM endpoint is actually up** - `curl -s http://127.0.0.1:1234/v1/models`.
   A dead endpoint doesn't fail the run; it silently degrades to a fallback poem
   with no metre rewrite.
2. **Do a `--no-motion` pass first** if you have changed anything. It exercises
   every stage in ~25 minutes instead of hours. This is how the character
   auto-registration bug was caught before it wasted a 10-hour render.

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

Then `make run INPUT=data/my_rhyme.txt`. Turning it into a scene-by-scene
script happens automatically as the first pipeline stage - there is no separate
"generate a script" step.

Outputs are named after the input file's stem (`my_rhyme.txt` ->
`my_rhyme.json` / `my_rhyme.mp4`), except the shared per-stage manifests noted
below.

## Where output files are saved

| Stage | Output |
|---|---|
| Generated poem | `data/generated/<name>.txt` (only with `--generate`) |
| Script | `data/scripts/<name>.json` (generated, not tracked) - scenes with line, speaker, stage_direction, scene_description, mood, duration |
| Images | `data/images/scene_NNN.png` + `data/images/manifest.json` |
| Motion | `data/motion/scene_NNN.mp4` + `data/motion/manifest.json` |
| Audio | `data/audio/scene_NNN.wav` + `data/audio/manifest.json` |
| **Final video** | `data/output/<name>.mp4` |
| Logs | `logs/pipeline.log` (structured JSON, one line per event) |
| Characters | `data/characters/` - reference portraits, persisted across runs |

**Per-scene files are not namespaced by rhyme name** - every run writes to
`data/images/`, `data/audio/` and `data/motion/`, so a new run overwrites the
previous run's scene files. Copy anything worth keeping before re-running.

**Old files are not silently reused, though.** Each stage stamps its manifest
with a hash of the input it consumed - the script stage with the poem's hash,
the later stages with the script's - and a stamp that does not match marks the
output stale, so it is regenerated. That covers:

- a different poem of the same length (scene count matched, so this used to
  sing the new words over the old pictures)
- the same poem edited in place
- a different poem run under the same `--name`
- manifests left by any older version of the pipeline, which carry no stamp

`--force` regenerates regardless, and `make clean` wipes everything generated.

## Configuration

Toggles live in `config/pipeline.yaml`, `config/sd_config.yml`,
`config/tts_config.yml`:

- `script.creative_rewrite` / `script.metre_rewrite` - creative line rewriting,
  and rewriting to a shared syllable count so lines sing to one melody
- `script.scene_planning.target_scene_duration_s` - lower gives more, shorter
  scenes (and so more images)
- **`motion.enabled`** - AnimateDiff clips instead of Ken Burns pans (default
  off; use `--motion` / `--no-motion` to override per run). This is
  the big one: **on, a run takes hours; off, minutes.** See below.
- `music.enabled` / `music.volume` / `music.duck` - MusicGen backing track
- `visual.character_bank` / `visual.ip_adapter` - hold a character's appearance
  steady across scenes and across episodes
- `backend` (in tts_config.yml) - `bark` sings; `piper` speaks and is warped
  into a melodic contour, and is far faster

### Animated scenes vs. Ken Burns

The motion stage is **off by default**, and each scene instead gets a slow
pan/zoom across its still image. Override per run without editing config:

```bash
python -m src.orchestration data/my_rhyme.txt --no-motion   # stills + Ken Burns (minutes)
python -m src.orchestration data/my_rhyme.txt --motion      # AnimateDiff clips (hours)
```

Measured on an 8-scene poem, GTX 1050 4GB:

| | `--no-motion` | `--motion` |
|---|---|---|
| motion stage | - | 5 h 8 min |
| **total run** | **~30 min** | **6 h 13 min** |
| per second of video | ~1 min | ~13 min |

`--motion` works and produces genuine animation, but on this card it renders at
384px from 16 frames looped to fill each scene, which visibly shimmers and has a
loop seam. It is parked as future work for a larger GPU - see STATUS.md for what
has already been ruled out, and what to try first when revisiting.

## Other Makefile commands

```bash
make test              # run the test suite (fast, ~30s)
pytest -m slow         # the real end-to-end pipeline test (GPU; run it alone)
make clean             # remove all generated artifacts (keeps inputs, characters, models, venv)
make clean-characters  # forget every character's established appearance
```

## Not included: YouTube upload

The pipeline stops at the finished video. Publishing to YouTube needs your
own API credentials and is intentionally a separate manual step - see the
`GPT-14` issue in Linear for that design when you're ready for it.
