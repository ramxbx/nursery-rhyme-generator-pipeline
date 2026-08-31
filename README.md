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

To have the pipeline write its own poem instead of supplying one, pass
`--generate`. `--lines` caps the length, which matters a great deal with
`--motion` - every line costs a Bark take, an image, and ~39 minutes of
animation:

```bash
.venv/Scripts/python.exe -m src.orchestration --generate --lines 4
.venv/Scripts/python.exe -m src.orchestration --generate --seed "Five Little Ducks"
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
| Motion | `data/motion/scene_NNN.mp4` + `data/motion/manifest.json` |
| Audio | `data/audio/scene_NNN.wav` + `data/audio/manifest.json` |
| **Final video** | `data/output/<name>.mp4` |
| Logs | `logs/pipeline.log` (structured JSON, one line per event) |
| Characters | `data/characters/` - reference portraits, persisted across runs |

**Note**: the per-scene image/audio files and their manifests are *not*
namespaced by rhyme name - running a second, different rhyme without
`--force` will reuse/overwrite the first run's scene images and audio if the
manifests already exist for that stage. Run `make clean` between different
rhymes, or use `--force`, to avoid stale cross-run reuse.

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
make test              # run the test suite (pytest)
make clean             # remove all generated artifacts (keeps inputs, characters, models, venv)
make clean-characters  # forget every character's established appearance
```

## Not included: YouTube upload

The pipeline stops at the finished video. Publishing to YouTube needs your
own API credentials and is intentionally a separate manual step - see the
`GPT-14` issue in Linear for that design when you're ready for it.
