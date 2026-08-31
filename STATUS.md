# Project status

Working notes for picking this up cold. Linear holds the per-issue detail and
commit messages hold the reasoning; this file is the map.

**Last updated:** 2026-08-31 · branch `project-status-and-spikes`, 12 commits
ahead of master · 162 tests passing in ~30s, plus 1 slow end-to-end test that is
deselected by default (`pytest -m slow`).

## What the pipeline does

`python -m src.orchestration --generate` writes a poem and builds a video from
nothing. Stages, each an isolated subprocess so the GPU is released between them:

| stage | what | roughly |
|---|---|---|
| rhyme | writes an original poem from a seed rhyme's subject/setting/mood | seconds |
| script | scenes, metre rewrite, SD-ready descriptions | ~1 min/scene |
| visual | SD1.5 + hires fix, IP-Adapter character conditioning | ~2 min/scene |
| motion | AnimateDiff clips — **OFF by default**, `--motion` to enable | ~39 min/scene |
| audio | Bark singing, per-line, with fallback to Piper | ~3 min/line |
| animate | ffmpeg: Ken Burns or motion clips, crossfades, MusicGen bed, subtitles | ~10 min |

Measured on 8 scenes: **~1 h without motion, 6 h 13 min with it.** Use
`--lines 4` when iterating — every extra line costs a Bark take and an image.

## Proven, with numbers

- **Audio levels.** Was −32.4 dBFS with words buried; now −17.6, zero clipping,
  voice ~20 dB above the music. Three causes: Bark's unnormalised output,
  `amix` halving every input by default, and no loudness normalisation.
- **Framing.** Shots cycle wide/medium/close; each has its own CLIP bar because a
  wide shot can't reach a close-up's subject-similarity score. Mean attempts 5 → 1.4.
- **Hires fix.** 512 composition → upscale → re-render at low denoise, 768 out.
  This is what makes eyes possible on a subject that's 60 px tall at 512.
- **IP-Adapter character conditioning.** The only thing that moved identity at all.
  Lamb poem: all 6 scenes cleared their bar first try, mean CLIP 0.290.
- **MusicGen ceiling.** Hard limit of 40.96 s; generation capped at 38 s and looped.

## Disproven — don't retry these

- **Text descriptors cannot enforce character identity on SD1.5.** A descriptor
  including "wearing a blue neckerchief" was verified in all four prompts of a
  test run and absent from all four images. Species and colour hold; specifics don't.
- **Processing speech cannot produce singing.** Three iterations of score-driven
  Piper warping were rejected by ear, while pitch error *improved* 3.15 → 0.04
  semitones. Singing differs in phonation, vibrato, formant tuning and breath;
  Rubber Band can't manufacture those. See `spikes/sing_*.py`.
- **CLIP cannot see anatomy defects.** On a two-snout pig it scored "two noses"
  *above* "one nose" and "deformed" lowest of all probes. More best-of-N attempts
  buy nothing against this failure class.
- **Per-character LoRA isn't needed** (GPT-38) now that IP-Adapter works.
- **Compositing a sharp subject over an animated background looks worse**, not
  better, than animating the whole scene. The subject reads as a sticker on a
  moving plate however much drift is added. Judged by eye and removed; the code
  is recoverable from commit 4615b4d (`src/utils/compositing.py`,
  `build_composite_scene_clip`).
- **The end-to-end test must not run automatically.** It calls the real
  `run_pipeline`, loading SD1.5 onto the GPU and writing the shared
  data/images and data/audio manifests - so `pytest` competed with an actual
  run for a 4GB card and for the same files, and both failed. It was gated only
  on the LLM endpoint being reachable, so it was invisible whenever that was
  down and fired the moment it came up, which also made the suite look like it
  hung (30s vs 13min). Now marked `slow` and deselected by default.
- **Scene count does not identify a run.** Resume compared only the number of
  entries in a cached manifest, so two poems of the same length matched and the
  second run reused the first's images - new words sung over old pictures, with
  nothing in the log to say so. Stages now stamp their output with a hash of the
  input they consumed. Only differing-length runs were ever safe.
- **Wide aspect ratios break SD1.5 composition**, at any pixel count. 512x288
  costs exactly what 384x384 costs (148 vs 147 s/frame, same pixels) and wastes
  nothing to the 16:9 crop, but the model draws a herd of duplicated subjects
  instead of one. Detail must come from an upscaler, not frame shape. Details
  under GPT-26 below.

## Open

| issue | state | note |
|---|---|---|
| GPT-26 AnimateDiff | In Progress | **Integrated; one full 4-scene run done at 36 min/scene.** Clarity work in flight — see below |
| GPT-41 score-driven singing | Backlog | ACE-Step (MIT, <4 GB) or DiffSinger. Bark still invents its own melody |
| GPT-40 anatomy defects | Backlog | Needs a metric that sees defects, or inpainting |
| GPT-14 YouTube upload | Todo | Never started |
| GPT-22 / GPT-25 | Backlog | Older style + prompt-ordering items |

## GPT-26 in detail — AnimateDiff IS viable, config chosen

Earlier this file said AnimateDiff was impractical at 508 s/frame. That was
measured at 512x512/20 steps and is no longer the operating point.

**Chosen config: 384x384, 10 steps, 16 frames, 2-second clips.**
Judged best by eye against seven alternatives; nothing else came close.
Costs ~40 min per 2-second scene, so a 6-scene poem is ~4 hours of animation.

### The grid (all 2-second clips, same lamb scene, same prompt)

| res | steps | frames | time | per frame | verdict |
|---|---|---|---|---|---|
| 256² | 8 | 8 | 6.4 min | 48 s | dissolved |
| 256² | 10 | 8 | 7.6 min | 57 s | dissolved |
| 256² | 10 | 16 | 16 min | 60 s | dissolved |
| 320² | 10 | 16 | 28 min | 105 s | not chosen |
| 384² | 6 | 16 | 23.7 min | 89 s | not chosen |
| 384² | 8 | 16 | 29.4 min | 110 s | not chosen |
| 384² | 10 | 8 | 18 min | 135 s | not chosen |
| **384²** | **10** | **16** | **40 min** | **150 s** | **chosen** |
| 384² | 12 | 16 | 45 min | 168 s | worse than 10 |

### What the grid shows

- **Resolution dominates cost.** 384² runs 135-150 s/frame whether you diffuse 8
  frames or 16; 256² runs 48-60 s/frame either way.
- **Frame count is nearly free** per frame, so it trades motion smoothness against
  wall-clock almost linearly — no quadratic blowup from temporal attention.
- **Cost does not scale with pixel area.** 320² is 1.56x the area of 256² but took
  1.75x the time; there is fixed per-step overhead that low resolutions cannot escape.
- **512² is SD1.5 native.** Everything below it is off-distribution, and the
  "watercolour" look is the model degrading gracefully. 256² degrades past legibility.
- **CPU offload does not share compute, only storage.** Offloaded 508 s/frame vs
  no-offload 527 s/frame at 512²/20 steps — the GPU does all the maths either way.

### Step count is at an optimum, not a ceiling

12 steps was tested and judged **worse** than 10, so more denoising is not simply
better. The encoded file sizes point the same way: 458KB at 10 steps against
241KB at 12, and across this grid larger files have tracked more retained detail
(H.264 compresses smooth, low-detail footage harder). The extra steps appear to
smooth away the texture that gives the 10-step version its character.

No reason to test 15. The config is converged.

### PARKED as future work — needs a bigger GPU

The technique works and is fully integrated; it is the hardware that does not
suit it. Turned OFF by default (`motion.enabled: false`), kept in the tree, and
switchable per run with `--motion` / `--no-motion`.

Two reasons, both measured on an 8-scene run:

1. **Cost.** 38.6 min/scene, 5.14 h of a 6h13m run - 83% of wall clock, for
   28.7 s of video. That is ~13 min of compute per second of output.
2. **Flicker, and it is not a tuning fault.** 16 frames diffused at 384px, each
   denoised independently, looped 2-3x per scene and motion-interpolated to
   24fps. Independent per-frame denoising shimmers on fine detail; the loop adds
   a periodic seam. Both are inherent to those numbers. Fixing them means more
   frames at higher resolution, which is exactly what a 4GB card cannot do.

Everything cheap has already been tried and is recorded below: the step count
and frame count are at an optimum rather than a ceiling (12 steps and 8 frames
both judged worse), and wide aspect ratios are disproven. What is left needs
VRAM.

**When revisiting on better hardware, in order:**

- More frames before more resolution. 32-48 frames removes the loop entirely -
  the seam is currently the most visible artefact, and it exists only because a
  2-second clip has to cover a 3-6 second scene.
- 512x512, SD1.5's native resolution, which should reduce the per-frame
  instability that reads as shimmer.
- Only then a learned upscaler (see the Real-ESRGAN notes) for detail.

Do not re-run the resolution/steps sweep. Do not retry wide aspect ratios.

### Integration status — the run that was completed

Wired into the pipeline as stage 2b (`src/agents/motion_agent.py`), on by
default via `motion.enabled`. A scene with a clip uses it; a scene whose
generation OOMed falls back to a Ken Burns pan over its still, per scene rather
than all-or-nothing.

First full animated run, 4-line lamb poem, 2026-08-30:

| stage | wall clock |
|---|---|
| script | 5.5 min |
| visual | 10 min (4 images, all first-attempt) |
| **motion** | **2.4 h — 36 min/scene** (2152/2144/2140/2338 s) |
| audio + assembly | ~13 min |

36 min/scene against the 40 predicted. Output: 14.9 s, 1920x1080, 4 scenes.

### What the first full run revealed about clarity

The generated frames are **better than the delivered video**. Two losses sit
between them, both in delivery rather than diffusion:

1. **44% of every frame is cropped away.** The clip is square, the video is
   16:9, so `build_motion_scene_clip` scales 384x384 up 5x to cover 1920x1080
   and centre-crops. Only 216 of 384 source rows survive; **96% of delivered
   pixels are lanczos invention**, and compositional detail at the top and
   bottom of frame is paid for and then discarded.
2. **Wide shots do not survive 384px.** Scene 1's `wide establishing shot` gave
   the lamb ~40 source pixels — an unreadable grey smudge at 1080p. Scene 2's
   medium shot, same resolution, same run, kept the lamb, the cottage and its
   window frames all legible. Shot type, not resolution, is what broke scene 1.

Fixed for (2): `ANIMATED_SHOT_CYCLE` in `visual_agent.py` drops wide shots
whenever the motion stage is on.

**Tested and DISPROVEN for (1).** `motion.width`/`motion.height` are now config
keys, and 512x288 is 147,456 pixels - identical to 384x384 - so it should have
been the same cost with nothing cropped. Cost held up exactly: 39.4 min against
the control's 39.1 (148 vs 147 s/frame), same scene, same seed, same prompt,
same delivery path (`spikes/aspect_spike.py`).

The image did not. At 512x288 SD1.5 stopped drawing one lamb and drew **a herd
of ghostly half-lambs smeared along the bottom of the frame** - subject
duplication along the long axis. Same failure class as GPT-28: strong aspect
deviation triggers it as readily as excess resolution does, independent of
pixel count.

What it did show: the background at 512 wide is genuinely more detailed - real
thatch texture, readable stonework - so "more horizontal resolution buys
detail" was correct. It just cannot be bought this way, because composition
breaks before the detail helps. **Do not retry wide aspects on SD1.5 at this
scale.** Detail has to come from a post-hoc upscaler, which does not touch what
the model composes.

Caution recorded because it nearly misled this decision: by laplacian variance
the broken clip scores 22.7 against the good one's 9.9, i.e. 2.3x "sharper".
The metric reads the duplication smear as detail. Same trap as CLIP scoring
anatomically broken subjects highly (GPT-40) and the singing spikes whose pitch
error fell to 0.04 semitones while sounding worse. Judge motion output by eye.

### Character auto-registration leaked a fourth time (fixed)

Each scene resolved its own subject - the first comma-segment of its own
description - so a four-line poem about an egg registered "village stones",
"shadows dancing and creeping across the wall" and "sleepy little birdies
sleeping on the ground" as characters. CLIP then verified each image against a
phrase that is not a subject, scored ~0.17 against a 0.26 bar, and burned all
five attempts on three of four scenes.

The "one character per poem" logic already existed and its comment described
this exact failure - but it lived inside the IP-Adapter branch and the
per-scene loop never used its result. Now resolved once outside that branch and
used by every scene for descriptor, seed and CLIP target. Scenes register
nothing, so there is no longer any per-scene inference left to leak.

Measured on the same poem before and after:

| | before | after |
|---|---|---|
| characters registered | 4 (3 bogus) | 1 |
| image generations | 16 | 6 |
| scenes at the attempt cap | 3 of 4 | 0 |
| CLIP scores | 0.175-0.284 | 0.267-0.302 |
| visual stage | 25 min | 12 min |

Caught by running a fast `motion.enabled: false` pass before committing to a
long render. **Do that before every long run** - it exercises every stage in
~25 minutes instead of hours.

### The seed search records the wrong seed (fixed)

`generate_best_image` tries up to 5 consecutive seeds and keeps the highest
CLIP scorer, but the manifest recorded the **base** seed rather than the
winner. The hires pass refined from the base seed, and the motion stage
regenerated the whole scene from it - so a scene that took three attempts was
animated from a composition CLIP had already rejected.

Latent rather than benign: every scene of the lamb run passed first try
(attempts=1 throughout), so base and winning seed coincided. The framing fix
that dropped mean attempts to ~1.4 is what had been hiding it.

Not recoverable after the fact either: on the early-return path the winner is
`seed + attempts - 1`, but when all five candidates miss the bar the best can
be any of them and that index was never recorded. `generate_best_image` now
returns the winning seed explicitly.

### The motion stage never uses the still image

`animate_scene` is pure text-to-video: it reuses the image stage's prompt and
seed so the animation depicts the same moment, but the 768x768 still itself is
discarded. When motion is on, the image stage's real output is CLIP validation
and a prompt — not pixels. Relevant to any future "sharpen the subject" idea,
and to whether the image stage needs the hires pass at all in animated runs.

### Encoder chain

A finished video is encoded three times — scene clips, crossfade concat,
subtitle burn-in — and all three ran at libx264's default CRF 23, so the two
intermediates discarded detail before the final encode saw it. (The music mix
copies the video stream and costs nothing.) Now centralised in `_x264()` in
`ffmpeg_helper.py`: CRF 12 intermediates, CRF 18 final, preset `slow`.

Measured on a re-assembly of the same manifests: sharpness (laplacian variance)
206.4 -> 212.0, i.e. **+2.7%**, for 10.9MB -> 22.1MB. Real but marginal, which
confirms the codec was never the main loss. Kept because YouTube re-encodes on
upload (GPT-14), so a cleaner source is worth more than the local file size.

## Gotchas

- `data/images/`, `data/audio/`, `data/output/`, `data/characters/` are gitignored
  and get overwritten by every run. Copy anything worth keeping.
- Stage manifests aren't namespaced per run; a count mismatch marks them stale
  and re-runs (fixed, but that's why).
- **`Bionic.exe` serves the OpenAI-compatible API on `127.0.0.1:1234`**, which is
  what `config/pipeline.yaml` points at — not LM Studio, despite the `lms` CLI
  being installed and appearing to work. Check with
  `curl -s http://127.0.0.1:1234/v1/models` rather than `lms ps`; the models the
  pipeline names (`lfm2-1.2b-bench`, `gemma-4-e2b-bench`) must appear in that list.
- Ollama may also be running on `:11434`. Nothing here uses it, and it holds
  ~3 GB of host RAM even with no models loaded — worth stopping before a long run.
- Python block-buffers stdout when redirected — use `python -u` for background
  runs or you see nothing until exit.
- Don't pipe a pipeline run through `tail`: it masks the real exit code. That
  once hid a crash and I reported the run as successful.
