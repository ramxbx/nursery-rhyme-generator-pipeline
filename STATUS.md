# Project status

Working notes for picking this up cold. Linear holds the per-issue detail and
commit messages hold the reasoning; this file is the map.

**Last updated:** 2026-08-15 · master at `a3d3626` (PR #6 merged) · 134 tests passing
(plus 1 slow end-to-end test, deselected by default — it runs the real pipeline
and takes ~15 min).

## What the pipeline does

`python -m src.orchestration --generate` writes a poem and builds a video from
nothing. Stages, each an isolated subprocess so the GPU is released between them:

| stage | what | roughly |
|---|---|---|
| rhyme | writes an original poem from a seed rhyme's subject/setting/mood | ~2 min |
| script | scenes, metre rewrite, SD-ready descriptions | ~5 min |
| visual | SD1.5 + hires fix, IP-Adapter character conditioning | ~5 min/scene |
| audio | Bark singing, per-line, with fallback to Piper | ~1.5 min/line + retries |
| animate | ffmpeg: Ken Burns, crossfades, MusicGen bed, karaoke subtitles | ~5 min |

A 6-line poem is ~45 min end to end. A 16-line poem is ~2 h. Use
`--lines 6` when iterating — every extra line costs a Bark take and an image.

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

## Open

| issue | state | note |
|---|---|---|
| GPT-26 AnimateDiff | In Progress | **Runs, but 508 s/frame.** See below |
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

### Integration status

Not wired into `animate_agent` yet. Spike lives at `spikes/animate_spike2.py`,
parameterised by `AD_FRAMES` and `AD_STEPS` env vars; resolution is a constant in
the file. Frames export at `FRAMES/2` fps so a clip is always 2 seconds, then
ffmpeg `minterpolate` (mci/aobmc/bidir) fills to 24 fps in ~5 s.

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
