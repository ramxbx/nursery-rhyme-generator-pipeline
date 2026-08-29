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

## GPT-26 in detail — where it actually stands

AnimateDiff **works** on this machine. 8 frames at 512², IP-Adapter conditioned
on the lamb, produced `anim_scene_003.mp4`.

The problem is speed: **4067 s for 8 frames = 508 s/frame.** One second of
animation took 68 minutes. A 6-scene poem at 8 frames each would be ~7 hours.

VRAM peaked at 2.16 GB of 4.29 and *fell* to 0.90 GB during generation — the card
is barely working. The cost is `enable_model_cpu_offload` shuttling weights for
every step of every frame.

**Untested next step:** `spikes/animate_spike2.py` inverts this — loads straight
to CUDA, no offload, VAE slicing only. Should use ~3.5 GB of VRAM and far less
host RAM. If it reaches even 30 s/frame that's a 17× speedup and changes the answer.

Two earlier failures were misdiagnosed as memory limits; both were actually a
missing `opencv-python` (now installed) which made `export_to_video` fail *after*
generating everything. Capture the real exception before theorising about memory.

## Gotchas

- `data/images/`, `data/audio/`, `data/output/`, `data/characters/` are gitignored
  and get overwritten by every run. Copy anything worth keeping.
- Stage manifests aren't namespaced per run; a count mismatch marks them stale
  and re-runs (fixed, but that's why).
- LM Studio must be running with `lfm2-1.2b-bench` and `gemma-4-e2b-bench` loaded.
  `lms ps` to check, `lms load <model> --identifier <name>` to fix.
- Python block-buffers stdout when redirected — use `python -u` for background
  runs or you see nothing until exit.
- Don't pipe a pipeline run through `tail`: it masks the real exit code. That
  once hid a crash and I reported the run as successful.
