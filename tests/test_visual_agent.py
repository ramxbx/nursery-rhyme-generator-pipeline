"""Unit tests for visual_agent (GPT-16). The SD pipeline is stubbed - these
test deterministic logic, not live generation."""
from src.agents import visual_agent as va


class FakeTokenizer:
    """Whitespace tokenizer standing in for CLIP's - enough to exercise the
    truncation branch without loading the real pipeline."""

    def __call__(self, text, add_special_tokens=True):
        return {"input_ids": text.split()}

    def decode(self, ids):
        return " ".join(ids)


class FakePipe:
    tokenizer = FakeTokenizer()


def test_seed_for_character_is_deterministic():
    assert va.seed_for_character("Star") == va.seed_for_character("Star")
    assert va.seed_for_character("Star") == va.seed_for_character("  star  ")  # normalized


def test_seed_for_character_differs_across_names():
    assert va.seed_for_character("Star") != va.seed_for_character("Narrator")


def test_build_prompt_includes_style_anchor_and_description():
    prompt = va.build_prompt(FakePipe(), {"scene_description": "a sheep in a field"})
    assert prompt.startswith(va.STYLE_ANCHOR)
    assert "a sheep in a field" in prompt


def test_build_prompt_falls_back_when_description_missing():
    prompt = va.build_prompt(FakePipe(), {})
    assert va.STYLE_ANCHOR in prompt
    assert "picture-book" in prompt


def test_build_prompt_truncates_to_clip_budget():
    long_description = " ".join(f"word{i}" for i in range(200))
    prompt = va.build_prompt(FakePipe(), {"scene_description": long_description})
    assert len(prompt.split()) == va.CLIP_TOKEN_BUDGET


def test_build_prompt_keeps_style_anchor_when_truncating():
    long_description = " ".join(f"word{i}" for i in range(200))
    prompt = va.build_prompt(FakePipe(), {"scene_description": long_description})
    assert prompt.startswith(va.STYLE_ANCHOR)  # style anchor survives truncation


def test_shot_cycles_so_a_video_is_not_all_close_ups():
    """The bug this fixes: every scene of every video was a portrait of the
    subject, so nothing the poem named ever appeared in frame."""
    shots = [va.shot_for_scene(i).name for i in range(1, 17)]
    assert shots[0] == "wide", "the opening scene should establish the setting"
    assert len(set(shots)) == 3, f"expected all three framings across 16 scenes, got {set(shots)}"
    assert shots.count("close") < len(shots) / 2, "close-ups should not dominate"


def test_shot_assignment_is_deterministic():
    """Re-running the same poem must frame it identically, or resumed runs
    would mix framings from different attempts."""
    assert [va.shot_for_scene(i).name for i in range(1, 9)] == \
           [va.shot_for_scene(i).name for i in range(1, 9)]


def test_wide_shots_are_judged_on_a_lower_clip_bar_than_close_ups():
    """CLIP subject-similarity measures how much of the frame is the subject,
    so a wide shot cannot reach a close-up's score however good it is."""
    assert va.SHOTS["wide"].clip_good_enough < va.SHOTS["medium"].clip_good_enough
    assert va.SHOTS["medium"].clip_good_enough < va.SHOTS["close"].clip_good_enough


def test_framing_reaches_the_prompt_and_style_still_leads():
    class FakeTok:
        def __call__(self, text, add_special_tokens=False):
            return {"input_ids": list(range(len(text.split())))}

    class FakePipe:
        tokenizer = FakeTok()

    scene = {"scene_description": "pink pig, green barnyard, blue sky"}
    prompt = va.build_prompt(FakePipe(), scene, va.SHOTS["wide"])
    assert prompt.startswith(va.STYLE_ANCHOR)
    assert "wide establishing shot" in prompt


def test_style_anchor_no_longer_forces_close_framing():
    """The anchor previously hardcoded 'subject close up and centered, simple
    background' onto every scene, which is what caused the problem."""
    assert "close up" not in va.STYLE_ANCHOR
    assert "simple background" not in va.STYLE_ANCHOR


def test_hires_target_is_a_multiple_of_eight():
    """SD's VAE downsamples by 8; a non-multiple dimension is silently cropped,
    which would shift the refined image against the composition it started from."""
    for w, h, scale in [(512, 512, 1.5), (512, 288, 1.5), (500, 300, 1.37)]:
        tw, th = va.hires_size(w, h, scale)
        assert tw % 8 == 0 and th % 8 == 0


def test_hires_actually_upscales():
    assert va.hires_size(512, 512, 1.5) == (768, 768)


def test_hires_never_collapses_to_zero():
    assert va.hires_size(4, 4, 0.1) == (8, 8)


def test_character_identity_reaches_the_prompt_ahead_of_the_scene():
    """Identity and framing must precede the scene's own wording so they are
    the parts that survive CLIP's 77-token truncation."""
    class FakeTok:
        def __call__(self, text, add_special_tokens=False):
            return {"input_ids": list(range(len(text.split())))}

    class FakePipe:
        tokenizer = FakeTok()

    scene = {"scene_description": "little pig, rolling in mud, barnyard"}
    prompt = va.build_prompt(FakePipe(), scene, va.SHOTS["medium"], "pink pig, dark spot on flank")
    assert prompt.index("pink pig, dark spot") < prompt.index("rolling in mud")


def test_scene_subject_is_not_repeated_when_the_bank_supplies_identity():
    """The description's first segment is the subject, so leaving it in front of
    a canonical descriptor names the subject twice - re-weighting its colour
    terms and bleeding them across the frame."""
    class FakeTok:
        def __call__(self, text, add_special_tokens=False):
            return {"input_ids": list(range(len(text.split())))}

    class FakePipe:
        tokenizer = FakeTok()

    scene = {"scene_description": "little pig, rosy pink nose, roaming in a wooden barn"}
    prompt = va.build_prompt(FakePipe(), scene, va.SHOTS["wide"], "small round rosy pink pig, curled tail")
    assert prompt.count("little pig") == 0
    assert "roaming in a wooden barn" in prompt
    assert "small round rosy pink pig" in prompt


def test_scene_description_is_untouched_without_a_character():
    class FakeTok:
        def __call__(self, text, add_special_tokens=False):
            return {"input_ids": list(range(len(text.split())))}

    class FakePipe:
        tokenizer = FakeTok()

    scene = {"scene_description": "little pig, rosy pink nose, in a barn"}
    prompt = va.build_prompt(FakePipe(), scene, va.SHOTS["wide"])
    assert "little pig" in prompt


def test_a_long_first_segment_is_kept_because_it_carries_more_than_the_subject():
    """If the model packs action into the opening segment instead of just the
    subject label, dropping it would lose what the scene is actually about."""
    class FakeTok:
        def __call__(self, text, add_special_tokens=False):
            return {"input_ids": list(range(len(text.split())))}

    class FakePipe:
        tokenizer = FakeTok()

    scene = {"scene_description": "a pig leaping over a wooden fence at sunset, barnyard"}
    prompt = va.build_prompt(FakePipe(), scene, va.SHOTS["medium"], "pink pig, curled tail")
    assert "leaping over a wooden fence" in prompt


def test_reference_key_comes_from_the_scene_not_the_canonical_descriptor():
    """The bug: the loop replaced `subject` with the canonical descriptor before
    deriving the lookup key, so character_key ran on
    "...pink pig, wearing a blue neckerchief, dark spot on left flank" and
    returned "flank". The reference lookup then missed on every scene and the
    adapter was handed nothing, which crashed the UNet."""
    from src.utils.character_bank import character_key

    scene_subject = "little pig"
    canonical = "small round pink pig, wearing a blue neckerchief, dark spot on left flank"
    assert character_key(scene_subject) == "pig"
    assert character_key(canonical) == "flank", "documents why the key must not come from the descriptor"


def test_reference_prompt_leads_with_the_subject_not_portrait_framing():
    """The bug: the reference prompt led with "character reference portrait,
    full body, standing, facing viewer" - all human-portrait cues - so asking
    for "tiny spider" produced a boy in a brown jacket, which IP-Adapter then
    propagated into every scene."""
    for cue in ("character reference portrait", "standing", "facing viewer"):
        assert cue not in va.REFERENCE_FRAMING, f"{cue!r} biases the reference toward a human"


def test_reference_is_held_to_a_stricter_bar_than_a_scene():
    """A wrong reference poisons every scene that character appears in, which is
    far worse than one bad scene image."""
    assert va.REFERENCE_CLIP_MIN > va.SHOTS["wide"].clip_good_enough
    assert va.REFERENCE_CLIP_MIN >= va.CLIP_GOOD_ENOUGH


def test_animated_runs_never_use_a_wide_shot():
    """A wide shot re-generated by AnimateDiff at 384px gives the subject ~40
    pixels, which upscaled to 1080p is an unreadable smudge - the exact failure
    seen in scene 1 of the first full animated run."""
    shots = [va.shot_for_scene(i, animated=True).name for i in range(1, 17)]
    assert "wide" not in shots, f"wide shot survived into an animated run: {shots}"


def test_animated_runs_still_vary_their_framing():
    """Dropping wide must not collapse every scene onto one framing - that was
    the original bug the shot cycle exists to fix."""
    shots = [va.shot_for_scene(i, animated=True).name for i in range(1, 17)]
    assert len(set(shots)) > 1, f"animated framing collapsed to {set(shots)}"


def test_still_runs_keep_their_establishing_shot():
    """Without the motion stage there is no 384px downscale, so wide shots are
    still worth having."""
    assert va.shot_for_scene(1, animated=False).name == "wide"
    assert va.shot_for_scene(1).name == "wide", "the default must stay unchanged"


def test_animated_shot_assignment_is_deterministic():
    assert [va.shot_for_scene(i, animated=True).name for i in range(1, 9)] == \
           [va.shot_for_scene(i, animated=True).name for i in range(1, 9)]


def _search_stub(monkeypatch, scores):
    """generate_best_image driven by a canned score per consecutive seed."""
    seen = []

    def fake_generate(pipe, prompt, seed, sd_config, reference=None):
        seen.append(seed)
        return f"image@{seed}", pipe

    monkeypatch.setattr(va, "generate_image", fake_generate)
    monkeypatch.setattr(va, "clip_subject_similarity",
                        lambda verifier, image, subject: scores[seen.index(int(image.split("@")[1]))])
    return seen


def test_the_winning_seed_is_returned_not_the_base_seed(monkeypatch):
    """The seed the search STARTED from is not the seed that made the image.
    The hires pass refines from it and AnimateDiff regenerates the whole scene
    from it, so recording the base seed animated a composition CLIP rejected."""
    _search_stub(monkeypatch, [0.10, 0.15, 0.99])
    image, _, score, attempts, winning = va.generate_best_image(
        object(), "a prompt", 500, {}, object(), "lamb", good_enough=0.9)
    assert attempts == 3
    assert winning == 502, f"expected the third seed, got {winning}"
    assert image == "image@502"


def test_the_winning_seed_is_right_when_no_candidate_clears_the_bar(monkeypatch):
    """Exhausting the budget keeps the highest scorer, which is not necessarily
    the last one tried - so the winner cannot be recomputed from `attempts`."""
    scores = [0.10, 0.42, 0.20, 0.31, 0.15]
    _search_stub(monkeypatch, scores)
    image, _, score, attempts, winning = va.generate_best_image(
        object(), "a prompt", 700, {}, object(), "lamb", good_enough=0.9)
    assert attempts == va.CLIP_ATTEMPTS
    assert winning == 701, f"best score was the second candidate, got seed {winning}"
    assert score == 0.42


def test_a_first_try_win_returns_the_base_seed(monkeypatch):
    """The case that hid this bug: every scene of the lamb run passed first
    try, so base and winning seed coincided."""
    _search_stub(monkeypatch, [0.99])
    _, _, _, attempts, winning = va.generate_best_image(
        object(), "a prompt", 900, {}, object(), "lamb", good_enough=0.9)
    assert (attempts, winning) == (1, 900)


def _script(descriptions):
    return {"scenes": [{"speaker": "narrator", "line": "x", "scene_description": d}
                       for d in descriptions]}


def test_only_the_poems_main_character_is_registered(monkeypatch, tmp_path):
    """A four-line poem about an egg registered "village stones", "shadows
    dancing and creeping across the wall" and "sleepy little birdies sleeping on
    the ground" as three separate characters, because each scene resolved its
    own opening noun. CLIP was then scored against phrases that are not
    subjects: ~0.17 against a 0.26 bar, all five attempts burned on three of
    four scenes."""
    registered = []

    def fake_resolve(bank, subject, seed, description=""):
        registered.append(subject)
        return subject, 1234, True

    monkeypatch.setattr(va, "resolve_character", fake_resolve)
    monkeypatch.setattr(va, "load_bank", lambda: {})
    monkeypatch.setattr(va, "save_bank", lambda b: None)
    monkeypatch.setattr(va, "build_pipeline", lambda cfg: object())
    monkeypatch.setattr(va, "build_clip_verifier", lambda: object())
    monkeypatch.setattr(va, "ensure_dirs", lambda p: {"images_dir": tmp_path})
    monkeypatch.setattr(va, "hires_fix", lambda *a, **k: a[1])

    seen_subjects = []

    def fake_best(pipe, prompt, seed, sd, verifier, subject, bar, reference=None):
        seen_subjects.append(subject)

        class Img:
            width = height = 512
            def save(self, p): open(p, "wb").write(b"x")
        return Img(), pipe, 0.9, 1, seed

    monkeypatch.setattr(va, "generate_best_image", fake_best)
    monkeypatch.setattr(va, "build_prompt", lambda pipe, scene, shot, ch: "prompt")

    class Cfg:
        sd = {"hires_fix": False}
        pipeline = {"visual": {"character_bank": True, "ip_adapter": False},
                    "motion": {"enabled": False}}
        paths = {"images_dir": tmp_path}

    script = _script([
        "a vibrant egg, sitting on a wall",
        "village stones, standing very tall",
        "shadows creeping, across the wall",
        "sleepy little birdies, on the ground",
    ])
    va.generate_visuals(script, Cfg())

    assert len(registered) == 1, f"registered {len(registered)} characters: {registered}"
    assert "egg" in registered[0]
    # And every scene verifies against that character, not its own scenery.
    assert set(seen_subjects) == {registered[0]}, seen_subjects
