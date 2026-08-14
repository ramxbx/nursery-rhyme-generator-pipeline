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
