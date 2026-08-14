"""Unit tests for the cross-episode character bank (GPT-36)."""
from src.utils import character_bank as cb


def test_the_same_animal_described_differently_is_one_character():
    """Poems describe their subject in their own words, so keying on the whole
    phrase would register a new character every episode - exactly the drift the
    bank exists to prevent."""
    keys = {cb.character_key(s) for s in
            ["a little pink pig", "the happy pig", "small spotty pig", "pig"]}
    assert keys == {"pig"}


def test_different_species_stay_separate():
    assert cb.character_key("a brown bear") != cb.character_key("a little duck")


def test_key_survives_a_subject_that_is_all_modifiers():
    assert cb.character_key("the little") != ""
    assert cb.character_key("") == "character"


def test_first_sighting_registers_the_character():
    bank = {}
    descriptor, seed, is_new = cb.resolve_character(
        bank, "a little pink pig", 4242, "a little pink pig, curled tail, in the barnyard")
    assert is_new
    assert bank["pig"]["subject"] == "a little pink pig"
    assert bank["pig"]["seed"] == 4242
    assert "curled tail" in descriptor


def test_a_known_character_keeps_its_stored_look_not_the_new_poem_s():
    """The whole mechanism: episode two must render episode one's pig, not
    redescribe its own."""
    bank = {"pig": {"subject": "pig", "build": "small round", "colour": "pink",
                    "features": ["dark spot on left flank"], "seed": 111}}
    descriptor, seed, is_new = cb.resolve_character(bank, "a huge grey pig", 999, "a huge grey pig, tusks")
    assert not is_new
    assert "pink" in descriptor and "grey" not in descriptor
    assert seed == 111, "the stored seed must win so the character starts from the same noise"


def test_descriptor_is_capped_so_it_cannot_crowd_out_the_scene():
    """SD1.5 reads 77 tokens. An unbounded identity string would hold the
    character steady while everything it was doing fell off the end."""
    long_desc = ", ".join(["a very long descriptive phrase here"] * 8)
    descriptor, _, _ = cb.resolve_character({}, "a pig", 1, long_desc)
    assert len(descriptor.split()) <= cb.MAX_DESCRIPTOR_WORDS


def test_bank_round_trips_through_disk(tmp_path):
    path = tmp_path / "bank.json"
    cb.save_bank({"pig": {"subject": "pig", "colour": "pink", "seed": 7}}, path)
    assert cb.load_bank(path)["pig"]["seed"] == 7


def test_a_corrupt_bank_does_not_take_the_run_down(tmp_path):
    """Losing cross-episode consistency is recoverable; failing the image stage
    over a bad cache file is not worth it."""
    path = tmp_path / "bank.json"
    path.write_text("{not json", encoding="utf-8")
    assert cb.load_bank(path) == {}


def test_missing_bank_starts_empty(tmp_path):
    assert cb.load_bank(tmp_path / "nope.json") == {}


def test_schema_is_generic_across_animals_and_people():
    """The subject will not always be an animal - boys, girls, men and women
    have to compose the same way, or the bank only works for half the seeds."""
    boy = {"build": "young", "subject": "boy", "features": ["curly brown hair"],
           "wearing": "a red striped shirt"}
    turtle = {"build": "small", "colour": "leaf green", "subject": "turtle",
              "features": ["patterned brown shell"]}
    assert cb.descriptor_for(boy).startswith("young boy")
    assert "wearing a red striped shirt" in cb.descriptor_for(boy)
    assert cb.descriptor_for(turtle).startswith("small leaf green turtle")


def test_people_and_animals_key_separately():
    keys = {cb.character_key(s) for s in
            ["a young boy", "a little girl", "an old woman", "a small turtle", "a woolly sheep"]}
    assert keys == {"boy", "girl", "woman", "turtle", "sheep"}


def test_over_budget_descriptor_drops_whole_slots_not_part_words():
    """A mid-phrase cut can end on a dangling colour word, which SD then applies
    to whatever follows it in the prompt."""
    entry = {"build": "small round", "colour": "rosy pink", "subject": "pig",
             "features": ["curled tail", "floppy ears", "dark spot on left flank"],
             "wearing": "a blue neckerchief"}
    d = cb.descriptor_for(entry)
    assert len(d.split()) <= cb.MAX_DESCRIPTOR_WORDS
    assert d.startswith("small round rosy pink pig")
    assert not d.rstrip().endswith(",")


def test_every_shipped_bank_entry_composes_within_budget():
    for key, entry in cb.load_bank().items():
        d = cb.descriptor_for(entry)
        assert d, f"{key} composed to nothing"
        assert len(d.split()) <= cb.MAX_DESCRIPTOR_WORDS, f"{key}: {d}"


def test_a_new_subject_is_registered_from_the_poem():
    """New characters must join the bank automatically - the seed list grows and
    poems invent subjects nobody pre-registered."""
    bank = {}
    d, seed, is_new = cb.resolve_character(
        bank, "little fox", 7, "little fox, bushy red tail, in a snowy forest, pine trees")
    assert is_new and "fox" in bank
    assert "bushy red tail" in d


def test_setting_is_not_registered_as_appearance():
    """A fox first seen in snow must not be described as "in a snowy forest" in
    every later episode - that pins the character to its debut forever."""
    bank = {}
    cb.resolve_character(bank, "little fox", 7,
                          "little fox, bushy red tail, in a snowy forest, pine trees")
    assert bank["fox"]["features"] == ["bushy red tail"]


def test_pose_is_not_registered_as_appearance():
    bank = {}
    cb.resolve_character(bank, "a wise owl", 7,
                          "a wise owl, large round eyes, perched on an old oak branch, night sky")
    assert bank["owl"]["features"] == ["large round eyes"]


def test_appearance_classifier_keeps_traits_and_drops_situations():
    assert cb.is_appearance("bushy red tail")
    assert cb.is_appearance("wearing a blue scarf")
    assert not cb.is_appearance("in a snowy forest")
    assert not cb.is_appearance("perched on an old oak branch")
    assert not cb.is_appearance("running across the meadow")
    assert not cb.is_appearance("")


def test_a_second_poem_reuses_the_registered_character(tmp_path):
    """End to end: register from poem one, then poem two must render poem one's
    character rather than its own description."""
    path = tmp_path / "bank.json"
    bank = {}
    cb.resolve_character(bank, "little fox", 7, "little fox, bushy red tail, in a snowy forest")
    cb.save_bank(bank, path)

    reloaded = cb.load_bank(path)
    d, seed, is_new = cb.resolve_character(
        reloaded, "a huge blue fox", 999, "a huge blue fox, metal wings, on the moon")
    assert not is_new
    assert "bushy red tail" in d and "metal wings" not in d
    assert seed == 7


def test_bare_scenery_nouns_are_not_registered_as_appearance():
    """Filtering by opening word rejects "in a snowy forest", but "pine trees"
    and "night sky" pass any such test - so registration stops at the first
    feature rather than scanning down the list."""
    bank = {}
    cb.resolve_character(bank, "little fox", 7,
                          "little fox, bushy red tail, in a snowy forest, pine trees")
    assert bank["fox"]["features"] == ["bushy red tail"]
    assert "pine trees" not in cb.descriptor_for(bank["fox"])


def test_a_subject_with_no_usable_feature_still_registers():
    bank = {}
    d, _, is_new = cb.resolve_character(bank, "a duck", 7, "a duck, on the pond, reeds")
    assert is_new and bank["duck"]["features"] == []
    assert d == "a duck"


def test_an_accessory_is_never_dropped_before_a_generic_trait():
    """The bug: a pig's blue neckerchief - its single most distinctive feature -
    was silently cut from every prompt by the word budget while "floppy ears",
    true of every cartoon pig, survived in its place."""
    entry = {"build": "small round", "colour": "pink", "subject": "pig",
             "features": ["curled tail", "floppy ears", "dark spot on left flank"],
             "wearing": "a blue neckerchief"}
    d = cb.descriptor_for(entry)
    assert "wearing a blue neckerchief" in d, d
    assert len(d.split()) <= cb.MAX_DESCRIPTOR_WORDS


def test_every_shipped_accessory_survives_the_budget():
    """A character bank whose accessories never reach the model is doing
    nothing - this is the acceptance test for the whole file."""
    for key, entry in cb.load_bank().items():
        if not entry.get("wearing"):
            continue
        d = cb.descriptor_for(entry)
        assert "wearing" in d, f"{key} lost its accessory: {d}"
