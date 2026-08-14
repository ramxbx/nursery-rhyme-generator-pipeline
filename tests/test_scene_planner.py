"""Unit tests for scene planning. Duration estimates come from script_agent's
real estimator so these exercise the same arithmetic the pipeline uses."""
from src.agents.script_agent import estimate_duration
from src.utils.scene_planner import partition_evenly, plan_scenes, split_into_atoms


def _durations(plans):
    return [p.duration_s for p in plans]


def _spread(plans):
    d = _durations(plans)
    return max(d) / min(d)


def test_partition_minimises_the_largest_group_rather_than_packing_greedily():
    """A greedy pack fills early groups and dumps the remainder in the last
    one - which for a poem is the scene that resolves the rhyme."""
    assert partition_evenly([1.0, 1.0, 1.0, 9.0], 2) == [3, 1]
    assert partition_evenly([5.0, 1.0, 1.0, 1.0], 2) == [1, 3]


def test_partition_handles_degenerate_counts():
    assert partition_evenly([1.0, 2.0, 3.0], 1) == [3]
    assert partition_evenly([1.0, 2.0], 5) == [1, 1]
    assert partition_evenly([], 3) == []


def test_every_source_line_reaches_a_scene():
    """The pipeline's core invariant: planning may regroup the poem but must
    never drop part of it."""
    lines = ["Twinkle, twinkle, little star,", "How I wonder what you are.",
             "Up above the world so high,", "Like a diamond in the sky."]
    plans = plan_scenes(lines, estimate_duration, target_scene_duration_s=5.0)
    covered = {i for p in plans for i in p.line_indices}
    assert covered == set(range(len(lines)))


def test_an_over_long_line_is_split_at_its_commas():
    """One rambling line would otherwise park a single image on screen for ten
    seconds while the rest of the poem races past."""
    long_line = ("And then the enormous spotted dog, who had been sleeping all afternoon, "
                 "woke up with a start, and barked.")
    atoms = split_into_atoms([long_line], estimate_duration, target_scene_duration_s=4.0)
    assert len(atoms) > 1
    assert all(a.line_index == 0 for a in atoms)


def test_a_short_line_is_never_split():
    """Line boundaries are rhyme boundaries, so they are only broken when
    leaving the line whole would unbalance the video."""
    atoms = split_into_atoms(["Twinkle, twinkle, little star,"], estimate_duration,
                              target_scene_duration_s=4.0)
    assert len(atoms) == 1


def test_planning_evens_out_a_lopsided_poem():
    lines = ["The cat sat.",
             ("And then the enormous spotted dog, who had been sleeping all afternoon in the warm sun, "
              "woke up with a start, and barked."),
             "Birds flew."]
    before = [estimate_duration(line) for line in lines]
    plans = plan_scenes(lines, estimate_duration, target_scene_duration_s=4.0)
    assert _spread(plans) < max(before) / min(before) / 2


def test_planning_leaves_an_already_even_poem_alone():
    """Merging whole lines is lumpy - forcing a count onto a poem whose lines
    are already uniform makes it less even, not more, so the planner should
    decline to act."""
    lines = ["Twinkle, twinkle, little star,", "How I wonder what you are.",
             "Up above the world so high,", "Like a diamond in the sky."]
    plans = plan_scenes(lines, estimate_duration, target_scene_duration_s=3.0)
    assert [p.text for p in plans] == lines


def test_scene_count_responds_to_the_target_duration():
    """The knob has to actually move: a longer target should yield fewer,
    longer scenes for the same poem."""
    lines = ["Twinkle, twinkle, little star,", "How I wonder what you are.",
             "Up above the world so high,", "Like a diamond in the sky.",
             "When the blazing sun is gone,", "When he nothing shines upon,"]
    short = plan_scenes(lines, estimate_duration, target_scene_duration_s=3.0)
    long = plan_scenes(lines, estimate_duration, target_scene_duration_s=7.0)
    assert len(short) > len(long)


def test_word_cap_keeps_subtitles_off_the_picture():
    """A merged lyric past the cap wraps to three lines at the subtitle size
    used here and covers the image."""
    lines = ["Twinkle, twinkle, little star,", "How I wonder what you are.",
             "Up above the world so high,", "Like a diamond in the sky."]
    plans = plan_scenes(lines, estimate_duration, target_scene_duration_s=20.0,
                         max_words_per_scene=6)
    assert all(len(p.text.split()) <= 6 for p in plans)


def test_empty_poem_plans_nothing():
    assert plan_scenes([], estimate_duration) == []
