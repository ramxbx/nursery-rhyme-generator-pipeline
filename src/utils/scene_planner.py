"""Scene planning: decide how many scenes a rhyme becomes, and where they cut.

Until now a scene *was* a source line - script_agent split the rhyme on
newlines and every stage downstream ran 1:1 from there. That makes the scene
count an accident of how the input file happens to be typed, and it leaves the
pipeline no way to even out a poem whose lines are wildly different lengths, or
to break up a single long line that would otherwise park one image on screen
for ten seconds.

This module replaces "a scene is a line" with "a scene is a contiguous run of
the poem, chosen so the scenes come out roughly equal". Three deliberate
choices shape it:

* No model is involved. script_agent's standing rule (see its module docstring)
  is that code owns structure and the local model only annotates - asking a
  1.2B CPU model to decompose a rhyme is the exact failure that disqualified it
  during the GPT-18 benchmark. Scene count and cut points are arithmetic here,
  and stay testable and reproducible.
* Cuts prefer line boundaries, because a line boundary is a rhyme boundary. The
  last word of a source line is the rhyme word, and singing.py already holds
  every line's final note - so a scene that ends mid-line ends on an unresolved
  phrase and sounds wrong. Lines are only broken apart when one is long enough
  to unbalance everything around it, and then only at its commas.
* Scene count is chosen by trying every plausible count and keeping the most
  even one, rather than by dividing total duration by a target. A raw division
  regularly lands on a count that no contiguous partition can fill evenly - for
  an 8-line poem it suggested 6 scenes, whose best possible partition was
  *less* even (1.85x spread) than simply keeping the 8 lines (1.4x).

A caveat worth keeping in view: the durations planned here are syllable-based
estimates. With the Bark backend the video's real scene lengths follow the
generated audio instead, so an even plan is necessary but not sufficient - the
acceptance band in bark_tts/audio_agent is what makes the plan stick.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable

# A line is only broken apart when it is at least this much longer than the
# target scene, i.e. when leaving it whole would visibly unbalance the video.
# Below that the rhyme boundary is worth more than the extra evenness.
SPLIT_LINE_RATIO = 1.5

# Phrase boundaries usable as cut points inside an over-long line. These are
# where a singer would already draw breath, so cutting here costs the least.
PHRASE_SPLIT_RE = re.compile(r"(?<=[,;:])\s+")

# How much a candidate scene count is penalised for landing far from the target
# scene duration, relative to how much it is penalised for being uneven.
# Evenness is what was asked for, so it carries full weight and duration drift
# is only a tie-breaker between similarly even options.
DURATION_DRIFT_WEIGHT = 1.0


@dataclass(frozen=True)
class Atom:
    """The smallest unit a scene can be built from - a whole source line, or a
    phrase of one that was too long to leave intact."""
    text: str
    line_index: int
    duration_s: float


@dataclass(frozen=True)
class ScenePlan:
    text: str
    line_indices: tuple[int, ...]
    duration_s: float


def split_into_atoms(lines: list[str], duration_of: Callable[[str], float],
                      target_scene_duration_s: float) -> list[Atom]:
    """One atom per line, except that a line far longer than a target scene is
    broken at its commas so the partition has somewhere to cut."""
    atoms: list[Atom] = []
    for line_index, line in enumerate(lines):
        duration = duration_of(line)
        pieces = [line]
        if duration > target_scene_duration_s * SPLIT_LINE_RATIO:
            candidate = [p.strip() for p in PHRASE_SPLIT_RE.split(line) if p.strip()]
            if len(candidate) > 1:
                pieces = candidate
        atoms.extend(Atom(text=p, line_index=line_index, duration_s=duration_of(p)) for p in pieces)
    return atoms


def partition_evenly(weights: list[float], k: int) -> list[int]:
    """Split `weights` into k contiguous groups, minimising the largest group
    total. Returns the group sizes.

    This is the classic linear-partition DP rather than a greedy pack: greedy
    fills early groups and pushes the imbalance into the last one, which for a
    poem means the final scene - the one that resolves the rhyme - ends up the
    odd length. At these sizes (tens of atoms) the exact O(n^2 k) solution is
    instant, so there is no reason to approximate."""
    n = len(weights)
    if k <= 1 or n == 0:
        return [n] if n else []
    if k >= n:
        return [1] * n

    prefix = [0.0]
    for w in weights:
        prefix.append(prefix[-1] + w)

    inf = float("inf")
    # best[i][j]: smallest achievable largest-group total when the first i atoms
    # are split into j groups. cut[i][j] remembers where the last group started.
    best = [[inf] * (k + 1) for _ in range(n + 1)]
    cut = [[0] * (k + 1) for _ in range(n + 1)]
    best[0][0] = 0.0
    for i in range(1, n + 1):
        for j in range(1, min(k, i) + 1):
            for p in range(j - 1, i):
                cost = max(best[p][j - 1], prefix[i] - prefix[p])
                if cost < best[i][j]:
                    best[i][j] = cost
                    cut[i][j] = p

    sizes, i = [], n
    for j in range(k, 0, -1):
        start = cut[i][j]
        sizes.append(i - start)
        i = start
    return list(reversed(sizes))


def _groups_from_sizes(atoms: list[Atom], sizes: list[int]) -> list[list[Atom]]:
    groups, i = [], 0
    for size in sizes:
        groups.append(atoms[i:i + size])
        i += size
    return groups


def _score(groups: list[list[Atom]], target_scene_duration_s: float) -> float:
    """Lower is better. Imbalance (longest scene / shortest scene) dominates;
    distance from the target scene duration breaks ties."""
    totals = [sum(a.duration_s for a in g) for g in groups]
    imbalance = max(totals) / max(min(totals), 1e-6)
    mean = sum(totals) / len(totals)
    drift = abs(mean - target_scene_duration_s) / max(target_scene_duration_s, 1e-6)
    return imbalance + DURATION_DRIFT_WEIGHT * drift


def plan_scenes(lines: list[str], duration_of: Callable[[str], float],
                 target_scene_duration_s: float = 4.0, min_scenes: int = 1,
                 max_scenes: int = 16, max_words_per_scene: int = 14) -> list[ScenePlan]:
    """Group a rhyme's lines into as-even-as-possible scenes.

    Every candidate scene count in range is partitioned and scored, and the
    best-scoring one wins. Counts that would produce a scene with more words
    than the subtitle can hold are discarded outright - a lyric that wraps to
    three lines at the subtitle size used here covers the picture."""
    if not lines:
        return []

    atoms = split_into_atoms(lines, duration_of, target_scene_duration_s)
    weights = [a.duration_s for a in atoms]

    lo = max(1, min(min_scenes, len(atoms)))
    hi = max(lo, min(max_scenes, len(atoms)))

    best_groups, best_score = None, float("inf")
    for k in range(lo, hi + 1):
        groups = _groups_from_sizes(atoms, partition_evenly(weights, k))
        if any(len(" ".join(a.text for a in g).split()) > max_words_per_scene for g in groups):
            continue
        score = _score(groups, target_scene_duration_s)
        if score < best_score:
            best_groups, best_score = groups, score

    if best_groups is None:
        # Every candidate breached the word cap, so the poem's own lines are
        # longer than the cap allows. Take the finest split available - it has
        # the fewest words per scene - and let the cap yield rather than cutting
        # mid-phrase, which would cost more than a wrapped subtitle does.
        best_groups = _groups_from_sizes(atoms, partition_evenly(weights, hi))

    groups = best_groups

    plans = []
    for group in groups:
        text = " ".join(a.text for a in group)
        # Recomputed from the joined text rather than summed: the estimator
        # counts inter-clause pauses, which differ once phrases are rejoined.
        plans.append(ScenePlan(text=text,
                                line_indices=tuple(dict.fromkeys(a.line_index for a in group)),
                                duration_s=duration_of(text)))
    return plans
