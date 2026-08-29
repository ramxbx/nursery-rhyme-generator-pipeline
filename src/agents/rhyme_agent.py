"""Rhyme generation agent - writes the poem the rest of the pipeline turns
into a video.

Every other stage until now started from a rhyme the user had already typed
into a text file. This one produces that file, so a run can begin from nothing
but a random seed.

The model is asked to write an ORIGINAL poem in the spirit of a well-known
nursery rhyme, never to reproduce one. That is both the interesting creative
task and the safe one: the famous rhyme supplies mood, subject and shape, and
the words have to be new. `_is_original` enforces it rather than trusting the
prompt, because a model asked to imitate something it has memorised will
sometimes simply recite it.

Note on "searching" for a famous rhyme: the pipeline's model is served locally
by LM Studio with no network access, so there is no live lookup to make. The
seeds live in data/seed_rhymes.json - a curated list of traditional rhymes,
all long out of copyright, each reduced to the handful of facts the prompt
actually needs (subject, setting, mood). Adding to that file is how you widen
the pool.

As everywhere else in this project, code owns the structure and the model only
fills it in: line count, couplet rhyming, metre, subject consistency and
originality are all checked here, and a poem failing them is regenerated. When
the model cannot produce a usable poem at all - unloaded, timing out, or just
having a bad day - the run falls back to a pre-written poem from
data/fallback_rhymes/ so the pipeline never blocks on it.
"""
from __future__ import annotations

import argparse
import difflib
import json
import random
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.agents.script_agent import count_line_syllables
from src.config import PipelineConfig, load_config
from src.utils.llm_client import LLMError, call_with_fallback
from src.utils.logger import get_logger, log_with_fields
from src.utils.prompt_builder import build_rhyme_prompt

logger = get_logger("rhyme_agent")

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
SEED_PATH = REPO_ROOT / "data" / "seed_rhymes.json"
FALLBACK_DIR = REPO_ROOT / "data" / "fallback_rhymes"

# Default length. Overridable per run: a shorter poem is the practical choice
# for end-to-end testing, since every line costs a Bark take (~1.5 min) and an
# image (~2 min), so 16 lines is roughly an hour before anything can be judged.
RHYME_LINES = 16
TARGET_SYLLABLES = 8
SYLLABLE_TOLERANCE = 4          # 4-12 syllables; wider than script_agent's rewrite
                                # tolerance because this is first-draft writing,
                                # and the metre rewrite tightens it up later.
MIN_WORDS_PER_LINE = 3
MAX_WORDS_PER_LINE = 12

# Writing sixteen rhyming lines in one pass is a structuring task, and the
# GPT-18 benchmark is unambiguous that the 1B-class models cannot hold structure
# over that length. The same 4.6B model that draws the scene descriptions is
# used here for the same reason.
RHYME_MODEL = "google/gemma-4-e2b"
RHYME_MAX_TOKENS = 1400         # generous: this is a reasoning model and burns a
                                # large hidden pass before visible output starts.
RHYME_TIMEOUT_S = 300
GENERATE_RETRIES = 3

# Couplets that must actually rhyme for the poem to be accepted, out of the 8 in
# a 16-line poem. Not all 8, because the rhyme test below is spelling-based and
# has real blind spots (it cannot see that "high" rhymes with "sky"), so
# demanding a perfect score would reject good poems. A majority is enough to
# separate a rhyming poem from prose with line breaks.
MIN_RHYMING_COUPLETS = 5

# Above this similarity to a seed's title or to any pre-written fallback poem,
# the "new" poem is treated as recital rather than composition.
ORIGINALITY_THRESHOLD = 0.7


def load_seeds() -> list[dict]:
    return json.loads(SEED_PATH.read_text(encoding="utf-8"))


# English spells the same rhyming sound many ways, and a key taken straight
# from the letters misses most real rhymes: "Sue"/"grew", "low"/"go" and
# "trees"/"breeze" all rhyme but share no ending. Folding the common vowel
# spellings onto one representative catches them. Ordered longest-first so
# "ough" is consumed before "ou".
_VOWEL_ALIASES = [
    ("ough", "u"), ("ue", "u"), ("ew", "u"), ("oo", "u"), ("ou", "u"),
    ("ow", "o"), ("oa", "o"), ("oe", "o"),
    ("ee", "e"), ("ea", "e"), ("ie", "e"),
    ("ai", "a"), ("ay", "a"),
]


def _normalise_key(key: str) -> str:
    for spelling, canonical in _VOWEL_ALIASES:
        key = key.replace(spelling, canonical)
    # A final "s" and "z" are the same sound after a vowel ("trees"/"breeze").
    if key.endswith("z"):
        key = key[:-1] + "s"
    return key


def _rhyme_key(word: str) -> str:
    """The part of a word that has to match for it to rhyme, approximated from
    spelling.

    There is no pronunciation dictionary among this project's dependencies, so
    this works on letters: drop a silent final 'e', take everything from the
    last vowel group onward ("star" -> "ar", "are" -> "ar"), then fold the
    common alternative spellings of the same vowel together.

    It stays blind to rhymes whose spellings diverge completely - "high"/"sky"
    and "flower"/"hour" both fail - which is why the caller requires only a
    majority of couplets to pass rather than all of them."""
    w = re.sub(r"[^a-z]", "", word.lower())
    if not w:
        return ""
    # Soft final "c" is the same sound as "s" ("face" rhymes with "case").
    # Done before the silent "e" is dropped, because the "e" is what makes the
    # "c" soft in the first place.
    if w.endswith("ce"):
        w = w[:-2] + "se"
    groups = list(re.finditer(r"[aeiouy]+", w))
    if w.endswith("e") and len(groups) > 1:
        w = w[:-1]
        groups = list(re.finditer(r"[aeiouy]+", w))
    if not groups:
        return _normalise_key(w[-2:])
    return _normalise_key(w[groups[-1].start():])


def lines_rhyme(a: str, b: str) -> bool:
    last = lambda line: (re.findall(r"[A-Za-z']+", line) or [""])[-1]
    wa, wb = last(a).lower(), last(b).lower()
    if not wa or not wb:
        return False
    if wa == wb:
        return False  # repeating the same word is not a rhyme
    ka, kb = _rhyme_key(wa), _rhyme_key(wb)
    if not ka or not kb:
        return False
    if ka == kb or wa[-2:] == wb[-2:]:
        return True
    # One key ending inside the other covers a trailing plural or inflection
    # that survived normalisation ("es" against "ees").
    return len(ka) >= 2 and len(kb) >= 2 and (ka.endswith(kb) or kb.endswith(ka))


def count_rhyming_couplets(lines: list[str]) -> int:
    return sum(1 for i in range(0, len(lines) - 1, 2) if lines_rhyme(lines[i], lines[i + 1]))


def _is_original(lines: list[str], seed: dict) -> bool:
    """Reject a poem that recites something instead of composing it.

    Checked against the pre-written fallback poems (the only full poems this
    repo ships) and against the seed's own title, which is the fragment of the
    original rhyme a model is most likely to echo back."""
    text = " ".join(lines).lower()
    if difflib.SequenceMatcher(None, seed["title"].lower(), text[:len(seed["title"]) + 10]).ratio() > ORIGINALITY_THRESHOLD:
        return False
    for path in sorted(FALLBACK_DIR.glob("*.txt")):
        other = " ".join(path.read_text(encoding="utf-8").split()).lower()
        if difflib.SequenceMatcher(None, other, text).ratio() > ORIGINALITY_THRESHOLD:
            return False
    return True


def validate_poem(text: str, seed: dict, check_originality: bool = True,
                   line_total: int = RHYME_LINES) -> tuple[list[str], str]:
    """Returns (lines, "") when the poem is usable, or ([], reason) when not.
    All-or-nothing: a poem with the wrong number of lines cannot be trimmed
    into shape without breaking its couplets.

    `check_originality` is only meaningful for model output. The pre-written
    fallback poems are themselves the corpus that check compares against, so
    they would always fail it - their structure still has to hold up, which is
    what the rest of this function tests."""
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    # Models like to introduce themselves; drop a leading title-ish line.
    if len(lines) == line_total + 1 and len(lines[0].split()) <= 6 and not lines[0].endswith((",", ".")):
        lines = lines[1:]
    if len(lines) != line_total:
        return [], f"expected {line_total} lines, got {len(lines)}"
    if len({line.lower() for line in lines}) != line_total:
        return [], "poem repeats a line"
    for line in lines:
        words = len(line.split())
        if not MIN_WORDS_PER_LINE <= words <= MAX_WORDS_PER_LINE:
            return [], f"line has {words} words: {line!r}"
        syllables = count_line_syllables(line)
        if abs(syllables - TARGET_SYLLABLES) > SYLLABLE_TOLERANCE:
            return [], f"line has {syllables} syllables: {line!r}"
    rhyming = count_rhyming_couplets(lines)
    # Scaled to length so a short poem is not held to a 16-line poem's count.
    needed = max(2, round(MIN_RHYMING_COUPLETS * line_total / RHYME_LINES))
    if rhyming < needed:
        return [], f"only {rhyming}/{line_total // 2} couplets rhyme"
    if check_originality and not _is_original(lines, seed):
        return [], "poem reproduces an existing rhyme"
    return lines, ""


def pick_fallback(rng: random.Random | None = None,
                   line_total: int = RHYME_LINES) -> tuple[str, str]:
    """A pre-written poem, used when generation fails. Returns (name, text).

    The source files carry blank lines between stanzas because they are meant
    to be read and edited by hand. Those are stripped here so this returns the
    identical shape to a generated poem - 16 lines, nothing else - and callers
    never have to care which path a run took."""
    paths = sorted(FALLBACK_DIR.glob("*.txt"))
    if not paths:
        raise FileNotFoundError(f"no fallback rhymes in {FALLBACK_DIR}")
    path = (rng or random).choice(paths)
    lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    # Truncated to the requested length, on an even boundary so couplets
    # stay intact. Without this a `--lines 4` run that fell back silently
    # produced a 16-line poem - which at ~40 min per animated scene is the
    # difference between a 2.7 hour render and a 10.7 hour one.
    keep = max(2, min(line_total - line_total % 2, len(lines)))
    return path.stem, "\n".join(lines[:keep]) + "\n"


def generate_rhyme(config: PipelineConfig | None = None, seed_title: str | None = None,
                    rng: random.Random | None = None, line_total: int = RHYME_LINES) -> tuple[str, str, bool]:
    """Write a new 16-line nursery rhyme.

    Returns (name, poem_text, generated) - `generated` is False when the poem
    came from the fallback folder, which callers log so a run is never silently
    assumed to be original work."""
    config = config or load_config()
    rng = rng or random.Random()
    seeds = load_seeds()
    if seed_title:
        seeds = [s for s in seeds if s["title"].lower() == seed_title.lower()] or seeds

    rhyme_config = {
        "endpoint": config.llm["endpoint"],
        "primary": RHYME_MODEL,
        "fallback": RHYME_MODEL,
        "request_timeout_s": RHYME_TIMEOUT_S,
        "max_retries": 1,
        "excluded": config.llm.get("excluded", []),
    }

    for attempt in range(1, GENERATE_RETRIES + 1):
        seed = rng.choice(seeds)
        prompt = build_rhyme_prompt(seed, line_total, TARGET_SYLLABLES)
        try:
            result = call_with_fallback(system_prompt=prompt,
                                         user_prompt="Write the poem now.",
                                         llm_config=rhyme_config, parse_json=False,
                                         max_tokens=RHYME_MAX_TOKENS)
        except LLMError as e:
            log_with_fields(logger, 30, "rhyme generation call failed", attempt=attempt, error=str(e))
            continue

        lines, reason = validate_poem(result.text.strip(), seed, line_total=line_total)
        if lines:
            log_with_fields(logger, 20, "rhyme generated", attempt=attempt, seed=seed["title"],
                             rhyming_couplets=count_rhyming_couplets(lines))
            name = re.sub(r"[^a-z0-9]+", "_", seed["subject"].lower()).strip("_")[:40] or "rhyme"
            return name, "\n".join(lines) + "\n", True
        log_with_fields(logger, 30, "rhyme rejected, retrying", attempt=attempt,
                         seed=seed["title"], reason=reason)

    name, text = pick_fallback(rng, line_total)
    log_with_fields(logger, 30, "rhyme generation exhausted retries, using fallback poem", name=name)
    return name, text, False


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Write a new 16-line nursery rhyme, inspired by a famous one but original.")
    parser.add_argument("--out", type=Path, default=None, help="Where to write the poem (default: data/<name>.txt)")
    parser.add_argument("--seed", default=None, help="Force a particular seed rhyme by title")
    parser.add_argument("--lines", type=int, default=RHYME_LINES,
                        help=f"How many lines to write (default {RHYME_LINES}; must be even)")
    parser.add_argument("--fallback-only", action="store_true",
                        help="Skip the model entirely and take a pre-written poem")
    args = parser.parse_args()

    config = load_config()
    if args.fallback_only:
        name, text = pick_fallback(line_total=args.lines)
        generated = False
    else:
        name, text, generated = generate_rhyme(config, args.seed, line_total=args.lines)

    out_path = args.out or (REPO_ROOT / "data" / f"{name}.txt")
    out_path.write_text(text, encoding="utf-8")
    print(f"Wrote {out_path} ({'generated' if generated else 'fallback'})")


if __name__ == "__main__":
    main()
