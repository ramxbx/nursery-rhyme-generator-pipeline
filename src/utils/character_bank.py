"""Persistent character identities, reused across poems (GPT-36, Tier 1).

The seed rhyme list is deliberately small, so the same handful of characters
recur across many generated poems. Without a bank, each poem invents its pig
from scratch and the audience sees a different pig every time. With one, the
pig stays recognisably the same pig from episode to episode - which is how
children's content builds an audience, and the reason the whole seed-based
generator is worth having.

Entries are structured attribute slots rather than one opaque string, because
the subject will not always be a pig. A boy, a woman, a turtle, a sheep and a
cat all need the same treatment, and they share a shape:

    <build> <colour> <subject>, <feature>, <feature>, <wearing>

    small round rosy pink pig, curled tail, floppy ears, blue neckerchief
    young boy, curly brown hair, round glasses, red striped shirt
    small green turtle, patterned brown shell, yellow spots

Slots keep the pieces separately editable and let the descriptor be rebuilt to
fit SD's prompt budget by dropping the least important slots first, which a
single pre-joined string cannot do.

What this tier delivers is RECOGNISABLE, not IDENTICAL. Prompt and seed
conditioning on SD1.5 holds species, colour and markings steady while pose and
composition still drift. Going further needs image conditioning (GPT-37,
IP-Adapter) or a per-character LoRA (GPT-38); both build on this file rather
than replacing it, since each needs a canonical reference to condition on.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

from src.utils.logger import get_logger, log_with_fields

logger = get_logger("character_bank")

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
BANK_PATH = REPO_ROOT / "data" / "character_bank.json"

# Words that describe a character without identifying it. Stripped when deriving
# the key so "a little pink pig", "the pig" and "small pink pig" all land on
# "pig" - and equally so "a young boy" and "the little boy" land on "boy".
_GENERIC = {
    "a", "an", "the", "little", "small", "big", "large", "tiny", "young", "old",
    "baby", "cute", "happy", "sad", "jolly", "friendly", "brave", "kind", "clever",
    "brown", "black", "white", "pink", "red", "blue", "green", "yellow", "orange",
    "purple", "grey", "gray", "golden", "silver", "spotty", "spotted", "striped",
    "fluffy", "woolly", "furry", "shiny",
}

# A descriptor longer than this crowds out the scene's own content in SD1.5's
# 77-token prompt budget - the character would stay consistent while nothing it
# was doing survived truncation.
MAX_DESCRIPTOR_WORDS = 14

# Slot order is prompt order. SD weights the opening tokens most heavily, so the
# subject noun and its colour lead; accessories trail because losing a
# neckerchief costs less identity than losing the species.
_SLOT_ORDER = ["build", "colour", "subject"]
_TRAIL_SLOTS = ["features", "wearing"]


def character_key(subject: str) -> str:
    """Stable identity for a subject phrase.

    Keys on the head noun rather than the full phrase so the same character
    matches across poems that describe it differently - "a little pink pig" and
    "the happy pig" are one character and should look alike. Works the same for
    people: "a young boy" and "the brave boy" both key to "boy"."""
    words = re.findall(r"[a-z]+", subject.lower())
    meaningful = [w for w in words if w not in _GENERIC]
    return meaningful[-1] if meaningful else (words[-1] if words else "character")


def descriptor_for(entry: dict) -> str:
    """Compose an entry's slots into a prompt fragment, trimming from the back
    until it fits the budget.

    Dropping trailing slots rather than truncating mid-phrase matters: a cut
    string can end on a dangling colour word that SD then applies to whatever
    follows it in the prompt."""
    head = " ".join(str(entry.get(s, "")).strip() for s in _SLOT_ORDER if entry.get(s)).strip()
    parts = [head] if head else []
    # Accessory before the feature list, because trailing parts are what get
    # dropped when the budget bites and an accessory is the most distinctive
    # thing a character owns. Ordering it last meant a pig's blue neckerchief
    # was silently cut on every prompt while "floppy ears" - true of every
    # cartoon pig, and drawn by the model anyway - survived in its place.
    if entry.get("wearing"):
        parts.append(f"wearing {entry['wearing']}")
    features = entry.get("features") or []
    if isinstance(features, str):
        features = [features]
    parts.extend(str(f).strip() for f in features if str(f).strip())

    while parts and len(" ".join(parts).split()) > MAX_DESCRIPTOR_WORDS and len(parts) > 1:
        parts.pop()
    descriptor = ", ".join(parts)
    words = descriptor.split()
    return " ".join(words[:MAX_DESCRIPTOR_WORDS]) if len(words) > MAX_DESCRIPTOR_WORDS else descriptor


# A descriptor segment that opens with one of these is describing where the
# character is or what it is doing, not what it looks like. Registering those as
# appearance would pin a character to its debut forever - a fox first seen in
# snow would be described as "in a snowy forest" in every later episode, and an
# owl as "perched on an old oak branch" even while flying. Both observed.
_LOCATION_OPENERS = {
    "in", "on", "at", "under", "over", "beside", "near", "through", "across",
    "inside", "outside", "among", "behind", "beneath", "by", "against", "around",
    "within", "atop", "amid", "before", "between", "toward", "towards", "from",
}
_ACTION_OPENERS = {
    "perched", "standing", "stands", "sitting", "sits", "lying", "resting",
    "walking", "walks", "running", "runs", "roaming", "roams", "bounding",
    "bounces", "leaping", "leaps", "playing", "plays", "flying", "flies",
    "swimming", "swims", "looking", "looks", "facing", "faces", "holding",
    "holds", "jumping", "jumps", "dancing", "sleeping", "eating", "munching",
    "splashing", "climbing", "peeking", "watching", "smiling",
}


def is_appearance(segment: str) -> bool:
    """Whether a descriptor segment describes the character rather than its
    situation. Deliberately conservative - a missed appearance trait costs a
    thinner descriptor, whereas a captured setting is baked into the character
    permanently."""
    words = re.findall(r"[a-z]+", segment.lower())
    if not words:
        return False
    return words[0] not in _LOCATION_OPENERS and words[0] not in _ACTION_OPENERS


def entry_from_description(subject: str, scene_description: str, seed: int) -> dict:
    """Build a first-sighting entry from what the script actually gives us.

    The scene description is a comma-separated descriptor list whose leading
    segments are appearance and whose later ones drift into setting and action.
    The first segment is the subject and the second is taken as its one feature.

    Deliberately conservative: a thin descriptor costs some cross-episode
    consistency, whereas a wrongly captured setting is baked into the character
    permanently and shows up in every future episode. Auto-registered entries
    are therefore a starting point - they carry no build/colour slots at all.

    Entries are plain JSON and meant to be edited by hand; enriching a character
    after its first episode is expected, not a workaround, and is how the
    shipped six were written."""
    segments = [s.strip() for s in (scene_description or "").split(",") if s.strip()]
    # Only the segment immediately after the subject, and only if it reads as
    # appearance. Reaching further down the list captures scenery: filtering by
    # opening word rejects "in a snowy forest" and "perched on an oak branch",
    # but bare noun phrases like "pine trees" and "night sky" pass any such test
    # and would be registered as if they were part of the animal. Chasing those
    # with a blocklist is endless, so the rule is simply to stop early.
    features = [segments[1]] if len(segments) > 1 and is_appearance(segments[1]) else []
    return {
        "subject": subject.strip() or "character",
        "features": features,
        "seed": int(seed),
    }


REFERENCE_DIR = REPO_ROOT / "data" / "characters"


def reference_path(key: str) -> Path:
    """Where a character's canonical portrait lives.

    One image per character, generated on first sight and reused forever after
    - it is what IP-Adapter conditions on, so it defines that character's look
    for every future episode. Deliberately a purpose-made portrait rather than a
    crop of a scene: scene renders carry backgrounds, odd poses and sometimes
    other animals, all of which image conditioning would drag into every
    subsequent scene."""
    return REFERENCE_DIR / f"{key}.png"


def load_bank(path: Path = BANK_PATH) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (ValueError, OSError):
        # A corrupt bank must not take the run down - a fresh one costs only
        # cross-episode consistency, which is what we had before this existed.
        log_with_fields(logger, 30, "character bank unreadable, starting empty", path=str(path))
        return {}


def save_bank(bank: dict, path: Path = BANK_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(bank, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def resolve_character(bank: dict, subject: str, seed: int,
                       scene_description: str = "") -> tuple[str, int, bool]:
    """Look a character up, registering it on first sight.

    Returns (descriptor, seed, is_new). For a character already in the bank the
    descriptor is the STORED one, never the current poem's - that is the whole
    mechanism. The stored seed wins likewise, so the character starts from the
    same noise it did in its first episode."""
    key = character_key(subject)
    entry = bank.get(key)
    if entry:
        return descriptor_for(entry), int(entry.get("seed", seed)), False
    entry = entry_from_description(subject, scene_description, seed)
    bank[key] = entry
    return descriptor_for(entry), int(entry["seed"]), True
