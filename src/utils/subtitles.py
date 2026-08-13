"""Kid-readable burned-in subtitles (GPT-21).

ASS format rather than plain SRT so font size/weight/outline/position can
be controlled directly - large bold text with a strong outline so it stays
legible over any background image, bottom-center.

The lyrics carry more of this video than the images do, so the styling is
deliberately prominent: large, bright colours, heavy outline, in a rounded
handwriting face rather than a standard UI font, with karaoke-style
highlighting that sweeps word by word as the line is sung so a child can
follow along with where they are in the lyric.
"""
from __future__ import annotations

import re
from pathlib import Path

# Segoe Print: a rounded handwriting face that ships with Windows, so it needs
# no font bundling. Friendlier and more storybook-like than Comic Sans (the
# previous choice) while staying highly legible at large sizes for children
# who are still learning to read. Ink Free and Segoe Script are the other
# stock options - Script is cursive and much harder for early readers.
SUBTITLE_FONT = "Segoe Print"

# ASS colours are &HAABBGGRR (alpha, then BLUE-GREEN-RED - not RGB).
# PrimaryColour is the "already sung" colour that karaoke sweeps in;
# SecondaryColour is what a word looks like before its turn.
SUNG_COLOUR = "&H0000FFFF"      # bright yellow
UNSUNG_COLOUR = "&H00FFFFFF"    # white
OUTLINE_COLOUR = "&H00202020"   # near-black, softer than pure black
SHADOW_COLOUR = "&H80000000"    # semi-transparent drop shadow

ASS_HEADER_TEMPLATE = """[Script Info]
ScriptType: v4.00+
PlayResX: {width}
PlayResY: {height}
WrapStyle: 0
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Kid,{font},{font_size},{sung},{unsung},{outline_colour},{shadow_colour},-1,0,0,0,100,100,2,0,1,{outline},3,2,40,40,{margin_v},1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""

# Word-art flourish applied to every line: a slight pop-in scale and fade so
# the lyric arrives rather than snapping on. Kept short so it never delays the
# highlight sweep, which has to stay aligned with the singing.
LINE_INTRO_EFFECT = r"{\fad(150,150)\t(0,180,\fscx110\fscy110)\t(180,320,\fscx100\fscy100)}"


def _ass_time(t: float) -> str:
    t = max(0.0, t)
    h = int(t // 3600)
    m = int((t % 3600) // 60)
    s = int(t % 60)
    cs = int(round((t - int(t)) * 100))
    if cs >= 100:
        cs, s = 0, s + 1
    return f"{h}:{m:02d}:{s:02d}.{cs:02d}"


def _escape(text: str) -> str:
    """ASS treats braces as override-tag delimiters and backslashes as escapes,
    so any of either in the lyric itself would corrupt the styling."""
    return text.replace("\\", "\\\\").replace("{", "(").replace("}", ")").replace("\n", "\\N")


def _syllable_weight(word: str) -> int:
    """Rough syllable count, used to share a line's duration between words.

    Weighting by syllables rather than splitting evenly keeps the highlight
    roughly in step with the singing - "little" should hold about twice as
    long as "the". This only has to be approximately right; it is a reading
    aid, not a transcript alignment."""
    letters = re.sub(r"[^a-z]", "", word.lower())
    if not letters:
        return 1
    groups = re.findall(r"[aeiouy]+", letters)
    count = len(groups)
    if letters.endswith("e") and count > 1:
        count -= 1
    return max(count, 1)


def _karaoke_text(line: str, duration_s: float) -> str:
    """Build the line with per-word karaoke timings.

    `\\kf` sweeps the fill smoothly across each word (as opposed to `\\k`,
    which flips whole words at once) - closer to a progress bar running along
    the lyric, which is easier to follow than a hard switch."""
    words = line.split()
    if not words:
        return _escape(line)

    weights = [_syllable_weight(w) for w in words]
    total_weight = sum(weights)
    # ASS karaoke durations are in centiseconds.
    total_cs = max(1, int(round(duration_s * 100)))

    parts, spent = [], 0
    for i, (word, weight) in enumerate(zip(words, weights)):
        if i == len(words) - 1:
            cs = max(1, total_cs - spent)  # absorb rounding drift in the last word
        else:
            cs = max(1, int(round(total_cs * weight / total_weight)))
            spent += cs
        parts.append(rf"{{\kf{cs}}}{_escape(word)}")
    return " ".join(parts)


def write_ass_subtitles(lines: list[str], timeline: list[tuple[float, float]], out_path: Path,
                         width: int, height: int, font_size: int = 120, outline: int = 6,
                         margin_v: int = 80, karaoke: bool = True) -> Path:
    if len(lines) != len(timeline):
        raise ValueError(f"lines ({len(lines)}) and timeline ({len(timeline)}) length mismatch")

    header = ASS_HEADER_TEMPLATE.format(
        width=width, height=height, font=SUBTITLE_FONT, font_size=font_size,
        sung=SUNG_COLOUR, unsung=UNSUNG_COLOUR, outline_colour=OUTLINE_COLOUR,
        shadow_colour=SHADOW_COLOUR, outline=outline, margin_v=margin_v)

    events = []
    for text, (start, end) in zip(lines, timeline):
        body = _karaoke_text(text, max(0.1, end - start)) if karaoke else _escape(text)
        events.append(
            f"Dialogue: 0,{_ass_time(start)},{_ass_time(end)},Kid,,0,0,0,,{LINE_INTRO_EFFECT}{body}")

    out_path.write_text(header + "\n".join(events) + "\n", encoding="utf-8")
    return out_path
