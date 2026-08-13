"""Kid-readable burned-in subtitles (GPT-21).

ASS format rather than plain SRT so font size/weight/outline/position can
be controlled directly - large bold text with a strong outline so it stays
legible over any background image, bottom-center.

The lyrics carry more of this video than the images do, so the styling is
deliberately prominent: large, bright yellow, heavy black outline, in a
rounded handwriting face rather than a standard UI font.
"""
from __future__ import annotations

from pathlib import Path

# Segoe Print: a rounded handwriting face that ships with Windows, so it needs
# no font bundling. Friendlier and more storybook-like than Comic Sans (the
# previous choice) while staying highly legible at large sizes for children
# who are still learning to read. Ink Free and Segoe Script are the other
# stock options - Script is cursive and much harder for early readers.
SUBTITLE_FONT = "Segoe Print"

ASS_HEADER_TEMPLATE = """[Script Info]
ScriptType: v4.00+
PlayResX: {width}
PlayResY: {height}
WrapStyle: 0

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Kid,{font},{font_size},&H0000FFFF,&H000000FF,&H00000000,&H00000000,-1,0,0,0,100,100,0,0,1,{outline},2,2,40,40,{margin_v},1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""


def _ass_time(t: float) -> str:
    t = max(0.0, t)
    h = int(t // 3600)
    m = int((t % 3600) // 60)
    s = int(t % 60)
    cs = int(round((t - int(t)) * 100))
    if cs >= 100:
        cs, s = 0, s + 1
    return f"{h}:{m:02d}:{s:02d}.{cs:02d}"


def write_ass_subtitles(lines: list[str], timeline: list[tuple[float, float]], out_path: Path,
                         width: int, height: int, font_size: int = 120, outline: int = 6,
                         margin_v: int = 80) -> Path:
    if len(lines) != len(timeline):
        raise ValueError(f"lines ({len(lines)}) and timeline ({len(timeline)}) length mismatch")

    header = ASS_HEADER_TEMPLATE.format(width=width, height=height, font=SUBTITLE_FONT,
                                         font_size=font_size, outline=outline, margin_v=margin_v)
    events = []
    for text, (start, end) in zip(lines, timeline):
        escaped = text.replace("\\", "\\\\").replace("{", "(").replace("}", ")").replace("\n", "\\N")
        events.append(f"Dialogue: 0,{_ass_time(start)},{_ass_time(end)},Kid,,0,0,0,,{escaped}")

    out_path.write_text(header + "\n".join(events) + "\n", encoding="utf-8")
    return out_path
