"""Render caption groups into an ASS subtitle file.

Each word gets its own dialogue event showing the whole group, with the
currently spoken word recoloured and scaled up. That produces the karaoke
highlight clippers use, without relying on \\k timing tags.
"""

from __future__ import annotations

from pathlib import Path

import config
from models import NO_SPACE_BEFORE, CaptionGroup

_HEADER_TEMPLATE = """[Script Info]
ScriptType: v4.00+
PlayResX: {play_x}
PlayResY: {play_y}
WrapStyle: 0
ScaledBorderAndShadow: yes
YCbCr Matrix: TV.709

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Clip,{font},{size},&H00{base},&H00{active},&H00{outline},&H80000000,-1,0,0,0,100,100,0,0,1,{outline_w},{shadow},2,{margin_h},{margin_h},{margin_v},1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""


def _format_time(seconds: float) -> str:
    """ASS timestamps are H:MM:SS.CC with centisecond precision."""
    seconds = max(0.0, seconds)
    hours, remainder = divmod(int(seconds), 3600)
    minutes, secs = divmod(remainder, 60)
    centis = int(round((seconds - int(seconds)) * 100))
    if centis == 100:  # rounding can tip a whole second
        centis = 0
        secs += 1
        if secs == 60:
            secs = 0
            minutes += 1
    return f"{hours}:{minutes:02d}:{secs:02d}.{centis:02d}"


def _escape(text: str) -> str:
    """Neutralise characters that libass would treat as markup."""
    return (
        text.replace("\\", "\\\\")
        .replace("{", "(")
        .replace("}", ")")
        .replace("\n", " ")
        .replace("\r", " ")
    )


def _word_text(raw: str) -> str:
    text = _escape(raw)
    return text.upper() if config.UPPERCASE_CAPTIONS else text


def _render_group_line(group: CaptionGroup, active_index: int) -> str:
    """Build the on-screen text with one word highlighted."""
    scale = config.ACTIVE_SCALE_PERCENT
    line = ""

    for index, word in enumerate(group.words):
        text = _word_text(word.text)

        # Whisper splits punctuation into its own token, so a space here would
        # render "4 ,000" on screen.
        if line and not text.startswith(NO_SPACE_BEFORE):
            line += " "

        if index == active_index:
            line += (
                f"{{\\1c&H{config.COLOR_ACTIVE}&\\fscx{scale}\\fscy{scale}}}"
                f"{text}"
                f"{{\\1c&H{config.COLOR_BASE}&\\fscx100\\fscy100}}"
            )
        else:
            line += text

    return line


def _group_events(group: CaptionGroup) -> list[tuple[float, float, str]]:
    """One event per word, tiled so the group never blinks out mid-phrase."""
    events: list[tuple[float, float, str]] = []
    words = group.words

    for index, word in enumerate(words):
        start = group.start_s if index == 0 else word.start_s
        if index + 1 < len(words):
            end = words[index + 1].start_s
        else:
            end = group.end_s

        if end <= start:
            continue

        events.append((start, end, _render_group_line(group, index)))

    return events


def write_ass(
    groups: list[CaptionGroup],
    destination: Path,
    language: str | None = None,
) -> Path:
    """Write an ASS file covering every caption group.

    The font follows the language, because the default carries Latin glyphs
    only and anything else would render as empty boxes.
    """
    header = _HEADER_TEMPLATE.format(
        play_x=config.TARGET_WIDTH,
        play_y=config.TARGET_HEIGHT,
        font=config.font_for_language(language),
        size=config.FONT_SIZE,
        base=config.COLOR_BASE,
        active=config.COLOR_ACTIVE,
        outline=config.COLOR_OUTLINE,
        outline_w=config.OUTLINE_WIDTH,
        shadow=config.SHADOW_DEPTH,
        margin_h=config.CAPTION_MARGIN_H,
        margin_v=config.CAPTION_MARGIN_V,
    )

    lines = [header]
    for group in groups:
        for start, end, text in _group_events(group):
            lines.append(
                f"Dialogue: 0,{_format_time(start)},{_format_time(end)},Clip,,0,0,0,,{text}\n"
            )

    destination.parent.mkdir(parents=True, exist_ok=True)
    # libass reads this as UTF-8; BOM would land in the first style name.
    destination.write_text("".join(lines), encoding="utf-8")
    return destination
