"""Turn a flat word stream into short, punchy caption groups."""

from __future__ import annotations

import config
from models import CaptionGroup, Word

# A group is forced to end after these, so captions break where a human would.
_SENTENCE_ENDINGS = (".", "!", "?")
_CLAUSE_ENDINGS = (",", ";", ":")


def _ends_sentence(text: str) -> bool:
    return text.endswith(_SENTENCE_ENDINGS)


def _ends_clause(text: str) -> bool:
    return text.endswith(_CLAUSE_ENDINGS)


def _group_width(words: list[Word], candidate: Word) -> int:
    parts = [word.text for word in words] + [candidate.text]
    return len(" ".join(parts))


def _should_break(current: list[Word], candidate: Word) -> bool:
    """Decide whether candidate starts a new on-screen group."""
    if not current:
        return False

    if len(current) >= config.MAX_WORDS_PER_GROUP:
        return True

    if _group_width(current, candidate) > config.MAX_CHARS_PER_GROUP:
        return True

    previous = current[-1]
    if _ends_sentence(previous.text):
        return True

    # A clause break only splits when the group already has some weight,
    # otherwise single-word flashes pile up.
    if _ends_clause(previous.text) and len(current) >= 2:
        return True

    if candidate.start_s - previous.end_s >= config.GROUP_SPLIT_SILENCE_S:
        return True

    return False


def build_groups(words: list[Word]) -> list[CaptionGroup]:
    """Split words into groups sized for fast, readable captions."""
    groups: list[CaptionGroup] = []
    current: list[Word] = []

    for word in words:
        if _should_break(current, word):
            groups.append(CaptionGroup(words=current))
            current = []
        current.append(word)

    if current:
        groups.append(CaptionGroup(words=current))

    return _pad_short_groups(groups)


def _pad_short_groups(groups: list[CaptionGroup]) -> list[CaptionGroup]:
    """Stretch very brief groups so they do not flicker on screen.

    A group is only extended into the gap before the next one, never over it,
    so captions can never overlap.
    """
    for index, group in enumerate(groups):
        duration = group.end_s - group.start_s
        if duration >= config.MIN_GROUP_DURATION_S:
            continue

        wanted = config.MIN_GROUP_DURATION_S - duration
        if index + 1 < len(groups):
            available = max(0.0, groups[index + 1].start_s - group.end_s)
            wanted = min(wanted, available)

        if wanted <= 0:
            continue

        last = group.words[-1]
        group.words[-1] = Word(
            text=last.text,
            start_s=last.start_s,
            end_s=last.end_s + wanted,
        )

    return groups
