"""End a clip where a thought finishes, not merely where speech pauses.

Snapping to a pause stops clips cutting mid-word, but a pause happens between
sentences *and* inside them, so a clip can still stop halfway through an idea.
Punctuation is what marks the difference, and the transcript has it.

Length gives a little either way: finishing the sentence matters more than
hitting an exact runtime, but a clip that runs on and on defeats the point of
the format, so the search is bounded at both ends.
"""

from __future__ import annotations

import config
from models import CaptionGroup

# A thought ends at one of these. A comma or dash does not.
_TERMINATORS = (".", "!", "?")

# Breathing room after the final word so its tail is not clipped.
_TAIL_S = 0.35


def sentence_end_times(groups: list[CaptionGroup]) -> list[float]:
    """Times at which a complete sentence finishes."""
    ends: list[float] = []

    for group in groups:
        for word in group.words:
            if word.text.rstrip().endswith(_TERMINATORS):
                ends.append(word.end_s + _TAIL_S)

    return ends


def choose_end(
    groups: list[CaptionGroup],
    natural_end_s: float,
) -> float:
    """Pick the clip's end: the sentence ending nearest the natural cut.

    Nearest rather than latest. The natural end came from where audience
    activity died down, so it marks the moment the clip is about; drifting to
    the furthest sentence inside the budget would pad every clip out to the
    ceiling and bury the point.
    """
    if not config.END_ON_COMPLETE_THOUGHT:
        return natural_end_s

    candidates = [
        end
        for end in sentence_end_times(groups)
        if config.THOUGHT_MIN_SECONDS <= end <= config.THOUGHT_MAX_SECONDS
    ]
    if not candidates:
        return natural_end_s

    return min(candidates, key=lambda end: abs(end - natural_end_s))


def trim_groups(groups: list[CaptionGroup], end_s: float) -> list[CaptionGroup]:
    """Drop caption groups that fall past the chosen end."""
    return [group for group in groups if group.end_s <= end_s]
