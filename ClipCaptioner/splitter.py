"""Split a long clip into Shorts-length parts on natural speech gaps."""

from __future__ import annotations

import config
from models import CaptionGroup, ClipPart


def _cut_points(groups: list[CaptionGroup], total_duration_s: float) -> list[float]:
    """Choose split times, preferring the gap after a caption group.

    Cutting mid-word is what makes auto-split clips feel cheap, so each cut
    lands on the last group boundary that fits inside the budget.
    """
    cuts: list[float] = []
    window_start = 0.0
    index = 0

    while total_duration_s - window_start > config.MAX_PART_DURATION_S:
        budget_end = window_start + config.MAX_PART_DURATION_S
        best: float | None = None

        while index < len(groups) and groups[index].end_s <= budget_end:
            best = groups[index].end_s
            index += 1

        # No group boundary in this window (long silence or music) - cut flat.
        if best is None or best <= window_start:
            best = budget_end
            while index < len(groups) and groups[index].start_s < best:
                index += 1

        cuts.append(best)
        window_start = best

    return cuts


def _groups_in_range(
    groups: list[CaptionGroup], start_s: float, end_s: float
) -> list[CaptionGroup]:
    """Groups fully inside the window, rebased so the part starts at zero."""
    selected = [
        group for group in groups if group.start_s >= start_s and group.end_s <= end_s
    ]
    return [group.shifted(-start_s) for group in selected]


def split_into_parts(
    groups: list[CaptionGroup], total_duration_s: float
) -> list[ClipPart]:
    """Break a clip into renderable parts, or return a single whole part."""
    if not config.SPLIT_LONG_CLIPS or total_duration_s <= config.MAX_PART_DURATION_S:
        return [
            ClipPart(
                index=1,
                total=1,
                start_s=0.0,
                end_s=total_duration_s,
                groups=_groups_in_range(groups, 0.0, total_duration_s),
            )
        ]

    boundaries = [0.0, *_cut_points(groups, total_duration_s), total_duration_s]

    # Fold a stubby tail into the previous part rather than shipping it.
    if len(boundaries) >= 3:
        tail = boundaries[-1] - boundaries[-2]
        if tail < config.MIN_PART_DURATION_S:
            boundaries.pop(-2)

    parts: list[ClipPart] = []
    total = len(boundaries) - 1
    for position in range(total):
        start_s = boundaries[position]
        end_s = boundaries[position + 1]
        parts.append(
            ClipPart(
                index=position + 1,
                total=total,
                start_s=start_s,
                end_s=end_s,
                groups=_groups_in_range(groups, start_s, end_s),
            )
        )

    return parts
