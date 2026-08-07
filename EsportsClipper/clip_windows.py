"""Turn detected peaks into clip windows that hold the fight and nothing else.

Two departures from the podcast pipeline, both deliberate:

* The window is **not centred on the peak**. The caster screams at the kill,
  which ends an engagement, so a centred window spends half its length on the
  aftermath and cuts off the fight that earned it.
* The start is **not a fixed offset**. It follows the gunfire backwards and
  stops at the first real gap, so a clip opens when shooting opens instead of
  on thirty seconds of someone driving to the zone.
"""

from __future__ import annotations

import numpy as np

import config
import shared  # noqa: F401  (puts the sibling project on sys.path)
from fight_detector import FightCurves
from peak_detector import ReplaySegment


def _buckets(seconds: float, width_s: float) -> int:
    return max(1, int(round(seconds / width_s)))


def _find_start(curves: FightCurves, anchor: int) -> int:
    """Walk back from the peak for as long as shooting continues."""
    width = curves.width_s
    max_back = _buckets(config.MAX_PRE_ROLL_SECONDS, width)
    min_back = _buckets(config.MIN_PRE_ROLL_SECONDS, width)

    limit = max(0, anchor - max_back)
    window = curves.gunfire[limit : anchor + 1]
    if window.size == 0:
        return limit

    # Measured against this fight's own intensity, so a quiet exchange is not
    # judged by the standard of the loudest battle of the day.
    floor = float(np.max(window)) * config.ACTION_FLOOR

    start = anchor
    while start > limit and curves.gunfire[start - 1] >= floor:
        start -= 1

    # Always keep a little run-up, even for a burst that starts abruptly.
    return max(0, min(start, anchor - min_back))


def build_window(curves: FightCurves, peak_s: float, score: float) -> ReplaySegment | None:
    """Build one clip around a detected peak, or None if it is too slight."""
    width = curves.width_s
    anchor = int(np.clip(peak_s / width, 0, len(curves.gunfire) - 1))

    start_s = _find_start(curves, anchor) * width
    end_s = peak_s + config.POST_ROLL_SECONDS

    # Trim the front rather than the payoff: the reaction is the point.
    if end_s - start_s > config.MAX_CLIP_SECONDS:
        start_s = end_s - config.MAX_CLIP_SECONDS

    if end_s - start_s < config.MIN_CLIP_SECONDS:
        return None

    return ReplaySegment(
        start_s=max(0.0, start_s),
        end_s=end_s,
        peak_s=peak_s,
        peak_score=score,
        prominence=score,
    )


def windows_from_segments(
    curves: FightCurves,
    segments: list[ReplaySegment],
) -> list[ReplaySegment]:
    """Rebuild every detected span as a gunfire-driven window."""
    built: list[ReplaySegment] = []
    for segment in segments:
        window = build_window(curves, segment.peak_s, segment.peak_score)
        if window is not None:
            built.append(window)
    return built


def pick_best(segments: list[ReplaySegment], limit: int) -> list[ReplaySegment]:
    """Strongest fights first, skipping any that overlap one already taken."""
    chosen: list[ReplaySegment] = []

    for candidate in sorted(segments, key=lambda s: -s.peak_score):
        if len(chosen) >= limit:
            break

        overlaps = any(
            min(candidate.end_s, picked.end_s) > max(candidate.start_s, picked.start_s)
            for picked in chosen
        )
        if not overlaps:
            chosen.append(candidate)

    return sorted(chosen, key=lambda s: s.start_s)


def clip_budget(duration_s: float) -> int:
    """How many fights to keep from a broadcast of this length."""
    if duration_s <= 0:
        return config.MIN_CLIPS

    wanted = round((duration_s / 3600.0) * config.CLIPS_PER_HOUR)
    return int(max(config.MIN_CLIPS, min(config.MAX_CLIPS, wanted)))
