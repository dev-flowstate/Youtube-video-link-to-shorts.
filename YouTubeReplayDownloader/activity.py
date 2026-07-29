"""Turn raw activity timestamps into heatmap points.

Chat messages and audio loudness are different signals, but both reduce to
"how much was happening at time T". Expressing them as HeatmapPoint lets the
existing peak detector consume them unchanged.
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter1d

from replay_fetcher import HeatmapPoint

# Resolution of the generated curve. Fine enough to locate a moment inside a
# long stream, coarse enough that chat noise averages out.
BUCKET_SECONDS = 5.0

# Very long streams would otherwise produce tens of thousands of buckets.
MAX_BUCKETS = 4000


def bucket_count(duration_s: float) -> tuple[int, float]:
    """Choose a bucket count and width for a video of this length."""
    if duration_s <= 0:
        return 0, BUCKET_SECONDS

    count = int(np.ceil(duration_s / BUCKET_SECONDS))
    if count <= MAX_BUCKETS:
        return count, BUCKET_SECONDS

    width = duration_s / MAX_BUCKETS
    return MAX_BUCKETS, width


# Chat arrives in bursts, so a raw curve is full of one-bucket spikes that are
# not moments. YouTube's own heatmap is only 100 points for a whole video and
# is already smooth; these curves are far finer, so they need smoothing over a
# comparable timescale before peaks mean anything.
SMOOTH_SECONDS = 30.0


def smooth_curve(values: np.ndarray, width_s: float, span_s: float = SMOOTH_SECONDS) -> np.ndarray:
    """Blur an activity curve over a fixed timespan, not a fixed bucket count."""
    if values.size < 3 or width_s <= 0:
        return values

    sigma = max(0.5, span_s / width_s / 2.0)
    return gaussian_filter1d(values.astype(float), sigma=sigma)


def _normalize(values: np.ndarray) -> np.ndarray:
    """Scale to 0-1, matching the range YouTube's own heatmap uses."""
    if values.size == 0:
        return values

    low = float(np.min(values))
    high = float(np.max(values))
    if high - low < 1e-9:
        return np.zeros_like(values)

    return (values - low) / (high - low)


def curve_to_points(values: np.ndarray, width_s: float) -> list[HeatmapPoint]:
    """Convert a per-bucket activity curve into heatmap points."""
    normalized = _normalize(values.astype(float))
    duration_ms = int(round(width_s * 1000))

    return [
        HeatmapPoint(
            start_ms=int(round(index * width_s * 1000)),
            duration_ms=duration_ms,
            score=float(score),
        )
        for index, score in enumerate(normalized)
    ]


# Chat rate at the very start and end of a stream reflects people arriving and
# saying goodbye, not anything happening on screen. Left alone, the opening
# surge outscores every real moment.
EDGE_TRIM_SECONDS = 90.0


def flatten_edges(values: np.ndarray, width_s: float, trim_s: float = EDGE_TRIM_SECONDS) -> np.ndarray:
    """Damp the leading and trailing spike so it cannot win as a peak.

    The edges are pulled down to the median of the middle rather than zeroed,
    since a trough would distort the baseline the peak detector derives.
    """
    if values.size == 0 or trim_s <= 0 or width_s <= 0:
        return values

    edge = int(round(trim_s / width_s))
    if edge <= 0 or values.size <= edge * 2 + 2:
        return values

    middle = values[edge:-edge]
    if middle.size == 0:
        return values

    level = float(np.median(middle))
    damped = values.copy()
    damped[:edge] = np.minimum(damped[:edge], level)
    damped[-edge:] = np.minimum(damped[-edge:], level)
    return damped


def timestamps_to_points(
    timestamps_s: list[float],
    duration_s: float,
    *,
    trim_edges: bool = True,
) -> list[HeatmapPoint]:
    """Bucket event times into an activity curve.

    Used for chat: each message is one event, so bucket counts are message
    rate over time.
    """
    count, width = bucket_count(duration_s)
    if count == 0:
        return []

    counts = np.zeros(count, dtype=float)
    for time_s in timestamps_s:
        if time_s < 0 or time_s > duration_s:
            continue
        index = min(count - 1, int(time_s / width))
        counts[index] += 1.0

    if trim_edges:
        counts = flatten_edges(counts, width)

    return curve_to_points(smooth_curve(counts, width), width)
