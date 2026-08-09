"""Detect significant replay peaks and their natural segment boundaries."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks

from replay_fetcher import HeatmapPoint


# How long a clip may spend on the run-up before reaching its peak. Callers
# that care set it explicitly; this is the fallback for those that do not.
DEFAULT_LEAD_IN_S = 10.0


@dataclass(frozen=True)
class ReplaySegment:
    """A detected most-replayed clip window."""

    start_s: float
    end_s: float
    peak_s: float
    peak_score: float
    prominence: float

    @property
    def duration_s(self) -> float:
        return max(0.0, self.end_s - self.start_s)


def _scores_and_times(points: list[HeatmapPoint]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    scores = np.array([p.score for p in points], dtype=float)
    starts = np.array([p.start_s for p in points], dtype=float)
    ends = np.array([p.end_s for p in points], dtype=float)
    centers = (starts + ends) / 2.0
    return scores, starts, ends, centers


def _smooth_scores(scores: np.ndarray, sigma: float = 1.2) -> np.ndarray:
    if len(scores) < 3:
        return scores.copy()
    return gaussian_filter1d(scores, sigma=sigma)


def _baseline_and_threshold(smoothed: np.ndarray) -> tuple[float, float, float]:
    baseline = float(np.percentile(smoothed, 20))
    peak = float(np.max(smoothed))
    spread = max(peak - baseline, 1e-6)

    # Segment ends when activity returns close to baseline.
    threshold = baseline + spread * 0.15
    return baseline, threshold, spread


def _peak_prominence(smoothed: np.ndarray, peak_index: int) -> float:
    peak_value = float(smoothed[peak_index])
    left_min = float(np.min(smoothed[: peak_index + 1])) if peak_index >= 0 else peak_value
    right_min = float(np.min(smoothed[peak_index:])) if peak_index < len(smoothed) else peak_value
    return peak_value - max(left_min, right_min)


def _find_peak_indices(
    smoothed: np.ndarray,
    baseline: float,
    spread: float,
    *,
    min_prominence_ratio: float,
    min_height_ratio: float,
    min_distance_buckets: int,
) -> list[int]:
    prominence_floor = max(spread * min_prominence_ratio, 0.015)
    height_floor = baseline + spread * min_height_ratio

    peak_indices, properties = find_peaks(
        smoothed,
        height=height_floor,
        prominence=prominence_floor,
        distance=min_distance_buckets,
    )

    indices = [int(i) for i in peak_indices]

    # Edge peaks (like strong video openings) often fail prominence checks.
    global_peak = int(np.argmax(smoothed))
    if global_peak not in indices and smoothed[global_peak] >= height_floor:
        indices.append(global_peak)

    # Local-max fallback catches secondary peaks find_peaks may skip.
    for i in range(1, len(smoothed) - 1):
        if smoothed[i] < height_floor:
            continue
        if smoothed[i] >= smoothed[i - 1] and smoothed[i] > smoothed[i + 1]:
            prominence = _peak_prominence(smoothed, i)
            if prominence >= prominence_floor and i not in indices:
                indices.append(i)

    indices.sort()
    filtered: list[int] = []
    for index in indices:
        if not filtered or index - filtered[-1] >= min_distance_buckets:
            filtered.append(index)
        elif smoothed[index] > smoothed[filtered[-1]]:
            filtered[-1] = index

    return filtered


def _expand_segment(
    smoothed: np.ndarray,
    starts: np.ndarray,
    ends: np.ndarray,
    peak_index: int,
    threshold: float,
    baseline: float,
) -> tuple[float, float]:
    left = peak_index
    while left > 0 and smoothed[left - 1] >= threshold:
        left -= 1

    right = peak_index
    while right < len(smoothed) - 1 and smoothed[right + 1] >= threshold:
        right += 1

    if left > 0 and smoothed[left - 1] > baseline:
        left -= 1
    if right < len(smoothed) - 1 and smoothed[right + 1] > baseline:
        right += 1

    return float(starts[left]), float(ends[right])


def _trim_to_peak(
    start_s: float,
    end_s: float,
    peak_s: float,
    max_seconds: float,
    lead_in_s: float = DEFAULT_LEAD_IN_S,
) -> tuple[float, float]:
    """Shorten a segment to a budget, opening just before the peak.

    Deliberately **not** centred. The peak is the moment people came back for,
    and centring it spends the first half of the clip on the run-up - which is
    the part a viewer scrolls past before the clip has made its case. On a 90s
    clip that was 45 seconds of buildup before anything happened.

    A short lead-in is kept rather than none: a podcast payoff is a sentence,
    and opening exactly on it lands mid-thought. Ten seconds carries the setup
    without burying the point, and boundary snapping widens it to the start of
    that sentence afterwards.

    The window is slid back inside the original bounds if the lead-in would
    push it past either edge, so a peak near the start still gets a full clip.
    """
    if max_seconds <= 0 or (end_s - start_s) <= max_seconds:
        return start_s, end_s

    # Never give more than half the clip to the run-up, however it is set.
    lead = max(0.0, min(lead_in_s, max_seconds / 2.0))

    new_start = peak_s - lead
    new_end = new_start + max_seconds

    if new_start < start_s:
        new_start = start_s
        new_end = start_s + max_seconds
    elif new_end > end_s:
        new_end = end_s
        new_start = end_s - max_seconds

    return max(start_s, new_start), min(end_s, new_end)


def _segments_overlap(a: ReplaySegment, b: ReplaySegment, overlap_ratio: float = 0.5) -> bool:
    overlap = min(a.end_s, b.end_s) - max(a.start_s, b.start_s)
    if overlap <= 0:
        return False

    shorter = min(a.duration_s, b.duration_s)
    if shorter <= 0:
        return False

    return (overlap / shorter) >= overlap_ratio


def _merge_overlapping(segments: list[ReplaySegment]) -> list[ReplaySegment]:
    if not segments:
        return []

    ordered = sorted(segments, key=lambda s: s.start_s)
    merged: list[ReplaySegment] = [ordered[0]]

    for segment in ordered[1:]:
        last = merged[-1]
        if _segments_overlap(last, segment):
            merged[-1] = ReplaySegment(
                start_s=min(last.start_s, segment.start_s),
                end_s=max(last.end_s, segment.end_s),
                peak_s=last.peak_s if last.peak_score >= segment.peak_score else segment.peak_s,
                peak_score=max(last.peak_score, segment.peak_score),
                prominence=max(last.prominence, segment.prominence),
            )
        else:
            merged.append(segment)

    return merged


def detect_replay_segments(
    points: list[HeatmapPoint],
    *,
    min_prominence_ratio: float = 0.07,
    min_height_ratio: float = 0.14,
    min_distance_buckets: int = 2,
    min_segment_seconds: float = 4.0,
    max_segment_seconds: float = 90.0,
    lead_in_seconds: float = DEFAULT_LEAD_IN_S,
) -> list[ReplaySegment]:
    """
    Find all confident replay peaks and their start/end boundaries.

    Boundaries are chosen where smoothed replay activity rises above a
    baseline-derived threshold and returns near normal again, then trimmed
    around the peak so no clip outstays a viewer's attention.
    """
    if len(points) < 4:
        return []

    scores, starts, ends, centers = _scores_and_times(points)
    smoothed = _smooth_scores(scores)
    baseline, threshold, spread = _baseline_and_threshold(smoothed)

    peak_indices = _find_peak_indices(
        smoothed,
        baseline,
        spread,
        min_prominence_ratio=min_prominence_ratio,
        min_height_ratio=min_height_ratio,
        min_distance_buckets=min_distance_buckets,
    )

    if not peak_indices:
        return []

    segments: list[ReplaySegment] = []

    for peak_index in peak_indices:
        start_s, end_s = _expand_segment(
            smoothed,
            starts,
            ends,
            peak_index,
            threshold,
            baseline,
        )

        if end_s - start_s < min_segment_seconds:
            continue

        peak_s = float(centers[peak_index])
        start_s, end_s = _trim_to_peak(
            start_s, end_s, peak_s, max_segment_seconds, lead_in_seconds
        )

        segments.append(
            ReplaySegment(
                start_s=start_s,
                end_s=end_s,
                peak_s=peak_s,
                peak_score=float(smoothed[peak_index]),
                prominence=_peak_prominence(smoothed, peak_index),
            )
        )

    segments = _merge_overlapping(segments)

    # Merging spans two windows, so it can exceed the budget again.
    trimmed: list[ReplaySegment] = []
    for segment in segments:
        start_s, end_s = _trim_to_peak(
            segment.start_s,
            segment.end_s,
            segment.peak_s,
            max_segment_seconds,
            lead_in_seconds,
        )
        trimmed.append(
            ReplaySegment(
                start_s=start_s,
                end_s=end_s,
                peak_s=segment.peak_s,
                peak_score=segment.peak_score,
                prominence=segment.prominence,
            )
        )

    trimmed.sort(key=lambda s: s.start_s)
    return trimmed
