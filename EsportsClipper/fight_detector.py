"""Turn audio features into a "this is a fight" curve.

Two signals with different timing:

* **gunfire** happens *during* the fight
* **caster excitement** happens *after* it - the scream lands on the knock,
  which is the end of the engagement

So caster energy is folded backwards before the two are combined, and they are
combined by multiplication rather than addition: gunfire gates, hype amplifies.
A ceremony is the loudest audio of the day and has no gunfire, so it has to
score zero however loud it gets.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import maximum_filter1d, median_filter

import config
import shared  # noqa: F401  (puts the sibling project on sys.path)
from activity import curve_to_points, flatten_edges, smooth_curve
from audio_features import FrameFeatures
from replay_fetcher import HeatmapPoint

# Window for the adaptive flux threshold, in frames. About 1.6s at a 16ms hop -
# long enough to span a burst, short enough to track a changing mix.
_MEDIAN_FRAMES = 101


@dataclass(frozen=True)
class FightCurves:
    """Per-bucket curves over the whole broadcast."""

    gunfire: np.ndarray      # onsets per bucket, raw counts
    hype: np.ndarray         # caster energy, folded backwards
    score: np.ndarray        # the gated combination
    width_s: float

    def times(self) -> np.ndarray:
        return np.arange(len(self.score), dtype=float) * self.width_s


def robust_normalise(values: np.ndarray, percentile: float | None = None) -> np.ndarray:
    """Scale to 0-1 without letting a few enormous moments set the range.

    A match-winning scream is many times louder than an ordinary good moment.
    Dividing by the maximum pushes every ordinary fight towards zero and makes
    the feature useless for the ~45 clips that are not chicken dinners.

    **Log compression** always: the gap between a celebration and a good fight
    becomes a factor of two rather than twenty, and every value keeps its
    ordering. Counts and loudness are both perceived this way anyway - forty
    shots is not twice the fight that twenty is.

    **Percentile clipping only when asked**, because it is not free: it ties
    everything above the cut at 1.0, and the strongest fights are exactly what
    has to stay rankable. So it is used for caster energy, where the ~6
    match-winning screams are genuine outliers and saturating them costs
    nothing, and *not* for gunfire, which has no such outliers - a big fight
    has perhaps three times the shooting of a small one, never twenty.
    """
    if values.size == 0:
        return values

    compressed = np.log1p(np.clip(values.astype(float), 0.0, None))

    low = float(np.min(compressed))
    if percentile is None:
        high = float(np.max(compressed))
    else:
        high = float(np.percentile(compressed, percentile))
        if high - low < 1e-9:
            high = float(np.max(compressed))

    if high - low < 1e-9:
        return np.zeros_like(compressed)

    return np.clip((compressed - low) / (high - low), 0.0, 1.0)


def find_gunfire_onsets(features: FrameFeatures) -> np.ndarray:
    """Frame indices that look like a gunshot rather than speech.

    Two conditions together: a sharp rise against the local median, and most of
    the energy above 4 kHz. Loud speech satisfies the first but not the second.
    """
    flux = features.flux
    if flux.size < 3:
        return np.empty(0, dtype=int)

    baseline = median_filter(flux, size=_MEDIAN_FRAMES, mode="nearest")
    loud_enough = flux > baseline * config.ONSET_FLUX_FACTOR

    # A peak, so one long rumble counts once rather than for every frame.
    is_peak = np.zeros_like(loud_enough)
    is_peak[1:-1] = (flux[1:-1] >= flux[:-2]) & (flux[1:-1] > flux[2:])

    broadband = features.hf_ratio >= config.HF_RATIO_MIN
    return np.flatnonzero(loud_enough & is_peak & broadband)


def _bucket_index(times_s: np.ndarray, width_s: float, count: int) -> np.ndarray:
    return np.clip((times_s / width_s).astype(int), 0, count - 1)


def _music_penalty(onset_times: np.ndarray, bucket_of: np.ndarray, count: int) -> np.ndarray:
    """Damp buckets whose onsets are evenly spaced.

    Ceremony music and sting-outs are broadband transients too - a drum track
    reads like gunfire on flux alone. The difference is rhythm: music is
    metronomic, a firefight is ragged.
    """
    penalty = np.ones(count, dtype=float)
    if onset_times.size < 4:
        return penalty

    for bucket in np.unique(bucket_of):
        times = onset_times[bucket_of == bucket]
        if times.size < 4:
            continue

        gaps = np.diff(times)
        mean = float(np.mean(gaps))
        if mean <= 0:
            continue

        variation = float(np.std(gaps)) / mean
        if variation < config.MUSIC_REGULARITY_MIN:
            # Scale down towards zero as the pattern gets more regular.
            penalty[bucket] = variation / config.MUSIC_REGULARITY_MIN

    return penalty


def build_curves(features: FrameFeatures, duration_s: float) -> FightCurves:
    """Bucket the frame features and score every bucket."""
    width = config.BUCKET_SECONDS
    count = max(1, int(np.ceil(duration_s / width)))

    onsets = find_gunfire_onsets(features)
    onset_times = onsets * features.hop_seconds
    onset_bucket = _bucket_index(onset_times, width, count)

    gunfire = np.bincount(onset_bucket, minlength=count).astype(float)[:count]
    gunfire *= _music_penalty(onset_times, onset_bucket, count)

    # Caster loudness per bucket, from the speech band only.
    frame_bucket = _bucket_index(features.times(), width, count)
    caster = np.bincount(frame_bucket, weights=features.speech_energy, minlength=count)
    frames_per = np.bincount(frame_bucket, minlength=count)
    caster = caster[:count] / np.maximum(frames_per[:count], 1)

    # Fold the reaction backwards over the moment that caused it.
    lookahead = max(1, int(round(config.CASTER_LOOKAHEAD_SECONDS / width)))
    # origin shifts the window forward, so each bucket sees what is coming.
    caster_ahead = maximum_filter1d(
        caster, size=lookahead, mode="nearest", origin=-(lookahead // 2)
    )

    # Gunfire has no extreme outliers, so it keeps its full range and the
    # biggest fights stay rankable against each other. Caster energy does have
    # them - every chicken dinner - so its tail is clipped.
    fight = robust_normalise(gunfire)
    hype = robust_normalise(caster_ahead, config.NORMALISE_PERCENTILE)

    score = fight * (config.HYPE_BASE + config.HYPE_GAIN * hype)

    # A handful of stray shots is not a fight, however loud what follows is.
    score[gunfire < config.MIN_GUNFIRE_ONSETS] = 0.0

    score = flatten_edges(score, width, config.EDGE_TRIM_SECONDS)
    score = smooth_curve(score, width, config.SMOOTH_SECONDS)

    return FightCurves(gunfire=gunfire, hype=hype, score=score, width_s=width)


def to_points(curves: FightCurves) -> list[HeatmapPoint]:
    """Hand the score to the existing peak detector in the shape it expects."""
    return curve_to_points(curves.score, curves.width_s)
