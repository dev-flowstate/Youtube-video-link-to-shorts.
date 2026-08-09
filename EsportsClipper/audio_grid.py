"""Audio features on the same half-second grid the vision features use.

`audio_features.extract` works at a 16 ms hop, which is far finer than any
decision this makes. These aggregate it down to the model's grid and add the
shapes that carry meaning on a broadcast.

Two of those shapes are the whole point.

**Causal z-score.** Absolute caster loudness is useless here - the casters are
loud continuously for five hours, so a plain energy feature is high everywhere.
What marks a moment is the caster getting louder *than he has been*, which is a
local comparison. It is computed against the preceding minute only, never the
whole track, so nothing about the future leaks backwards into training.

**Lookahead.** The scream lands on the kill, and the kill ends the fight. A
feature reading only the present scores the run-up at nothing. Taking a forward
maximum lets the shooting inherit the weight of the reaction it earned. That
makes the model non-causal by half a minute, which is free when clipping a
recording.

The gunfire count is here as a **negative control**. It was measured inverted on
this footage - studio talk out-scored real firefights 1.03 to 0.43 onsets per
second. It stays so the importance table can put a number on it instead of
relying on an argument.
"""

from __future__ import annotations

import numpy as np

import config
import shared  # noqa: F401  (puts the sibling project on sys.path)
from audio_features import FrameFeatures
from fight_detector import find_gunfire_onsets

FEATURE_NAMES: tuple[str, ...] = (
    "aud_flux_mean",
    "aud_flux_p95",
    "aud_gunfire_onsets",     # negative control - measured dead on this footage
    "aud_hf_ratio_mean",
    "aud_speech_energy",
    "aud_speech_z60",         # the caster spike, as a local comparison
    "aud_speech_ahead30",     # the reaction, folded back onto its cause
    "aud_silence_frac",
    "aud_low_energy",         # crowd rumble and explosions
)


def _bucket(values: np.ndarray, edges: np.ndarray, reducer) -> np.ndarray:
    """Reduce a fine-grained series into the coarse grid, bucket by bucket."""
    out = np.full(len(edges) - 1, np.nan, dtype=np.float32)
    for index in range(len(edges) - 1):
        chunk = values[edges[index] : edges[index + 1]]
        if chunk.size:
            out[index] = reducer(chunk)
    return out


def _causal_z(values: np.ndarray, window: int) -> np.ndarray:
    """How unusual each value is against the ones before it, and only before.

    A trailing window rather than a centred one. A centred window would let a
    moment be judged partly by what follows it, which is future information and
    would flatter every score measured afterwards.
    """
    out = np.full(len(values), np.nan, dtype=np.float32)
    if len(values) == 0:
        return out

    filled = np.nan_to_num(values, nan=0.0).astype(np.float64)
    cumulative = np.concatenate(([0.0], np.cumsum(filled)))
    cumulative_sq = np.concatenate(([0.0], np.cumsum(filled * filled)))

    for index in range(len(values)):
        start = max(0, index - window)
        count = index - start
        if count < window // 4:
            # Too little history to say what normal looks like yet.
            continue
        total = cumulative[index] - cumulative[start]
        total_sq = cumulative_sq[index] - cumulative_sq[start]
        mean = total / count
        variance = max(total_sq / count - mean * mean, 0.0)
        spread = float(np.sqrt(variance))
        if spread < 1e-9:
            out[index] = 0.0
        else:
            out[index] = (filled[index] - mean) / spread

    return out


def _forward_max(values: np.ndarray, window: int) -> np.ndarray:
    """The largest value in the next `window` steps, this one included."""
    out = np.full(len(values), np.nan, dtype=np.float32)
    filled = np.nan_to_num(values, nan=-np.inf)
    for index in range(len(values)):
        chunk = filled[index : index + window + 1]
        if chunk.size:
            best = float(np.max(chunk))
            out[index] = best if np.isfinite(best) else np.nan
    return out


def build(features: FrameFeatures, times_s: np.ndarray) -> np.ndarray:
    """Audio features aligned to `times_s`, one row each.

    `times_s` comes from the vision pass and is authoritative. Audio is fitted
    to it rather than the other way round, so the two halves of a row always
    describe the same instant.
    """
    step = float(times_s[1] - times_s[0]) if len(times_s) > 1 else 0.5
    hop = features.hop_seconds

    # Bucket boundaries in audio-frame indices, one bucket per output row.
    edges = np.searchsorted(
        np.arange(len(features.flux), dtype=float) * hop,
        np.append(times_s, times_s[-1] + step if len(times_s) else step),
    )

    flux_mean = _bucket(features.flux, edges, np.mean)
    flux_p95 = _bucket(features.flux, edges, lambda c: np.percentile(c, 95))
    hf_mean = _bucket(features.hf_ratio, edges, np.mean)
    speech = _bucket(features.speech_energy, edges, np.mean)
    low = _bucket(features.low_energy, edges, np.mean)

    # Onsets are already the audio detector's own notion of a gunshot, so the
    # control measures exactly what that detector believed.
    onset_flags = np.zeros(len(features.flux), dtype=np.float32)
    onset_flags[find_gunfire_onsets(features)] = 1.0
    onsets = _bucket(onset_flags, edges, np.sum)

    # Silence is measured against this broadcast's own speech level, not an
    # absolute threshold, because mixes differ between feeds.
    reference = float(np.nanpercentile(speech, 60)) if np.any(np.isfinite(speech)) else 0.0
    quiet = (features.speech_energy < reference * config.SILENCE_FRACTION).astype(np.float32)
    silence = _bucket(quiet, edges, np.mean)

    z_window = max(1, int(round(config.SPEECH_Z_WINDOW_SECONDS / step)))
    ahead = max(1, int(round(config.CASTER_LOOKAHEAD_SECONDS / step)))

    columns = [
        flux_mean,
        flux_p95,
        onsets,
        hf_mean,
        speech,
        _causal_z(speech, z_window),
        _forward_max(_causal_z(speech, z_window), ahead),
        silence,
        low,
    ]
    return np.column_stack(columns).astype(np.float32)


def missing(rows: int) -> np.ndarray:
    """All-NaN audio, for footage whose soundtrack says nothing about fights.

    Highlight reels usually have the broadcast audio replaced with music. Left
    as real numbers those rows would teach the model that music means combat,
    which is true of an edit and false of a live broadcast. NaN says "unknown",
    which is what it is, and the classifier handles that natively.
    """
    return np.full((rows, len(FEATURE_NAMES)), np.nan, dtype=np.float32)
