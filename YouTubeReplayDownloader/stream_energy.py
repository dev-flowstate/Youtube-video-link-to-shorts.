"""Find moments from the audio alone, when no audience data exists.

A live stream has no most-replayed graph, and a stream that just ended has no
chat replay yet either. Both leave nothing to measure the audience by, so this
measures the speaker instead.

Raw loudness was tried once and abandoned: it flags an intro sting or
background music as readily as a real moment. The difference here is that
loudness is weighted by how much a bucket actually sounds like *speech*.
Talking carries a steady rhythm of short gaps - breaths, syllable breaks,
turn-taking - while music, applause and room noise run continuous. A bucket
with no gaps at all is not someone making a point, however loud it is.
"""

from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path

import numpy as np

import activity
import speech
from ffmpeg_utils import ensure_ffmpeg_available
from replay_fetcher import HeatmapPoint
from utils import parse_youtube_url

SAMPLE_RATE = 16_000


class EnergyAnalysisFailed(Exception):
    """Raised when the audio could not be analysed."""


# Pause rates that read as conversation, in gaps per minute. Below the floor
# the audio is continuous - music, applause, a held note. Above the ceiling it
# is broken speech or silence with occasional noise.
MIN_PAUSE_RATE = 2.0
IDEAL_PAUSE_RATE = 9.0
MAX_PAUSE_RATE = 30.0

# Floor on the speech weighting, so a loud moment in an unusual passage is
# damped rather than erased.
MIN_SPEECH_WEIGHT = 0.15


def _read_samples(audio_path: Path) -> np.ndarray:
    """Decode to mono 16 kHz PCM in memory."""
    ensure_ffmpeg_available()
    args = [
        "ffmpeg",
        "-v",
        "error",
        "-i",
        str(audio_path),
        "-vn",
        "-ac",
        "1",
        "-ar",
        str(SAMPLE_RATE),
        "-f",
        "s16le",
        "-",
    ]
    result = subprocess.run(args, capture_output=True, check=False)
    if result.returncode != 0 or not result.stdout:
        stderr = result.stderr.decode("utf-8", "replace").strip()
        raise EnergyAnalysisFailed(f"Could not decode audio: {stderr[-300:]}")

    return np.frombuffer(result.stdout, dtype=np.int16).astype(np.float32) / 32768.0


def _loudness_db(samples: np.ndarray, width_s: float, count: int) -> np.ndarray:
    """RMS energy per bucket, in dB.

    dB because loudness is perceived logarithmically; on a linear scale one
    shout flattens everything else into the baseline.
    """
    per_bucket = max(1, int(round(width_s * SAMPLE_RATE)))
    curve = np.zeros(count, dtype=float)
    filled = 0

    for index in range(count):
        chunk = samples[index * per_bucket : (index + 1) * per_bucket]
        if chunk.size == 0:
            break
        rms = float(np.sqrt(np.mean(np.square(chunk))))
        curve[index] = 20.0 * np.log10(max(rms, 1e-6))
        filled = index + 1

    if filled == 0:
        raise EnergyAnalysisFailed("Audio track is too short to analyse.")

    # Trailing buckets past the end of the audio would read as a deep trough.
    if filled < count:
        curve[filled:] = float(np.median(curve[:filled]))

    return curve


def _pause_rate(silences: list[tuple[float, float]], width_s: float, count: int) -> np.ndarray:
    """Gaps per minute in each bucket."""
    counts = np.zeros(count, dtype=float)
    for start_s, _end_s in silences:
        index = int(start_s / width_s)
        if 0 <= index < count:
            counts[index] += 1.0

    return counts * (60.0 / width_s)


def _speech_weight(pause_rate: np.ndarray) -> np.ndarray:
    """How much each bucket sounds like conversation, from 0 to 1.

    Peaks at a natural speaking rhythm and falls away on either side, so
    continuous audio - music, applause, a droning room - cannot score highly
    on loudness alone.
    """
    weight = np.zeros_like(pause_rate)

    rising = (pause_rate >= MIN_PAUSE_RATE) & (pause_rate < IDEAL_PAUSE_RATE)
    weight[rising] = (pause_rate[rising] - MIN_PAUSE_RATE) / (IDEAL_PAUSE_RATE - MIN_PAUSE_RATE)

    falling = (pause_rate >= IDEAL_PAUSE_RATE) & (pause_rate <= MAX_PAUSE_RATE)
    weight[falling] = 1.0 - 0.6 * (
        (pause_rate[falling] - IDEAL_PAUSE_RATE) / (MAX_PAUSE_RATE - IDEAL_PAUSE_RATE)
    )

    return np.clip(weight, MIN_SPEECH_WEIGHT, 1.0)


def _excitement(loudness_db: np.ndarray, weight: np.ndarray) -> np.ndarray:
    """Combine relative loudness with how much a bucket sounds like speech.

    Loudness is rescaled across the whole recording rather than measured
    against a threshold, which keeps quiet recordings and loud ones on the
    same footing and preserves the ordering of everything below average.
    Subtracting a baseline and clipping at zero was tried first and flattened
    the entire quieter half of a stream into a tie.
    """
    low = float(np.min(loudness_db))
    high = float(np.max(loudness_db))
    if high - low < 1e-9:
        return np.zeros_like(loudness_db)

    scaled = (loudness_db - low) / (high - low)
    return scaled * weight


def measure_duration(audio_path: Path) -> float:
    """Length of a downloaded track, in seconds."""
    ensure_ffmpeg_available()
    args = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(audio_path),
    ]
    result = subprocess.run(args, capture_output=True, text=True, check=False)
    try:
        return float(result.stdout.strip())
    except (TypeError, ValueError):
        return 0.0


def fetch_energy_activity(
    youtube_url: str,
    duration_s: float = 0.0,
    live_from_start: bool = False,
) -> tuple[list[HeatmapPoint], float]:
    """Build an activity curve from how the speaker sounds over time.

    Returns the curve and the duration it covers. A stream that is still
    running reports no duration of its own, so the downloaded audio is
    measured instead - that length is also how much video exists to cut from.
    """
    with tempfile.TemporaryDirectory(prefix="energy_") as tmp:
        try:
            path = speech.download_audio(youtube_url, Path(tmp), live_from_start)
        except speech.SpeechAnalysisFailed as exc:
            raise EnergyAnalysisFailed(str(exc)) from exc

        measured = measure_duration(path)
        if measured > 0:
            duration_s = measured
        if duration_s <= 0:
            raise EnergyAnalysisFailed("Could not determine how long the audio is.")

        count, width = activity.bucket_count(duration_s)
        if count == 0:
            raise EnergyAnalysisFailed("Audio is too short to analyse.")

        samples = _read_samples(path)
        speech_map = speech.analyse_audio(path)

    loudness = _loudness_db(samples, width, count)
    rate = _pause_rate(speech_map.silences, width, count)
    curve = _excitement(loudness, _speech_weight(rate))

    curve = activity.flatten_edges(curve, width)
    points = activity.curve_to_points(activity.smooth_curve(curve, width), width)
    return points, duration_s
