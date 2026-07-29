"""Find moments from audio loudness when no better signal exists.

Shouting, laughing and crowd reactions all show up as sustained loudness.
This is the last-resort detector: it works on literally any video, but it
cannot tell excitement from a loud intro or background music.
"""

from __future__ import annotations

import subprocess
import tempfile
import uuid
from pathlib import Path

import numpy as np
import yt_dlp

import activity
from ffmpeg_utils import ensure_ffmpeg_available, ffmpeg_location_option
from replay_fetcher import HeatmapPoint
from utils import parse_youtube_url

SAMPLE_RATE = 16_000


class AudioAnalysisFailed(Exception):
    """Raised when loudness analysis cannot be completed."""


def _download_audio(youtube_url: str, work_dir: Path) -> Path:
    """Grab the audio track only - far smaller and faster than the video."""
    base = work_dir / f"audio_{uuid.uuid4().hex}"

    options = {
        "format": "bestaudio/best",
        "outtmpl": str(base) + ".%(ext)s",
        "quiet": True,
        "no_warnings": True,
        "noprogress": True,
        "retries": 5,
    }
    options.update(ffmpeg_location_option())

    try:
        with yt_dlp.YoutubeDL(options) as ydl:
            ydl.download([youtube_url])
    except Exception as exc:
        raise AudioAnalysisFailed(f"Could not download audio: {exc}") from exc

    produced = [path for path in work_dir.glob(f"{base.name}*") if path.is_file()]
    if not produced:
        raise AudioAnalysisFailed("Audio download produced no file.")

    return produced[0]


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
        raise AudioAnalysisFailed(f"Could not decode audio: {stderr[-300:]}")

    return np.frombuffer(result.stdout, dtype=np.int16).astype(np.float32) / 32768.0


def _loudness_curve(samples: np.ndarray, width_s: float, count: int) -> np.ndarray:
    """RMS energy per bucket, in dB.

    dB rather than raw amplitude because loudness is perceived
    logarithmically - without it, one very loud moment flattens everything
    else into the baseline.
    """
    per_bucket = max(1, int(round(width_s * SAMPLE_RATE)))
    usable = min(count, int(np.ceil(len(samples) / per_bucket)))
    if usable == 0:
        raise AudioAnalysisFailed("Audio track is too short to analyse.")

    curve = np.zeros(count, dtype=float)
    for index in range(usable):
        chunk = samples[index * per_bucket : (index + 1) * per_bucket]
        if chunk.size == 0:
            continue
        rms = float(np.sqrt(np.mean(np.square(chunk))))
        curve[index] = 20.0 * np.log10(max(rms, 1e-6))

    # Silent tail buckets would otherwise read as a huge negative outlier.
    if usable < count:
        curve[usable:] = curve[:usable].min() if usable else 0.0

    return curve


def fetch_audio_activity(youtube_url: str, duration_s: float) -> list[HeatmapPoint]:
    """Build an activity curve from how loud the audio is over time."""
    canonical_url = parse_youtube_url(youtube_url)
    count, width = activity.bucket_count(duration_s)
    if count == 0:
        raise AudioAnalysisFailed("Unknown video duration.")

    with tempfile.TemporaryDirectory(prefix="audio_") as tmp:
        work_dir = Path(tmp)
        audio_path = _download_audio(canonical_url, work_dir)
        samples = _read_samples(audio_path)

    curve = _loudness_curve(samples, width, count)
    curve = activity.flatten_edges(curve, width)
    return activity.curve_to_points(activity.smooth_curve(curve, width), width)
