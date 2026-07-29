"""Pick the best available signal for locating clip-worthy moments.

Most-replayed data is preferred whenever it exists: it is a direct measure of
what viewers rewatched. Live streams do not have one until well after the
broadcast ends, so chat and audio stand in.
"""

from __future__ import annotations

from dataclasses import dataclass

import yt_dlp

import audio_peaks
import chat_fetcher
from replay_fetcher import HeatmapPoint, ReplayDataNotAvailable, fetch_replay_data
from utils import parse_youtube_url


class NoMomentSignal(Exception):
    """Raised when every detection method failed."""


@dataclass(frozen=True)
class MomentData:
    """Activity points plus a note on where they came from."""

    points: list[HeatmapPoint]
    source: str
    duration_s: float


def fetch_video_facts(youtube_url: str) -> tuple[float, bool]:
    """Return the video's duration and whether it is still broadcasting."""
    canonical_url = parse_youtube_url(youtube_url)
    options = {"quiet": True, "no_warnings": True, "skip_download": True}

    with yt_dlp.YoutubeDL(options) as ydl:
        info = ydl.extract_info(canonical_url, download=False)

    duration = info.get("duration") or 0.0
    is_live = bool(info.get("is_live"))
    return float(duration), is_live


def find_moments(youtube_url: str) -> MomentData:
    """Try each signal in order of quality and return the first that works."""
    duration_s, is_live = fetch_video_facts(youtube_url)

    if is_live:
        raise NoMomentSignal(
            "This stream is still live. Wait until the broadcast ends - a "
            "clip cannot be cut from a video that is still growing."
        )

    if duration_s <= 0:
        raise NoMomentSignal("Could not determine the video's duration.")

    attempts: list[str] = []

    try:
        points = fetch_replay_data(youtube_url)
        return MomentData(points=points, source="most-replayed", duration_s=duration_s)
    except (ReplayDataNotAvailable, RuntimeError) as exc:
        attempts.append(f"most-replayed: {exc}")

    try:
        points = chat_fetcher.fetch_chat_activity(youtube_url, duration_s)
        return MomentData(points=points, source="chat activity", duration_s=duration_s)
    except chat_fetcher.ChatDataNotAvailable as exc:
        attempts.append(f"chat: {exc}")

    try:
        points = audio_peaks.fetch_audio_activity(youtube_url, duration_s)
        return MomentData(points=points, source="audio loudness", duration_s=duration_s)
    except audio_peaks.AudioAnalysisFailed as exc:
        attempts.append(f"audio: {exc}")

    detail = "\n  ".join(attempts)
    raise NoMomentSignal(f"No usable signal for this video:\n  {detail}")
