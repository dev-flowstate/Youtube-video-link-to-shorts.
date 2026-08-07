"""Pick the best available signal for locating clip-worthy moments.

Most-replayed data is preferred whenever it exists: it is a direct measure of
what viewers rewatched. Live streams do not have one until well after the
broadcast ends, so chat activity stands in.

Both signals measure audience reaction. Loudness detection was tried and
removed: it flags loud intros and background music as readily as real
moments, and a wrong clip costs more than a missing one.
"""

from __future__ import annotations

from dataclasses import dataclass

import yt_dlp

import chat_fetcher
import stream_energy
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


def _energy_moments(youtube_url: str, duration_s: float, live: bool) -> MomentData:
    """Last resort: read the audio itself."""
    points, measured = stream_energy.fetch_energy_activity(
        youtube_url, duration_s, live_from_start=live
    )
    return MomentData(points=points, source="speech energy", duration_s=measured)


def find_moments(youtube_url: str) -> MomentData:
    """Try each signal in order of quality and return the first that works."""
    duration_s, is_live = fetch_video_facts(youtube_url)

    # A running broadcast has no audience data of any kind and reports no
    # duration, so go straight to the audio and measure what exists so far.
    if is_live:
        print(
            "Stream is still live - clipping the part broadcast so far.\n"
            "  This pulls the broadcast from its beginning, so a stream that\n"
            "  has been running for hours takes a long time to fetch before\n"
            "  anything can be analysed. Clipping it after it ends is much\n"
            "  faster and gives better moments."
        )
        return _energy_moments(youtube_url, 0.0, live=True)

    attempts: list[str] = []

    if duration_s > 0:
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
    else:
        attempts.append("duration unknown, skipped audience signals")

    # A stream that has just ended has neither a replay graph nor published
    # chat yet, which would otherwise leave nothing to clip for days.
    for attempt in attempts:
        print(f"  {attempt}")
    print("Falling back to speech energy.")

    try:
        return _energy_moments(youtube_url, duration_s, live=False)
    except stream_energy.EnergyAnalysisFailed as exc:
        attempts.append(f"speech energy: {exc}")

    detail = "\n  ".join(attempts)
    raise NoMomentSignal(f"No usable signal for this video:\n  {detail}")
