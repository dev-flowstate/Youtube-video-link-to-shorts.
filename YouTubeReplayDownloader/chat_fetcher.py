"""Find hype moments from live chat replay.

A spike in message rate is the strongest signal a stream has for "something
just happened" - it is what human clippers scrub for. Only works on past
broadcasts whose chat replay is still available.
"""

from __future__ import annotations

import json
import tempfile
import uuid
from pathlib import Path

import yt_dlp

import activity
from replay_fetcher import HeatmapPoint
from utils import parse_youtube_url


class ChatDataNotAvailable(Exception):
    """Raised when chat replay cannot be retrieved for a video."""


# Chat reacts after the fact: a viewer sees something, then types. Left
# uncorrected, every chat-derived clip lands late and the moment it exists for
# sits near the start of the window, or just before it. Shifting messages
# earlier puts the peak back over its cause. Tune by clipping one stream at a
# few values and seeing which centres best.
REACTION_LAG_S = 4.0


def _download_chat(youtube_url: str, work_dir: Path) -> Path:
    """Fetch the live chat replay track without downloading the video."""
    base = work_dir / f"chat_{uuid.uuid4().hex}"

    options = {
        "writesubtitles": True,
        "subtitleslangs": ["live_chat"],
        "skip_download": True,
        "outtmpl": str(base) + ".%(ext)s",
        "quiet": True,
        "no_warnings": True,
        "noprogress": True,
    }

    try:
        with yt_dlp.YoutubeDL(options) as ydl:
            ydl.download([youtube_url])
    except Exception as exc:
        raise ChatDataNotAvailable(f"Could not download chat replay: {exc}") from exc

    matches = list(work_dir.glob(f"{base.name}*live_chat*"))
    if not matches:
        raise ChatDataNotAvailable("This video has no chat replay.")

    return matches[0]


def _parse_offsets(chat_path: Path) -> list[float]:
    """Read each message's offset from the start of the stream, in seconds.

    The file is JSON Lines - one chat action per line - so it is read
    incrementally rather than loaded whole. A busy multi-hour stream can
    produce hundreds of megabytes here.
    """
    offsets: list[float] = []

    with chat_path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue

            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue

            action = payload.get("replayChatItemAction")
            if not isinstance(action, dict):
                continue

            raw_offset = action.get("videoOffsetTimeMsec")
            if raw_offset is None:
                continue

            try:
                offsets.append(float(raw_offset) / 1000.0)
            except (TypeError, ValueError):
                continue

    return offsets


def fetch_chat_activity(youtube_url: str, duration_s: float) -> list[HeatmapPoint]:
    """Build an activity curve from chat message rate over the stream."""
    canonical_url = parse_youtube_url(youtube_url)

    with tempfile.TemporaryDirectory(prefix="chat_") as tmp:
        work_dir = Path(tmp)
        chat_path = _download_chat(canonical_url, work_dir)
        offsets = _parse_offsets(chat_path)

    if len(offsets) < 50:
        raise ChatDataNotAvailable(
            f"Only {len(offsets)} chat message(s) found - too few to find peaks."
        )

    corrected = [max(0.0, offset - REACTION_LAG_S) for offset in offsets]
    points = activity.timestamps_to_points(corrected, duration_s)
    if not points:
        raise ChatDataNotAvailable("Chat activity curve came out empty.")

    return points
