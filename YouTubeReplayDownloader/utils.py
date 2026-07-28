"""Shared helpers for URL validation, timestamps, and filenames."""

from __future__ import annotations

import re
from pathlib import Path
from urllib.parse import parse_qs, urlparse


YOUTUBE_HOSTS = {
    "youtube.com",
    "www.youtube.com",
    "m.youtube.com",
    "music.youtube.com",
    "youtu.be",
}


class InvalidYouTubeURL(Exception):
    """Raised when the URL is not a supported normal YouTube video link."""


def parse_youtube_url(url: str) -> str:
    """
    Validate and normalize a YouTube watch URL.

    Rejects Shorts, playlists, live-only paths, and non-YouTube links.
    Returns the canonical watch URL.
    """
    url = url.strip()
    if not url:
        raise InvalidYouTubeURL("URL is empty.")

    parsed = urlparse(url)
    host = parsed.netloc.lower().removeprefix("www.")

    if host not in YOUTUBE_HOSTS and not host.endswith(".youtube.com"):
        raise InvalidYouTubeURL(f"Not a YouTube URL: {url}")

    path = parsed.path.rstrip("/")

    if "/shorts/" in path.lower():
        raise InvalidYouTubeURL("Shorts URLs are not supported.")

    video_id: str | None = None

    if host == "youtu.be":
        video_id = path.lstrip("/").split("/")[0] or None
    elif path == "/watch":
        query = parse_qs(parsed.query)
        ids = query.get("v", [])
        video_id = ids[0] if ids else None
    elif path.startswith("/embed/"):
        video_id = path.split("/embed/")[1].split("/")[0]
    else:
        raise InvalidYouTubeURL(
            "Use a normal watch URL like https://www.youtube.com/watch?v=VIDEO_ID"
        )

    if not video_id or not re.fullmatch(r"[\w-]{11}", video_id):
        raise InvalidYouTubeURL(f"Could not extract a valid video ID from: {url}")

    return f"https://www.youtube.com/watch?v={video_id}"


def ms_to_seconds(ms: int | float) -> float:
    return float(ms) / 1000.0


def format_timestamp(seconds: float) -> str:
    """Format seconds as 03m42s."""
    total = max(0, int(round(seconds)))
    minutes, secs = divmod(total, 60)
    hours, minutes = divmod(minutes, 60)

    if hours:
        return f"{hours:02d}h{minutes:02d}m{secs:02d}s"
    return f"{minutes:02d}m{secs:02d}s"


def format_section_range(start_s: float, end_s: float) -> str:
    """Format a yt-dlp download-sections range like *3:42-4:12."""
    return f"*{seconds_to_hms(start_s)}-{seconds_to_hms(end_s)}"


def seconds_to_hms(seconds: float) -> str:
    total = max(0, int(round(seconds)))
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)

    if hours:
        return f"{hours}:{minutes:02d}:{secs:02d}"
    return f"{minutes}:{secs:02d}"


_INVALID_FILENAME_CHARS = re.compile(r'[<>:"/\\|?*\x00-\x1f]')


def sanitize_filename(name: str, max_length: int = 180) -> str:
    cleaned = _INVALID_FILENAME_CHARS.sub("", name)
    cleaned = re.sub(r"\s+", " ", cleaned).strip(" .")
    if not cleaned:
        cleaned = "video"
    if len(cleaned) > max_length:
        cleaned = cleaned[:max_length].rstrip(" .")
    return cleaned


def build_output_path(
    output_dir: Path,
    title: str,
    start_s: float,
    end_s: float,
) -> Path:
    safe_title = sanitize_filename(title)
    start_label = format_timestamp(start_s)
    end_label = format_timestamp(end_s)
    filename = f"{safe_title} [{start_label}-{end_label}].mp4"
    return output_dir / filename
