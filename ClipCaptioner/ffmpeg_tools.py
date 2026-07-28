"""FFmpeg discovery and command execution.

Deliberately standalone rather than importing YouTubeReplayDownloader's
ffmpeg_utils, so the two packages stay independently runnable.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path

from models import VideoInfo


class FFmpegNotFoundError(Exception):
    """Raised when FFmpeg or FFprobe cannot be located."""


class FFmpegError(Exception):
    """Raised when an FFmpeg invocation fails."""


_TOOL_DIR: Path | None = None


def _candidate_dirs() -> list[Path]:
    candidates: list[Path] = []

    local_app_data = os.environ.get("LOCALAPPDATA")
    if local_app_data:
        packages = Path(local_app_data) / "Microsoft" / "WinGet" / "Packages"
        if packages.exists():
            for match in packages.glob("Gyan.FFmpeg_*"):
                candidates.extend(
                    sub for sub in match.glob("**/bin") if (sub / "ffmpeg.exe").exists()
                )

    candidates.append(Path(r"C:\ffmpeg\bin"))
    candidates.append(Path(r"C:\Program Files\ffmpeg\bin"))
    return candidates


def resolve_tool_dir() -> Path:
    """Return the directory holding ffmpeg and ffprobe."""
    global _TOOL_DIR
    if _TOOL_DIR is not None:
        return _TOOL_DIR

    on_path = shutil.which("ffmpeg")
    if on_path:
        _TOOL_DIR = Path(on_path).parent
        return _TOOL_DIR

    for directory in _candidate_dirs():
        if (directory / "ffmpeg.exe").exists():
            _TOOL_DIR = directory
            return _TOOL_DIR

    try:
        import imageio_ffmpeg
    except ImportError:
        pass
    else:
        exe = Path(imageio_ffmpeg.get_ffmpeg_exe())
        if exe.exists():
            _TOOL_DIR = exe.parent
            return _TOOL_DIR

    raise FFmpegNotFoundError(
        "FFmpeg was not found. Install it (winget install Gyan.FFmpeg) "
        "or run: pip install imageio-ffmpeg"
    )


def tool_path(name: str) -> str:
    """Absolute path to ffmpeg or ffprobe."""
    directory = resolve_tool_dir()
    suffix = ".exe" if os.name == "nt" else ""
    candidate = directory / f"{name}{suffix}"
    if candidate.exists():
        return str(candidate)

    fallback = shutil.which(name)
    if fallback:
        return fallback

    raise FFmpegNotFoundError(f"{name} not found alongside {directory}")


def run(args: list[str], *, cwd: Path | None = None) -> None:
    """Run an FFmpeg command, raising with its stderr tail on failure."""
    result = subprocess.run(
        args,
        capture_output=True,
        text=True,
        check=False,
        cwd=str(cwd) if cwd else None,
    )
    if result.returncode != 0:
        stderr = (result.stderr or "").strip()
        raise FFmpegError(f"FFmpeg failed ({result.returncode}): {stderr[-800:]}")


def probe_video(path: Path) -> VideoInfo:
    """Read width, height and duration from a video file."""
    args = [
        tool_path("ffprobe"),
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=width,height:format=duration",
        "-of",
        "json",
        str(path),
    ]
    result = subprocess.run(args, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        raise FFmpegError(f"Could not probe {path.name}: {result.stderr.strip()[-400:]}")

    payload = json.loads(result.stdout or "{}")
    streams = payload.get("streams") or []
    if not streams:
        raise FFmpegError(f"No video stream found in {path.name}")

    try:
        duration = float((payload.get("format") or {}).get("duration", 0.0))
    except (TypeError, ValueError):
        duration = 0.0

    return VideoInfo(
        width=int(streams[0]["width"]),
        height=int(streams[0]["height"]),
        duration_s=duration,
    )
