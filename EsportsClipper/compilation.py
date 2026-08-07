"""Join every fight clip into one long video.

Each clip was cut from the same source with identical codec parameters, so the
concat demuxer can stitch them with a stream copy - seconds of work rather than
a full re-encode. The dead air between fights is never included, which is the
whole point of a compilation.
"""

from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path

import shared  # noqa: F401  (puts the sibling project on sys.path)
from ffmpeg_utils import ensure_ffmpeg_available
from utils import sanitize_filename


class CompilationFailed(Exception):
    """Raised when the clips could not be joined."""


def _escape(path: Path) -> str:
    """Quote a path for the concat demuxer's list file."""
    # The demuxer reads single-quoted paths and needs its own quotes escaped.
    return str(path.resolve()).replace("'", r"'\''")


def build(clips: list[Path], output_dir: Path, title: str) -> Path:
    """Concatenate clips in the order given and return the finished file."""
    usable = [clip for clip in clips if clip.exists() and clip.stat().st_size > 0]
    if len(usable) < 2:
        raise CompilationFailed("Need at least two clips to build a compilation.")

    ensure_ffmpeg_available()
    output_path = output_dir / f"{sanitize_filename(title)} - all fights.mp4"

    with tempfile.TemporaryDirectory(prefix="compilation_") as tmp:
        listing = Path(tmp) / "clips.txt"
        listing.write_text(
            "".join(f"file '{_escape(clip)}'\n" for clip in usable),
            encoding="utf-8",
        )

        args = [
            "ffmpeg",
            "-y",
            "-v",
            "error",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            str(listing),
            "-c",
            "copy",
            "-movflags",
            "+faststart",
            str(output_path),
        ]
        result = subprocess.run(args, capture_output=True, text=True, check=False)

    if result.returncode != 0 or not output_path.exists():
        raise CompilationFailed(f"FFmpeg concat failed: {result.stderr.strip()[-400:]}")

    return output_path
