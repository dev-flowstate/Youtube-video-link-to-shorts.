"""Make YouTubeReplayDownloader's modules importable from here.

This project needs the sibling's URL parsing, audio fetching, peak detection
and clip cutting. Only the *signal* is new, so duplicating roughly a thousand
lines to keep the folders separate would be worse than one documented path
entry.

The path is **appended**, not inserted, so a module defined here always wins on
a name collision. Both projects define main.py, which is why nothing may
`import main` across the boundary.
"""

from __future__ import annotations

import sys
from pathlib import Path

SIBLING = Path(__file__).resolve().parent.parent / "YouTubeReplayDownloader"


class SiblingMissing(RuntimeError):
    """Raised when the downloader project cannot be found."""


def bootstrap() -> Path:
    """Put the sibling project on sys.path and return where it was found."""
    if not (SIBLING / "peak_detector.py").exists():
        raise SiblingMissing(
            f"Could not find YouTubeReplayDownloader next to this project.\n"
            f"  Looked in: {SIBLING}\n"
            f"  EsportsClipper reuses its downloader and peak detection, so the "
            f"two folders must sit side by side."
        )

    path = str(SIBLING)
    if path not in sys.path:
        sys.path.append(path)

    return SIBLING


bootstrap()
