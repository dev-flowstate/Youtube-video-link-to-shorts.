"""Make YouTubeReplayDownloader's modules importable from here.

This project needs the sibling's URL parsing, audio fetching, peak detection
and clip cutting. Only the *signal* is new, so duplicating roughly a thousand
lines to keep the folders separate would be worse than one documented path
entry.

The path goes in at position 1: after this project's own directory, which is
sys.path[0], but ahead of site-packages.

Both halves of that matter. Behind the local directory, so a module defined
here still wins a name collision - both projects define config.py and main.py.
Ahead of site-packages, because the sibling has modules with everyday names
and PyPI has packages with the same ones: appending put the sibling last, and
an installed distribution called `utils` silently shadowed the project's
utils.py, breaking every import beneath it.
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
    if path in sys.path:
        sys.path.remove(path)
    sys.path.insert(1, path)

    return SIBLING


bootstrap()
