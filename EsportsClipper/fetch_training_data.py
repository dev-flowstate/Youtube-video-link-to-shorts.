"""Fetch the footage the fight detector trains on.

Two kinds, and they do different jobs.

Highlight reels are the positive class. Almost every second of one is a fight,
so they hand over thousands of labelled examples without an API call or a
hand-marked frame. What they cannot give is a negative: a reel contains no
studio talk and no quiet gameplay, because those are exactly what the editor
removed.

Full broadcasts supply that other half, and a held-out one supplies the only
honest accuracy number.

Run:
    py fetch_training_data.py
Everything already on disk is skipped, so a re-run after an interruption costs
nothing.
"""

from __future__ import annotations

import sys
from pathlib import Path

import yt_dlp

import config
import shared  # noqa: F401  (puts the sibling project on sys.path)
from ffmpeg_utils import ffmpeg_location_option

REELS_DIR = config.DATA_DIR / "reels"
BROADCASTS_DIR = config.DATA_DIR / "broadcasts"


class FetchFailed(Exception):
    """Raised when nothing at all could be downloaded."""


def _existing(destination: Path) -> Path | None:
    """The already-downloaded file for this id, whatever extension it took."""
    for path in destination.parent.glob(f"{destination.stem}.*"):
        if path.is_file() and path.stat().st_size > 0 and path.suffix != ".part":
            return path
    return None


def fetch(video_id: str, destination_dir: Path, max_height: int) -> Path:
    """Download one video, named by its id.

    Named by id rather than title on purpose. Two of these titles contain
    characters this console cannot encode, and the labels are keyed on the id
    anyway - a title would be a second name for the same thing and a chance for
    the two to disagree.
    """
    destination_dir.mkdir(parents=True, exist_ok=True)
    target = destination_dir / f"{video_id}.mp4"

    already = _existing(target)
    if already is not None:
        size_gb = already.stat().st_size / 1e9
        print(f"  {video_id}  already here ({size_gb:.2f} GB) - skipped")
        return already

    options = {
        "format": (
            f"bestvideo[height<={max_height}]+bestaudio/"
            f"best[height<={max_height}]/best"
        ),
        "merge_output_format": "mp4",
        # Both the finished file and yt-dlp's part files land here. The system
        # temp drive has little room free and a five hour broadcast would fill
        # it, so nothing is allowed to default there.
        "outtmpl": str(destination_dir / f"{video_id}.%(ext)s"),
        "paths": {"home": str(destination_dir), "temp": str(destination_dir)},
        "retries": 5,
        "fragment_retries": 5,
        "quiet": True,
        "no_warnings": True,
        "noprogress": False,
        "live_from_start": True,
    }
    options.update(ffmpeg_location_option())

    try:
        with yt_dlp.YoutubeDL(options) as ydl:  # type: ignore[arg-type]  # yt-dlp's stubs declare a TypedDict; a plain options dict is what the library actually takes
            ydl.download([f"https://www.youtube.com/watch?v={video_id}"])
    except Exception as exc:
        raise FetchFailed(f"{video_id}: {exc}") from exc

    produced = _existing(target)
    if produced is None:
        raise FetchFailed(f"{video_id}: download produced no file")
    return produced


def fetch_all(include_broadcast_b: bool = False) -> list[Path]:
    """Fetch every training source, carrying on past any one failure."""
    wanted: list[tuple[str, Path, int]] = [
        (vid, REELS_DIR, config.REEL_MAX_HEIGHT) for vid in config.TRAINING_REELS
    ]
    wanted.append(
        (config.TRAINING_BROADCAST_A, BROADCASTS_DIR, config.SOURCE_MAX_HEIGHT)
    )
    if include_broadcast_b:
        wanted.append(
            (config.TRAINING_BROADCAST_B, BROADCASTS_DIR, config.SOURCE_MAX_HEIGHT)
        )

    got: list[Path] = []
    failed: list[str] = []

    for index, (video_id, folder, height) in enumerate(wanted, start=1):
        print(f"[{index}/{len(wanted)}] {video_id} -> {folder.name}/ ({height}p cap)")
        try:
            got.append(fetch(video_id, folder, height))
        except FetchFailed as exc:
            # One bad link must not cost the rest. A five hour broadcast is an
            # hour of downloading; losing the reels because it failed would be
            # the worst possible trade.
            print(f"  failed: {exc}")
            failed.append(str(exc))

    print(f"\n{len(got)} file(s) ready, {len(failed)} failed.")
    for note in failed:
        print(f"  - {note}")

    if not got:
        raise FetchFailed("Nothing downloaded.")
    return got


def main() -> int:
    print("Training data")
    print("=" * 13)
    print(f"Into: {config.DATA_DIR}\n")

    try:
        got = fetch_all(include_broadcast_b="--with-b" in sys.argv)
    except FetchFailed as exc:
        print(f"Failed: {exc}")
        return 1

    print("\nOn disk:")
    for path in sorted(got):
        print(f"  {path.stat().st_size / 1e9:6.2f} GB  {path.parent.name}/{path.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
