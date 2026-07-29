"""
YouTube highlight clip downloader.

Works on normal videos and on past live broadcasts. Edit YOUTUBE_URL below,
then run:
    python main.py
"""

from __future__ import annotations

import sys
from pathlib import Path

from downloader import DownloadError, download_all_segments, fetch_video_title
from ffmpeg_utils import FFmpegNotFoundError
from moment_finder import NoMomentSignal, find_moments
from peak_detector import ReplaySegment, detect_replay_segments
from utils import InvalidYouTubeURL, format_timestamp, parse_youtube_url


# ---------------------------------------------------------------------------
# Edit this URL before running
# ---------------------------------------------------------------------------
YOUTUBE_URL = "https://youtu.be/Rni7Fz7208c?si=RfD5qbBfVaoakq8A"

# Where the finished clips are written. Created automatically if missing.
# Defaults to an "output" folder next to this script so the project works
# anywhere. Replace with an absolute path to send clips elsewhere, e.g.
#     OUTPUT_DIR = Path(r"E:\Youtube Videos\Videos")
OUTPUT_DIR = Path(__file__).resolve().parent / "output"

# Upper bound on clips from a single video. A long stream can yield dozens of
# peaks, and every one costs a download plus a captioning pass, so only the
# strongest are kept. Set to None for no limit.
MAX_CLIPS: int | None = 15


def _print_segments(segments: list[ReplaySegment]) -> None:
    print(f"\nFound {len(segments)} replay segment(s):\n")
    for index, segment in enumerate(segments, start=1):
        print(
            f"  {index}. "
            f"{format_timestamp(segment.start_s)} -> {format_timestamp(segment.end_s)} "
            f"(peak {format_timestamp(segment.peak_s)}, "
            f"score {segment.peak_score:.3f})"
        )
    print()


def main() -> int:
    output_dir = OUTPUT_DIR

    print("YouTube Highlight Downloader")
    print("=" * 28)
    print(f"URL: {YOUTUBE_URL}")
    print(f"Output folder: {output_dir}\n")

    try:
        canonical_url = parse_youtube_url(YOUTUBE_URL)

        print("Looking for clip-worthy moments...")
        moments = find_moments(canonical_url)
        print(f"Signal: {moments.source} ({len(moments.points)} data points)")

        segments = detect_replay_segments(moments.points)

        if not segments:
            print(f"No significant peaks were detected using {moments.source}.")
            return 0

        if MAX_CLIPS is not None and len(segments) > MAX_CLIPS:
            print(f"Keeping the {MAX_CLIPS} strongest of {len(segments)} peaks.")
            strongest = sorted(segments, key=lambda s: -s.peak_score)[:MAX_CLIPS]
            segments = sorted(strongest, key=lambda s: s.start_s)

        _print_segments(segments)

        title = fetch_video_title(canonical_url)
        print(f"Video title: {title}\n")

        saved_files = download_all_segments(
            youtube_url=canonical_url,
            title=title,
            segments=segments,
            output_dir=output_dir,
        )

        print("Download complete:\n")
        for path in saved_files:
            print(f"  - {path.name}")

        return 0

    except InvalidYouTubeURL as exc:
        print(f"Invalid URL: {exc}")
        return 1
    except NoMomentSignal as exc:
        print(f"Could not find moments: {exc}")
        print("Nothing was downloaded.")
        return 0
    except FFmpegNotFoundError as exc:
        print(str(exc))
        return 1
    except DownloadError as exc:
        print(f"Download failed: {exc}")
        return 1
    except Exception as exc:
        print(f"Unexpected error: {exc}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
