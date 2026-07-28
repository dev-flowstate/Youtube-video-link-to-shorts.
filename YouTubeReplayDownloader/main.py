"""
YouTube Most-Replayed clip downloader.

Edit YOUTUBE_URL below, then run:
    python main.py
"""

from __future__ import annotations

import sys
from pathlib import Path

from downloader import DownloadError, download_all_segments, fetch_video_title
from ffmpeg_utils import FFmpegNotFoundError
from peak_detector import ReplaySegment, detect_replay_segments
from replay_fetcher import ReplayDataNotAvailable, fetch_replay_data
from utils import InvalidYouTubeURL, format_timestamp, parse_youtube_url


# ---------------------------------------------------------------------------
# Edit this URL before running
# ---------------------------------------------------------------------------
YOUTUBE_URL = "https://youtu.be/VT3CtCEh8HQ?si=9KYwYy006bc-SCgt"

# Where the finished clips are written. Created automatically if missing.
# Defaults to an "output" folder next to this script so the project works
# anywhere. Replace with an absolute path to send clips elsewhere, e.g.
#     OUTPUT_DIR = Path(r"E:\Youtube Videos\Videos")
OUTPUT_DIR = Path(__file__).resolve().parent / "output"


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

    print("YouTube Most-Replayed Downloader")
    print("=" * 34)
    print(f"URL: {YOUTUBE_URL}")
    print(f"Output folder: {output_dir}\n")

    try:
        canonical_url = parse_youtube_url(YOUTUBE_URL)
        replay_points = fetch_replay_data(canonical_url)
        segments = detect_replay_segments(replay_points)

        if not segments:
            print("No significant replay peaks were detected for this video.")
            return 0

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
    except ReplayDataNotAvailable as exc:
        print(f"No replay data available: {exc}")
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
