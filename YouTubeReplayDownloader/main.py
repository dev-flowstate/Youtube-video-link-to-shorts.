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
YOUTUBE_URL = "https://youtu.be/lwzADAdjqvE?si=P15lzh3SobT8P2ur"

# Where the finished clips are written. Created automatically if missing.
# Defaults to an "output" folder next to this script so the project works
# anywhere. Replace with an absolute path to send clips elsewhere, e.g.
#     OUTPUT_DIR = Path(r"E:\Youtube Videos\Videos")
OUTPUT_DIR = Path(__file__).resolve().parent / "output"

# How many clips to keep from a single video. Only the strongest peaks
# survive, so this is "the N hottest moments". Set to None for no limit.
MAX_CLIPS: int | None = 5

# Longest a clip may run, in seconds. Clips are trimmed around their peak, so
# the best moment stays in frame. Short clips hold attention; 90s is about the
# ceiling before viewers drop off.
MAX_CLIP_SECONDS = 90.0


def _pick_hottest(segments: list[ReplaySegment], limit: int) -> list[ReplaySegment]:
    """Take the strongest peaks, skipping any that repeat an earlier pick.

    Two adjacent peaks can survive detection with windows that overlap. With
    only a handful of slots, any shared footage wastes part of the output, so
    candidates must not touch a pick already made. Returning four distinct
    clips beats five that partly repeat each other.
    """
    chosen: list[ReplaySegment] = []

    for candidate in sorted(segments, key=lambda s: -s.peak_score):
        if len(chosen) >= limit:
            break

        overlaps = any(
            min(candidate.end_s, picked.end_s) > max(candidate.start_s, picked.start_s)
            for picked in chosen
        )
        if not overlaps:
            chosen.append(candidate)

    return sorted(chosen, key=lambda s: s.start_s)


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

        segments = detect_replay_segments(
            moments.points,
            max_segment_seconds=MAX_CLIP_SECONDS,
        )

        if not segments:
            print(f"No significant peaks were detected using {moments.source}.")
            return 0

        if MAX_CLIPS is not None and len(segments) > MAX_CLIPS:
            print(f"Keeping the {MAX_CLIPS} hottest of {len(segments)} peaks.")
            segments = _pick_hottest(segments, MAX_CLIPS)

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
