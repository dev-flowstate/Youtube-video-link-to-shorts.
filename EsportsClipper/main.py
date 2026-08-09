"""Cut the fights out of an esports broadcast.

Edit STREAM_URL in config.py, then run:
    py main.py

Produces one clip per fight, plus a compilation of all of them. Feed the clips
to ClipCaptioner to turn them into Shorts.
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

# Must come first: it puts YouTubeReplayDownloader on sys.path, and every
# import below it that is not a local module lives there. Do not let an import
# sorter move this - alphabetical order would break the run.
import shared  # noqa: F401  isort:skip

import audio_features
import clip_windows
import compilation
import config
import downloader
import fight_detector
import speech
import vision_detector
from downloader import DownloadError, download_all_segments, fetch_video_title
from ffmpeg_utils import FFmpegNotFoundError
from moment_finder import fetch_video_facts
from peak_detector import detect_replay_segments
from utils import InvalidYouTubeURL, format_timestamp, parse_youtube_url


def _apply_quality_cap() -> None:
    """Hold the source download to a sane size for a five hour broadcast.

    Set on the sibling at runtime rather than editing it, so the podcast
    pipeline keeps its own full-quality default.
    """
    height = config.SOURCE_MAX_HEIGHT
    downloader.SOURCE_FORMAT = (
        f"bestvideo[height<={height}]+bestaudio/best[height<={height}]/best"
    )


def _mark_output_as_gameplay(output_dir: Path) -> None:
    """Leave a note saying what these clips are.

    ClipCaptioner reads this and changes two things.

    Face tracking goes off: gameplay has no speaker to follow, so the detector
    locks onto a webcam corner, a crowd shot or a logo and pans the crop around
    chasing it.

    Cropping becomes "fit": a match feed carries the minimap, the kill feed and
    often the fight itself out at the edges, and cutting a tall slice out of the
    middle throws all of it away. Fitting the whole frame keeps everything on
    screen.
    """
    marker = output_dir / "content.json"
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        marker.write_text(
            json.dumps(
                {"content_type": "gameplay", "face_tracking": False, "crop_mode": "fit"},
                indent=2,
            ),
            encoding="utf-8",
        )
    except OSError as exc:
        # Only a hint for the next stage; never lose clips over it.
        print(f"Could not write {marker.name}: {exc}")


def _find_fights_by_watching(source_video: Path) -> list:
    """Look at the broadcast and keep the moments that show combat."""
    labelled = vision_detector.detect(source_video)

    seen: dict[str, int] = {}
    for item in labelled:
        seen[item.label] = seen.get(item.label, 0) + 1
    summary = ", ".join(f"{count} {label.lower()}" for label, count in sorted(seen.items()))
    print(f"    saw {summary}")

    spans = vision_detector.fight_windows(labelled)
    return clip_windows.windows_from_vision(spans, config.VISION_SAMPLE_SECONDS)


def _find_fights_by_listening(canonical_url: str, is_live: bool, duration_s: float) -> list:
    """Fall back to the audio detector.

    Kept for when there is no API key, but it is much the weaker of the two: on
    a real broadcast the game is mixed so far under the casters that studio
    segments scored higher on gunfire than the actual fights did.
    """
    print("Fetching audio and looking for fights...")
    with tempfile.TemporaryDirectory(prefix="esports_") as tmp:
        audio_path = speech.download_audio(canonical_url, Path(tmp), is_live)
        features = audio_features.extract(audio_path)

    measured = features.duration_s
    if measured > 0:
        duration_s = measured
    print(f"Analysed {duration_s / 60:.0f} min of audio.")

    curves = fight_detector.build_curves(features, duration_s)
    active = int((curves.gunfire >= config.MIN_GUNFIRE_ONSETS).sum())
    print(f"Gunfire in {active} of {len(curves.gunfire)} buckets.")

    # Peaks only - the spans it returns are centred, which is wrong here, so
    # clip_windows rebuilds every one of them around the gunfire.
    min_gap = clip_windows._buckets(config.MAX_CLIP_SECONDS, curves.width_s)
    peaks = detect_replay_segments(
        fight_detector.to_points(curves),
        min_distance_buckets=min_gap,
        min_segment_seconds=config.BUCKET_SECONDS,
        max_segment_seconds=config.MAX_CLIP_SECONDS,
    )
    if not peaks:
        return []

    return clip_windows.windows_from_segments(curves, peaks)


def _print_fights(segments) -> None:
    print(f"\nFound {len(segments)} fight(s):\n")
    for index, segment in enumerate(segments, start=1):
        print(
            f"  {index:2d}. "
            f"{format_timestamp(segment.start_s)} -> {format_timestamp(segment.end_s)} "
            f"({segment.duration_s:.0f}s, score {segment.peak_score:.2f})"
        )
    print()


def main() -> int:
    output_dir = config.OUTPUT_DIR

    print("Esports Fight Clipper")
    print("=" * 21)
    print(f"URL: {config.STREAM_URL}")
    print(f"Output folder: {output_dir}\n")

    try:
        canonical_url = parse_youtube_url(config.STREAM_URL)
        _apply_quality_cap()

        duration_s, is_live = fetch_video_facts(canonical_url)
        if is_live:
            print(
                "Stream is still live - clipping the part broadcast so far.\n"
                "  This fetches from the beginning of the broadcast, so a long\n"
                "  running stream takes a while before anything is analysed."
            )

        title = fetch_video_title(canonical_url)
        print(f"Broadcast: {title}\n")

        watching = config.USE_VISION and vision_detector.available()
        if watching:
            # Before the download, not after: the source is the expensive part
            # and there is no point fetching it to find the key is spent.
            vision_detector.check_ready()

            # Watching needs the file before it can choose anything, so the
            # download comes first. It is cached, so cutting reuses it rather
            # than fetching the broadcast twice.
            print("Downloading the source once...")
            source_video = downloader.fetch_source_video(canonical_url, output_dir)
            print(f"Source ready: {source_video.name}\n")

            print(f"Detector: watching the video ({config.VISION_MODEL})")
            segments = _find_fights_by_watching(source_video)
        else:
            if config.USE_VISION and not config.ALLOW_AUDIO_FALLBACK:
                # A missing key must stop the run rather than quietly falling
                # through to a detector that is measured to prefer studio talk
                # over real fights (see config.ALLOW_AUDIO_FALLBACK).
                print(
                    "Watching is switched on but unavailable "
                    f"({vision_detector.ENV_KEY} unset, or google-genai missing).\n"
                    "\n"
                    "  Most likely cause: the key is saved in the Windows environment,\n"
                    "  but a terminal inherits its environment from when its parent\n"
                    "  window opened - a VS Code window opened before the key was\n"
                    "  saved cannot see it. Restart VS Code and try again.\n"
                    "\n"
                    "  If the key is set, check that google-genai is installed.\n"
                    "\n"
                    "  The audio detector is a measured negative result, not a\n"
                    "  fallback, so it stays off by default. Set\n"
                    "  ALLOW_AUDIO_FALLBACK = True in config.py to run it anyway.\n"
                )
                return 1
            if config.USE_VISION:
                print(
                    "Watching is switched on but unavailable "
                    f"({vision_detector.ENV_KEY} unset, or google-genai missing).\n"
                    "  Falling back to audio anyway (ALLOW_AUDIO_FALLBACK is on): this\n"
                    "  detector is known to pick studio talk over real fights.\n"
                )
            print("Detector: listening to audio (known-bad, ALLOW_AUDIO_FALLBACK is on)")
            segments = _find_fights_by_listening(canonical_url, is_live, duration_s)

        if not segments:
            print("No fights were detected in this broadcast.")
            return 0

        budget = clip_windows.clip_budget(duration_s)
        if len(segments) > budget:
            print(f"Keeping the {budget} best of {len(segments)} fights.")
        segments = clip_windows.pick_best(segments, budget)

        _print_fights(segments)
        print("Cutting every fight from the source.")

        saved = download_all_segments(
            youtube_url=canonical_url,
            title=title,
            segments=segments,
            output_dir=output_dir,
        )

        print(f"\n{len(saved)} clip(s) written.")
        _mark_output_as_gameplay(output_dir)

        if config.MAKE_COMPILATION and len(saved) >= 2:
            try:
                joined = compilation.build(saved, output_dir, title)
                print(f"Compilation: {joined.name}")
            except compilation.CompilationFailed as exc:
                print(f"Compilation skipped: {exc}")

        print("\nNext, for Shorts:")
        print(f'  cd ../ClipCaptioner && py main.py --input "{output_dir}"')
        return 0

    except InvalidYouTubeURL as exc:
        print(f"Invalid URL: {exc}")
        return 1
    except FFmpegNotFoundError as exc:
        print(str(exc))
        return 1
    except audio_features.FeatureExtractionFailed as exc:
        print(f"Could not analyse the audio: {exc}")
        return 1
    except vision_detector.VisionUnavailable as exc:
        print(f"Could not watch the broadcast: {exc}")
        return 1
    except DownloadError as exc:
        print(f"Download failed: {exc}")
        return 1
    except KeyboardInterrupt:
        print("\nInterrupted.")
        return 130
    finally:
        # The source is downloaded before detection now, so every early return
        # above would otherwise leave a multi-gigabyte file in the output
        # folder. Cutting already cleans up after itself, so this is a no-op on
        # the path that reaches it.
        downloader.cleanup_cached_videos()


if __name__ == "__main__":
    sys.exit(main())
