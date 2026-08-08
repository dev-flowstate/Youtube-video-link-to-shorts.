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

    ClipCaptioner reads this and turns face tracking off. Gameplay has no
    speaker to follow - the detector would lock onto a webcam corner, a crowd
    shot or a logo and pan the crop around chasing it, which is worse than
    holding the centre where the action is.
    """
    marker = output_dir / "content.json"
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        marker.write_text(
            json.dumps({"content_type": "gameplay", "face_tracking": False}, indent=2),
            encoding="utf-8",
        )
    except OSError as exc:
        # Only a hint for the next stage; never lose clips over it.
        print(f"Could not write {marker.name}: {exc}")


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

        # Peaks only - the spans it returns are centred, which is wrong here,
        # so clip_windows rebuilds every one of them around the gunfire.
        min_gap = clip_windows._buckets(config.MAX_CLIP_SECONDS, curves.width_s)
        peaks = detect_replay_segments(
            fight_detector.to_points(curves),
            min_distance_buckets=min_gap,
            min_segment_seconds=config.BUCKET_SECONDS,
            max_segment_seconds=config.MAX_CLIP_SECONDS,
        )

        if not peaks:
            print("No fights were detected in this broadcast.")
            return 0

        segments = clip_windows.windows_from_segments(curves, peaks)
        if not segments:
            print("Every candidate was too brief to be a fight.")
            return 0

        budget = clip_windows.clip_budget(duration_s)
        if len(segments) > budget:
            print(f"Keeping the {budget} best of {len(segments)} fights.")
        segments = clip_windows.pick_best(segments, budget)

        _print_fights(segments)

        title = fetch_video_title(canonical_url)
        print(f"Broadcast: {title}\n")
        print("Downloading the source once, then cutting every fight locally.")

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
    except DownloadError as exc:
        print(f"Download failed: {exc}")
        return 1
    except KeyboardInterrupt:
        print("\nInterrupted.")
        return 130


if __name__ == "__main__":
    sys.exit(main())
