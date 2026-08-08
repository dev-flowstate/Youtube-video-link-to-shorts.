"""Download replay segments with yt-dlp and FFmpeg."""

from __future__ import annotations

import subprocess
import uuid
from pathlib import Path

import yt_dlp
from tqdm import tqdm

from ffmpeg_utils import ensure_ffmpeg_available, ffmpeg_location_option, run_ffprobe
from peak_detector import ReplaySegment
from utils import build_output_path, parse_youtube_url


class DownloadError(Exception):
    """Raised when a segment download fails."""


# Full source quality.
#
# Downloading only the replay ranges was tried and abandoned: yt-dlp delegates
# partial downloads to FFmpeg's own HTTP client, which stalls indefinitely
# against YouTube (a 15s 720p range wrote zero bytes in seven minutes) while
# yt-dlp's native downloader sustains normal speed. So fetch the source once
# with the native downloader, then cut every clip locally.
SOURCE_FORMAT = "bestvideo+bestaudio/best"

# A clip running past its requested end is normal - a stream copy can only cut
# on a keyframe. This is not that: it is the ceiling above which the cut
# clearly never happened at all, and the file is a large piece of the source
# rather than a clip. Set far above any plausible keyframe interval so ordinary
# drift is never mistaken for it.
RUNAWAY_CUT_S = 300.0

_VIDEO_CACHE: dict[str, Path] = {}


def fetch_video_title(youtube_url: str) -> str:
    canonical_url = parse_youtube_url(youtube_url)
    ydl_opts = {
        "quiet": True,
        "no_warnings": True,
        "skip_download": True,
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:  # type: ignore[arg-type]  # yt-dlp's stubs declare a TypedDict; a plain options dict is what the library actually takes
        info = ydl.extract_info(canonical_url, download=False)

    title = info.get("title")
    if not title:
        raise DownloadError("Could not read video title from YouTube.")
    return str(title)


def _download_full_video(youtube_url: str, work_dir: Path) -> Path:
    """Download the full source video once and reuse it for all segment cuts."""
    cache_key = parse_youtube_url(youtube_url)
    cached = _VIDEO_CACHE.get(cache_key)
    if cached and cached.exists() and cached.stat().st_size > 0:
        return cached

    ensure_ffmpeg_available()
    temp_base = work_dir / f"_source_{uuid.uuid4().hex}"

    options = {
        "format": SOURCE_FORMAT,
        "merge_output_format": "mp4",
        "outtmpl": str(temp_base.with_suffix(".%(ext)s")),
        "retries": 5,
        "fragment_retries": 5,
        "quiet": True,
        "no_warnings": True,
        "noprogress": True,
        # Pulls a running broadcast from its beginning instead of the live
        # edge. Ignored for anything that is not live, so it is always on.
        "live_from_start": True,
    }
    options.update(ffmpeg_location_option())

    try:
        with yt_dlp.YoutubeDL(options) as ydl:  # type: ignore[arg-type]  # yt-dlp's stubs declare a TypedDict; a plain options dict is what the library actually takes
            ydl.download([cache_key])
    except Exception as exc:
        raise DownloadError(f"Failed to download source video: {exc}") from exc

    produced = None
    for path in work_dir.iterdir():
        if not path.is_file():
            continue
        if path.suffix.lower() != ".mp4":
            continue
        if path.stem == temp_base.stem:
            produced = path
            break

    if produced is None or not produced.exists():
        raise DownloadError("Source video download did not produce an MP4 file.")

    _VIDEO_CACHE[cache_key] = produced
    return produced


def _ffmpeg_cut(source_video: Path, segment: ReplaySegment, output_path: Path) -> None:
    ensure_ffmpeg_available()

    if output_path.exists():
        output_path.unlink()

    duration = max(0.1, segment.end_s - segment.start_s)
    cmd = [
        "ffmpeg",
        "-y",
        "-ss",
        f"{segment.start_s:.3f}",
        "-i",
        str(source_video),
        "-t",
        f"{duration:.3f}",
        "-map",
        "0",
        "-c",
        "copy",
        "-avoid_negative_ts",
        "make_zero",
        "-movflags",
        "+faststart",
        str(output_path),
    ]

    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        stderr = (result.stderr or "").strip()
        raise DownloadError(f"FFmpeg cut failed: {stderr[-500:]}")


def download_segment(
    youtube_url: str,
    segment: ReplaySegment,
    output_path: Path,
    source_video: Path,
) -> Path:
    """Cut one replay segment from a cached full-quality source video."""
    if output_path.exists() and output_path.stat().st_size > 0:
        return output_path

    _ffmpeg_cut(source_video, segment, output_path)
    return output_path


def _verify_clip(output_path: Path, segment: ReplaySegment) -> None:
    if not output_path.exists() or output_path.stat().st_size == 0:
        raise DownloadError(f"Download produced an empty file: {output_path.name}")

    try:
        run_ffprobe(output_path)
    except RuntimeError as exc:
        raise DownloadError(str(exc)) from exc

    ensure_ffmpeg_available()
    ffprobe = "ffprobe"
    cmd = [
        ffprobe,
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(output_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        raise DownloadError(f"Could not read duration for {output_path.name}.")

    try:
        actual_duration = float(result.stdout.strip())
    except ValueError as exc:
        raise DownloadError(f"Invalid duration for {output_path.name}.") from exc

    expected = segment.duration_s
    overshoot = actual_duration - expected

    # Running long is not a fault. Cutting with a stream copy can only land on
    # a keyframe, so a clip routinely runs past its requested end, and the
    # extra footage costs nothing - the captioner trims to a complete thought
    # afterwards regardless. Rejecting these threw away perfectly good clips
    # over something that was never going to matter.
    #
    # The one overshoot worth catching is the pathological kind, where -t was
    # ignored and the "clip" is the rest of the broadcast. That is orders of
    # magnitude out, not seconds, so the ceiling is deliberately far above any
    # keyframe interval.
    if overshoot > max(RUNAWAY_CUT_S, expected * 3.0):
        raise DownloadError(
            f"Cut did not take for {output_path.name}: "
            f"expected ~{expected:.1f}s, got {actual_duration:.1f}s."
        )

    # Coming up short is the real fault - it means the cut was truncated and
    # the moment the clip exists for may not be in it.
    if -overshoot > max(2.0, expected * 0.15):
        raise DownloadError(
            f"Clip is short for {output_path.name}: "
            f"expected ~{expected:.1f}s, got {actual_duration:.1f}s."
        )


def cleanup_cached_videos() -> None:
    """Delete the downloaded source video(s) once every clip has been cut."""
    for path in list(_VIDEO_CACHE.values()):
        path.unlink(missing_ok=True)
    _VIDEO_CACHE.clear()


def download_all_segments(
    youtube_url: str,
    title: str,
    segments: list[ReplaySegment],
    output_dir: Path,
) -> list[Path]:
    """Download every detected replay segment."""
    output_dir.mkdir(parents=True, exist_ok=True)

    targets = [
        (segment, build_output_path(output_dir, title, segment.start_s, segment.end_s))
        for segment in segments
    ]
    missing = [
        (segment, path)
        for segment, path in targets
        if not (path.exists() and path.stat().st_size > 0)
    ]

    # Downloading the source is by far the expensive step, so skip it entirely
    # when every clip is already on disk.
    if not missing:
        print(f"All {len(targets)} clip(s) already present - nothing to download.\n")
        return [path for _, path in targets]

    print(f"{len(missing)} of {len(targets)} clip(s) missing.")

    saved: list[Path] = []
    failed: list[str] = []

    try:
        print("Downloading source video once (highest quality)...")
        source_video = _download_full_video(youtube_url, output_dir)
        print(f"Source ready: {source_video.name}\n")

        for segment, output_path in tqdm(missing, desc="Cutting clips", unit="clip"):
            # One bad cut must not lose the batch. The source download is the
            # expensive part and it has already happened by this point, so
            # aborting here throws away every remaining clip and the source
            # with it, forcing the whole download again for one bad segment.
            try:
                saved_path = download_segment(youtube_url, segment, output_path, source_video)
                _verify_clip(saved_path, segment)
                saved.append(saved_path)
            except DownloadError as exc:
                failed.append(f"{output_path.name}: {exc}")
                output_path.unlink(missing_ok=True)
                tqdm.write(f"  skipped {output_path.name}")
    finally:
        cleanup_cached_videos()

    if failed:
        print(f"\n{len(failed)} clip(s) could not be cut:")
        for note in failed[:10]:
            print(f"  - {note}")
        if len(failed) > 10:
            print(f"  ... and {len(failed) - 10} more")

    if not saved and failed:
        raise DownloadError(
            f"Every clip failed to cut ({len(failed)} of {len(failed)}). "
            "The source download may be damaged."
        )

    return saved
