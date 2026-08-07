"""Locate pauses in speech, so clips can start and end where a sentence does.

Activity peaks say *when* something interesting happened, not where the
sentence around it begins. Cutting on the raw peak boundary lands mid-word,
which is the clearest giveaway of an auto-generated clip.

Silence is measured relative to the material's own loudness rather than at a
fixed dB. Real podcast and vlog audio carries constant room tone: a clip
measured here averaged -18 dB and never once dropped below -30 dB, so a fixed
threshold finds nothing at all on some sources and flags quiet speech as a
pause on others.
"""

from __future__ import annotations

import re
import subprocess
import uuid
from dataclasses import dataclass
from pathlib import Path

import yt_dlp

from ffmpeg_utils import ensure_ffmpeg_available, ffmpeg_location_option
from peak_detector import ReplaySegment
from utils import parse_youtube_url

# How far below the track's mean volume counts as a pause. Measured over 338s
# of real vlog audio: mean - 4 dB yields about nine pauses a minute, so a
# boundary nearly always finds one within tolerance. Deeper thresholds get too
# sparse to snap to, shallower ones start catching breaths mid-sentence.
SILENCE_OFFSET_DB = 4.0

# Shorter dips are breaths and stop consonants, not sentence boundaries.
MIN_SILENCE_S = 0.20

# How far a boundary may move to reach a pause. Beyond this the clip would no
# longer contain the moment it was chosen for.
SNAP_TOLERANCE_S = 5.0

# A candidate whose audio is mostly pause has nothing worth captioning.
MAX_SILENCE_RATIO = 0.70


class SpeechAnalysisFailed(Exception):
    """Raised when the audio could not be analysed."""


@dataclass(frozen=True)
class SpeechMap:
    """Where the pauses are in a piece of audio."""

    silences: list[tuple[float, float]]
    mean_volume_db: float

    def silence_seconds_between(self, start_s: float, end_s: float) -> float:
        total = 0.0
        for silence_start, silence_end in self.silences:
            overlap = min(end_s, silence_end) - max(start_s, silence_start)
            if overlap > 0:
                total += overlap
        return total

    def speech_ratio(self, start_s: float, end_s: float) -> float:
        span = end_s - start_s
        if span <= 0:
            return 0.0
        return 1.0 - (self.silence_seconds_between(start_s, end_s) / span)


def download_audio(
    youtube_url: str,
    work_dir: Path,
    live_from_start: bool = False,
) -> Path:
    """Fetch the audio track only - far smaller and faster than the video.

    live_from_start pulls a broadcast that is still running from its
    beginning rather than from the live edge, which is what makes an
    in-progress stream analysable at all.
    """
    canonical_url = parse_youtube_url(youtube_url)
    base = work_dir / f"speech_{uuid.uuid4().hex}"

    options = {
        "format": "bestaudio/best",
        "outtmpl": str(base) + ".%(ext)s",
        "quiet": True,
        "no_warnings": True,
        "noprogress": True,
        "retries": 5,
    }
    if live_from_start:
        options["live_from_start"] = True
    options.update(ffmpeg_location_option())

    try:
        with yt_dlp.YoutubeDL(options) as ydl:
            ydl.download([canonical_url])
    except Exception as exc:
        raise SpeechAnalysisFailed(f"Could not download audio: {exc}") from exc

    produced = [path for path in work_dir.glob(f"{base.name}*") if path.is_file()]
    if not produced:
        raise SpeechAnalysisFailed("Audio download produced no file.")
    return produced[0]


def _run_filter(audio_path: Path, audio_filter: str) -> str:
    ensure_ffmpeg_available()
    args = [
        "ffmpeg",
        "-hide_banner",
        "-nostats",
        "-i",
        str(audio_path),
        "-af",
        audio_filter,
        "-f",
        "null",
        "-",
    ]
    result = subprocess.run(args, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        raise SpeechAnalysisFailed(f"FFmpeg analysis failed: {result.stderr.strip()[-300:]}")
    # These filters report through stderr even on success.
    return result.stderr


_MEAN_VOLUME = re.compile(r"mean_volume:\s*(-?[\d.]+) dB")
_SILENCE_START = re.compile(r"silence_start:\s*(-?[\d.]+)")
_SILENCE_END = re.compile(r"silence_end:\s*(-?[\d.]+)")


def _measure_mean_volume(audio_path: Path) -> float:
    output = _run_filter(audio_path, "volumedetect")
    match = _MEAN_VOLUME.search(output)
    if not match:
        raise SpeechAnalysisFailed("Could not measure the audio's mean volume.")
    return float(match.group(1))


def _find_silences(audio_path: Path, threshold_db: float) -> list[tuple[float, float]]:
    output = _run_filter(
        audio_path,
        f"silencedetect=noise={threshold_db:.1f}dB:d={MIN_SILENCE_S}",
    )

    starts = [float(m.group(1)) for m in _SILENCE_START.finditer(output)]
    ends = [float(m.group(1)) for m in _SILENCE_END.finditer(output)]

    silences: list[tuple[float, float]] = []
    for index, start in enumerate(starts):
        if index < len(ends):
            end = ends[index]
            if end > start:
                silences.append((start, end))
        else:
            # A trailing silence runs to the end of the file and is unclosed.
            silences.append((start, start + MIN_SILENCE_S))

    return silences


def analyse_audio(audio_path: Path) -> SpeechMap:
    """Measure loudness, then find pauses relative to it."""
    mean_db = _measure_mean_volume(audio_path)
    silences = _find_silences(audio_path, mean_db - SILENCE_OFFSET_DB)
    return SpeechMap(silences=silences, mean_volume_db=mean_db)


def _snap_start(speech: SpeechMap, target_s: float) -> float | None:
    """Find where speech resumes, at or before the requested start.

    Widening is preferred over narrowing: snapping forwards would clip the
    opening words off the very sentence the clip exists for. Only if no pause
    sits before the target does it accept a later one.
    """
    before = [
        end for _start, end in speech.silences if target_s - SNAP_TOLERANCE_S <= end <= target_s
    ]
    if before:
        return max(before)

    after = [
        end for _start, end in speech.silences if target_s < end <= target_s + SNAP_TOLERANCE_S
    ]
    return min(after) if after else None


def _snap_end(speech: SpeechMap, target_s: float) -> float | None:
    """Find where speech stops, at or after the requested end.

    Same reasoning mirrored: prefer running on to the next pause rather than
    truncating the moment mid-thought.
    """
    after = [
        start
        for start, _end in speech.silences
        if target_s <= start <= target_s + SNAP_TOLERANCE_S
    ]
    if after:
        return min(after)

    before = [
        start
        for start, _end in speech.silences
        if target_s - SNAP_TOLERANCE_S <= start < target_s
    ]
    return max(before) if before else None


def _valid(segment: ReplaySegment, start_s: float, end_s: float, max_duration_s: float) -> bool:
    if end_s <= start_s:
        return False
    if not (start_s <= segment.peak_s <= end_s):
        return False
    return max_duration_s <= 0 or (end_s - start_s) <= max_duration_s


def snap_segment(
    segment: ReplaySegment,
    speech: SpeechMap,
    max_duration_s: float = 0.0,
) -> ReplaySegment:
    """Move a segment's edges onto nearby pauses, keeping the peak inside.

    Snapping outwards can push a clip past its length budget, so the snapped
    edges are tried together first, then one at a time, and the original is
    kept if nothing fits.
    """
    start_s = _snap_start(speech, segment.start_s)
    end_s = _snap_end(speech, segment.end_s)

    candidates = (
        (start_s, end_s),
        (start_s, segment.end_s),
        (segment.start_s, end_s),
    )

    for new_start, new_end in candidates:
        if new_start is None or new_end is None:
            continue
        if _valid(segment, new_start, new_end, max_duration_s):
            return ReplaySegment(
                start_s=new_start,
                end_s=new_end,
                peak_s=segment.peak_s,
                peak_score=segment.peak_score,
                prominence=segment.prominence,
            )

    return segment


def refine_segments(
    segments: list[ReplaySegment],
    speech: SpeechMap,
    max_duration_s: float = 0.0,
) -> tuple[list[ReplaySegment], int]:
    """Snap every segment to speech and drop the ones that are mostly pause.

    Returns the surviving segments and how many were dropped.
    """
    kept: list[ReplaySegment] = []
    dropped = 0

    for segment in segments:
        if speech.speech_ratio(segment.start_s, segment.end_s) < (1.0 - MAX_SILENCE_RATIO):
            dropped += 1
            continue
        kept.append(snap_segment(segment, speech, max_duration_s))

    return kept, dropped
