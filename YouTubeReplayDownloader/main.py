"""
YouTube highlight clip downloader.

Works on normal videos and on past live broadcasts. Edit YOUTUBE_URL below,
then run:
    python main.py
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import tempfile

import speech
from downloader import DownloadError, download_all_segments, fetch_video_title
from ffmpeg_utils import FFmpegNotFoundError
from moment_finder import NoMomentSignal, find_moments
import peak_detector as detect
from peak_detector import ReplaySegment, detect_replay_segments
from utils import InvalidYouTubeURL, format_timestamp, parse_youtube_url


# ---------------------------------------------------------------------------
# Edit this URL before running
# ---------------------------------------------------------------------------
YOUTUBE_URL = "https://youtu.be/Q2hOryHdgAk?si=ZtRptZqkR7UQCxlx"

# Where the finished clips are written. Created automatically if missing.
# Defaults to an "output" folder next to this script so the project works
# anywhere. Replace with an absolute path to send clips elsewhere, e.g.
#     OUTPUT_DIR = Path(r"E:\Youtube Videos\Videos")
OUTPUT_DIR = Path(__file__).resolve().parent / "output"

# How many clips to keep, scaled to how long the source runs. A three hour
# stream holds far more worth clipping than a ten minute video, and a fixed
# count threw most of a stream away. Only the strongest peaks survive, so
# these are always "the N hottest moments".
#
# Supply runs out on its own: clips may not overlap, so a 55 minute stream
# yields about 27 distinct ones however many are asked for.
# Was 15, which turned a two hour episode into thirty clips and spent most of
# them on moments that were merely above average. Four an hour keeps only
# moments with a clear peak behind them: the same episode now yields about
# eight, and a clip has to earn its slot against the whole hour rather than
# against its immediate neighbours.
CLIPS_PER_HOUR = 4

# Floor for short videos and ceiling for very long ones. The floor is low on
# purpose - a ten minute video that holds three good moments should give three
# clips, not five padded out to meet a quota.
MIN_CLIPS = 3
MAX_CLIPS = 20

# Set a number here to ignore the scaling and always take exactly that many.
FIXED_CLIP_COUNT: int | None = None

# Longest a clip may run, in seconds. Shorter clips loop, and loops are reach,
# so this is a ceiling rather than a target. Was 90s, which is long enough that
# a viewer decides to leave before the point arrives.
MAX_CLIP_SECONDS = 45.0

# How much of that clip may be spent getting to the peak.
#
# The peak is the moment people replayed, so it is the hook, and the opening
# seconds are the whole audition - in a Shorts feed there is no thumbnail, the
# first frames are it.
#
# Was 10.0, which spent the opening ten seconds getting to the moment people
# actually replayed. On a Short that is most of the window in which a viewer
# decides to stay, and the research on why these stall is blunt about it: lead
# with the peak, do not build up to it.
#
# Four rather than one, because a podcast payoff is a sentence and a hard cut
# onto the peak lands mid-clause. Four seconds is roughly a sentence of run-up:
# enough that the line makes grammatical sense, short enough that the payoff
# arrives while the viewer is still deciding.
#
# What makes this safe is the hook card, which did not exist when this was 10.
# The objection to opening near the peak was that a cold viewer has no context;
# the card now states the point on screen while the speaker is still reaching
# it, so the run-up no longer has to carry that job.
HOOK_LEAD_SECONDS = 4.0

# What kind of video this is, which decides what evidence is trusted.
#   "talk"    - podcasts and interviews. Most-replayed and chat only.
#   "general" - anything else. Also allows speech energy.
#   None      - ask each run.
#
# Speech energy is deliberately off for talk content: it measures how loud the
# speaker is, so on a podcast it finds the shouting rather than the point being
# made. For a stream with no audience data it is the only thing available, but
# a podcast has a most-replayed graph and should say so plainly when it does
# not, rather than quietly picking clips on volume.
CONTENT_TYPE: str | None = None

# Align clip edges with pauses in speech so they start and end on a sentence
# rather than mid-word, and skip candidates that are mostly silence. Costs one
# audio-only download before the video is fetched.
SNAP_TO_SPEECH = True

# Extra seconds kept on the end of every clip. ClipCaptioner reads the
# transcript and stops the finished video where a sentence actually ends,
# which it can only do if there is footage past the cut to reach into.
# Unused padding is dropped there, so it costs a little download, not runtime.
TAIL_PADDING_SECONDS = 20.0


def _ask_content_type() -> str | None:
    """Ask what the video is, because it decides which signals are trusted.

    Speech energy suits a stream with no audience data, but for a podcast it
    finds whoever is loudest rather than whatever is interesting. Gameplay
    needs a different detector entirely, so that answer points elsewhere
    instead of producing clips picked on the wrong evidence.

    Returns the chosen type, or None if the user should use EsportsClipper.
    """
    print("What kind of video is this?")
    print("  [1] Podcast, interview or talk   - most-replayed and chat only")
    print("  [2] General video or stream      - also allows speech energy")
    print("  [3] Gaming or esports            - use EsportsClipper instead")

    try:
        answer = input("Choose [1]: ").strip() or "1"
    except EOFError:
        # Piped or scheduled run: the safe default is the strict one.
        return "talk"

    if answer == "3":
        return None
    return "general" if answer == "2" else "talk"


def _clip_budget(duration_s: float) -> int:
    """How many clips to take from a source of this length."""
    if FIXED_CLIP_COUNT is not None:
        return max(1, FIXED_CLIP_COUNT)

    if duration_s <= 0:
        return MIN_CLIPS

    wanted = round((duration_s / 3600.0) * CLIPS_PER_HOUR)
    return int(max(MIN_CLIPS, min(MAX_CLIPS, wanted)))


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


def _refine_with_speech(
    canonical_url: str,
    segments: list[ReplaySegment],
) -> list[ReplaySegment]:
    """Align clip edges with pauses and drop candidates that are mostly silent.

    Analysing the audio track alone is cheap next to the full video download
    that follows, and it happens before the hottest clips are chosen so a
    rejected candidate frees its slot for the next best one.
    """
    if not SNAP_TO_SPEECH:
        return segments

    try:
        with tempfile.TemporaryDirectory(prefix="speech_") as tmp:
            audio_path = speech.download_audio(canonical_url, Path(tmp))
            speech_map = speech.analyse_audio(audio_path)
    except speech.SpeechAnalysisFailed as exc:
        # Rough boundaries beat no clips at all.
        print(f"Speech analysis unavailable ({exc}) - using raw boundaries.")
        return segments

    refined, dropped = speech.refine_segments(segments, speech_map, MAX_CLIP_SECONDS)
    print(
        f"Speech: {len(speech_map.silences)} pause(s) found; "
        f"edges aligned, {dropped} silent candidate(s) dropped."
    )
    return refined


def _ask_about_captioning() -> list[str] | None:
    """Ask up front whether to caption, and how, so the run can be left alone.

    Asked before the download rather than after it, because the download is the
    long part. A question waiting at the end means coming back to a prompt
    instead of to finished Shorts.

    Every answer ClipCaptioner would otherwise ask for is collected here and
    passed as flags, so it never stops to ask once it starts. Moving only the
    first question would have achieved nothing: the other three would still be
    sitting there an hour later.

    Returns the flags to run it with, or None to skip captioning.
    """
    try:
        if input("Caption the clips afterwards? [Y/n] ").strip().lower() in {"n", "no"}:
            return None

        flags: list[str] = []

        answer = input("  Burn captions in? [Y/n] ").strip().lower()
        flags.append("--no-captions" if answer in {"n", "no"} else "--captions")

        print("  Layout:  [1] podcast  [2] streamer  [3] gameplay")
        layout = input("  Choose [1]: ").strip() or "1"
        flags += ["--layout", {"2": "stacked", "3": "fit"}.get(layout, "crop")]

        print("  Alongside the speaker:  [1] cutaways  [2] filler  [3] nothing")
        extras = input("  Choose [1]: ").strip() or "1"
        if extras == "2":
            flags += ["--parkour", "--no-broll"]
        elif extras == "3":
            flags += ["--no-parkour", "--no-broll"]
        else:
            flags += ["--broll", "--no-parkour"]

        return flags
    except EOFError:
        # Piped or scheduled: caption with the defaults rather than stopping,
        # since a half-finished job helps nobody when there is no one watching.
        return ["--captions", "--layout", "crop", "--broll", "--no-parkour"]


def _run_clip_captioner(output_dir: Path, flags: list[str]) -> int:
    """Hand the finished clips to ClipCaptioner, in its own process.

    A subprocess rather than an import. Both projects define config.py and
    main.py, so importing the sibling puts two different modules under the same
    names and whichever loads first wins - the exact collision EsportsClipper
    needs shared.py to work around. A separate process keeps each project's
    imports its own, and inherits this terminal so ClipCaptioner's prompts
    still reach you.
    """
    # Resolved before handing over, because the subprocess runs with its cwd
    # set to ClipCaptioner. A relative path would be read from there instead,
    # quietly pointing the captioner at its own output folder.
    output_dir = output_dir.resolve()

    captioner = Path(__file__).resolve().parent.parent / "ClipCaptioner"
    entry = captioner / "main.py"
    if not entry.exists():
        print(f"\nClipCaptioner not found at {captioner}")
        print(f"Clips are in {output_dir}")
        return 1

    print(f"\nHanding {len(list(output_dir.glob('*.mp4')))} clip(s) to ClipCaptioner...\n")
    result = subprocess.run(
        [sys.executable, "-u", str(entry), "--input", str(output_dir), *flags],
        cwd=str(captioner),
        check=False,
    )
    return result.returncode


def _mark_output_as_talk(output_dir: Path) -> None:
    """Tell ClipCaptioner how much of each clip is padding rather than clip.

    Every clip carries TAIL_PADDING_SECONDS of extra footage so the captioner
    can reach past the cut to finish a sentence. It has no way to tell that
    tail apart from the clip itself, so without this it treats the padded
    length as the intended length and keeps nearly all of it - which is how a
    45s clip becomes a 65s one.
    """
    marker = output_dir / "content.json"
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        marker.write_text(
            json.dumps(
                {
                    "content_type": "talk",
                    "tail_padding_seconds": TAIL_PADDING_SECONDS,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
    except OSError as exc:
        # Only a hint for the next stage; never lose clips over it.
        print(f"Could not write {marker.name}: {exc}")


def _add_tail_padding(
    segments: list[ReplaySegment],
    video_duration_s: float,
) -> list[ReplaySegment]:
    """Keep extra footage past each cut for the captioner to reach into.

    ClipCaptioner ends the finished video where a sentence actually finishes,
    which it can only do if the clip contains footage past the activity-based
    cut. Whatever it does not use is dropped there.
    """
    if TAIL_PADDING_SECONDS <= 0:
        return segments

    padded: list[ReplaySegment] = []
    for segment in segments:
        end_s = segment.end_s + TAIL_PADDING_SECONDS
        if video_duration_s > 0:
            end_s = min(end_s, video_duration_s)

        padded.append(
            ReplaySegment(
                start_s=segment.start_s,
                end_s=end_s,
                peak_s=segment.peak_s,
                peak_score=segment.peak_score,
                prominence=segment.prominence,
            )
        )

    return padded


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
    # Clip filenames are built from YouTube titles, and those carry emoji.
    # In a console Windows copes; the moment stdout is a pipe it falls back
    # to the ANSI codepage and the first emoji title raises
    # UnicodeEncodeError from a print, part way through a run that has
    # already downloaded gigabytes.
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            reconfigure(encoding="utf-8", errors="replace")

    output_dir = OUTPUT_DIR

    print("YouTube Highlight Downloader")
    print("=" * 28)
    print(f"URL: {YOUTUBE_URL}\n")

    content = CONTENT_TYPE or _ask_content_type()
    caption_flags = _ask_about_captioning()
    if content is None:
        print("\nGameplay needs a different detector - casters and gunfire, not")
        print("speech. Use EsportsClipper, which is built for it:\n")
        print("  cd ../EsportsClipper")
        print("  py main.py")
        return 0

    allow_energy = content != "talk"
    print(f"\nContent: {content}")
    print(f"Signals: most-replayed, chat{', speech energy' if allow_energy else ''}")
    print(f"Output folder: {output_dir}\n")

    try:
        canonical_url = parse_youtube_url(YOUTUBE_URL)

        print("Looking for clip-worthy moments...")
        moments = find_moments(canonical_url, allow_speech_energy=allow_energy)
        print(f"Signal: {moments.source} ({len(moments.points)} data points)")

        # Leave room for boundary snapping to widen a clip out to the nearest
        # pause. Without this headroom every outward snap would breach the
        # length cap and be rejected, so edges could only ever move inwards.
        detect_budget = MAX_CLIP_SECONDS
        if SNAP_TO_SPEECH:
            detect_budget -= 2 * speech.SNAP_TOLERANCE_S

        segments = detect_replay_segments(
            moments.points,
            max_segment_seconds=detect_budget,
            lead_in_seconds=HOOK_LEAD_SECONDS,
        )

        if not segments:
            print(f"No significant peaks were detected using {moments.source}.")
            return 0

        segments = _refine_with_speech(canonical_url, segments)
        if not segments:
            print("Every candidate was mostly silence. Nothing to download.")
            return 0

        budget = _clip_budget(moments.duration_s)
        if len(segments) > budget:
            print(
                f"Source runs {moments.duration_s / 60:.0f} min - "
                f"keeping the {budget} hottest of {len(segments)} peaks."
            )
            segments = _pick_hottest(segments, budget)
        elif len(segments) < budget:
            # Peak detection came up short. It asks a moment to be a local
            # maximum clearing a prominence and a height bar, which on a long
            # podcast discards a great deal: on a 150 minute episode the raw
            # curve holds 32 local maxima, smoothing leaves 13 and the
            # thresholds pass 9, against a budget of 10. The scores behind
            # those cuts decline smoothly with no cliff, so the moments just
            # under the bar are not meaningfully worse than the ones just over.
            extra = detect.backfill(
                moments.points,
                segments,
                budget,
                max_segment_seconds=detect_budget,
                lead_in_seconds=HOOK_LEAD_SECONDS,
            )
            if extra:
                print(
                    f"Only {len(segments)} clear peak(s) for a budget of {budget} - "
                    f"adding the {len(extra)} next best moment(s)."
                )
                segments = sorted(segments + extra, key=lambda s: s.start_s)

        # Padding is added last so it cannot push two candidates into overlap
        # during selection, which would cost one of the slots.
        segments = _add_tail_padding(segments, moments.duration_s)

        _print_segments(segments)

        title = fetch_video_title(canonical_url)
        print(f"Video title: {title}\n")

        saved_files = download_all_segments(
            youtube_url=canonical_url,
            title=title,
            segments=segments,
            output_dir=output_dir,
        )

        _mark_output_as_talk(output_dir)

        print("Download complete:\n")
        for path in saved_files:
            print(f"  - {path.name}")

        if saved_files and caption_flags is not None:
            return _run_clip_captioner(output_dir, caption_flags)

        print(f"\nTo caption them later:\n  cd ../ClipCaptioner && py main.py")
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
