"""Orchestrate clip -> transcript -> captions -> vertical render."""

from __future__ import annotations

import queue
import tempfile
import threading
from collections.abc import Iterator
from pathlib import Path

import audio
import caption_builder
import config
import ffmpeg_tools
import renderer
import splitter
import thought
import titler
import tracker
import transcriber
from models import CaptionGroup, VideoInfo, Word


class PipelineError(Exception):
    """Raised when a clip cannot be processed."""


def find_clips(input_dir: Path, output_dir: Path | None = None) -> list[Path]:
    """Source clips to caption, ignoring anything we produced ourselves.

    Rendered clips are named after their title, so they cannot be recognised
    by name any more. Location is the reliable test: anything sitting in the
    output folder is our own work.
    """
    if not input_dir.exists():
        raise PipelineError(f"Input folder does not exist: {input_dir}")

    resolved_output = output_dir.resolve() if output_dir else None

    clips: list[Path] = []
    for path in sorted(input_dir.glob("*.mp4")):
        if resolved_output and resolved_output in path.resolve().parents:
            continue
        clips.append(path)

    return clips


def transcribe_clip(clip: Path) -> list[CaptionGroup]:
    """Extract audio, transcribe it, and group the words for display."""
    with tempfile.TemporaryDirectory(prefix="clipcaptioner_audio_") as tmp:
        wav_path = Path(tmp) / "audio.wav"
        audio.extract_audio(clip, wav_path)
        # The filename carries the video title, which primes the decoder.
        words = transcriber.transcribe_words(wav_path, title=clip.stem)

    return caption_builder.build_groups(words)


def _crop_path(clip: Path, info: VideoInfo) -> list[tracker.CropKeyframe] | None:
    """Face-tracked crop path, or None to fall back to centre framing."""
    if not config.TRACK_FACES:
        return None

    crop_w, _crop_h, _x, _y = renderer.compute_crop(info)
    try:
        path = tracker.build_crop_path(clip, info, crop_w)
    except tracker.TrackingUnavailable as exc:
        # Tracking is an enhancement, never a reason to lose the clip.
        print(f"    face tracking off ({exc}) - centre crop")
        return None

    print(f"    face tracking: {len(path)} crop keyframe(s)")
    return path


def process_clip(clip: Path, output_dir: Path) -> list[Path]:
    """Caption one clip, returning every rendered file."""
    return render_clip(clip, transcribe_clip(clip), output_dir)


def render_clip(clip: Path, groups: list[CaptionGroup], output_dir: Path) -> list[Path]:
    """Render an already-transcribed clip."""
    info = ffmpeg_tools.probe_video(clip)
    duration = info.duration_s

    # Stop where the sentence does. Applied before splitting so the chosen end
    # is the clip's real end, and any trailing part-boundary follows from it.
    # Capped at the footage available: the downloader leaves a tail to reach
    # into, but there is nothing beyond the file itself.
    usable_end = min(thought.choose_end(groups, duration), duration)

    if duration - usable_end > 0.5:
        print(
            f"    ended on a complete thought: "
            f"{duration:.1f}s -> {usable_end:.1f}s"
        )
        groups = thought.trim_groups(groups, usable_end)

    parts = splitter.split_into_parts(groups, usable_end)

    # The title decides the filename, so it has to be worked out before the
    # output path exists, not written alongside it afterwards.
    planned = []
    for part in parts:
        words = [word for group in part.groups for word in group.words]
        title = titler.make_title(words, clip.stem)
        planned.append((part, title, words, renderer.build_output_path(output_dir, clip, part, title)))

    pending = [entry for entry in planned if not entry[3].exists()]
    crop_path = _crop_path(clip, info) if pending else None

    rendered: list[Path] = []
    for part, title, words, output_path in planned:
        if output_path.exists() and output_path.stat().st_size > 0:
            print(f"    part {part.index}/{part.total}: already rendered, skipping")
            rendered.append(output_path)
            continue

        print(
            f"    part {part.index}/{part.total}: "
            f"{part.duration_s:.0f}s, {len(part.groups)} caption group(s) - rendering"
        )
        print(f"    title: {title}")
        rendered.append(
            renderer.render_part(
                clip, part, info, output_path, crop_path=crop_path, title=title
            )
        )
        _write_description(title, words, clip.name, output_path)

    return rendered


def _write_description(
    title: str,
    words: list[Word],
    source_name: str,
    output_path: Path,
) -> None:
    """Save the description and hashtags to paste when uploading."""
    if not config.WRITE_TITLES:
        return

    try:
        output_path.with_suffix(".txt").write_text(
            titler.build_description(title, words, source_name) + "\n",
            encoding="utf-8",
        )
    except OSError as exc:
        # Upload text is a convenience; never lose a rendered clip over one.
        print(f"    could not write description: {exc}")


def _transcribe_ahead(clips: list[Path]) -> Iterator[tuple[Path, list[CaptionGroup] | Exception]]:
    """Transcribe the next clip while the current one is still encoding.

    Transcription is CPU-bound and encoding runs on the GPU, so they barely
    contend. Overlapping them recovers most of the transcription time. The
    queue holds one item, which keeps at most one clip's work in flight.
    """
    results: queue.Queue = queue.Queue(maxsize=1)

    def worker() -> None:
        for clip in clips:
            try:
                results.put((clip, transcribe_clip(clip)))
            except Exception as exc:  # handed to the consumer, not swallowed
                results.put((clip, exc))
        results.put(None)

    thread = threading.Thread(target=worker, daemon=True)
    thread.start()

    while True:
        item = results.get()
        if item is None:
            break
        yield item

    thread.join()


def run(input_dir: Path | None = None, output_dir: Path | None = None) -> list[Path]:
    """Caption every clip in the input folder."""
    input_dir = input_dir or config.INPUT_DIR
    output_dir = output_dir or config.OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    clips = find_clips(input_dir, output_dir)
    if not clips:
        print(f"No clips found in {input_dir}")
        return []

    print(f"Found {len(clips)} clip(s) in {input_dir}\n")

    produced: list[Path] = []
    for position, (clip, outcome) in enumerate(_transcribe_ahead(clips), start=1):
        print(f"[{position}/{len(clips)}] {clip.name}")

        if isinstance(outcome, Exception):
            # One bad clip should not abandon the rest of the batch.
            print(f"    skipped: {outcome}")
            print()
            continue

        try:
            produced.extend(render_clip(clip, outcome, output_dir))
        except ffmpeg_tools.FFmpegError as exc:
            print(f"    skipped: {exc}")
        print()

    return produced
