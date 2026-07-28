"""Orchestrate clip -> transcript -> captions -> vertical render."""

from __future__ import annotations

import tempfile
from pathlib import Path

import audio
import caption_builder
import config
import ffmpeg_tools
import renderer
import splitter
import tracker
import transcriber
from models import CaptionGroup, VideoInfo


class PipelineError(Exception):
    """Raised when a clip cannot be processed."""


def find_clips(input_dir: Path) -> list[Path]:
    """Source clips to caption, ignoring anything we produced ourselves."""
    if not input_dir.exists():
        raise PipelineError(f"Input folder does not exist: {input_dir}")

    clips = [
        path
        for path in sorted(input_dir.glob("*.mp4"))
        if "[vertical" not in path.stem
    ]
    return clips


def transcribe_clip(clip: Path) -> list[CaptionGroup]:
    """Extract audio, transcribe it, and group the words for display."""
    with tempfile.TemporaryDirectory(prefix="clipcaptioner_audio_") as tmp:
        wav_path = Path(tmp) / "audio.wav"
        audio.extract_audio(clip, wav_path)
        words = transcriber.transcribe_words(wav_path)

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
    info = ffmpeg_tools.probe_video(clip)
    duration = info.duration_s

    groups = transcribe_clip(clip)
    parts = splitter.split_into_parts(groups, duration)

    pending = [
        part
        for part in parts
        if not renderer.build_output_path(output_dir, clip, part).exists()
    ]
    crop_path = _crop_path(clip, info) if pending else None

    rendered: list[Path] = []
    for part in parts:
        output_path = renderer.build_output_path(output_dir, clip, part)
        if output_path.exists() and output_path.stat().st_size > 0:
            print(f"    part {part.index}/{part.total}: already rendered, skipping")
            rendered.append(output_path)
            continue

        print(
            f"    part {part.index}/{part.total}: "
            f"{part.duration_s:.0f}s, {len(part.groups)} caption group(s) - rendering"
        )
        rendered.append(
            renderer.render_part(clip, part, info, output_path, crop_path=crop_path)
        )

    return rendered


def run(input_dir: Path | None = None, output_dir: Path | None = None) -> list[Path]:
    """Caption every clip in the input folder."""
    input_dir = input_dir or config.INPUT_DIR
    output_dir = output_dir or config.OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    clips = find_clips(input_dir)
    if not clips:
        print(f"No clips found in {input_dir}")
        return []

    print(f"Found {len(clips)} clip(s) in {input_dir}\n")

    produced: list[Path] = []
    for position, clip in enumerate(clips, start=1):
        print(f"[{position}/{len(clips)}] {clip.name}")
        try:
            produced.extend(process_clip(clip, output_dir))
        except (
            ffmpeg_tools.FFmpegError,
            transcriber.TranscriptionError,
        ) as exc:
            # One bad clip should not abandon the rest of the batch.
            print(f"    skipped: {exc}")
        print()

    return produced
