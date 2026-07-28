"""Crop to 9:16, burn in captions, and encode the final clip."""

from __future__ import annotations

import tempfile
from pathlib import Path

import ass_writer
import config
import ffmpeg_tools
from models import ClipPart, VideoInfo

_TARGET_ASPECT = config.TARGET_WIDTH / config.TARGET_HEIGHT


def _even(value: int) -> int:
    """H.264 needs even dimensions."""
    return value - (value % 2)


def compute_crop(info: VideoInfo) -> tuple[int, int, int, int]:
    """Centre crop box (w, h, x, y) that matches the target aspect ratio."""
    source_aspect = info.width / info.height

    if source_aspect > _TARGET_ASPECT:
        # Landscape source: full height, trim the sides.
        crop_h = _even(info.height)
        crop_w = _even(int(round(info.height * _TARGET_ASPECT)))
    else:
        # Already narrow: full width, trim top and bottom.
        crop_w = _even(info.width)
        crop_h = _even(int(round(info.width / _TARGET_ASPECT)))

    crop_w = min(crop_w, _even(info.width))
    crop_h = min(crop_h, _even(info.height))
    x = max(0, (info.width - crop_w) // 2)
    y = max(0, (info.height - crop_h) // 2)
    return crop_w, crop_h, x, y


def _build_filter(info: VideoInfo, subtitle_name: str) -> str:
    crop_w, crop_h, x, y = compute_crop(info)
    return (
        f"crop={crop_w}:{crop_h}:{x}:{y},"
        f"scale={config.TARGET_WIDTH}:{config.TARGET_HEIGHT}:flags=lanczos,"
        f"setsar=1,"
        # Referenced by bare name; ffmpeg runs with cwd set to the file's
        # folder so Windows drive letters never reach the filtergraph parser,
        # where a colon separates arguments.
        f"subtitles={subtitle_name}"
    )


def render_part(
    source: Path,
    part: ClipPart,
    info: VideoInfo,
    output_path: Path,
) -> Path:
    """Render one part of a clip as a captioned vertical video."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="clipcaptioner_") as tmp:
        work_dir = Path(tmp)
        subtitle_name = "subs.ass"
        ass_writer.write_ass(part.groups, work_dir / subtitle_name)

        args = [
            ffmpeg_tools.tool_path("ffmpeg"),
            "-y",
            "-loglevel",
            "error",
            "-ss",
            f"{part.start_s:.3f}",
            "-t",
            f"{part.duration_s:.3f}",
            "-i",
            str(source),
            "-vf",
            _build_filter(info, subtitle_name),
            "-c:v",
            "libx264",
            "-preset",
            config.VIDEO_PRESET,
            "-crf",
            str(config.VIDEO_CRF),
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            "-b:a",
            config.AUDIO_BITRATE,
            "-movflags",
            "+faststart",
            str(output_path),
        ]
        ffmpeg_tools.run(args, cwd=work_dir)

    if not output_path.exists() or output_path.stat().st_size == 0:
        raise ffmpeg_tools.FFmpegError(f"Render produced nothing for {output_path.name}")

    return output_path


def build_output_path(output_dir: Path, source: Path, part: ClipPart) -> Path:
    """Name the rendered file, numbering parts only when there are several."""
    stem = source.stem
    if part.total <= 1:
        return output_dir / f"{stem} [vertical].mp4"
    return output_dir / f"{stem} [vertical part {part.index} of {part.total}].mp4"
