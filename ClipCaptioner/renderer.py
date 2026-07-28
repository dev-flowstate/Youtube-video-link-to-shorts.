"""Crop to 9:16, burn in captions, and encode the final clip."""

from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path

import ass_writer
import config
import ffmpeg_tools
import tracker
from models import ClipPart, VideoInfo

_TARGET_ASPECT = config.TARGET_WIDTH / config.TARGET_HEIGHT

_hardware_encoder_ok: bool | None = None


def hardware_encoder_available() -> bool:
    """Probe once whether the GPU encoder actually works on this machine.

    Listing the encoder is not proof it runs - QSV is present in most FFmpeg
    builds but fails without a matching Intel iGPU and driver, so this
    encodes a throwaway frame instead of trusting `-encoders`.
    """
    global _hardware_encoder_ok
    if _hardware_encoder_ok is not None:
        return _hardware_encoder_ok

    if not config.PREFER_HARDWARE_ENCODER:
        _hardware_encoder_ok = False
        return False

    probe = [
        ffmpeg_tools.tool_path("ffmpeg"),
        "-hide_banner",
        "-loglevel",
        "error",
        "-f",
        "lavfi",
        "-i",
        "color=c=black:s=256x256:d=0.1",
        "-c:v",
        config.HARDWARE_ENCODER,
        "-f",
        "null",
        "-",
    ]
    result = subprocess.run(probe, capture_output=True, text=True, check=False)
    _hardware_encoder_ok = result.returncode == 0

    if not _hardware_encoder_ok:
        print(f"    {config.HARDWARE_ENCODER} unavailable, using libx264")

    return _hardware_encoder_ok


def _encoder_args() -> list[str]:
    if hardware_encoder_available():
        return [
            "-c:v",
            config.HARDWARE_ENCODER,
            "-global_quality",
            str(config.HARDWARE_QUALITY),
        ]
    return [
        "-c:v",
        "libx264",
        "-preset",
        config.VIDEO_PRESET,
        "-crf",
        str(config.VIDEO_CRF),
    ]


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


def _build_filter(
    info: VideoInfo,
    subtitle_name: str,
    sendcmd_name: str | None,
) -> str:
    """Filter chain: crop to 9:16, scale, then burn in the captions.

    Every file is referenced by bare name; FFmpeg runs with cwd set to their
    folder so Windows drive letters never reach the filtergraph parser, where
    a colon separates arguments.
    """
    crop_w, crop_h, x, y = compute_crop(info)

    stages = []
    if sendcmd_name:
        # sendcmd rewrites crop's x as the clip plays, panning the window.
        stages.append(f"sendcmd=f={sendcmd_name}")
    stages.append(f"crop={crop_w}:{crop_h}:{x}:{y}")
    stages.append(f"scale={config.TARGET_WIDTH}:{config.TARGET_HEIGHT}:flags=lanczos")
    stages.append("setsar=1")
    stages.append(f"subtitles={subtitle_name}")
    return ",".join(stages)


def render_part(
    source: Path,
    part: ClipPart,
    info: VideoInfo,
    output_path: Path,
    crop_path: list[tracker.CropKeyframe] | None = None,
) -> Path:
    """Render one part of a clip as a captioned vertical video."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="clipcaptioner_") as tmp:
        work_dir = Path(tmp)
        subtitle_name = "subs.ass"
        ass_writer.write_ass(part.groups, work_dir / subtitle_name)

        sendcmd_name = None
        if crop_path:
            # Rebase onto this part's timeline; input seeking zeroes the clock.
            local = [
                tracker.CropKeyframe(time_s=key.time_s - part.start_s, x=key.x)
                for key in crop_path
                if part.start_s - 1.0 <= key.time_s <= part.end_s
            ]
            local = [
                tracker.CropKeyframe(time_s=max(0.0, key.time_s), x=key.x) for key in local
            ]
            if local:
                sendcmd_name = "crop.cmd"
                tracker.write_sendcmd(local, work_dir / sendcmd_name)

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
            _build_filter(info, subtitle_name, sendcmd_name),
            *_encoder_args(),
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
