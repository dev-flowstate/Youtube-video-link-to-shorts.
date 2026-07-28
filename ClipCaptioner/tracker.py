"""Follow the speaker's face so the 9:16 crop frames them, not the centre.

Sampling is deliberately sparse. Decoding only keyframes costs almost nothing
and, because encoders place a keyframe at every scene cut, it puts a sample
exactly where the framing needs to change. The result is a list of crop
positions over time that gets handed to FFmpeg's sendcmd filter.
"""

from __future__ import annotations

import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

import config
import ffmpeg_tools
from models import VideoInfo

_MODEL_PATH = Path(__file__).resolve().parent / "models" / "face_detection_yunet_2023mar.onnx"


class TrackingUnavailable(Exception):
    """Raised when tracking cannot run, so the caller can centre-crop instead."""


@dataclass(frozen=True)
class CropKeyframe:
    """Where the crop window's left edge sits at a moment in time."""

    time_s: float
    x: int


def _keyframe_times(video: Path) -> list[float]:
    args = [
        ffmpeg_tools.tool_path("ffprobe"),
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-skip_frame",
        "nokey",
        "-show_entries",
        "frame=pts_time",
        "-of",
        "csv=p=0",
        str(video),
    ]
    result = subprocess.run(args, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        raise TrackingUnavailable(f"Could not read keyframes: {result.stderr.strip()[-300:]}")

    times: list[float] = []
    for line in result.stdout.splitlines():
        value = line.strip().rstrip(",")
        if not value:
            continue
        try:
            times.append(float(value))
        except ValueError:
            continue
    return times


def _extract_frames(video: Path, work_dir: Path) -> list[tuple[float, Path]]:
    """Write downscaled sample frames and pair each with its timestamp."""
    args = [ffmpeg_tools.tool_path("ffmpeg"), "-y", "-loglevel", "error"]

    if config.TRACKING_SAMPLE_MODE == "keyframes":
        times = _keyframe_times(video)
        args += ["-skip_frame", "nokey", "-i", str(video)]
        args += ["-vf", f"scale={config.TRACKING_FRAME_WIDTH}:-2"]
    else:
        fps = max(0.1, config.TRACKING_DENSE_FPS)
        args += ["-i", str(video)]
        args += ["-vf", f"fps={fps},scale={config.TRACKING_FRAME_WIDTH}:-2"]
        times = []

    args += ["-vsync", "0", "-q:v", "4", str(work_dir / "f_%06d.jpg")]
    ffmpeg_tools.run(args)

    frames = sorted(work_dir.glob("f_*.jpg"))
    if not frames:
        raise TrackingUnavailable("No sample frames were produced")

    if not times:
        step = 1.0 / max(0.1, config.TRACKING_DENSE_FPS)
        times = [index * step for index in range(len(frames))]

    # Frame extraction and the pts listing can disagree by one at the tail.
    count = min(len(frames), len(times))
    return list(zip(times[:count], frames[:count]))


def _detect_centers(samples: list[tuple[float, Path]], info: VideoInfo) -> list[tuple[float, float]]:
    """Return (time, face centre x in source pixels) for frames with a face."""
    try:
        import cv2
    except ImportError as exc:
        raise TrackingUnavailable("opencv-python is not installed") from exc

    if not _MODEL_PATH.exists():
        raise TrackingUnavailable(f"Face model missing: {_MODEL_PATH.name}")

    detector = None
    centers: list[tuple[float, float]] = []

    for time_s, frame_path in samples:
        image = cv2.imread(str(frame_path))
        if image is None:
            continue

        height, width = image.shape[:2]
        if detector is None:
            detector = cv2.FaceDetectorYN.create(
                str(_MODEL_PATH),
                "",
                (width, height),
                score_threshold=config.TRACKING_MIN_SCORE,
            )

        _, faces = detector.detect(image)
        if faces is None or len(faces) == 0:
            continue

        # Largest face wins - the speaker is nearer the camera than bystanders.
        best = max(faces, key=lambda face: float(face[2]) * float(face[3]))
        center_x_small = float(best[0]) + float(best[2]) / 2.0
        centers.append((time_s, center_x_small / width * info.width))

    return centers


def _smooth(
    centers: list[tuple[float, float]],
    info: VideoInfo,
    crop_width: int,
) -> list[CropKeyframe]:
    """Turn raw detections into a steady, speed-limited crop path."""
    half = crop_width / 2.0
    max_x = max(0, info.width - crop_width)
    default_center = info.width / 2.0

    smoothed: list[CropKeyframe] = []
    current = default_center
    previous_time: float | None = None
    max_jump = config.TRACKING_MAX_JUMP_FRACTION * info.width

    for time_s, target in centers:
        # A detection miles from the current framing is a bystander, not a cut.
        if abs(target - current) > max_jump:
            target = current

        blended = current + (target - current) * config.TRACKING_SMOOTHING

        if previous_time is not None:
            elapsed = max(1e-3, time_s - previous_time)
            limit = config.TRACKING_MAX_PAN_PX_PER_S * elapsed
            delta = blended - current
            if abs(delta) > limit:
                blended = current + (limit if delta > 0 else -limit)

        current = blended
        previous_time = time_s

        x = int(round(current - half))
        smoothed.append(CropKeyframe(time_s=time_s, x=max(0, min(max_x, x))))

    return smoothed


def build_crop_path(video: Path, info: VideoInfo, crop_width: int) -> list[CropKeyframe]:
    """Compute the crop path for a clip, or raise TrackingUnavailable."""
    with tempfile.TemporaryDirectory(prefix="clipcaptioner_track_") as tmp:
        work_dir = Path(tmp)
        samples = _extract_frames(video, work_dir)
        centers = _detect_centers(samples, info)

    if not centers:
        raise TrackingUnavailable("No faces detected")

    return _smooth(centers, info, crop_width)


def write_sendcmd(path: list[CropKeyframe], destination: Path) -> Path:
    """Write an FFmpeg sendcmd script that pans the crop window."""
    lines = [f"{keyframe.time_s:.3f} crop x {keyframe.x};\n" for keyframe in path]
    destination.write_text("".join(lines), encoding="utf-8")
    return destination
