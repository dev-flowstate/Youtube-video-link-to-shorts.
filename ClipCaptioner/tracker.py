"""Follow the speaker's face so the 9:16 crop frames them, not the centre.

The output is a dense, continuous crop path rather than a handful of
positions. Three things make the motion read as smooth:

* faces are sampled several times a second, so the path has real detail
* gaps are interpolated and the path is filtered forwards *and* backwards,
  which smooths without adding the lag a one-way filter would
* the result is resampled to a fine time grid, so FFmpeg's crop moves in
  steps too small to see

Scene cuts are detected and treated as hard boundaries: the crop repositions
instantly across a cut, because sliding through one looks like a mistake.
"""

from __future__ import annotations

import statistics
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


@dataclass
class _Sample:
    """One inspected frame."""

    time_s: float
    center_x: float | None  # source pixels, None when no face was found
    starts_shot: bool = False


# ---------------------------------------------------------------------------
# Sampling and detection
# ---------------------------------------------------------------------------


def _extract_frames(
    video: Path,
    work_dir: Path,
    width: int = config.TRACKING_FRAME_WIDTH,
) -> list[tuple[float, Path]]:
    """Write downscaled sample frames at a fixed rate.

    The width is an argument rather than read straight from config because the
    camera hunt needs a bigger one than the crop tracker - see
    _CAMERA_FRAME_WIDTH.
    """
    fps = max(0.5, config.TRACKING_SAMPLE_FPS)
    args = [
        ffmpeg_tools.tool_path("ffmpeg"),
        "-y",
        "-loglevel",
        "error",
        "-i",
        str(video),
        "-vf",
        f"fps={fps},scale={width}:-2",
        "-vsync",
        "0",
        "-q:v",
        "4",
        str(work_dir / "f_%06d.jpg"),
    ]
    ffmpeg_tools.run(args)

    frames = sorted(work_dir.glob("f_*.jpg"))
    if not frames:
        raise TrackingUnavailable("No sample frames were produced")

    step = 1.0 / fps
    return [(index * step, path) for index, path in enumerate(frames)]


def _inspect_frames(samples: list[tuple[float, Path]], info: VideoInfo) -> list[_Sample]:
    """Detect the subject's face and flag scene cuts, in one pass."""
    try:
        import cv2
        import numpy as np
    except ImportError as exc:
        raise TrackingUnavailable("opencv-python is not installed") from exc

    if not _MODEL_PATH.exists():
        raise TrackingUnavailable(f"Face model missing: {_MODEL_PATH.name}")

    detector = None
    previous_thumb = None
    previous_center: float | None = None
    results: list[_Sample] = []

    for time_s, frame_path in samples:
        image = cv2.imread(str(frame_path))
        if image is None:
            continue

        height, width = image.shape[:2]

        # Cut detection: compare a tiny greyscale thumbnail against the last
        # frame. Cheap, and robust enough since cuts change nearly every pixel.
        thumb = cv2.cvtColor(cv2.resize(image, (64, 36)), cv2.COLOR_BGR2GRAY).astype("float32")
        starts_shot = previous_thumb is None or (
            float(np.mean(np.abs(thumb - previous_thumb))) >= config.TRACKING_CUT_THRESHOLD
        )
        previous_thumb = thumb

        if detector is None:
            detector = cv2.FaceDetectorYN.create(
                str(_MODEL_PATH),
                "",
                (width, height),
                score_threshold=config.TRACKING_MIN_SCORE,
            )

        _, faces = detector.detect(image)
        center_x = None
        if faces is not None and len(faces) > 0:
            scale = info.width / width
            # After a cut there is no framing to stay near, so size decides.
            anchor = None if starts_shot else previous_center
            best = _choose_face(faces, anchor, scale, info.width)
            center_x = (float(best[0]) + float(best[2]) / 2.0) * scale

        if center_x is not None:
            previous_center = center_x

        results.append(_Sample(time_s=time_s, center_x=center_x, starts_shot=starts_shot))

    return results


def _choose_face(faces, anchor: float | None, scale: float, source_width: int):
    """Pick which face the crop should follow.

    Choosing the largest face makes two people at similar distance swap the
    frame back and forth, because tiny changes flip which is momentarily
    bigger. Staying with whoever is nearest the current framing keeps the
    shot on one person; size only decides when there is nothing to stay near.
    """
    largest = max(faces, key=lambda face: float(face[2]) * float(face[3]))
    if anchor is None:
        return largest

    def distance(face) -> float:
        centre = (float(face[0]) + float(face[2]) / 2.0) * scale
        return abs(centre - anchor)

    nearest = min(faces, key=distance)

    # A face far outside the current framing is a different subject, not drift.
    if distance(nearest) > config.TRACKING_MAX_JUMP_FRACTION * source_width:
        return largest
    return nearest


# ---------------------------------------------------------------------------
# Path building
# ---------------------------------------------------------------------------


def _split_shots(samples: list[_Sample]) -> list[list[_Sample]]:
    """Group samples into shots, so smoothing never crosses a cut."""
    shots: list[list[_Sample]] = []
    current: list[_Sample] = []

    for sample in samples:
        if sample.starts_shot and current:
            shots.append(current)
            current = []
        current.append(sample)

    if current:
        shots.append(current)
    return _merge_short_shots(shots)


def _merge_short_shots(shots: list[list[_Sample]]) -> list[list[_Sample]]:
    """Fold very short shots into their neighbour.

    Handheld camera movement trips the cut detector, producing runs of
    one-sample "shots". Each of those would become an instant jump, which is
    exactly the stutter this is meant to avoid. Real cuts are rarely this
    close together, so anything brief is treated as motion, not an edit.
    """
    if not shots:
        return shots

    merged: list[list[_Sample]] = [shots[0]]
    for shot in shots[1:]:
        duration = shot[-1].time_s - shot[0].time_s
        span = 1.0 / max(0.5, config.TRACKING_SAMPLE_FPS)
        if duration + span < config.TRACKING_MIN_SHOT_S:
            merged[-1].extend(shot)
        else:
            merged.append(shot)

    return merged


def _fill_gaps(shot: list[_Sample], fallback: float) -> list[float]:
    """Linearly interpolate across frames where no face was found.

    Detected positions are copied into a fully populated list up front rather
    than filling holes in place. Every slot then provably holds a number, so
    the arithmetic below needs no None handling and the return type is honest.
    """
    detected = [sample.center_x for sample in shot]
    known = [index for index, value in enumerate(detected) if value is not None]
    if not known:
        return [fallback] * len(detected)

    values: list[float] = [fallback] * len(detected)
    for index in known:
        found = detected[index]
        assert found is not None  # by construction of `known`
        values[index] = float(found)

    # Hold the first and last known values out to the shot's edges.
    for index in range(known[0]):
        values[index] = values[known[0]]
    for index in range(known[-1] + 1, len(values)):
        values[index] = values[known[-1]]

    for left, right in zip(known, known[1:]):
        span = right - left
        if span <= 1:
            continue
        start_value = values[left]
        end_value = values[right]
        for offset in range(1, span):
            ratio = offset / span
            values[left + offset] = start_value + (end_value - start_value) * ratio

    return values


def _reject_outliers(values: list[float], info: VideoInfo) -> list[float]:
    """Drop detections far from the running position - usually bystanders."""
    limit = config.TRACKING_MAX_JUMP_FRACTION * info.width
    cleaned: list[float] = []
    anchor = values[0]

    for value in values:
        if abs(value - anchor) > limit:
            value = anchor
        cleaned.append(value)
        anchor = value

    return cleaned


def _clamp_speed(values: list[float], dt: float) -> list[float]:
    """Cap how far the target may move between samples."""
    limit = config.TRACKING_MAX_PAN_PX_PER_S * dt
    clamped = [values[0]]

    for value in values[1:]:
        delta = value - clamped[-1]
        if abs(delta) > limit:
            value = clamped[-1] + (limit if delta > 0 else -limit)
        clamped.append(value)

    return clamped


def _smooth_zero_phase(values: list[float], dt: float) -> list[float]:
    """Moving average applied forwards then backwards, so it adds no lag.

    A one-way filter is what makes tracking feel like it is dragging behind
    the subject; running it in both directions cancels that out.
    """
    if len(values) < 3:
        return list(values)

    window = max(1, int(round(config.TRACKING_SMOOTHING_WINDOW_S / max(dt, 1e-6))))
    if window < 2:
        return list(values)

    def moving_average(series: list[float]) -> list[float]:
        out: list[float] = []
        half = window // 2
        for index in range(len(series)):
            start = max(0, index - half)
            end = min(len(series), index + half + 1)
            window_slice = series[start:end]
            out.append(sum(window_slice) / len(window_slice))
        return out

    forward = moving_average(values)
    backward = list(reversed(moving_average(list(reversed(forward)))))
    return backward


def _apply_deadzone(values: list[float]) -> list[float]:
    """Hold position through micro-movement so the frame is not always drifting."""
    if not values:
        return values

    held = [values[0]]
    for value in values[1:]:
        if abs(value - held[-1]) < config.TRACKING_DEADZONE_PX:
            held.append(held[-1])
        else:
            held.append(value)
    return held


def _resample(
    times: list[float],
    values: list[float],
    start_s: float,
    end_s: float,
) -> list[tuple[float, float]]:
    """Interpolate the path onto a fine, evenly spaced grid."""
    if not times:
        return []

    step = 1.0 / max(1.0, config.TRACKING_OUTPUT_FPS)
    out: list[tuple[float, float]] = []
    cursor = 0

    time_s = start_s
    while time_s <= end_s + 1e-6:
        while cursor + 1 < len(times) and times[cursor + 1] < time_s:
            cursor += 1

        if cursor + 1 >= len(times) or time_s <= times[0]:
            value = values[min(cursor, len(values) - 1)] if time_s >= times[0] else values[0]
        else:
            span = times[cursor + 1] - times[cursor]
            ratio = 0.0 if span <= 0 else (time_s - times[cursor]) / span
            ratio = min(1.0, max(0.0, ratio))
            value = values[cursor] + (values[cursor + 1] - values[cursor]) * ratio

        out.append((time_s, value))
        time_s += step

    return out


def build_crop_path(video: Path, info: VideoInfo, crop_width: int) -> list[CropKeyframe]:
    """Compute a smooth crop path for a clip, or raise TrackingUnavailable."""
    with tempfile.TemporaryDirectory(prefix="clipcaptioner_track_") as tmp:
        work_dir = Path(tmp)
        frames = _extract_frames(video, work_dir)
        samples = _inspect_frames(frames, info)

    if not samples:
        raise TrackingUnavailable("No frames could be inspected")

    if not any(sample.center_x is not None for sample in samples):
        raise TrackingUnavailable("No faces detected")

    dt = 1.0 / max(0.5, config.TRACKING_SAMPLE_FPS)
    half = crop_width / 2.0
    max_x = max(0, info.width - crop_width)
    fallback = info.width / 2.0

    points: list[tuple[float, float]] = []
    carried = fallback

    for shot in _split_shots(samples):
        # A shot with no face holds the previous framing rather than snapping
        # back to the centre of the frame and out again.
        values = _fill_gaps(shot, carried)
        values = _reject_outliers(values, info)
        values = _clamp_speed(values, dt)
        # Deadzone before smoothing: afterwards it would reintroduce the very
        # steps the smoothing just removed.
        values = _apply_deadzone(values)
        values = _smooth_zero_phase(values, dt)
        carried = values[-1]

        times = [sample.time_s for sample in shot]
        # Resample within the shot only, so the jump at a cut stays instant.
        points.extend(_resample(times, values, times[0], times[-1]))

    path: list[CropKeyframe] = []
    previous_x: int | None = None
    for time_s, center in points:
        x = int(round(center - half))
        x = max(0, min(max_x, x))
        # Skip repeats - a static shot needs one command, not hundreds.
        if previous_x is not None and x == previous_x:
            continue
        path.append(CropKeyframe(time_s=time_s, x=x))
        previous_x = x

    if not path:
        raise TrackingUnavailable("Crop path came out empty")

    return path


def write_sendcmd(path: list[CropKeyframe], destination: Path) -> Path:
    """Write an FFmpeg sendcmd script that pans the crop window."""
    lines = [f"{keyframe.time_s:.3f} crop x {keyframe.x};\n" for keyframe in path]
    destination.write_text("".join(lines), encoding="utf-8")
    return destination


# ---------------------------------------------------------------------------
# Finding the streamer's camera
# ---------------------------------------------------------------------------

# Detection width for the camera hunt, in place of TRACKING_FRAME_WIDTH.
#
# 480 is the right width for the crop tracker, whose subject is a head filling
# a fair share of the frame. A streamer's camera is a picture-in-picture box
# and the face inside it is far smaller: on the PMWC broadcast it measures
# about 40 px across a 1920-wide frame, which is 10 px at 480 and well under
# what YuNet will find. Measured over five clips of that broadcast, the share
# of sampled frames holding a face ran 0.00-0.35 at 480 against 0.40-0.98 at
# 1280, and only from 960 up did the median position agree between clips of
# the same layout - at 480 four of the five clips would have been abandoned as
# having no camera at all. This costs about twice the detection time, which is
# affordable because the pass runs once per clip.
_CAMERA_FRAME_WIDTH = 1280


@dataclass(frozen=True)
class CameraBox:
    """The streamer's camera as fractions of frame width and height.

    Normalised rather than pixels so one measurement survives a rescale: the
    box is found on downscaled sample frames and applied to the full-size
    source.
    """

    x: float
    y: float
    width: float
    height: float

    def crop(self, width: int, height: int) -> tuple[int, int, int, int]:
        """This box as an FFmpeg crop (w, h, x, y), clamped inside the frame."""
        crop_w = max(2, min(width, int(round(self.width * width))))
        crop_h = max(2, min(height, int(round(self.height * height))))
        x = max(0, min(width - crop_w, int(round(self.x * width))))
        y = max(0, min(height - crop_h, int(round(self.y * height))))
        return crop_w, crop_h, x, y


def _sample_faces(
    samples: list[tuple[float, Path]],
) -> list[tuple[float, float, float, float] | None]:
    """Largest face per frame, normalised: centre x, centre y, width, height.

    Normalised on the way out, so the caller never has to know what size the
    sample frames happened to be. One entry per frame, None where nothing was
    found, because the share of frames carrying a face is itself the signal
    that decides whether there is a camera worth enlarging.
    """
    try:
        import cv2
    except ImportError as exc:
        raise TrackingUnavailable("opencv-python is not installed") from exc

    if not _MODEL_PATH.exists():
        raise TrackingUnavailable(f"Face model missing: {_MODEL_PATH.name}")

    detector = None
    found: list[tuple[float, float, float, float] | None] = []

    for _time_s, frame_path in samples:
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
            found.append(None)
            continue

        # A camera holds one face. Where the broadcast cuts to a crowd or a
        # caster desk the frame holds several, and the largest is the nearest
        # the lens; the median across the clip then discards those shots
        # anyway, since they are a minority of a stream.
        best = max(faces, key=lambda face: float(face[2]) * float(face[3]))
        found.append(
            (
                (float(best[0]) + float(best[2]) / 2.0) / width,
                (float(best[1]) + float(best[3]) / 2.0) / height,
                float(best[2]) / width,
                float(best[3]) / height,
            )
        )

    return found


def find_camera_box(video: Path, info: VideoInfo) -> CameraBox | None:
    """Locate the streamer's camera from where faces actually appear.

    The camera sits somewhere different in every streamer's layout, so a fixed
    corner is guesswork. What is reliable is that the face inside it stays put
    for the whole clip, which makes the clip-wide median a far better
    description of the box than any single frame.

    Median rather than mean throughout: one crowd shot, a poster on the wall or
    a false positive on a game character drags a mean across the frame and
    leaves a median where it was.

    Returns None unless a face lands in the same place for at least
    STACKED_MIN_FACE_RATE of the sampled frames. There is no camera to enlarge
    below that, and enlarging a guess fills half the screen with a patch of
    wall.
    """
    with tempfile.TemporaryDirectory(prefix="clipcaptioner_camera_") as tmp:
        work_dir = Path(tmp)
        frames = _extract_frames(video, work_dir, width=_CAMERA_FRAME_WIDTH)
        faces = _sample_faces(frames)

    if not faces:
        raise TrackingUnavailable("No frames could be inspected")

    detected = [face for face in faces if face is not None]
    if not detected:
        return None

    centre_x = statistics.median(face[0] for face in detected)
    centre_y = statistics.median(face[1] for face in detected)
    face_width = statistics.median(face[2] for face in detected)
    face_height = statistics.median(face[3] for face in detected)

    # Room around the face for hair and shoulders, applied to both sides of the
    # detection rather than to its width alone.
    #
    # Padding the width and taking the height from the aspect ratio looks like
    # the same thing and is not. YuNet's box runs brow to chin and is about
    # 1.25x taller than it is wide, so at STACKED_CAMERA_PADDING 1.9 that route
    # leaves 0.09 of a face width above the head - 3 px on the PMWC webcam.
    # Rendered out it is a mugshot with the hair cut off the top and the chin
    # off the bottom, which is the exact thing the padding exists to prevent.
    box_width = face_width * config.STACKED_CAMERA_PADDING
    box_height = face_height * config.STACKED_CAMERA_PADDING

    # Now shape it to the panel. These fractions are of two different lengths,
    # so the frame's own shape has to come out of the sum: a 0.1-wide box on a
    # 16:9 frame is 0.178 of the height, not 0.1. Whichever side is short grows
    # to meet the aspect and neither shrinks, so both paddings survive.
    frame_aspect = info.width / info.height
    if box_width * frame_aspect / box_height > config.STACKED_CAMERA_ASPECT:
        box_height = box_width * frame_aspect / config.STACKED_CAMERA_ASPECT
    else:
        box_width = box_height * config.STACKED_CAMERA_ASPECT / frame_aspect

    # A camera tucked into a corner pushes its padded box off the frame. Both
    # sides shrink together rather than whichever overflowed, so the panel
    # keeps the shape it is about to be scaled into.
    overflow = max(1.0, box_width, box_height)
    box_width /= overflow
    box_height /= overflow

    box = CameraBox(
        x=max(0.0, min(1.0 - box_width, centre_x - box_width / 2.0)),
        y=max(0.0, min(1.0 - box_height, centre_y - box_height / 2.0)),
        width=box_width,
        height=box_height,
    )

    # Counting detections anywhere asks only whether something face-shaped
    # turned up, not whether it kept turning up in the same place. On a PMWC
    # clip carrying no webcam at all, false positives scattered over vehicles
    # and scenery cleared 25% between them and the median landed on a car in
    # mid-frame - the filtergraph runs and enlarges the wrong thing, which is
    # the failure worth guarding against. Counting only the frames whose face
    # lands inside the box makes the test say what it means: a face is in this
    # one place for at least STACKED_MIN_FACE_RATE of the clip.
    agreeing = sum(
        1
        for face in detected
        if box.x <= face[0] <= box.x + box.width
        and box.y <= face[1] <= box.y + box.height
    )
    if agreeing < config.STACKED_MIN_FACE_RATE * len(faces):
        return None

    return box
