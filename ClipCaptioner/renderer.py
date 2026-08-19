"""Crop to 9:16, burn in captions, and encode the final clip."""

from __future__ import annotations

import hashlib
import re
import subprocess
import tempfile
from pathlib import Path

import ass_writer
import config
import ffmpeg_tools
import stock
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
    subtitle_name: str | None,
    sendcmd_name: str | None,
    camera: tracker.CameraBox | None = None,
) -> str:
    """Filter chain: crop to 9:16, scale, then burn in the captions.

    Every file is referenced by bare name; FFmpeg runs with cwd set to their
    folder so Windows drive letters never reach the filtergraph parser, where
    a colon separates arguments.

    subtitle_name is None when captions are turned off - the clip is still
    cropped, tracked and scaled, it simply carries no burned-in text.

    camera is the box the stacked layout enlarges, and is None when no camera
    could be found - see _camera_box.
    """
    tail = ["setsar=1"]
    if subtitle_name:
        tail.append(f"subtitles={subtitle_name}")

    if config.CROP_MODE == "stacked":
        # No camera means no stack. Falling back to "fit" keeps the whole
        # frame, where falling back to "crop" would throw two thirds of it
        # away to fix a problem the viewer does not have.
        if camera is None:
            return _fit_filter(tail)
        return _stacked_filter(info, camera, tail)

    if config.CROP_MODE == "fit":
        return _fit_filter(tail)

    crop_w, crop_h, x, y = compute_crop(info)

    stages = []
    if sendcmd_name:
        # sendcmd rewrites crop's x as the clip plays, panning the window.
        stages.append(f"sendcmd=f={sendcmd_name}")
    stages.append(f"crop={crop_w}:{crop_h}:{x}:{y}")
    stages.append(f"scale={config.TARGET_WIDTH}:{config.TARGET_HEIGHT}:flags=lanczos")
    stages.extend(tail)
    return ",".join(stages)


def _fit_filter(tail: list[str]) -> str:
    """Whole frame, centred, over a blurred copy of itself.

    Cropping gameplay to 9:16 discards the minimap, the kill feed and often
    the fight, so the frame is shrunk to fit the width instead. The blurred
    backdrop fills what would otherwise be black bars, which read as wasted
    space in a Shorts feed.

    The blur is done by shrinking and regrowing rather than with gblur, which
    is accurate and far too slow to run over every frame of every clip.
    """
    width, height = config.TARGET_WIDTH, config.TARGET_HEIGHT
    small = max(8, width // config.FIT_BACKDROP_BLUR)

    return (
        "split=2[bg][fg];"
        f"[bg]scale={width}:{height}:force_original_aspect_ratio=increase,"
        f"crop={width}:{height},scale={small}:-2,scale={width}:{height}:flags=bilinear[bgb];"
        f"[fg]scale={width}:-2:flags=lanczos[fgs];"
        "[bgb][fgs]overlay=(W-w)/2:(H-h)/2,"
        + ",".join(tail)
    )


def _stacked_filter(info: VideoInfo, camera: tracker.CameraBox, tail: list[str]) -> str:
    """The streamer's camera on top, the whole frame below.

    Cropping a stream to 9:16 leaves a face reacting to something the viewer
    cannot see, and fitting the whole frame leaves the action about a third of
    the screen tall. Stacking gives the reaction real size without losing the
    thing it is a reaction to.

    The camera panel is scaled to cover and then centre-cropped, so it fills
    its share exactly and carries no bars. The lower panel is the *whole*
    frame - the small original camera included, which is normal in this style
    of edit and not worth masking - over the same shrink-and-regrow blur
    _fit_filter uses, for the same reason: gblur over a whole clip is far too
    slow.
    """
    width = config.TARGET_WIDTH
    top = _even(int(round(config.TARGET_HEIGHT * config.STACKED_CAMERA_SHARE)))
    bottom = config.TARGET_HEIGHT - top
    small = max(8, width // config.FIT_BACKDROP_BLUR)

    crop_w, crop_h, x, y = camera.crop(info.width, info.height)
    crop_w, crop_h = _even(crop_w), _even(crop_h)

    return (
        "split=3[cam][bg][fg];"
        f"[cam]crop={crop_w}:{crop_h}:{x}:{y},"
        f"scale={width}:{top}:force_original_aspect_ratio=increase:flags=lanczos,"
        f"crop={width}:{top}[camera];"
        f"[bg]scale={width}:{bottom}:force_original_aspect_ratio=increase,"
        f"crop={width}:{bottom},scale={small}:-2,scale={width}:{bottom}:flags=bilinear[bgb];"
        f"[fg]scale={width}:-2:flags=lanczos[fgs];"
        "[bgb][fgs]overlay=(W-w)/2:(H-h)/2[screen];"
        "[camera][screen]vstack=inputs=2,"
        + ",".join(tail)
    )


def _filler_filter(
    info: VideoInfo,
    subtitle_name: str | None,
    sendcmd_name: str | None,
    camera: tracker.CameraBox | None,
) -> str:
    """The podcast on top, looping ambient filler footage below.

    Filler is peripheral - motion to hold a scroll, not something the viewer
    came for - so the split leans hard towards the podcast rather than the
    near-even one _stacked_filter gives a reacting streamer, where both
    halves are genuinely worth watching. See FILLER_PODCAST_SHARE.

    The podcast panel is exactly whatever _build_filter already produces for
    the chosen CROP_MODE - crop, fit or stacked - with no captions burned in
    yet, covered and centre-cropped down into its share of the height. That
    reuses the existing framing logic outright, rather than a third copy of
    it. The filler panel (input 1, a second physical file) gets the same
    cover-and-crop treatment, so it fills its strip with no bars of its own.
    Captions are burned once, after the two panels are stacked into the
    finished frame, so they land exactly where they do without filler.

    Filler shorter than the clip is looped by the caller with -stream_loop -1
    on its input, not here - vstack stops at the shorter of its two inputs,
    so the podcast (already trimmed by -t) still decides the output length.

    The filler is resampled to the podcast's own frame rate before stacking.
    Stock footage rarely shares the source's rate - 25fps against a 29.97fps
    source, say - and vstack combining two different rates hands the encoder
    a variable one, which h264_qsv refuses outright ("Current frame rate is
    unsupported"). Matching it keeps the composite as steady as an ordinary
    render.
    """
    width = config.TARGET_WIDTH
    top = _even(int(round(config.TARGET_HEIGHT * config.FILLER_PODCAST_SHARE)))
    bottom = config.TARGET_HEIGHT - top

    # No captions here - they belong on the finished, stacked frame below.
    podcast_chain = _build_filter(info, None, sendcmd_name, camera)

    tail = ["setsar=1"]
    if subtitle_name:
        tail.append(f"subtitles={subtitle_name}")

    return (
        f"[0:v]{podcast_chain}[podcast];"
        f"[podcast]scale={width}:{top}:force_original_aspect_ratio=increase,"
        f"crop={width}:{top}[podtop];"
        f"[1:v]fps={info.fps},scale={width}:{bottom}:force_original_aspect_ratio=increase,"
        f"crop={width}:{bottom}[fillbot];"
        "[podtop][fillbot]vstack=inputs=2," + ",".join(tail) + "[vout]"
    )


# Resolved once per run and reused for every clip: the choice does not depend
# on which clip is being rendered, and re-fetching per clip would repeat the
# same cache lookups stock.py already does on disk.
_filler_source: Path | None = None
_filler_resolved = False


def _filler_footage() -> Path | None:
    """One piece of ambient filler footage, or None if none could be had.

    Tries FILLER_TOPICS in order and stops at the first with results - a
    niche topic like "subway surfers gameplay" can come back empty on a stock
    site built for real-world footage, where the earlier, more generic topics
    almost always have something.

    Never raises: filler is a bonus on top of the normal render, and
    render_part falls back to the ordinary layout - never a black strip -
    when this comes back empty.
    """
    global _filler_source, _filler_resolved
    if _filler_resolved:
        return _filler_source

    _filler_resolved = True
    for topic in config.FILLER_TOPICS:
        found = stock.fetch(topic, want=1)
        if found:
            _filler_source = found[0]
            return _filler_source

    print(
        f"    no filler footage found for {', '.join(config.FILLER_TOPICS)} "
        "- using the normal layout"
    )
    return None


# Detection costs a frame extraction and a pass of the face detector over the
# whole file, and render_part runs once per part, so the answer is remembered.
_camera_boxes: dict[Path, tracker.CameraBox | None] = {}


def _camera_box(source: Path, info: VideoInfo) -> tracker.CameraBox | None:
    """Where this clip's camera sits, or None if it has no usable one.

    Never raises: a clip that cannot be stacked is still worth rendering, and
    _build_filter fits the whole frame instead. The reason is printed once,
    because a layout quietly turning into a different layout is exactly the
    kind of thing that goes unnoticed until the upload.
    """
    key = source.resolve()
    if key in _camera_boxes:
        return _camera_boxes[key]

    try:
        box = tracker.find_camera_box(source, info)
    except tracker.TrackingUnavailable as exc:
        print(f"    no camera found ({exc}) - fitting the whole frame instead")
        box = None
    else:
        if box is None:
            print(
                f"    no camera found (a face appears in under "
                f"{config.STACKED_MIN_FACE_RATE:.0%} of frames) - "
                f"fitting the whole frame instead"
            )
        else:
            print(
                f"    camera at {box.x:.3f},{box.y:.3f} "
                f"sized {box.width:.3f}x{box.height:.3f} of the frame"
            )

    _camera_boxes[key] = box
    return box


def render_part(
    source: Path,
    part: ClipPart,
    info: VideoInfo,
    output_path: Path,
    crop_path: list[tracker.CropKeyframe] | None = None,
    title: str | None = None,
    language: str | None = None,
) -> Path:
    """Render one part of a clip as a captioned vertical video."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Found from the source rather than the part: the camera does not move
    # between parts, and one detection pass over the file answers for all of
    # them.
    camera = _camera_box(source, info) if config.CROP_MODE == "stacked" else None

    filler_source = _filler_footage() if config.PARKOUR_FILLER else None

    with tempfile.TemporaryDirectory(prefix="clipcaptioner_") as tmp:
        work_dir = Path(tmp)

        subtitle_name = None
        if config.BURN_CAPTIONS:
            subtitle_name = "subs.ass"
            # The font follows the language: the default has Latin glyphs only.
            ass_writer.write_ass(part.groups, work_dir / subtitle_name, language)

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
        ]

        if filler_source is not None:
            # Looped indefinitely so shorter filler never freezes or cuts to
            # black. That leaves nothing to stop vstack on its own: it holds
            # the podcast's last frame once that (finite, -t above) input
            # ends rather than closing, so without a hard stop the output
            # would run forever against the looped filler. An explicit -t on
            # the *output* is that stop, independent of the input-side one.
            args += [
                "-stream_loop",
                "-1",
                "-i",
                str(filler_source),
                "-filter_complex",
                _filler_filter(info, subtitle_name, sendcmd_name, camera),
                "-map",
                "[vout]",
                "-map",
                "0:a",
                "-t",
                f"{part.duration_s:.3f}",
            ]
        else:
            args += ["-vf", _build_filter(info, subtitle_name, sendcmd_name, camera)]

        args += [
            *_encoder_args(),
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            "-b:a",
            config.AUDIO_BITRATE,
            "-movflags",
            "+faststart",
        ]

        # Stored in the container itself, so the title survives even if the
        # file is renamed on the way to an upload.
        if title:
            args += ["-metadata", f"title={title}"]

        args.append(str(output_path))
        ffmpeg_tools.run(args, cwd=work_dir)

    if not output_path.exists() or output_path.stat().st_size == 0:
        raise ffmpeg_tools.FFmpegError(f"Render produced nothing for {output_path.name}")

    return output_path


# Characters Windows will not accept in a filename.
_ILLEGAL = re.compile(r'[<>:"/\\|?*\x00-\x1f]')

# Source clips carry their position in the video, like "[20m40s-22m10s]".
_SOURCE_RANGE = re.compile(r"\[(\d+h)?\d+m\d+s-(\d+h)?\d+m\d+s\]")


def sanitize_filename(text: str, max_length: int = 120) -> str:
    """Make a title safe to use as a filename."""
    cleaned = _ILLEGAL.sub("", text)
    cleaned = re.sub(r"\s+", " ", cleaned).strip(" .")
    if len(cleaned) > max_length:
        cleaned = cleaned[:max_length].rsplit(" ", 1)[0].rstrip(" .")
    return cleaned or "clip"


def build_output_path(
    output_dir: Path,
    source: Path,
    part: ClipPart,
    title: str | None = None,
) -> Path:
    """Name the rendered file after its title.

    YouTube pre-fills the title field from the filename on upload, so naming
    the file properly saves retyping it for every clip.

    The source clip's timestamp range is kept on the end. Two different
    moments can score the same title, and without something unique the second
    would look already-rendered and be silently skipped.
    """
    suffix = ""

    if title:
        stem = sanitize_filename(title)
        # Two different moments can score the same title. Without something
        # tied to the source, the second would look already-rendered and be
        # skipped in silence. The timestamp range reads well and is already
        # unique; a short digest covers sources that lack one.
        match = _SOURCE_RANGE.search(source.stem)
        if match:
            suffix = f" {match.group(0)}"
        else:
            digest = hashlib.sha1(source.stem.encode("utf-8")).hexdigest()[:6]
            suffix = f" [{digest}]"
    else:
        stem = source.stem

    if part.total > 1:
        suffix += f" (part {part.index} of {part.total})"

    return output_dir / f"{stem}{suffix}.mp4"
