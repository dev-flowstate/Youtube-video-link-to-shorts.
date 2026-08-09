"""Per-frame vision features for the fight detector.

Audio was tried first and is a documented failure on this footage: the game is
mixed so far under the casters that studio segments scored 2.4x *higher* on
gunfire than the real firefights did. The difference between a firefight, quiet
rotation and an interview is on the screen, so this measures the screen.

One sequential FFmpeg pass produces every feature. Frames arrive as raw BGR on
a pipe at `fps` per second and 960 wide; nothing is written to disk, because a
five hour broadcast is roughly 40,000 frames and JPEG-ing them costs more than
the whole analysis.

Timestamps are arithmetic - `t = index / fps` - and never come from a second
pass. Pairing a decode against a separate ffprobe timeline is the single most
expensive bug available here: a container with a non-zero start time shifts one
timeline against the other by a constant, and every training label afterwards
lands on the wrong row without anything looking wrong.

Four rules the features follow:

* NaN is the only missing marker. Zero means "measured, and it was zero" - a
  zero kill notice count says there were no kills, a NaN says the strip could
  not be read. Zero-filling the second invents evidence.
* Nothing keys on stored bitmaps. The overlay is detected as *structure that
  persists while the world moves*, because a template keys the model to one
  tournament's art and the held-out broadcast deliberately carries a different
  one (a streamer facecam).
* Regions come from hud_layout, in fractions of the frame, never in pixels.
* HSV, greyscale and the bright/fire masks are computed once per frame and
  shared between groups.
"""

from __future__ import annotations

import json
import subprocess
import tempfile
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from scipy.ndimage import maximum_filter1d
from tqdm import tqdm

import hud_layout
import shared  # noqa: F401  (puts the sibling project on sys.path)
from ffmpeg_utils import ensure_ffmpeg_available

# ---------------------------------------------------------------------------
# Decoding
# ---------------------------------------------------------------------------

# Frames are scaled to this width. hud_layout is in fractions, so the number is
# free to move, but it sets how small the smallest thing anything here counts
# gets: at 960 a team-bar tick zone is 43x23 px and a damage digit about 5x12.
# Going lower loses the ticks; going higher costs decode time for detail
# nothing reads.
FRAME_WIDTH = 960

# Height is halved as well, so a frame is a quarter of the pixels of 1080p.
_EDGE_SCALE = 2

# ---------------------------------------------------------------------------
# Colour thresholds
# ---------------------------------------------------------------------------
#
# OpenCV packs hue into 0-179 (degrees halved) and S/V into 0-255, so every
# constant below is the fraction from the brief times 255, and every hue is the
# angle halved.

# The muzzle-flash primitive: near-white and blown out. 0.92 * 255 and
# 0.25 * 255.
_BRIGHT_V = 235
_BRIGHT_S = 64

# Molotov, blast and vehicle fire: 5-35 degrees becomes 3-17 after halving
# (the nearest whole steps, so 6-34 degrees in practice), 0.55 and 0.60 of 255.
_FIRE_H_LO, _FIRE_H_HI = 3, 17
_FIRE_S = 141
_FIRE_V = 153

# Canny on 0-255. Loose enough to keep overlay hairlines, tight enough that
# grass texture does not become a wall of edges.
_CANNY_LO, _CANNY_HI = 60, 160

# ---------------------------------------------------------------------------
# HUD presence
# ---------------------------------------------------------------------------

# How far back a region is compared with itself. The overlay is by definition
# what stays identical while the world beneath it moves, and two seconds is
# long enough that even a slow observer pan has changed the world underneath.
PERSISTENCE_LAG_SECONDS = 2.0

# Persistence alone is not enough - a locked-off studio wide shot also holds
# still - so each score is persistence times how much crisp structure the
# region contains. Measured on five frames of the Group A reel at 960 wide:
# the minimap runs 0.20-0.29 edge density and the top-left stats strip
# 0.16-0.21, while a stretch of live world in the same box sits near 0.01-0.09.
# Anything past this counts as fully structured.
_HUD_EDGE_REFERENCE = 0.06

# The overlay is pixel-locked frame to frame, but AV1 and h264 shift a stroke
# by a pixel now and then, so the earlier edge map is dilated by one before
# intersecting. Without it a static minimap scores about half what it should.
_EDGE_JITTER = 3

# Below this, the game overlay is not on screen and Group C cannot be trusted.
_HUD_PRESENT_FLOOR = 0.15

# Below this the team bar is covered (a lower third, a replay wipe, or a
# production that simply is not showing it) and Group E is unreadable.
_TEAM_BAR_FLOOR = 0.15

# A standings table is a large flat panel pasted over the world, so what marks
# it is pixels that do not change at all rather than edges that persist. Six
# levels out of 255 is above compression noise and below any real content.
_STILL_TOLERANCE = 6

# ---------------------------------------------------------------------------
# Scene cuts
# ---------------------------------------------------------------------------

# Total-variation distance between consecutive luma histograms, in 0-1. A pan
# moves a few percent of the mass; a cut from the desk to the game moves most
# of it.
_SCENE_CUT_DISTANCE = 0.35

# ---------------------------------------------------------------------------
# Damage numbers
# ---------------------------------------------------------------------------
#
# Not OCR - only "are floating damage numbers present".
#
# A damage number measured off the reel is a 20x10 px group of white digits on
# a translucent plate, and two things separate it from the rest of a bright
# frame. It stands well above its immediate surroundings, which a cloud or a
# sun-glint does not, because those *are* their surroundings. And it is short:
# digits are counted as whole words, so a nametag or a playzone warning merges
# into something far wider than any three-digit number.
#
# The plate cannot be found by an absolute threshold - it is translucent, so
# over smoke it reads about 150, not 40. Local contrast is what survives that.

_DMG_LOCAL_WINDOW = 0.016      # of frame width; a little wider than a digit
_DMG_LOCAL_LIFT = 50           # levels a glyph must stand above its ground
_DMG_MERGE = 0.0031            # of frame width; bridges digits, not words

# Word size, as fractions of frame width and height. At 960x540 that is 6-38 px
# wide and 6-18 px tall, which brackets every damage number measured and
# excludes every nametag.
_DMG_MIN_H, _DMG_MAX_H = 0.011, 0.033
_DMG_MIN_W, _DMG_MAX_W = 0.006, 0.040

# Strokes, not a solid lump, but a merged word is denser than a single digit.
_DMG_MIN_FILL, _DMG_MAX_FILL = 0.30, 0.95

# ---------------------------------------------------------------------------
# Kill and knock notices
# ---------------------------------------------------------------------------
#
# Verified on a real notice off the Group A reel: raw pixel change in this
# strip is dominated by camera motion, so it is not used. A notice is a dark
# semi-transparent plate carrying bright glyphs, which camera motion does not
# produce, and that is what is counted.

_KILLFEED_DARK = 90            # the plate, which is translucent, not black
_KILLFEED_BRIGHT = 175         # the glyphs on it
_KILLFEED_DARK_FRAC = 0.40     # most of a notice row is plate

# A row with no bright pixels is just a shadow; a row that is nearly all bright
# is sky. A notice row sits between.
_KILLFEED_INK_LO, _KILLFEED_INK_HI = 0.02, 0.40

# A notice is a box, not a scanline. Measured height of the one verified notice
# was 23 px at 960 wide, of which the glyph rows were about half.
_KILLFEED_MIN_ROWS = 0.011     # of frame height

KILLFEED_LAG_SECONDS = 1.0

# A notice appearing is a rising edge, held between two levels. Differencing
# alone counted one notice several times over: measured on the Group A reel,
# the ink under a single banner wobbles 0.29, 0.30, 0.19, 0.11, 0.08 as the
# camera moves behind the translucent plate, and every wobble past a plain
# threshold looked like another kill.
_KILLFEED_ON = 0.05
_KILLFEED_OFF = 0.02

# Notices land at the *end* of an exchange, so a trailing window would mark the
# aftermath and score the run-up at zero - the same mistake
# CASTER_LOOKAHEAD_SECONDS exists to avoid in the audio path. Centred instead.
KILLFEED_WINDOWS = (10.0, 30.0)

# ---------------------------------------------------------------------------
# Alive ticks
# ---------------------------------------------------------------------------
#
# Measured off a real team bar: green ticks come out at hue 79-85 (158-170
# degrees, a mint green) with S about 140 and V about 200, and grey ticks at
# S under 25 with V 133-165. The bands below are wider than those readings so a
# re-skin toward a pure green still lands inside, but stop short of hue 100,
# where sky begins.

_TICK_MIN_S, _TICK_MIN_V = 90, 90
_GREEN_H_LO, _GREEN_H_HI = 45, 95
_RED_H_LO, _RED_H_HI = 0, 8
_RED_WRAP_LO, _RED_WRAP_HI = 172, 179

# Grey ticks are dimmer than the bar's own white lettering, which is what the
# upper bound keeps out.
_GREY_MAX_S = 55
_GREY_V_LO, _GREY_V_HI = 100, 210

# A tick is a thin slash most of the height of its zone.
_TICK_MIN_HEIGHT = 0.30        # of the tick zone's height
_TICK_MAX_WIDTH = 0.40         # of the tick zone's width
_TICK_MIN_AREA = 4             # px

# ---------------------------------------------------------------------------
# Windows
# ---------------------------------------------------------------------------

MAX2_SECONDS = 2.0             # the "+/- 2s" of the _max2 features
HUD_MEAN_SECONDS = 30.0
ALIVE_DELTA_SECONDS = 30.0

FEATURE_NAMES: tuple[str, ...] = (
    # Group A - motion and light, whole frame with the overlay masked out.
    "luma_mean",
    "luma_p99",
    "bright_frac",
    "bright_frac_d",
    "bright_frac_max2",
    "fire_frac",
    "fire_frac_d",
    "fire_frac_max2",
    "frame_diff",
    "frame_diff_p95",
    "scene_cut",
    "edge_density",
    # Group B - whether the game overlay is on screen at all.
    "hud_minimap_score",
    "hud_topleft_score",
    "hud_teambar_score",
    "hud_present",
    "hud_present_frac30",
    "standings_overlay_present",
    # Group C - floating damage numbers.
    "dmg_blobs",
    "dmg_blobs_max2",
    # Group D - kill and knock notices.
    "killfeed_ink",
    "killfeed_change",
    "killfeed_events_10s",
    "killfeed_events_30s",
    # Group E - alive ticks in the team bar.
    "alive_ticks_total",
    "alive_teams",
    "alive_ticks_d30",
    "knocked_ticks",
)

# What one frame yields before any windowing. Everything in FEATURE_NAMES is
# either one of these or derived from them.
_RAW_NAMES: tuple[str, ...] = (
    "luma_mean",
    "luma_p99",
    "bright_frac",
    "fire_frac",
    "frame_diff",
    "frame_diff_p95",
    "hist_distance",
    "edge_density",
    "hud_minimap_score",
    "hud_topleft_score",
    "hud_teambar_score",
    "standings_overlay_present",
    "dmg_blobs",
    "killfeed_ink",
    "alive_ticks_total",
    "alive_teams",
    "knocked_ticks",
)

_LEVELS = np.arange(256, dtype=np.float64)

# The order the three presence scores are reported in. Named rather than
# positional so a rename in hud_layout stops the run instead of quietly
# scoring the minimap as the team bar.
_PRESENCE_ORDER = ("minimap", "topleft", "teambar")
_TEAMBAR = _PRESENCE_ORDER.index("teambar")


class FeatureExtractionFailed(Exception):
    """Raised when the video could not be decoded or analysed."""


# ---------------------------------------------------------------------------
# Geometry, resolved once per run
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _Layout:
    """Every rectangle and mask hud_layout implies, in pixels, computed once."""

    width: int
    height: int
    world: np.ndarray
    world_pixels: int
    world_small: np.ndarray
    world_small_pixels: int
    small_size: tuple[int, int]
    presence: tuple[tuple[int, int, int, int], ...]
    killfeed: tuple[int, int, int, int]
    damage: tuple[int, int, int, int]
    standings: tuple[int, int, int, int]
    tick_zones: tuple[tuple[int, int, int, int], ...]
    word_h: tuple[int, int]
    word_w: tuple[int, int]
    banner_min_rows: int
    dmg_window: int
    dmg_merge: np.ndarray

    @classmethod
    def build(cls, width: int, height: int) -> _Layout:
        if set(_PRESENCE_ORDER) != set(hud_layout.PRESENCE_REGIONS):
            raise FeatureExtractionFailed(
                f"hud_layout.PRESENCE_REGIONS is {sorted(hud_layout.PRESENCE_REGIONS)}, "
                f"but the features are named for {sorted(_PRESENCE_ORDER)}."
            )

        world = np.full((height, width), 255, dtype=np.uint8)
        for region in hud_layout.OVERLAY_REGIONS:
            x0, y0, x1, y1 = region.pixels(width, height)
            world[y0:y1, x0:x1] = 0

        small = (width // _EDGE_SCALE, height // _EDGE_SCALE)
        world_small = cv2.resize(world, small, interpolation=cv2.INTER_NEAREST)

        bar = hud_layout.TEAM_BAR.pixels(width, height)
        slots = hud_layout.TEAM_BAR_SLOTS
        slot_w = (bar[2] - bar[0]) / slots
        lo, hi = hud_layout.TEAM_SLOT_TICKS
        zones = tuple(
            (
                int(round(bar[0] + slot_w * (s + lo))),
                bar[1],
                int(round(bar[0] + slot_w * (s + hi))),
                bar[3],
            )
            for s in range(slots)
        )

        return cls(
            width=width,
            height=height,
            world=world,
            world_pixels=int(cv2.countNonZero(world)),
            world_small=world_small,
            world_small_pixels=int(cv2.countNonZero(world_small)),
            small_size=small,
            presence=tuple(
                hud_layout.PRESENCE_REGIONS[name].pixels(width, height)
                for name in _PRESENCE_ORDER
            ),
            killfeed=hud_layout.KILLFEED_STRIP.pixels(width, height),
            damage=hud_layout.DAMAGE_ZONE.pixels(width, height),
            standings=hud_layout.STANDINGS_OVERLAY.pixels(width, height),
            tick_zones=zones,
            word_h=(max(2, int(_DMG_MIN_H * height)), max(3, int(_DMG_MAX_H * height))),
            word_w=(max(1, int(_DMG_MIN_W * width)), max(3, int(_DMG_MAX_W * width))),
            banner_min_rows=max(3, int(_KILLFEED_MIN_ROWS * height)),
            dmg_window=max(3, int(_DMG_LOCAL_WINDOW * width)),
            dmg_merge=np.ones((1, max(2, int(_DMG_MERGE * width))), np.uint8),
        )


# ---------------------------------------------------------------------------
# FFmpeg
# ---------------------------------------------------------------------------


def _decode_args(
    video: Path, fps: float, max_seconds: float | None, frames: int | None = None
) -> list[str]:
    """One filter chain, shared by the size probe and the real pass.

    The two must agree exactly: the probe's job is to say how many bytes a
    frame is, and it can only answer for the chain it actually ran.
    """
    args = ["ffmpeg", "-v", "error", "-nostdin"]
    if max_seconds is not None:
        # Before -i, so FFmpeg stops reading the container rather than
        # decoding five hours and discarding it.
        args += ["-t", f"{max_seconds:.3f}"]
    args += [
        "-i", str(video),
        "-an", "-sn",
        "-vf", f"fps={fps},scale={FRAME_WIDTH}:-2",
    ]
    if frames is not None:
        args += ["-frames:v", str(frames)]
    args += ["-f", "rawvideo", "-pix_fmt", "bgr24", "-"]
    return args


def _frame_height(video: Path, fps: float) -> int:
    """Decode one frame and measure it, rather than predicting the scaler.

    `scale=960:-2` rounds to an even height, and how it gets there depends on
    the sample aspect ratio the container declares. Guessing wrong would not
    raise anything - the byte stream would simply be read at the wrong stride
    and every frame after the first would come out sheared.
    """
    result = subprocess.run(
        _decode_args(video, fps, None, frames=1), capture_output=True, check=False
    )
    row = FRAME_WIDTH * 3
    if result.returncode != 0 or not result.stdout:
        detail = result.stderr.decode("utf-8", "replace").strip()[-300:]
        raise FeatureExtractionFailed(f"Could not decode {video.name}: {detail}")
    if len(result.stdout) % row:
        raise FeatureExtractionFailed(
            f"{video.name}: first frame is {len(result.stdout)} bytes, which is "
            f"not a whole number of {FRAME_WIDTH}px BGR rows."
        )
    return len(result.stdout) // row


def _expected_frames(video: Path, fps: float, max_seconds: float | None) -> int | None:
    """Roughly how many frames the pass will yield, for the progress bar only.

    Deliberately not used for anything else. This number comes from the
    container's own duration and the timestamps do not - mixing the two is how
    a constant offset gets into the labels.
    """
    args = [
        "ffprobe", "-v", "error", "-show_entries", "format=duration",
        "-of", "json", str(video),
    ]
    result = subprocess.run(args, capture_output=True, text=True, check=False)
    try:
        duration = float(json.loads(result.stdout)["format"]["duration"])
    except (ValueError, KeyError, TypeError, json.JSONDecodeError):
        return None
    if max_seconds is not None:
        duration = min(duration, max_seconds)
    return max(1, int(duration * fps))


def _read_exact(stream, view: memoryview) -> bool:
    """Fill the whole buffer, or report the stream ended."""
    got = 0
    size = len(view)
    while got < size:
        read = stream.readinto(view[got:])
        if not read:
            return False
        got += read
    return True


# ---------------------------------------------------------------------------
# Per-frame measurement
# ---------------------------------------------------------------------------


class _Running:
    """What one frame needs to know about the frames before it."""

    def __init__(self, lag: int) -> None:
        self.previous_gray: np.ndarray | None = None
        self.previous_hist: np.ndarray | None = None
        # Long enough that index 0 is exactly `lag` frames back once full.
        self.edges: deque[tuple[np.ndarray, ...]] = deque(maxlen=lag + 1)
        self.standings: deque[np.ndarray] = deque(maxlen=lag + 1)
        self.lag = lag


def _masked_stats(values: np.ndarray, layout: _Layout) -> tuple[float, float, np.ndarray]:
    """Mean, 99th percentile and the histogram, over the world only.

    A histogram rather than np.percentile because a percentile has to partition
    half a million pixels per frame and this does not - and the histogram is
    wanted anyway, for the scene cut.
    """
    hist = cv2.calcHist([values], [0], layout.world, [256], [0, 256]).ravel()
    total = layout.world_pixels
    mean = float(hist @ _LEVELS) / total
    p99 = float(np.searchsorted(np.cumsum(hist), 0.99 * total))
    return mean / 255.0, p99 / 255.0, hist / total


def _persistence(now: np.ndarray, before: np.ndarray) -> float:
    """How much of this region's edge structure survived the lag, in 0-1.

    Intersection over union of the two edge maps, with the older one dilated by
    a pixel so compression jitter does not read as movement. Empty on both
    sides scores 0: no structure is not the same as a stationary overlay.
    """
    union = cv2.countNonZero(cv2.bitwise_or(now, before))
    if union == 0:
        return 0.0
    grown = cv2.dilate(before, np.ones((_EDGE_JITTER, _EDGE_JITTER), np.uint8))
    return cv2.countNonZero(cv2.bitwise_and(now, grown)) / union


def _structure(edges: np.ndarray) -> float:
    """How much crisp detail a region holds, saturating at overlay levels."""
    density = cv2.countNonZero(edges) / edges.size
    return min(1.0, density / _HUD_EDGE_REFERENCE)


def _damage_blobs(bright: np.ndarray, fire: np.ndarray, gray: np.ndarray,
                  layout: _Layout) -> int:
    """Count short bright words standing above their own local background."""
    x0, y0, x1, y1 = layout.damage
    glyphs = cv2.bitwise_or(bright[y0:y1, x0:x1], fire[y0:y1, x0:x1])

    # Only glyphs that stand out from what is immediately around them. A cloud
    # is bright and so is everything next to it, so it lifts by nothing.
    patch = np.ascontiguousarray(gray[y0:y1, x0:x1])
    ground = cv2.blur(patch, (layout.dmg_window, layout.dmg_window))
    lifted = cv2.inRange(cv2.subtract(patch, ground), _DMG_LOCAL_LIFT, 255)

    # Merge digits into words so a number counts once and a nametag comes out
    # as one over-wide blob rather than several digit-sized ones.
    words = cv2.dilate(cv2.bitwise_and(glyphs, lifted), layout.dmg_merge)

    count, _labels, stats, _centroids = cv2.connectedComponentsWithStats(
        words, connectivity=8
    )
    if count <= 1:
        return 0

    widths = stats[1:, cv2.CC_STAT_WIDTH]
    heights = stats[1:, cv2.CC_STAT_HEIGHT]
    areas = stats[1:, cv2.CC_STAT_AREA]
    fill = areas / np.maximum(widths * heights, 1)
    keep = (
        (heights >= layout.word_h[0]) & (heights <= layout.word_h[1])
        & (widths >= layout.word_w[0]) & (widths <= layout.word_w[1])
        & (fill >= _DMG_MIN_FILL) & (fill <= _DMG_MAX_FILL)
    )
    return int(np.count_nonzero(keep))


def _killfeed_ink(gray: np.ndarray, layout: _Layout) -> float:
    """What share of the strip is covered by notice banners.

    A notice row is mostly dark plate carrying a minority of bright glyphs, and
    it belongs to a run of such rows - a notice is a box. A single qualifying
    row is a nametag or a shadow edge and is dropped.
    """
    x0, y0, x1, y1 = layout.killfeed
    strip = gray[y0:y1, x0:x1]
    dark = (strip < _KILLFEED_DARK).mean(axis=1)
    ink = (strip > _KILLFEED_BRIGHT).mean(axis=1)

    row_ok = (
        (dark >= _KILLFEED_DARK_FRAC)
        & (ink >= _KILLFEED_INK_LO)
        & (ink <= _KILLFEED_INK_HI)
    )
    if not row_ok.any():
        return 0.0

    # Keep only runs at least a banner tall.
    padded = np.concatenate(([False], row_ok, [False]))
    edges = np.flatnonzero(padded[1:] != padded[:-1])
    starts, ends = edges[0::2], edges[1::2]
    long_enough = (ends - starts) >= layout.banner_min_rows
    return float((ends - starts)[long_enough].sum()) / len(row_ok)


def _count_ticks(mask: np.ndarray, min_height: int, max_width: int) -> int:
    count, _labels, stats, _centroids = cv2.connectedComponentsWithStats(
        mask, connectivity=8
    )
    if count <= 1:
        return 0
    keep = (
        (stats[1:, cv2.CC_STAT_AREA] >= _TICK_MIN_AREA)
        & (stats[1:, cv2.CC_STAT_HEIGHT] >= min_height)
        & (stats[1:, cv2.CC_STAT_WIDTH] <= max_width)
    )
    return int(np.count_nonzero(keep))


def _alive_ticks(hsv: np.ndarray, layout: _Layout) -> tuple[int, int, int]:
    """Alive ticks, teams still holding one, and knocked ticks.

    Green means the team is in frame, grey means it is alive somewhere else,
    red means a knocked player - so alive is green plus grey, and red is the
    knock count.
    """
    alive = 0
    teams = 0
    knocked = 0

    for x0, y0, x1, y1 in layout.tick_zones:
        zone = np.ascontiguousarray(hsv[y0:y1, x0:x1])
        min_height = max(2, int(_TICK_MIN_HEIGHT * (y1 - y0)))
        max_width = max(2, int(_TICK_MAX_WIDTH * (x1 - x0)))

        green = cv2.inRange(
            zone, (_GREEN_H_LO, _TICK_MIN_S, _TICK_MIN_V), (_GREEN_H_HI, 255, 255)
        )
        red = cv2.bitwise_or(
            cv2.inRange(
                zone, (_RED_H_LO, _TICK_MIN_S, _TICK_MIN_V), (_RED_H_HI, 255, 255)
            ),
            cv2.inRange(
                zone,
                (_RED_WRAP_LO, _TICK_MIN_S, _TICK_MIN_V),
                (_RED_WRAP_HI, 255, 255),
            ),
        )
        grey = cv2.inRange(zone, (0, 0, _GREY_V_LO), (179, _GREY_MAX_S, _GREY_V_HI))

        lit = _count_ticks(green, min_height, max_width) + _count_ticks(
            grey, min_height, max_width
        )
        alive += lit
        teams += 1 if lit else 0
        knocked += _count_ticks(red, min_height, max_width)

    return alive, teams, knocked


def _measure(frame: np.ndarray, layout: _Layout, state: _Running) -> tuple[float, ...]:
    """Everything one frame can say for itself, in _RAW_NAMES order."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Group A -------------------------------------------------------------
    luma_mean, luma_p99, hist = _masked_stats(gray, layout)

    bright = cv2.inRange(hsv, (0, 0, _BRIGHT_V), (179, _BRIGHT_S, 255))
    fire = cv2.inRange(hsv, (_FIRE_H_LO, _FIRE_S, _FIRE_V), (_FIRE_H_HI, 255, 255))
    bright_frac = (
        cv2.countNonZero(cv2.bitwise_and(bright, layout.world)) / layout.world_pixels
    )
    fire_frac = (
        cv2.countNonZero(cv2.bitwise_and(fire, layout.world)) / layout.world_pixels
    )

    if state.previous_gray is None:
        # No predecessor, so the change since the last frame is unknown rather
        # than zero.
        frame_diff = np.nan
        frame_diff_p95 = np.nan
        hist_distance = np.nan
    else:
        delta = cv2.absdiff(gray, state.previous_gray)
        diff_hist = cv2.calcHist([delta], [0], layout.world, [256], [0, 256]).ravel()
        frame_diff = float(diff_hist @ _LEVELS) / layout.world_pixels / 255.0
        frame_diff_p95 = (
            float(np.searchsorted(np.cumsum(diff_hist), 0.95 * layout.world_pixels))
            / 255.0
        )
        hist_distance = 0.5 * float(np.abs(hist - state.previous_hist).sum())

    small = cv2.resize(gray, layout.small_size, interpolation=cv2.INTER_AREA)
    world_edges = cv2.bitwise_and(
        cv2.Canny(small, _CANNY_LO, _CANNY_HI), layout.world_small
    )
    edge_density = cv2.countNonZero(world_edges) / layout.world_small_pixels

    # Group B -------------------------------------------------------------
    now_edges = tuple(
        cv2.Canny(np.ascontiguousarray(gray[y0:y1, x0:x1]), _CANNY_LO, _CANNY_HI)
        for x0, y0, x1, y1 in layout.presence
    )
    sx0, sy0, sx1, sy1 = layout.standings
    standings_now = np.ascontiguousarray(gray[sy0:sy1, sx0:sx1])

    state.edges.append(now_edges)
    state.standings.append(standings_now)

    if len(state.edges) <= state.lag:
        # Nothing to compare against yet. Two seconds of a five hour broadcast.
        scores = [np.nan] * len(now_edges)
        standings_score = np.nan
    else:
        older = state.edges[0]
        scores = [
            _persistence(now, before) * _structure(now)
            for now, before in zip(now_edges, older)
        ]
        still = cv2.absdiff(standings_now, state.standings[0]) <= _STILL_TOLERANCE
        standings_edges = cv2.Canny(standings_now, _CANNY_LO, _CANNY_HI)
        standings_score = float(still.mean()) * _structure(standings_edges)

    hud_present = float(np.median(scores))

    # Group C -------------------------------------------------------------
    if not np.isfinite(hud_present) or hud_present < _HUD_PRESENT_FLOOR:
        # Without the overlay there is no telling whether nothing was hit or
        # the shot is simply not of the game.
        dmg = np.nan
    else:
        dmg = float(_damage_blobs(bright, fire, gray, layout))

    # Group D -------------------------------------------------------------
    killfeed_ink = _killfeed_ink(gray, layout)

    # Group E -------------------------------------------------------------
    teambar_score = scores[_TEAMBAR]
    if not np.isfinite(teambar_score) or teambar_score < _TEAM_BAR_FLOOR:
        alive, teams, knocked = np.nan, np.nan, np.nan
    else:
        alive, teams, knocked = _alive_ticks(hsv, layout)

    state.previous_gray = gray
    state.previous_hist = hist

    return (
        luma_mean, luma_p99, bright_frac, fire_frac,
        frame_diff, frame_diff_p95, hist_distance, edge_density,
        scores[0], scores[1], scores[2], standings_score,
        dmg, killfeed_ink, alive, teams, knocked,
    )


# ---------------------------------------------------------------------------
# Windowed features
# ---------------------------------------------------------------------------


def _lagged_delta(values: np.ndarray, lag: int) -> np.ndarray:
    """values[i] - values[i - lag], NaN where there is no earlier value."""
    out = np.full(len(values), np.nan, dtype=np.float64)
    if 0 < lag < len(values):
        out[lag:] = values[lag:] - values[:-lag]
    return out


def _rolling_max(values: np.ndarray, half: int) -> np.ndarray:
    """Centred maximum over +/- half samples, ignoring NaN."""
    filled = np.where(np.isnan(values), -np.inf, values)
    out = maximum_filter1d(filled, size=2 * half + 1, mode="nearest")
    return np.where(np.isneginf(out), np.nan, out)


def _window_bounds(count: int, half: int) -> tuple[np.ndarray, np.ndarray]:
    index = np.arange(count)
    return np.maximum(index - half, 0), np.minimum(index + half + 1, count)


def _rolling_nanmean(values: np.ndarray, half: int) -> np.ndarray:
    """Centred mean over +/- half samples, NaN only where the window is empty."""
    valid = ~np.isnan(values)
    totals = np.concatenate(([0.0], np.cumsum(np.where(valid, values, 0.0))))
    counts = np.concatenate(([0], np.cumsum(valid.astype(np.int64))))
    lo, hi = _window_bounds(len(values), half)
    seen = counts[hi] - counts[lo]
    out = np.full(len(values), np.nan, dtype=np.float64)
    np.divide(totals[hi] - totals[lo], seen, out=out, where=seen > 0)
    return out


def _rising_edges(values: np.ndarray, on: float, off: float) -> np.ndarray:
    """Mark where a banner appears, holding the verdict between two levels.

    One notice has to produce one event. A plain threshold on the change since
    a second ago produced four or five, because the ink under a single banner
    wobbles as the camera moves behind it.
    """
    flags = np.zeros(len(values), dtype=bool)
    present = False
    for i, value in enumerate(values):
        if not present and value >= on:
            present = True
            flags[i] = True
        elif present and value < off:
            present = False
    return flags


def _window_count(flags: np.ndarray, half: int) -> np.ndarray:
    """Centred count of True over +/- half samples."""
    running = np.concatenate(([0], np.cumsum(flags.astype(np.int64))))
    lo, hi = _window_bounds(len(flags), half)
    return (running[hi] - running[lo]).astype(np.float64)


def _assemble(raw: dict[str, np.ndarray], fps: float) -> np.ndarray:
    """Turn the per-frame measurements into the published feature matrix."""
    max2 = max(1, int(round(MAX2_SECONDS * fps)))
    hud_half = max(1, int(round(HUD_MEAN_SECONDS * fps / 2)))
    kill_lag = max(1, int(round(KILLFEED_LAG_SECONDS * fps)))
    alive_lag = max(1, int(round(ALIVE_DELTA_SECONDS * fps)))

    # The middle of the three, so a lower third covering one region cannot
    # sink the verdict on its own - two of the three still have to agree. The
    # three are unreadable together or not at all, so a plain median is right
    # here and nanmedian would only hide a bug.
    hud = np.median(
        np.vstack(
            (
                raw["hud_minimap_score"],
                raw["hud_topleft_score"],
                raw["hud_teambar_score"],
            )
        ),
        axis=0,
    )

    killfeed_change = _lagged_delta(raw["killfeed_ink"], kill_lag)
    appeared = _rising_edges(raw["killfeed_ink"], _KILLFEED_ON, _KILLFEED_OFF)

    columns = {
        "luma_mean": raw["luma_mean"],
        "luma_p99": raw["luma_p99"],
        "bright_frac": raw["bright_frac"],
        "bright_frac_d": _lagged_delta(raw["bright_frac"], 1),
        "bright_frac_max2": _rolling_max(raw["bright_frac"], max2),
        "fire_frac": raw["fire_frac"],
        "fire_frac_d": _lagged_delta(raw["fire_frac"], 1),
        "fire_frac_max2": _rolling_max(raw["fire_frac"], max2),
        "frame_diff": raw["frame_diff"],
        "frame_diff_p95": raw["frame_diff_p95"],
        "scene_cut": np.where(
            np.isnan(raw["hist_distance"]),
            np.nan,
            (raw["hist_distance"] > _SCENE_CUT_DISTANCE).astype(np.float64),
        ),
        "edge_density": raw["edge_density"],
        "hud_minimap_score": raw["hud_minimap_score"],
        "hud_topleft_score": raw["hud_topleft_score"],
        "hud_teambar_score": raw["hud_teambar_score"],
        "hud_present": hud,
        "hud_present_frac30": _rolling_nanmean(hud, hud_half),
        "standings_overlay_present": raw["standings_overlay_present"],
        "dmg_blobs": raw["dmg_blobs"],
        "dmg_blobs_max2": _rolling_max(raw["dmg_blobs"], max2),
        "killfeed_ink": raw["killfeed_ink"],
        "killfeed_change": killfeed_change,
        "killfeed_events_10s": _window_count(
            appeared, max(1, int(round(KILLFEED_WINDOWS[0] * fps / 2)))
        ),
        "killfeed_events_30s": _window_count(
            appeared, max(1, int(round(KILLFEED_WINDOWS[1] * fps / 2)))
        ),
        "alive_ticks_total": raw["alive_ticks_total"],
        "alive_teams": raw["alive_teams"],
        "alive_ticks_d30": _lagged_delta(raw["alive_ticks_total"], alive_lag),
        "knocked_ticks": raw["knocked_ticks"],
    }

    missing = set(FEATURE_NAMES) - set(columns)
    if missing:
        raise FeatureExtractionFailed(f"Features never computed: {sorted(missing)}")

    return np.column_stack([columns[name] for name in FEATURE_NAMES]).astype(np.float32)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def extract(
    video: Path, fps: float = 2.0, max_seconds: float | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Returns (times_s, values); values is (len(times), len(FEATURE_NAMES)) float32."""
    ensure_ffmpeg_available()

    height = _frame_height(video, fps)
    layout = _Layout.build(FRAME_WIDTH, height)
    expected = _expected_frames(video, fps, max_seconds)

    frame_bytes = FRAME_WIDTH * height * 3
    buffer = bytearray(frame_bytes)
    view = memoryview(buffer)
    frame = np.frombuffer(buffer, dtype=np.uint8).reshape(height, FRAME_WIDTH, 3)

    state = _Running(lag=max(1, int(round(PERSISTENCE_LAG_SECONDS * fps))))
    rows: list[tuple[float, ...]] = []

    print(
        f"    {video.name}: {FRAME_WIDTH}x{height} at {fps} fps"
        + (f", about {expected} frames" if expected else "")
    )

    # FFmpeg's own diagnostics go to a file rather than a pipe. A five hour
    # source with a few damaged packets can emit more than a pipe buffer holds,
    # and nothing is draining it until the loop below ends - which it never
    # would, because FFmpeg would be blocked writing.
    with tempfile.TemporaryFile() as log:
        process = subprocess.Popen(
            _decode_args(video, fps, max_seconds),
            stdout=subprocess.PIPE,
            stderr=log,
            bufsize=frame_bytes,
        )
        assert process.stdout is not None
        code = -1
        try:
            with tqdm(total=expected, unit="frame", desc="    frames") as progress:
                while _read_exact(process.stdout, view):
                    rows.append(_measure(frame, layout, state))
                    progress.update(1)
            code = process.wait()
        finally:
            if process.poll() is None:
                process.kill()

        if code != 0:
            log.seek(0)
            detail = log.read().decode("utf-8", "replace").strip()[-300:]
            raise FeatureExtractionFailed(f"FFmpeg failed on {video.name}: {detail}")

    if not rows:
        raise FeatureExtractionFailed(f"{video.name} produced no frames.")

    measured = np.asarray(rows, dtype=np.float64)
    raw = {name: measured[:, i] for i, name in enumerate(_RAW_NAMES)}
    values = _assemble(raw, fps)

    # Arithmetic, never a second ffprobe pass. See the module docstring.
    times = np.arange(len(rows), dtype=np.float64) / fps

    if values.shape != (len(times), len(FEATURE_NAMES)):
        raise FeatureExtractionFailed(
            f"Feature matrix is {values.shape}, expected "
            f"({len(times)}, {len(FEATURE_NAMES)})."
        )
    return times, values
