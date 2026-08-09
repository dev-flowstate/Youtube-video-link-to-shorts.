"""Find fights by looking at the video instead of listening to it.

Audio was tried first and measured against known fights on a real broadcast.
It does not work: the game is mixed so far under the casters that gunfire
barely reaches the mix, and the wrongly-picked studio segments scored 2.4x
*higher* on gunfire onsets than the real firefights did. Five candidate
features - low-frequency energy, spectral flatness, crest factor, onset rate,
low-frequency transient rate - all came out flat between the two.

The difference is obvious on screen and invisible in the waveform, so this
looks at frames. Keyframes are pulled out (nearly free, and they land on the
broadcast's own scene cuts), tiled into grids, and Gemini says what each tile
is showing. The result is a timeline of gameplay, combat and studio talk.
"""

from __future__ import annotations

import math
import os
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

import config
import shared  # noqa: F401  (puts the sibling project on sys.path)
from ffmpeg_utils import ensure_ffmpeg_available

ENV_KEY = "GEMINI_API_KEY"

# One label per sampled frame.
FIGHT = "FIGHT"
GAME = "GAME"
STUDIO = "STUDIO"
UNKNOWN = "UNKNOWN"

_VALID = {FIGHT, GAME, STUDIO}

_PROMPT = """Each tile below is one frame from a PUBG Mobile esports broadcast,
numbered {first} to {last} in reading order (left to right, top to bottom).

Label every tile with exactly one word:

FIGHT  - live gameplay with combat visibly happening right now: muzzle flash,
         bullet tracers, a hit marker or damage number, a knock or a kill
         notification, an explosion, smoke or a grenade going off.
GAME   - live gameplay with no combat visible in this frame: looting, driving,
         running, rotating, waiting in the zone, an empty map view. A player
         crouched behind cover, aiming, or holding an angle is GAME unless a
         shot is actually landing or being fired in the frame.
STUDIO - anything that is not live gameplay: casters, an analysis desk, an
         interview, the crowd, a replay board, standings, a trophy, an advert,
         a transition or a logo.

Reply with one line per tile and nothing else, in this exact form:

{first}: WORD
"""


class VisionUnavailable(Exception):
    """Raised when frames cannot be classified."""


class DailyQuotaExhausted(VisionUnavailable):
    """The key has no requests left today.

    Kept separate from an ordinary rate limit because the two need opposite
    responses. A per-minute limit clears on its own in under a minute and is
    worth waiting for; a daily one does not clear until midnight Pacific, so
    waiting just burns an hour to arrive at the same place.
    """


@dataclass(frozen=True)
class Labelled:
    """What was on screen at a moment in the video."""

    time_s: float
    label: str

    @property
    def is_fight(self) -> bool:
        return self.label == FIGHT

    @property
    def is_gameplay(self) -> bool:
        return self.label in (FIGHT, GAME)


def available() -> bool:
    """Whether frames can be classified at all."""
    if not os.environ.get(ENV_KEY):
        return False
    try:
        from google import genai  # noqa: F401
    except ImportError:
        return False
    return True


def check_ready() -> None:
    """Confirm the key works and has requests left, before anything expensive.

    Detection cannot start until the source is downloaded, and that is ten-odd
    gigabytes and the best part of an hour. Finding out afterwards that the
    day's quota was already gone wastes all of it, so one cheap call settles the
    question first.
    """
    if not available():
        raise VisionUnavailable(
            f"{ENV_KEY} is not set, or google-genai is not installed."
        )

    from google import genai

    client = genai.Client(api_key=os.environ[ENV_KEY])
    try:
        client.models.generate_content(model=config.VISION_MODEL, contents="OK")
    except Exception as exc:
        if _is_daily_limit(exc):
            raise DailyQuotaExhausted(
                "This key has no requests left today. The free-tier daily quota "
                "resets at midnight Pacific."
            ) from exc
        if _is_rate_limit(exc):
            return  # a per-minute limit clears on its own; not a reason to stop
        raise VisionUnavailable(f"{type(exc).__name__}: {exc}") from exc


def _keyframe_times(video: Path) -> list[float]:
    args = [
        "ffprobe", "-v", "error", "-select_streams", "v:0", "-skip_frame", "nokey",
        "-show_entries", "frame=pts_time", "-of", "csv=p=0", str(video),
    ]
    result = subprocess.run(args, capture_output=True, text=True, check=False)
    times = []
    for line in result.stdout.splitlines():
        value = line.strip().rstrip(",")
        try:
            times.append(float(value))
        except ValueError:
            continue
    return times


def sample_frames(video: Path, work_dir: Path) -> list[tuple[float, Path]]:
    """Pull out keyframes, thinned to roughly one per SAMPLE_SECONDS.

    Keyframes cost almost nothing to decode - the encoder already made them
    independent - and a broadcast puts one at every scene cut, which is exactly
    where it switches between the game and the desk.
    """
    ensure_ffmpeg_available()

    times = _keyframe_times(video)
    if not times:
        raise VisionUnavailable("No keyframes found in the source.")

    args = [
        "ffmpeg", "-y", "-v", "error", "-skip_frame", "nokey", "-i", str(video),
        "-vf", f"scale={config.VISION_FRAME_WIDTH}:-2", "-vsync", "0", "-q:v", "5",
        str(work_dir / "k_%06d.jpg"),
    ]
    subprocess.run(args, check=False)

    frames = sorted(work_dir.glob("k_*.jpg"))
    if not frames:
        raise VisionUnavailable("No frames could be extracted.")

    # The timestamps come from one pass and the images from another, paired by
    # position. The two agree in practice - both ask for exactly the keyframes -
    # but if they ever did not, every frame after the disagreement would carry
    # its neighbour's timestamp and the clips would be cut at the wrong moments.
    # A wrong clip is worse than no clip, so this refuses rather than guesses.
    if len(frames) != len(times):
        raise VisionUnavailable(
            f"Found {len(times)} keyframe timestamps but {len(frames)} frames. "
            "Timestamps could not be matched to images reliably."
        )

    paired = list(zip(times, frames))

    # Thin to the requested spacing so a densely-keyframed source does not
    # multiply the number of calls.
    kept: list[tuple[float, Path]] = []
    last = -1e9
    for t, p in paired:
        if t - last >= config.VISION_SAMPLE_SECONDS:
            kept.append((t, p))
            last = t
    return kept


def _grid(images: list[Path], destination: Path, columns: int) -> Path:
    """Tile frames into one image so a single call covers many moments."""
    rows = math.ceil(len(images) / columns)
    layout = "|".join(
        f"{'0' if c == 0 else '+'.join(f'w{i}' for i in range(c))}_"
        f"{'0' if r == 0 else '+'.join(f'h{i * columns}' for i in range(r))}"
        for r in range(rows)
        for c in range(columns)
        if r * columns + c < len(images)
    )

    args = ["ffmpeg", "-y", "-v", "error"]
    for image in images:
        args += ["-i", str(image)]

    count = len(images)
    if count == 1:
        args += ["-c", "copy", str(destination)]
    else:
        args += ["-filter_complex", f"xstack=inputs={count}:layout={layout}:fill=black",
                 str(destination)]

    subprocess.run(args, check=False)
    if not destination.exists():
        raise VisionUnavailable("Could not build a frame grid.")
    return destination


def _parse(text: str, first: int, count: int) -> dict[int, str]:
    labels: dict[int, str] = {}
    for line in (text or "").splitlines():
        if ":" not in line:
            continue
        head, _, tail = line.partition(":")
        try:
            index = int(head.strip())
        except ValueError:
            continue
        word = tail.strip().upper().split()[0] if tail.strip() else ""
        if index in range(first, first + count) and word in _VALID:
            labels[index] = word
    return labels


def _is_rate_limit(exc: Exception) -> bool:
    text = str(exc)
    return "429" in text or "RESOURCE_EXHAUSTED" in text


def _is_daily_limit(exc: Exception) -> bool:
    """Whether the quota that ran out was the per-day one.

    Google names the quota in the error, e.g.
    GenerateRequestsPerDayPerProjectPerModel-FreeTier.
    """
    return _is_rate_limit(exc) and "PerDay" in str(exc)


def _ask(client, types, prompt: str, image: Path):
    """One call, retried through the quota.

    A free-tier key runs out of requests per minute long before it runs out of
    anything else, and the quota comes back on its own within about a minute.
    Waiting is the correct response; giving up would leave a stretch of the
    broadcast unlabelled, and unlabelled reads as "no fight here".
    """
    delay = config.VISION_RETRY_SECONDS

    for attempt in range(config.VISION_MAX_RETRIES + 1):
        try:
            return client.models.generate_content(
                model=config.VISION_MODEL,
                contents=[
                    prompt,
                    types.Part.from_bytes(
                        data=image.read_bytes(), mime_type="image/jpeg"
                    ),
                ],
            )
        except Exception as exc:
            if _is_daily_limit(exc):
                raise DailyQuotaExhausted(str(exc)) from exc
            if not _is_rate_limit(exc) or attempt == config.VISION_MAX_RETRIES:
                raise
            time.sleep(delay)
            delay *= 2

    raise VisionUnavailable("Ran out of retries.")


def classify(frames: list[tuple[float, Path]]) -> list[Labelled]:
    """Ask Gemini what is on screen in each sampled frame."""
    if not available():
        raise VisionUnavailable(
            f"{ENV_KEY} is not set, or google-genai is not installed."
        )

    from google import genai
    from google.genai import types

    client = genai.Client(api_key=os.environ[ENV_KEY])
    columns = config.VISION_GRID_COLUMNS
    per_call = columns * config.VISION_GRID_ROWS

    # Held below the quota rather than sprinting into it and backing off, which
    # is both faster overall and quieter.
    interval = 60.0 / max(1.0, config.VISION_REQUESTS_PER_MINUTE)
    last_call = 0.0

    out: list[Labelled] = []
    failures: list[str] = []
    ran_dry = False

    with tempfile.TemporaryDirectory(prefix="vision_grid_") as tmp:
        grid_dir = Path(tmp)
        total = math.ceil(len(frames) / per_call)

        for batch_no, start in enumerate(range(0, len(frames), per_call), start=1):
            batch = frames[start : start + per_call]
            first = start + 1

            if ran_dry:
                # Nothing left to spend, so stop asking. The remaining frames
                # are recorded unlabelled and the check below decides whether
                # what did come back is worth anything.
                for time_s, _path in batch:
                    out.append(Labelled(time_s, UNKNOWN))
                continue

            grid = _grid([p for _, p in batch], grid_dir / f"g{batch_no}.jpg", columns)

            pause = interval - (time.monotonic() - last_call)
            if pause > 0:
                time.sleep(pause)
            last_call = time.monotonic()

            try:
                response = _ask(
                    client,
                    types,
                    _PROMPT.format(first=first, last=first + len(batch) - 1),
                    grid,
                )
                labels = _parse(getattr(response, "text", "") or "", first, len(batch))
            except DailyQuotaExhausted:
                print(f"    out of requests for today at batch {batch_no}/{total}")
                ran_dry = True
                labels = {}
            except Exception as exc:
                # A failed batch leaves its frames unknown rather than ending
                # the run; one bad call should not cost the whole broadcast.
                failures.append(f"batch {batch_no}: {type(exc).__name__} {exc}"[:160])
                labels = {}

            for offset, (time_s, _path) in enumerate(batch):
                out.append(Labelled(time_s, labels.get(first + offset, UNKNOWN)))

            if batch_no % 10 == 0 or batch_no == total:
                print(f"    classified {batch_no}/{total} batches")

    unknown = sum(1 for item in out if item.label == UNKNOWN)
    if failures:
        print(f"    {len(failures)} batch(es) failed, {unknown} frame(s) unlabelled:")
        for note in failures[:3]:
            print(f"      {note}")
        if len(failures) > 3:
            print(f"      ... and {len(failures) - 3} more")

    # An unlabelled frame reads as "no fight here", so a run that mostly failed
    # would quietly report a broadcast with no fights in it. Say so instead.
    if out and unknown > len(out) * config.VISION_MAX_UNLABELLED:
        done = len(out) - unknown
        if ran_dry:
            raise DailyQuotaExhausted(
                f"The key ran out of requests after {done} of {len(out)} frames. "
                f"Free-tier daily quota is small and resets at midnight Pacific; "
                f"enable billing, or raise VISION_SAMPLE_SECONDS to look less often."
            )
        raise VisionUnavailable(
            f"{unknown} of {len(out)} frames could not be labelled. "
            "The results would not be trustworthy, so nothing was clipped."
        )

    return out


def fight_windows(labelled: list[Labelled]) -> list[tuple[float, float]]:
    """Merge neighbouring FIGHT frames into continuous windows.

    A fight is rarely one frame, and the sampling is coarse enough that a real
    engagement shows up as a run. Gaps shorter than the bridge are closed so a
    single mislabelled frame mid-fight does not split one engagement in two.
    """
    hits = [item.time_s for item in labelled if item.is_fight]
    if not hits:
        return []

    bridge = config.VISION_BRIDGE_SECONDS
    windows: list[list[float]] = [[hits[0], hits[0]]]
    for t in hits[1:]:
        if t - windows[-1][1] <= bridge:
            windows[-1][1] = t
        else:
            windows.append([t, t])

    return [(a, b) for a, b in windows]


def detect(video: Path) -> list[Labelled]:
    """Sample the video and label every sampled moment."""
    with tempfile.TemporaryDirectory(prefix="vision_frames_") as tmp:
        frames = sample_frames(video, Path(tmp))

        per_call = config.VISION_GRID_COLUMNS * config.VISION_GRID_ROWS
        calls = math.ceil(len(frames) / per_call)
        minutes = calls / max(1.0, config.VISION_REQUESTS_PER_MINUTE)
        print(
            f"    sampled {len(frames)} frames -> {calls} request(s), "
            f"about {minutes:.0f} min at the current rate"
        )

        return classify(frames)
