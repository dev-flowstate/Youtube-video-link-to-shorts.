"""Collect training labels from the Gemini classifier, a day's quota at a time.

The classifier is good - 17 of 18 frames correct on hand-checked footage, and 12
of 12 studio frames correctly excluded - but a free key allows only a few
hundred requests a day, and a broadcast wants more than that. So collection has
to survive being stopped and resumed, which is what separates this from simply
calling `vision_detector.classify`.

Three differences from that function, each learned the hard way.

**Results are appended to disk as each batch returns.** `classify` gathers
everything in memory and, past a threshold of failures, raises - discarding the
whole run. For a detector that is right: half a broadcast labelled is not worth
clipping from. For label collection it is the opposite; every verdict already
paid for is worth keeping.

**Batches are sent in a shuffled order.** Frames stay grouped with their
neighbours inside a batch, because that is the arrangement the classifier was
measured on, but the batches themselves are shuffled. Run out of quota halfway
and you have a sample spread over the whole broadcast rather than a detailed
picture of its first hour and nothing after.

**Work already on disk is skipped.** Re-running tomorrow continues where
yesterday stopped instead of paying twice for the same frames.
"""

from __future__ import annotations

import json
import math
import random
import tempfile
import time
from pathlib import Path

import config
import shared  # noqa: F401  (puts the sibling project on sys.path)
import vision_detector
from vision_detector import DailyQuotaExhausted, VisionUnavailable


def label_file(source_id: str) -> Path:
    return config.DATA_DIR / "labels" / f"{source_id}.jsonl"


def already_done(path: Path) -> set[float]:
    """Timestamps already carrying a verdict, so quota is never spent twice."""
    if not path.exists():
        return set()

    done: set[float] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            done.add(round(float(json.loads(line)["time_s"]), 2))
        except (json.JSONDecodeError, KeyError, TypeError, ValueError):
            # A line torn in half by an interrupted write costs one frame, not
            # the file. Skipping it means it gets asked again, which is right.
            continue
    return done


def _append(path: Path, source_id: str, rows: list[tuple[float, str]]) -> None:
    """Write this batch's verdicts before asking for the next one."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for time_s, label in rows:
            handle.write(
                json.dumps({"source": source_id, "time_s": round(time_s, 2), "label": label})
                + "\n"
            )
        handle.flush()


def collect(
    video: Path,
    source_id: str,
    budget: int | None = None,
    sample_seconds: float | None = None,
) -> Path:
    """Label frames from one video, appending as it goes.

    `budget` caps how many requests this run may spend, so a day's quota can be
    split deliberately between broadcasts instead of the first one taking it
    all.
    """
    if not vision_detector.available():
        raise VisionUnavailable(
            f"{vision_detector.ENV_KEY} is not set, or google-genai is not installed."
        )

    destination = label_file(source_id)
    done = already_done(destination)
    if done:
        print(f"  {len(done)} frame(s) already labelled - continuing from there")

    spacing = sample_seconds or config.VISION_SAMPLE_SECONDS
    previous = config.VISION_SAMPLE_SECONDS
    config.VISION_SAMPLE_SECONDS = spacing

    try:
        with tempfile.TemporaryDirectory(prefix="teacher_") as tmp:
            frames = vision_detector.sample_frames(video, Path(tmp))
            wanted = [(t, p) for t, p in frames if round(t, 2) not in done]
            print(f"  {len(frames)} sampled, {len(wanted)} still to label")
            if not wanted:
                return destination

            _run(wanted, destination, source_id, budget)
    finally:
        config.VISION_SAMPLE_SECONDS = previous

    return destination


def _run(
    wanted: list[tuple[float, Path]],
    destination: Path,
    source_id: str,
    budget: int | None,
) -> None:
    from google import genai
    from google.genai import types

    import os

    client = genai.Client(api_key=os.environ[vision_detector.ENV_KEY])
    columns = config.VISION_GRID_COLUMNS
    per_call = columns * config.VISION_GRID_ROWS

    batches = [wanted[i : i + per_call] for i in range(0, len(wanted), per_call)]

    # Shuffled so that running dry leaves a sample of the whole broadcast
    # rather than its opening hour in detail and nothing afterwards. Seeded, so
    # a resumed run does not reshuffle into a different order.
    random.Random(len(wanted)).shuffle(batches)

    if budget is not None:
        batches = batches[:budget]

    interval = 60.0 / max(1.0, config.VISION_REQUESTS_PER_MINUTE)
    last_call = 0.0
    saved = 0

    with tempfile.TemporaryDirectory(prefix="teacher_grid_") as tmp:
        grid_dir = Path(tmp)

        for number, batch in enumerate(batches, start=1):
            grid = vision_detector._grid(
                [p for _, p in batch], grid_dir / f"g{number}.jpg", columns
            )

            pause = interval - (time.monotonic() - last_call)
            if pause > 0:
                time.sleep(pause)
            last_call = time.monotonic()

            try:
                response = vision_detector._ask(
                    client,
                    types,
                    vision_detector._PROMPT.format(first=1, last=len(batch)),
                    grid,
                )
            except DailyQuotaExhausted:
                print(f"  out of quota after {number - 1} request(s); {saved} label(s) saved")
                print("  run again tomorrow - it will carry on from here")
                return
            except Exception as exc:
                print(f"  batch {number} failed ({type(exc).__name__}), carrying on")
                continue

            verdicts = vision_detector._parse(
                getattr(response, "text", "") or "", 1, len(batch)
            )
            rows = [
                (time_s, verdicts[index + 1])
                for index, (time_s, _path) in enumerate(batch)
                if index + 1 in verdicts
            ]

            # Written now, not at the end. Everything already paid for survives
            # whatever happens to the rest of the run.
            _append(destination, source_id, rows)
            saved += len(rows)

            if number % 10 == 0 or number == len(batches):
                print(f"  {number}/{len(batches)} request(s), {saved} label(s) saved")


def plan(video_seconds: float, sample_seconds: float | None = None) -> int:
    """How many requests labelling this video would take."""
    spacing = sample_seconds or config.VISION_SAMPLE_SECONDS
    per_call = config.VISION_GRID_COLUMNS * config.VISION_GRID_ROWS
    return math.ceil((video_seconds / spacing) / per_call)
