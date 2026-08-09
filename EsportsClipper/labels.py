"""Where the training labels come from.

Two sources, because neither can do the job alone.

A highlight reel is almost entirely fighting - that is what an editor made it
out of - so every row of one is a positive, free of charge. What a reel cannot
provide is a negative, because the editor removed precisely the studio talk and
the quiet rotations we need the model to reject.

A raw broadcast has those in abundance but says nothing about which of its rows
are which. The Gemini classifier answers that, and it was measured at 17 of 18
frames correct with 12 of 12 studio frames correctly excluded, so its answers
are worth spending a limited daily quota on. It is spent only on the raw
broadcast, and only on the boundary the reels cannot teach: a firefight against
a squad that is merely moving.

Nothing here trusts a reel blindly. An edit opens on a logo sting, closes on a
podium shot, and cuts to a crowd reaction partway through, none of which is
combat. Rows are kept only where the game overlay is actually on screen.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

import config
import shared  # noqa: F401  (puts the sibling project on sys.path)

UNKNOWN = -1
STUDIO = 0
GAME = 1
FIGHT = 2

NAMES = {UNKNOWN: "unknown", STUDIO: "studio", GAME: "game", FIGHT: "fight"}

_FROM_TEACHER = {"STUDIO": STUDIO, "GAME": GAME, "FIGHT": FIGHT}


@dataclass(frozen=True)
class Labelled:
    """Labels for one source, aligned row-for-row with its feature table."""

    source_id: str
    times_s: np.ndarray
    labels: np.ndarray          # int8, one per row
    from_reel: bool

    def counts(self) -> dict[str, int]:
        return {
            NAMES[value]: int((self.labels == value).sum())
            for value in (UNKNOWN, STUDIO, GAME, FIGHT)
        }


def _column(values: np.ndarray, names: tuple[str, ...], wanted: str) -> np.ndarray:
    if wanted not in names:
        raise KeyError(f"Feature table has no column {wanted!r}.")
    return values[:, names.index(wanted)]


def from_reel(
    source_id: str,
    times_s: np.ndarray,
    values: np.ndarray,
    names: tuple[str, ...],
) -> Labelled:
    """Label an edited highlights reel: fighting, except where it plainly is not.

    Two filters, both necessary.

    The ends go, because a reel opens on a title card and closes on a podium or
    an outro, and neither is combat.

    The rest is kept only where the game overlay is on screen. A reel cuts to
    the crowd, to a caster reacting, to a replay wipe; those frames look nothing
    like a firefight and labelling them FIGHT would teach the model that a
    cheering crowd is one. Rows without the overlay are marked unknown rather
    than negative - a reel is not evidence about what studio talk looks like,
    and pretending otherwise would poison the negative class.
    """
    labels = np.full(len(times_s), UNKNOWN, dtype=np.int8)
    if len(times_s) == 0:
        return Labelled(source_id, times_s, labels, from_reel=True)

    hud = _column(values, names, "hud_present")
    trim = config.REEL_TRIM_SECONDS
    end = float(times_s[-1])

    inside = (times_s >= trim) & (times_s <= end - trim)
    playing = np.nan_to_num(hud, nan=0.0) >= config.REEL_MIN_HUD
    labels[inside & playing] = FIGHT

    return Labelled(source_id, times_s, labels, from_reel=True)


def from_teacher(
    source_id: str,
    times_s: np.ndarray,
    label_file: Path,
) -> Labelled:
    """Label a raw broadcast from the classifier's saved verdicts.

    Each verdict lands on the nearest grid row, and only if that row is close
    enough in time to be the same moment. Snapping a 10s-spaced verdict onto
    every row between would invent labels the classifier never gave.
    """
    labels = np.full(len(times_s), UNKNOWN, dtype=np.int8)
    if not label_file.exists() or len(times_s) == 0:
        return Labelled(source_id, times_s, labels, from_reel=False)

    step = float(times_s[1] - times_s[0]) if len(times_s) > 1 else config.GRID_SECONDS
    tolerance = max(step, config.LABEL_MATCH_SECONDS)

    applied = 0
    for line in label_file.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            # One malformed line must not discard a day of quota.
            continue
        if record.get("source") != source_id:
            continue
        value = _FROM_TEACHER.get(str(record.get("label", "")).upper())
        if value is None:
            continue

        index = int(np.searchsorted(times_s, float(record["time_s"])))
        index = max(0, min(len(times_s) - 1, index))
        if abs(times_s[index] - float(record["time_s"])) <= tolerance:
            labels[index] = value
            applied += 1

    if applied:
        print(f"    {source_id}: applied {applied} teacher label(s)")
    return Labelled(source_id, times_s, labels, from_reel=False)


def spread(labelled: Labelled, seconds: float) -> Labelled:
    """Extend each verdict over the moments either side of it.

    The classifier looked once every ten seconds; the grid has twenty rows in
    that time. Those rows are the same moment and a label that covers only one
    of them wastes nineteen. Spreading stops short of the next verdict, so a
    label never crosses into territory the classifier judged differently.
    """
    if seconds <= 0 or len(labelled.times_s) == 0:
        return labelled

    step = (
        float(labelled.times_s[1] - labelled.times_s[0])
        if len(labelled.times_s) > 1
        else config.GRID_SECONDS
    )
    reach = max(1, int(round(seconds / step)))

    out = labelled.labels.copy()
    known = np.flatnonzero(labelled.labels != UNKNOWN)

    for position in known:
        value = labelled.labels[position]
        low = max(0, position - reach)
        high = min(len(out), position + reach + 1)
        for index in range(low, high):
            if out[index] == UNKNOWN:
                out[index] = value

    return Labelled(labelled.source_id, labelled.times_s, out, labelled.from_reel)


def summarise(items: list[Labelled]) -> None:
    """Print what the training set actually contains."""
    print("\nLabels:")
    total: dict[str, int] = {}
    for item in items:
        counts = item.counts()
        kind = "reel" if item.from_reel else "broadcast"
        known = sum(v for k, v in counts.items() if k != "unknown")
        print(
            f"  {item.source_id:14s} {kind:9s} "
            f"{known:6d} labelled  "
            + "  ".join(f"{k}={v}" for k, v in counts.items() if k != "unknown")
        )
        for key, value in counts.items():
            total[key] = total.get(key, 0) + value

    usable = sum(v for k, v in total.items() if k != "unknown")
    print(f"  {'TOTAL':14s} {'':9s} {usable:6d} labelled  "
          + "  ".join(f"{k}={v}" for k, v in total.items() if k != "unknown"))

    fights = total.get("fight", 0)
    if usable:
        print(f"  fight share: {100 * fights / usable:.1f}%")
