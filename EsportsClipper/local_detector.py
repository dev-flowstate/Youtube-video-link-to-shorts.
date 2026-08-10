"""Find the fights with the trained model, locally and for free.

Same output as `vision_detector.detect` - a list of `Labelled` - so everything
downstream is untouched. The difference is what it costs and how closely it
looks. The Gemini pass samples every ten seconds because every look is a
request against a small daily quota; this samples twenty times a second's worth
of that, because a look costs nothing once the model exists.

Two heads run in order. The first asks whether the screen shows live gameplay
at all, which is the interview filter and the reason any of this exists. Only
rows it accepts reach the second, which asks whether that gameplay is a fight.

A single frame never opens a clip. A lens flare, a bright transition or one
confusing graphic can clear a threshold; what none of them can do is clear it
for several consecutive seconds. Requiring a run turns a merely good frame
score into a clip precision worth publishing, and costs almost no recall
because a real engagement produces dozens of positives in a row.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np

import config
import feature_table
import shared  # noqa: F401  (puts the sibling project on sys.path)
from vision_detector import FIGHT, GAME, STUDIO, Labelled

MODEL_PATH = Path(__file__).resolve().parent / "models" / "fight_model.joblib"


class ModelUnavailable(Exception):
    """Raised when the trained model cannot be loaded or does not fit."""


@dataclass(frozen=True)
class Scored:
    """Per-row output, kept so a run can be inspected rather than just trusted."""

    times_s: np.ndarray
    gameplay: np.ndarray      # Stage A probability
    fight: np.ndarray         # Stage B probability, NaN where Stage A said no
    labels: list[Labelled]


def available() -> bool:
    return MODEL_PATH.exists()


def _load() -> dict:
    if not MODEL_PATH.exists():
        raise ModelUnavailable(
            f"No trained model at {MODEL_PATH.name}. Run train_detector.py first."
        )
    try:
        payload = joblib.load(MODEL_PATH)
    except Exception as exc:
        raise ModelUnavailable(f"Could not load {MODEL_PATH.name}: {exc}") from exc

    if payload.get("synthetic"):
        # The self-test writes a model that learned invented data. It exists to
        # prove the machinery runs, and it knows nothing about a real fight.
        raise ModelUnavailable(
            "This model came from the synthetic self-test and cannot detect "
            "anything real. Train on actual footage before using it."
        )
    return payload


def _order_columns(payload: dict, values: np.ndarray) -> np.ndarray:
    """Put the table's columns into the order the model was trained on.

    The one failure this cannot survive is a silent column mismatch: the model
    still returns a probability, it is simply about the wrong feature, and
    nothing in the output would say so. The names travel inside the payload
    precisely so the two can be compared instead of assumed.
    """
    trained = list(payload["feature_names"])
    current = list(feature_table.FEATURE_NAMES)

    if trained == current:
        return values

    missing = [name for name in trained if name not in current]
    if missing:
        raise ModelUnavailable(
            f"The model expects {len(missing)} feature(s) this build no longer "
            f"produces (first: {missing[0]!r}). Retrain against the current schema."
        )

    index = [current.index(name) for name in trained]
    return values[:, index]


def _runs(flags: np.ndarray, need: int, window: int) -> np.ndarray:
    """Keep only positives inside a run of at least `need` in `window` steps."""
    if len(flags) == 0:
        return flags

    counts = np.convolve(flags.astype(np.int32), np.ones(window, dtype=np.int32), "same")
    return flags & (counts >= need)


def detect(video: Path, source_id: str | None = None) -> Scored:
    """Score every half-second of a video and label it."""
    payload = _load()

    name = source_id or video.stem
    times, values = feature_table.load_or_build(video, name, use_audio=True)
    values = _order_columns(payload, values)

    stage_a = payload["stage_a"]
    stage_b = payload["stage_b"]

    gameplay = stage_a["model"].predict_proba(values)[:, 1]
    is_gameplay = gameplay >= stage_a["threshold"]

    fight = np.full(len(times), np.nan, dtype=np.float64)
    if is_gameplay.any():
        fight[is_gameplay] = stage_b["model"].predict_proba(values[is_gameplay])[:, 1]

    hot = is_gameplay & (np.nan_to_num(fight, nan=0.0) >= stage_b["threshold"])

    # A run, not a frame. One bright graphic can clear a threshold; it cannot
    # stay cleared for several seconds running, and a real fight does.
    window = max(1, int(round(config.VOTE_WINDOW_SECONDS / config.GRID_SECONDS)))
    hot = _runs(hot, config.VOTE_MIN_HITS, window)

    labels: list[Labelled] = []
    for index, time_s in enumerate(times):
        if hot[index]:
            labels.append(Labelled(float(time_s), FIGHT))
        elif is_gameplay[index]:
            labels.append(Labelled(float(time_s), GAME))
        else:
            labels.append(Labelled(float(time_s), STUDIO))

    return Scored(times_s=times, gameplay=gameplay, fight=fight, labels=labels)


def summarise(scored: Scored) -> None:
    """What the model saw, so a run can be judged before its clips are watched."""
    counts: dict[str, int] = {}
    for item in scored.labels:
        counts[item.label] = counts.get(item.label, 0) + 1

    total = max(1, len(scored.labels))
    parts = ", ".join(
        f"{count} {label.lower()} ({100 * count / total:.0f}%)"
        for label, count in sorted(counts.items())
    )
    print(f"    saw {parts}")
