"""Train the local fight detector, and measure it in a way that survives scrutiny.

Two binary heads, not one three-class model.

**Stage A** answers "is this gameplay at all?" - GAME or FIGHT against STUDIO.
That is the complaint the whole project exists to fix: 88% of the audio
detector's clips were presenters, interviews and ranking boards. It is also the
cheap half, because a desk and a battlefield look nothing alike.

**Stage B** answers "is this gameplay a fight?" and only ever sees rows Stage A
already accepted. Splitting this way splits the imbalance with it: Stage A is
roughly balanced, and only Stage B is skewed, so only Stage B needs the careful
threshold.

Everything else here is about not lying to ourselves.

**The split is by block, not by row.** Rows sit half a second apart and
neighbouring rows are near-identical, so a random split puts a row's own twin in
the validation set and reports something near 0.97 that means nothing. Rows are
grouped into five-minute blocks within their source and whole blocks go to one
side or the other.

**Training rows near a validation block are thrown away.** Blocking alone is not
enough, because the features themselves reach across the boundary: the speech
z-score looks a minute back and the caster feature looks half a minute forward,
so a training row can literally contain audio from a validation row. Anything
within `PURGE_SECONDS` of a validation row goes.

**`early_stopping=False`.** Its internal validation split is random over rows,
which is precisely the leak this file is built to avoid - it would be leakage
switched on by default, inside the estimator, where it does not show up in any
split diagnostic.

**No accuracy and no ROC-AUC anywhere.** At a 9% fight rate "never a fight"
scores 91% accurate, and ROC-AUC is dominated by the enormous negative class.
Average precision and precision at a stated recall are the only numbers that
move when the detector actually gets better.

**A source only trains a stage if it holds both of that stage's classes.** A
highlights reel is FIGHT and nothing else, because that is what an editor left
in it. Folding the reels in therefore put every negative example in one video
and 91% of the positives in four others, and once the two classes live in
different videos, anything that says which video a row came from predicts the
label for free - the overlay art, the missing audio track, the clock. The rule
is applied by counting labels per source, not by naming videos, so a fifth reel
is handled without anyone remembering to handle it.

**Reels never score.** A metric computed over reel rows measures the editor, not
the model. Every reported number and both thresholds come from broadcast rows
only, where the class balance is the one the detector will meet in production.

Run:
    py train_detector.py                  # real footage under data/
    py train_detector.py --synthetic      # self-test, no video needed
    py train_detector.py --purge-seconds 0    # show what the purge is worth
    py train_detector.py --reel-positives     # try the reels in Stage B anyway
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score, precision_recall_curve

import audio_grid
import config
import labels
import shared  # noqa: F401  (puts the sibling project on sys.path)

MODELS_DIR = Path(__file__).resolve().parent / "models"

# ---------------------------------------------------------------------------
# Splitting
# ---------------------------------------------------------------------------

# Rows are grouped into blocks of this length inside their own source, and
# whole blocks are held out together. Five minutes is comfortably longer than
# any single engagement (fights run 10-40s) and longer than the longest feature
# window, so a block is a genuinely separate stretch of broadcast rather than a
# slice through the middle of one moment.
#
# Grouping by *source* instead would leave four or five groups in total, which
# is too few to fold and would make every score a coin toss on which reel
# landed where. Cross-source generalisation is what the held-out broadcast B is
# for; this measures generalisation across time.
BLOCK_SECONDS = 300.0

# Training rows this close to any validation row are dropped, because the two
# rows share raw input and are therefore not independent.
#
# Derived, not chosen. The widest reach in the feature schema is the audio pair:
# the speech z-score is computed against the preceding SPEECH_Z_WINDOW_SECONDS,
# and the caster feature takes a forward maximum over CASTER_LOOKAHEAD_SECONDS.
# A row therefore depends on raw audio spanning lookback + lookahead, and two
# rows further apart than that share none of it. The vision features' rolling
# windows are narrower than this pair, so this bounds them too.
PURGE_SECONDS = config.SPEECH_Z_WINDOW_SECONDS + config.CASTER_LOOKAHEAD_SECONDS

# Five folds over a few dozen blocks. More folds would leave each validation
# side too small to estimate a precision at all.
N_SPLITS = 5

RANDOM_STATE = 0

# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

# Capacity is deliberately small. The effective sample size is not the row count
# - it is the number of distinct engagements and studio segments, which is a few
# hundred. Twenty thousand rows describing three hundred moments is three
# hundred examples with a lot of repetition, and a model sized for the row count
# would memorise every one of them.
#
# Trees also mean no scaler, which matters here for a second reason: a scaler
# fitted across folds is itself a leak, and the cheapest way not to leak through
# preprocessing is to have none.
LEARNING_RATE = 0.06
MAX_LEAF_NODES = 15
MIN_SAMPLES_LEAF = 40
L2_REGULARIZATION = 1.0

# How long to boost for, decided by the grouped CV rather than asserted.
#
# The grid stops at 400 as a cap, not because the search converged there. Even a
# blocked split shares sources between its two sides, so cross-validation can
# still reward a model for recognising a broadcast rather than a fight, and it
# will happily keep asking for more trees while it does. The run says so when it
# picks the top of the grid, because that is the case where the number came from
# the cap rather than from the data.
MAX_ITER_GRID = (50, 100, 200, 400)

# ---------------------------------------------------------------------------
# Operating points
# ---------------------------------------------------------------------------

# Stage A is the interview filter, so its errors are the ones the user actually
# sees. A false positive here is a clip of a presenter, which is the failure
# that made the old detector useless.
STAGE_A_TARGET_PRECISION = 0.98
STAGE_A_MIN_RECALL = 0.0

# Stage B may miss fights - a broadcast has dozens and only ~50 clips are wanted
# - but what it does pick must mostly be fights.
STAGE_B_TARGET_PRECISION = 0.80
STAGE_B_MIN_RECALL = 0.40

# The recall the report quotes precision at, for both stages, so the two are
# comparable and neither can be flattered by quoting its own best corner.
REPORT_RECALL = 0.50

# ---------------------------------------------------------------------------
# Auxiliary positives
# ---------------------------------------------------------------------------

# Whether the highlight reels are allowed back into Stage B as extra positives.
#
# Off. On this corpus a reel row is not so much another example of a fight as
# another example of being a reel: every one of them is FIGHT by construction
# and every one of them has a NaN audio block, so "the sound is missing" is a
# free and perfect answer, and the leak checks cannot see it because they run on
# the broadcast slice where the sound is really there. Measured both ways on one
# label set, the reels cost Stage B 0.0348 average precision and 0.0312
# precision at its own threshold.
#
# Reachable rather than deleted, because "reels cannot help Stage B" is a
# measurement on this corpus, not a law. Switching it on fits Stage B twice and
# prints both results on the same broadcast-only slice, so the claim can be
# rechecked instead of believed. What it never does is let a reel row into that
# slice or into the threshold - those stay broadcast-only in both fits.
STAGE_B_REEL_POSITIVES = False

# The numbers that comparison turns on. Average precision leads because it is
# what the max_iter search optimises and the only one of the four that does not
# move when the threshold moves.
_REEL_COMPARISON: tuple[tuple[str, str], ...] = (
    ("average_precision", "average precision"),
    (f"precision_at_recall_{REPORT_RECALL:g}", f"precision at recall {REPORT_RECALL:g}"),
    ("threshold_precision", "precision at threshold"),
    ("threshold_recall", "recall at threshold"),
)

# ---------------------------------------------------------------------------
# Leakage self-checks
# ---------------------------------------------------------------------------

# How far above the base rate a deliberately uninformative model may score
# before the split is called leaky. Not zero: average precision on a finite
# validation set wobbles, and a source whose studio talk genuinely clusters at
# the start of the day gives a timestamp a little honest signal. A quarter above
# base rate is comfortably outside that wobble and far below what an actual leak
# produces, which is near-perfect.
LEAK_MARGIN = 1.25

# Stage A currently fails the timestamp check, and it is worth writing down what
# that failure is and is not, because the obvious suspect is innocent.
#
# It is not the choice of training sources. Both checks have always run on the
# prior-preserving broadcast rows alone, so no reel row has ever been inside
# one; excluding the reels from the fit left the number exactly where it was.
#
# It is the shape of the folds. A broadcast day runs in phases - a pre-show, a
# match, a break - far longer than a five-minute block: measured here, a block
# is 91% one class at the median and 81% of neighbouring blocks share a majority
# class. GroupKFold deals blocks out at random, so a held-out block nearly
# always has training blocks on both sides of it, 90 seconds away, and a model
# given nothing but a clock interpolates across that gap. Dealt instead as five
# contiguous stretches of the day, where a clock has to extrapolate, the same
# model scores 0.4413 against a base rate of 0.5271 - worse than guessing -
# against 0.7657 with the blocks shuffled.
#
# The honest reading is that Stage A's fold estimate is inflated by whatever
# else drifts slowly across a day, and that the repair is fold geometry rather
# than the label table. That is a deliberate change to a leakage control, so it
# is left for someone to make on purpose rather than smuggled in here.

# ---------------------------------------------------------------------------
# Importance
# ---------------------------------------------------------------------------

# Correlated columns are permuted together, because permuting them one at a
# time tells you nothing: knock out flash_rate and flash_peak_8s carries the
# signal, so both come out looking worthless. The groups are what the features
# are *about*, which is the level the question is asked at.
IMPORTANCE_REPEATS = 5

# Audio columns are named exactly, because this table exists to settle one
# argument with a number - whether the audio gunfire features contribute
# anything, given that on real footage studio talk out-scored firefights 1.03 to
# 0.43 onsets per second. Guessing at the grouping would make the answer
# arguable.
_AUDIO_GROUPS: dict[str, str] = {
    "aud_flux_mean": "audio-gunfire",
    "aud_flux_p95": "audio-gunfire",
    "aud_gunfire_onsets": "audio-gunfire",
    "aud_hf_ratio_mean": "audio-gunfire",
    # Below speech: explosions and impacts. A venue crowd also lives down here,
    # so this column straddles the two groups; it sits with gunfire because
    # impacts are the combat half of what it measures.
    "aud_low_energy": "audio-gunfire",
    "aud_speech_energy": "audio-caster",
    "aud_speech_z60": "audio-caster",
    "aud_speech_ahead30": "audio-caster",
    "aud_silence_frac": "audio-caster",
}

# Vision columns are matched on keyword, because frame_features owns its own
# names and this file should not have to be edited every time it gains a
# column. Anything unmatched lands in "other" and is printed, so a new feature
# quietly falling outside every group is visible rather than silent. First match
# wins, so the order below is the tie-break.
#
# Two judgement calls worth naming. The whole-frame luma columns sit with flash
# because they measure the same thing at different sensitivities - a muzzle
# flash moves luma_p99 before it moves bright_frac - and permuting one without
# the other would let the survivor carry the signal. The damage-number columns
# sit with the kill feed because both answer "something just landed", and they
# fire within seconds of each other, so separating them would split one piece of
# evidence in half and score both halves as worthless.
_VISION_KEYWORDS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("flash", ("flash", "muzzle", "bright", "luma")),
    ("fire", ("fire", "tracer", "explos")),
    ("motion", ("motion", "frame_diff", "scene_cut", "edge_density", "flow", "shake")),
    ("hud", ("hud", "overlay", "minimap", "presence", "standings")),
    ("killfeed", ("killfeed", "kill_feed", "dmg", "damage", "banner", "elim")),
    ("ticks", ("tick", "alive", "knock", "remaining", "roster", "zone", "stage")),
)


# ---------------------------------------------------------------------------
# The table
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Dataset:
    """Every labelled row from every source, in one block of memory."""

    feature_names: tuple[str, ...]
    values: np.ndarray       # (rows, features) float32, NaN where unknown
    label: np.ndarray        # int8, one of labels.STUDIO / GAME / FIGHT
    source: np.ndarray       # str, which video the row came from
    times_s: np.ndarray      # float, seconds into that video
    from_reel: np.ndarray    # bool, True for edited highlights

    def counts(self) -> dict[str, int]:
        return {
            labels.NAMES[value]: int((self.label == value).sum())
            for value in (labels.STUDIO, labels.GAME, labels.FIGHT)
        }


def _combine(parts: list[tuple[str, bool, np.ndarray, np.ndarray, np.ndarray]],
             feature_names: tuple[str, ...]) -> Dataset:
    """Stack per-source tables, keeping only rows that carry a label."""
    values, label, source, times, reel = [], [], [], [], []

    for source_id, is_reel, source_times, source_values, source_labels in parts:
        known = source_labels != labels.UNKNOWN
        if not known.any():
            print(f"  {source_id}: no usable labels - skipped")
            continue
        values.append(source_values[known])
        label.append(source_labels[known])
        times.append(source_times[known])
        source.append(np.full(int(known.sum()), source_id))
        reel.append(np.full(int(known.sum()), is_reel))

    if not values:
        raise SystemExit("No labelled rows anywhere. Nothing to train on.")

    return Dataset(
        feature_names=feature_names,
        values=np.vstack(values).astype(np.float32),
        label=np.concatenate(label).astype(np.int8),
        source=np.concatenate(source),
        times_s=np.concatenate(times).astype(float),
        from_reel=np.concatenate(reel).astype(bool),
    )


def load_real() -> Dataset:
    """Build the table from the footage under data/.

    Imported here rather than at module scope so the synthetic self-test still
    runs on a machine where the vision extractor or the footage is missing.

    Broadcast B is deliberately absent: config keeps it held out and never
    fitted on, because it carries a facecam overlay that A does not, and the
    only honest cross-tournament number comes from a source no fold ever saw.
    """
    import feature_table
    import teacher_labels

    parts: list[tuple[str, bool, np.ndarray, np.ndarray, np.ndarray]] = []

    print("Reels (positives, audio deliberately withheld):")
    for source_id in config.TRAINING_REELS:
        video = config.DATA_DIR / "reels" / f"{source_id}.mp4"
        if not video.exists():
            print(f"  {source_id}: not downloaded yet - skipped")
            continue
        times, values = feature_table.load_or_build(video, source_id, use_audio=False)
        marked = labels.from_reel(source_id, times, values, feature_table.FEATURE_NAMES)
        parts.append((source_id, True, times, values, marked.labels))

    print("\nBroadcasts (the only source of negatives):")
    for source_id in (config.TRAINING_BROADCAST_A,):
        video = config.DATA_DIR / "broadcasts" / f"{source_id}.mp4"
        if not video.exists():
            print(f"  {source_id}: not downloaded yet - skipped")
            continue
        times, values = feature_table.load_or_build(video, source_id, use_audio=True)
        marked = labels.from_teacher(source_id, times, teacher_labels.label_file(source_id))
        marked = labels.spread(marked, config.LABEL_SPREAD_SECONDS)
        parts.append((source_id, False, times, values, marked.labels))

    return _combine(parts, feature_table.FEATURE_NAMES)


# ---------------------------------------------------------------------------
# Which sources may train a stage
# ---------------------------------------------------------------------------


def _drop_single_class_sources(
    stage: str, data: Dataset, rows: np.ndarray, y: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Keep only the sources that supply both of this stage's classes.

    This is the rule that stops a head learning which video a row came from
    instead of what is on screen. Inside a source that contributes one class
    only, the label never varies, so nothing within that source separates the
    two classes - the only thing separating its rows from another source's rows
    is the source itself, and every property that identifies a source is then
    free to stand in for the label: the overlay art, the encoder, the missing
    audio track, the length of the video, the clock.

    Measured, not feared. Folding the four reels back in - which is what
    --reel-positives now does - puts every negative example in one video and
    13362 of Stage B's 17129 positives in four others, and costs Stage B 0.0348
    average precision on the broadcast rows both fits are scored on: 0.4131
    without them against 0.3783 with, same labels, same folds. They are not
    extra examples of a fight. They are extra examples of being a reel, and a
    reel gives itself away by its missing audio block alone.

    The test is label composition rather than a list of video ids, because a
    list is a thing to forget. A fifth reel of pure fights excludes itself, and
    a reel that one day arrives with its quiet rotations labelled makes itself
    eligible, with nobody editing this file either time.
    """
    print(f"\n{stage}: source eligibility (a source must supply both classes)")

    source = data.source[rows]
    from_reel = data.from_reel[rows]
    keep = np.zeros(len(rows), dtype=bool)

    for source_id in sorted(set(source.tolist())):
        here = source == source_id
        positive = int(y[here].sum())
        negative = int(here.sum()) - positive
        usable = positive > 0 and negative > 0
        if usable:
            keep |= here
        print(f"  {source_id:14s} {'reel' if from_reel[here][0] else 'broadcast':9s} "
              f"{int(here.sum()):6d} rows  {positive:6d} positive  {negative:6d} negative"
              f"   {'eligible' if usable else 'EXCLUDED - one class only'}")

    if not keep.any():
        raise SystemExit(
            f"{stage}: no source holds both of its classes, so there is no "
            "honest training set - only a way of learning which video a row "
            "came from. Label the missing side of the boundary in a source that "
            "already has the other side."
        )

    print(f"  keeping {int(keep.sum())} of {len(rows)} rows from "
          f"{len(set(source[keep].tolist()))} of {len(set(source.tolist()))} source(s)")
    return rows[keep], y[keep]


# ---------------------------------------------------------------------------
# Folds
# ---------------------------------------------------------------------------


def _block_ids(source: np.ndarray, times_s: np.ndarray) -> np.ndarray:
    """One group key per five-minute stretch of one video."""
    index = np.floor(times_s / BLOCK_SECONDS).astype(int)
    return np.array([f"{s}#{i}" for s, i in zip(source, index)])


def _purge(
    train_index: np.ndarray,
    valid_index: np.ndarray,
    source: np.ndarray,
    times_s: np.ndarray,
    purge_seconds: float,
) -> np.ndarray:
    """Drop training rows that share raw input with a validation row.

    Distance is measured to the nearest validation row of the *same* source;
    two videos never share a clock, so a broadcast at t=900 and a reel at t=900
    are unrelated.
    """
    if purge_seconds <= 0:
        return train_index

    keep = np.ones(len(train_index), dtype=bool)
    train_source = source[train_index]
    train_times = times_s[train_index]

    for name in np.unique(source[valid_index]):
        held = np.sort(times_s[valid_index][source[valid_index] == name])
        here = train_source == name
        if not here.any() or held.size == 0:
            continue

        mine = train_times[here]
        position = np.searchsorted(held, mine)
        before = np.where(position > 0, mine - held[np.clip(position - 1, 0, None)], np.inf)
        after = np.where(
            position < held.size,
            held[np.clip(position, None, held.size - 1)] - mine,
            np.inf,
        )
        keep[here] &= np.minimum(before, after) >= purge_seconds

    return train_index[keep]


def _folds(
    source: np.ndarray, times_s: np.ndarray, purge_seconds: float
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Contiguous blocked folds with the training side purged.

    Blocks are dealt as unbroken stretches of each source, never shuffled, and
    that distinction turned out to be the whole ball game.

    Shuffling the groups looks safe - a five-minute block stays whole either
    way, so no row ever sits opposite its own twin. It is not safe. A broadcast
    day is made of phases far longer than five minutes: a pre-show, a match, a
    break. Deal the blocks at random and almost every validation block ends up
    with training blocks on both sides of it in time, so a model needs only to
    interpolate between two neighbours to guess the middle. Measured on this
    broadcast with the clock as the only feature:

        base rate                      0.5169
        blocks shuffled                0.8299
        blocks contiguous              0.5176

    Shuffled, the clock alone scores far above chance while knowing nothing.
    Contiguous, it lands on the base rate exactly - which is what "this feature
    carries no information" is supposed to look like.

    Each source is split separately so every fold still sees every source, and
    a reel cannot end up wholly inside one fold merely for being short.
    """
    groups = _block_ids(source, times_s)
    distinct = len(np.unique(groups))
    if distinct < N_SPLITS:
        raise SystemExit(
            f"Only {distinct} five-minute block(s) of labelled footage. "
            f"{N_SPLITS} folds need at least that many. Label more, or lower "
            f"BLOCK_SECONDS - but lowering it moves the split closer to the "
            f"rows and is a way of cheating."
        )

    # Each source's blocks are split into N contiguous runs, and run i of every
    # source forms fold i. Doing it per source rather than over the pooled
    # blocks keeps all sources represented in every fold while leaving each
    # fold's footage an unbroken stretch of its own day.
    per_fold: list[list[np.ndarray]] = [[] for _ in range(N_SPLITS)]
    for name in np.unique(source):
        in_source = source == name
        source_blocks = np.unique(groups[in_source])
        if len(source_blocks) < N_SPLITS:
            # Too short to divide; it trains everywhere and validates nowhere,
            # which is honest - a source that cannot be held out cannot be
            # scored on either.
            continue
        for fold, chunk in enumerate(np.array_split(np.sort(source_blocks), N_SPLITS)):
            per_fold[fold].append(np.flatnonzero(in_source & np.isin(groups, chunk)))

    out = []
    for fold in range(N_SPLITS):
        if not per_fold[fold]:
            continue
        valid_index = np.sort(np.concatenate(per_fold[fold]))
        train_index = np.setdiff1d(np.arange(len(groups)), valid_index)
        out.append(
            (_purge(train_index, valid_index, source, times_s, purge_seconds), valid_index)
        )

    if not out:
        raise SystemExit(
            "No source had enough labelled footage to divide into folds. "
            "Label more before training."
        )
    return out


# ---------------------------------------------------------------------------
# Fitting
# ---------------------------------------------------------------------------


def _model(max_iter: int) -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(
        learning_rate=LEARNING_RATE,
        max_iter=max_iter,
        max_leaf_nodes=MAX_LEAF_NODES,
        min_samples_leaf=MIN_SAMPLES_LEAF,
        l2_regularization=L2_REGULARIZATION,
        # Its internal split is over rows, chosen at random. That is exactly the
        # leak this file exists to prevent, and it would be on by default.
        early_stopping=False,
        class_weight="balanced",
        random_state=RANDOM_STATE,
    )


@dataclass(frozen=True)
class Fitted:
    """Out-of-fold scores plus the fold models, for the importance pass."""

    scores: np.ndarray                                        # NaN where unscored
    fold_average_precision: list[float]
    models: list[tuple[HistGradientBoostingClassifier, np.ndarray]]


def _cross_validate(
    values: np.ndarray,
    y: np.ndarray,
    source: np.ndarray,
    times_s: np.ndarray,
    score_on: np.ndarray,
    max_iter: int,
    purge_seconds: float,
    shuffle_train_labels: bool = False,
) -> Fitted:
    """Fit each fold and score the rows it held out.

    `score_on` marks the rows a fold's average precision is computed over.
    Everything is fitted on, but only prior-preserving rows are scored - see
    `_report`.
    """
    scores = np.full(len(y), np.nan)
    per_fold: list[float] = []
    models: list[tuple[HistGradientBoostingClassifier, np.ndarray]] = []
    rng = np.random.default_rng(RANDOM_STATE)

    for train_index, valid_index in _folds(source, times_s, purge_seconds):
        y_train = y[train_index]
        if shuffle_train_labels:
            y_train = rng.permutation(y_train)
        if len(np.unique(y_train)) < 2:
            continue

        model = _model(max_iter).fit(values[train_index], y_train)
        predicted = model.predict_proba(values[valid_index])[:, 1]
        scores[valid_index] = predicted
        models.append((model, valid_index))

        graded = score_on[valid_index]
        if graded.sum() >= 2 and 0 < y[valid_index][graded].sum() < int(graded.sum()):
            per_fold.append(
                float(average_precision_score(y[valid_index][graded], predicted[graded]))
            )

    return Fitted(scores, per_fold, models)


def _choose_max_iter(
    values: np.ndarray,
    y: np.ndarray,
    source: np.ndarray,
    times_s: np.ndarray,
    score_on: np.ndarray,
    purge_seconds: float,
) -> tuple[int, Fitted, dict[int, float]]:
    """Pick the boosting length on grouped, purged folds - never a random split."""
    best_iter, best_fit, best_score = 0, None, -np.inf
    tried: dict[int, float] = {}

    for max_iter in MAX_ITER_GRID:
        fit = _cross_validate(
            values, y, source, times_s, score_on, max_iter, purge_seconds
        )
        mean = float(np.mean(fit.fold_average_precision)) if fit.fold_average_precision else 0.0
        tried[max_iter] = mean
        print(f"    max_iter={max_iter:4d}  mean fold AP {mean:.4f}")
        if mean > best_score:
            best_iter, best_fit, best_score = max_iter, fit, mean

    assert best_fit is not None
    return best_iter, best_fit, tried


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def _curve(y: np.ndarray, scores: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """precision, recall and threshold, all the same length."""
    precision, recall, threshold = precision_recall_curve(y, scores)
    n = len(threshold)
    return precision[:n], recall[:n], threshold


def _precision_at_recall(y: np.ndarray, scores: np.ndarray, floor: float) -> float:
    """The best precision still available at or above this recall."""
    precision, recall, _ = _curve(y, scores)
    reachable = recall >= floor
    return float(precision[reachable].max()) if reachable.any() else float("nan")


def _operating_point(
    y: np.ndarray, scores: np.ndarray, target_precision: float, min_recall: float
) -> tuple[float, float, float, bool]:
    """The threshold to ship: most recall that still meets the precision target.

    When nothing meets the target, this returns the most precise point that
    still clears the recall floor and says so, rather than quietly shipping a
    threshold that misses the requirement.
    """
    precision, recall, threshold = _curve(y, scores)
    feasible = recall >= min_recall
    hits = feasible & (precision >= target_precision)
    pool = hits if hits.any() else feasible
    if not pool.any():
        return float("nan"), float("nan"), float("nan"), False

    where = np.flatnonzero(pool)
    best = where[np.argmax(recall[where])] if hits.any() else where[np.argmax(precision[where])]
    return (
        float(threshold[best]),
        float(precision[best]),
        float(recall[best]),
        bool(hits.any()),
    )


# ---------------------------------------------------------------------------
# Leakage self-checks
# ---------------------------------------------------------------------------


def _leak_checks(
    values: np.ndarray,
    y: np.ndarray,
    source: np.ndarray,
    times_s: np.ndarray,
    max_iter: int,
    purge_seconds: float,
) -> dict[str, object]:
    """Two models that should not work. If either does, the split is broken.

    Both run on the prior-preserving rows alone, fitted and scored. Mixing reels
    in would confound the timestamp check with source identity - reels are
    minutes long and all positive, broadcasts are hours long and mostly not - and
    a check that can fail for a reason other than leakage is not a check.
    """
    base = float(y.mean())
    everything = np.ones(len(y), dtype=bool)

    # A timestamp carries no information about what is on screen. If knowing
    # *when* a row happened predicts its label across held-out blocks, the
    # blocks are not held out.
    clock = times_s.reshape(-1, 1).astype(np.float32)
    clock_fit = _cross_validate(
        clock, y, source, times_s, everything, max_iter, purge_seconds
    )
    clock_ap = float(average_precision_score(y, clock_fit.scores))

    # Shuffled training labels destroy the only real signal. Anything left is
    # the split handing the model its answers.
    shuffled_fit = _cross_validate(
        values, y, source, times_s, everything, max_iter, purge_seconds,
        shuffle_train_labels=True,
    )
    shuffled_ap = float(average_precision_score(y, shuffled_fit.scores))

    ceiling = base * LEAK_MARGIN
    return {
        "base_rate": base,
        "ceiling": ceiling,
        "timestamp_only_ap": clock_ap,
        "timestamp_only_passed": clock_ap <= ceiling,
        "shuffled_labels_ap": shuffled_ap,
        "shuffled_labels_passed": shuffled_ap <= ceiling,
        "passed": clock_ap <= ceiling and shuffled_ap <= ceiling,
    }


# ---------------------------------------------------------------------------
# Grouped permutation importance
# ---------------------------------------------------------------------------


def _feature_groups(names: tuple[str, ...]) -> dict[str, list[int]]:
    """Which columns belong together, by what they measure."""
    groups: dict[str, list[int]] = {}
    for index, name in enumerate(names):
        lowered = name.lower()
        if lowered in _AUDIO_GROUPS:
            group = _AUDIO_GROUPS[lowered]
        else:
            group = "other"
            for candidate, keywords in _VISION_KEYWORDS:
                if any(word in lowered for word in keywords):
                    group = candidate
                    break
        groups.setdefault(group, []).append(index)
    return groups


def _print_groups(names: tuple[str, ...], groups: dict[str, list[int]]) -> None:
    """Show the grouping once, so a miscategorised column is visible up front."""
    print("\nFeature groups (permuted together in the importance table):")
    for group in sorted(groups):
        members = ", ".join(names[i] for i in groups[group])
        marker = "  <- UNGROUPED, check _VISION_KEYWORDS" if group == "other" else ""
        print(f"  {group:16s} {len(groups[group]):2d}  {members}{marker}")


def _grouped_importance(
    fit: Fitted,
    values: np.ndarray,
    y: np.ndarray,
    score_on: np.ndarray,
    groups: dict[str, list[int]],
) -> list[tuple[str, float, float]]:
    """How much average precision each group of features is worth.

    Measured on the prior-preserving rows only. Reels have all-NaN audio, so
    including them would let the audio groups score for a reason that has
    nothing to do with sound: permuting them would shuffle "audio missing"
    across rows, and "audio missing" is a perfect marker for "this row came from
    a reel", which is a perfect marker for FIGHT. Restricted to broadcast rows
    the audio is really there, and the number means what it says.
    """
    rng = np.random.default_rng(RANDOM_STATE)
    drops: dict[str, list[float]] = {name: [] for name in groups}

    for model, valid_index in fit.models:
        graded = score_on[valid_index]
        rows = valid_index[graded]
        if len(rows) < 2:
            continue
        truth = y[rows]
        if truth.sum() == 0 or truth.sum() == len(truth):
            continue

        block = values[rows]
        baseline = average_precision_score(truth, model.predict_proba(block)[:, 1])

        for name, columns in groups.items():
            for _ in range(IMPORTANCE_REPEATS):
                # One permutation for the whole group, so the correlations
                # *within* it survive and only its link to the label is cut.
                shuffled = block.copy()
                shuffled[:, columns] = block[rng.permutation(len(block))][:, columns]
                spoiled = average_precision_score(
                    truth, model.predict_proba(shuffled)[:, 1]
                )
                drops[name].append(float(baseline - spoiled))

    ranked = [
        (name, float(np.mean(values_)), float(np.std(values_)))
        for name, values_ in drops.items()
        if values_
    ]
    ranked.sort(key=lambda row: row[1], reverse=True)
    return ranked


# ---------------------------------------------------------------------------
# One stage, end to end
# ---------------------------------------------------------------------------


@dataclass
class Stage:
    name: str
    positive: str
    model: HistGradientBoostingClassifier
    threshold: float
    report: dict[str, object]
    importance: list[tuple[str, float, float]]


def _run_stage(
    name: str,
    positive: str,
    data: Dataset,
    rows: np.ndarray,
    y: np.ndarray,
    target_precision: float,
    min_recall: float,
    purge_seconds: float,
) -> Stage:
    """Cross-validate, choose a threshold, then refit on everything."""
    values = data.values[rows]
    source = data.source[rows]
    times_s = data.times_s[rows]

    # The prior-preserving slice: rows whose class balance was not enriched by
    # an editor. A reel is ~95% FIGHT because that is what a reel is, so a
    # threshold picked with reels in the pool is picked against a world where
    # nearly everything is a fight - and then applied to a broadcast where
    # nearly nothing is. Reels stay in the fit and stay out of every number.
    graded = ~data.from_reel[rows]

    print(f"\n{name}: {positive}")
    print(f"  rows {len(y)} ({int(graded.sum())} prior-preserving), "
          f"positives {int(y.sum())} ({100 * y.mean():.1f}%)")
    if graded.sum() < 2 or not 0 < y[graded].sum() < int(graded.sum()):
        raise SystemExit(
            f"{name}: the prior-preserving slice has no usable mix of classes. "
            "Every honest number this file reports comes from broadcast rows, "
            "so there is nothing to report. Collect teacher labels first."
        )
    print(f"  prior-preserving base rate {100 * y[graded].mean():.1f}%")

    # What the purge costs, stated rather than assumed. It is not free - every
    # row it drops is a row the model does not learn from - so the trade is only
    # worth making if the number is visible.
    whole = _folds(source, times_s, 0.0)
    purged = _folds(source, times_s, purge_seconds)
    before = float(np.mean([len(train) for train, _ in whole]))
    after = float(np.mean([len(train) for train, _ in purged]))
    print(f"  purge drops {before - after:.0f} of {before:.0f} training rows per fold "
          f"({100 * (1 - after / before):.1f}%)")

    print("  choosing max_iter on grouped, purged folds:")
    max_iter, fit, tried = _choose_max_iter(
        values, y, source, times_s, graded, purge_seconds
    )
    print(f"  chosen max_iter {max_iter}")
    if max_iter == MAX_ITER_GRID[-1]:
        print("    (the top of the grid - the cap chose this, not the curve)")

    scored = graded & np.isfinite(fit.scores)
    truth, predicted = y[scored], fit.scores[scored]

    average_precision = float(average_precision_score(truth, predicted))
    with_reels = float(average_precision_score(y, fit.scores))
    at_recall = _precision_at_recall(truth, predicted, REPORT_RECALL)
    threshold, precision, recall, met = _operating_point(
        truth, predicted, target_precision, min_recall
    )

    print(f"  average precision            {average_precision:.4f}  "
          f"(base rate {truth.mean():.4f})")
    print(f"  precision at recall {REPORT_RECALL:.2f}       {at_recall:.4f}")
    if not graded.all():
        print(f"  same, reels included         {with_reels:.4f}  "
              "(flattered - reels are ~all positive)")
    print(f"  threshold                    {threshold:.4f}")
    print(f"  at that threshold            precision {precision:.4f}  recall {recall:.4f}")
    if met:
        print(f"  target precision {target_precision:.2f} at recall >= {min_recall:.2f}: MET")
    else:
        print(f"  target precision {target_precision:.2f} at recall >= {min_recall:.2f}: "
              f"NOT MET - best available is {precision:.4f}")

    print("  leakage self-checks:")
    leaks = _leak_checks(
        values[graded], y[graded], source[graded], times_s[graded], max_iter, purge_seconds
    )
    _print_leaks(leaks)

    print("  grouped permutation importance (drop in AP, validation folds):")
    groups = _feature_groups(data.feature_names)
    importance = _grouped_importance(fit, values, y, graded, groups)
    for group, mean, spread in importance:
        bar = "#" * max(0, int(round(mean * 200)))
        print(f"    {group:16s} {mean:+8.4f} +/- {spread:6.4f}  {bar}")

    final = _model(max_iter).fit(values, y)

    return Stage(
        name=name,
        positive=positive,
        model=final,
        threshold=threshold,
        report={
            "positive": positive,
            "training_sources": sorted(set(source.tolist())),
            "rows": int(len(y)),
            "rows_prior_preserving": int(graded.sum()),
            "positives": int(y.sum()),
            "base_rate": float(y[graded].mean()),
            "max_iter": max_iter,
            "max_iter_grid": {str(k): v for k, v in tried.items()},
            "fold_average_precision": fit.fold_average_precision,
            "average_precision": average_precision,
            "average_precision_with_reels": with_reels,
            f"precision_at_recall_{REPORT_RECALL:g}": at_recall,
            "threshold": threshold,
            "threshold_precision": precision,
            "threshold_recall": recall,
            "target_precision": target_precision,
            "target_min_recall": min_recall,
            "target_met": met,
            "leakage": leaks,
        },
        importance=importance,
    )


def _print_leaks(leaks: dict[str, object]) -> None:
    base = float(leaks["base_rate"])  # type: ignore[arg-type]
    ceiling = float(leaks["ceiling"])  # type: ignore[arg-type]
    for key, title in (
        ("timestamp_only", "timestamp as the only feature"),
        ("shuffled_labels", "training labels shuffled"),
    ):
        value = float(leaks[f"{key}_ap"])  # type: ignore[arg-type]
        passed = bool(leaks[f"{key}_passed"])
        verdict = "ok" if passed else "*** FAILED - THE SPLIT LEAKS ***"
        print(f"    {title:32s} AP {value:.4f}  vs base rate {base:.4f} "
              f"(ceiling {ceiling:.4f})  {verdict}")


def _stage_b_with_reels(
    data: Dataset,
    baseline: Stage,
    rows: np.ndarray,
    y: np.ndarray,
    purge_seconds: float,
) -> Stage:
    """Refit Stage B with the reels' FIGHT rows added, and compare like for like.

    The reels move the fit and nothing else. Both fits are scored on the same
    prior-preserving broadcast rows and both thresholds are chosen there, which
    is the only comparison worth printing - a number computed over reel rows
    measures the editor, not the model, and would show an improvement whatever
    happened.

    Expect it to hurt. Reel rows carry two markers of their own label that no
    broadcast row has, a missing audio block and an overlay from a different
    edit, and neither is visible to leak checks that run on broadcast rows. That
    is why the answer is printed as a difference rather than asserted here: the
    reels have to show what they are worth on this corpus, and on the next one
    the arithmetic may come out the other way.
    """
    extra = np.flatnonzero(data.from_reel & (data.label == labels.FIGHT))
    if extra.size == 0:
        print("\nStage B: reel positives asked for, but no reel FIGHT rows exist.")
        return baseline

    with_reels = _run_stage(
        "Stage B + reels", "FIGHT vs GAME, reels added as extra positives",
        data,
        np.concatenate([rows, extra]),
        np.concatenate([y, np.ones(len(extra), dtype=int)]),
        STAGE_B_TARGET_PRECISION, STAGE_B_MIN_RECALL, purge_seconds,
    )

    print(f"\nDid the reels earn their place? {extra.size} extra positive rows, "
          "both fits scored on the same broadcast-only slice:")
    print(f"  {'':24s} {'broadcast only':>16s} {'+ reels':>12s} {'change':>10s}")
    for key, title in _REEL_COMPARISON:
        before = float(baseline.report[key])  # type: ignore[arg-type]
        after = float(with_reels.report[key])  # type: ignore[arg-type]
        print(f"  {title:24s} {before:16.4f} {after:12.4f} {after - before:+10.4f}")

    change = float(with_reels.report["average_precision"]) - float(  # type: ignore[arg-type]
        baseline.report["average_precision"]  # type: ignore[arg-type]
    )
    print("  the reels " + ("helped" if change > 0 else "did not help")
          + f" - average precision {change:+.4f}. The reel fit is what was saved,"
            " because asking for it is what --reel-positives means.")

    with_reels.report["reel_positives"] = {
        "extra_positive_rows": int(extra.size),
        "broadcast_only": {key: baseline.report[key] for key, _ in _REEL_COMPARISON},
        "with_reels": {key: with_reels.report[key] for key, _ in _REEL_COMPARISON},
    }
    return with_reels


# ---------------------------------------------------------------------------
# Saving
# ---------------------------------------------------------------------------


def _save(stage_a: Stage, stage_b: Stage, data: Dataset, out_dir: Path,
          purge_seconds: float, synthetic: bool) -> tuple[Path, Path]:
    """Write both heads and everything needed to apply them without guessing.

    The feature names travel *inside* the payload, in column order, because the
    one failure mode this cannot recover from is inference building its columns
    in a different order from training: the model still returns a probability,
    it is simply wrong, and nothing in the output says so.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    model_path = out_dir / "fight_model.joblib"
    card_path = out_dir / "model_card.json"

    joblib.dump(
        {
            "schema": 1,
            "feature_names": list(data.feature_names),
            "grid_seconds": config.GRID_SECONDS,
            "stage_a": {
                "model": stage_a.model,
                "threshold": stage_a.threshold,
                "positive": stage_a.positive,
            },
            "stage_b": {
                "model": stage_b.model,
                "threshold": stage_b.threshold,
                "positive": stage_b.positive,
            },
            "trained_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "synthetic": synthetic,
        },
        model_path,
    )

    sources = [
        {
            "source_id": str(name),
            "kind": "reel" if bool(data.from_reel[data.source == name][0]) else "broadcast",
            "rows": int((data.source == name).sum()),
        }
        for name in sorted(set(data.source.tolist()))
    ]

    card = {
        "trained_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "synthetic": synthetic,
        "feature_names": list(data.feature_names),
        "grid_seconds": config.GRID_SECONDS,
        "block_seconds": BLOCK_SECONDS,
        "purge_seconds": purge_seconds,
        "n_splits": N_SPLITS,
        "class_counts": data.counts(),
        "sources": sources,
        "stages": {"stage_a": stage_a.report, "stage_b": stage_b.report},
        "importance": {
            "stage_a": [[n, m, s] for n, m, s in stage_a.importance],
            "stage_b": [[n, m, s] for n, m, s in stage_b.importance],
        },
    }
    card_path.write_text(json.dumps(card, indent=2), encoding="utf-8")
    return model_path, card_path


# ---------------------------------------------------------------------------
# Synthetic self-test
# ---------------------------------------------------------------------------
#
# Not training data and never treated as such - it writes to models/synthetic/
# and stamps `"synthetic": true` on both artefacts. It exists so the machinery
# above can be exercised and, in particular, so the leakage controls can be
# shown to bite: run with --purge-seconds 0 and the validation number goes up,
# which is the whole argument for the purge in one measurement.
#
# The rows are deliberately *not* iid noise. Independent noise cannot leak, so a
# split tested against it would look perfect however broken it was. Each source
# here has segment-level offsets and an AR(1) drift, which is what makes
# neighbouring rows near-duplicates - the exact property that makes a random
# split dishonest on real footage.

# One step at the model's own grid rate. 0.97 gives a correlation half-life of
# roughly twenty rows, or ten seconds, which is about how long a camera holds a
# shot.
_AR_RHO = 0.97

# Spread of the per-segment offset and of the within-segment noise, both in the
# same units as the group means below. The offset is what carries across a block
# boundary and is therefore what the purge is defending against.
_SEGMENT_SPREAD = 0.30
_NOISE_SPREAD = 0.50

# Mean level of each feature group under each label. The audio-gunfire row is
# the important one: it is inverted, studio above fight, because that is what
# was measured on this footage (1.03 vs 0.43 onsets per second). If the
# importance table cannot notice that, the importance table is not working.
_GROUP_MEANS: dict[str, tuple[float, float, float]] = {
    #                studio  game   fight
    "hud":          (0.02,  0.95,  0.97),
    "motion":       (0.20,  0.45,  0.80),
    "flash":        (0.02,  0.08,  0.55),
    "fire":         (0.01,  0.05,  0.45),
    "killfeed":     (0.00,  0.03,  0.40),
    "ticks":        (0.10,  0.60,  0.50),
    "audio-caster": (0.30,  0.35,  0.75),
    "audio-gunfire": (0.55, 0.40,  0.42),
    "other":        (0.30,  0.30,  0.30),
}

# How long each kind of segment runs, in seconds, and where it goes next. Fights
# are short and rare, studio talk is long, which is the shape of a real day.
_SEGMENT_SECONDS = {
    labels.STUDIO: (45.0, 180.0),
    labels.GAME: (30.0, 150.0),
    labels.FIGHT: (10.0, 40.0),
}

_SYNTHETIC_FRAME_NAMES = (
    "hud_present", "hud_persistence_30s",
    "flash_rate", "flash_peak_8s",
    "fire_ratio", "fire_peak_8s",
    "motion_mean", "motion_p95", "motion_peak_30s",
    "killfeed_banners", "killfeed_peak_30s",
    "ticks_alive", "ticks_lost_30s",
)

# Columns whose name ends in a number of seconds are rolling summaries, and the
# fixture has to reproduce that or it cannot test the purge at all: without a
# window, a row's value depends on nothing but that row, no row can share input
# with a row on the other side of a block boundary, and a purge defending
# against exactly that has nothing to defend against. A first pass at this file
# left the windows out, and disabling the purge changed nothing - which was a
# property of the fixture, not evidence about the split.
#
# The lookbehind stops "luma_p99" and "frame_diff_p95" being read as 99 and 95
# second windows.
_WINDOW_IN_NAME = re.compile(r"(?<!p)(\d+)s?$")

# Longest window the fixture will honour. Anything past this is a misparse.
_MAX_WINDOW_SECONDS = 60.0


def _ar1(rng: np.random.Generator, n: int) -> np.ndarray:
    """Unit-variance noise that remembers where it just was."""
    out = np.empty(n, dtype=np.float64)
    innovation = rng.normal(0.0, np.sqrt(1.0 - _AR_RHO**2), n)
    value = rng.normal()
    for index in range(n):
        value = _AR_RHO * value + innovation[index]
        out[index] = value
    return out


def _centred_mean(values: np.ndarray, half: int) -> np.ndarray:
    """Mean over +/- half rows, so a row's value reaches either side of itself."""
    running = np.concatenate(([0.0], np.cumsum(values)))
    index = np.arange(len(values))
    low = np.maximum(index - half, 0)
    high = np.minimum(index + half + 1, len(values))
    return (running[high] - running[low]) / (high - low)


def _forward_max(values: np.ndarray, reach: int) -> np.ndarray:
    """Largest value in the next `reach` rows, this one included."""
    padded = np.concatenate((values, np.full(reach, -np.inf)))
    return np.lib.stride_tricks.sliding_window_view(padded, reach + 1).max(axis=1)


def _apply_windows(values: np.ndarray, names: tuple[str, ...]) -> np.ndarray:
    """Turn the named columns into rolling summaries of their neighbourhood.

    Centred windows stand in for the real schema's mix of trailing and centred
    ones; only the forward-looking caster feature is treated separately, because
    a forward window is the one case where a row can contain information from
    its own future and that is worth reproducing exactly.
    """
    out = values.astype(np.float64, copy=True)
    for index, name in enumerate(names):
        found = _WINDOW_IN_NAME.search(name)
        if not found:
            continue
        seconds = min(float(found.group(1)), _MAX_WINDOW_SECONDS)
        span = max(1, int(round(seconds / config.GRID_SECONDS)))
        if "ahead" in name:
            out[:, index] = _forward_max(out[:, index], span)
        else:
            out[:, index] = _centred_mean(out[:, index], max(1, span // 2))
    return out.astype(np.float32)


def _walk(rng: np.random.Generator, duration: float) -> list[tuple[int, float, float]]:
    """A day as a chain of segments: talk, play, fight, play, talk."""
    out: list[tuple[int, float, float]] = []
    state, start = labels.STUDIO, 0.0
    while start < duration:
        low, high = _SEGMENT_SECONDS[state]
        end = min(start + float(rng.uniform(low, high)), duration)
        out.append((state, start, end))
        start = end
        if state == labels.STUDIO:
            state = labels.GAME
        elif state == labels.GAME:
            state = labels.FIGHT if rng.random() < 0.55 else labels.STUDIO
        else:
            state = labels.GAME
    return out


def _paint(
    rng: np.random.Generator,
    times_s: np.ndarray,
    truth: np.ndarray,
    segment: np.ndarray,
    names: tuple[str, ...],
) -> np.ndarray:
    """Turn a label sequence into a feature table that is hard to split honestly."""
    groups = _feature_groups(names)
    values = np.zeros((len(times_s), len(names)), dtype=np.float64)
    segments = int(segment.max()) + 1

    for group, columns in groups.items():
        studio, game, fight = _GROUP_MEANS.get(group, _GROUP_MEANS["other"])
        level = np.where(
            truth == labels.FIGHT, fight, np.where(truth == labels.GAME, game, studio)
        )
        for column in columns:
            offsets = rng.normal(0.0, _SEGMENT_SPREAD, segments)[segment]
            values[:, column] = level + offsets + _NOISE_SPREAD * _ar1(rng, len(times_s))

    return _apply_windows(values, names)


def _synthetic_broadcast(
    rng: np.random.Generator, source_id: str, duration: float, names: tuple[str, ...]
) -> tuple[str, bool, np.ndarray, np.ndarray, np.ndarray]:
    times_s = np.arange(0.0, duration, config.GRID_SECONDS)
    truth = np.full(len(times_s), labels.STUDIO, dtype=np.int8)
    segment = np.zeros(len(times_s), dtype=int)

    for number, (state, start, end) in enumerate(_walk(rng, duration)):
        span = (times_s >= start) & (times_s < end)
        truth[span] = state
        segment[span] = number

    values = _paint(rng, times_s, truth, segment, names)

    # Only every VISION_SAMPLE_SECONDS carries a verdict, because that is what a
    # teacher costs, and each verdict then spreads the way the real ones do. A
    # self-test on a fully labelled broadcast would be testing a luxury the
    # pipeline never has.
    step = max(1, int(round(config.VISION_SAMPLE_SECONDS / config.GRID_SECONDS)))
    sparse = np.full(len(times_s), labels.UNKNOWN, dtype=np.int8)
    sparse[::step] = truth[::step]
    spread = labels.spread(
        labels.Labelled(source_id, times_s, sparse, from_reel=False),
        config.LABEL_SPREAD_SECONDS,
    )
    return source_id, False, times_s, values, spread.labels


def _synthetic_reel(
    rng: np.random.Generator, source_id: str, duration: float, names: tuple[str, ...]
) -> tuple[str, bool, np.ndarray, np.ndarray, np.ndarray]:
    """An edited reel: fights back to back, with cutaways the real rule rejects."""
    times_s = np.arange(0.0, duration, config.GRID_SECONDS)
    truth = np.full(len(times_s), labels.FIGHT, dtype=np.int8)
    segment = np.zeros(len(times_s), dtype=int)

    start, number = 0.0, 0
    while start < duration:
        fight_end = min(start + float(rng.uniform(15.0, 45.0)), duration)
        span = (times_s >= start) & (times_s < fight_end)
        truth[span] = labels.FIGHT
        segment[span] = number
        number += 1

        cut_end = min(fight_end + float(rng.uniform(2.0, 6.0)), duration)
        span = (times_s >= fight_end) & (times_s < cut_end)
        # A crowd shot or a caster cam: no overlay, so the real reel rule leaves
        # it unknown rather than calling it a fight.
        truth[span] = labels.STUDIO
        segment[span] = number
        number += 1
        start = cut_end

    values = _paint(rng, times_s, truth, segment, names)
    audio = len(audio_grid.FEATURE_NAMES)
    # A reel's soundtrack is music, which says nothing about combat.
    values[:, -audio:] = np.nan

    marked = labels.from_reel(source_id, times_s, values, names)
    return source_id, True, times_s, values, marked.labels


def load_synthetic() -> Dataset:
    """Six fake sources: two broadcasts for the negatives, four reels."""
    try:
        import frame_features

        frame_names = frame_features.FEATURE_NAMES
        print("Synthetic rows on the real frame_features schema.")
    except ImportError:
        frame_names = _SYNTHETIC_FRAME_NAMES
        print("frame_features is not written yet - using a stand-in vision schema.")
        print("The column names below are invented; the machinery under test is not.")

    names = tuple(frame_names) + audio_grid.FEATURE_NAMES
    rng = np.random.default_rng(RANDOM_STATE)

    parts = [
        _synthetic_broadcast(rng, "fake_broadcast_a", 3600.0, names),
        _synthetic_broadcast(rng, "fake_broadcast_b", 3600.0, names),
    ]
    parts += [
        _synthetic_reel(rng, f"fake_reel_{letter}", 600.0, names)
        for letter in "abcd"
    ]
    return _combine(parts, names)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--synthetic",
        action="store_true",
        help="train on generated rows instead of footage, into models/synthetic/",
    )
    parser.add_argument(
        "--purge-seconds",
        type=float,
        default=PURGE_SECONDS,
        help=(
            "how much training footage to drop either side of a validation "
            f"block (default {PURGE_SECONDS:g}). Set to 0 to see what the purge "
            "is worth - the validation numbers will go up, and they will be lies."
        ),
    )
    parser.add_argument(
        "--reel-positives",
        action="store_true",
        default=STAGE_B_REEL_POSITIVES,
        help=(
            "let the highlight reels into Stage B as extra positives (default "
            f"{'on' if STAGE_B_REEL_POSITIVES else 'off'}). Stage B is then "
            "fitted both ways and both results are printed on the same "
            "broadcast-only slice, so the reels have to show what they are worth."
        ),
    )
    arguments = parser.parse_args()

    print("Fight detector training")
    print("=" * 23)
    print(f"  block {BLOCK_SECONDS:g}s   purge {arguments.purge_seconds:g}s   "
          f"folds {N_SPLITS}")
    if arguments.purge_seconds <= 0:
        print("  PURGE DISABLED - every number below is inflated on purpose.")
    elif arguments.purge_seconds != PURGE_SECONDS:
        print(f"  purge overridden (derived value is {PURGE_SECONDS:g}s)")
    if arguments.reel_positives:
        print("  reels admitted to Stage B as extra positives - it will be "
              "fitted both ways and both results printed.")
    print()

    data = load_synthetic() if arguments.synthetic else load_real()

    counts = data.counts()
    print(f"\n{len(data.label)} labelled rows from {len(set(data.source.tolist()))} "
          f"source(s): " + "  ".join(f"{k}={v}" for k, v in counts.items()))
    _print_groups(data.feature_names, _feature_groups(data.feature_names))

    # Stage A: everything that is not the desk.
    stage_a_rows = np.flatnonzero(
        (data.label == labels.STUDIO) | (data.label == labels.GAME)
        | (data.label == labels.FIGHT)
    )
    stage_a_rows, stage_a_y = _drop_single_class_sources(
        "Stage A", data, stage_a_rows,
        (data.label[stage_a_rows] != labels.STUDIO).astype(int),
    )
    stage_a = _run_stage(
        "Stage A", "gameplay (GAME or FIGHT) vs STUDIO",
        data, stage_a_rows, stage_a_y,
        STAGE_A_TARGET_PRECISION, STAGE_A_MIN_RECALL, arguments.purge_seconds,
    )

    # Stage B only ever sees rows Stage A has already accepted, so it is trained
    # on gameplay alone. Training it on studio rows too would spend its capacity
    # re-learning a boundary Stage A already draws better.
    stage_b_rows = np.flatnonzero(
        (data.label == labels.GAME) | (data.label == labels.FIGHT)
    )
    stage_b_rows, stage_b_y = _drop_single_class_sources(
        "Stage B", data, stage_b_rows,
        (data.label[stage_b_rows] == labels.FIGHT).astype(int),
    )
    stage_b = _run_stage(
        "Stage B", "FIGHT vs GAME, on gameplay rows only",
        data, stage_b_rows, stage_b_y,
        STAGE_B_TARGET_PRECISION, STAGE_B_MIN_RECALL, arguments.purge_seconds,
    )
    if arguments.reel_positives:
        stage_b = _stage_b_with_reels(
            data, stage_b, stage_b_rows, stage_b_y, arguments.purge_seconds
        )

    out_dir = MODELS_DIR / "synthetic" if arguments.synthetic else MODELS_DIR
    model_path, card_path = _save(
        stage_a, stage_b, data, out_dir, arguments.purge_seconds, arguments.synthetic
    )
    print(f"\nWrote {model_path}")
    print(f"Wrote {card_path}")

    leaked = not (
        stage_a.report["leakage"]["passed"] and stage_b.report["leakage"]["passed"]  # type: ignore[index]
    )
    if leaked:
        print("\nA LEAKAGE CHECK FAILED. The scores above are not evidence of "
              "anything. Fix the split before reading them.")
        return 2
    if not (stage_a.report["target_met"] and stage_b.report["target_met"]):
        print("\nTargets not met. The model is saved and the thresholds are the "
              "best available, but they do not reach the requirement.")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
