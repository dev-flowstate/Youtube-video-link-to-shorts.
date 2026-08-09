"""One table per video: every feature, one row per half-second.

The vision pass owns the clock. It decodes at a fixed rate and its timestamps
are arithmetic - row `i` is at `i / fps` seconds, by construction. Audio is
then fitted onto those rows.

That direction matters. Deriving the two independently and matching them
afterwards is how a constant offset creeps in, and a constant offset does not
announce itself: every label attaches to the wrong row, the model learns noise,
and it still reports a plausible-looking number. Making one side authoritative
removes the opportunity.

The finished table is cached. Decoding five hours is the expensive step by a
wide margin, and once it is done the labels can be revised, the split changed
and the model retrained as often as needed without touching the video again.
The cache, not the mp4, is the artefact worth keeping.
"""

from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path

import numpy as np

import audio_features
import audio_grid
import config
import frame_features
import shared  # noqa: F401  (puts the sibling project on sys.path)
from ffmpeg_utils import ensure_ffmpeg_available

FEATURE_NAMES: tuple[str, ...] = frame_features.FEATURE_NAMES + audio_grid.FEATURE_NAMES

CACHE_VERSION = 1


class FeatureTableFailed(Exception):
    """Raised when a table could not be built."""


def cache_path(source_id: str) -> Path:
    return config.DATA_DIR / "features" / f"{source_id}.npz"


def _extract_audio(video: Path, work_dir: Path) -> Path:
    """Pull the soundtrack out to a file the audio pass can stream."""
    ensure_ffmpeg_available()
    destination = work_dir / "audio.wav"
    result = subprocess.run(
        [
            "ffmpeg", "-y", "-v", "error", "-i", str(video),
            "-vn", "-ac", "1", "-ar", str(audio_features.SAMPLE_RATE),
            str(destination),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0 or not destination.exists():
        raise FeatureTableFailed(
            f"Could not read audio from {video.name}: {(result.stderr or '').strip()[-200:]}"
        )
    return destination


def build(
    video: Path,
    source_id: str,
    use_audio: bool = True,
    max_seconds: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Build the full table for one video. Returns (times_s, values).

    `use_audio=False` is for edited reels. Their soundtrack is usually replaced
    with music, and music that plays over every positive example and nowhere
    else is a shortcut the model would take instead of learning what a fight
    looks like. Marked unknown, which is the truth, rather than zero.
    """
    fps = 1.0 / config.GRID_SECONDS
    times, vision = frame_features.extract(video, fps=fps, max_seconds=max_seconds)

    if len(times) == 0:
        raise FeatureTableFailed(f"{video.name} produced no frames.")

    if use_audio:
        try:
            with tempfile.TemporaryDirectory(prefix="table_") as tmp:
                audio_path = _extract_audio(video, Path(tmp))
                sound = audio_features.extract(audio_path)
            audio = audio_grid.build(sound, times)
        except (FeatureTableFailed, audio_features.FeatureExtractionFailed) as exc:
            # A silent or damaged track is a reason to know less, not to lose
            # the vision features that carry the signal anyway.
            print(f"    audio unavailable ({exc}) - continuing without it")
            audio = audio_grid.missing(len(times))
    else:
        audio = audio_grid.missing(len(times))

    values = np.hstack([vision, audio]).astype(np.float32)

    if values.shape[1] != len(FEATURE_NAMES):
        raise FeatureTableFailed(
            f"Built {values.shape[1]} columns but {len(FEATURE_NAMES)} are named. "
            "The vision and audio schemas have drifted apart."
        )
    return times, values


def load_or_build(
    video: Path,
    source_id: str,
    use_audio: bool = True,
    max_seconds: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """The cached table if it matches the current schema, else a fresh one."""
    path = cache_path(source_id)

    if path.exists():
        try:
            cached = np.load(path, allow_pickle=False)
            names_match = (
                list(cached["names"]) == list(FEATURE_NAMES)
                and int(cached["version"]) == CACHE_VERSION
            )
            if names_match:
                print(f"    {source_id}: cached table ({cached['values'].shape})")
                return cached["times"], cached["values"]
            print(f"    {source_id}: cache is from an older schema - rebuilding")
        except (OSError, KeyError, ValueError) as exc:
            print(f"    {source_id}: cache unreadable ({exc}) - rebuilding")

    times, values = build(video, source_id, use_audio=use_audio, max_seconds=max_seconds)

    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        times=times,
        values=values,
        names=np.array(FEATURE_NAMES),
        version=np.array(CACHE_VERSION),
    )
    print(f"    {source_id}: built and cached {values.shape}")
    return times, values


def describe(times: np.ndarray, values: np.ndarray) -> None:
    """Print each column's range and how often it is missing."""
    print(f"\n{values.shape[0]} rows x {values.shape[1]} features "
          f"({times[-1] / 60:.1f} min)\n")
    print(f"  {'feature':26s} {'min':>10s} {'max':>10s} {'mean':>10s} {'NaN%':>7s}")
    for index, name in enumerate(FEATURE_NAMES):
        column = values[:, index]
        finite = column[np.isfinite(column)]
        missing = 100.0 * (1.0 - len(finite) / max(1, len(column)))
        if len(finite) == 0:
            print(f"  {name:26s} {'-':>10s} {'-':>10s} {'-':>10s} {missing:6.1f}%")
            continue
        print(
            f"  {name:26s} {finite.min():10.3f} {finite.max():10.3f} "
            f"{finite.mean():10.3f} {missing:6.1f}%"
        )
