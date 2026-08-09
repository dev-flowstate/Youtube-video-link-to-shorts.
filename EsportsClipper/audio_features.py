"""Per-frame audio features, computed streaming.

A five hour broadcast is roughly 1.3 GB of 16 kHz float32, so nothing here
loads the track whole. Audio arrives in chunks from an FFmpeg pipe and each
chunk is turned into frame features immediately; only the small per-frame
arrays are kept.

Three features come out of one FFT, which is why they are computed together
rather than in separate passes:

* spectral flux - how sharply the spectrum changed, i.e. onset strength
* high-frequency ratio - what share of energy sits above 4 kHz
* speech-band energy - loudness where a voice actually lives
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np

import shared  # noqa: F401  (puts the sibling project on sys.path)
from ffmpeg_utils import ensure_ffmpeg_available

SAMPLE_RATE = 16_000
FFT_SIZE = 512
HOP = 256

# Frequency bins. rfft over 512 samples at 16 kHz gives 257 bins of 31.25 Hz.
_BIN_HZ = SAMPLE_RATE / FFT_SIZE
_HF_FROM = int(4000 / _BIN_HZ)          # gunfire is broadband, voice is not
_SPEECH_FROM = int(300 / _BIN_HZ)
_SPEECH_TO = int(3400 / _BIN_HZ)

# Below speech. Explosions and a venue crowd both live down here, and so does
# the rumble a mix picks up when a room is full. Summed in the same pass as the
# others, so it costs one more slice per chunk and nothing else.
_LF_FROM = int(60 / _BIN_HZ)
_LF_TO = int(250 / _BIN_HZ)

# Samples pulled from the pipe at a time. 60s keeps peak memory near 4 MB.
_CHUNK_SAMPLES = SAMPLE_RATE * 60


class FeatureExtractionFailed(Exception):
    """Raised when the audio could not be decoded or analysed."""


@dataclass(frozen=True)
class FrameFeatures:
    """One value per analysis frame, all the same length."""

    flux: np.ndarray
    hf_ratio: np.ndarray
    speech_energy: np.ndarray
    low_energy: np.ndarray
    hop_seconds: float

    @property
    def duration_s(self) -> float:
        return len(self.flux) * self.hop_seconds

    def times(self) -> np.ndarray:
        return np.arange(len(self.flux), dtype=float) * self.hop_seconds


def _frame(samples: np.ndarray) -> np.ndarray:
    """Slice a chunk into overlapping windows without copying per frame."""
    count = 1 + (len(samples) - FFT_SIZE) // HOP
    if count <= 0:
        return np.empty((0, FFT_SIZE), dtype=np.float32)

    index = np.arange(FFT_SIZE)[None, :] + HOP * np.arange(count)[:, None]
    return samples[index]


def _spectra(frames: np.ndarray, window: np.ndarray) -> np.ndarray:
    if frames.size == 0:
        return np.empty((0, FFT_SIZE // 2 + 1), dtype=np.float32)
    return np.abs(np.fft.rfft(frames * window, axis=1)).astype(np.float32)


def _open_audio(audio_path: Path) -> subprocess.Popen:
    ensure_ffmpeg_available()
    args = [
        "ffmpeg",
        "-v",
        "error",
        "-i",
        str(audio_path),
        "-vn",
        "-ac",
        "1",
        "-ar",
        str(SAMPLE_RATE),
        "-f",
        "s16le",
        "-",
    ]
    return subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE)


def extract(audio_path: Path) -> FrameFeatures:
    """Stream the whole track and return one row of features per frame."""
    window = np.hanning(FFT_SIZE).astype(np.float32)

    flux_parts: list[np.ndarray] = []
    hf_parts: list[np.ndarray] = []
    speech_parts: list[np.ndarray] = []
    low_parts: list[np.ndarray] = []

    previous_spectrum: np.ndarray | None = None
    carry = np.empty(0, dtype=np.float32)

    process = _open_audio(audio_path)
    assert process.stdout is not None

    try:
        while True:
            raw = process.stdout.read(_CHUNK_SAMPLES * 2)
            if not raw:
                break

            block = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
            # Frames must span the chunk seam, so the tail of the previous
            # block is prepended rather than dropped.
            samples = np.concatenate((carry, block)) if carry.size else block

            frames = _frame(samples)
            if frames.shape[0] == 0:
                carry = samples
                continue

            consumed = (frames.shape[0] - 1) * HOP + FFT_SIZE
            carry = samples[consumed - (FFT_SIZE - HOP) :].copy()

            spectra = _spectra(frames, window)

            # Flux is a difference, so it needs a frame to look back at. Later
            # chunks borrow the last frame of the previous one, which keeps the
            # seam from reading as a false onset. The very first frame of the
            # track has no predecessor, so it is compared against itself and
            # scores zero.
            #
            # Prepending something either way matters: without it the first
            # chunk yields one value fewer than it has frames, while every
            # other feature yields one per frame, and the arrays drift apart by
            # exactly one for the rest of the run.
            leading = previous_spectrum if previous_spectrum is not None else spectra[0]
            diff = np.diff(np.vstack((leading[None, :], spectra)), axis=0)
            flux_parts.append(np.maximum(diff, 0.0).sum(axis=1))
            previous_spectrum = spectra[-1]

            total = spectra.sum(axis=1)
            safe_total = np.maximum(total, 1e-9)
            hf_parts.append(spectra[:, _HF_FROM:].sum(axis=1) / safe_total)
            speech_parts.append(spectra[:, _SPEECH_FROM:_SPEECH_TO].sum(axis=1))
            low_parts.append(spectra[:, _LF_FROM:_LF_TO].sum(axis=1))

        stderr = process.stderr.read().decode("utf-8", "replace") if process.stderr else ""
        if process.wait() != 0:
            raise FeatureExtractionFailed(f"Could not decode audio: {stderr.strip()[-300:]}")
    finally:
        if process.poll() is None:
            process.kill()

    if not flux_parts:
        raise FeatureExtractionFailed("Audio produced no analysable frames.")

    flux = np.concatenate(flux_parts)
    hf_ratio = np.concatenate(hf_parts)
    speech_energy = np.concatenate(speech_parts)
    low_energy = np.concatenate(low_parts)

    # These are combined element-wise downstream, so any drift between them
    # surfaces hours later as an unreadable broadcast error. Caught here, where
    # the cause is still obvious.
    lengths = {len(flux), len(hf_ratio), len(speech_energy), len(low_energy)}
    if len(lengths) != 1:
        raise FeatureExtractionFailed(
            "Feature arrays came out different lengths "
            f"(flux {len(flux)}, hf {len(hf_ratio)}, speech {len(speech_energy)}, "
            f"low {len(low_energy)}) - the chunk seam handling is wrong."
        )

    return FrameFeatures(
        flux=flux,
        hf_ratio=hf_ratio,
        speech_energy=speech_energy,
        low_energy=low_energy,
        hop_seconds=HOP / SAMPLE_RATE,
    )
