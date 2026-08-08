"""FFmpeg discovery and PATH setup for yt-dlp."""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path


class FFmpegNotFoundError(Exception):
    """Raised when FFmpeg is not available."""


_FFMPEG_DIR: Path | None = None


def _common_windows_paths() -> list[Path]:
    candidates: list[Path] = []

    local_app_data = os.environ.get("LOCALAPPDATA")
    if local_app_data:
        winget_packages = Path(local_app_data) / "Microsoft" / "WinGet" / "Packages"
        if winget_packages.exists():
            for match in winget_packages.glob("Gyan.FFmpeg_*"):
                for sub in match.glob("**/bin"):
                    if (sub / "ffmpeg.exe").exists():
                        candidates.append(sub)

    program_files = os.environ.get("ProgramFiles")
    if program_files:
        candidates.append(Path(program_files) / "ffmpeg" / "bin")

    for fixed in (
        Path(r"C:\ffmpeg\bin"),
        Path(r"C:\Program Files\ffmpeg\bin"),
    ):
        candidates.append(fixed)

    return candidates


def _imageio_ffmpeg_dir() -> Path | None:
    try:
        import imageio_ffmpeg
    except ImportError:
        return None

    ffmpeg_exe = Path(imageio_ffmpeg.get_ffmpeg_exe())
    if ffmpeg_exe.exists():
        return ffmpeg_exe.parent
    return None


def resolve_ffmpeg_dir() -> Path:
    """Return the directory containing ffmpeg (and ideally ffprobe)."""
    global _FFMPEG_DIR
    if _FFMPEG_DIR is not None:
        return _FFMPEG_DIR

    if shutil.which("ffmpeg"):
        ffmpeg_path = shutil.which("ffmpeg")
        assert ffmpeg_path is not None
        _FFMPEG_DIR = Path(ffmpeg_path).parent
        return _FFMPEG_DIR

    for directory in _common_windows_paths():
        if (directory / "ffmpeg.exe").exists():
            _FFMPEG_DIR = directory
            return _FFMPEG_DIR

    imageio_dir = _imageio_ffmpeg_dir()
    if imageio_dir is not None:
        _FFMPEG_DIR = imageio_dir
        return _FFMPEG_DIR

    raise FFmpegNotFoundError(
        "FFmpeg was not found. Install FFmpeg, run setup.bat, or: pip install imageio-ffmpeg"
    )


def _register_with_yt_dlp(ffmpeg_dir: Path) -> None:
    """Point yt-dlp's global FFmpeg lookup at the directory we resolved.

    Partial downloads are gated on FFmpegFD.available(), a classmethod that
    cannot see the per-download ffmpeg_location option and caches a failed
    lookup process-wide. Setting the ContextVar it consults re-keys that cache
    on the absolute binary path, so a stale negative result cannot stick.
    """
    try:
        from yt_dlp.postprocessor.ffmpeg import FFmpegPostProcessor
    except ImportError:
        return

    FFmpegPostProcessor._ffmpeg_location.set(str(ffmpeg_dir))  # type: ignore[arg-type]  # declared ContextVar[None] upstream


def ensure_ffmpeg_available() -> Path:
    """Ensure FFmpeg is available and prepend its folder to PATH."""
    ffmpeg_dir = resolve_ffmpeg_dir()
    ffmpeg_exe = ffmpeg_dir / ("ffmpeg.exe" if os.name == "nt" else "ffmpeg")

    if not ffmpeg_exe.exists():
        raise FFmpegNotFoundError(f"ffmpeg executable missing: {ffmpeg_exe}")

    path_entries = os.environ.get("PATH", "").split(os.pathsep)
    ffmpeg_dir_str = str(ffmpeg_dir)
    if ffmpeg_dir_str not in path_entries:
        os.environ["PATH"] = ffmpeg_dir_str + os.pathsep + os.environ.get("PATH", "")

    _register_with_yt_dlp(ffmpeg_dir)
    return ffmpeg_dir


def ffmpeg_location_option() -> dict[str, str]:
    """yt-dlp option pointing at the FFmpeg binary directory."""
    ffmpeg_dir = ensure_ffmpeg_available()
    return {"ffmpeg_location": str(ffmpeg_dir)}


def run_ffprobe(output_path: Path) -> None:
    ensure_ffmpeg_available()
    ffprobe = shutil.which("ffprobe")
    if ffprobe is None:
        return

    cmd = [
        ffprobe,
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(output_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"ffprobe could not validate {output_path.name}.")
