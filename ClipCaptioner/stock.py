"""Fetch stock footage from Pexels for b-roll and filler.

Two features need video that is not the podcast: a cutaway when the speaker
mentions something concrete, and a strip of ambient motion under the talking
head to hold attention.

Both could be fed by grabbing clips off YouTube, and that is how the channel
gets struck. Pexels footage is licensed for commercial and monetised use with
no attribution required, which is the difference between a feature and a
liability.

Everything fetched is cached under STOCK_DIR, so a second clip mentioning the
sea costs nothing and the library grows into something usable offline.
"""

from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

import config

ENV_KEY = "PEXELS_API_KEY"

_SEARCH = "https://api.pexels.com/videos/search"

# Pexels answers an unrecognised client with 403 rather than anything
# descriptive, so a plain urllib request looks exactly like a bad key. Cost an
# hour to find; the header is not optional.
_HEADERS = {"User-Agent": "Mozilla/5.0", "Accept": "application/json"}


class StockUnavailable(Exception):
    """Raised when footage cannot be fetched."""


def available() -> bool:
    return bool(os.environ.get(ENV_KEY))


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")[:60] or "clip"


def _cache_dir(topic: str) -> Path:
    path = config.STOCK_DIR / _slug(topic)
    path.mkdir(parents=True, exist_ok=True)
    return path


def cached(topic: str) -> list[Path]:
    """Footage already on disk for this topic."""
    folder = config.STOCK_DIR / _slug(topic)
    if not folder.exists():
        return []
    return sorted(p for p in folder.glob("*.mp4") if p.stat().st_size > 0)


def _request(url: str) -> dict:
    key = os.environ.get(ENV_KEY)
    if not key:
        raise StockUnavailable(f"{ENV_KEY} is not set.")

    request = urllib.request.Request(url, headers={**_HEADERS, "Authorization": key})
    try:
        with urllib.request.urlopen(request, timeout=config.STOCK_TIMEOUT_S) as response:
            return json.load(response)
    except urllib.error.HTTPError as exc:
        raise StockUnavailable(f"Pexels returned HTTP {exc.code}") from exc
    except Exception as exc:
        raise StockUnavailable(f"{type(exc).__name__}: {exc}") from exc


def _short_side(file: dict) -> int:
    return min(file.get("width") or 0, file.get("height") or 0)


def _pick_file(video: dict) -> str | None:
    """The smallest file whose short side still covers the frame.

    Judged on the short side rather than the height. Portrait stock is tall by
    definition, so a height test happily accepts a 720x1280 clip that is only
    720 wide and then upscales it half again to fill a 1080-wide frame.

    Among the files that clear the bar, the smallest wins: biggest-available
    would fetch 4K to show for one second.
    """
    files = [f for f in video.get("video_files", []) if f.get("link")]
    if not files:
        return None

    big_enough = [f for f in files if _short_side(f) >= config.STOCK_MIN_SHORT_SIDE]
    if big_enough:
        return min(big_enough, key=_short_side)["link"]
    return max(files, key=_short_side)["link"]


def fetch(topic: str, want: int = 1) -> list[Path]:
    """Footage for a topic, from the cache if it is there and Pexels if not.

    Never raises for want of footage. A missing cutaway is a clip without a
    cutaway; a raised exception here would cost the whole render.
    """
    have = cached(topic)
    if len(have) >= want:
        return have[:want]

    if not available():
        return have

    query = urllib.parse.urlencode(
        {
            "query": topic,
            "per_page": max(want * 2, 4),
            "orientation": config.STOCK_ORIENTATION,
            "size": "medium",
        }
    )

    try:
        payload = _request(f"{_SEARCH}?{query}")
    except StockUnavailable as exc:
        print(f"    stock footage unavailable ({exc})")
        return have

    folder = _cache_dir(topic)
    got = list(have)

    for video in payload.get("videos", []):
        if len(got) >= want:
            break

        # Long clips are whole scenes; a cutaway needs a second or two and a
        # long download is wasted on footage that is cut away from immediately.
        if (video.get("duration") or 0) > config.STOCK_MAX_SOURCE_SECONDS:
            continue

        link = _pick_file(video)
        if not link:
            continue

        destination = folder / f"{video.get('id', 'x')}.mp4"
        if destination.exists() and destination.stat().st_size > 0:
            got.append(destination)
            continue

        try:
            request = urllib.request.Request(link, headers=_HEADERS)
            with urllib.request.urlopen(request, timeout=config.STOCK_TIMEOUT_S) as response:
                destination.write_bytes(response.read())
        except Exception as exc:
            print(f"    could not download stock clip ({type(exc).__name__})")
            destination.unlink(missing_ok=True)
            continue

        got.append(destination)

    return got[:want]
