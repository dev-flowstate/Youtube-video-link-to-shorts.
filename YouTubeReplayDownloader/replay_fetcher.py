"""Fetch YouTube 'Most Replayed' heatmap data from page JSON."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

import requests

from utils import parse_youtube_url


class ReplayDataNotAvailable(Exception):
    """Raised when the video has no replay heatmap."""


HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/138.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
}


@dataclass(frozen=True)
class HeatmapPoint:
    start_ms: int
    duration_ms: int
    score: float

    @property
    def center_s(self) -> float:
        return (self.start_ms + self.duration_ms / 2) / 1000.0

    @property
    def start_s(self) -> float:
        return self.start_ms / 1000.0

    @property
    def end_s(self) -> float:
        return (self.start_ms + self.duration_ms) / 1000.0


def get_html(url: str) -> str:
    response = requests.get(url, headers=HEADERS, timeout=30)
    if response.status_code != 200:
        raise RuntimeError(f"Failed to fetch page ({response.status_code}).")
    return response.text


def extract_yt_initial_data(html: str) -> dict[str, Any]:
    patterns = (
        r"var ytInitialData = (.*?);</script>",
        r"window\['ytInitialData'\]\s*=\s*(.*?);</script>",
        r'window\["ytInitialData"\]\s*=\s*(.*?);</script>',
    )

    for pattern in patterns:
        match = re.search(pattern, html, re.DOTALL)
        if match:
            return json.loads(match.group(1))

    raise ReplayDataNotAvailable("ytInitialData not found on the page.")


def recursive_find_key(obj: Any, key: str) -> Any | None:
    if isinstance(obj, dict):
        if key in obj:
            return obj[key]
        for value in obj.values():
            result = recursive_find_key(value, key)
            if result is not None:
                return result
    elif isinstance(obj, list):
        for item in obj:
            result = recursive_find_key(item, key)
            if result is not None:
                return result
    return None


def _normalize_legacy_markers(heat_markers: list[dict[str, Any]]) -> list[HeatmapPoint]:
    points: list[HeatmapPoint] = []

    for marker in heat_markers:
        try:
            renderer = marker["heatMarkerRenderer"]
            points.append(
                HeatmapPoint(
                    start_ms=int(renderer["timeRangeStartMillis"]),
                    duration_ms=int(renderer["markerDurationMillis"]),
                    score=float(renderer["heatMarkerIntensityScoreNormalized"]),
                )
            )
        except (KeyError, TypeError, ValueError):
            continue

    return points


def _normalize_modern_markers(markers: list[dict[str, Any]]) -> list[HeatmapPoint]:
    points: list[HeatmapPoint] = []

    for marker in markers:
        try:
            points.append(
                HeatmapPoint(
                    start_ms=int(marker["startMillis"]),
                    duration_ms=int(marker["durationMillis"]),
                    score=float(marker["intensityScoreNormalized"]),
                )
            )
        except (KeyError, TypeError, ValueError):
            continue

    return points


def _extract_legacy_heatmap(yt_initial: dict[str, Any]) -> list[HeatmapPoint] | None:
    markers_map = recursive_find_key(yt_initial, "markersMap")
    if not markers_map:
        return None

    for marker in markers_map:
        try:
            renderer = marker["value"]["heatmap"]["heatmapRenderer"]
            heat_markers = renderer["heatMarkers"]
            if heat_markers:
                points = _normalize_legacy_markers(heat_markers)
                if points:
                    return points
        except (KeyError, TypeError):
            continue

    return None


def _extract_modern_heatmap(yt_initial: dict[str, Any]) -> list[HeatmapPoint] | None:
    entities: list[dict[str, Any]] = []

    def collect_entities(obj: Any) -> None:
        if isinstance(obj, dict):
            if "macroMarkersListEntity" in obj:
                entity = obj["macroMarkersListEntity"]
                if isinstance(entity, dict):
                    entities.append(entity)
            for value in obj.values():
                collect_entities(value)
        elif isinstance(obj, list):
            for item in obj:
                collect_entities(item)

    collect_entities(yt_initial)

    for entity in entities:
        markers_list = entity.get("markersList", {})
        marker_type = markers_list.get("markerType")
        if marker_type != "MARKER_TYPE_HEATMAP":
            continue

        markers = markers_list.get("markers", [])
        points = _normalize_modern_markers(markers)
        if points:
            return points

    # Fallback: any direct markersList with heatmap type.
    def find_markers_list(obj: Any) -> list[HeatmapPoint] | None:
        if isinstance(obj, dict):
            if obj.get("markerType") == "MARKER_TYPE_HEATMAP" and "markers" in obj:
                points = _normalize_modern_markers(obj["markers"])
                if points:
                    return points
            for value in obj.values():
                result = find_markers_list(value)
                if result is not None:
                    return result
        elif isinstance(obj, list):
            for item in obj:
                result = find_markers_list(item)
                if result is not None:
                    return result
        return None

    return find_markers_list(yt_initial)


def fetch_replay_data(youtube_url: str) -> list[HeatmapPoint]:
    """
    Fetch normalized replay heatmap points for a YouTube video.

    Supports both legacy and current YouTube page JSON formats.
    """
    canonical_url = parse_youtube_url(youtube_url)
    html = get_html(canonical_url)
    yt_initial = extract_yt_initial_data(html)

    points = _extract_modern_heatmap(yt_initial)
    if not points:
        points = _extract_legacy_heatmap(yt_initial)

    if not points:
        raise ReplayDataNotAvailable("No replay heatmap exists for this video.")

    points.sort(key=lambda point: point.start_ms)
    return points
