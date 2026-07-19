# Free-text place search bounded to Manhattan, like typing into Google Maps.
#
# Primary: OSM Nominatim with a hard viewbox (bounded=1) around the camera
# coverage area, so "Times Square" resolves but "Newark Airport" does not.
# Fallback: fuzzy match against DOT camera names, which keeps search working
# fully offline (Docker demos, no network).
#
# Every result is re-validated against the coverage box; anything outside
# returns outcome="outside" so the UI can show a friendly "Manhattan only"
# message instead of routing into the void.

from __future__ import annotations

import time
from typing import Any

import pandas as pd
import requests

NOMINATIM_URL = "https://nominatim.openstreetmap.org/search"
# Nominatim usage policy requires an identifying agent.
USER_AGENT = "emergency-routing-smart-response/1.0"
REQUEST_TIMEOUT_S = 8

# Nominatim allows at most 1 request/second; space our calls accordingly so a
# start+destination pair never gets rate-limited into a fake "not found".
_MIN_INTERVAL_S = 1.05
_last_request_at = 0.0


def _throttle() -> None:
    global _last_request_at
    wait = _MIN_INTERVAL_S - (time.monotonic() - _last_request_at)
    if wait > 0:
        time.sleep(wait)
    _last_request_at = time.monotonic()

# Coverage box around the DOT cameras (west, south, east, north), padded a
# touch so riverside addresses still hit.
COVERAGE = (-74.03, 40.69, -73.90, 40.88)


def in_coverage(lat: float, lon: float) -> bool:
    west, south, east, north = COVERAGE
    return south <= lat <= north and west <= lon <= east


def _nominatim(query: str) -> dict[str, Any] | None:
    """One bounded Nominatim lookup.  Returns None on any failure."""
    try:
        _throttle()
        r = requests.get(
            NOMINATIM_URL,
            params={
                "q": query,
                "format": "json",
                "limit": 1,
                "viewbox": ",".join(str(c) for c in COVERAGE),
                "bounded": 1,
                "countrycodes": "us",
            },
            headers={"User-Agent": USER_AGENT},
            timeout=REQUEST_TIMEOUT_S,
        )
        r.raise_for_status()
        hits = r.json()
        if not hits:
            return None
        hit = hits[0]
        return {
            "lat": float(hit["lat"]),
            "lon": float(hit["lon"]),
            "label": hit.get("display_name", query).split(",")[0],
            "source": "nominatim",
        }
    except Exception:
        return None


def _camera_match(query: str, cameras_csv: str = "manhattan_cameras.csv") -> dict[str, Any] | None:
    """Offline fallback: substring match on DOT camera names.

    'amsterdam 60' matches 'Amsterdam Ave @ 60 St' - every query token must
    appear in the camera name.
    """
    try:
        cams = pd.read_csv(cameras_csv).dropna(subset=["lat", "lon"])
    except FileNotFoundError:
        return None

    tokens = [t for t in query.lower().replace("@", " ").split() if t]
    if not tokens:
        return None

    names = cams["name"].str.lower()
    mask = pd.Series(True, index=cams.index)
    for t in tokens:
        mask &= names.str.contains(t, regex=False)

    matches = cams[mask]
    if matches.empty:
        return None
    hit = matches.iloc[0]
    return {
        "lat": float(hit["lat"]),
        "lon": float(hit["lon"]),
        "label": str(hit["name"]),
        "source": "camera",
    }


def geocode_manhattan(query: str) -> dict[str, Any]:
    """Resolve a free-text place to coordinates inside the coverage area.

    Returns {"outcome": "ok", "lat", "lon", "label", "source"} on success,
    {"outcome": "outside"} when the place exists but is out of Manhattan,
    {"outcome": "not_found"} when nothing matches at all.
    """
    query = (query or "").strip()
    if not query:
        return {"outcome": "not_found"}

    # Bias the online search towards Manhattan without polluting the label.
    result = _nominatim(f"{query}, Manhattan, New York")
    if result is None:
        result = _nominatim(query)

    if result is None:
        # Did the place exist somewhere else?  Unbounded probe purely to give
        # the user the right error message ("outside" vs "not found").
        try:
            _throttle()
            r = requests.get(
                NOMINATIM_URL,
                params={"q": query, "format": "json", "limit": 1},
                headers={"User-Agent": USER_AGENT},
                timeout=REQUEST_TIMEOUT_S,
            )
            r.raise_for_status()
            hits = r.json()
            if hits and not in_coverage(float(hits[0]["lat"]), float(hits[0]["lon"])):
                return {"outcome": "outside"}
        except Exception:
            pass
        # Fully offline or genuinely unknown: try the camera-name fallback.
        result = _camera_match(query)

    if result is None:
        return {"outcome": "not_found"}
    if not in_coverage(result["lat"], result["lon"]):
        return {"outcome": "outside"}

    result["outcome"] = "ok"
    return result


if __name__ == "__main__":
    # Offline fallback must always work.
    cam = _camera_match("amsterdam 60")
    assert cam is not None and "Amsterdam" in cam["label"], cam
    print(f"Camera fallback: {cam['label']} ({cam['lat']:.5f}, {cam['lon']:.5f})")

    # Bounded online search (tolerated to fail without network).
    ts = geocode_manhattan("Times Square")
    print(f"Times Square -> {ts}")
    assert ts["outcome"] == "ok", "Times Square should resolve inside coverage"

    outside = geocode_manhattan("Newark Airport")
    print(f"Newark Airport -> outcome={outside['outcome']}")
    assert outside["outcome"] in ("outside", "not_found")

    nonsense = geocode_manhattan("zzqqxx nowhere")
    assert nonsense["outcome"] == "not_found"
    print("geocode.py self-test OK")
