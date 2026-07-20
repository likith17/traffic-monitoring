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
from functools import lru_cache
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


def _throttle(blocking: bool = True) -> bool:
    """Space out Nominatim calls to respect the 1 req/s policy.

    blocking=True  (address lookups): wait out the interval, then proceed.
    blocking=False (type-ahead): never sleep - return False so the caller can
    skip the network entirely.  Sleeping here would freeze the Streamlit
    script thread on every keystroke, which looks like the page reloading
    before results appear.
    """
    global _last_request_at
    wait = _MIN_INTERVAL_S - (time.monotonic() - _last_request_at)
    if wait > 0:
        if not blocking:
            return False
        time.sleep(wait)
    _last_request_at = time.monotonic()
    return True

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


@lru_cache(maxsize=4)
def _load_cameras(cameras_csv: str) -> pd.DataFrame | None:
    """Camera list with a pre-lowercased name column, read from disk once.

    Cached because type-ahead calls this on every keystroke; re-parsing the
    CSV each time is pure waste.  Returns None when the file is missing.
    """
    try:
        cams = pd.read_csv(cameras_csv).dropna(subset=["lat", "lon"])
    except FileNotFoundError:
        return None
    cams["_name_lower"] = cams["name"].str.lower()
    return cams


def _camera_matches(
    query: str,
    cameras_csv: str = "manhattan_cameras.csv",
    limit: int = 7,
) -> list[dict[str, Any]]:
    """Offline substring match on DOT camera names.

    'amsterdam 60' matches 'Amsterdam Ave @ 60 St' — every query token must
    appear in the camera name.  Returns up to `limit` hits for type-ahead.
    """
    cams = _load_cameras(cameras_csv)
    if cams is None:
        return []

    tokens = [t for t in query.lower().replace("@", " ").split() if t]
    if not tokens:
        return []

    names = cams["_name_lower"]
    mask = pd.Series(True, index=cams.index)
    for t in tokens:
        mask &= names.str.contains(t, regex=False)

    matches = cams[mask].head(limit)
    return [
        {
            "lat": float(row["lat"]),
            "lon": float(row["lon"]),
            "label": str(row["name"]),
            "source": "camera",
        }
        for _, row in matches.iterrows()
    ]


def _camera_match(query: str, cameras_csv: str = "manhattan_cameras.csv") -> dict[str, Any] | None:
    """Best single camera-name match (used by geocode fallback)."""
    hits = _camera_matches(query, cameras_csv=cameras_csv, limit=1)
    return hits[0] if hits else None


def _nominatim_suggest(query: str, limit: int = 5) -> list[dict[str, Any]]:
    """Multi-hit bounded Nominatim lookup used for type-ahead suggestions.

    Non-blocking: if the rate-limit window hasn't elapsed we return nothing
    rather than sleeping, because this runs on every keystroke.  The local
    landmark and camera-name suggestions still answer instantly.
    """
    try:
        if not _throttle(blocking=False):
            return []
        r = requests.get(
            NOMINATIM_URL,
            params={
                "q": query,
                "format": "json",
                "limit": limit,
                "viewbox": ",".join(str(c) for c in COVERAGE),
                "bounded": 1,
                "countrycodes": "us",
            },
            headers={"User-Agent": USER_AGENT},
            timeout=REQUEST_TIMEOUT_S,
        )
        r.raise_for_status()
        out = []
        for hit in r.json():
            lat, lon = float(hit["lat"]), float(hit["lon"])
            if not in_coverage(lat, lon):
                continue
            # "Times Square, Broadway, Manhattan, ..." -> "Times Square, Broadway"
            label = ", ".join(hit.get("display_name", query).split(",")[:2]).strip()
            out.append({"lat": lat, "lon": lon, "label": label, "source": "nominatim"})
        return out
    except Exception:
        return []


# Curated landmarks so typing "time" / "wall" / "columbia" always suggests
# something useful even before Nominatim answers (and when offline).
LANDMARKS: list[dict[str, Any]] = [
    {"label": "Times Square", "lat": 40.7580, "lon": -73.9855},
    {"label": "Wall Street", "lat": 40.7074, "lon": -74.0113},
    {"label": "Columbia University", "lat": 40.8075, "lon": -73.9626},
    {"label": "Columbus Circle", "lat": 40.7681, "lon": -73.9819},
    {"label": "Grand Central Terminal", "lat": 40.7527, "lon": -73.9772},
    {"label": "Empire State Building", "lat": 40.7484, "lon": -73.9857},
    {"label": "Penn Station", "lat": 40.7506, "lon": -73.9935},
    {"label": "Union Square", "lat": 40.7359, "lon": -73.9911},
    {"label": "Washington Square Park", "lat": 40.7308, "lon": -73.9973},
    {"label": "Central Park (south)", "lat": 40.7660, "lon": -73.9760},
    {"label": "Harlem Hospital", "lat": 40.8140, "lon": -73.9405},
    {"label": "Battery Park", "lat": 40.7033, "lon": -74.0170},
    {"label": "Lincoln Center", "lat": 40.7725, "lon": -73.9835},
    {"label": "Brooklyn Bridge (Manhattan)", "lat": 40.7061, "lon": -73.9969},
    {"label": "Port Authority Bus Terminal", "lat": 40.7570, "lon": -73.9900},
]


def suggest_places(query: str, max_results: int = 7) -> list[dict[str, Any]]:
    """Google-Maps-style suggestions while typing.

    Priority: curated landmarks → DOT camera names (instant, offline) →
    Nominatim (only if we still need more hits).  Every returned dict has
    lat / lon / label inside the coverage area, so picking a suggestion can
    never produce an "outside Manhattan" error.
    """
    query = (query or "").strip()
    if len(query) < 2:
        return []

    results: list[dict[str, Any]] = []
    seen: set[str] = set()
    q = query.lower()

    def _add(hit: dict[str, Any]) -> None:
        key = hit["label"].lower()
        if key in seen or not in_coverage(hit["lat"], hit["lon"]):
            return
        seen.add(key)
        results.append({**hit, "source": hit.get("source", "landmark")})

    for lm in LANDMARKS:
        if q in lm["label"].lower():
            _add(lm)

    for cam in _camera_matches(query, limit=max_results):
        _add(cam)

    # Only hit the network when local suggestions aren't enough — keeps
    # type-ahead snappy and respects Nominatim's 1 req/s limit.
    if len(results) < 3:
        for hit in _nominatim_suggest(f"{query}, Manhattan, New York", limit=max_results):
            _add(hit)

    return results[:max_results]


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

    sugg = suggest_places("union sq")
    print(f"Suggestions for 'union sq': {[s['label'] for s in sugg]}")
    assert all(in_coverage(s["lat"], s["lon"]) for s in sugg)

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
