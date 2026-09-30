# Phase 2, step 1: get real ground truth to calibrate congestion against.
#
# The current congestion score uses hand-picked weights (vehicle 1.0, pedestrian
# 0.3, signal 0.5). Phase 2 aims to replace those guesses with values learned
# from data. That needs a real target to learn against, and this module supplies
# one: NYC's live traffic-speed feed, which reports measured vehicle speed on
# instrumented road links.
#
# The honest limitation, measured before building on it: these sensors cover
# highways and major arterials (FDR Drive, the West Side Highway, the East River
# bridges), not the surface streets most cameras watch. Only about 35 of the 373
# cameras sit within 60 m of a sensored link, and those are the ones physically
# on those roads. So this feed is valid ground truth for that subset only.
# Surface-street cameras cannot be speed-labelled this way, and this module does
# not pretend otherwise: it returns the matched subset with a distance, and the
# caller decides what is close enough to trust.
#
# A speed of 0 in the feed almost always means the sensor returned no reading,
# not gridlock, so it is surfaced as has_speed=False rather than as "stopped".

from __future__ import annotations

import math

import pandas as pd
import requests

# NYC Open Data, Real-Time Traffic Speed Data (Socrata dataset i4gi-tjb9).
SPEED_URL = "https://data.cityofnewyork.us/resource/i4gi-tjb9.json"

# A camera must be at least this close to a link to count as on that road.
# Chosen from the coverage measurement: at 60 m the matches are genuine (the
# camera's own road), and beyond ~100 m they start catching a parallel highway.
DEFAULT_MATCH_M = 60.0


def haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in metres between two lat/lon points."""
    r = 6_371_000.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


def _parse_points(link_points: str) -> list[tuple]:
    """Parse a link_points string ('lat,lon lat,lon ...') into coordinate pairs."""
    pts = []
    for tok in (link_points or "").replace("\n", " ").split():
        if "," in tok:
            try:
                a, b = tok.split(",")[:2]
                pts.append((float(a), float(b)))
            except ValueError:
                continue
    return pts


def fetch_speed_links(borough: str = "Manhattan", limit: int = 2000) -> list[dict]:
    """Pull the current speed feed for one borough.

    Each record carries a measured `speed` (mph), a `link_name`, and a
    `link_points` polyline of the road segment. Returns the raw records; parsing
    and matching happen in match_cameras.
    """
    r = requests.get(SPEED_URL, params={"borough": borough, "$limit": limit}, timeout=20)
    r.raise_for_status()
    return r.json()


def _nearest_link(clat: float, clon: float, links_parsed: list[tuple]) -> tuple:
    """Closest sensored link to a camera. Returns (dist_m, link_name, speed, link_id)."""
    best = (float("inf"), None, None, None)
    for speed, name, link_id, pts in links_parsed:
        for plat, plon in pts:
            d = haversine_m(clat, clon, plat, plon)
            if d < best[0]:
                best = (d, name, speed, link_id)
    return best


def match_cameras(
    cameras_csv: str = "manhattan_cameras.csv",
    max_dist_m: float = DEFAULT_MATCH_M,
    links: list[dict] | None = None,
) -> pd.DataFrame:
    """Match each camera to its nearest sensored link, keep only close matches.

    Returns one row per camera within max_dist_m of a link, with the measured
    speed and the match distance. Rows where the sensor returned 0 are kept but
    flagged has_speed=False, since 0 means "no reading" far more often than
    "stopped".
    """
    if links is None:
        links = fetch_speed_links()

    parsed = []
    for link in links:
        try:
            speed = float(link["speed"])
        except (KeyError, TypeError, ValueError):
            continue
        pts = _parse_points(link.get("link_points", ""))
        if pts:
            parsed.append((speed, link.get("link_name", "?"), link.get("link_id"), pts))

    cams = pd.read_csv(cameras_csv).dropna(subset=["lat", "lon"])

    rows = []
    for _, cam in cams.iterrows():
        dist, name, speed, link_id = _nearest_link(
            float(cam["lat"]), float(cam["lon"]), parsed
        )
        if dist <= max_dist_m:
            rows.append({
                "camera_id": cam["camera_id"],
                "camera_name": cam["name"],
                "lat": cam["lat"],
                "lon": cam["lon"],
                "link_id": link_id,
                "link_name": name,
                "speed_mph": speed,
                "has_speed": speed > 0,
                "dist_m": round(dist, 1),
            })

    return pd.DataFrame(rows)


if __name__ == "__main__":
    print("Fetching NYC real-time speed feed and matching cameras...")
    matched = match_cameras()

    if matched.empty:
        print("No cameras matched. Check the network or the feed URL.")
        raise SystemExit(1)

    with_speed = matched[matched["has_speed"]]
    print(f"\nCameras on a sensored road (within {DEFAULT_MATCH_M:.0f} m): {len(matched)}")
    print(f"  of those with a live speed reading (>0):            {len(with_speed)}")
    if len(with_speed):
        print(f"  speed range: {with_speed['speed_mph'].min():.0f}"
              f"-{with_speed['speed_mph'].max():.0f} mph, "
              f"mean {with_speed['speed_mph'].mean():.1f}")

    matched.to_csv("speed_matched_cameras.csv", index=False)
    print("\nSaved speed_matched_cameras.csv (the calibration subset for Phase 2).")
    print("\nThis is valid ground truth only for cameras genuinely on a sensored")
    print("road. Surface-street cameras keep the heuristic score; they cannot be")
    print("speed-labelled from this feed.")
