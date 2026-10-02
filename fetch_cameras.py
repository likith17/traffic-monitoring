# Step 1 of the pipeline: pull the full NYC DOT camera list for all five boroughs.
# Run this once (or whenever you want a fresh camera index) before
# update_camera_stats.py. Output: manhattan_cameras.csv
#
# Note on the filename: it is kept as manhattan_cameras.csv because ~17 modules
# reference that path; it now holds the whole city, not just Manhattan. Renaming
# it to nyc_cameras.csv is a worthwhile later cleanup.

import requests
import pandas as pd

# Public NYC DOT API - no key required, returns all ~900 city-wide cameras.
API_URL = "https://webcams.nyctmc.org/api/cameras"

# The five NYC boroughs, as the API tags them in the "area" field. Anything
# outside these (if the feed ever includes it) is dropped, so coverage is the
# city and nothing beyond it.
NYC_BOROUGHS = {"Manhattan", "Brooklyn", "Queens", "Bronx", "Staten Island"}


def main():
    print("[INFO] Fetching camera list from NYC DOT API...")
    r = requests.get(API_URL, timeout=15)
    r.raise_for_status()
    cams = r.json()
    print(f"[INFO] Total cameras returned: {len(cams)}")

    kept = [c for c in cams if c.get("area") in NYC_BOROUGHS]
    print(f"[INFO] City-wide cameras kept (5 boroughs): {len(kept)}")

    rows = []
    for c in kept:
        rows.append({
            "camera_id": c["id"],
            "name": c["name"],
            "lat": c["latitude"],
            "lon": c["longitude"],
            # The API uses different key names across camera models; fall back to
            # the canonical image endpoint if neither key is present.
            "image_url": (
                c.get("imageUrl")
                or c.get("imageURL")
                or f"https://webcams.nyctmc.org/api/cameras/{c['id']}/image"
            ),
            "area": c.get("area", ""),
            "isOnline": c.get("isOnline", True),
        })

    df = pd.DataFrame(rows)
    df.to_csv("manhattan_cameras.csv", index=False)

    print("[INFO] Saved manhattan_cameras.csv (now city-wide):")
    print(df["area"].value_counts())
    print(f"[INFO] Total cameras stored: {len(df)}")


if __name__ == "__main__":
    main()
