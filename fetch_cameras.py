# Step 1 of the pipeline: pull the full NYC DOT camera list and keep only Manhattan cameras.
# Run this once (or whenever you want a fresh camera index) before running update_camera_stats.py.
# Output: manhattan_cameras.csv

import requests
import pandas as pd

# Public NYC DOT API — no key required, returns all ~900 city-wide cameras.
API_URL = "https://webcams.nyctmc.org/api/cameras"


def main():
    print("[INFO] Fetching camera list from NYC DOT API...")
    r = requests.get(API_URL, timeout=10)
    r.raise_for_status()
    cams = r.json()

    print(f"[INFO] Total cameras returned: {len(cams)}")

    # Keep only cameras that the API tags as "Manhattan"; skip all other boroughs.
    manhattan = [c for c in cams if c.get("area") == "Manhattan"]
    print(f"[INFO] Manhattan cameras found: {len(manhattan)}")

    rows = []
    for c in manhattan:
        rows.append({
            "camera_id": c["id"],
            "name": c["name"],
            "lat": c["latitude"],
            "lon": c["longitude"],
            # The API uses different key names across camera models; fall back to the
            # canonical image endpoint if neither key is present.
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

    print("[INFO] Saved manhattan_cameras.csv:")
    print(df.head())
    print(f"[INFO] Total Manhattan cameras stored: {len(df)}")


if __name__ == "__main__":
    main()
