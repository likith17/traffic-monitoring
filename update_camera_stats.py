# Step 2 of the pipeline: downloads a live snapshot from every Manhattan camera,
# runs YOLOv12 on each image, and writes congestion scores to camera_stats.csv.
# Run this after fetch_cameras.py.  Expect it to take several minutes for ~200 cameras.

import time
import requests
import numpy as np
import pandas as pd
import cv2
from pathlib import Path

from routing.detect import get_detector


def compute_congestion(counts: dict):
    """Convert a {class_name: count} dict from one YOLO frame into a congestion score.

    Vehicles carry the most weight (1.0 each).  Pedestrians matter less (0.3) because
    they don't block lanes.  Traffic signals count partially (0.5) as they mark busy
    intersections.  Score thresholds: <5 low, 5–14 medium, 15+ high.
    """
    vehicles = (
        counts.get("car", 0)
        + counts.get("bus", 0)
        + counts.get("truck", 0)
        + counts.get("motorcycle", 0)
    )
    pedestrians = counts.get("person", 0)
    signals = (
        counts.get("traffic light", 0)
        + counts.get("stop sign", 0)
    )

    score = vehicles * 1.0 + pedestrians * 0.3 + signals * 0.5

    if score < 5:
        level = "low"
    elif score < 15:
        level = "medium"
    else:
        level = "high"

    return score, level, vehicles, pedestrians, signals


def load_model():
    """Load the detector once - building the session parses the whole model."""
    print("[INFO] Loading YOLOv12 (ONNX Runtime) ...")
    model = get_detector()
    print(f"[INFO] Model loaded, {len(model.names)} classes.")
    return model


def fetch_frame(url: str):
    """Download a JPEG snapshot from the camera URL and decode it into a NumPy array.

    Returns None (with a warning) if the request fails or the image can't be decoded.
    """
    try:
        r = requests.get(url, timeout=8)
        r.raise_for_status()
        arr = np.frombuffer(r.content, np.uint8)
        frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if frame is None:
            # imdecode returns None for corrupt or non-image responses.
            raise ValueError("cv2.imdecode returned None")
        return frame
    except Exception as e:
        print(f"[WARN] Failed to fetch frame from {url}: {e}")
        return None


def analyze_camera_row(model, row: pd.Series):
    """Fetch a snapshot for one camera row and run YOLO on it.

    Returns a dict of stats, or None if the camera is unreachable or inference fails.
    """
    frame = fetch_frame(row["image_url"])
    if frame is None:
        return None

    try:
        counts = model.class_counts(frame)
    except Exception as e:
        print(f"[WARN] YOLO inference failed for {row['camera_id']}: {e}")
        return None

    score, level, vehicles, pedestrians, signals = compute_congestion(counts)

    return {
        "camera_id": row["camera_id"],
        "name": row["name"],
        "lat": row["lat"],
        "lon": row["lon"],
        "area": row.get("area", "Manhattan"),
        "score": score,
        "level": level,
        "vehicles": vehicles,
        "pedestrians": pedestrians,
        "signals": signals,
    }


def main():
    cams = pd.read_csv("manhattan_cameras.csv")
    print(f"[INFO] Computing congestion for {len(cams)} Manhattan cameras...")

    model = load_model()
    rows = []

    for idx, row in cams.iterrows():
        print(f"[INFO] Analyzing camera {idx+1}/{len(cams)} – {row['name']}")
        info = analyze_camera_row(model, row)
        if info is not None:
            rows.append(info)
        # Brief pause to avoid hammering the NYC DOT image server.
        time.sleep(0.5)

    if not rows:
        print("[ERROR] No camera stats could be computed.")
        return

    stats_df = pd.DataFrame(rows)
    stats_df.to_csv("camera_stats.csv", index=False)

    print("[INFO] Saved camera_stats.csv:")
    print(stats_df.head())
    print(f"[INFO] Total cameras with stats: {len(stats_df)}")


if __name__ == "__main__":
    main()
