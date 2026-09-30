# Step 2 of the pipeline: downloads a live snapshot from every Manhattan camera,
# runs YOLOv12 on each image, and writes congestion scores to camera_stats.csv.
# Run this after fetch_cameras.py.  Expect it to take several minutes for ~200 cameras.

import argparse
import statistics
import time
import requests
import numpy as np
import pandas as pd
import cv2
from pathlib import Path

from routing.detect import get_detector, assess_frame_quality


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


def analyze_camera_temporal(model, row: pd.Series, polls: int = 5, spacing: float = 2.0):
    """Score a camera from several frames instead of one, and take the median.

    A single DOT snapshot is noisy: measured across live cameras, a lone frame
    misses the multi-frame median by about 38 percent of the score, because a
    momentary gap or a passing truck swings a single count. Polling the feed a
    few times (it returns a fresh frame every few seconds) and taking the median
    of each object count removes that swing. The median is used rather than the
    mean so a single corrupt or mis-detected frame cannot drag the result.

    Returns the same dict shape as analyze_camera_row, plus the number of frames
    that actually contributed, or None if too few frames were usable.
    """
    per_frame = []  # (vehicles, pedestrians, signals) for each usable frame
    for i in range(polls):
        frame = fetch_frame(row["image_url"])
        if frame is not None:
            usable, why = assess_frame_quality(frame)
            if not usable:
                # A dark or placeholder frame's zero count is not "clear road";
                # skip it rather than let it drag the median toward empty.
                print(f"[WARN] skipping {why} frame for {row['camera_id']}")
            else:
                try:
                    _, _, v, p, s = compute_congestion(model.class_counts(frame))
                    per_frame.append((v, p, s))
                except Exception as e:
                    print(f"[WARN] inference failed for {row['camera_id']}: {e}")
        if i < polls - 1:
            time.sleep(spacing)

    if len(per_frame) < max(2, polls // 2):
        return None  # too few usable frames to be more reliable than a single shot

    veh = statistics.median(f[0] for f in per_frame)
    ped = statistics.median(f[1] for f in per_frame)
    sig = statistics.median(f[2] for f in per_frame)

    # Recompute the score and level from the median component counts, so the
    # thresholds apply to the stabilised value rather than to a noisy frame.
    counts = {"car": veh, "person": ped, "traffic light": sig}
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
        "frames_used": len(per_frame),
    }


def main():
    parser = argparse.ArgumentParser(description="Score Manhattan cameras for congestion")
    parser.add_argument("--temporal", action="store_true",
                        help="poll each camera several times and take the median "
                             "(slower, but far less noisy than a single frame)")
    parser.add_argument("--polls", type=int, default=5,
                        help="frames per camera in temporal mode (default 5)")
    args = parser.parse_args()

    cams = pd.read_csv("manhattan_cameras.csv")
    mode = "temporal (median of frames)" if args.temporal else "single-frame"
    print(f"[INFO] Computing congestion for {len(cams)} cameras, {mode} mode...")
    if args.temporal:
        print(f"[INFO] Temporal mode polls {args.polls} frames per camera; "
              f"expect roughly {args.polls}x the runtime.")

    model = load_model()
    rows = []

    for idx, row in cams.iterrows():
        print(f"[INFO] Analyzing camera {idx+1}/{len(cams)} - {row['name']}")
        if args.temporal:
            info = analyze_camera_temporal(model, row, polls=args.polls)
        else:
            info = analyze_camera_row(model, row)
            time.sleep(0.5)  # brief pause; temporal mode already spaces its polls
        if info is not None:
            rows.append(info)

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
