# Offline segment analysis: runs YOLOv12 on pre-recorded MP4 clips and writes
# per-segment averages to segment_stats.csv.  Used by the "Segments" tab in dashboard.py.
# Run once after you have the video clips in the videos/ directory.

import cv2
import numpy as np
import pandas as pd
from routing.detect import get_detector
from pathlib import Path


# Each entry maps a human-readable road segment to a local video file.
# Add or remove entries here to analyse different locations.
SEGMENTS = [
    {
        "segment_id": "seg_a",
        "name": "Segment A – Midtown",
        "lat": 40.7580,
        "lon": -73.9855,
        "video_path": "videos/seg_a.mp4",
    },
    {
        "segment_id": "seg_b",
        "name": "Segment B – 7th Avenue",
        "lat": 40.7527,
        "lon": -73.9772,
        "video_path": "videos/seg_b.mp4",
    },
    {
        "segment_id": "seg_c",
        "name": "Segment C – Broadway",
        "lat": 40.7420,
        "lon": -73.9920,
        "video_path": "videos/seg_c.mp4",
    },
]


def compute_congestion(counts: dict):
    """Turn a {class_name: count} dict into a numeric congestion score and label.

    Weights: each vehicle counts as 1.0, each pedestrian as 0.3 (lower road impact),
    and each traffic signal as 0.5 (indicates an intersection, not just volume).
    Thresholds: <5 = low, 5–14 = medium, 15+ = high.
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


def analyze_segment(seg: dict, model, max_frames: int = 60, stride: int = 10):
    """Sample up to max_frames evenly spaced frames from the video and average the scores.

    stride=10 means we look at every 10th frame, which is fast enough for offline
    analysis while still capturing traffic variation across the clip.
    The model is passed in so it is loaded only once across all segments.
    """
    video_path = Path(seg["video_path"])
    if not video_path.exists():
        print(f"[WARN] Video not found for {seg['segment_id']}: {video_path}")
        return None

    print(f"[INFO] Analyzing segment {seg['segment_id']} from {video_path}...")


    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"[WARN] Could not open video: {video_path}")
        return None

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"[INFO] Video frames: {frame_count}")

    scores = []
    vehicles_list = []
    peds_list = []
    signals_list = []

    processed = 0
    idx = 0  # absolute frame index (including skipped frames)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Skip frames that don't fall on the stride boundary.
        if idx % stride != 0:
            idx += 1
            continue

        counts = model.class_counts(frame)
        score, level, vehicles, peds, signals = compute_congestion(counts)
        scores.append(score)
        vehicles_list.append(vehicles)
        peds_list.append(peds)
        signals_list.append(signals)

        processed += 1
        idx += 1

        if processed >= max_frames:
            break

    cap.release()

    if processed == 0:
        print(f"[WARN] No frames processed for segment {seg['segment_id']}")
        return None

    # Average all sampled frames into a single representative score.
    avg_score = float(np.mean(scores))
    avg_vehicles = float(np.mean(vehicles_list))
    avg_peds = float(np.mean(peds_list))
    avg_signals = float(np.mean(signals_list))

    if avg_score < 5:
        level = "low"
    elif avg_score < 15:
        level = "medium"
    else:
        level = "high"

    return {
        "segment_id": seg["segment_id"],
        "name": seg["name"],
        "lat": seg["lat"],
        "lon": seg["lon"],
        "avg_score": avg_score,
        "avg_vehicles": avg_vehicles,
        "avg_pedestrians": avg_peds,
        "avg_signals": avg_signals,
        "level": level,
        "video_path": str(video_path),
    }


def main():
    # Load the model once here so we don't pay the 5–10 second startup cost per segment.
    print("[INFO] Loading YOLOv12 (ONNX Runtime) ...")
    model = get_detector()
    print("[INFO] YOLO model loaded.")

    rows = []
    for seg in SEGMENTS:
        info = analyze_segment(seg, model)
        if info is not None:
            rows.append(info)

    if not rows:
        print("[ERROR] No segment stats produced.")
        return

    df = pd.DataFrame(rows)
    df.to_csv("segment_stats.csv", index=False)

    print("[INFO] Saved segment_stats.csv:")
    print(df.head())


if __name__ == "__main__":
    main()
