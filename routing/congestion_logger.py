# Phase 2, step 3: accumulate a real training set over time.
#
# Step 2 showed the current score tracks measured speed only weakly on a single
# snapshot, and that one moment gives too little congestion variation to trust a
# fit. The honest fix is not a cleverer model, it is more data across more
# conditions. This logger is that collector.
#
# Each run is one time-slice: for every camera that sits on a sensored road, it
# polls the live image several times, runs Phase 1 tracking over those frames,
# and appends one row pairing the perception features with the feed's measured
# speed at that moment. Run it repeatedly (by hand, cron, or a scheduled task)
# and congestion_log.csv fills up with real (features, speed) pairs spanning rush
# hours, midday lulls, and nights. Only once that set is broad enough does
# fitting weights become honest, and that fit is a later step, not this one.
#
# Crucially it logs the richer Phase 1 flow features (queue length, distinct
# vehicles, relative speed), not just the raw count score. Step 2 hinted these
# may track speed better on fast roads, where the count score barely moves. The
# log lets that be tested on real data instead of asserted.

from __future__ import annotations

import argparse
import statistics as st
import time
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import requests

from routing.detect import get_detector, assess_frame_quality
from routing.ground_truth import match_cameras
from routing.tracking import (
    VehicleTracker, VEHICLE_CLASSES, HIGH_CONF, flow_stats_from,
)
from update_camera_stats import compute_congestion

LOG_PATH = "congestion_log.csv"

# Column order for the log. Fixed here so every run appends compatibly.
LOG_COLUMNS = [
    "timestamp_utc", "camera_id", "camera_name", "link_name", "dist_m",
    "score", "queue_length", "unique_vehicles", "mean_density", "mean_speed_px",
    "frames_used", "speed_mph",
]


def _sample_camera(detector, image_url: str, polls: int, spacing: float,
                   stationary_px: float = 2.0):
    """Poll one camera several times and return (score, FlowStats, frames_used).

    Each frame is detected exactly once; that single detection feeds both the
    per-frame congestion score and the tracker, so no frame is processed twice.
    Returns None if too few frames were usable to be better than a single shot.
    """
    tracker = VehicleTracker()
    scores, veh_counts = [], []

    for i in range(polls):
        try:
            r = requests.get(image_url, timeout=8)
            r.raise_for_status()
            frame = cv2.imdecode(np.frombuffer(r.content, np.uint8), cv2.IMREAD_COLOR)
        except Exception:
            frame = None

        # A dark or placeholder frame gives all-zero features that are not real
        # (can't see, not empty). Logging them would teach a later fit that
        # "no cars -> free flow" from frames where nothing was visible, so skip.
        if frame is not None and assess_frame_quality(frame)[0]:
            dets = detector.detect(frame)
            counts: dict = {}
            for d in dets:
                counts[d.name] = counts.get(d.name, 0) + 1
            scores.append(compute_congestion(counts)[0])
            tracker.update(dets)
            veh_counts.append(sum(1 for d in dets
                                  if d.name in VEHICLE_CLASSES and d.conf >= HIGH_CONF))

        if i < polls - 1:
            time.sleep(spacing)

    if len(scores) < max(2, polls // 2):
        return None

    flow = flow_stats_from(tracker, veh_counts, len(scores), stationary_px)
    return st.median(scores), flow, len(scores)


def run_once(polls: int, spacing: float, cameras_csv: str = "manhattan_cameras.csv",
             log_path: str = LOG_PATH, require_speed: bool = True) -> pd.DataFrame:
    """One collection pass: sample every sensored-road camera, append rows."""
    matched = match_cameras(cameras_csv=cameras_csv)
    if require_speed:
        matched = matched[matched["has_speed"]]
    if matched.empty:
        print("No sensored-road cameras with a reading right now. Nothing logged.")
        return pd.DataFrame(columns=LOG_COLUMNS)

    cams = pd.read_csv(cameras_csv)[["camera_id", "image_url"]]
    matched = matched.merge(cams, on="camera_id", how="left")

    detector = get_detector()
    ts = datetime.now(timezone.utc).isoformat(timespec="seconds")
    print(f"Sampling {len(matched)} cameras, {polls} frames each, at {ts}...\n")

    rows = []
    for _, cam in matched.iterrows():
        sample = _sample_camera(detector, cam["image_url"], polls, spacing)
        if sample is None:
            continue
        score, flow, frames_used = sample
        rows.append({
            "timestamp_utc": ts,
            "camera_id": cam["camera_id"],
            "camera_name": cam["camera_name"],
            "link_name": cam["link_name"],
            "dist_m": cam["dist_m"],
            "score": round(score, 2),
            "queue_length": flow.queue_length,
            "unique_vehicles": flow.unique_vehicles,
            "mean_density": round(flow.mean_density, 2),
            "mean_speed_px": round(flow.mean_speed_px, 2),
            "frames_used": frames_used,
            "speed_mph": cam["speed_mph"],
        })
        print(f"  {str(cam['camera_name'])[:32]:<32} score={score:4.1f} "
              f"queue={flow.queue_length} uniq={flow.unique_vehicles} "
              f"spd_px={flow.mean_speed_px:4.1f} | {cam['speed_mph']:4.1f} mph")

    df = pd.DataFrame(rows, columns=LOG_COLUMNS)
    if df.empty:
        print("\nNo cameras returned enough frames this pass.")
        return df

    header = not Path(log_path).exists()
    df.to_csv(log_path, mode="a", header=header, index=False)

    total = pd.read_csv(log_path)
    spans = total["timestamp_utc"].nunique()
    print(f"\nAppended {len(df)} rows to {log_path}.")
    print(f"  Log now holds {len(total)} rows across {spans} time-slice(s).")
    if spans < 6:
        print("  Keep collecting across different hours before fitting anything;")
        print("  a handful of slices is still one narrow slice of conditions.")
    return df


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Phase 2 collector: log perception features vs measured speed")
    ap.add_argument("--polls", type=int, default=6, help="frames per camera")
    ap.add_argument("--spacing", type=float, default=1.5, help="seconds between polls")
    ap.add_argument("--log", default=LOG_PATH, help="CSV to append to")
    args = ap.parse_args()
    run_once(args.polls, args.spacing, log_path=args.log)


if __name__ == "__main__":
    main()
