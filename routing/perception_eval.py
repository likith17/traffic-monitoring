# Phase 1 evaluation: does multi-frame scoring actually reduce noise?
#
# The claim behind temporal aggregation is that a single camera frame gives a
# noisy congestion score, and taking the median over several frames removes that
# noise. This script measures it directly on live cameras. For each camera it
# polls the feed several times, scores every frame on its own, and reports how
# far a single frame lands from the median of the group. A large gap means one
# frame alone is unreliable and multi-frame aggregation is worth its cost.
#
# Run:  python -m routing.perception_eval --cameras 8 --polls 5
# Output: a summary table plus perception_stability.csv, so the result is
# reproducible rather than a one-off measurement.

from __future__ import annotations

import argparse
import statistics as st
import time

import cv2
import numpy as np
import pandas as pd
import requests

from routing.detect import get_detector
from update_camera_stats import compute_congestion


def score_frame(detector, frame) -> float:
    return compute_congestion(detector.class_counts(frame))[0]


def evaluate(cameras: int, polls: int, spacing: float) -> pd.DataFrame:
    detector = get_detector()
    cams = pd.read_csv("manhattan_cameras.csv").dropna(subset=["lat", "lon"])

    rows = []
    for _, cam in cams.iterrows():
        if len(rows) >= cameras:
            break
        scores = []
        for i in range(polls):
            try:
                r = requests.get(cam["image_url"], timeout=8)
                r.raise_for_status()
                f = cv2.imdecode(np.frombuffer(r.content, np.uint8), cv2.IMREAD_COLOR)
                if f is not None:
                    scores.append(score_frame(detector, f))
            except Exception:
                pass
            if i < polls - 1:
                time.sleep(spacing)

        if len(scores) < 3:
            continue  # too few usable frames to compare

        median = st.median(scores)
        rows.append({
            "camera": cam["name"],
            "n_frames": len(scores),
            "median_score": round(median, 2),
            "single_frame_std": round(st.pstdev(scores), 2),
            # Worst a single frame deviates from the group median: the size of
            # the error you accept by trusting one frame.
            "max_abs_dev": round(max(abs(s - median) for s in scores), 2),
        })
        print(f"  {cam['name'][:34]:<34} frames={[round(s,1) for s in scores]}")

    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 1 multi-frame stability evaluation")
    parser.add_argument("--cameras", type=int, default=8)
    parser.add_argument("--polls", type=int, default=5)
    parser.add_argument("--spacing", type=float, default=2.0, help="seconds between polls")
    args = parser.parse_args()

    print(f"Polling {args.cameras} cameras, {args.polls} frames each...\n")
    df = evaluate(args.cameras, args.polls, args.spacing)

    if df.empty:
        print("No cameras returned enough frames. Check the network and retry.")
        return

    df.to_csv("perception_stability.csv", index=False)

    mean_std = df["single_frame_std"].mean()
    mean_dev = df["max_abs_dev"].mean()
    mean_median = df["median_score"].mean() or 1.0
    pct = 100 * mean_dev / mean_median

    print(f"\n=== Multi-frame stability over {len(df)} cameras, {args.polls} frames each ===")
    print(f"  Mean single-frame standard deviation : {mean_std:.2f}")
    print(f"  Mean worst single-frame deviation    : {mean_dev:.2f}  "
          f"({pct:.0f}% of the median score)")
    print("\n  A lone frame can miss the multi-frame median by this much; taking")
    print("  the median across frames removes that swing. Saved perception_stability.csv")


if __name__ == "__main__":
    main()
