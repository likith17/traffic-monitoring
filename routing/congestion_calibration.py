# Phase 2, step 2: does the camera congestion score actually track real speed?
#
# Before replacing the hand-picked weights with anything learned, we have to
# know the current score means something. This is the go/no-go check. For each
# camera that genuinely sits on a sensored road (from ground_truth.py), it polls
# the live image a few times, computes the heuristic congestion score with a
# temporal median to cut single-frame noise, and pairs that score with the
# measured speed from the feed.
#
# The expected relationship is negative: more congestion (higher score) should
# mean lower speed. We report Pearson (linear) and Spearman (rank) correlation.
# If the score and speed are correlated, calibrating the weights against speed is
# justified. If they are not, the honest conclusion is that the perception
# features need work before any learning step, and we say so rather than fitting
# a model to noise.
#
# This is a single-snapshot experiment over ~20 cameras: a first look, not proof.
# One moment gives limited congestion variation, so a real training set needs the
# time-logger (step 3). The number here says whether that is worth building.
#
# Run:  python -m routing.congestion_calibration --polls 3

from __future__ import annotations

import argparse
import statistics as st
import time

import cv2
import numpy as np
import requests
from scipy import stats

from routing.detect import get_detector
from routing.ground_truth import match_cameras
from update_camera_stats import compute_congestion


def _score_once(detector, image_url: str, timeout: float = 8.0) -> float | None:
    """Fetch one frame and return its heuristic congestion score, or None."""
    try:
        r = requests.get(image_url, timeout=timeout)
        r.raise_for_status()
        frame = cv2.imdecode(np.frombuffer(r.content, np.uint8), cv2.IMREAD_COLOR)
        if frame is None:
            return None
        return compute_congestion(detector.class_counts(frame))[0]
    except Exception:
        return None


def _temporal_score(detector, image_url: str, polls: int, spacing: float) -> float | None:
    """Median congestion score over several frames, to damp single-frame noise."""
    scores = []
    for i in range(polls):
        s = _score_once(detector, image_url)
        if s is not None:
            scores.append(s)
        if i < polls - 1:
            time.sleep(spacing)
    return st.median(scores) if scores else None


def run(polls: int, spacing: float, cameras_csv: str = "manhattan_cameras.csv"):
    matched = match_cameras(cameras_csv=cameras_csv)
    valid = matched[matched["has_speed"]].copy()
    if valid.empty:
        print("No cameras with a live speed reading right now. Retry later.")
        return None

    # Need the image_url, which lives in the full camera table, not the match.
    import pandas as pd
    cams = pd.read_csv(cameras_csv)[["camera_id", "image_url"]]
    valid = valid.merge(cams, on="camera_id", how="left")

    detector = get_detector()
    print(f"Scoring {len(valid)} cameras on sensored roads, {polls} frames each...\n")

    rows = []
    for _, cam in valid.iterrows():
        score = _temporal_score(detector, cam["image_url"], polls, spacing)
        if score is None:
            continue
        rows.append({
            "camera_name": cam["camera_name"],
            "score": round(score, 2),
            "speed_mph": cam["speed_mph"],
            "dist_m": cam["dist_m"],
        })
        print(f"  {str(cam['camera_name'])[:34]:<34} score={score:5.1f}  "
              f"speed={cam['speed_mph']:5.1f} mph")

    df = pd.DataFrame(rows)
    if len(df) < 5:
        print(f"\nOnly {len(df)} usable cameras returned frames; too few to correlate.")
        return df

    pear_r, pear_p = stats.pearsonr(df["score"], df["speed_mph"])
    spear_r, spear_p = stats.spearmanr(df["score"], df["speed_mph"])

    df.to_csv("congestion_vs_speed.csv", index=False)

    print(f"\n=== Score vs measured speed over {len(df)} cameras (one snapshot) ===")
    print(f"  Pearson  r = {pear_r:+.2f}  (p = {pear_p:.3f})   linear")
    print(f"  Spearman r = {spear_r:+.2f}  (p = {spear_p:.3f})   rank")
    print("\n  Expected sign is negative: more congestion, less speed.")
    if spear_r < -0.3 and spear_p < 0.1:
        print("  Result: the score tracks real speed. Calibrating weights against")
        print("  speed is justified; the time-logger (step 3) can build the set.")
    elif spear_r < 0:
        print("  Result: weak negative link, in the right direction but not strong")
        print("  on one snapshot. Accumulate data over time before trusting a fit.")
    else:
        print("  Result: no negative link on this snapshot. The perception features")
        print("  likely need work before any learning step. Do not fit to this.")
    print("  Saved congestion_vs_speed.csv")
    return df


def main() -> None:
    ap = argparse.ArgumentParser(description="Phase 2 score-vs-speed correlation check")
    ap.add_argument("--polls", type=int, default=3, help="frames per camera")
    ap.add_argument("--spacing", type=float, default=1.5, help="seconds between polls")
    args = ap.parse_args()
    run(args.polls, args.spacing)


if __name__ == "__main__":
    main()
