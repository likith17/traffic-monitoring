# Where does the congestion score lose accuracy? Measure it, do not guess.
#
# The detector is parity-checked against the PyTorch model, so detection itself
# is faithful. The open question is whether the number built from those
# detections - the congestion score - is a good measure of congestion. This
# script audits that on real camera frames, quantifying four known weak points
# so any fix is justified by data rather than asserted:
#
#   1. Confidence sensitivity. How much does the vehicle count move if the
#      threshold shifts? A count that swings with an arbitrary cutoff is fragile.
#   2. Static-infrastructure share. Traffic lights and stop signs are fixtures in
#      a camera's view, present every frame no matter the traffic, yet each adds
#      0.5 to the score. This measures how much of the score is that fixed offset
#      rather than actual congestion.
#   3. Small-object drop-off. Vehicles far down the street are only a few pixels
#      after letterboxing to 640 and fall below the confidence floor. This
#      reports how much of what is detected sits near that limit (so how much is
#      probably being missed just past it).
#   4. Blank or dark frames. A "camera unavailable" placeholder or a night frame
#      yields zero detections, which is indistinguishable from a genuinely clear
#      road unless we check the image itself.
#
# Run:  python -m routing.perception_audit --cameras 15
# Output: a summary with interpretation + perception_audit.csv.

from __future__ import annotations

import argparse

import cv2
import numpy as np
import pandas as pd
import requests

from routing.detect import get_detector, assess_frame_quality

VEHICLE = {"car", "bus", "truck", "motorcycle"}
SIGNAL = {"traffic light", "stop sign"}
CONF_LEVELS = (0.15, 0.25, 0.40)

# A vehicle box smaller than this fraction of the frame is near the size at which
# detection becomes unreliable. (Brightness/placeholder thresholds live with the
# shared guard in routing.detect, so there is one definition.)
TINY_AREA_FRAC = 0.0015


def _fetch(url: str):
    try:
        r = requests.get(url, timeout=8)
        r.raise_for_status()
        return cv2.imdecode(np.frombuffer(r.content, np.uint8), cv2.IMREAD_COLOR)
    except Exception:
        return None


def audit_frame(detector, frame) -> dict:
    """All four measurements for a single frame."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    brightness = float(gray.mean())
    variation = float(gray.std())
    usable, _reason = assess_frame_quality(frame)
    blank = not usable
    frame_area = frame.shape[0] * frame.shape[1]

    # Vehicle count at each confidence level.
    veh_by_conf = {}
    dets_25 = None
    for c in CONF_LEVELS:
        dets = detector.detect(frame, conf=c)
        if c == 0.25:
            dets_25 = dets
        veh_by_conf[c] = sum(1 for d in dets if d.name in VEHICLE)

    # Score breakdown and small-object share at the operating threshold (0.25).
    vehicles = sum(1 for d in dets_25 if d.name in VEHICLE)
    peds = sum(1 for d in dets_25 if d.name == "person")
    signals = sum(1 for d in dets_25 if d.name in SIGNAL)
    score = vehicles * 1.0 + peds * 0.3 + signals * 0.5

    veh_boxes = [d for d in dets_25 if d.name in VEHICLE]
    tiny = 0
    for d in veh_boxes:
        x1, y1, x2, y2 = d.xyxy
        if (max(x2 - x1, 0) * max(y2 - y1, 0)) / frame_area < TINY_AREA_FRAC:
            tiny += 1

    return {
        "brightness": round(brightness, 1),
        "variation": round(variation, 1),
        "blank_or_dark": blank,
        "veh_conf15": veh_by_conf[0.15],
        "veh_conf25": veh_by_conf[0.25],
        "veh_conf40": veh_by_conf[0.40],
        "score": round(score, 2),
        "signal_contrib": round(signals * 0.5, 2),
        "vehicles": vehicles,
        "tiny_vehicles": tiny,
    }


def run(cameras: int, cameras_csv: str = "manhattan_cameras.csv") -> pd.DataFrame:
    detector = get_detector()
    cams = pd.read_csv(cameras_csv).dropna(subset=["image_url"])

    rows = []
    for _, cam in cams.iterrows():
        if len(rows) >= cameras:
            break
        frame = _fetch(cam["image_url"])
        if frame is None:
            continue
        rec = audit_frame(detector, frame)
        rec["camera"] = cam["name"]
        rows.append(rec)
        print(f"  {str(cam['name'])[:30]:<30} bright={rec['brightness']:5.1f} "
              f"veh@25={rec['veh_conf25']:2d} (15:{rec['veh_conf15']:2d}/40:{rec['veh_conf40']:2d}) "
              f"score={rec['score']:5.1f} signal={rec['signal_contrib']:.1f} "
              f"tiny={rec['tiny_vehicles']}" + ("  [BLANK/DARK]" if rec["blank_or_dark"] else ""))

    return pd.DataFrame(rows)


def summarise(df: pd.DataFrame) -> None:
    if df.empty:
        print("No frames audited.")
        return

    usable = df[~df["blank_or_dark"]]
    n_blank = int(df["blank_or_dark"].sum())

    print(f"\n=== Perception audit over {len(df)} cameras "
          f"({len(usable)} usable, {n_blank} blank/dark) ===")

    if not usable.empty:
        v25 = usable["veh_conf25"].sum()
        v15 = usable["veh_conf15"].sum()
        v40 = usable["veh_conf40"].sum()
        if v25:
            print(f"\n1. Confidence sensitivity (total vehicles, usable frames):")
            print(f"   conf 0.15: {v15}   conf 0.25: {v25}   conf 0.40: {v40}")
            print(f"   Lowering 0.25->0.15 surfaces {100*(v15-v25)/v25:+.0f}% more; "
                  f"raising 0.25->0.40 drops {100*(v40-v25)/v25:+.0f}%.")
            print("   Large swings mean many vehicles sit right at the threshold.")

        total_score = usable["score"].sum()
        signal_share = 100 * usable["signal_contrib"].sum() / total_score if total_score else 0
        with_signal = int((usable["signal_contrib"] > 0).sum())
        print(f"\n2. Static-infrastructure share:")
        print(f"   {signal_share:.0f}% of the total score comes from traffic-light/"
              f"stop-sign detections,")
        print(f"   present in {with_signal}/{len(usable)} usable frames. These are "
              "fixtures, not traffic;")
        print("   they add a fixed per-camera offset that is noise for comparing cameras.")

        tiny = usable["tiny_vehicles"].sum()
        print(f"\n3. Small-object drop-off:")
        print(f"   {tiny} of {v25} detected vehicles ({100*tiny/v25 if v25 else 0:.0f}%) are "
              "tiny boxes near the")
        print("   detection limit; a comparable number just past it is likely missed.")

    print(f"\n4. Blank/dark frames: {n_blank}/{len(df)} returned no trustworthy image.")
    print("   Their zero counts must not be read as 'clear road'.")


def main() -> None:
    ap = argparse.ArgumentParser(description="Audit congestion-score accuracy on live frames")
    ap.add_argument("--cameras", type=int, default=15)
    args = ap.parse_args()

    print(f"Auditing perception on up to {args.cameras} live cameras...\n")
    df = run(args.cameras)
    if df.empty:
        print("No cameras returned frames.")
        return
    df.to_csv("perception_audit.csv", index=False)
    summarise(df)
    print("\nSaved perception_audit.csv")


if __name__ == "__main__":
    main()
