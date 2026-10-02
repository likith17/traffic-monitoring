# Compare detector variants on live camera frames, to decide whether a bigger
# model (YOLOv12-M) is worth adopting over the shipped YOLOv12-S - especially at
# night, where the small model misses vehicles.
#
# Honest limit: there is no ground truth here, so this measures RELATIVE recall
# (how many vehicles each variant finds) and inference latency, not absolute
# accuracy. More detections is a recall proxy, not proof of correctness - a
# bigger model can also add false positives. Read the numbers as "how much more
# does M see, and what does it cost", then eyeball a few frames before adopting.
#
# Dev tool: needs the training deps (torch + ultralytics), not the slim runtime.
# Run:  python -m routing.model_compare --cameras 14

from __future__ import annotations

import argparse
import time

import cv2
import numpy as np
import pandas as pd
import requests

from routing.detect import enhance_low_light

VEHICLES = {"car", "bus", "truck", "motorcycle"}


def _fetch(url: str):
    try:
        r = requests.get(url, timeout=8)
        r.raise_for_status()
        return cv2.imdecode(np.frombuffer(r.content, np.uint8), cv2.IMREAD_COLOR)
    except Exception:
        return None


def _count_vehicles(model, frame, conf: float) -> tuple[int, float]:
    """Return (vehicle_count, inference_ms) for one frame."""
    t0 = time.perf_counter()
    res = model(frame, imgsz=640, conf=conf, verbose=False)[0]
    ms = (time.perf_counter() - t0) * 1000
    names = res.names
    n = sum(1 for c in res.boxes.cls.tolist() if names[int(c)] in VEHICLES)
    return n, ms


def main() -> None:
    ap = argparse.ArgumentParser(description="Compare YOLOv12-S vs -M on live frames")
    ap.add_argument("--cameras", type=int, default=14)
    ap.add_argument("--conf", type=float, default=0.25)
    args = ap.parse_args()

    from ultralytics import YOLO
    print("Loading YOLOv12-S and YOLOv12-M ...")
    s_model = YOLO("weights/yolov12s.pt")
    m_model = YOLO("yolov12m.pt")

    cams = pd.read_csv("manhattan_cameras.csv").dropna(subset=["image_url"])

    # Totals per variant: (vehicles, ms, frames, brightness sum)
    agg = {k: [0, 0.0, 0] for k in ("S", "S+enh", "M", "M+enh")}
    bright_sum = 0.0
    used = 0
    print(f"\nSampling up to {args.cameras} live cameras (conf {args.conf})...\n")
    print(f"{'camera':30} {'bright':>6} {'S':>4} {'S+e':>4} {'M':>4} {'M+e':>4}")

    for _, cam in cams.iterrows():
        if used >= args.cameras:
            break
        frame = _fetch(cam["image_url"])
        if frame is None:
            continue
        bright = float(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).mean())
        enh = enhance_low_light(frame)

        sv, sms = _count_vehicles(s_model, frame, args.conf)
        sev, sems = _count_vehicles(s_model, enh, args.conf)
        mv, mms = _count_vehicles(m_model, frame, args.conf)
        mev, mems = _count_vehicles(m_model, enh, args.conf)

        for k, (v, ms) in zip(agg, ((sv, sms), (sev, sems), (mv, mms), (mev, mems))):
            agg[k][0] += v
            agg[k][1] += ms
            agg[k][2] += 1
        bright_sum += bright
        used += 1
        print(f"{str(cam['name'])[:30]:30} {bright:6.0f} {sv:4d} {sev:4d} {mv:4d} {mev:4d}")

    if used == 0:
        print("No frames fetched.")
        return

    base = agg["S"][0] or 1
    print(f"\n=== Over {used} live frames (mean brightness {bright_sum/used:.0f}/255) ===")
    print(f"{'variant':8} {'vehicles':>9} {'vs S':>7} {'mean ms':>9}")
    for k, (veh, ms, n) in agg.items():
        print(f"{k:8} {veh:9d} {100*(veh-base)/base:+6.0f}% {ms/n:9.0f}")
    print("\nReading: 'vehicles' is total detected (recall proxy, no ground truth).")
    print("Higher usually means better recall at night, but verify a few frames")
    print("for false positives before adopting. 'mean ms' is CPU latency per frame.")


if __name__ == "__main__":
    main()
