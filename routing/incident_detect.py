# Phase 3: incident detection with a vision-language model.
#
# Counting vehicles tells you a road is busy; it does not tell you a car has
# crashed, stalled in a live lane, or that the street is flooded. Those are the
# events an emergency router most needs to route around, and they do not show up
# as a number. This module adds that layer.
#
# It follows the same shape as the rest of the system: a cheap detector first, a
# model second, and a fallback so it never hard-fails. Stage 1 reuses Phase 1
# tracking to decide *when* an image is worth a closer look (a backup forming, or
# unusually high volume). Stage 2 sends that frame to a vision-language model
# (traffic_llm.vision_completion) and asks for a strict-JSON verdict on whether a
# real incident is visible. If no vision model is configured, or the call fails,
# it returns the Stage-1 assessment marked unconfirmed rather than crashing.
#
# Honest scope, stated so it is not oversold:
#   - This reads a single still frame. It cannot see motion, and a VLM can be
#     wrong or hallucinate, so a positive is an advisory flag for a dispatcher to
#     verify, not a confirmed fact.
#   - Stage 1 geometry from a low-frame-rate still camera is weak on its own; its
#     job is to gate the cost of Stage 2, not to declare incidents.
#   - "heavy_congestion" is not an incident; it is reported separately so a plain
#     jam is never mislabelled as a crash.

from __future__ import annotations

import argparse
import json
import re
import time
from dataclasses import dataclass, asdict

import cv2
import numpy as np
import requests

from routing.detect import get_detector
from routing.tracking import (
    VehicleTracker, VEHICLE_CLASSES, HIGH_CONF, flow_stats_from, FlowStats,
)
from update_camera_stats import compute_congestion
from traffic_llm import vision_completion

# Incident categories the model is allowed to return. Kept closed so a verdict is
# always one of a known set, and so "heavy_congestion" stays distinct from a real
# blocking incident.
CATEGORIES = [
    "accident", "stalled_vehicle", "debris", "flooding",
    "emergency_vehicle_present", "heavy_congestion", "none",
]
BLOCKING = {"accident", "stalled_vehicle", "debris", "flooding"}

# Stage-1 gate thresholds. A single idle car at a light is not a candidate; a
# real backup or clearly high volume is. score>=15 is the existing "high" band.
MIN_QUEUE = 2
HIGH_SCORE = 15.0


@dataclass
class IncidentReport:
    camera_id: str
    camera_name: str
    candidate: bool           # did Stage 1 think this frame worth a VLM look?
    stage1_reason: str
    source: str               # "vlm", "heuristic" (VLM unavailable), or "none"
    incident: bool | None     # None when unconfirmed (no VLM)
    category: str
    confidence: float
    note: str
    score: float
    queue_length: int
    unique_vehicles: int

    def as_dict(self) -> dict:
        return asdict(self)


def prefilter(score: float, flow: FlowStats) -> tuple[bool, str]:
    """Cheap Stage-1 gate: is this frame worth a vision-model call?

    Returns (candidate, reason). Conservative on purpose: it only flags a real
    backup or clearly high volume, so most quiet frames cost nothing.
    """
    reasons = []
    if flow.queue_length >= MIN_QUEUE and flow.queue_length >= 0.5 * max(flow.unique_vehicles, 1):
        reasons.append(f"backup ({flow.queue_length} stopped)")
    if score >= HIGH_SCORE:
        reasons.append(f"high volume (score {score:.0f})")
    return (len(reasons) > 0, "; ".join(reasons) if reasons else "quiet")


_VLM_SYSTEM = (
    "You review a single still frame from a New York City traffic camera for "
    "emergency dispatch. Report only clearly visible conditions that would block "
    "or slow an emergency vehicle. You see one frame with no motion, so be "
    "conservative: normal moving traffic, even if heavy, is not an incident."
)

_VLM_PROMPT = (
    "Look at this traffic camera frame. Respond with STRICT JSON only, no prose, "
    "in exactly this form:\n"
    '{"incident": true|false, "category": one of '
    '["accident","stalled_vehicle","debris","flooding",'
    '"emergency_vehicle_present","heavy_congestion","none"], '
    '"confidence": 0.0-1.0, "note": "short phrase"}\n'
    "Rules: incident=true only for a crash, a vehicle stopped in a live lane, "
    "debris, or flooding. A plain jam is category=heavy_congestion with "
    "incident=false. Nothing notable is category=none, incident=false."
)


def _parse_verdict(text: str) -> dict | None:
    """Pull the JSON verdict out of the model reply, tolerating code fences.

    Returns a normalised dict, or None if nothing parseable is found. Never
    raises: a malformed reply must not take down an emergency tool.
    """
    if not text:
        return None
    # Grab the first {...} block, so a fenced ```json ... ``` or stray prose
    # around it does not break parsing.
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return None
    try:
        raw = json.loads(match.group(0))
    except (json.JSONDecodeError, ValueError):
        return None

    category = str(raw.get("category", "none")).strip().lower()
    if category not in CATEGORIES:
        category = "none"
    try:
        confidence = float(raw.get("confidence", 0.0))
    except (TypeError, ValueError):
        confidence = 0.0
    confidence = max(0.0, min(1.0, confidence))
    incident = bool(raw.get("incident", False))
    # A model may set incident=true but category=heavy_congestion; keep them
    # consistent, since congestion is explicitly not an incident here.
    if category in ("heavy_congestion", "none"):
        incident = False
    if category == "emergency_vehicle_present":
        incident = False  # informative, not a blockage
    note = str(raw.get("note", ""))[:200]
    return {"incident": incident, "category": category,
            "confidence": confidence, "note": note}


def _encode_jpeg(frame: np.ndarray, max_w: int = 1024) -> bytes | None:
    """JPEG-encode a BGR frame, downscaling wide images to keep the payload small."""
    if frame is None:
        return None
    h, w = frame.shape[:2]
    if w > max_w:
        frame = cv2.resize(frame, (max_w, int(h * max_w / w)))
    ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return buf.tobytes() if ok else None


def assess(
    camera_id: str,
    camera_name: str,
    score: float,
    flow: FlowStats,
    frame: np.ndarray | None,
    *,
    use_vlm: bool = True,
    force_vlm: bool = False,
    timeout: int = 60,
) -> IncidentReport:
    """Combine the Stage-1 gate and the optional Stage-2 vision verdict.

    Kept free of any network fetch so it can be unit-tested with synthetic
    frames. analyze_camera() does the polling and calls this.
    """
    candidate, reason = prefilter(score, flow)

    base = IncidentReport(
        camera_id=camera_id, camera_name=camera_name,
        candidate=candidate, stage1_reason=reason,
        source="none", incident=None, category="none", confidence=0.0, note="",
        score=round(score, 2), queue_length=flow.queue_length,
        unique_vehicles=flow.unique_vehicles,
    )

    if not (use_vlm and frame is not None and (candidate or force_vlm)):
        # No vision step taken. If Stage 1 flagged volume, say so honestly as
        # congestion, but never assert an incident without confirmation.
        if candidate:
            base.source = "heuristic"
            base.category = "heavy_congestion"
            base.note = "flagged by counts only; not vision-confirmed"
        return base

    jpeg = _encode_jpeg(frame)
    if jpeg is None:
        base.source = "heuristic"
        base.note = "frame could not be encoded"
        return base

    reply, err = vision_completion(_VLM_PROMPT, [jpeg], system=_VLM_SYSTEM, timeout=timeout)
    verdict = _parse_verdict(reply) if reply else None
    if verdict is None:
        # VLM unavailable or unparseable: fall back to the honest heuristic view.
        base.source = "heuristic"
        base.category = "heavy_congestion" if candidate else "none"
        base.note = f"vision unavailable ({(err or 'unparseable')[:80]})"
        return base

    base.source = "vlm"
    base.incident = verdict["incident"]
    base.category = verdict["category"]
    base.confidence = verdict["confidence"]
    base.note = verdict["note"]
    return base


def analyze_camera(
    detector, camera_id: str, camera_name: str, image_url: str,
    *, polls: int = 5, spacing: float = 1.5, use_vlm: bool = True,
    force_vlm: bool = False,
) -> IncidentReport | None:
    """Poll one camera, run Stage-1 tracking, then assess. None if unreachable."""
    tracker = VehicleTracker()
    scores, veh_counts, last_frame = [], [], None

    for i in range(polls):
        try:
            r = requests.get(image_url, timeout=8)
            r.raise_for_status()
            frame = cv2.imdecode(np.frombuffer(r.content, np.uint8), cv2.IMREAD_COLOR)
        except Exception:
            frame = None
        if frame is not None:
            last_frame = frame
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

    if not scores:
        return None

    import statistics as st
    flow = flow_stats_from(tracker, veh_counts, len(scores))
    return assess(camera_id, camera_name, st.median(scores), flow, last_frame,
                  use_vlm=use_vlm, force_vlm=force_vlm)


def main() -> None:
    ap = argparse.ArgumentParser(description="Phase 3 incident detection over cameras")
    ap.add_argument("--cameras", type=int, default=10, help="how many cameras to scan")
    ap.add_argument("--polls", type=int, default=5)
    ap.add_argument("--no-vlm", action="store_true", help="skip the vision model (Stage 1 only)")
    ap.add_argument("--force-vlm", action="store_true", help="call the VLM even on quiet frames")
    args = ap.parse_args()

    import pandas as pd
    cams = pd.read_csv("manhattan_cameras.csv").dropna(subset=["image_url"])
    detector = get_detector()

    print(f"Scanning {min(args.cameras, len(cams))} cameras for incidents...\n")
    reports = []
    for _, cam in cams.head(args.cameras).iterrows():
        rep = analyze_camera(detector, str(cam["camera_id"]), str(cam["name"]),
                             cam["image_url"], polls=args.polls,
                             use_vlm=not args.no_vlm, force_vlm=args.force_vlm)
        if rep is None:
            continue
        reports.append(rep.as_dict())
        flag = "INCIDENT" if rep.incident else ("~" if rep.candidate else " ")
        print(f"  [{flag:^8}] {rep.camera_name[:30]:<30} "
              f"cat={rep.category:<18} src={rep.source:<9} "
              f"conf={rep.confidence:.2f}  ({rep.stage1_reason})")

    if reports:
        pd.DataFrame(reports).to_csv("incident_reports.csv", index=False)
        n_inc = sum(1 for r in reports if r["incident"])
        print(f"\nSaved incident_reports.csv. {n_inc} vision-confirmed incident(s), "
              f"{sum(1 for r in reports if r['candidate'])} Stage-1 candidate(s).")
        if not any(r["source"] == "vlm" for r in reports):
            print("Note: no vision model configured, so these are Stage-1 only "
                  "(set LLM_PROVIDER + an API key to enable confirmation).")


if __name__ == "__main__":
    main()
