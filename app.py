# FastAPI backend for the emergency-routing UI, replacing the Streamlit app.
#
# Why this exists: Streamlit re-runs the whole script on every interaction, which
# is slow and caused state bugs (vanishing results, tab resets). Here the heavy
# objects - the ONNX detector and the OSM street graph - load once at startup and
# stay in memory, and the browser talks to small JSON endpoints instead. The map
# itself is a real Leaflet map in web/index.html, so panning, zooming and drawing
# routes never touch Python.
#
# Endpoints:
#   GET  /                     the Leaflet page
#   GET  /api/health           liveness + what loaded
#   GET  /api/cameras          camera congestion points for the map
#   POST /api/route            plan a vision-confirmed route between two places
#
# Run (dev):  .venv/Scripts/python -m uvicorn app:app --reload --port 8000
# Run (prod): uvicorn app:app --host 0.0.0.0 --port 8000

from __future__ import annotations

import base64
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import requests
from fastapi import FastAPI, Query
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel

from routing.graph import nearest_node, build_default_graph
from routing.planners import astar_route, route_metrics, static_baseline_route
from routing.vision_gate import plan_confirmed_route
from routing.geo import route_map_payload
from routing.geocode import geocode_manhattan, suggest_places
from routing.external_route import fetch_external_route
from routing.detect import get_detector, assess_frame_quality
from routing.incident_detect import check_frame_for_incident, BLOCKING
from update_camera_stats import compute_congestion

app = FastAPI(title="Emergency Routing API")

# Loaded once at startup; see load_state().
STATE: dict = {}
WEB_DIR = Path(__file__).parent / "web"


def load_state() -> None:
    """Load the street graph and camera data once, at process start."""
    try:
        from routing.streets import build_street_graph
        graph, real = build_street_graph(), True
    except Exception:
        graph, real = build_default_graph(), False

    cams = pd.read_csv("manhattan_cameras.csv")
    stats_path = Path("camera_stats.csv")
    if stats_path.exists():
        stats = pd.read_csv(stats_path)[
            ["camera_id", "score", "level", "vehicles", "pedestrians", "signals"]
        ]
        cams = cams.merge(stats, on="camera_id", how="left")

    STATE["graph"] = graph
    STATE["real_streets"] = real
    STATE["cameras"] = cams
    STATE["model"] = get_detector()
    # Fast lookup from camera id to its row (for the live-view endpoints).
    STATE["cam_by_id"] = {str(r["camera_id"]): r for _, r in cams.iterrows()}


def _fetch_frame(url: str):
    """Download and decode one camera snapshot, or None if unreachable."""
    try:
        r = requests.get(url, timeout=8)
        r.raise_for_status()
        frame = cv2.imdecode(np.frombuffer(r.content, np.uint8), cv2.IMREAD_COLOR)
        return frame
    except Exception:
        return None


@app.on_event("startup")
def _startup() -> None:
    load_state()


@app.get("/api/health")
def health() -> dict:
    g = STATE.get("graph")
    return {
        "ok": g is not None,
        "real_streets": STATE.get("real_streets", False),
        "nodes": g.number_of_nodes() if g else 0,
        "cameras": int(len(STATE.get("cameras", []))),
    }


@app.get("/api/cameras")
def cameras() -> JSONResponse:
    """Camera congestion points for the map overlay."""
    df = STATE["cameras"].copy()
    df["lat"] = pd.to_numeric(df["lat"], errors="coerce")
    df["lon"] = pd.to_numeric(df["lon"], errors="coerce")
    df = df.dropna(subset=["lat", "lon"])
    out = []
    for _, r in df.iterrows():
        score = r.get("score")
        out.append({
            "id": str(r["camera_id"]),
            "name": str(r["name"]),
            "lat": float(r["lat"]),
            "lon": float(r["lon"]),
            "score": None if pd.isna(score) else float(score),
            "level": None if pd.isna(r.get("level")) else str(r.get("level")),
        })
    return JSONResponse(out)


@app.get("/api/suggest")
def suggest(q: str = Query("", min_length=0)) -> JSONResponse:
    """Type-ahead place suggestions (landmarks, camera names, then Nominatim)."""
    hits = suggest_places(q) if len(q.strip()) >= 2 else []
    return JSONResponse([
        {"label": h["label"], "lat": float(h["lat"]), "lon": float(h["lon"])}
        for h in hits
    ])


@app.get("/api/camera/{camera_id}/detect")
def camera_detect(camera_id: str) -> JSONResponse:
    """Run YOLOv12 on a fresh snapshot from one camera and return the annotated
    image plus its congestion counts - the live-camera test from the old UI."""
    row = STATE["cam_by_id"].get(str(camera_id))
    if row is None:
        return JSONResponse({"ok": False, "error": "unknown camera"}, status_code=404)
    frame = _fetch_frame(row["image_url"])
    if frame is None:
        return JSONResponse({"ok": False, "error": "camera unreachable or offline"},
                            status_code=502)

    model = STATE["model"]
    # Slightly lower confidence for the live view so more of what is on screen
    # gets a box; scoring elsewhere keeps the default threshold.
    dets = model.detect(frame, conf=0.20)
    annotated = model.annotate(frame, dets)
    counts: dict = {}
    for d in dets:
        counts[d.name] = counts.get(d.name, 0) + 1
    score, level, vehicles, peds, signals = compute_congestion(counts)

    ok, buf = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 82])
    img = base64.b64encode(buf.tobytes()).decode("ascii") if ok else None
    usable, why = assess_frame_quality(frame)
    return JSONResponse({
        "ok": True, "name": str(row["name"]),
        "image": f"data:image/jpeg;base64,{img}" if img else None,
        "score": round(score, 1), "level": level,
        "vehicles": vehicles, "pedestrians": peds, "signals": signals,
        "detections": len(dets),
        "usable": usable, "quality": why,
    })


@app.get("/api/camera/{camera_id}/incident")
def camera_incident(camera_id: str) -> JSONResponse:
    """Vision-language incident check on one camera (crash, stall, debris,
    flooding). Degrades to a clear message when no vision model is configured."""
    row = STATE["cam_by_id"].get(str(camera_id))
    if row is None:
        return JSONResponse({"ok": False, "error": "unknown camera"}, status_code=404)
    frame = _fetch_frame(row["image_url"])
    if frame is None:
        return JSONResponse({"ok": False, "error": "camera unreachable or offline"},
                            status_code=502)

    rep = check_frame_for_incident(frame, str(camera_id), str(row["name"]))
    blocking = rep.source == "vlm" and rep.incident and rep.category in BLOCKING
    return JSONResponse({
        "ok": True, "name": str(row["name"]), "source": rep.source,
        "incident": rep.incident, "category": rep.category,
        "confidence": rep.confidence, "note": rep.note, "blocking": bool(blocking),
    })


class RouteRequest(BaseModel):
    start: str = "Times Square"
    end: str = "Wall Street"
    # When a suggestion was picked, the page sends its coordinates so we skip a
    # second geocode round-trip; free-typed text still geocodes server-side.
    start_lat: float | None = None
    start_lon: float | None = None
    end_lat: float | None = None
    end_lon: float | None = None
    compare: bool = True  # also fetch the external router for comparison


def _resolve(query: str) -> dict:
    """Geocode a place string to a point, or return an error outcome."""
    geo = geocode_manhattan(query)
    return geo


@app.post("/api/route")
def route(req: RouteRequest) -> JSONResponse:
    g = STATE["graph"]

    # Use coordinates from a picked suggestion when present; otherwise geocode.
    if req.start_lat is not None and req.start_lon is not None:
        start_geo = {"outcome": "ok", "lat": req.start_lat, "lon": req.start_lon,
                     "label": req.start}
    else:
        start_geo = _resolve(req.start)
    if req.end_lat is not None and req.end_lon is not None:
        end_geo = {"outcome": "ok", "lat": req.end_lat, "lon": req.end_lon,
                   "label": req.end}
    else:
        end_geo = _resolve(req.end)

    for label, geo in (("start", start_geo), ("end", end_geo)):
        if geo.get("outcome") != "ok":
            return JSONResponse(
                {"ok": False,
                 "error": f"Could not place the {label} ('{geo.get('label', '?')}'). "
                          "Try a NYC landmark, address, or intersection."},
                status_code=422,
            )

    src = nearest_node(g, start_geo["lat"], start_geo["lon"])
    dst = nearest_node(g, end_geo["lat"], end_geo["lon"])
    if src == dst:
        return JSONResponse(
            {"ok": False, "error": "Start and destination resolve to the same "
             "intersection - pick places further apart."},
            status_code=422,
        )

    # Vision-confirmed route (offline: trust stored camera scores) + baseline.
    our_path, gate_info = plan_confirmed_route(
        g, src, dst, planner=astar_route, mode="offline")
    baseline_path = static_baseline_route(g, src, dst)

    m = route_metrics(g, our_path)
    mb = route_metrics(g, baseline_path)

    payload = route_map_payload(
        g, our_path, baseline_path,
        start=(start_geo["lat"], start_geo["lon"]),
        end=(end_geo["lat"], end_geo["lon"]),
        gate_info=gate_info,
    )

    external = None
    if req.compare:
        try:
            ext = fetch_external_route(
                g, start_geo["lat"], start_geo["lon"],
                end_geo["lat"], end_geo["lon"], src_node=src, dst_node=dst)
            external = {
                "provider": ext.get("provider"),
                "coords": ext.get("coords", []),
                "duration_min": round(ext.get("duration_s", 0) / 60, 1),
                "congested_min": round(ext.get("congested_time_s", 0) / 60, 1),
            }
        except Exception:
            external = None

    return JSONResponse({
        "ok": True,
        "start_label": start_geo.get("label", req.start),
        "end_label": end_geo.get("label", req.end),
        "confirmed": gate_info.get("confirmed", False),
        "our_minutes": round(m["travel_time_s"] / 60, 1),
        "baseline_minutes": round(mb["travel_time_s"] / 60, 1),
        "saved_minutes": round(max(mb["travel_time_s"] - m["travel_time_s"], 0) / 60, 1),
        "hops": m["hops"],
        "km": round(m["length_km"], 2),
        "blocked_cameras": [
            {"camera": c.get("camera"), "score": round(c.get("score", 0), 1),
             "reason": c.get("reason", "score")}
            for c in gate_info.get("blocked_cameras", [])
        ],
        "payload": payload,
        "external": external,
    })


@app.get("/", response_class=HTMLResponse)
def index() -> HTMLResponse:
    html = (WEB_DIR / "index.html").read_text(encoding="utf-8")
    return HTMLResponse(html)
