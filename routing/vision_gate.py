# The vision gate: before a route is dispatched, the cameras along it are
# checked with YOLOv12 to make sure the streets are actually passable.
#
# Two modes:
#   offline - trust the scores already stored on the graph (from
#             camera_stats.csv).  Fast, needs no network, used by the
#             simulator and Docker demos.
#   live    - re-download a fresh snapshot from each camera on the route and
#             run YOLOv12 on it right now.  Slower but reflects the street
#             this very minute.  Used by the dashboard when requested.
#
# Any camera whose congestion score reaches BLOCK_THRESHOLD marks its
# intersection as impassable.  The planner is then re-run on a copy of the
# graph with that intersection cut out, and the new route is checked again,
# until a confirmed-clear route is found (or we run out of attempts).

from __future__ import annotations

from typing import Callable

import networkx as nx

# A camera score at or above this means the intersection is effectively
# blocked for an emergency vehicle (double-parked jam, incident, closure).
# 15+ is already "high" congestion in the scoring scheme; 25 is severe.
BLOCK_THRESHOLD = 25.0

# Give up re-planning after this many attempts and return the least-bad route.
MAX_REPLANS = 5


def cameras_on_route(g: nx.DiGraph, path: list[tuple]) -> list[tuple]:
    """The intersections along the route that actually host a camera.
    Only these can be visually confirmed - the rest of the route is trusted."""
    return [n for n in path if "cam_id" in g.nodes[n]]


def offline_score(g: nx.DiGraph, node: tuple) -> float:
    """Congestion score for a camera node as recorded in camera_stats.csv."""
    return float(g.nodes[node].get("cam_score", 0.0))


def _live_score_and_frame(g: nx.DiGraph, node: tuple, model=None):
    """Fetch a fresh snapshot for the camera at this node, score it now, and
    return (score, frame). The frame is handed back so the incident check can
    reuse it instead of downloading the image a second time. Returns
    (offline_score, None) whenever the camera is unreachable, so a dead camera
    never blocks dispatch by itself.

    The heavy imports happen inside the function on purpose: offline users
    (the simulator, CI, Docker demos) never pay the ultralytics startup cost.
    """
    import pandas as pd

    from update_camera_stats import compute_congestion, fetch_frame, load_model

    cam_id = g.nodes[node].get("cam_id")
    if cam_id is None:
        return 0.0, None

    # camera_stats.csv has no image URLs, so look the camera up in the
    # original camera list to find where to download the snapshot from.
    cams = pd.read_csv("manhattan_cameras.csv")
    match = cams[cams["camera_id"] == cam_id]
    if match.empty:
        return offline_score(g, node), None

    frame = fetch_frame(match.iloc[0]["image_url"])
    if frame is None:
        return offline_score(g, node), None

    if model is None:
        model = load_model()

    counts = model.class_counts(frame)
    score, _level, _v, _p, _s = compute_congestion(counts)
    return float(score), frame


def live_score(g: nx.DiGraph, node: tuple, model=None) -> float:
    """Fetch a fresh snapshot for the camera at this node and score it with
    YOLOv12 right now. Thin wrapper over _live_score_and_frame for callers that
    only need the score."""
    return _live_score_and_frame(g, node, model=model)[0]


# A vision-confirmed incident needs at least this confidence to block a route.
# Below it, the model is too unsure to override a route on one still frame.
INCIDENT_MIN_CONF = 0.5


def confirm_route(
    g: nx.DiGraph,
    path: list[tuple],
    mode: str = "offline",
    block_threshold: float = BLOCK_THRESHOLD,
    model=None,
    check_incidents: bool = False,
    incident_min_conf: float = INCIDENT_MIN_CONF,
    vlm_timeout: int = 30,
) -> tuple[bool, list[dict]]:
    """Check every camera along the route and report which ones say 'blocked'.

    A camera blocks the route for either reason, independently: its congestion
    score reaches block_threshold, or (when check_incidents and mode='live') a
    vision-language model confirms a blocking incident in the frame - a crash, a
    vehicle stalled in a live lane, debris, or flooding. Incidents are only
    checked in live mode, since they need a fresh frame and a model call; offline
    mode stays score-only and network-free.

    Returns (route_is_clear, checked_cameras). Each entry carries the camera
    name, its score, whether it blocks, why ('score', 'incident', or 'clear'),
    and any incident details.
    """
    checked: list[dict] = []
    clear = True

    for node in cameras_on_route(g, path):
        incident = None
        if mode == "live":
            score, frame = _live_score_and_frame(g, node, model=model)
            if check_incidents and frame is not None:
                from routing.incident_detect import check_frame_for_incident, BLOCKING
                rep = check_frame_for_incident(
                    frame,
                    str(g.nodes[node].get("cam_id", "")),
                    g.nodes[node].get("cam_name", "unknown"),
                    timeout=vlm_timeout,
                )
                # Only a vision-confirmed, high-enough-confidence blocking
                # category counts. Congestion and emergency-vehicle sightings do
                # not, and the score path already covers plain heavy traffic.
                if (rep.source == "vlm" and rep.incident
                        and rep.category in BLOCKING
                        and rep.confidence >= incident_min_conf):
                    incident = {"category": rep.category,
                                "confidence": rep.confidence, "note": rep.note}
        else:
            score = offline_score(g, node)

        blocked_by_score = score >= block_threshold
        blocked = blocked_by_score or incident is not None
        clear = clear and not blocked
        entry = {
            "node": node,
            "camera": g.nodes[node].get("cam_name", "unknown"),
            "score": score,
            "blocked": blocked,
            "reason": ("incident" if incident is not None
                       else "score" if blocked_by_score else "clear"),
        }
        if incident is not None:
            entry["incident"] = incident
        checked.append(entry)

    return clear, checked


def plan_confirmed_route(
    g: nx.DiGraph,
    src: tuple,
    dst: tuple,
    planner: Callable[[nx.DiGraph, tuple, tuple], list[tuple]],
    mode: str = "offline",
    block_threshold: float = BLOCK_THRESHOLD,
    model=None,
    check_incidents: bool = False,
    incident_min_conf: float = INCIDENT_MIN_CONF,
    vlm_timeout: int = 30,
) -> tuple[list[tuple], dict]:
    """Plan a route, confirm it visually, and re-plan around blocked spots.

    The planner argument is any function with the (graph, src, dst) signature -
    Dijkstra, A*, or a wrapper around the trained RL agent - so the gate works
    with every routing strategy in this package.

    With check_incidents=True in live mode, a vision-confirmed incident (crash,
    stall, debris, flooding) blocks a segment and forces a re-plan just as a
    high congestion score does, so the router avoids not only jams but active
    hazards. Returns (path, info); info records every re-plan and every blocked
    camera, which the LLM explanation module later turns into plain English.
    """
    # Work on a copy so cutting out blocked intersections never damages the
    # shared graph used by other parts of the app.
    work = g.copy()

    all_blocked: list[dict] = []
    attempts = 0
    path = planner(work, src, dst)

    while attempts < MAX_REPLANS:
        attempts += 1
        clear, checked = confirm_route(
            work, path, mode=mode, block_threshold=block_threshold, model=model,
            check_incidents=check_incidents, incident_min_conf=incident_min_conf,
            vlm_timeout=vlm_timeout,
        )
        blocked_here = [c for c in checked if c["blocked"]]

        # A blockage at the start or destination cannot be routed around -
        # the vehicle is already there / must get there.  Report it as a
        # warning instead of failing the route forever.
        avoidable = [c for c in blocked_here if c["node"] not in (src, dst)]
        unavoidable = [c for c in blocked_here if c["node"] in (src, dst)]
        all_blocked.extend(avoidable)

        if not avoidable:
            return path, {
                "confirmed": True,
                "attempts": attempts,
                "blocked_cameras": all_blocked,
                "endpoint_warnings": unavoidable,
                "checked_cameras": checked,
            }

        # Cut every avoidable blocked intersection out of the working graph
        # and try again with the same planner.
        for cam in avoidable:
            node = cam["node"]
            work.remove_edges_from(list(work.in_edges(node)) + list(work.out_edges(node)))

        try:
            path = planner(work, src, dst)
        except nx.NetworkXNoPath:
            # The blockages disconnect the map entirely; return the last
            # route we had, clearly marked as unconfirmed.
            break

    return path, {
        "confirmed": False,
        "attempts": attempts,
        "blocked_cameras": all_blocked,
        "endpoint_warnings": [],
        "checked_cameras": [],
    }


if __name__ == "__main__":
    # Self-test in offline mode: use a low threshold so some real camera
    # scores count as blockages, and verify the gate routes around them.
    from routing.graph import build_default_graph
    from routing.planners import astar_route, route_metrics

    g = build_default_graph()

    # Pick start/destination intersections that do NOT host a camera, so any
    # blockage found is mid-route and the avoidance logic gets exercised.
    free = [n for n in g.nodes if "cam_id" not in g.nodes[n]]
    src = min(free)   # bottom-left-most camera-free intersection
    dst = max(free)   # top-right-most camera-free intersection
    print(f"Routing {src} -> {dst}")

    naive = astar_route(g, src, dst)
    naive_cams = cameras_on_route(g, naive)
    print(f"Naive A* route passes {len(naive_cams)} cameras")
    assert naive_cams, "Test route should pass at least one camera"

    # Simulate an incident: force the first camera on the naive route to
    # report a severe blockage, so the gate MUST route around it.
    incident_node = naive_cams[0]
    g.nodes[incident_node]["cam_score"] = 99.0
    print(f"Simulated blockage at {g.nodes[incident_node]['cam_name']}")

    path, info = plan_confirmed_route(
        g, src, dst, planner=astar_route, mode="offline"
    )

    print(f"Confirmed: {info['confirmed']} after {info['attempts']} attempt(s)")
    print(f"Cameras routed around: {len(info['blocked_cameras'])}")
    for cam in info["blocked_cameras"][:5]:
        print(f"  AVOIDED {cam['camera']}: score {cam['score']:.1f}")
    for cam in info.get("endpoint_warnings", []):
        print(f"  WARNING endpoint camera {cam['camera']}: score {cam['score']:.1f}")

    m = route_metrics(g, path)
    print(f"Final route: {m['travel_time_s']:.0f}s over {m['length_km']:.2f} km")

    assert info["confirmed"], "Gate should find a confirmed route around the incident"
    assert incident_node not in path, "Confirmed route must avoid the blocked intersection"
    assert len(info["blocked_cameras"]) >= 1, "The simulated blockage should be recorded"
    print("Self-test passed.")

    # --- Second self-test: incident-based blocking (no network, no API key) ---
    # A vision-confirmed incident must block a segment even when its congestion
    # score is low. We inject a fake fresh frame for every camera and a fake VLM
    # verdict that flags exactly one camera as a crash, then confirm the gate
    # routes around it. This exercises the live+incident path deterministically.
    print("\n--- incident-blocking self-test ---")
    import numpy as _np
    import routing.vision_gate as _vg
    import routing.incident_detect as _inc

    g2 = build_default_graph()
    free2 = [n for n in g2.nodes if "cam_id" not in g2.nodes[n]]
    s2, d2 = min(free2), max(free2)
    naive2 = astar_route(g2, s2, d2)
    cams2 = cameras_on_route(g2, naive2)
    assert cams2, "Test route should pass a camera"
    crash_node = cams2[0]
    crash_name = g2.nodes[crash_node].get("cam_name", "unknown")

    _real_score_frame = _vg._live_score_and_frame
    _real_check = _inc.check_frame_for_incident
    try:
        # Every camera reads a harmless low score and returns a dummy frame, so
        # nothing blocks on score and the incident path is what decides.
        _vg._live_score_and_frame = lambda g, node, model=None: (
            1.0, _np.zeros((8, 8, 3), dtype=_np.uint8))

        def _fake_check(frame, cam_id="", cam_name="", *, use_vlm=True, timeout=30):
            is_crash = cam_name == crash_name
            return _inc.IncidentReport(
                camera_id=cam_id, camera_name=cam_name,
                candidate=is_crash, stage1_reason="test",
                source="vlm", incident=is_crash,
                category="accident" if is_crash else "none",
                confidence=0.9 if is_crash else 0.0, note="test",
                score=1.0, queue_length=0, unique_vehicles=0)
        _inc.check_frame_for_incident = _fake_check

        path2, info2 = plan_confirmed_route(
            g2, s2, d2, planner=astar_route, mode="live", check_incidents=True)
    finally:
        _vg._live_score_and_frame = _real_score_frame
        _inc.check_frame_for_incident = _real_check

    blocked_by_incident = [c for c in info2["blocked_cameras"]
                           if c.get("reason") == "incident"]
    print(f"Confirmed: {info2['confirmed']}, incident blocks: {len(blocked_by_incident)}")
    for c in blocked_by_incident[:3]:
        print(f"  AVOIDED (incident) {c['camera']}: {c['incident']['category']} "
              f"@ {c['incident']['confidence']:.0%} (score only {c['score']:.0f})")
    assert info2["confirmed"], "Gate should route around the incident"
    assert crash_node not in path2, "Confirmed route must avoid the incident camera"
    assert blocked_by_incident, "The incident should be recorded as an incident block"
    assert blocked_by_incident[0]["score"] < BLOCK_THRESHOLD, \
        "Incident must block on its own, not because of a high score"
    print("Incident-blocking self-test passed.")
