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


def live_score(g: nx.DiGraph, node: tuple, model=None) -> float:
    """Fetch a fresh snapshot for the camera at this node and score it with
    YOLOv12 right now.  Falls back to the stored offline score if the camera
    is unreachable, so a dead camera never blocks dispatch by itself.

    The heavy imports happen inside the function on purpose: offline users
    (the simulator, CI, Docker demos) never pay the ultralytics startup cost.
    """
    import pandas as pd

    from update_camera_stats import compute_congestion, fetch_frame, load_model

    cam_id = g.nodes[node].get("cam_id")
    if cam_id is None:
        return 0.0

    # camera_stats.csv has no image URLs, so look the camera up in the
    # original camera list to find where to download the snapshot from.
    cams = pd.read_csv("manhattan_cameras.csv")
    match = cams[cams["camera_id"] == cam_id]
    if match.empty:
        return offline_score(g, node)

    frame = fetch_frame(match.iloc[0]["image_url"])
    if frame is None:
        return offline_score(g, node)

    if model is None:
        model = load_model()

    counts = model.class_counts(frame)
    score, _level, _v, _p, _s = compute_congestion(counts)
    return float(score)


def confirm_route(
    g: nx.DiGraph,
    path: list[tuple],
    mode: str = "offline",
    block_threshold: float = BLOCK_THRESHOLD,
    model=None,
) -> tuple[bool, list[dict]]:
    """Check every camera along the route and report which ones say 'blocked'.

    Returns (route_is_clear, checked_cameras) where checked_cameras is a list
    of dicts with the camera name, its score, and whether it blocks the route.
    """
    checked: list[dict] = []
    clear = True

    for node in cameras_on_route(g, path):
        if mode == "live":
            score = live_score(g, node, model=model)
        else:
            score = offline_score(g, node)

        blocked = score >= block_threshold
        clear = clear and not blocked
        checked.append({
            "node": node,
            "camera": g.nodes[node].get("cam_name", "unknown"),
            "score": score,
            "blocked": blocked,
        })

    return clear, checked


def plan_confirmed_route(
    g: nx.DiGraph,
    src: tuple,
    dst: tuple,
    planner: Callable[[nx.DiGraph, tuple, tuple], list[tuple]],
    mode: str = "offline",
    block_threshold: float = BLOCK_THRESHOLD,
    model=None,
) -> tuple[list[tuple], dict]:
    """Plan a route, confirm it visually, and re-plan around blocked spots.

    The planner argument is any function with the (graph, src, dst) signature -
    Dijkstra, A*, or a wrapper around the trained RL agent - so the gate works
    with every routing strategy in this package.

    Returns (path, info).  info records every re-plan and every blocked
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
            work, path, mode=mode, block_threshold=block_threshold, model=model
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
