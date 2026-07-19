# En-route navigation with Google-Maps-style recalculation.
#
# Production navigation (Google, Waze, OSRM) works in a loop:
#   1. plan on the current edge weights,
#   2. drive; every few seconds fresh probe/incident data updates the weights,
#   3. if the remaining route got significantly worse - or the road ahead is
#      closed - reroute FROM THE CURRENT POSITION, not from the start.
#
# This module reproduces that loop with our extra ingredient: the "incident
# data" is the YOLO vision check on the cameras the vehicle is about to pass.
# When a camera ahead reports a blockage, the intersection is cut and the
# remaining route is recalculated immediately - the same UX as taking a wrong
# turn in Google Maps and watching the route redraw.
#
# The result is a step-by-step trace the dashboard can replay on the map.

from __future__ import annotations

from typing import Callable

import networkx as nx

from routing.planners import route_metrics
from routing.vision_gate import BLOCK_THRESHOLD, live_score, offline_score

# How many upcoming cameras the vehicle checks before entering them.
# Small on purpose: a camera 5 km ahead will be re-checked when it is close,
# with fresher data - checking it now would waste live YOLO calls.
LOOKAHEAD_CAMERAS = 3


def _next_cameras(g: nx.DiGraph, path: list, limit: int) -> list:
    """The first `limit` camera intersections on the remaining path
    (excluding the node the vehicle is standing on)."""
    found = []
    for node in path[1:]:
        if "cam_id" in g.nodes[node]:
            found.append(node)
            if len(found) == limit:
                break
    return found


def drive_route(
    g: nx.DiGraph,
    src,
    dst,
    planner: Callable[[nx.DiGraph, object, object], list],
    mode: str = "offline",
    block_threshold: float = BLOCK_THRESHOLD,
    model=None,
    max_reroutes: int = 8,
) -> dict:
    """Simulate driving from src to dst with continuous vision checks.

    At every intersection the vehicle looks at the next few cameras on its
    route.  A blocked camera triggers an immediate reroute from the CURRENT
    position on a graph copy with that intersection cut out - exactly how a
    consumer navigator recalculates, but using vision instead of GPS probes.

    Returns a trace dict:
      driven          - nodes actually driven, in order
      segments        - list of {path, reason} for each planned leg (for the
                        map: the abandoned legs are drawn faded)
      reroutes        - list of {at_node, blocked_camera, score}
      confirmed       - True if the vehicle reached dst
      travel_time_s   - time spent on the driven path (congested weights)
    """
    work = g.copy()
    score_fn = (
        (lambda n: live_score(work, n, model=model))
        if mode == "live"
        else (lambda n: offline_score(work, n))
    )

    current = src
    driven: list = [src]
    segments: list[dict] = []
    reroutes: list[dict] = []
    checked_scores: dict = {}

    path = planner(work, current, dst)
    segments.append({"path": list(path), "reason": "initial plan"})

    while current != dst:
        # Look ahead: check the next cameras before committing to drive.
        blocked_ahead = None
        for cam_node in _next_cameras(work, path, LOOKAHEAD_CAMERAS):
            if cam_node in (src, dst):
                continue  # cannot route around the origin / destination
            if cam_node not in checked_scores:
                checked_scores[cam_node] = score_fn(cam_node)
            if checked_scores[cam_node] >= block_threshold:
                blocked_ahead = cam_node
                break

        if blocked_ahead is not None:
            if len(reroutes) >= max_reroutes:
                # Too many blockages - keep the current plan, flag as risky.
                return {
                    "driven": driven,
                    "segments": segments,
                    "reroutes": reroutes,
                    "confirmed": False,
                    "travel_time_s": _driven_time(g, driven),
                    "final_path": path,
                }
            reroutes.append({
                "at_node": current,
                "blocked_camera": work.nodes[blocked_ahead].get("cam_name", "?"),
                "blocked_node": blocked_ahead,
                "score": checked_scores[blocked_ahead],
            })
            work.remove_edges_from(
                list(work.in_edges(blocked_ahead)) + list(work.out_edges(blocked_ahead))
            )
            try:
                path = planner(work, current, dst)
            except (nx.NetworkXNoPath, nx.NodeNotFound):
                return {
                    "driven": driven,
                    "segments": segments,
                    "reroutes": reroutes,
                    "confirmed": False,
                    "travel_time_s": _driven_time(g, driven),
                    "final_path": segments[-1]["path"],
                }
            segments.append({
                "path": list(path),
                "reason": f"rerouted around {reroutes[-1]['blocked_camera']}",
            })
            continue

        # Road ahead confirmed - drive one intersection forward.
        if len(path) < 2:
            break
        path = path[1:]
        current = path[0]
        driven.append(current)

    return {
        "driven": driven,
        "segments": segments,
        "reroutes": reroutes,
        "confirmed": current == dst,
        "travel_time_s": _driven_time(g, driven),
        "final_path": driven,
    }


def _driven_time(g: nx.DiGraph, driven: list) -> float:
    """Congested travel time along the nodes actually driven."""
    total = 0.0
    for u, v in zip(driven[:-1], driven[1:]):
        if g.has_edge(u, v):
            total += float(g.edges[u, v]["travel_time"])
    return total


if __name__ == "__main__":
    # Self-test on the real street network: force a blockage on the initial
    # route and verify the navigator reroutes mid-drive and still arrives.
    from routing.planners import astar_route
    from routing.streets import build_street_graph
    from routing.vision_gate import cameras_on_route

    g = build_street_graph()
    nodes = list(g.nodes)
    src = min(nodes, key=lambda n: g.nodes[n]["lat"])
    dst = max(nodes, key=lambda n: g.nodes[n]["lat"])

    naive = astar_route(g, src, dst)
    cams = cameras_on_route(g, naive)
    assert cams, "Cross-island route should pass cameras"

    incident = cams[len(cams) // 2]  # block a camera mid-route
    g.nodes[incident]["cam_score"] = 99.0
    print(f"Blocking mid-route camera: {g.nodes[incident]['cam_name']}")

    trace = drive_route(g, src, dst, planner=astar_route, mode="offline")

    print(f"Arrived: {trace['confirmed']}")
    print(f"Reroutes: {len(trace['reroutes'])}")
    for r in trace["reroutes"]:
        print(f"  at {r['at_node']}: avoided {r['blocked_camera']} "
              f"(score {r['score']:.0f})")
    m = route_metrics(g, trace["driven"])
    print(f"Driven: {m['hops']} hops, {m['length_km']:.2f} km, "
          f"{trace['travel_time_s'] / 60:.1f} min")

    assert trace["confirmed"], "Vehicle must reach the destination"
    assert incident not in trace["driven"], "Must never drive through the blockage"
    assert len(trace["reroutes"]) >= 1, "The blockage must trigger a reroute"
    assert len(trace["segments"]) >= 2, "Trace should keep the abandoned leg"
    print("navigator.py self-test OK")
