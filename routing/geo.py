# Geo helpers: turn graph paths into map-ready polylines and markers.
#
# The planners work on abstract (row, col) grid nodes.  The public UI needs
# real lat/lon sequences so Folium / Leaflet can draw a Google-Maps-like path.
# Keeping this conversion in one place means the dashboard never has to know
# about NetworkX node ids.

from __future__ import annotations

from typing import Any

import networkx as nx


def path_to_latlon(g: nx.DiGraph, path: list[tuple]) -> list[list[float]]:
    """Convert a list of graph nodes into a Folium-ready [[lat, lon], ...] polyline.

    Raises ValueError on an empty path so callers fail loudly instead of
    rendering a blank map.
    """
    if not path:
        raise ValueError("Cannot convert an empty path to lat/lon")
    return [[float(g.nodes[n]["lat"]), float(g.nodes[n]["lon"])] for n in path]


def path_bounds(coords: list[list[float]]) -> list[list[float]]:
    """Tight [[south, west], [north, east]] bounds for a polyline."""
    if not coords:
        raise ValueError("Cannot compute bounds of an empty coordinate list")
    lats = [c[0] for c in coords]
    lons = [c[1] for c in coords]
    return [[min(lats), min(lons)], [max(lats), max(lons)]]


def camera_markers_from_gate(g: nx.DiGraph, gate_info: dict | None) -> list[dict[str, Any]]:
    """Extract map markers for blocked / warned cameras from vision-gate output.

    Each marker is a plain dict so the UI layer can render it without importing
    NetworkX.  Missing gate_info returns an empty list (nothing to highlight).
    """
    if not gate_info:
        return []

    markers: list[dict[str, Any]] = []
    for kind, key in (("blocked", "blocked_cameras"), ("warning", "endpoint_warnings")):
        for cam in gate_info.get(key, []):
            node = cam.get("node")
            if node is None or node not in g.nodes:
                continue
            markers.append(
                {
                    "lat": float(g.nodes[node]["lat"]),
                    "lon": float(g.nodes[node]["lon"]),
                    "name": cam.get("camera", "camera"),
                    "score": float(cam.get("score", 0.0)),
                    "kind": kind,
                }
            )
    return markers


def route_map_payload(
    g: nx.DiGraph,
    our_path: list[tuple],
    baseline_path: list[tuple] | None = None,
    start: tuple[float, float] | None = None,
    end: tuple[float, float] | None = None,
    gate_info: dict | None = None,
) -> dict[str, Any]:
    """Bundle everything the map UI needs into one JSON-serialisable dict.

    start/end are (lat, lon) of the user-picked cameras.  When omitted they
    default to the first/last node of our_path so the payload is always complete.
    """
    our_coords = path_to_latlon(g, our_path)
    baseline_coords = path_to_latlon(g, baseline_path) if baseline_path else []

    if start is None:
        start = (our_coords[0][0], our_coords[0][1])
    if end is None:
        end = (our_coords[-1][0], our_coords[-1][1])

    all_coords = our_coords + baseline_coords + [[start[0], start[1]], [end[0], end[1]]]
    return {
        "our_route": our_coords,
        "baseline_route": baseline_coords,
        "start": {"lat": float(start[0]), "lon": float(start[1])},
        "end": {"lat": float(end[0]), "lon": float(end[1])},
        "cameras": camera_markers_from_gate(g, gate_info),
        "bounds": path_bounds(all_coords),
    }


if __name__ == "__main__":
    from routing.graph import build_default_graph
    from routing.planners import astar_route, static_baseline_route

    g = build_default_graph()
    nodes = list(g.nodes)
    src, dst = nodes[0], nodes[-1]
    route = astar_route(g, src, dst)
    baseline = static_baseline_route(g, src, dst)

    coords = path_to_latlon(g, route)
    assert len(coords) == len(route), "Polyline length must match hop count"
    assert all(len(c) == 2 for c in coords), "Each point must be [lat, lon]"

    payload = route_map_payload(
        g,
        route,
        baseline_path=baseline,
        gate_info={"blocked_cameras": [], "endpoint_warnings": []},
    )
    assert payload["our_route"], "our_route must be non-empty"
    assert payload["baseline_route"], "baseline_route must be non-empty"
    assert "bounds" in payload

    print(f"Route points : {len(payload['our_route'])}")
    print(f"Baseline pts : {len(payload['baseline_route'])}")
    print(f"Bounds       : {payload['bounds']}")
    print("geo.py self-test OK")
