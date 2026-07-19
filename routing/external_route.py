# External directions baseline: Google Directions when a key is set, otherwise
# the public OSRM demo server, otherwise the in-graph static free-flow path.
#
# Labelling is intentional and honest:
#   - "Google Maps" only when the Google Directions API actually answered
#   - "OSRM (OpenStreetMap)" for the free public router
#   - "Naive free-flow path" for the local graph fallback
# Never claim a route is Google Maps when it is not.

from __future__ import annotations

import os
from typing import Any

import networkx as nx
import requests

from routing.graph import haversine_m, nearest_node
from routing.planners import route_metrics, static_baseline_route

OSRM_URL = "https://router.project-osrm.org/route/v1/driving"
GOOGLE_DIRECTIONS_URL = "https://maps.googleapis.com/maps/api/directions/json"
REQUEST_TIMEOUT_S = 12


def _polyline_length_km(coords: list[list[float]]) -> float:
    """Approximate path length from a [[lat, lon], ...] sequence."""
    if len(coords) < 2:
        return 0.0
    total = 0.0
    for a, b in zip(coords, coords[1:]):
        total += haversine_m(a[0], a[1], b[0], b[1])
    return total / 1000.0


def _estimate_travel_time_s(g: nx.DiGraph, coords: list[list[float]]) -> float:
    """Score an external polyline against our congestion-weighted graph.

    Snaps each consecutive pair of points to graph nodes and sums the
    congested travel_time along the shortest connecting hops.  This lets us
    compare Google/OSRM geometry against our vision-aware ETA on the same
    cost model — without pretending their ETA equals ours.
    """
    if len(coords) < 2:
        return 0.0

    snapped = [nearest_node(g, lat, lon) for lat, lon in coords]
    # Collapse consecutive duplicates from dense polylines.
    nodes: list[tuple] = [snapped[0]]
    for n in snapped[1:]:
        if n != nodes[-1]:
            nodes.append(n)

    total = 0.0
    for u, v in zip(nodes, nodes[1:]):
        if g.has_edge(u, v):
            total += float(g.edges[u, v]["travel_time"])
            continue
        # External routes may cut corners across our coarser grid — walk the
        # Dijkstra path between snapped nodes when a direct edge is missing.
        try:
            hop = nx.shortest_path(g, u, v, weight="travel_time")
            for a, b in zip(hop, hop[1:]):
                total += float(g.edges[a, b]["travel_time"])
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            # Last resort: free-flow haversine at FREE_FLOW_KMH equivalent.
            dist = haversine_m(
                g.nodes[u]["lat"], g.nodes[u]["lon"],
                g.nodes[v]["lat"], g.nodes[v]["lon"],
            )
            total += dist / (40.0 / 3.6)
    return total


def fetch_google_directions(
    start_lat: float,
    start_lon: float,
    end_lat: float,
    end_lon: float,
    api_key: str | None = None,
) -> dict[str, Any] | None:
    """Call Google Directions.  Returns None on any failure."""
    key = api_key or os.getenv("GOOGLE_MAPS_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not key:
        return None
    try:
        r = requests.get(
            GOOGLE_DIRECTIONS_URL,
            params={
                "origin": f"{start_lat},{start_lon}",
                "destination": f"{end_lat},{end_lon}",
                "mode": "driving",
                "key": key,
            },
            timeout=REQUEST_TIMEOUT_S,
        )
        r.raise_for_status()
        data = r.json()
        if data.get("status") != "OK" or not data.get("routes"):
            return None
        route = data["routes"][0]
        leg = route["legs"][0]
        # Decode overview polyline into lat/lon pairs.
        coords = _decode_google_polyline(route["overview_polyline"]["points"])
        return {
            "provider": "Google Maps",
            "coords": coords,
            "distance_km": leg["distance"]["value"] / 1000.0,
            "duration_s": float(leg["duration"]["value"]),
            "source": "google_directions",
        }
    except Exception:
        return None


def _decode_google_polyline(encoded: str) -> list[list[float]]:
    """Decode a Google encoded polyline into [[lat, lon], ...]."""
    coords: list[list[float]] = []
    index = 0
    lat = 0
    lon = 0
    length = len(encoded)
    while index < length:
        for coord_name in ("lat", "lon"):
            result = 0
            shift = 0
            while True:
                b = ord(encoded[index]) - 63
                index += 1
                result |= (b & 0x1F) << shift
                shift += 5
                if b < 0x20:
                    break
            delta = ~(result >> 1) if result & 1 else (result >> 1)
            if coord_name == "lat":
                lat += delta
            else:
                lon += delta
        coords.append([lat / 1e5, lon / 1e5])
    return coords


def fetch_osrm_directions(
    start_lat: float,
    start_lon: float,
    end_lat: float,
    end_lon: float,
) -> dict[str, Any] | None:
    """Call the public OSRM demo server.  Returns None on any failure."""
    url = (
        f"{OSRM_URL}/{start_lon},{start_lat};{end_lon},{end_lat}"
        f"?overview=full&geometries=geojson"
    )
    try:
        r = requests.get(url, timeout=REQUEST_TIMEOUT_S)
        r.raise_for_status()
        data = r.json()
        if data.get("code") != "Ok" or not data.get("routes"):
            return None
        route = data["routes"][0]
        # GeoJSON is [lon, lat] — flip to [lat, lon] for Folium.
        coords = [[c[1], c[0]] for c in route["geometry"]["coordinates"]]
        return {
            "provider": "OSRM (OpenStreetMap)",
            "coords": coords,
            "distance_km": float(route["distance"]) / 1000.0,
            "duration_s": float(route["duration"]),
            "source": "osrm",
        }
    except Exception:
        return None


def fetch_external_route(
    g: nx.DiGraph,
    start_lat: float,
    start_lon: float,
    end_lat: float,
    end_lon: float,
    src_node: tuple | None = None,
    dst_node: tuple | None = None,
) -> dict[str, Any]:
    """Resolve the best available external/naive baseline for comparison.

    Preference order: Google Directions → OSRM → in-graph static free-flow.
    Always returns a dict with provider, coords, distance_km, duration_s,
    congested_time_s (our cost model), and source.
    """
    result = fetch_google_directions(start_lat, start_lon, end_lat, end_lon)
    if result is None:
        result = fetch_osrm_directions(start_lat, start_lon, end_lat, end_lon)

    if result is not None:
        result["congested_time_s"] = _estimate_travel_time_s(g, result["coords"])
        result["length_km"] = result["distance_km"]
        return result

    # Offline / no-network fallback: free-flow Dijkstra on our grid.
    if src_node is None:
        src_node = nearest_node(g, start_lat, start_lon)
    if dst_node is None:
        dst_node = nearest_node(g, end_lat, end_lon)
    path = static_baseline_route(g, src_node, dst_node)
    metrics = route_metrics(g, path)
    coords = [[float(g.nodes[n]["lat"]), float(g.nodes[n]["lon"])] for n in path]
    return {
        "provider": "Naive free-flow path",
        "coords": coords,
        "distance_km": metrics["length_km"],
        "duration_s": metrics["free_flow_time_s"],
        "congested_time_s": metrics["travel_time_s"],
        "length_km": metrics["length_km"],
        "source": "static_baseline",
        "graph_path": path,
    }


def external_crosses_blockages(
    g: nx.DiGraph,
    coords: list[list[float]],
    blocked_nodes: set[tuple],
    proximity_m: float = 180.0,
) -> list[dict[str, Any]]:
    """Check whether an external polyline passes near known blocked cameras.

    Used in the compare UI to show *why* the vision-confirmed route diverges
    from Google/OSRM: their path still threads the incident.
    """
    if not blocked_nodes or not coords:
        return []
    hits: list[dict[str, Any]] = []
    seen: set[tuple] = set()
    for lat, lon in coords:
        for node in blocked_nodes:
            if node in seen:
                continue
            d = haversine_m(lat, lon, g.nodes[node]["lat"], g.nodes[node]["lon"])
            if d <= proximity_m:
                seen.add(node)
                hits.append(
                    {
                        "node": node,
                        "camera": g.nodes[node].get("cam_name", str(node)),
                        "distance_m": d,
                        "score": g.nodes[node].get("cam_score", 0.0),
                    }
                )
    return hits


if __name__ == "__main__":
    from routing.graph import build_default_graph

    g = build_default_graph()
    cams = [n for n, d in g.nodes(data=True) if "cam_id" in d]
    assert len(cams) >= 2, "Need camera nodes for the self-test"
    a, b = cams[0], cams[-1]
    start = (g.nodes[a]["lat"], g.nodes[a]["lon"])
    end = (g.nodes[b]["lat"], g.nodes[b]["lon"])

    # Offline fallback must always work without network.
    path = static_baseline_route(g, a, b)
    metrics = route_metrics(g, path)
    assert metrics["length_km"] > 0

    # Decode a tiny known Google polyline fragment (pair of points).
    # Encoded "_p~iF~ps|U_ulLnnqC" is the classic Google sample.
    sample = _decode_google_polyline("_p~iF~ps|U_ulLnnqC")
    assert len(sample) == 2, f"Expected 2 points, got {len(sample)}"

    # Prefer whatever network gives us; fall back silently.
    result = fetch_external_route(g, start[0], start[1], end[0], end[1], a, b)
    assert result["coords"], "External route must return coordinates"
    assert result["provider"], "Provider label required for honest UI"
    print(f"Provider : {result['provider']}")
    print(f"Points   : {len(result['coords'])}")
    print(f"Distance : {result['distance_km']:.2f} km")
    print(f"Source   : {result['source']}")
    print("external_route.py self-test OK")
