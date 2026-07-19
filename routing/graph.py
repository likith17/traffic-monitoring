# Builds the road network the routing algorithms run on.
#
# The graph is a Manhattan-style grid stretched over the real bounding box of
# the NYC DOT cameras, so every camera from camera_stats.csv can be snapped to
# a nearby intersection and its YOLO congestion score used to slow down the
# streets around it.  Every edge carries:
#   base_time   - free-flow travel time in seconds (no traffic at all)
#   congestion  - multiplier >= 1 coming from nearby camera scores
#   travel_time - base_time * congestion, what the planners actually minimise
#
# The builder is deliberately kept behind two small functions
# (build_grid_graph + attach_congestion) so a real street network from OSMnx
# could replace the grid later without touching the planners or the RL agent.

from __future__ import annotations

import math
from pathlib import Path

import networkx as nx
import pandas as pd

# Free-flow speed assumed on every street, in km/h.  Emergency vehicles move
# faster than regular traffic, but the exact number only scales all times
# equally, so the ranking of routes does not depend on it.
FREE_FLOW_KMH = 40.0

# How strongly one unit of camera congestion score slows an edge down.
# score 0  -> multiplier 1.0 (free flow)
# score 10 -> multiplier 2.0 (twice as slow)
# score 30 -> multiplier 4.0 (heavy jam)
CONGESTION_PER_POINT = 0.1


def haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Distance in metres between two lat/lon points on the Earth's surface."""
    r = 6_371_000.0  # Earth radius in metres
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


def build_grid_graph(
    cameras_csv: str | Path = "manhattan_cameras.csv",
    n_rows: int = 30,
    n_cols: int = 12,
) -> nx.DiGraph:
    """Build a directed grid road network covering the camera area.

    Rows follow latitude (like numbered streets) and columns follow longitude
    (like avenues).  Every neighbouring pair of intersections gets one edge in
    each direction, because congestion is not necessarily symmetric.
    """
    df = pd.read_csv(cameras_csv)
    lat = pd.to_numeric(df["lat"], errors="coerce").dropna()
    lon = pd.to_numeric(df["lon"], errors="coerce").dropna()

    # Stretch the grid slightly past the outermost cameras so none of them
    # sit exactly on the boundary.
    pad = 0.002
    lat_min, lat_max = lat.min() - pad, lat.max() + pad
    lon_min, lon_max = lon.min() - pad, lon.max() + pad

    g = nx.DiGraph()

    # Create one node per grid intersection with its real-world position.
    for i in range(n_rows):
        for j in range(n_cols):
            node_lat = lat_min + (lat_max - lat_min) * i / (n_rows - 1)
            node_lon = lon_min + (lon_max - lon_min) * j / (n_cols - 1)
            g.add_node((i, j), lat=node_lat, lon=node_lon)

    def add_two_way(a: tuple, b: tuple) -> None:
        """Connect two intersections with an edge in each direction."""
        dist = haversine_m(
            g.nodes[a]["lat"], g.nodes[a]["lon"],
            g.nodes[b]["lat"], g.nodes[b]["lon"],
        )
        seconds = dist / (FREE_FLOW_KMH / 3.6)  # km/h -> m/s
        for u, v in ((a, b), (b, a)):
            g.add_edge(
                u, v,
                length_m=dist,
                base_time=seconds,
                congestion=1.0,
                travel_time=seconds,
                blocked=False,
            )

    # Wire up the grid: each node connects to its right-hand and upward
    # neighbour, which covers every adjacent pair exactly once.
    for i in range(n_rows):
        for j in range(n_cols):
            if i + 1 < n_rows:
                add_two_way((i, j), (i + 1, j))
            if j + 1 < n_cols:
                add_two_way((i, j), (i, j + 1))

    return g


def nearest_node(g: nx.DiGraph, lat: float, lon: float) -> tuple:
    """Snap an arbitrary lat/lon (a camera, an incident, a hospital) to the
    closest intersection in the graph."""
    return min(
        g.nodes,
        key=lambda n: haversine_m(lat, lon, g.nodes[n]["lat"], g.nodes[n]["lon"]),
    )


def attach_congestion(
    g: nx.DiGraph,
    stats_csv: str | Path = "camera_stats.csv",
) -> nx.DiGraph:
    """Push the YOLO congestion scores from camera_stats.csv onto the graph.

    Each camera is snapped to its nearest intersection.  All edges touching
    that intersection get slowed down in proportion to the camera's score,
    because a jammed intersection delays every street that feeds it.
    The camera id and score are also stored on the node so the vision gate
    can later re-check exactly the cameras that sit along a planned route.
    """
    path = Path(stats_csv)
    if not path.exists():
        # No scores yet: the graph simply stays at free-flow travel times.
        print(f"[WARN] {stats_csv} not found - graph keeps free-flow times.")
        return g

    stats = pd.read_csv(path)
    for _, cam in stats.iterrows():
        cam_lat = pd.to_numeric(cam["lat"], errors="coerce")
        cam_lon = pd.to_numeric(cam["lon"], errors="coerce")
        score = pd.to_numeric(cam["score"], errors="coerce")
        if pd.isna(cam_lat) or pd.isna(cam_lon) or pd.isna(score):
            continue

        node = nearest_node(g, float(cam_lat), float(cam_lon))

        # A node can end up hosting several cameras; keep the worst score,
        # since the most congested view is the safest planning assumption.
        prev = g.nodes[node].get("cam_score", -1.0)
        if score > prev:
            g.nodes[node]["cam_id"] = cam["camera_id"]
            g.nodes[node]["cam_name"] = cam["name"]
            g.nodes[node]["cam_score"] = float(score)
            g.nodes[node]["cam_level"] = cam.get("level", "")

        multiplier = 1.0 + float(score) * CONGESTION_PER_POINT
        for u, v in list(g.in_edges(node)) + list(g.out_edges(node)):
            edge = g.edges[u, v]
            # Again keep the worst multiplier if several cameras affect the edge.
            if multiplier > edge["congestion"]:
                edge["congestion"] = multiplier
                edge["travel_time"] = edge["base_time"] * multiplier

    return g


def build_default_graph() -> nx.DiGraph:
    """Convenience one-liner used by the dashboard and the simulator:
    grid over the camera area with the latest congestion attached."""
    g = build_grid_graph()
    return attach_congestion(g)


if __name__ == "__main__":
    # Quick self-test: build the graph and show that congestion actually
    # changed some travel times.
    g = build_default_graph()
    print(f"Nodes: {g.number_of_nodes()}, edges: {g.number_of_edges()}")

    slowed = [
        (u, v, d) for u, v, d in g.edges(data=True) if d["congestion"] > 1.0
    ]
    print(f"Edges slowed by camera congestion: {len(slowed)}")
    for u, v, d in sorted(slowed, key=lambda e: -e[2]["congestion"])[:5]:
        print(
            f"  {u} -> {v}: base {d['base_time']:.0f}s, "
            f"x{d['congestion']:.2f} -> {d['travel_time']:.0f}s"
        )

    cam_nodes = [n for n, d in g.nodes(data=True) if "cam_id" in d]
    print(f"Intersections hosting a camera: {len(cam_nodes)}")
