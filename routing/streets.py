# Real street network for routing, following the production-maps recipe:
# a directed road graph whose edges carry free-flow travel times, with live
# congestion applied as edge-weight MULTIPLIERS on top (the topology never
# changes at query time - same split Google/OSRM use).
#
# The graph is downloaded once from OpenStreetMap (drivable roads inside the
# camera bounding box) and cached as GraphML in data/, so Docker and offline
# runs never touch the network.  Edges keep the exact schema graph.py uses
# (length_m, base_time, congestion, travel_time), which means every planner,
# the RL agent, the vision gate and the navigator work unchanged.
#
# Build/refresh the cache:  python -m routing.streets --build

from __future__ import annotations

import argparse
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd

GRAPHML_PATH = Path("data/manhattan_streets.graphml")
BBOX_PAD_DEG = 0.004
# Speed assumed for edges where OSM has no maxspeed and osmnx could not impute.
DEFAULT_SPEED_KMH = 40.0


def camera_bbox(cameras_csv: str | Path = "manhattan_cameras.csv") -> tuple:
    """(west, south, east, north) box covering every camera, slightly padded."""
    df = pd.read_csv(cameras_csv)
    lat = pd.to_numeric(df["lat"], errors="coerce").dropna()
    lon = pd.to_numeric(df["lon"], errors="coerce").dropna()
    return (
        float(lon.min()) - BBOX_PAD_DEG,
        float(lat.min()) - BBOX_PAD_DEG,
        float(lon.max()) + BBOX_PAD_DEG,
        float(lat.max()) + BBOX_PAD_DEG,
    )


def download_streets(cameras_csv: str | Path = "manhattan_cameras.csv") -> Path:
    """Fetch the drivable street network from OSM and cache it as GraphML.

    Run rarely (the road topology is stable); everything else loads the cache.
    """
    import osmnx as ox

    bbox = camera_bbox(cameras_csv)
    print(f"Downloading drivable streets for bbox {bbox} ...")
    mg = ox.graph_from_bbox(bbox, network_type="drive")
    mg = ox.routing.add_edge_speeds(mg, fallback=DEFAULT_SPEED_KMH)
    mg = ox.routing.add_edge_travel_times(mg)

    GRAPHML_PATH.parent.mkdir(parents=True, exist_ok=True)
    ox.io.save_graphml(mg, GRAPHML_PATH)
    print(f"Saved {mg.number_of_nodes()} nodes / {mg.number_of_edges()} edges "
          f"to {GRAPHML_PATH}")
    return GRAPHML_PATH


def _multi_to_digraph(mg) -> nx.DiGraph:
    """Collapse the OSMnx MultiDiGraph into our planner schema.

    Parallel edges keep the fastest one.  Curved street geometry is preserved
    as a [lat, lon] list so the map can hug the road instead of drawing
    straight chords between intersections.
    """
    g = nx.DiGraph()
    for n, d in mg.nodes(data=True):
        g.add_node(n, lat=float(d["y"]), lon=float(d["x"]))

    for u, v, d in mg.edges(data=True):
        length = float(d.get("length", 0.0))
        base_time = float(
            d.get("travel_time") or length / (DEFAULT_SPEED_KMH / 3.6)
        )
        if base_time <= 0:
            continue
        if g.has_edge(u, v) and g.edges[u, v]["base_time"] <= base_time:
            continue

        geom = d.get("geometry")
        coords = None
        if geom is not None:
            coords = [[float(lat), float(lon)] for lon, lat in geom.coords]
            # OSMnx stores geometry in the digitised direction; flip it if it
            # runs v -> u so polylines always flow with the edge.
            du = (coords[0][0] - g.nodes[u]["lat"]) ** 2 + (coords[0][1] - g.nodes[u]["lon"]) ** 2
            dv = (coords[-1][0] - g.nodes[u]["lat"]) ** 2 + (coords[-1][1] - g.nodes[u]["lon"]) ** 2
            if dv < du:
                coords = coords[::-1]

        g.add_edge(
            u, v,
            length_m=length,
            base_time=base_time,
            congestion=1.0,
            travel_time=base_time,
            blocked=False,
            geometry_latlon=coords,
        )
    return g


def attach_congestion_fast(
    g: nx.DiGraph,
    stats_csv: str | Path = "camera_stats.csv",
) -> nx.DiGraph:
    """Vectorised version of graph.attach_congestion for large street graphs.

    Same behaviour: each camera snaps to its nearest intersection, stores its
    id/score on the node, and slows every street touching that intersection
    by a multiplier proportional to the YOLO congestion score.
    """
    from routing.graph import CONGESTION_PER_POINT

    path = Path(stats_csv)
    if not path.exists():
        print(f"[WARN] {stats_csv} not found - streets keep free-flow times.")
        return g

    nodes = list(g.nodes)
    lats = np.array([g.nodes[n]["lat"] for n in nodes])
    lons = np.array([g.nodes[n]["lon"] for n in nodes])

    stats = pd.read_csv(path)
    for _, cam in stats.iterrows():
        cam_lat = pd.to_numeric(cam["lat"], errors="coerce")
        cam_lon = pd.to_numeric(cam["lon"], errors="coerce")
        score = pd.to_numeric(cam["score"], errors="coerce")
        if pd.isna(cam_lat) or pd.isna(cam_lon) or pd.isna(score):
            continue

        # Equirectangular approximation is plenty for nearest-node snapping
        # at city scale and lets numpy do all cameras in milliseconds.
        d2 = (lats - float(cam_lat)) ** 2 + ((lons - float(cam_lon)) * 0.755) ** 2
        node = nodes[int(np.argmin(d2))]

        prev = g.nodes[node].get("cam_score", -1.0)
        if float(score) > prev:
            g.nodes[node]["cam_id"] = cam["camera_id"]
            g.nodes[node]["cam_name"] = cam["name"]
            g.nodes[node]["cam_score"] = float(score)
            g.nodes[node]["cam_level"] = cam.get("level", "")

        multiplier = 1.0 + float(score) * CONGESTION_PER_POINT
        for u, v in list(g.in_edges(node)) + list(g.out_edges(node)):
            edge = g.edges[u, v]
            if multiplier > edge["congestion"]:
                edge["congestion"] = multiplier
                edge["travel_time"] = edge["base_time"] * multiplier

    return g


def build_street_graph(
    stats_csv: str | Path = "camera_stats.csv",
    graphml: str | Path = GRAPHML_PATH,
) -> nx.DiGraph:
    """Load the cached OSM street network with camera congestion attached.

    Raises FileNotFoundError when the cache is missing so callers can fall
    back to the synthetic grid explicitly (never silently).
    """
    graphml = Path(graphml)
    if not graphml.exists():
        raise FileNotFoundError(
            f"{graphml} missing - run `python -m routing.streets --build` once."
        )

    import osmnx as ox

    mg = ox.io.load_graphml(graphml)
    g = _multi_to_digraph(mg)

    # Keep only the largest strongly connected component so every pair of
    # nodes is routable (bbox clipping leaves dangling one-way stubs).
    biggest = max(nx.strongly_connected_components(g), key=len)
    g = g.subgraph(biggest).copy()

    # Admissible A* heuristic needs the true fastest speed on this graph.
    speeds = [
        d["length_m"] / d["base_time"] * 3.6
        for _, _, d in g.edges(data=True) if d["base_time"] > 0
    ]
    g.graph["max_speed_kmh"] = max(speeds) if speeds else DEFAULT_SPEED_KMH

    return attach_congestion_fast(g, stats_csv)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Real street network builder")
    parser.add_argument("--build", action="store_true",
                        help="download the OSM network and refresh the cache")
    args = parser.parse_args()

    if args.build:
        download_streets()

    g = build_street_graph()
    print(f"Street graph: {g.number_of_nodes()} nodes, {g.number_of_edges()} edges")
    print(f"Max speed on graph: {g.graph['max_speed_kmh']:.0f} km/h")

    slowed = [(u, v, d) for u, v, d in g.edges(data=True) if d["congestion"] > 1.0]
    cam_nodes = [n for n, d in g.nodes(data=True) if "cam_id" in d]
    print(f"Edges slowed by camera congestion: {len(slowed)}")
    print(f"Intersections hosting a camera: {len(cam_nodes)}")

    # Route across the island and sanity-check the result is a real path.
    from routing.planners import astar_route, route_metrics

    nodes = list(g.nodes)
    south = min(nodes, key=lambda n: g.nodes[n]["lat"])
    north = max(nodes, key=lambda n: g.nodes[n]["lat"])
    path = astar_route(g, south, north)
    m = route_metrics(g, path)
    print(f"South->north route: {m['hops']} hops, {m['length_km']:.2f} km, "
          f"{m['travel_time_s'] / 60:.1f} min")

    assert len(cam_nodes) > 100, "Most cameras should snap to street intersections"
    assert m["length_km"] > 5.0, "A cross-island route should be several km"
    print("streets.py self-test OK")
