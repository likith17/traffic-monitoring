# Classical shortest-path planners: Dijkstra and A*.
#
# Both minimise the congestion-aware travel_time that graph.py puts on every
# edge.  The static baseline planner ignores congestion on purpose - it plans
# on free-flow base_time only, which is what a naive dispatch system would do.
# The benchmark in simulate.py measures how much time the smarter planners
# save compared to that baseline.

from __future__ import annotations

import networkx as nx

from routing.graph import FREE_FLOW_KMH, haversine_m


def dijkstra_route(g: nx.DiGraph, src: tuple, dst: tuple) -> list[tuple]:
    """Plain Dijkstra on congestion-aware travel times.
    Returns the list of nodes from src to dst (inclusive)."""
    return nx.dijkstra_path(g, src, dst, weight="travel_time")


def _time_heuristic(g: nx.DiGraph):
    """Admissible A* heuristic: straight-line distance at free-flow speed.

    No real route can beat flying straight to the goal at the fastest speed
    any edge allows, so this never overestimates and A* stays optimal.
    """
    speed_ms = FREE_FLOW_KMH / 3.6

    def h(a: tuple, b: tuple) -> float:
        dist = haversine_m(
            g.nodes[a]["lat"], g.nodes[a]["lon"],
            g.nodes[b]["lat"], g.nodes[b]["lon"],
        )
        return dist / speed_ms

    return h


def astar_route(g: nx.DiGraph, src: tuple, dst: tuple) -> list[tuple]:
    """A* on congestion-aware travel times with a straight-line heuristic.
    Finds the same optimal route as Dijkstra but explores fewer nodes."""
    return nx.astar_path(g, src, dst, heuristic=_time_heuristic(g), weight="travel_time")


def static_baseline_route(g: nx.DiGraph, src: tuple, dst: tuple) -> list[tuple]:
    """The naive baseline: shortest path on free-flow times, blind to traffic.
    Whatever it picks is then EVALUATED against real congested times, which is
    exactly the mistake a static dispatch system makes."""
    return nx.dijkstra_path(g, src, dst, weight="base_time")


def route_metrics(g: nx.DiGraph, path: list[tuple]) -> dict:
    """Sum up what travelling this route actually costs.

    travel_time_s uses the congested edge times - even for routes that were
    planned while ignoring congestion - so all planners are compared fairly
    on the same 'real world'.
    """
    travel_time = 0.0
    base_time = 0.0
    length_m = 0.0
    worst_congestion = 1.0

    for u, v in zip(path[:-1], path[1:]):
        edge = g.edges[u, v]
        travel_time += edge["travel_time"]
        base_time += edge["base_time"]
        length_m += edge["length_m"]
        worst_congestion = max(worst_congestion, edge["congestion"])

    return {
        "hops": len(path) - 1,
        "length_km": length_m / 1000.0,
        "travel_time_s": travel_time,
        "free_flow_time_s": base_time,
        "worst_congestion": worst_congestion,
    }


if __name__ == "__main__":
    # Self-test: route across the whole grid and confirm that
    # 1) Dijkstra and A* agree on the optimal congested time, and
    # 2) the congestion-blind baseline is no faster once evaluated on
    #    real congested times.
    from routing.graph import build_default_graph

    g = build_default_graph()
    src, dst = (0, 0), (29, 11)  # opposite corners of the grid

    dij = dijkstra_route(g, src, dst)
    ast = astar_route(g, src, dst)
    base = static_baseline_route(g, src, dst)

    m_dij = route_metrics(g, dij)
    m_ast = route_metrics(g, ast)
    m_base = route_metrics(g, base)

    print(f"Dijkstra : {m_dij['travel_time_s']:.0f}s over {m_dij['length_km']:.2f} km")
    print(f"A*       : {m_ast['travel_time_s']:.0f}s over {m_ast['length_km']:.2f} km")
    print(f"Baseline : {m_base['travel_time_s']:.0f}s over {m_base['length_km']:.2f} km "
          f"(planned blind to congestion)")

    assert abs(m_dij["travel_time_s"] - m_ast["travel_time_s"]) < 1e-6, \
        "Dijkstra and A* should find equally fast routes"
    assert m_dij["travel_time_s"] <= m_base["travel_time_s"] + 1e-6, \
        "Congestion-aware routing should never be slower than the blind baseline"
    print("Self-test passed.")
