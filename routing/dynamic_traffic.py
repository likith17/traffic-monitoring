# Phase 4, step 1: traffic that changes while you drive.
#
# Every routing result so far is measured on a static snapshot: the congestion on
# each street is fixed for the whole trip. On a fixed snapshot A* is provably
# optimal, which is exactly why the learned agent has nothing to beat and the
# benchmark cannot reward anticipating the future. This module builds the missing
# piece - an environment where congestion rises and fades over time - so that
# planning which accounts for *when* a street is reached can do better than
# planning that freezes the map at departure.
#
# The model stays consistent with graph.py: a street's cost is still
# base_time * multiplier, and one congestion "point" still slows it by
# CONGESTION_PER_POINT. The only new thing is that the points are a function of
# time. Congestion arrives as events - a surge that builds to a peak and fades,
# localised around a point in the city with a spatial falloff - layered on top of
# the static congestion already on the graph. Events are interpretable (you can
# say where and when each jam is) and reproducible from a seed.
#
# What this module provides:
#   DynamicTraffic.edge_time(u, v, t)  - a street's travel time at clock t
#   DynamicTraffic.snapshot(t)         - a graph frozen at t, for a normal planner
#   simulate_drive(dyn, path, t0)      - the true cost of a path as it is driven
#   time_dependent_route(dyn, s, d, t0) - the best route given the full forecast
#                                         (an omniscient ceiling, not something a
#                                          real dispatcher has)

from __future__ import annotations

import heapq
import random
from dataclasses import dataclass

import networkx as nx

from routing.graph import CONGESTION_PER_POINT, haversine_m


@dataclass
class CongestionEvent:
    """One localised surge of congestion that builds to a peak and fades.

    center is a graph node the surge is centred on. The extra congestion at the
    centre rises linearly from t_start to t_peak, then falls to zero by t_end,
    and drops off with distance out to radius_m.
    """

    center: tuple
    t_start: float
    t_peak: float
    t_end: float
    peak_points: float
    radius_m: float

    def time_factor(self, t: float) -> float:
        """0 outside the event, rising to 1 at the peak, back to 0 at the end."""
        if t <= self.t_start or t >= self.t_end:
            return 0.0
        if t < self.t_peak:
            return (t - self.t_start) / (self.t_peak - self.t_start)
        return (self.t_end - t) / (self.t_end - self.t_peak)

    def extra_points_at(self, dist_m: float, t: float) -> float:
        """Extra congestion points this event adds at a distance and a time."""
        if dist_m >= self.radius_m:
            return 0.0
        space = 1.0 - dist_m / self.radius_m       # linear spatial falloff
        return self.peak_points * self.time_factor(t) * space


class DynamicTraffic:
    """A graph whose edge travel times vary with time via congestion events."""

    def __init__(self, g: nx.DiGraph, events: list[CongestionEvent]):
        self.g = g
        self.events = events
        # Cache each node's position and each edge's midpoint so the per-edge,
        # per-time cost does not recompute geometry on every query.
        self._pos = {n: (d["lat"], d["lon"]) for n, d in g.nodes(data=True)}
        self._mid: dict[tuple, tuple] = {}
        for u, v in g.edges():
            ulat, ulon = self._pos[u]
            vlat, vlon = self._pos[v]
            self._mid[(u, v)] = ((ulat + vlat) / 2.0, (ulon + vlon) / 2.0)

    def extra_points(self, u: tuple, v: tuple, t: float) -> float:
        """Total time-varying congestion points on edge (u, v) at clock t."""
        if not self.events:
            return 0.0
        mlat, mlon = self._mid[(u, v)]
        total = 0.0
        for ev in self.events:
            clat, clon = self._pos[ev.center]
            d = haversine_m(mlat, mlon, clat, clon)
            total += ev.extra_points_at(d, t)
        return total

    def edge_multiplier(self, u: tuple, v: tuple, t: float) -> float:
        """Congestion multiplier on (u, v) at clock t: static plus time-varying."""
        static = self.g.edges[u, v]["congestion"]          # >= 1.0, from cameras
        return static + CONGESTION_PER_POINT * self.extra_points(u, v, t)

    def edge_time(self, u: tuple, v: tuple, t: float) -> float:
        """Travel time in seconds to cross (u, v) entering at clock t."""
        return self.g.edges[u, v]["base_time"] * self.edge_multiplier(u, v, t)

    def snapshot(self, t: float) -> nx.DiGraph:
        """A copy of the graph with travel_time frozen at clock t.

        This is what a planner sees if it assumes current conditions hold for the
        whole trip - the honest information a dispatcher has at departure (or at a
        mid-route replan), with no knowledge of how the jams will evolve.
        """
        h = self.g.copy()
        for u, v in h.edges():
            mult = self.edge_multiplier(u, v, t)
            h.edges[u, v]["congestion"] = mult
            h.edges[u, v]["travel_time"] = h.edges[u, v]["base_time"] * mult
        return h


def simulate_drive(dyn: DynamicTraffic, path: list[tuple], t0: float = 0.0) -> float:
    """The true travel time of a path, paying each street's cost as it is reached.

    The clock advances edge by edge, so a jam that has faded by the time the
    vehicle arrives costs nothing, and one that peaks on arrival costs the most.
    This is the honest score for any route under the dynamic model.
    """
    t = t0
    for u, v in zip(path[:-1], path[1:]):
        t += dyn.edge_time(u, v, t)
    return t - t0


def time_dependent_route(
    dyn: DynamicTraffic, src: tuple, dst: tuple, t0: float = 0.0
) -> list[tuple]:
    """Earliest-arrival route given the full congestion forecast.

    Time-dependent Dijkstra on arrival time: the label of a node is the earliest
    clock it can be reached, and an edge is relaxed using its cost at that clock.
    With costs that vary smoothly this is the best a router could do if it knew
    the future exactly, so it serves as the ceiling the learned methods aim at -
    not something a real dispatcher can run, since it needs perfect foresight.
    """
    best = {src: t0}
    prev: dict[tuple, tuple] = {}
    pq = [(t0, src)]
    while pq:
        t, u = heapq.heappop(pq)
        if u == dst:
            break
        if t > best.get(u, float("inf")):
            continue
        for v in dyn.g.successors(u):
            arrive = t + dyn.edge_time(u, v, t)
            if arrive < best.get(v, float("inf")):
                best[v] = arrive
                prev[v] = u
                heapq.heappush(pq, (arrive, v))

    if dst not in prev and dst != src:
        raise nx.NetworkXNoPath(f"No time-dependent route {src} -> {dst}")

    path = [dst]
    while path[-1] != src:
        path.append(prev[path[-1]])
    path.reverse()
    return path


def random_events(
    g: nx.DiGraph, rng: random.Random, n_events: int, horizon_s: float,
    peak_points: float = 40.0, radius_m: float = 500.0,
) -> list[CongestionEvent]:
    """Scatter n_events congestion surges over the map and the time horizon.

    Each surge is centred on a random intersection, starts at a random time in
    the horizon, and lasts a few minutes. Seeded through rng so an episode is
    reproducible. peak_points of 40 is a severe jam (4x slowdown at the centre),
    matching the blockage scale used elsewhere in the benchmark.
    """
    nodes = list(g.nodes)
    events = []
    for _ in range(n_events):
        start = rng.uniform(0, horizon_s * 0.7)
        duration = rng.uniform(180, 420)  # 3-7 minutes
        events.append(CongestionEvent(
            center=rng.choice(nodes),
            t_start=start,
            t_peak=start + duration * 0.4,
            t_end=start + duration,
            peak_points=peak_points * rng.uniform(0.6, 1.0),
            radius_m=radius_m * rng.uniform(0.7, 1.3),
        ))
    return events


if __name__ == "__main__":
    # Deterministic self-test: build a jam that peaks right where and when a
    # straight-through route would arrive, and check that (a) the dynamic cost
    # exceeds the static one, and (b) the time-dependent router, seeing the
    # forecast, arrives no later than a static plan that drives blindly into it.
    from routing.graph import build_default_graph
    from routing.planners import astar_route

    g = build_default_graph()
    free = [n for n in g.nodes if "cam_id" not in g.nodes[n]]
    src, dst = min(free), max(free)

    static_path = astar_route(g, src, dst)

    # Place a severe, long jam centred on the middle of that static route, active
    # across the whole drive so blind driving is certain to hit it.
    mid_node = static_path[len(static_path) // 2]
    ev = CongestionEvent(
        center=mid_node, t_start=0.0, t_peak=300.0, t_end=3000.0,
        peak_points=80.0, radius_m=800.0,
    )
    dyn = DynamicTraffic(g, [ev])

    # (a) time factor is triangular and bounded.
    assert ev.time_factor(-1) == 0.0 and ev.time_factor(3001) == 0.0
    assert abs(ev.time_factor(300.0) - 1.0) < 1e-9
    assert 0.0 < ev.time_factor(150.0) < 1.0

    # (b) the same street costs more mid-event than with no event.
    u, v = static_path[len(static_path) // 2], static_path[len(static_path) // 2 + 1]
    base = g.edges[u, v]["base_time"] * g.edges[u, v]["congestion"]
    assert dyn.edge_time(u, v, 300.0) > base, "event should slow the edge"

    static_cost = simulate_drive(dyn, static_path, 0.0)
    td_path = time_dependent_route(dyn, src, dst, 0.0)
    td_cost = simulate_drive(dyn, td_path, 0.0)

    print(f"Static route driven through the jam : {static_cost/60:.1f} min "
          f"({len(static_path)} nodes)")
    print(f"Time-dependent route (knows forecast): {td_cost/60:.1f} min "
          f"({len(td_path)} nodes)")
    print(f"Foresight saves: {(static_cost - td_cost)/60:.1f} min")

    assert td_cost <= static_cost + 1e-6, "forecast-aware route must not be worse"
    print("dynamic_traffic.py self-test OK")
