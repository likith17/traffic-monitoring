# Phase 4, step 3: does anticipation pay off more on the real street graph?
#
# The grid benchmark (routing/predictive_benchmark.py) showed that a predictive
# planner matches the perfect-foresight oracle on recurring congestion, but its
# edge over simple reactive replanning was small, because a uniform grid has so
# many parallel streets that reacting to a jam late is almost as good as seeing it
# coming. The open question it left was whether that edge grows on a network with
# genuine bottlenecks, where committing toward the wrong river crossing is
# expensive to undo. This benchmark answers that on the real five-borough street
# graph.
#
# It reuses the dynamic-congestion model and the recurring, forecastable pattern,
# but runs on routing/streets.py's actual OpenStreetMap graph (about 55,100
# intersections). The reactive and forecast planners use the goal-directed,
# no-copy A* in dynamic_traffic.py, because the old snapshot-copy approach is far
# too slow at this scale. Origins and destinations are drawn far apart, so routes
# are long enough to cross the city and meet its bottlenecks.
#
# Because recurring jams sit on specific corridors rather than everywhere, their
# effect on any one trip is all-or-nothing: a jam either lands on the route or it
# does not. So alongside the mean travel time, this reports how often the
# predictive planner strictly beats the reactive one, which captures the
# "when it hits a bottleneck" behaviour that an average alone would wash out.
#
# Run:  python -m routing.real_benchmark --episodes 12

from __future__ import annotations

import argparse
import random

import networkx as nx
import pandas as pd

from routing.dynamic_traffic import (
    DynamicTraffic, simulate_drive, fixed_time_astar, time_dependent_astar,
)
from routing.graph import haversine_m
from routing.planners import static_baseline_route
from routing.recurrent_traffic import (
    recurring_pattern, forecast_events, realize_events,
)


def _drive_segment(actual, plan, t, cur, steps, replan_every):
    """Advance along the first replan_every hops of a plan; return new state."""
    for u, v in list(zip(plan[:-1], plan[1:]))[:replan_every]:
        t += actual.edge_time(u, v, t)
        cur = v
        steps += 1
    return t, cur, steps


def reactive_drive(actual, src, dst, t0=0.0, replan_every=5, cap_hops=4000):
    """Re-plan on CURRENT conditions every few hops, no forecast (no graph copy)."""
    t, cur, steps = t0, src, 0
    while cur != dst:
        if steps > cap_hops:
            return t - t0, False
        try:
            plan = fixed_time_astar(actual, cur, dst, t)
        except nx.NetworkXNoPath:
            return t - t0, False
        t, cur, steps = _drive_segment(actual, plan, t, cur, steps, replan_every)
    return t - t0, True


def predictive_adaptive_drive(actual, forecast, src, dst, t0=0.0,
                              replan_every=5, cap_hops=4000):
    """Re-plan on the FORECAST from the current clock every few hops."""
    t, cur, steps = t0, src, 0
    while cur != dst:
        if steps > cap_hops:
            return t - t0, False
        try:
            plan = time_dependent_astar(forecast, cur, dst, t)
        except nx.NetworkXNoPath:
            return t - t0, False
        t, cur, steps = _drive_segment(actual, plan, t, cur, steps, replan_every)
    return t - t0, True


def high_flow_centers(g, rng, n_centers, min_km, k_samples=60):
    """Find the arterials and crossings that carry the most traffic.

    Recurring congestion does not fall on random side streets; it concentrates on
    the corridors most trips use, which on this graph are the avenues and the
    limited river crossings. We approximate those by routing many random far-apart
    trips on free-flow times and counting how often each intersection is used. The
    busiest intersections are where recurring jams belong, and also where
    anticipating a jam matters most, since they are the hard-to-avoid bottlenecks.
    Computed once from a fixed seed so the set is stable across episodes.
    """
    from collections import Counter
    tally: Counter = Counter()
    picks = random.Random(1234)  # fixed, independent of the episode stream
    for _ in range(k_samples):
        src, dst = pick_od(g, picks, min_km)
        try:
            path = nx.dijkstra_path(g, src, dst, weight="base_time")
        except nx.NetworkXNoPath:
            continue
        for n in path[1:-1]:
            tally[n] += 1
    return [n for n, _ in tally.most_common(n_centers)]


def pick_od(g, rng, min_km):
    """A reachable origin/destination pair at least min_km apart (straight line)."""
    nodes = [n for n in g.nodes if "cam_id" not in g.nodes[n]]
    for _ in range(500):
        src, dst = rng.sample(nodes, 2)
        km = haversine_m(g.nodes[src]["lat"], g.nodes[src]["lon"],
                         g.nodes[dst]["lat"], g.nodes[dst]["lon"]) / 1000.0
        if km >= min_km and nx.has_path(g, src, dst):
            return src, dst
    raise RuntimeError("Could not find a far, reachable O/D pair")


def run_benchmark(episodes, seed, horizon_s=3000.0, min_km=8.0,
                  n_corridors=25, radius_m=1000.0, peak_points=60.0):
    rng = random.Random(seed)
    from routing.streets import build_street_graph
    base = build_street_graph()
    print(f"Street graph: {base.number_of_nodes():,} nodes, "
          f"{base.number_of_edges():,} edges")

    # Place the recurring jams on the busiest corridors, not random streets.
    centers = high_flow_centers(base, rng, n_corridors, min_km)
    pattern = recurring_pattern(base, n_corridors=n_corridors, horizon_s=horizon_s,
                                peak_points=peak_points, radius_m=radius_m,
                                centers=centers)
    forecast = DynamicTraffic(base, forecast_events(pattern))

    rows = []
    for ep in range(episodes):
        src, dst = pick_od(base, rng, min_km)
        actual = DynamicTraffic(base, realize_events(
            pattern, rng, horizon_s=horizon_s, surprise_radius_m=radius_m, g=base))

        naive_s = simulate_drive(actual, static_baseline_route(actual.g, src, dst), 0.0)
        static_s = simulate_drive(actual, fixed_time_astar(actual, src, dst, 0.0), 0.0)
        reactive_s, r_ok = reactive_drive(actual, src, dst, 0.0)
        predictive_s = simulate_drive(actual, time_dependent_astar(forecast, src, dst, 0.0), 0.0)
        predadapt_s, pa_ok = predictive_adaptive_drive(actual, forecast, src, dst, 0.0)
        oracle_s = simulate_drive(actual, time_dependent_astar(actual, src, dst, 0.0), 0.0)

        rows.append({
            "episode": ep, "naive_s": naive_s, "static_s": static_s,
            "reactive_s": reactive_s if r_ok else float("nan"),
            "predictive_s": predictive_s,
            "predadapt_s": predadapt_s if pa_ok else float("nan"),
            "oracle_s": oracle_s,
        })
        print(f"[EP {ep+1:>3}/{episodes}] naive {naive_s/60:5.1f} | "
              f"static {static_s/60:5.1f} | reactive {reactive_s/60:5.1f} | "
              f"predict {predictive_s/60:5.1f} | pred+adapt {predadapt_s/60:5.1f} | "
              f"oracle {oracle_s/60:5.1f}  (min)")

    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser(description="Predictive routing on the real street graph")
    ap.add_argument("--episodes", type=int, default=12)
    ap.add_argument("--seed", type=int, default=17)
    ap.add_argument("--min-km", type=float, default=8.0, help="min straight-line O/D distance")
    args = ap.parse_args()

    df = run_benchmark(args.episodes, args.seed, min_km=args.min_km)
    df.to_csv("real_benchmark.csv", index=False)

    naive = df["naive_s"].mean()
    static = df["static_s"].mean()
    reactive = df["reactive_s"].mean()
    predictive = df["predictive_s"].mean()
    predadapt = df["predadapt_s"].mean()
    oracle = df["oracle_s"].mean()

    def pct(x):
        return f"{(1 - x / naive) * 100:+5.1f}% vs naive"

    gap = reactive - oracle
    def closed(x):
        return "n/a" if abs(gap) <= 1e-9 else f"{100 * (reactive - x) / gap:4.0f}% of reactive gap"

    # How often does anticipation actually matter on a trip?
    wins = int((df["predadapt_s"] < df["reactive_s"] - 5.0).sum())
    big = int((df["reactive_s"] - df["predadapt_s"] > 60.0).sum())

    print(f"\n=== Mean travel time over {len(df)} real-graph episodes ===")
    print(f"  Naive free-flow path        : {naive/60:6.2f} min   (baseline)")
    print(f"  Static A* (departure only)  : {static/60:6.2f} min   {pct(static)}")
    print(f"  Reactive replan             : {reactive/60:6.2f} min   {pct(reactive)}")
    print(f"  Predictive (forecast, once) : {predictive/60:6.2f} min   {pct(predictive)}   closes {closed(predictive)}")
    print(f"  Predictive + adaptive       : {predadapt/60:6.2f} min   {pct(predadapt)}   closes {closed(predadapt)}")
    print(f"  Oracle time-dependent (ceil): {oracle/60:6.2f} min   {pct(oracle)}")
    print(f"\n  Predictive+adaptive beat reactive on {wins}/{len(df)} trips "
          f"({big} of them by over a minute).")
    print(f"  Reactive-to-oracle gap: {gap/60:.2f} min. Saved real_benchmark.csv")


if __name__ == "__main__":
    main()
