# Phase 4, step 1 (evaluation): how much does changing traffic cost a plan that
# does not account for it, and how much is left for a smarter method to win?
#
# The static benchmark (routing/simulate.py) cannot answer this, because it
# freezes congestion. Here congestion evolves while the vehicle drives
# (routing/dynamic_traffic.py), and four strategies are scored on the SAME
# episodes so the differences are only in the routing decision:
#
#   naive       - free-flow shortest path, blind to all congestion
#   static      - A* on the congestion known at departure, then drive blindly;
#                 optimal for the snapshot, but the snapshot goes stale
#   adaptive    - re-plan from the current position every few blocks using
#                 current conditions; reacts to jams but cannot see them coming
#   oracle (TD) - earliest-arrival route given the full forecast; needs perfect
#                 foresight, so it is a ceiling, not a deployable method
#
# The point of this step is not to declare a winner. It is to show that once
# traffic changes, "static" is no longer optimal, "adaptive" recovers part of the
# loss by reacting, and a gap to the oracle remains - the gap that a predictive or
# learned policy (Phase 4's next steps) would try to close. If that gap is real
# and sizable, the learning phase has something to beat; if it is not, that is
# worth knowing before building a model.
#
# Run:  python -m routing.dynamic_benchmark --episodes 25

from __future__ import annotations

import argparse
import random

import networkx as nx
import pandas as pd

from routing.dynamic_traffic import (
    DynamicTraffic, random_events, simulate_drive, time_dependent_route,
)
from routing.graph import build_default_graph
from routing.planners import astar_route, static_baseline_route


def adaptive_drive(
    dyn: DynamicTraffic, src: tuple, dst: tuple, t0: float = 0.0,
    replan_every: int = 2, max_steps_factor: int = 5,
) -> tuple[float, bool]:
    """Drive while re-planning from the current position on current conditions.

    Every replan_every blocks the route is recomputed on a snapshot of the
    congestion as it is right now - the honest information a live-traffic
    navigator has, with no forecast. Returns (realized_time, reached). The step
    cap guards against a pathological reroute oscillation; reaching it counts as
    not arriving, which is reported rather than hidden.
    """
    t = t0
    cur = src
    steps = 0
    # A generous cap: a sane reactive drive never needs this many blocks.
    cap = max_steps_factor * dyn.g.number_of_nodes() ** 0.5 * 4

    while cur != dst:
        if steps > cap:
            return t - t0, False
        snap = dyn.snapshot(t)
        try:
            plan = astar_route(snap, cur, dst)
        except nx.NetworkXNoPath:
            return t - t0, False
        # Follow the fresh plan for a few blocks before re-planning again.
        for u, v in list(zip(plan[:-1], plan[1:]))[:replan_every]:
            t += dyn.edge_time(u, v, t)
            cur = v
            steps += 1
            if cur == dst:
                break

    return t - t0, True


def make_episode(base: nx.DiGraph, rng: random.Random, n_events: int, horizon_s: float):
    """One dynamic emergency: a set of congestion events and a far-apart O/D."""
    events = random_events(base, rng, n_events, horizon_s)
    dyn = DynamicTraffic(base, events)

    free = [n for n in base.nodes if "cam_id" not in base.nodes[n]]
    while True:
        src, dst = rng.sample(free, 2)
        if abs(src[0] - dst[0]) + abs(src[1] - dst[1]) >= 12:
            return dyn, src, dst


def run_benchmark(episodes: int, n_events: int, seed: int) -> pd.DataFrame:
    rng = random.Random(seed)
    base = build_default_graph()
    # A time horizon comfortably longer than a typical trip on this network.
    horizon_s = 3000.0
    rows = []

    for ep in range(episodes):
        dyn, src, dst = make_episode(base, rng, n_events, horizon_s)

        naive_path = static_baseline_route(dyn.g, src, dst)
        static_path = astar_route(dyn.snapshot(0.0), src, dst)

        naive_s = simulate_drive(dyn, naive_path, 0.0)
        static_s = simulate_drive(dyn, static_path, 0.0)
        adaptive_s, reached = adaptive_drive(dyn, src, dst, 0.0)
        oracle_s = simulate_drive(dyn, time_dependent_route(dyn, src, dst, 0.0), 0.0)

        rows.append({
            "episode": ep, "naive_s": naive_s, "static_s": static_s,
            "adaptive_s": adaptive_s if reached else float("nan"),
            "oracle_s": oracle_s, "adaptive_reached": reached,
        })
        print(f"[EP {ep+1:>3}/{episodes}] naive {naive_s/60:5.1f} | "
              f"static {static_s/60:5.1f} | "
              f"adaptive {adaptive_s/60:5.1f}{'' if reached else '*'} | "
              f"oracle {oracle_s/60:5.1f}  (min)")

    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description="Dynamic-conditions routing benchmark")
    ap.add_argument("--episodes", type=int, default=25)
    ap.add_argument("--events", type=int, default=10, help="congestion surges per episode")
    ap.add_argument("--seed", type=int, default=11)
    args = ap.parse_args()

    df = run_benchmark(args.episodes, args.events, args.seed)
    df.to_csv("dynamic_benchmark.csv", index=False)

    naive = df["naive_s"].mean()
    static = df["static_s"].mean()
    adaptive = df["adaptive_s"].mean()  # NaN-safe: unreached episodes excluded
    oracle = df["oracle_s"].mean()
    reached = int(df["adaptive_reached"].sum())

    def pct(x):
        return f"{(1 - x / naive) * 100:+.1f}% vs naive"

    print(f"\n=== Mean travel time over {len(df)} dynamic episodes "
          f"({args.events} congestion events each) ===")
    print(f"  Naive free-flow path        : {naive/60:6.2f} min   (baseline)")
    print(f"  Static A* (departure only)  : {static/60:6.2f} min   {pct(static)}")
    print(f"  Adaptive replan (reactive)  : {adaptive/60:6.2f} min   {pct(adaptive)}"
          f"   [{reached}/{len(df)} reached]")
    print(f"  Oracle time-dependent (ceil): {oracle/60:6.2f} min   {pct(oracle)}")
    print(f"\n  Room reactive replanning leaves on the table vs the forecast "
          f"ceiling: {(adaptive - oracle)/60:.2f} min "
          f"({100*(adaptive-oracle)/adaptive:.0f}% of the adaptive time).")
    print("  That gap is what a predictive or learned policy (next steps) would")
    print("  try to close. Saved dynamic_benchmark.csv")


if __name__ == "__main__":
    main()
