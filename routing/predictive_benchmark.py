# Phase 4, step 2 (evaluation): does forecasting recurring congestion beat
# reacting to it?
#
# Step 1's benchmark scattered jams at random, so "the forecast" carried no
# information and a predictive planner could not pull ahead of a reactive one.
# This benchmark uses the recurring, forecastable structure from
# routing/recurrent_traffic.py and adds the two methods that structure makes
# meaningful, scoring six strategies on the SAME episodes:
#
#   naive        - free-flow shortest path, blind to all congestion
#   static       - A* on conditions known at departure, then drive blindly
#   adaptive     - re-plan from the current position on CURRENT conditions;
#                  reacts to jams but cannot see them coming (no forecast)
#   predictive   - plan once on the FORECAST (the recurring pattern), anticipating
#                  jams before reaching them; driven on the real realization
#   pred+adaptive- re-plan on the forecast from the current clock every few blocks;
#                  anticipation plus reaction, the realistic deployable policy
#   oracle (TD)  - earliest arrival given the REAL realization; perfect foresight,
#                  a ceiling, not deployable
#
# The question this answers: once congestion is predictable, how much does routing
# against the forecast (predictive, pred+adaptive) save over merely reacting
# (adaptive), and how much of the gap up to the perfect-foresight oracle does it
# close? The residual gap is the unforecastable part (the surprise incidents),
# which no forecast can remove - and saying so is the point.
#
# Run:  python -m routing.predictive_benchmark --episodes 40

from __future__ import annotations

import argparse
import random

import networkx as nx
import pandas as pd

from routing.dynamic_traffic import (
    DynamicTraffic, simulate_drive, time_dependent_route,
)
from routing.dynamic_benchmark import adaptive_drive
from routing.graph import build_default_graph
from routing.planners import astar_route, static_baseline_route
from routing.recurrent_traffic import (
    recurring_pattern, forecast_events, realize_events,
)


def predictive_adaptive_drive(
    actual: DynamicTraffic, forecast: DynamicTraffic,
    src: tuple, dst: tuple, t0: float = 0.0,
    replan_every: int = 2, max_steps_factor: int = 5,
) -> tuple[float, bool]:
    """Drive while re-planning on the FORECAST from the current clock.

    Unlike reactive adaptive (which snapshots current conditions and cannot see a
    jam coming), this recomputes an earliest-arrival route on the recurring
    forecast starting from where and when the vehicle is now - so it keeps
    anticipating scheduled jams, while the re-planning lets it adjust when the real
    drive drifts from the forecast. Costs are paid on the real realization.
    """
    t = t0
    cur = src
    steps = 0
    cap = max_steps_factor * actual.g.number_of_nodes() ** 0.5 * 4
    while cur != dst:
        if steps > cap:
            return t - t0, False
        try:
            plan = time_dependent_route(forecast, cur, dst, t)
        except nx.NetworkXNoPath:
            return t - t0, False
        for u, v in list(zip(plan[:-1], plan[1:]))[:replan_every]:
            t += actual.edge_time(u, v, t)
            cur = v
            steps += 1
            if cur == dst:
                break
    return t - t0, True


def make_episode(
    base: nx.DiGraph, forecast: DynamicTraffic, pattern, rng: random.Random,
    horizon_s: float,
):
    """One day: the fixed pattern realized with noise + surprises, and a far O/D."""
    actual_events = realize_events(pattern, rng, horizon_s=horizon_s, g=base)
    actual = DynamicTraffic(base, actual_events)
    interior = [n for n in base.nodes if "cam_id" not in base.nodes[n]]
    while True:
        src, dst = rng.sample(interior, 2)
        if abs(src[0] - dst[0]) + abs(src[1] - dst[1]) >= 12:
            return actual, src, dst


def run_benchmark(episodes: int, seed: int, horizon_s: float = 6000.0) -> pd.DataFrame:
    rng = random.Random(seed)
    base = build_default_graph()

    # The city's recurring rhythm, fixed for every episode - this is the history a
    # forecast is built from, so the predictive methods may know it exactly.
    pattern = recurring_pattern(base, horizon_s=horizon_s)
    forecast = DynamicTraffic(base, forecast_events(pattern))

    rows = []
    for ep in range(episodes):
        actual, src, dst = make_episode(base, forecast, pattern, rng, horizon_s)

        naive_path = static_baseline_route(actual.g, src, dst)
        static_path = astar_route(actual.snapshot(0.0), src, dst)

        naive_s = simulate_drive(actual, naive_path, 0.0)
        static_s = simulate_drive(actual, static_path, 0.0)
        adaptive_s, a_ok = adaptive_drive(actual, src, dst, 0.0)
        predictive_s = simulate_drive(
            actual, time_dependent_route(forecast, src, dst, 0.0), 0.0)
        predadapt_s, pa_ok = predictive_adaptive_drive(actual, forecast, src, dst, 0.0)
        oracle_s = simulate_drive(
            actual, time_dependent_route(actual, src, dst, 0.0), 0.0)

        rows.append({
            "episode": ep, "naive_s": naive_s, "static_s": static_s,
            "adaptive_s": adaptive_s if a_ok else float("nan"),
            "predictive_s": predictive_s,
            "predadapt_s": predadapt_s if pa_ok else float("nan"),
            "oracle_s": oracle_s,
        })
        print(f"[EP {ep+1:>3}/{episodes}] naive {naive_s/60:5.1f} | "
              f"static {static_s/60:5.1f} | adaptive {adaptive_s/60:5.1f} | "
              f"predict {predictive_s/60:5.1f} | pred+adapt {predadapt_s/60:5.1f} | "
              f"oracle {oracle_s/60:5.1f}  (min)")

    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description="Predictive routing benchmark (recurring congestion)")
    ap.add_argument("--episodes", type=int, default=40)
    ap.add_argument("--seed", type=int, default=17)
    args = ap.parse_args()

    df = run_benchmark(args.episodes, args.seed)
    df.to_csv("predictive_benchmark.csv", index=False)

    naive = df["naive_s"].mean()
    static = df["static_s"].mean()
    adaptive = df["adaptive_s"].mean()
    predictive = df["predictive_s"].mean()
    predadapt = df["predadapt_s"].mean()
    oracle = df["oracle_s"].mean()

    def pct(x):
        return f"{(1 - x / naive) * 100:+5.1f}% vs naive"

    # How much of the reactive-to-oracle gap does each forecast method close?
    gap = adaptive - oracle
    def closed(x):
        return "n/a" if gap <= 1e-9 else f"{100 * (adaptive - x) / gap:4.0f}% of the reactive gap"

    print(f"\n=== Mean travel time over {len(df)} episodes of recurring congestion ===")
    print(f"  Naive free-flow path        : {naive/60:6.2f} min   (baseline)")
    print(f"  Static A* (departure only)  : {static/60:6.2f} min   {pct(static)}")
    print(f"  Adaptive replan (reactive)  : {adaptive/60:6.2f} min   {pct(adaptive)}")
    print(f"  Predictive (forecast, once) : {predictive/60:6.2f} min   {pct(predictive)}   closes {closed(predictive)}")
    print(f"  Predictive + adaptive       : {predadapt/60:6.2f} min   {pct(predadapt)}   closes {closed(predadapt)}")
    print(f"  Oracle time-dependent (ceil): {oracle/60:6.2f} min   {pct(oracle)}")
    print(f"\n  Reactive-to-oracle gap (the forecastable room): {gap/60:.2f} min.")
    print(f"  Residual gap after pred+adaptive (the unforecastable surprises): "
          f"{(predadapt - oracle)/60:.2f} min.")
    print("  Saved predictive_benchmark.csv")


if __name__ == "__main__":
    main()
