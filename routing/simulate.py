# The benchmark: does the smart pipeline (congestion-aware planning + RL +
# vision confirmation) actually get emergency vehicles there faster than a
# naive static dispatcher?
#
# How one simulated emergency works:
#   1. A random incident location and destination are drawn on the map.
#   2. Random road blockages appear at some camera intersections.  The
#      planners' congestion map is STALE - it does not know about them.
#      Only the cameras (the vision gate) can reveal them at dispatch time.
#   3. Three strategies plan a route:
#        static   - shortest path on free-flow times, blind to everything
#        astar    - A* on the congestion map + vision gate re-planning
#        rl       - Q-learning agent + vision gate re-planning
#   4. Every route is then scored against the TRUE state of the streets,
#      where driving into a blockage costs heavily (the vehicle crawls
#      through or waits for it to clear).
#
# Run:  python -m routing.simulate --episodes 30
# Output: summary table on stdout + routing_benchmark.csv

from __future__ import annotations

import argparse
import random

import networkx as nx
import pandas as pd

from routing.graph import build_default_graph
from routing.planners import astar_route, static_baseline_route
from routing.rl_agent import QLearningRouter
from routing.vision_gate import plan_confirmed_route

# How much slower a street becomes when it is actually blocked: the vehicle
# inches through the jam / waits for it to be cleared.
BLOCKAGE_SLOWDOWN = 8.0

# Congestion score the incident cameras report during a blockage.
BLOCKED_CAM_SCORE = 60.0


def true_route_time(g_true: nx.DiGraph, path: list[tuple]) -> float:
    """What the route really costs in seconds, on the true street state."""
    return sum(g_true.edges[u, v]["travel_time"] for u, v in zip(path[:-1], path[1:]))


def make_episode(
    base: nx.DiGraph, rng: random.Random, n_incidents: int
) -> tuple[nx.DiGraph, nx.DiGraph, tuple, tuple]:
    """Set up one simulated emergency.

    Returns (g_plan, g_true, src, dst):
      g_plan - what the planners see: stale congestion weights, but camera
               scores updated to the truth (cameras always show 'now').
      g_true - reality: blocked streets are massively slower.
    """
    g_plan = base.copy()
    g_true = base.copy()

    # Blockages appear at random camera intersections.
    cam_nodes = [n for n, d in g_plan.nodes(data=True) if "cam_id" in d]
    incidents = rng.sample(cam_nodes, min(n_incidents, len(cam_nodes)))

    for node in incidents:
        # The camera sees the blockage - both graphs get the fresh score,
        # because a camera feed is always current.
        for g in (g_plan, g_true):
            g.nodes[node]["cam_score"] = BLOCKED_CAM_SCORE

        # But only reality slows the streets down; the planners' congestion
        # weights are stale and still show normal traffic there.
        for u, v in list(g_true.in_edges(node)) + list(g_true.out_edges(node)):
            edge = g_true.edges[u, v]
            edge["travel_time"] = edge["base_time"] * BLOCKAGE_SLOWDOWN

    # Emergency start and destination: random intersections without cameras,
    # far enough apart to make routing non-trivial.
    free = [n for n in g_plan.nodes if "cam_id" not in g_plan.nodes[n]]
    while True:
        src, dst = rng.sample(free, 2)
        # Require at least a third of the grid between them (manhattan metric).
        if abs(src[0] - dst[0]) + abs(src[1] - dst[1]) >= 12:
            return g_plan, g_true, src, dst


def rl_planner(g: nx.DiGraph, src: tuple, dst: tuple) -> list[tuple]:
    """Adapter so the Q-learning agent has the same (graph, src, dst)
    signature as the classical planners and works with the vision gate."""
    agent = QLearningRouter(g, seed=0)
    agent.train(src, dst, episodes=800)
    return agent.best_route()


def run_benchmark(episodes: int, n_incidents: int, seed: int) -> pd.DataFrame:
    """Simulate the requested number of emergencies and record every result."""
    rng = random.Random(seed)
    base = build_default_graph()
    rows = []

    for ep in range(episodes):
        g_plan, g_true, src, dst = make_episode(base, rng, n_incidents)

        # Strategy 1: the naive static dispatcher (no traffic, no cameras).
        static_path = static_baseline_route(g_plan, src, dst)

        # Strategy 2: A* on congestion weights, confirmed by the vision gate.
        astar_path, _ = plan_confirmed_route(g_plan, src, dst, planner=astar_route)

        # Strategy 3: RL policy, confirmed by the vision gate.
        rl_path, _ = plan_confirmed_route(g_plan, src, dst, planner=rl_planner)

        rows.append({
            "episode": ep,
            "src": str(src),
            "dst": str(dst),
            "static_s": true_route_time(g_true, static_path),
            "astar_vision_s": true_route_time(g_true, astar_path),
            "rl_vision_s": true_route_time(g_true, rl_path),
        })
        print(
            f"[EP {ep + 1:>3}/{episodes}] static {rows[-1]['static_s']:6.0f}s | "
            f"A*+vision {rows[-1]['astar_vision_s']:6.0f}s | "
            f"RL+vision {rows[-1]['rl_vision_s']:6.0f}s"
        )

    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Emergency routing benchmark")
    parser.add_argument("--episodes", type=int, default=30, help="simulated emergencies")
    parser.add_argument("--incidents", type=int, default=6, help="road blockages per episode")
    parser.add_argument("--seed", type=int, default=7, help="random seed")
    args = parser.parse_args()

    df = run_benchmark(args.episodes, args.incidents, args.seed)
    df.to_csv("routing_benchmark.csv", index=False)

    static = df["static_s"].mean()
    astar = df["astar_vision_s"].mean()
    rl = df["rl_vision_s"].mean()

    print("\n=== Mean simulated response time over "
          f"{len(df)} emergencies ({args.incidents} blockages each) ===")
    print(f"  Static shortest-path baseline : {static / 60:6.1f} min")
    print(f"  A* + vision gate              : {astar / 60:6.1f} min "
          f"({(1 - astar / static) * 100:+.1f}% vs baseline)")
    print(f"  Q-learning + vision gate      : {rl / 60:6.1f} min "
          f"({(1 - rl / static) * 100:+.1f}% vs baseline)")
    print("\nSaved per-episode results to routing_benchmark.csv")


if __name__ == "__main__":
    main()
