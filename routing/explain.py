# Turns a route decision into a plain-English justification a dispatcher can
# read out loud: which route was chosen, why, what was avoided, and how much
# time it saves over the naive alternative.
#
# Two layers:
#   1. build_route_summary() collects the hard facts (times, distances,
#      blocked cameras) into a compact text block.
#   2. explain_route() asks the LLM (via the existing traffic_llm helpers)
#      to phrase those facts naturally.  If no API key is configured, a
#      template-based fallback produces a decent explanation anyway, so the
#      module never fails in Docker demos or offline runs.

from __future__ import annotations

import networkx as nx

from routing.planners import route_metrics
from traffic_llm import chat_completion


def _fmt_minutes(seconds: float) -> str:
    """Human-friendly duration, e.g. 754s -> '12.6 min'."""
    return f"{seconds / 60:.1f} min"


def build_route_summary(
    g: nx.DiGraph,
    route: list[tuple],
    baseline: list[tuple] | None = None,
    gate_info: dict | None = None,
    strategy: str = "RL + vision",
) -> str:
    """Collect every fact the explanation needs into one small text block.

    This same text is used as the LLM prompt context AND as the source for
    the offline fallback, so both explanations are always consistent.
    """
    m = route_metrics(g, route)
    lines = [
        f"Routing strategy: {strategy}",
        f"Chosen route: {m['hops']} blocks, {m['length_km']:.2f} km, "
        f"estimated {_fmt_minutes(m['travel_time_s'])} in current traffic "
        f"({_fmt_minutes(m['free_flow_time_s'])} with no traffic).",
        f"Worst congestion multiplier along the route: x{m['worst_congestion']:.2f}.",
    ]

    if baseline is not None:
        mb = route_metrics(g, baseline)
        saved = mb["travel_time_s"] - m["travel_time_s"]
        lines.append(
            f"Static shortest-path baseline would take {_fmt_minutes(mb['travel_time_s'])} "
            f"in current traffic; the chosen route saves {_fmt_minutes(max(saved, 0.0))}."
        )

    if gate_info:
        if gate_info.get("confirmed"):
            lines.append(
                f"Vision check: route confirmed clear by camera review "
                f"in {gate_info['attempts']} attempt(s)."
            )
        else:
            lines.append("Vision check: could NOT fully confirm the route - dispatch with caution.")

        for cam in gate_info.get("blocked_cameras", []):
            lines.append(
                f"Avoided blocked intersection at camera '{cam['camera']}' "
                f"(congestion score {cam['score']:.0f})."
            )
        for cam in gate_info.get("endpoint_warnings", []):
            lines.append(
                f"Warning: heavy congestion at the start/destination itself "
                f"('{cam['camera']}', score {cam['score']:.0f}) - unavoidable."
            )

    return "\n".join(lines)


def _fallback_explanation(summary: str) -> str:
    """No LLM available: reshape the factual summary into readable prose.
    Intentionally boring but always correct and always available."""
    return (
        "Route decision summary (automatic, no LLM configured):\n"
        + summary
        + "\nThe route was selected because it minimises estimated travel time on "
        "congestion-weighted streets, and every camera along it was checked "
        "before dispatch."
    )


def explain_route(
    g: nx.DiGraph,
    route: list[tuple],
    baseline: list[tuple] | None = None,
    gate_info: dict | None = None,
    strategy: str = "RL + vision",
) -> str:
    """Produce the final human-readable justification for a route choice.

    Tries the configured LLM first (same environment variables as the
    dashboard chat tab).  Any error - missing key, network down, bad model -
    silently drops to the template fallback: an emergency tool must always
    answer.
    """
    summary = build_route_summary(g, route, baseline, gate_info, strategy)

    messages = [
        {
            "role": "system",
            "content": (
                "You are the dispatch assistant of an emergency routing system. "
                "Using ONLY the facts below, explain in 3-5 short sentences why "
                "this route was chosen. Mention avoided blockages and time saved "
                "if present. Plain language, no bullet points, no headers.\n\n"
                f"Facts:\n{summary}"
            ),
        },
        {"role": "user", "content": "Why was this route chosen?"},
    ]

    reply, err = chat_completion(messages, timeout=30)
    if err or not reply:
        return _fallback_explanation(summary)
    return reply


if __name__ == "__main__":
    # Self-test: run the full pipeline pieces together and print both the
    # factual summary and the final explanation (fallback text when no LLM
    # key is configured - that is expected and fine).
    from routing.graph import build_default_graph
    from routing.planners import astar_route, static_baseline_route
    from routing.vision_gate import cameras_on_route, plan_confirmed_route

    g = build_default_graph()
    free = [n for n in g.nodes if "cam_id" not in g.nodes[n]]
    src, dst = min(free), max(free)

    # Simulate one blocked camera so the explanation has something to say.
    naive = astar_route(g, src, dst)
    cams = cameras_on_route(g, naive)
    if cams:
        g.nodes[cams[0]]["cam_score"] = 99.0

    route, info = plan_confirmed_route(g, src, dst, planner=astar_route)
    baseline = static_baseline_route(g, src, dst)

    print("--- factual summary ---")
    print(build_route_summary(g, route, baseline, info, strategy="A* + vision"))
    print("\n--- explanation ---")
    print(explain_route(g, route, baseline, info, strategy="A* + vision"))
