# Phase 3 end-to-end demo: watch the router steer around a confirmed incident.
#
# The vision gate already reroutes around heavy congestion (a high camera
# score). Phase 3 adds a second, independent reason to avoid a street: a
# vision-language model confirming a blocking incident in the camera frame - a
# crash, a vehicle stalled in a live lane, debris, or flooding. This script
# shows that end to end.
#
# To stay reproducible and to run with no API key, the incident here is SEEDED:
# one camera on the initial route is made to return an "accident" verdict, while
# its congestion score is forced low. That combination is the whole point - it
# proves the reroute is caused by the incident itself, not by traffic volume.
# With a vision model configured and a real crash in view, the exact same gate
# path runs live; only the verdict's source changes.
#
# Run:  python -m routing.incident_demo

from __future__ import annotations

import numpy as np

import routing.incident_detect as incident_detect
import routing.vision_gate as vision_gate
from routing.explain import explain_route
from routing.graph import build_default_graph
from routing.planners import astar_route, route_metrics
from routing.vision_gate import cameras_on_route, plan_confirmed_route


def _seed_incident(crash_name: str):
    """Return patched (score_frame_fn, incident_fn) that stage one accident.

    Every camera reads a low score and hands back a dummy frame, so nothing
    blocks on congestion. Only the named camera returns a confirmed accident, so
    any reroute must be the incident's doing.
    """
    def fake_score_and_frame(g, node, model=None):
        return 1.0, np.zeros((8, 8, 3), dtype=np.uint8)

    def fake_incident(frame, camera_id="", camera_name="", *, use_vlm=True, timeout=30):
        is_crash = camera_name == crash_name
        return incident_detect.IncidentReport(
            camera_id=camera_id, camera_name=camera_name,
            candidate=is_crash, stage1_reason="seeded demo",
            source="vlm", incident=is_crash,
            category="accident" if is_crash else "none",
            confidence=0.92 if is_crash else 0.0,
            note="two vehicles blocking the lane" if is_crash else "clear",
            score=1.0, queue_length=0, unique_vehicles=0)

    return fake_score_and_frame, fake_incident


def main() -> None:
    g = build_default_graph()
    free = [n for n in g.nodes if "cam_id" not in g.nodes[n]]
    src, dst = min(free), max(free)

    naive = astar_route(g, src, dst)
    cams = [n for n in cameras_on_route(g, naive) if n not in (src, dst)]
    if not cams:
        print("This start/destination pair passes no mid-route camera; "
              "nothing to demonstrate. Try another pair.")
        return

    crash_node = cams[len(cams) // 2]  # put the incident mid-route
    crash_name = g.nodes[crash_node].get("cam_name", "unknown")
    naive_m = route_metrics(g, naive)

    print("=" * 68)
    print("Phase 3 demo: routing around a vision-confirmed incident")
    print("=" * 68)
    print(f"\nEmergency from {src} to {dst}.")
    print(f"Initial A* route: {naive_m['hops']} blocks, {naive_m['length_km']:.2f} km, "
          f"{naive_m['travel_time_s'] / 60:.1f} min in current traffic.")
    print(f"It passes {len(cams)} camera(s). Staging an accident mid-route at:")
    print(f"  >> {crash_name}")
    print("   (its congestion score is forced LOW, so only the incident can "
          "trigger a reroute.)")

    # Seed the incident and run the incident-aware gate in live mode.
    real_score_frame = vision_gate._live_score_and_frame
    real_incident = incident_detect.check_frame_for_incident
    fake_score_frame, fake_incident = _seed_incident(crash_name)
    try:
        vision_gate._live_score_and_frame = fake_score_frame
        incident_detect.check_frame_for_incident = fake_incident
        route, info = plan_confirmed_route(
            g, src, dst, planner=astar_route, mode="live", check_incidents=True)
    finally:
        vision_gate._live_score_and_frame = real_score_frame
        incident_detect.check_frame_for_incident = real_incident

    route_m = route_metrics(g, route)
    incident_blocks = [c for c in info["blocked_cameras"] if c.get("reason") == "incident"]

    print(f"\nVision gate: confirmed={info['confirmed']} after {info['attempts']} "
          f"attempt(s), {len(incident_blocks)} incident block(s).")
    for c in incident_blocks:
        inc = c["incident"]
        print(f"  AVOIDED {c['camera']}: {inc['category'].replace('_', ' ')} "
              f"at {inc['confidence']:.0%} confidence (congestion score only "
              f"{c['score']:.0f}).")

    print(f"\nConfirmed route: {route_m['hops']} blocks, {route_m['length_km']:.2f} km, "
          f"{route_m['travel_time_s'] / 60:.1f} min.")
    print(f"Incident intersection on the initial route? {crash_node in naive}")
    print(f"Incident intersection on the confirmed route? {crash_node in route}")

    print("\n--- Dispatcher explanation ---")
    print(explain_route(g, route, baseline=naive, gate_info=info,
                        strategy="A* + vision (incident-aware)"))

    # Make the outcome unambiguous for anyone reading the output.
    ok = (info["confirmed"] and crash_node in naive and crash_node not in route
          and incident_blocks)
    print("\n" + ("DEMO OK: the router avoided the confirmed incident."
                  if ok else "DEMO INCONCLUSIVE: see output above."))


if __name__ == "__main__":
    main()
