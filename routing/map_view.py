# Folium map builders for the public emergency-routing UI.
#
# Dark "mission control" cartography: CartoDB dark tiles, neon-cyan route,
# glowing congestion dots.  Two builders:
#   build_route_map   - dual-route comparison after planning
#   build_cameras_map - live congestion overview (the landing visual)

from __future__ import annotations

from typing import Any

import folium

TILES = "CartoDB dark_matter"

OUR_COLOR = "#22D3EE"       # neon cyan — the path actually driven
BASELINE_COLOR = "#94A3B8"  # slate — Google/OSRM/naive comparison
ABANDONED_COLOR = "#F59E0B" # amber — legs abandoned after a mid-drive reroute
START_COLOR = "#34D399"     # green
END_COLOR = "#F87171"       # signal red
BLOCKED_COLOR = "#0F172A"   # near-black marker for blockages

LEVEL_COLORS = {
    "low": "#34D399",
    "medium": "#FBBF24",
    "high": "#F87171",
}


def _dark_panel(html: str) -> str:
    """Shared style for on-map overlay panels."""
    return f"""
    <div style="
        position: fixed; bottom: 24px; left: 24px; z-index: 9999;
        background: rgba(10, 15, 30, 0.88); padding: 10px 14px;
        border: 1px solid rgba(148, 163, 184, 0.25); border-radius: 10px;
        font-family: 'Inter', 'Segoe UI', sans-serif; font-size: 12px;
        color: #E2E8F0; line-height: 1.6;
        box-shadow: 0 4px 18px rgba(0, 0, 0, 0.45);
        backdrop-filter: blur(6px);
    ">{html}</div>
    """


def build_route_map(
    payload: dict[str, Any],
    *,
    our_label: str = "Vision-confirmed route",
    baseline_label: str = "Standard directions",
    height: int = 520,
) -> folium.Map:
    """Interactive dual-route map from a route_map_payload() dict."""
    our = payload.get("our_route") or []
    baseline = payload.get("baseline_route") or []
    start = payload["start"]
    end = payload["end"]

    m = folium.Map(
        location=[(start["lat"] + end["lat"]) / 2, (start["lon"] + end["lon"]) / 2],
        zoom_start=13,
        tiles=TILES,
        control_scale=True,
        height=height,
    )

    if baseline and len(baseline) >= 2:
        folium.PolyLine(
            locations=baseline,
            color=BASELINE_COLOR,
            weight=5,
            opacity=0.7,
            dash_array="8 10",
            tooltip=baseline_label,
        ).add_to(m)

    # Abandoned legs from mid-drive reroutes (Google-style recalculation).
    for leg in payload.get("abandoned_routes", []):
        coords = leg.get("coords") or []
        if len(coords) >= 2:
            folium.PolyLine(
                locations=coords,
                color=ABANDONED_COLOR,
                weight=4,
                opacity=0.6,
                dash_array="2 8",
                tooltip=leg.get("label", "Abandoned after reroute"),
            ).add_to(m)

    if our and len(our) >= 2:
        # Soft glow underlay, then the crisp route line on top.
        folium.PolyLine(locations=our, color=OUR_COLOR, weight=10, opacity=0.25).add_to(m)
        folium.PolyLine(
            locations=our, color=OUR_COLOR, weight=4.5, opacity=1.0, tooltip=our_label
        ).add_to(m)

    for spot, color, label in (
        (start, START_COLOR, "Start (vehicle)"),
        (end, END_COLOR, "Incident / destination"),
    ):
        folium.CircleMarker(
            location=[spot["lat"], spot["lon"]],
            radius=9, color="#0B1220", weight=2,
            fill=True, fill_color=color, fill_opacity=1.0,
            tooltip=label,
        ).add_to(m)

    for cam in payload.get("cameras", []):
        kind = cam.get("kind", "blocked")
        label = (
            f"Blocked: {cam['name']} (score {cam['score']:.0f})"
            if kind == "blocked"
            else f"Congested endpoint: {cam['name']} (score {cam['score']:.0f})"
        )
        folium.Marker(
            location=[cam["lat"], cam["lon"]],
            tooltip=label,
            icon=folium.Icon(
                color="black" if kind == "blocked" else "orange",
                icon="times" if kind == "blocked" else "exclamation-triangle",
                prefix="fa",
            ),
        ).add_to(m)

    bounds = payload.get("bounds")
    if bounds:
        m.fit_bounds(bounds, padding=(30, 30))

    legend = _dark_panel(f"""
      <div style="font-weight:700; margin-bottom:4px; letter-spacing:.04em;">ROUTE LEGEND</div>
      <div><span style="color:{OUR_COLOR};">━━</span> {our_label}</div>
      <div><span style="color:{BASELINE_COLOR};">╌ ╌</span> {baseline_label}</div>
      <div><span style="color:{ABANDONED_COLOR};">· ·</span> Abandoned after reroute</div>
      <div><span style="color:{START_COLOR};">●</span> Start &nbsp;
           <span style="color:{END_COLOR};">●</span> Destination</div>
    """)
    m.get_root().html.add_child(folium.Element(legend))
    return m


def build_cameras_map(cams_df, height: int = 460) -> folium.Map:
    """Landing visual: every DOT camera as a glowing dot colored by congestion.

    cams_df needs lat / lon / name and optionally score / level.
    """
    df = cams_df.dropna(subset=["lat", "lon"])

    m = folium.Map(
        location=[40.7731, -73.9712],  # roughly central Manhattan
        zoom_start=12,
        tiles=TILES,
        height=height,
    )

    counts = {"low": 0, "medium": 0, "high": 0}
    for _, cam in df.iterrows():
        level = str(cam.get("level") or "").lower()
        color = LEVEL_COLORS.get(level, "#64748B")
        if level in counts:
            counts[level] += 1
        score = cam.get("score")
        score_txt = f" · score {score:.0f}" if score == score and score is not None else ""
        folium.CircleMarker(
            location=[float(cam["lat"]), float(cam["lon"])],
            radius=4.5,
            color=color,
            weight=1,
            fill=True,
            fill_color=color,
            fill_opacity=0.85,
            tooltip=f"{cam['name']}{score_txt}",
        ).add_to(m)

    legend = _dark_panel(f"""
      <div style="font-weight:700; margin-bottom:4px; letter-spacing:.04em;">LIVE CONGESTION</div>
      <div><span style="color:{LEVEL_COLORS['low']};">●</span> Low ({counts['low']}) &nbsp;
           <span style="color:{LEVEL_COLORS['medium']};">●</span> Medium ({counts['medium']}) &nbsp;
           <span style="color:{LEVEL_COLORS['high']};">●</span> High ({counts['high']})</div>
      <div style="color:#94A3B8;">{len(df)} NYC DOT cameras scored by YOLOv12</div>
    """)
    m.get_root().html.add_child(folium.Element(legend))
    return m


if __name__ == "__main__":
    import pandas as pd

    from routing.geo import route_map_payload
    from routing.graph import build_default_graph
    from routing.planners import astar_route, static_baseline_route

    g = build_default_graph()
    nodes = list(g.nodes)
    src, dst = nodes[0], nodes[-1]
    payload = route_map_payload(
        g, astar_route(g, src, dst), baseline_path=static_baseline_route(g, src, dst)
    )
    m = build_route_map(payload)
    html = m.get_root().render()
    assert OUR_COLOR in html
    print(f"Route map: {len(payload['our_route'])} pts")

    cams = pd.read_csv("manhattan_cameras.csv")
    try:
        stats = pd.read_csv("camera_stats.csv")
        cams = cams.merge(stats[["camera_id", "score", "level"]], on="camera_id", how="left")
    except FileNotFoundError:
        pass
    cm = build_cameras_map(cams)
    assert cm is not None
    print(f"Cameras map: {len(cams)} cameras")
    print("map_view.py self-test OK")
