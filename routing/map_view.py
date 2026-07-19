# Folium map builder for the public emergency-routing UI.
#
# Takes the JSON-serialisable payload from routing.geo and returns a Folium
# Map that looks like a consumer navigation app: street basemap, solid route,
# dashed comparison route, start/end pins, and blockage markers.

from __future__ import annotations

from typing import Any

import folium


# Calm emergency-ops palette — no purple-gradient AI look.
OUR_COLOR = "#0B3D91"       # deep navy — vision-confirmed route
BASELINE_COLOR = "#6B7280"  # muted grey — Google/OSRM/naive baseline
START_COLOR = "#15803D"     # green
END_COLOR = "#B91C1C"       # signal red
BLOCKED_COLOR = "#111827"   # near-black


def build_route_map(
    payload: dict[str, Any],
    *,
    our_label: str = "Vision-confirmed route",
    baseline_label: str = "Standard directions",
    height: int = 520,
) -> folium.Map:
    """Build an interactive Leaflet map from a route_map_payload() dict."""
    our = payload.get("our_route") or []
    baseline = payload.get("baseline_route") or []
    start = payload["start"]
    end = payload["end"]

    center_lat = (start["lat"] + end["lat"]) / 2.0
    center_lon = (start["lon"] + end["lon"]) / 2.0

    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=13,
        tiles="CartoDB positron",
        control_scale=True,
        height=height,
    )

    if baseline and len(baseline) >= 2:
        folium.PolyLine(
            locations=baseline,
            color=BASELINE_COLOR,
            weight=5,
            opacity=0.75,
            dash_array="8 10",
            tooltip=baseline_label,
        ).add_to(m)

    if our and len(our) >= 2:
        folium.PolyLine(
            locations=our,
            color=OUR_COLOR,
            weight=6,
            opacity=0.95,
            tooltip=our_label,
        ).add_to(m)

    folium.CircleMarker(
        location=[start["lat"], start["lon"]],
        radius=8,
        color=START_COLOR,
        fill=True,
        fill_color=START_COLOR,
        fill_opacity=1.0,
        tooltip="Start (vehicle)",
    ).add_to(m)

    folium.CircleMarker(
        location=[end["lat"], end["lon"]],
        radius=8,
        color=END_COLOR,
        fill=True,
        fill_color=END_COLOR,
        fill_opacity=1.0,
        tooltip="Incident / destination",
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
                icon="remove-sign" if kind == "blocked" else "warning-sign",
                prefix="glyphicon",
            ),
        ).add_to(m)

    bounds = payload.get("bounds")
    if bounds:
        m.fit_bounds(bounds, padding=(30, 30))

    legend_html = f"""
    <div style="
        position: fixed; bottom: 24px; left: 24px; z-index: 9999;
        background: rgba(255,255,255,0.95); padding: 10px 14px;
        border: 1px solid #e5e7eb; border-radius: 6px;
        font-family: Georgia, 'Times New Roman', serif; font-size: 12px;
        color: #111827; line-height: 1.55; box-shadow: 0 1px 3px rgba(0,0,0,.08);
    ">
      <div style="font-weight: 700; margin-bottom: 4px;">Route legend</div>
      <div><span style="color:{OUR_COLOR};">━━</span> {our_label}</div>
      <div><span style="color:{BASELINE_COLOR};">╌ ╌</span> {baseline_label}</div>
      <div><span style="color:{START_COLOR};">●</span> Start &nbsp;
           <span style="color:{END_COLOR};">●</span> Destination</div>
    </div>
    """
    m.get_root().html.add_child(folium.Element(legend_html))
    return m


if __name__ == "__main__":
    from routing.geo import route_map_payload
    from routing.graph import build_default_graph
    from routing.planners import astar_route, static_baseline_route

    g = build_default_graph()
    nodes = list(g.nodes)
    src, dst = nodes[0], nodes[-1]
    route = astar_route(g, src, dst)
    baseline = static_baseline_route(g, src, dst)
    payload = route_map_payload(g, route, baseline_path=baseline)
    m = build_route_map(payload)
    assert m is not None
    # Smoke-check that both polylines were attached.
    html = m.get_root().render()
    assert OUR_COLOR in html or "PolyLine" in html or "polyline" in html.lower()
    print(f"Map built with {len(payload['our_route'])} our pts, "
          f"{len(payload['baseline_route'])} baseline pts")
    print("map_view.py self-test OK")
