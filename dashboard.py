# Emergency Routing for Smart Response — public Streamlit app.
# Launch with: streamlit run dashboard.py
#
# Three views:
#   Route planner - type any two Manhattan places (like Google Maps), watch the
#                   vision-confirmed route drive and recalculate around blockages
#   Live cameras  - the YOLO congestion picture across all NYC DOT cameras
#   Ask the city  - LLM chat grounded in the current congestion data

import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import requests
import streamlit as st
from streamlit_folium import st_folium
from streamlit_searchbox import st_searchbox
from ultralytics import YOLO

from routing.graph import build_default_graph, nearest_node
from routing.planners import astar_route, dijkstra_route, route_metrics, static_baseline_route
from routing.rl_agent import rl_route
from routing.navigator import drive_route
from routing.explain import explain_route
from routing.geo import path_to_latlon, route_map_payload
from routing.geocode import geocode_manhattan, suggest_places
from routing.map_view import build_cameras_map, build_route_map
from routing.external_route import external_crosses_blockages, fetch_external_route
from traffic_llm import build_traffic_context, chat_completion

st.set_page_config(
    page_title="Emergency Routing — Smart Response",
    page_icon="◉",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Mission-control visual system ─────────────────────────────────────────────
st.markdown(
    """
    <style>
      @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;600;700&family=Inter:wght@400;500;600&display=swap');

      :root {
        --bg: #0B1220;
        --panel: rgba(148, 163, 184, 0.07);
        --line: rgba(148, 163, 184, 0.18);
        --ink: #E2E8F0;
        --muted: #94A3B8;
        --cyan: #22D3EE;
        --red: #F87171;
        --green: #34D399;
        --amber: #FBBF24;
      }

      html, body, [class*="css"] { font-family: 'Inter', 'Segoe UI', sans-serif; }

      .stApp {
        background:
          radial-gradient(900px 500px at 85% -10%, rgba(34, 211, 238, 0.10) 0%, transparent 55%),
          radial-gradient(700px 420px at 0% 0%, rgba(248, 113, 113, 0.07) 0%, transparent 50%),
          linear-gradient(180deg, #0B1220 0%, #0D1526 60%, #0B1220 100%);
        color: var(--ink);
      }

      h1, h2, h3, h4 { font-family: 'Space Grotesk', sans-serif !important; letter-spacing: -0.01em; }

      /* Hero */
      .hero { padding: 0.6rem 0 0.4rem 0; }
      .hero-eyebrow {
        font-size: 0.78rem; font-weight: 600; letter-spacing: 0.22em;
        color: var(--cyan); text-transform: uppercase; margin: 0 0 0.35rem 0;
      }
      .hero-title {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 2.6rem; font-weight: 700; line-height: 1.08;
        color: #F8FAFC; margin: 0;
      }
      .hero-title em { color: var(--cyan); font-style: normal; }
      .hero-sub {
        margin: 0.55rem 0 0 0; max-width: 46rem;
        color: var(--muted); font-size: 1.02rem; line-height: 1.55;
      }

      /* Live status chips */
      .chips { display: flex; gap: 0.6rem; flex-wrap: wrap; margin: 1.05rem 0 0.4rem 0; }
      .chip {
        background: var(--panel); border: 1px solid var(--line);
        border-radius: 999px; padding: 0.38rem 0.9rem;
        font-size: 0.85rem; color: var(--ink);
        backdrop-filter: blur(4px);
      }
      .chip b { font-family: 'Space Grotesk', sans-serif; }
      .chip .dot { display: inline-block; width: 8px; height: 8px; border-radius: 50%;
                   margin-right: 6px; vertical-align: 1px; }
      .pulse { animation: pulse 2.2s infinite; }
      @keyframes pulse { 0%,100% { opacity: 1; } 50% { opacity: 0.45; } }

      /* Tabs as pill nav */
      div[data-testid="stTabs"] div[role="tablist"] {
        gap: 0.4rem; border-bottom: none; margin-top: 0.6rem;
      }
      div[data-testid="stTabs"] button[role="tab"] {
        background: var(--panel); border: 1px solid var(--line);
        border-radius: 999px; padding: 0.35rem 1.1rem;
        color: var(--muted); font-weight: 600;
      }
      div[data-testid="stTabs"] button[role="tab"][aria-selected="true"] {
        color: #06222B; background: var(--cyan); border-color: var(--cyan);
      }
      div[data-testid="stTabs"] button[role="tab"][aria-selected="true"] p { color: #06222B; }

      /* Cards & metrics */
      div[data-testid="stMetric"] {
        background: var(--panel); border: 1px solid var(--line);
        border-radius: 14px; padding: 0.9rem 1.05rem;
        backdrop-filter: blur(4px);
      }
      div[data-testid="stMetric"] label { color: var(--muted) !important; }
      div[data-testid="stMetric"] [data-testid="stMetricValue"] {
        color: #F8FAFC !important; font-family: 'Space Grotesk', sans-serif;
      }

      .section-note { color: var(--muted); font-size: 0.93rem; margin: -0.3rem 0 0.9rem 0; }

      /* Inputs */
      div[data-baseweb="input"] > div, div[data-baseweb="select"] > div {
        background: rgba(15, 23, 42, 0.75) !important;
        border-color: var(--line) !important;
      }

      .stButton > button[kind="primary"] {
        background: linear-gradient(90deg, #EF4444 0%, #F87171 100%);
        border: none; font-weight: 700; letter-spacing: 0.02em;
        box-shadow: 0 4px 20px rgba(239, 68, 68, 0.35);
      }
    </style>
    """,
    unsafe_allow_html=True,
)


# ── Data loaders ──────────────────────────────────────────────────────────────

@st.cache_data
def load_cameras():
    return pd.read_csv("manhattan_cameras.csv")


@st.cache_data
def load_stats():
    path = Path("camera_stats.csv")
    if not path.exists():
        return None, None
    updated = datetime.fromtimestamp(path.stat().st_mtime)
    return pd.read_csv(path), updated


@st.cache_resource
def get_model():
    return YOLO("weights/yolov12s.pt")


@st.cache_data(ttl=24 * 3600, show_spinner=False)
def geocode_cached(query: str) -> dict:
    """Cache place lookups so repeat searches never re-hit Nominatim."""
    return geocode_manhattan(query)


@st.cache_resource
def get_route_graph():
    """Real OSM street network (cached GraphML) with camera congestion.
    Falls back to the synthetic grid only if the cache is missing."""
    try:
        from routing.streets import build_street_graph
        return build_street_graph(), True
    except FileNotFoundError:
        return build_default_graph(), False


def fetch_frame(url: str):
    try:
        r = requests.get(url, timeout=8)
        r.raise_for_status()
        arr = np.frombuffer(r.content, np.uint8)
        frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if frame is None:
            raise ValueError("response is not a decodable image")
        return frame
    except Exception as e:
        st.warning(f"Failed to fetch frame: {e}")
        return None


def run_yolo_on_camera(cam_row: pd.Series):
    model = get_model()
    frame = fetch_frame(cam_row["image_url"])
    if frame is None:
        return None, None
    res = model(frame, imgsz=640)[0]
    return res.plot(), res


cams_df = load_cameras()
stats_df, stats_updated = load_stats()

if stats_df is not None:
    merged = cams_df.merge(
        stats_df[["camera_id", "score", "level", "vehicles", "pedestrians", "signals"]],
        on="camera_id",
        how="left",
    )
else:
    merged = cams_df.copy()
    for col in ("score", "vehicles", "pedestrians", "signals"):
        merged[col] = np.nan
    merged["level"] = None


# ── Hero + live status chips ──────────────────────────────────────────────────

n_high = int((merged["level"] == "high").sum())
n_scored = int(merged["score"].notna().sum())
freshness = stats_updated.strftime("%b %d, %H:%M") if stats_updated else "n/a"

st.markdown(
    f"""
    <div class="hero">
      <p class="hero-eyebrow">Vision-confirmed dispatch · Manhattan</p>
      <p class="hero-title">Routes that <em>see</em> the road ahead.</p>
      <p class="hero-sub">
        Type any two places in Manhattan. We plan on real streets, check every
        traffic camera along the way with YOLOv12, and recalculate mid-drive the
        moment one shows a blockage — then show you what standard map directions
        would have done.
      </p>
      <div class="chips">
        <span class="chip"><span class="dot pulse" style="background:var(--green);"></span>
          <b>{len(cams_df)}</b>&nbsp;DOT cameras</span>
        <span class="chip"><span class="dot" style="background:var(--cyan);"></span>
          <b>{n_scored}</b>&nbsp;scored by YOLO</span>
        <span class="chip"><span class="dot" style="background:var(--red);"></span>
          <b>{n_high}</b>&nbsp;heavy congestion</span>
        <span class="chip"><span class="dot" style="background:var(--amber);"></span>
          updated&nbsp;<b>{freshness}</b></span>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

tab_route, tab_cams, tab_ask = st.tabs(["Route planner", "Live cameras", "Ask the city"])


# ── Route planner ─────────────────────────────────────────────────────────────
with tab_route:
    route_g, on_real_streets = get_route_graph()
    if not on_real_streets:
        st.warning(
            "Street network cache missing — run `python -m routing.streets --build` "
            "for real Manhattan streets."
        )

    def _search_manhattan(term: str) -> list:
        """Type-ahead: instant camera-name matches + bounded Nominatim hits."""
        return [
            (s["label"], (s["label"], s["lat"], s["lon"]))
            for s in suggest_places(term)
        ]

    SEARCHBOX_STYLE = {
        "searchbox": {
            "searchField": {
                "backgroundColor": "rgba(15, 23, 42, 0.75)",
                "color": "#E2E8F0",
                "border": "1px solid rgba(148, 163, 184, 0.18)",
                "borderRadius": "8px",
            },
            "menuList": {"backgroundColor": "#131C2E"},
            "option": {
                "color": "#E2E8F0",
                "backgroundColor": "#131C2E",
                "highlightColor": "rgba(34, 211, 238, 0.25)",
            },
        },
    }

    col_from, col_to = st.columns(2)
    with col_from:
        start_pick = st_searchbox(
            _search_manhattan,
            key="start_search",
            label="From",
            placeholder="Type a place… e.g. Columbus Circle",
            default_searchterm="Times Square",
            default_use_searchterm=True,
            debounce=500,
            style_overrides=SEARCHBOX_STYLE,
        )
    with col_to:
        dest_pick = st_searchbox(
            _search_manhattan,
            key="dest_search",
            label="To (the incident)",
            placeholder="Any address or landmark in Manhattan",
            default_searchterm="Wall Street",
            default_use_searchterm=True,
            debounce=500,
            style_overrides=SEARCHBOX_STYLE,
        )

    with st.expander("Routing options", expanded=False):
        vision_mode = st.selectbox(
            "Vision check",
            ["Live (fresh YOLO per camera)", "Offline (stored camera scores)"],
            help=(
                "Live re-downloads a snapshot from every camera ahead and "
                "runs YOLOv12 on it right now; offline trusts the scores "
                "from the last city-wide scan."
            ),
        )
        mode = "live" if vision_mode.startswith("Live") else "offline"
        # Staged blockage only makes sense against stored scores — hide it
        # entirely in live mode so users aren't offered a dead option.
        if mode == "offline":
            simulate_incident = st.selectbox(
                "En-route demo",
                ["Off", "Simulate a blockage mid-route"],
                help=(
                    "Stages a severe blockage on a camera along the initial "
                    "route after departure, so you can watch the route "
                    "recalculate from the vehicle's position."
                ),
            )
        else:
            simulate_incident = "Off"

    plan_clicked = st.button("Find vision-confirmed route", type="primary")

    def _resolve_place(pick, fallback_query: str):
        """A searchbox returns (label, lat, lon) when a suggestion was picked,
        or the raw typed string otherwise - geocode the latter."""
        if isinstance(pick, tuple) and len(pick) == 3:
            return {"outcome": "ok", "label": pick[0], "lat": pick[1],
                    "lon": pick[2], "source": "suggestion"}
        return geocode_cached(str(pick or fallback_query))

    if not plan_clicked:
        # Landing visual: the live congestion picture, not an empty form.
        st_folium(
            build_cameras_map(merged),
            width=None, height=460, returned_objects=[],
            key="landing_map",
        )
    else:
        with st.spinner("Locating places..."):
            start_geo = _resolve_place(start_pick, "Times Square")
            dest_geo = _resolve_place(dest_pick, "Wall Street")

        start_q = start_geo.get("label", str(start_pick or "Times Square"))
        dest_q = dest_geo.get("label", str(dest_pick or "Wall Street"))
        geo_ok = True
        for label, geo, q in (("Start", start_geo, start_q), ("Destination", dest_geo, dest_q)):
            if geo["outcome"] == "outside":
                st.error(
                    f"**{label} is outside Manhattan.** \u201c{q}\u201d exists, but this "
                    "system covers Manhattan only — try a place on the island."
                )
                geo_ok = False
            elif geo["outcome"] == "not_found":
                st.error(
                    f"**Couldn't find \u201c{q}\u201d within Manhattan.** This system "
                    "covers Manhattan only — try a landmark, an address, or an "
                    "intersection like \u201cAmsterdam Ave @ 60 St\u201d."
                )
                geo_ok = False

        if geo_ok:
            start_lat, start_lon = start_geo["lat"], start_geo["lon"]
            end_lat, end_lon = dest_geo["lat"], dest_geo["lon"]
            st.caption(f"From **{start_geo['label']}** to **{dest_geo['label']}**")

            src = nearest_node(route_g, start_lat, start_lon)
            dst = nearest_node(route_g, end_lat, end_lon)

            if src == dst:
                st.error("Start and destination snap to the same intersection — pick places further apart.")
                st.stop()

            yolo_model = get_model() if mode == "live" else None

            # The navigation loop mutates camera scores in the incident demo,
            # so work on a copy and leave the shared cached graph untouched.
            nav_g = route_g.copy()
            nav_g.graph.update(route_g.graph)

            ALGORITHMS = [
                ("A*", astar_route),
                ("Dijkstra", dijkstra_route),
                ("Q-learning (RL)", rl_route),
            ]

            spinner_msg = (
                "Driving the route with all three algorithms and comparing "
                "against standard map directions..."
                if mode == "offline"
                else "Checking cameras live with YOLOv12 and driving the route "
                "with all three algorithms — live mode takes a little longer..."
            )
            with st.spinner(spinner_msg):
                if simulate_incident != "Off":
                    from routing.vision_gate import cameras_on_route

                    initial = astar_route(nav_g, src, dst)
                    cams_ahead = [
                        n for n in cameras_on_route(nav_g, initial) if n not in (src, dst)
                    ]
                    if cams_ahead:
                        staged = cams_ahead[len(cams_ahead) // 2]
                        nav_g.nodes[staged]["cam_score"] = 99.0

                # One shared cache: each camera is downloaded and YOLO-scored at
                # most once even though three algorithms drive the network.
                shared_scores: dict = {}
                candidates = []
                for algo_name, algo_fn in ALGORITHMS:
                    t0 = time.perf_counter()
                    try:
                        algo_trace = drive_route(
                            nav_g, src, dst, planner=algo_fn, mode=mode,
                            model=yolo_model, score_cache=shared_scores,
                        )
                    except Exception:
                        continue
                    compute_s = time.perf_counter() - t0
                    algo_route = (
                        algo_trace["driven"] if algo_trace["confirmed"]
                        else algo_trace["final_path"]
                    )
                    algo_m = route_metrics(nav_g, algo_route)
                    candidates.append({
                        "name": algo_name,
                        "trace": algo_trace,
                        "route": algo_route,
                        "metrics": algo_m,
                        "compute_s": compute_s,
                    })

                external = fetch_external_route(
                    nav_g, start_lat, start_lon, end_lat, end_lon,
                    src_node=src, dst_node=dst,
                )

            if not candidates:
                st.error("No algorithm could find a route between these points.")
                st.stop()

            # The winner: fastest arrival among algorithms that actually got
            # there; if none arrived, least-bad ETA.
            arrived = [c for c in candidates if c["trace"]["confirmed"]]
            best = min(
                arrived or candidates,
                key=lambda c: c["metrics"]["travel_time_s"],
            )
            strategy = best["name"]
            trace = best["trace"]
            route = best["route"]

            gate_info = {
                "confirmed": trace["confirmed"],
                "attempts": len(trace["segments"]),
                "blocked_cameras": [
                    {"node": r["blocked_node"], "camera": r["blocked_camera"], "score": r["score"]}
                    for r in trace["reroutes"]
                ],
                "endpoint_warnings": [],
            }

            m_route = route_metrics(nav_g, route)
            our_min = m_route["travel_time_s"] / 60.0
            ext_cong_min = external["congested_time_s"] / 60.0
            ext_provider_min = external["duration_s"] / 60.0
            saved_s = external["congested_time_s"] - m_route["travel_time_s"]

            blocked_nodes = {
                cam["node"] for cam in gate_info["blocked_cameras"] if "node" in cam
            }
            ext_hits = external_crosses_blockages(nav_g, external["coords"], blocked_nodes)

            col1, col2, col3, col4 = st.columns(4)
            col1.metric(f"Our ETA — {strategy} (best of 3)", f"{our_min:.1f} min")
            col2.metric(f"{external['provider']} ETA", f"{ext_provider_min:.1f} min")
            col3.metric("Their path in our traffic model", f"{ext_cong_min:.1f} min")
            col4.metric(
                "Advantage vs map path",
                f"{max(saved_s, 0) / 60:.1f} min",
                delta="faster under congestion" if saved_s > 0 else "similar",
            )

            if gate_info["confirmed"]:
                if trace["reroutes"]:
                    st.success(
                        f"**{strategy}** won out of {len(candidates)} algorithms, "
                        f"arriving after {len(trace['reroutes'])} en-route "
                        "recalculation(s) — the route redrew from the vehicle's "
                        "position when a camera ahead reported a blockage."
                    )
                else:
                    st.success(
                        f"**{strategy}** won out of {len(candidates)} algorithms; "
                        "every camera ahead was checked before entering and "
                        "stayed clear the whole drive."
                    )
            else:
                st.warning("Route could not be fully confirmed — dispatch with caution.")

            if external["source"] == "google_directions":
                st.caption("Comparison baseline: live Google Directions API.")
            elif external["source"] == "osrm":
                st.caption(
                    "Comparison baseline: OSRM public router (OpenStreetMap). "
                    "Set GOOGLE_MAPS_API_KEY to compare against Google Maps directly."
                )
            else:
                st.caption("Comparison baseline: offline naive free-flow path.")

            for r in trace["reroutes"]:
                st.warning(
                    f"Recalculated en route: camera {r['blocked_camera']} ahead "
                    f"reported score {r['score']:.0f} — rerouted from the vehicle's position."
                )
            if ext_hits:
                names = ", ".join(h["camera"] for h in ext_hits)
                st.error(
                    f"{external['provider']} still passes near blocked camera(s): "
                    f"{names}. That is the gap vision confirmation closes."
                )

            payload = route_map_payload(
                nav_g, route,
                baseline_path=None,
                start=(start_lat, start_lon), end=(end_lat, end_lon),
                gate_info=gate_info,
            )
            payload["baseline_route"] = external["coords"]

            abandoned = []
            for seg, reroute in zip(trace["segments"][:-1], trace["reroutes"]):
                leg = seg["path"]
                if reroute["at_node"] in leg:
                    unused = leg[leg.index(reroute["at_node"]):]
                    if len(unused) >= 2:
                        abandoned.append({
                            "coords": path_to_latlon(nav_g, unused),
                            "label": f"Abandoned: {reroute['blocked_camera']} blocked",
                        })
            payload["abandoned_routes"] = abandoned

            all_pts = payload["our_route"] + payload["baseline_route"]
            for leg in abandoned:
                all_pts = all_pts + leg["coords"]
            payload["bounds"] = [
                [min(p[0] for p in all_pts), min(p[1] for p in all_pts)],
                [max(p[0] for p in all_pts), max(p[1] for p in all_pts)],
            ]

            fmap = build_route_map(
                payload,
                our_label=f"Vision-confirmed ({strategy})",
                baseline_label=external["provider"],
            )
            st_folium(fmap, width=None, height=560, returned_objects=[], key="route_map")

            st.markdown(
                '<p class="section-note">Cyan = the path actually driven · dashed slate = '
                "standard map directions · dotted amber = legs abandoned when the route "
                "recalculated · black markers = blockages avoided.</p>",
                unsafe_allow_html=True,
            )

            left, right = st.columns(2)
            with left:
                st.markdown(f"#### Our route ({strategy})")
                st.write(
                    f"{m_route['length_km']:.2f} km · {m_route['hops']} street segments · "
                    f"worst congestion ×{m_route['worst_congestion']:.2f}"
                )
            with right:
                st.markdown(f"#### {external['provider']}")
                st.write(
                    f"{external['distance_km']:.2f} km · provider ETA {ext_provider_min:.1f} min · "
                    f"in our model {ext_cong_min:.1f} min"
                )

            st.markdown("### How the three algorithms compared")
            st.markdown(
                '<p class="section-note">Every plan runs A*, Dijkstra and '
                "corridor Q-learning through the same drive simulation with the "
                "same camera checks; the fastest arrival is what you see on the "
                "map above.</p>",
                unsafe_allow_html=True,
            )
            comp_df = pd.DataFrame([
                {
                    "Algorithm": c["name"] + (" — shown on map" if c is best else ""),
                    "ETA (min)": round(c["metrics"]["travel_time_s"] / 60.0, 1),
                    "Distance (km)": round(c["metrics"]["length_km"], 2),
                    "Street segments": c["metrics"]["hops"],
                    "Reroutes": len(c["trace"]["reroutes"]),
                    "Arrived": "yes" if c["trace"]["confirmed"] else "no",
                    "Compute (s)": round(c["compute_s"], 2),
                }
                for c in sorted(candidates, key=lambda c: c["metrics"]["travel_time_s"])
            ])
            st.dataframe(comp_df, hide_index=True, width="stretch")

            graph_baseline = (
                external.get("graph_path") or static_baseline_route(nav_g, src, dst)
            )
            with st.spinner("Writing explanation..."):
                explanation = explain_route(
                    nav_g, route, graph_baseline, gate_info,
                    strategy=f"{strategy} + vision vs {external['provider']}",
                )
            st.markdown("### Why this route")
            st.write(explanation)


# ── Live cameras ──────────────────────────────────────────────────────────────
with tab_cams:
    if stats_df is None:
        st.warning("No camera_stats.csv found. Run update_camera_stats.py first.")
    else:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Cameras scored", n_scored)
        c2.metric("High congestion", n_high)
        c3.metric("Medium congestion", int((merged["level"] == "medium").sum()))
        c4.metric("Average score", f"{merged['score'].mean():.1f}")

    st.markdown(
        '<p class="section-note">Click any camera dot on the map — its live '
        "YOLOv12 view starts below automatically. That is the same inference "
        "the router uses to confirm routes.</p>",
        unsafe_allow_html=True,
    )

    map_state = st_folium(
        build_cameras_map(merged),
        width=None, height=440,
        returned_objects=["last_object_clicked", "last_object_clicked_tooltip"],
        key="cams_map",
    )

    cam_pick = cams_df.dropna(subset=["lat", "lon"]).reset_index(drop=True)

    # Resolve the camera: a map-dot click wins, the dropdown is the fallback.
    clicked_cam = None
    tip = (map_state or {}).get("last_object_clicked_tooltip") or ""
    if isinstance(tip, str) and tip.startswith("cam:"):
        cam_id = tip.split("|", 1)[0].removeprefix("cam:")
        hit = cam_pick[cam_pick["camera_id"].astype(str) == cam_id]
        if not hit.empty:
            clicked_cam = hit.iloc[0]
    elif (map_state or {}).get("last_object_clicked"):
        # Fallback for older folium builds that only return lat/lng.
        clicked = map_state["last_object_clicked"]
        if "lat" in clicked and "lng" in clicked:
            d2 = (cam_pick["lat"].astype(float) - float(clicked["lat"])) ** 2 + (
                cam_pick["lon"].astype(float) - float(clicked["lng"])
            ) ** 2
            if float(d2.min()) < (0.0004) ** 2:
                clicked_cam = cam_pick.loc[d2.idxmin()]

    if clicked_cam is not None:
        # Keep the dropdown in sync with the clicked camera.
        default_idx = int(
            cam_pick.index[cam_pick["camera_id"] == clicked_cam["camera_id"]][0]
        )
    else:
        default_idx = 0

    cam_choice = st.selectbox(
        "…or pick a camera from the list",
        options=cam_pick["camera_id"].tolist(),
        index=min(default_idx, len(cam_pick) - 1),
        format_func=lambda cid: cam_pick.loc[cam_pick["camera_id"] == cid, "name"].iloc[0],
        key="cam_list_pick",
    )

    if clicked_cam is not None:
        cam_row = clicked_cam
    else:
        cam_row = cam_pick[cam_pick["camera_id"] == cam_choice].iloc[0]

    # Auto-start on a fresh map click; the button covers dropdown picks.
    click_token = str(clicked_cam["camera_id"]) if clicked_cam is not None else None
    fresh_click = (
        click_token is not None
        and st.session_state.get("last_live_cam") != click_token
    )
    start_live = fresh_click or st.button("Start live view (15 frames, every 2s)")

    if start_live:
        if click_token is not None:
            st.session_state["last_live_cam"] = click_token
        st.markdown(f"#### Live — {cam_row['name']}")
        placeholder = st.empty()
        failed = False
        for i in range(15):
            annotated, _ = run_yolo_on_camera(cam_row)
            if annotated is None:
                st.error("Could not fetch a frame from this camera. It may be offline.")
                failed = True
                break
            placeholder.image(
                annotated, channels="BGR",
                caption=f"{cam_row['name']} — frame {i + 1}/15",
            )
            time.sleep(2)
        if not failed:
            st.info("Live view finished — click the camera again (or the button) to restart.")

    if stats_df is not None:
        with st.expander("Full congestion table"):
            st.dataframe(
                merged.dropna(subset=["lat", "lon"])[
                    ["name", "score", "level", "vehicles", "pedestrians", "signals"]
                ]
                .sort_values("score", ascending=False)
                .reset_index(drop=True)
            )


# ── Ask the city (LLM) ────────────────────────────────────────────────────────
with tab_ask:
    st.markdown(
        '<p class="section-note">Chat with an LLM grounded strictly in the '
        "current camera congestion data. Configure OPENAI_API_KEY (or "
        "LLM_PROVIDER=anthropic with ANTHROPIC_API_KEY) to enable it.</p>",
        unsafe_allow_html=True,
    )

    context_text = build_traffic_context(merged, None)
    with st.expander("Show text sent to the model"):
        st.text(context_text[:12000] + ("..." if len(context_text) > 12000 else ""))

    if "llm_messages" not in st.session_state:
        st.session_state.llm_messages = []

    for msg in st.session_state.llm_messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    user_q = st.chat_input('e.g. "How is traffic in Midtown right now?"')
    if user_q:
        st.session_state.llm_messages.append({"role": "user", "content": user_q})
        with st.chat_message("user"):
            st.markdown(user_q)

        system = (
            "You answer questions about Manhattan traffic using only the data below. "
            "If there are no scores, say so and mention update_camera_stats.py. "
            "Use the Midtown / Upper / Lower labels as given. Do not guess beyond the table. "
            "Keep it short.\n\n"
            f"Data:\n{context_text}"
        )
        api_messages = [{"role": "system", "content": system}] + [
            {"role": m["role"], "content": m["content"]}
            for m in st.session_state.llm_messages
        ]

        with st.chat_message("assistant"):
            with st.spinner("Calling model..."):
                reply, err = chat_completion(api_messages)
            if err:
                st.error(err)
                st.session_state.llm_messages.append(
                    {"role": "assistant", "content": f"(Model error: {err})"}
                )
            else:
                st.markdown(reply or "_Empty response_")
                st.session_state.llm_messages.append(
                    {"role": "assistant", "content": reply or ""}
                )

    if st.button("Clear chat history", key="clear_llm_chat"):
        st.session_state.llm_messages = []
        st.rerun()
