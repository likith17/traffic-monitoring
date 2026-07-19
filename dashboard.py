# Streamlit dashboard for the Manhattan traffic congestion project.
# Launch with: streamlit run dashboard.py
# Prerequisites: run fetch_cameras.py once, then update_camera_stats.py for live scores.

import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import requests
import streamlit as st
from ultralytics import YOLO

from routing.graph import build_default_graph, nearest_node
from routing.planners import astar_route, dijkstra_route, route_metrics, static_baseline_route
from routing.rl_agent import QLearningRouter
from routing.vision_gate import plan_confirmed_route
from routing.explain import explain_route
from traffic_llm import build_traffic_context, chat_completion

st.set_page_config(page_title="Manhattan Traffic - YOLOv12", layout="wide")


# --- Data loaders (cached so they don't re-read CSV on every Streamlit interaction) ---

@st.cache_data
def load_cameras():
    """Load the base camera list written by fetch_cameras.py."""
    return pd.read_csv("manhattan_cameras.csv")


@st.cache_data
def load_stats():
    """Load the per-camera congestion scores written by update_camera_stats.py.
    Returns None if the file doesn't exist yet (scores haven't been computed).
    """
    path = Path("camera_stats.csv")
    if not path.exists():
        return None
    return pd.read_csv(path)


@st.cache_data
def load_segment_stats():
    """Load the offline segment analysis written by compute_stats.py.
    Returns None if the file doesn't exist yet.
    """
    path = Path("segment_stats.csv")
    if not path.exists():
        return None
    return pd.read_csv(path)


@st.cache_resource
def get_model():
    """Load the YOLOv12 weights once and keep them in memory across re-renders.
    st.cache_resource is correct here because the model object is not serialisable
    (as required by st.cache_data), but it's safe to share between sessions.
    """
    return YOLO("weights/yolov12s.pt")


def fetch_frame(url: str):
    """Download a camera snapshot and decode it into a BGR NumPy array.

    Shows a Streamlit warning and returns None on any failure, including cases
    where the server responds 200 but the body is not a valid image.
    """
    try:
        r = requests.get(url, timeout=8)
        r.raise_for_status()
        arr = np.frombuffer(r.content, np.uint8)
        frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if frame is None:
            # imdecode silently returns None for corrupt or non-image content.
            raise ValueError("cv2.imdecode returned None — response may not be an image")
        return frame
    except Exception as e:
        st.warning(f"Failed to fetch frame: {e}")
        return None


def run_yolo_on_camera(cam_row: pd.Series):
    """Fetch a live snapshot for the given camera and run YOLO on it.

    Returns (annotated_image, result) on success, or (None, None) if the frame
    couldn't be fetched or inference failed.
    """
    model = get_model()
    frame = fetch_frame(cam_row["image_url"])
    if frame is None:
        return None, None

    res = model(frame, imgsz=640)[0]
    annotated = res.plot()  # draws bounding boxes and labels onto the frame
    return annotated, res


# --- App-level data: loaded once at startup and shared across all tabs ---

cams_df = load_cameras()
stats_df = load_stats()

if stats_df is not None:
    # Join the pre-computed scores onto the camera list so every tab can use merged.
    merged = cams_df.merge(
        stats_df[
            [
                "camera_id",
                "score",
                "level",
                "vehicles",
                "pedestrians",
                "signals",
            ]
        ],
        on="camera_id",
        how="left",  # keep cameras that have no score yet (will show NaN)
    )
else:
    # No scores yet — create the score columns as NaN so downstream code
    # doesn't have to branch on whether the columns exist.
    merged = cams_df.copy()
    merged["score"] = np.nan
    merged["level"] = None
    merged["vehicles"] = np.nan
    merged["pedestrians"] = np.nan
    merged["signals"] = np.nan

st.title("Manhattan traffic (YOLOv12)")

tab1, tab2, tab3, tab4, tab5 = st.tabs(
    [
        "Overview",
        "Camera explorer",
        "Segments",
        "Ask traffic (LLM)",
        "Emergency routing",
    ]
)

# ── Tab 1: city-wide overview with a map and sortable table ───────────────────
with tab1:
    st.subheader("Manhattan cameras")

    if stats_df is None:
        st.warning("No camera_stats.csv found. Run update_camera_stats.py first.")
    else:
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Cameras with Data", len(stats_df))
        col2.metric("High Congestion", (stats_df["level"] == "high").sum())
        col3.metric("Medium Congestion", (stats_df["level"] == "medium").sum())

        st.markdown("### Map of Cameras")

        # Make a copy so we don't mutate the shared merged dataframe.
        stats_map = merged.copy()
        stats_map["lat"] = pd.to_numeric(stats_map["lat"], errors="coerce")
        stats_map["lon"] = pd.to_numeric(stats_map["lon"], errors="coerce")
        # Drop rows with missing coordinates — st.map raises an error on NaN lat/lon.
        stats_map = stats_map.dropna(subset=["lat", "lon"])

        if stats_map.empty:
            st.warning("No valid lat/lon data available to plot on the map.")
        else:
            st.map(stats_map[["lat", "lon"]])

        st.markdown("### Camera Congestion Table")
        st.dataframe(
            stats_map[
                [
                    "camera_id",
                    "name",
                    "area",
                    "score",
                    "level",
                    "vehicles",
                    "pedestrians",
                    "signals",
                ]
            ]
            .sort_values("score", ascending=False)
            .reset_index(drop=True)
        )


# ── Tab 2: drill into a single camera with live YOLO inference ────────────────
with tab2:
    st.subheader("Single camera")

    def area_bucket(lat: float) -> str:
        """Assign a coarse Manhattan zone based on latitude.
        Thresholds chosen to roughly match Upper (above 96th St), Midtown, Lower.
        """
        if lat > 40.78:
            return "Upper Manhattan"
        elif lat > 40.75:
            return "Midtown"
        else:
            return "Lower Manhattan"

    # Work on a local copy so the cached cams_df is not mutated.
    tab2_cams = cams_df.copy()
    tab2_cams["zone"] = tab2_cams["lat"].apply(area_bucket)

    zone = st.selectbox(
        "Select Manhattan zone:", ["All", "Upper Manhattan", "Midtown", "Lower Manhattan"]
    )

    if zone != "All":
        filtered = tab2_cams[tab2_cams["zone"] == zone]
    else:
        filtered = tab2_cams

    st.write(f"{len(filtered)} cameras in this selection.")

    cam_choice = st.selectbox(
        "Choose a camera:",
        options=filtered["camera_id"].tolist(),
        format_func=lambda cid: filtered.loc[
            filtered["camera_id"] == cid, "name"
        ].iloc[0],
    )

    cam_row = filtered[filtered["camera_id"] == cam_choice].iloc[0]

    st.write(f"**Camera:** {cam_row['name']}")
    st.write(f"**Location:** lat={cam_row['lat']:.5f}, lon={cam_row['lon']:.5f}")

    mode = st.radio(
        "View mode:",
        ["Single snapshot", "Continuous live view"],
        horizontal=True,
    )

    placeholder = st.empty()
    model = get_model()
    names = model.names

    if mode == "Single snapshot":
        if st.button("Get latest frame"):
            with st.spinner("Running YOLO..."):
                annotated, res = run_yolo_on_camera(cam_row)

            if annotated is not None:
                placeholder.image(
                    annotated, channels="BGR", caption="Detections"
                )

                # Build a class-count dict from the detection results for display.
                counts: dict[str, int] = {}
                for box in res.boxes:
                    cls_id = int(box.cls[0])
                    name = names.get(cls_id, str(cls_id))
                    counts[name] = counts.get(name, 0) + 1

                st.markdown("**Detections:**")
                st.json(counts)
            else:
                st.error("Could not fetch or process frame for this camera.")

    else:  # Continuous live view — loops up to 15 frames, 2 seconds apart
        run_live = st.checkbox("Start continuous live view (updates every 2s)")

        if run_live:
            for i in range(15):
                with st.spinner(f"Frame {i+1}/15..."):
                    annotated, res = run_yolo_on_camera(cam_row)
                if annotated is None:
                    st.error("Could not fetch or process frame. Stopping.")
                    break

                placeholder.image(
                    annotated,
                    channels="BGR",
                    caption=f"Frame {i+1}",
                )

                time.sleep(2)

            st.info("Live demo finished. Uncheck and re-check to run again.")


# ── Tab 3: results from the offline video segment analysis ────────────────────
with tab3:
    st.subheader("Recorded Segment Evaluation")

    seg_df = load_segment_stats()
    if seg_df is None:
        st.warning("No segment_stats.csv found. Run compute_stats.py first.")
    else:
        st.markdown("### Segment Summary Table")
        st.dataframe(
            seg_df[
                [
                    "segment_id",
                    "name",
                    "lat",
                    "lon",
                    "avg_score",
                    "level",
                    "avg_vehicles",
                    "avg_pedestrians",
                    "avg_signals",
                ]
            ].reset_index(drop=True)
        )

        st.markdown("### Average Congestion per Segment")
        chart_df = seg_df.set_index("name")[["avg_score"]]
        st.bar_chart(chart_df)


# ── Tab 4: chat with an LLM about the current traffic data ───────────────────
with tab4:
    st.subheader("Traffic Q&A (LLM)")
    st.caption(
        "Uses merged stats from this repo (plus segments if you have them). "
        "Keys: OPENAI_API_KEY and optional OPENAI_BASE_URL / OPENAI_MODEL; "
        "or LLM_PROVIDER=anthropic with ANTHROPIC_API_KEY."
    )

    # Build the context string from the current in-memory dataframes so the LLM
    # always reflects the data loaded at app startup.
    seg_ctx = load_segment_stats()
    context_text = build_traffic_context(merged, seg_ctx)

    with st.expander("Show text sent to the model"):
        # Truncate the preview to 12 000 chars so it doesn't overwhelm the UI.
        st.text(context_text[:12000] + ("..." if len(context_text) > 12000 else ""))

    # Persist conversation history across Streamlit re-runs using session state.
    if "llm_messages" not in st.session_state:
        st.session_state.llm_messages = []

    for msg in st.session_state.llm_messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    user_q = st.chat_input('e.g. "How is traffic in Midtown today?"')
    if user_q:
        st.session_state.llm_messages.append({"role": "user", "content": user_q})
        with st.chat_message("user"):
            st.markdown(user_q)

        # The system prompt grounds the model strictly in the traffic data so it
        # doesn't hallucinate conditions beyond what the cameras captured.
        system = (
            "You answer questions about Manhattan traffic using only the data below. "
            "If there are no scores, say so and mention update_camera_stats.py. "
            "Use the Midtown / Upper / Lower labels as given. Do not guess beyond the table. "
            "Keep it short.\n\n"
            f"Data:\n{context_text}"
        )

        # Prepend the system message; the model sees the full conversation history.
        api_messages = (
            [{"role": "system", "content": system}]
            + [
                {"role": m["role"], "content": m["content"]}
                for m in st.session_state.llm_messages
            ]
        )

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
                st.session_state.llm_messages.append({"role": "assistant", "content": reply or ""})

    if st.button("Clear chat history", key="clear_llm_chat"):
        st.session_state.llm_messages = []
        st.rerun()


# ── Tab 5: emergency routing with RL, vision confirmation and explanation ─────
with tab5:
    st.subheader("Emergency routing")
    st.caption(
        "Plans a route on the congestion-weighted road graph, confirms it "
        "against the cameras along the way, and explains the decision."
    )

    @st.cache_resource
    def get_route_graph():
        """Build the road graph once per server start; it carries the
        congestion scores from camera_stats.csv."""
        return build_default_graph()

    route_g = get_route_graph()

    # Start and destination are picked as cameras purely because their names
    # are recognisable street corners; the route snaps to the nearest
    # intersection of the road grid.
    cam_options = cams_df.dropna(subset=["lat", "lon"])
    col_a, col_b = st.columns(2)
    with col_a:
        start_cam = st.selectbox(
            "Start (vehicle position):",
            cam_options["camera_id"].tolist(),
            format_func=lambda cid: cam_options.loc[
                cam_options["camera_id"] == cid, "name"
            ].iloc[0],
            key="route_start",
        )
    with col_b:
        end_cam = st.selectbox(
            "Destination (incident):",
            cam_options["camera_id"].tolist(),
            index=len(cam_options) - 1,
            format_func=lambda cid: cam_options.loc[
                cam_options["camera_id"] == cid, "name"
            ].iloc[0],
            key="route_end",
        )

    col_c, col_d = st.columns(2)
    with col_c:
        strategy = st.selectbox(
            "Routing strategy:", ["A*", "Dijkstra", "Q-learning (RL)"]
        )
    with col_d:
        vision_mode = st.selectbox(
            "Vision check:",
            ["Offline (stored camera scores)", "Live (fresh YOLO per camera)"],
        )

    if st.button("Plan route", type="primary"):
        start_row = cam_options[cam_options["camera_id"] == start_cam].iloc[0]
        end_row = cam_options[cam_options["camera_id"] == end_cam].iloc[0]

        src = nearest_node(route_g, float(start_row["lat"]), float(start_row["lon"]))
        dst = nearest_node(route_g, float(end_row["lat"]), float(end_row["lon"]))

        if src == dst:
            st.error("Start and destination snap to the same intersection - pick points further apart.")
            st.stop()

        # Wrap the chosen strategy in the common (graph, src, dst) signature
        # expected by the vision gate.
        if strategy == "Dijkstra":
            planner = dijkstra_route
        elif strategy == "Q-learning (RL)":
            def planner(g, a, b):
                agent = QLearningRouter(g, seed=0)
                agent.train(a, b, episodes=1500)
                return agent.best_route()
        else:
            planner = astar_route

        mode = "live" if vision_mode.startswith("Live") else "offline"
        yolo_model = get_model() if mode == "live" else None

        with st.spinner(f"Planning with {strategy} and confirming via cameras..."):
            route, gate_info = plan_confirmed_route(
                route_g, src, dst, planner=planner, mode=mode, model=yolo_model
            )
            baseline = static_baseline_route(route_g, src, dst)

        m_route = route_metrics(route_g, route)
        m_base = route_metrics(route_g, baseline)

        # Headline numbers: chosen route vs the congestion-blind baseline.
        saved_s = m_base["travel_time_s"] - m_route["travel_time_s"]
        col1, col2, col3 = st.columns(3)
        col1.metric("Estimated time", f"{m_route['travel_time_s'] / 60:.1f} min")
        col2.metric("Static baseline", f"{m_base['travel_time_s'] / 60:.1f} min")
        col3.metric("Time saved", f"{max(saved_s, 0) / 60:.1f} min")

        if gate_info["confirmed"]:
            st.success(
                f"Route visually confirmed in {gate_info['attempts']} attempt(s)."
            )
        else:
            st.warning("Route could not be fully confirmed - dispatch with caution.")

        for cam in gate_info.get("blocked_cameras", []):
            st.warning(
                f"Avoided blocked intersection: {cam['camera']} "
                f"(score {cam['score']:.0f})"
            )
        for cam in gate_info.get("endpoint_warnings", []):
            st.info(
                f"Heavy congestion at endpoint camera {cam['camera']} "
                f"(score {cam['score']:.0f}) - unavoidable."
            )

        # Map: static matplotlib figure - renders reliably inside Streamlit
        # tabs, unlike WebGL charts.  Grid streets are shaded by congestion,
        # the chosen route is blue, the blind baseline is dashed grey.
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(7, 9))

        for u, v, d in route_g.edges(data=True):
            x = [route_g.nodes[u]["lon"], route_g.nodes[v]["lon"]]
            y = [route_g.nodes[u]["lat"], route_g.nodes[v]["lat"]]
            # Darker orange = more congested street.
            heat = min((d["congestion"] - 1.0) / 2.0, 1.0)
            ax.plot(x, y, color=(1.0, 0.85 - 0.6 * heat, 0.7 - 0.6 * heat),
                    linewidth=0.6, zorder=1)

        def draw_path(path, **kwargs):
            ax.plot(
                [route_g.nodes[n]["lon"] for n in path],
                [route_g.nodes[n]["lat"] for n in path],
                **kwargs,
            )

        draw_path(baseline, color="grey", linestyle="--", linewidth=1.8,
                  zorder=2, label="Static baseline")
        draw_path(route, color="#1e64ff", linewidth=2.6, zorder=3, label="Chosen route")

        ax.scatter([float(start_row["lon"])], [float(start_row["lat"])],
                   color="green", s=90, zorder=4, label="Start")
        ax.scatter([float(end_row["lon"])], [float(end_row["lat"])],
                   color="red", s=90, zorder=4, label="Destination")

        for cam in gate_info.get("blocked_cameras", []):
            n = cam["node"]
            ax.scatter([route_g.nodes[n]["lon"]], [route_g.nodes[n]["lat"]],
                       color="black", marker="x", s=80, zorder=4)

        ax.set_aspect(1.3)  # roughly compensate for latitude distortion
        ax.set_xticks([])
        ax.set_yticks([])
        ax.legend(loc="upper left", fontsize=8)
        ax.set_title("Route on the congestion-weighted road grid", fontsize=10)
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)
        st.caption("Street shading = congestion (darker orange is slower). "
                   "Black X marks a blockage the route avoided.")

        # Finally the plain-English justification (LLM, or template fallback).
        with st.spinner("Writing explanation..."):
            explanation = explain_route(
                route_g, route, baseline, gate_info,
                strategy=f"{strategy} + vision",
            )
        st.markdown("### Why this route")
        st.write(explanation)
