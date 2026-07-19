# Emergency Routing for Smart Response

Real-time emergency vehicle routing for Manhattan that combines **reinforcement
learning (Q-learning)** with classical graph algorithms (**Dijkstra, A\***),
confirms route feasibility with a **YOLOv12 vision check** on live NYC DOT
traffic cameras, and explains every decision in plain English through an
**LLM module**. The public dashboard draws **Google-Maps-like dual routes**
(vision-confirmed vs standard directions) so non-experts can see the difference.
Ships with a Streamlit UI and a Docker image for reproducible Linux deployment.

Benchmark over 30 simulated emergencies with hidden road blockages
(`python -m routing.simulate`):

| Strategy | Mean response time | vs baseline |
|---|---|---|
| Static shortest-path baseline | 26.9 min | — |
| A* + vision gate | 21.8 min | **-18.8%** |
| Q-learning + vision gate | 22.5 min | **-16.4%** |

---

## Why not just Google Maps?

Standard directions optimise for typical traffic. They **cannot see** a blocked
intersection on a live DOT camera. This project adds that missing signal:

1. Plan on a congestion-weighted graph (Dijkstra / A* / Q-learning under noise).
2. Re-check every camera on the candidate path (YOLOv12).
3. Cut blocked nodes and re-plan until the path is vision-confirmed.
4. Overlay our path against Google Directions (or OSRM) so the divergence is visible.

**Constraint trade-off:** the planner runs on a reproducible Manhattan-style grid
snapped to real camera coordinates (works fully offline in Docker). External
comparison uses real street geometry from Google / OSRM when the network is up.

---

## How it works

```
NYC DOT cameras ──> fetch_cameras.py ──> manhattan_cameras.csv
                                                │
                              update_camera_stats.py (YOLOv12 per camera)
                                                │
                                         camera_stats.csv
                                                │
                    ┌───────────────────────────┴───────────────────────────┐
                    │                    routing/ package                    │
                    │                                                       │
                    │  graph.py ──────── road grid + congestion weights      │
                    │  planners.py ───── Dijkstra / A* / static baseline     │
                    │  rl_agent.py ───── Q-learning under traffic noise      │
                    │  vision_gate.py ── YOLO confirm / re-plan              │
                    │  external_route.py Google Directions → OSRM → naive    │
                    │  geo.py / map_view.py  public Folium dual-route map    │
                    │  explain.py ────── LLM justification (w/ fallback)     │
                    │  simulate.py ───── benchmark vs static baseline        │
                    └───────────────────────────┬───────────────────────────┘
                                                │
                         dashboard.py (Streamlit — Emergency routing first)
```

---

## Quick start (local)

Requires Python 3.11 and the YOLOv12 weights at `weights/yolov12s.pt`
(included in this repo). Prefer the project venv:

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt

# 1. Refresh the camera list and congestion scores (optional - CSVs included)
python fetch_cameras.py
python update_camera_stats.py

# 2. Run the routing benchmark
python -m routing.simulate --episodes 30

# 3. Launch the dashboard
streamlit run dashboard.py
```

Self-tests: `python -m routing.geo`, `python -m routing.map_view`,
`python -m routing.external_route`, `python -m routing.vision_gate`.

## Quick start (Docker)

```bash
docker build -t emergency-routing .
docker run -p 8501:8501 emergency-routing
```

Open http://localhost:8501. The image is CPU-only and bundles the weights and
pre-computed camera data. Without network it falls back to the naive free-flow
comparison path; with network it uses OSRM (or Google if a key is set).

---

## Dashboard

| Tab | What it shows |
|---|---|
| **Emergency routing** | Interactive street map: navy = vision-confirmed path, dashed grey = Google/OSRM; metrics compare both; LLM explanation |
| **Overview** | Camera map + congestion table |
| **Camera explorer** | Live YOLO on any camera |
| **Segments** | Offline clip scores (`compute_stats.py`) |
| **Ask traffic (LLM)** | Chat grounded in congestion data |

### Map comparison providers (honest labels)

| Provider shown in UI | When |
|---|---|
| **Google Maps** | `GOOGLE_MAPS_API_KEY` (or `GOOGLE_API_KEY`) is set and Directions succeeds |
| **OSRM (OpenStreetMap)** | Default online fallback — free public router |
| **Naive free-flow path** | Offline / no network — in-graph static baseline |

```bash
export GOOGLE_MAPS_API_KEY="your-key"   # optional, enables true Google compare
```

## LLM configuration (optional)

```bash
# OpenAI-compatible (OpenAI, OpenRouter, Ollama, ...)
export OPENAI_API_KEY="sk-..."
export OPENAI_MODEL="gpt-4o-mini"        # optional
export OPENAI_BASE_URL="https://..."     # optional

# or Anthropic
export LLM_PROVIDER="anthropic"
export ANTHROPIC_API_KEY="sk-ant-..."
```

Without a key, explanations fall back to an automatic template.

---

## Repository notes

- `routing_benchmark.csv` — per-episode results behind the table above.
- YOLO training assets (`train/`, `valid/`, `test/`) stay in history and are
  excluded from the Docker image.
- `compute_stats.py` scores MP4 clips into `segment_stats.csv`; place them under
  `videos/`.
