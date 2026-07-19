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
intersection on a live DOT camera. This project follows the same architecture
production navigators use — a directed road graph with live conditions applied
as edge-weight multipliers, never a topology rebuild — and adds the missing
vision signal:

1. **Plan** on real Manhattan streets from OpenStreetMap (Dijkstra / A* /
   corridor-trained Q-learning), weighted by YOLOv12 camera congestion.
2. **Drive with lookahead** — before the vehicle enters the next few camera
   intersections, each one is re-checked (fresh YOLO in live mode).
3. **Recalculate en route** — a blocked camera ahead cuts that intersection and
   replans from the vehicle's current position, exactly like Google Maps
   redrawing after a wrong turn. Abandoned legs stay on the map, faded.
4. **Compare honestly** — the final path is overlaid on Google Directions (or
   OSRM) so the divergence created by vision confirmation is visible.

**Constraint trade-off:** the street graph is cached as GraphML in `data/` so
Docker and offline runs never touch the network; tabular Q-learning trains
inside a corridor around the A* spine because exploring a 10k-node city graph
per query is not tractable for a demo that must answer in seconds.

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
                    │  streets.py ────── real OSM street graph (cached)      │
                    │  graph.py ──────── synthetic grid fallback             │
                    │  planners.py ───── Dijkstra / A* / static baseline     │
                    │  rl_agent.py ───── corridor Q-learning under noise     │
                    │  vision_gate.py ── YOLO confirm / re-plan (pre-drive)  │
                    │  navigator.py ──── en-route lookahead + recalculation  │
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
`python -m routing.external_route`, `python -m routing.vision_gate`,
`python -m routing.streets`, `python -m routing.navigator`.

The OSM street cache ships in `data/manhattan_streets.graphml`; refresh it with
`python -m routing.streets --build` if the road network ever changes.

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
| **Route planner** | Type any two Manhattan places (Nominatim search bounded to the island, offline camera-name fallback). Dark real-street map: cyan = path driven, dashed slate = Google/OSRM, amber = legs abandoned when the drive recalculated. An en-route demo stages a mid-drive blockage so you can watch the recalculation. Searches outside Manhattan get a clear "Manhattan only" error. |
| **Live cameras** | Congestion map of every DOT camera (colour = YOLO level), metrics, live YOLO on any camera, full sortable table |
| **Ask the city** | LLM chat grounded in the current congestion data |

Searching a place outside the island (e.g. "Newark Airport") is rejected with a
friendly error rather than routing off-coverage.

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
