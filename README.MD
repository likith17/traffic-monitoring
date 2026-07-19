# Emergency Routing for Smart Response

Real-time emergency vehicle routing for Manhattan that combines **reinforcement
learning (Q-learning)** with classical graph algorithms (**Dijkstra, A\***),
confirms route feasibility with a **YOLOv12 vision check** on live NYC DOT
traffic cameras, and explains every decision in plain English through an
**LLM module**. Ships with a Streamlit dashboard and a Docker image for
reproducible Linux deployment.

Benchmark over 30 simulated emergencies with hidden road blockages
(`python -m routing.simulate`):

| Strategy | Mean response time | vs baseline |
|---|---|---|
| Static shortest-path baseline | 26.9 min | — |
| A* + vision gate | 21.8 min | **-18.8%** |
| Q-learning + vision gate | 22.5 min | **-16.4%** |

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
                    │  graph.py ──── road grid, congestion-weighted edges    │
                    │  planners.py ─ Dijkstra / A* / static baseline         │
                    │  rl_agent.py ─ Q-learning trained under traffic noise  │
                    │  vision_gate.py ─ YOLO re-check of cameras on route,   │
                    │                   re-plans around blocked streets      │
                    │  explain.py ── LLM route justification (w/ fallback)   │
                    │  simulate.py ─ benchmark vs static baseline            │
                    └───────────────────────────┬───────────────────────────┘
                                                │
                                    dashboard.py (Streamlit, 5 tabs)
```

1. **Perception** – every Manhattan DOT camera snapshot is scored by YOLOv12
   (vehicles ×1.0, pedestrians ×0.3, signals ×0.5) into a congestion score.
2. **World model** – a directed road grid is stretched over the real camera
   bounding box; each camera snaps to its nearest intersection and slows down
   the streets feeding it.
3. **Planning** – Dijkstra and A* minimise congestion-aware travel time; the
   tabular Q-learning agent trains across episodes with randomly perturbed
   edge times, learning routes that stay fast when traffic is uncertain.
4. **Vision confirmation** – before dispatch, every camera along the candidate
   route is re-checked (fresh YOLO inference in live mode, stored scores in
   offline mode). Blocked intersections are cut out and the planner re-runs
   until the route is confirmed clear.
5. **Explanation** – the route facts (time saved, blockages avoided, vision
   verdict) are handed to an LLM that writes a short dispatcher-friendly
   justification; a template fallback answers when no API key is configured.

---

## Quick start (local)

Requires Python 3.11 and the YOLOv12 weights at `weights/yolov12s.pt`
(included in this repo).

```bash
pip install -r requirements.txt

# 1. Refresh the camera list and congestion scores (optional - CSVs included)
python fetch_cameras.py
python update_camera_stats.py

# 2. Run the routing benchmark
python -m routing.simulate --episodes 30

# 3. Launch the dashboard
streamlit run dashboard.py
```

Every routing module also runs standalone as a self-test, e.g.
`python -m routing.vision_gate`.

## Quick start (Docker)

```bash
docker build -t emergency-routing .
docker run -p 8501:8501 emergency-routing
```

Open http://localhost:8501. The image is CPU-only and bundles the weights and
pre-computed camera data, so it works fully offline out of the box.

---

## Dashboard tabs

| Tab | What it shows |
|---|---|
| **Overview** | Map of all cameras + sortable congestion table |
| **Camera explorer** | Live YOLO inference on any single camera |
| **Segments** | Averaged congestion from pre-recorded clips (`compute_stats.py`) |
| **Ask traffic (LLM)** | Chat with an LLM grounded in the current congestion data |
| **Emergency routing** | Plan a route (Dijkstra / A* / RL), vision-confirm it, see the map, time saved vs baseline, and the LLM explanation |

## LLM configuration (optional)

The chat tab and the route explanations use one of:

```bash
# OpenAI-compatible (OpenAI, OpenRouter, Ollama, ...)
export OPENAI_API_KEY="sk-..."           # required
export OPENAI_MODEL="gpt-4o-mini"        # optional
export OPENAI_BASE_URL="https://..."     # optional

# or Anthropic
export LLM_PROVIDER="anthropic"
export ANTHROPIC_API_KEY="sk-ant-..."
```

Without a key everything still works - explanations fall back to an
automatic template.

---

## Repository notes

- `routing_benchmark.csv` - per-episode results behind the table above.
- The YOLO training assets from the earlier phase of this project (the
  `train/ valid/ test` dataset, `Traffic_Analysis.ipynb`, `yolo11s.pt`) remain
  in the repository history and are excluded from the Docker image.
- `compute_stats.py` scores your own MP4 clips into `segment_stats.csv`; place
  them under `videos/` (see the `SEGMENTS` list at the top of the script).
