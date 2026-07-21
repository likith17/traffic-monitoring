# Emergency Routing for Smart Response

An emergency-vehicle router for Manhattan that checks live traffic cameras before it commits to a street.

Ordinary navigation optimises for *typical* traffic. It cannot see that the intersection two blocks ahead is blocked right now. This project adds that missing sensor: YOLOv12 scores 370-odd live NYC DOT camera feeds, those scores become edge weights on the real OpenStreetMap road network, and the router refuses to send a vehicle through an intersection a camera says is impassable — recalculating mid-drive when one turns out to be, the same way your phone reroutes after a wrong turn.

![Demo walkthrough](docs/assets/demo.gif)

**Headline result.** Over 30 simulated emergencies with road blockages the planner cannot see in advance, mean response time drops from **27.0 minutes to 21.4 minutes — 20.8% faster** than the shortest-path dispatch it replaces. Reproduce it with `python -m routing.simulate --episodes 30`.

---

## Contents

- [The problem, and how success is measured](#the-problem-and-how-success-is-measured)
- [How the system works](#how-the-system-works)
- [Results](#results)
- [Design decisions, and why](#design-decisions-and-why)
- [Problems hit along the way](#problems-hit-along-the-way)
- [Running it](#running-it)
- [Project layout](#project-layout)
- [What I know is wrong with it](#what-i-know-is-wrong-with-it)

---

## The problem, and how success is measured

**The situation.** Dispatchers send ambulances and fire trucks along routes chosen by systems that model traffic statistically. Those models are good at "this avenue is usually slow at 5pm" and blind to "this intersection is blocked right now." New York City already operates several hundred public traffic cameras that show exactly that, and nothing in the dispatch loop looks at them.

**The baseline being beaten.** Shortest-path routing on free-flow travel times — the naive dispatcher. It is not a strawman: it is what you get from any router that lacks a live congestion feed, and it is implemented here as `static_baseline_route()` so every comparison runs against real code rather than an assumed number.

**Success metric.** Minutes of response time, not detection accuracy. mAP on a COCO validation set says nothing about whether an ambulance arrives sooner. The number that matters is mean simulated response time versus that baseline, measured on the same road network with the same blockages.

**Guardrails I refused to trade away.**

- The router must always return a route. A vision check that fails, a camera that is offline, an RL policy that gets stuck — none of these may leave a vehicle without a plan. Every stage has a fallback.
- Comparisons against Google and OSRM must be honest. Their ETAs are re-scored inside *this* congestion model rather than taken at face value, because their number answers a different question.
- No fabricated data. Every camera score comes from an actual inference on an actual frame.

**Constraints that shaped the build.** A public demo has to answer in seconds, which rules out anything that trains over the full city graph per request. It also has to run on a free-tier host with no GPU, which is what eventually pushed PyTorch out of the container. And it has to work when the network does not, so the camera index, congestion scores and street graph all ship inside the repo.

---

## How the system works

```
NYC DOT cameras
      |
      |  fetch_cameras.py         one-time index of Manhattan cameras
      v
manhattan_cameras.csv
      |
      |  update_camera_stats.py   YOLOv12 per camera -> congestion score
      v
camera_stats.csv
      |
      +-------------------------------------------------+
      |                  routing/                        |
      |  streets.py     real OSM network (cached)        |
      |  graph.py       camera scores -> edge weights    |
      |  planners.py    Dijkstra / A* / static baseline  |
      |  rl_agent.py    corridor Q-learning              |
      |  vision_gate.py camera check before dispatch     |
      |  navigator.py   drive loop + recalculation       |
      |  detect.py      YOLOv12 via ONNX Runtime         |
      |  explain.py     plain-English justification      |
      |  simulate.py    benchmark vs the baseline        |
      +-------------------------------------------------+
      |
      v
dashboard.py  (Streamlit)
```

**Perception.** Each camera snapshot goes through YOLOv12. Detections collapse into one congestion score with weights chosen for road impact rather than object count: a vehicle counts 1.0 because it occupies a lane, a pedestrian 0.3 because they mostly do not, a traffic signal 0.5 because it marks an intersection where queueing happens. Under 5 is low, 5–14 medium, 15 and above high.

**World model.** `streets.py` pulls the drivable Manhattan network from OpenStreetMap once and caches it as GraphML (10,893 intersections, 24,852 one-way street segments). Each camera snaps to its nearest intersection and inflates the travel time of every street touching it: `travel_time = base_time × (1 + score × 0.1)`. A score of 10 makes a street twice as slow. The topology never changes; only the weights do.

**Planning.** A\* and Dijkstra minimise that congestion-aware time. A\*'s heuristic is straight-line distance at the network's fastest speed, which cannot overestimate, so it stays optimal while exploring fewer nodes. The Q-learning agent treats intersections as states and streets as actions, with reward equal to negative travel time plus a large arrival bonus — but it trains with ±25% random noise on every edge time, so it learns routes that stay fast when the congestion picture is stale rather than routes tuned to one snapshot.

**Vision gate.** Before dispatch, `vision_gate.py` walks the chosen route and checks every camera on it. Any intersection scoring at or above the blocking threshold gets cut out of a working copy of the graph and the planner runs again, up to five times. `navigator.py` does the same thing while driving: look a few cameras ahead, and if one is blocked, replan from the vehicle's current position rather than from the start.

**Explanation.** `explain.py` assembles the facts — time chosen, time saved, blockages avoided, vision verdict — and asks an LLM to phrase them in a few sentences. Without an API key it falls back to a template built from the same facts, so the feature degrades rather than breaks.

---

## Results

Thirty simulated emergencies, six road blockages each, on the real street network:

| Strategy | Mean response | vs baseline |
|---|---|---|
| Static shortest path (baseline) | 27.0 min | — |
| A\* + vision gate | 21.4 min | −20.8% |
| Corridor Q-learning + vision gate | 21.9 min | −18.9% |

The benchmark's design matters more than the numbers. Each episode plants blockages at random camera intersections and then maintains two versions of the city: the one the planners see, where congestion weights are *stale* and do not know about the blockages, and the real one, where a blocked street costs eight times its normal time. Cameras report the truth in both, because a camera feed is always current. So the only way to learn about a blockage is to look — which is precisely the capability being measured. Every route, including the baseline's, is then priced against the real city.

The win is not a better shortest-path algorithm. A\* and Dijkstra provably find the same route here. The win is excluding streets the cameras show are impassable before the vehicle commits to them.

Per-episode results are in `routing_benchmark.csv`.

One caveat on reproducibility: `camera_stats.csv` is a snapshot from one moment, so re-running `update_camera_stats.py` at a different hour changes the congestion field and shifts the absolute minutes. The gap between strategies is the stable part; the exact numbers move with the traffic. The committed scores were taken on a weekday evening (mean score 3.07 across 371 cameras).

**Where the RL agent actually stands.** It lands slightly behind A\* on this benchmark, and I have left that in rather than tuning until it wins. It is optimising for a different thing — robustness to congestion noise, which this benchmark only partly rewards — and on a static snapshot A\* is already optimal, so there is nothing for RL to beat. Reporting it as a loss is more useful than reporting a number I massaged.

---

## Design decisions, and why

Most of these were reversals. The first approach is listed because the reason it failed is the interesting part.

| What I tried first | Why it broke | What it is now |
|---|---|---|
| Synthetic grid graph over the camera bounding box | Fast to write, but routes cut straight across the Hudson and through Central Park. Fine for a benchmark, indefensible in a public demo. | Real OSM drive network, cached as GraphML, curved street geometry preserved (`routing/streets.py`) |
| Tabular Q-learning over the whole city graph | 10,893 nodes is far too large for tabular RL to explore in the seconds a demo has. It never converged. | Corridor Q-learning: train only in a band around the A\* route, which is what hierarchical routing engines do (`routing/rl_agent.py`) |
| Rebuilding the graph when traffic changes | Slow and fragile. Production navigators do not do this. | Fixed topology, congestion as edge-weight multipliers |
| Letting the user pick the algorithm | Nobody visiting a demo wants to choose between Dijkstra and A\*. | Run all three, show the fastest, put the full comparison in a table below the map |
| Trusting Google/OSRM ETAs directly | Their number answers "how long normally," not "how long given these cameras." Comparing directly would flatter whichever model was luckier. | Re-score their polyline inside this congestion model, so both sides answer the same question |
| Nominatim on every keystroke | 1 request/second rate limit turned into fake "place not found" errors. | Landmarks and camera names first, network only when local results are thin |
| PyTorch in the container | 728 MB of a 3.27 GB image, for inference that never needs autograd or CUDA. | ONNX Runtime (`routing/detect.py`), with a parity test against PyTorch |

A few decisions worth expanding.

**Why ONNX Runtime, and what it cost.** I measured the image before optimising rather than guessing: `du` inside the running container showed site-packages at 1,849 MB, of which torch alone was 728 MB — four times the next largest package. Since the container only ever runs forward passes, that is 728 MB spent on training machinery. Exporting to ONNX and running inference through ONNX Runtime removed it.

The honest cost is that ultralytics' pre- and post-processing had to be reimplemented: letterbox resize going in, box decode and non-maximum suppression coming out, about 90 lines in `routing/detect.py`. That code could be subtly wrong in a way nothing would crash on — it would just shift every congestion score, and with it the routing and the benchmark. So `tests/test_detect_parity.py` feeds both backends byte-identical input and compares raw output tensors; they agree to 0.0012, which is float32 noise. End-to-end detection counts differ by a few percent because ultralytics letterboxes to a stride-multiple rectangle while the exported model has a fixed 640×640 input and pads differently — a framing difference, not a correctness one, and the test explains that distinction rather than hiding it behind a tolerance.

PyTorch stays in `requirements-dev.txt`. Training, re-export and the parity test all still use it. It just does not ship.

**Why the topology is fixed and congestion is a multiplier.** This is how production navigators work, and the reason is cache-ability and speed: rebuilding a 10k-node graph per request would dominate the response time, and the road layout genuinely does not change minute to minute. Only conditions do.

**Why the benchmark hides blockages from the planner.** An earlier version let planners see the blockages in their congestion weights, and every strategy scored about the same — of course they did, they were all routing around known obstacles. The information asymmetry is the experiment. Without it there is nothing to measure.

---

## Problems hit along the way

These are the ones that cost real time. Several are environment problems rather than code problems, which is most of what actually goes wrong.

**The benchmark was measuring code that does not ship.** `simulate.py` trained full-graph Q-learning while the dashboard used the corridor version. The numbers were real but described a different system. Fixed by pointing the benchmark at `rl_route`, the same function the app calls, and re-running. This is the failure mode the hiring guides warn about — a benchmark that quietly measures the wrong thing — and I would not have caught it without re-reading the two files side by side.

**Type-ahead search froze the whole page.** Typing "bellevue" took 6.96 seconds. The cause was not what it looked like: I assumed the CSV re-read per keystroke, measured it, and found it cost 3 ms. The real culprit was the Nominatim rate limiter calling `time.sleep()` for up to 1.05 seconds — on Streamlit's script thread. Every keystroke blocked the entire UI, and Streamlit's rerun landed before the results did, which read as the page refreshing early. Type-ahead now skips the network instead of sleeping when the rate-limit window has not elapsed. Same input, 0.01 seconds.

**The Docker image would have leaked my API key.** The Dockerfile copied `.streamlit/` for the theme config, which also swept in `secrets.toml`. Baked-in secrets survive in image layers even if a later layer deletes them, and `docker history` will show them. Caught before any image was pushed. `.dockerignore` now excludes it explicitly, and the build is verified with a grep for real key patterns inside the finished image.

**WSL2 refused to start with a misleading error.** Docker Desktop reported "Virtualization support not detected," which points at BIOS. It was not BIOS: `HypervisorPresent` was true and Virtualization-Based Security was actively running, both of which require VT-x. What was missing was the Virtual Machine Platform Windows feature. `wsl --install --no-distribution` from an admin prompt plus a reboot fixed it. The lesson is that `VirtualizationFirmwareEnabled: False` is a reporting artifact once a hypervisor has claimed VT-x, not evidence that virtualization is off.

**Docker Hub would not serve image layers.** Pulls failed consistently with `EOF` from `production.cloudfront.docker.com` mid-transfer, while small manifest requests succeeded. MTU was fine (1500-byte packets passed), DNS was fine, and a pull from `gcr.io` worked on the first try — so the problem was specific to Docker Hub's CDN on this network, not Docker. Worked around by pulling `python:3.11-slim` through `mirror.gcr.io`, which serves the identical image. Adding `"registry-mirrors": ["https://mirror.gcr.io"]` to the Docker Engine config makes that permanent.

**ONNX export failed with a misleading missing module.** `ModuleNotFoundError: No module named 'onnxscript'` looks like a broken install. It is not: newer PyTorch exporters require `onnxscript` and it is not a declared dependency of either torch or ultralytics. It is pinned in `requirements-dev.txt` so the next person does not lose the same twenty minutes.

**The export wrote a redundant 36 MB sidecar.** ONNX export produced `yolov12s.onnx` plus `yolov12s.onnx.data`. Re-saving with `save_as_external_data=False` produced a single 37 MB file that loads standalone and gives bit-identical output — the sidecar was duplicating weights already in the main file. Two files became one, and the artifact halved.

---

## Running it

### Locally

Python 3.11. Everything the app needs — camera index, congestion scores, street graph, ONNX model — is committed, so there is no data-prep step.

```bash
python -m venv .venv
# Windows:      .venv\Scripts\activate
# Linux/macOS:  source .venv/bin/activate

pip install -r requirements.txt
streamlit run dashboard.py
```

Opens on http://localhost:8501.

For training, re-exporting the model, or running the ONNX parity test, add the development dependencies:

```bash
pip install -r requirements-dev.txt
```

Refreshing the data and reproducing the benchmark:

```bash
python fetch_cameras.py            # rebuild the camera index
python update_camera_stats.py      # rescore every camera (several minutes)
python -m routing.simulate --episodes 30
```

Every routing module self-tests on its own, which is the fastest way to check one piece in isolation:

```bash
python -m routing.detect
python -m routing.streets
python -m routing.navigator
python -m routing.vision_gate
python -m tests.test_detect_parity   # needs requirements-dev.txt
```

### Docker

```bash
docker build -t emergency-routing .
docker run -p 8501:8501 emergency-routing
```

The image is CPU-only, runs as a non-root user, has a health check, and contains no secrets. Keys are passed at run time and never built in:

```bash
docker run -p 8501:8501 \
  -e LLM_PROVIDER=anthropic -e ANTHROPIC_API_KEY=sk-ant-... \
  emergency-routing
```

If `docker pull` fails with an `EOF` from CloudFront, see the Docker Hub note in [Problems hit along the way](#problems-hit-along-the-way).

### Optional keys

Everything works without them. The router, the vision gate, the benchmark and the OSRM comparison need no credentials at all.

```bash
export ANTHROPIC_API_KEY=...      # LLM route explanations; template fallback otherwise
export LLM_PROVIDER=anthropic     # required alongside it, or the code defaults to OpenAI
export GOOGLE_MAPS_API_KEY=...    # compare against Google Directions instead of OSRM
```

Locally these can live in `.streamlit/secrets.toml` — copy `.streamlit/secrets.toml.example` and fill it in. That file is gitignored and excluded from the Docker image.

---

## Project layout

```
routing/
  detect.py          YOLOv12 inference through ONNX Runtime
  streets.py         OSM drive network, cached as GraphML
  graph.py           camera scores -> congestion-weighted edges
  planners.py        Dijkstra, A*, and the static baseline
  rl_agent.py        corridor Q-learning under traffic noise
  vision_gate.py     camera check and re-planning before dispatch
  navigator.py       drive loop with lookahead and recalculation
  geocode.py         Manhattan-bounded place search
  external_route.py  Google Directions / OSRM comparison
  geo.py             routes -> lat/lon polylines
  map_view.py        Folium map rendering
  explain.py         LLM route justification with offline fallback
  simulate.py        benchmark against the static baseline

tests/
  test_detect_parity.py   ONNX vs PyTorch equivalence

dashboard.py         Streamlit UI
fetch_cameras.py     build the camera index
update_camera_stats.py   score every camera
compute_stats.py     offline analysis of recorded clips

data/manhattan_streets.graphml   cached street network
weights/yolov12s.onnx            deployed inference model
weights/yolov12s.pt              source weights, for re-export and training
camera_stats.csv                 current congestion scores
routing_benchmark.csv            per-episode benchmark output
```

---

## What I know is wrong with it

Being straight about the limits, because most of these are the obvious follow-up questions.

**The congestion weights are guesses.** Vehicle 1.0, pedestrian 0.3, signal 0.5 are reasonable, and they are not calibrated against anything. Doing this properly means correlating camera scores with measured travel times — which needs ground-truth trip data I do not have.

**A camera sees one frame, not a street.** Scores come from a single snapshot at one point. A camera pointed at a clear block fifty metres from a jam reports "clear." Multi-frame temporal smoothing would help.

**The RL agent does not beat A\*.** Discussed above. It is included because robustness to stale congestion is a real property worth demonstrating, not because it wins.

**Blockage severity is synthetic.** The 8× slowdown for a blocked street in the benchmark is a plausible figure, not a measured one. The relative ranking of strategies is robust to it; the absolute minutes are not.

**No monitoring or retraining loop.** There is no drift detection on camera scores and no scheduled re-scoring. For a demo that is fine. For anything real, stale congestion data is the first thing that would silently degrade.

**1.5 GB is still not small.** Less than half of what it was, but Streamlit, pandas, pyarrow, OpenCV and osmnx account for most of the remainder. Dropping osmnx from the runtime — it is used for exactly one call, loading the cached GraphML — would remove geopandas, scipy and pyogrio, worth roughly another 250 MB. That is the next thing I would do.

---

## Stack

YOLOv12 · ONNX Runtime · NetworkX · OSMnx · Streamlit · Folium · Docker
