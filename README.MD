# Manhattan Traffic – YOLOv12 + Streamlit

Pulls live NYC DOT webcam snapshots for Manhattan, runs YOLOv12-S on each image to count vehicles/pedestrians/signals, stores the results, and displays everything in a Streamlit dashboard with an optional LLM chat tab.

---

## Table of contents

1. [Project overview](#project-overview)
2. [File layout](#file-layout)
3. [Requirements](#requirements)
4. [One-time setup](#one-time-setup)
5. [Running the pipeline](#running-the-pipeline)
6. [Running the dashboard](#running-the-dashboard)
7. [LLM chat tab](#llm-chat-tab)
8. [Offline video segments (optional)](#offline-video-segments-optional)
9. [Troubleshooting](#troubleshooting)

---

## Project overview

```
NYC DOT API  →  fetch_cameras.py  →  manhattan_cameras.csv
                                            │
                                  update_camera_stats.py  →  camera_stats.csv
                                            │
                                       dashboard.py  (Streamlit)
                                            │
                              (optional) compute_stats.py  →  segment_stats.csv
```

The dashboard has four tabs:

| Tab | What it shows |
|-----|---------------|
| **Overview** | Map of all cameras + sortable congestion table |
| **Camera explorer** | Live YOLO snapshot for any single camera |
| **Segments** | Averaged congestion scores from pre-recorded video clips |
| **Ask traffic (LLM)** | Chat with an LLM about the current congestion data |

---

## File layout

```
traffic-final/
├── fetch_cameras.py        # Step 1 – download camera list from NYC DOT API
├── update_camera_stats.py  # Step 2 – run YOLO on every live camera image
├── compute_stats.py        # Optional – run YOLO on local video files
├── dashboard.py            # Streamlit dashboard (all four tabs)
├── traffic_llm.py          # LLM helpers used by the chat tab
│
├── weights/
│   └── yolov12s.pt         # YOLOv12-S model weights  ← you must supply this
│
├── videos/                 # Only needed for compute_stats.py  ← you must supply these
│   ├── seg_a.mp4
│   ├── seg_b.mp4
│   └── seg_c.mp4
│
├── manhattan_cameras.csv   # Created by fetch_cameras.py
├── camera_stats.csv        # Created by update_camera_stats.py
└── segment_stats.csv       # Created by compute_stats.py
```

---

## Requirements

- **Python 3.10 or 3.11** (YOLOv12 / ultralytics may not support 3.12 yet)
- A virtual environment (the repo ships with `.venv/`)
- Internet access to reach the NYC DOT API and your chosen LLM endpoint

Install dependencies (if starting fresh):

```bash
pip install ultralytics streamlit opencv-python-headless pandas requests numpy
```

> If you already have the `.venv` in this folder, activate it and skip the install:
> ```powershell
> # Windows PowerShell
> .\.venv\Scripts\Activate.ps1
> ```
> ```bash
> # macOS / Linux
> source .venv/bin/activate
> ```

---

## One-time setup

### 1. Place the model weights

Copy or download `yolov12s.pt` into the `weights/` folder:

```
weights/yolov12s.pt
```

The file must be named exactly `yolov12s.pt`. If you have a different variant (e.g. `yolov12n.pt`), update the `WEIGHTS_PATH` constant at the top of both `compute_stats.py` and `update_camera_stats.py`.

### 2. Fetch the camera list

```bash
python fetch_cameras.py
```

This calls the public NYC DOT API and writes `manhattan_cameras.csv` (~200 rows). It takes only a few seconds. Re-run it whenever you want a refreshed list of cameras (cameras go offline or change URLs occasionally).

---

## Running the pipeline

### Step 1 – fetch cameras (one-time or periodic)

```bash
python fetch_cameras.py
```

Output: `manhattan_cameras.csv`

### Step 2 – score every camera with YOLO

```bash
python update_camera_stats.py
```

This downloads a live JPEG from each of the ~200 Manhattan cameras, runs YOLO on it, and writes `camera_stats.csv`. Expect it to take **5–15 minutes** depending on network speed and your hardware. A 0.5-second pause between cameras prevents overloading the NYC DOT image server.

Output: `camera_stats.csv`

> **Tip:** Run this step again any time you want updated congestion numbers. The dashboard reads whichever CSV is on disk when it starts.

---

## Running the dashboard

```bash
streamlit run dashboard.py
```

Streamlit will print a local URL (usually `http://localhost:8501`). Open it in your browser. The app reads `manhattan_cameras.csv` and `camera_stats.csv` at startup — make sure you have run Steps 1 and 2 first.

If `camera_stats.csv` is missing, the Overview tab will show a warning but the app will still load. The Camera Explorer tab can still perform live YOLO inference on demand.

---

## LLM chat tab

The **Ask traffic** tab sends the current congestion data as context to an LLM so you can ask plain-English questions like *"Which area is most congested right now?"*

Set **one** of the following sets of environment variables before launching the dashboard:

### Option A – OpenAI (or any OpenAI-compatible endpoint)

```powershell
# Windows PowerShell
$env:OPENAI_API_KEY = "sk-..."
# Optional overrides:
$env:OPENAI_MODEL    = "gpt-4o"           # default: gpt-4o-mini
$env:OPENAI_BASE_URL = "https://..."      # default: https://api.openai.com/v1
```

```bash
# macOS / Linux
export OPENAI_API_KEY="sk-..."
export OPENAI_MODEL="gpt-4o"           # optional
export OPENAI_BASE_URL="https://..."   # optional
```

Works with **OpenAI**, **OpenRouter**, and **Ollama** (set `OPENAI_BASE_URL=http://localhost:11434/v1` and any non-empty key).

### Option B – Anthropic Claude

```powershell
$env:LLM_PROVIDER    = "anthropic"
$env:ANTHROPIC_API_KEY = "sk-ant-..."
# Optional overrides:
$env:ANTHROPIC_MODEL      = "claude-haiku-4-5"  # default
$env:ANTHROPIC_MAX_TOKENS = "2048"              # default
```

```bash
export LLM_PROVIDER="anthropic"
export ANTHROPIC_API_KEY="sk-ant-..."
```

If no API key is set, the chat tab will display a clear error message explaining what is missing.

---

## Offline video segments (optional)

`compute_stats.py` analyses pre-recorded MP4 clips and writes `segment_stats.csv`, which appears in the **Segments** tab of the dashboard.

### Why the script reports "Video not found"

The script expects MP4 files at these exact paths relative to the project root:

```
videos/seg_a.mp4   (Midtown segment)
videos/seg_b.mp4   (7th Avenue segment)
videos/seg_c.mp4   (Broadway segment)
```

**The `videos/` folder and its contents are not included in this repository** — you must supply your own recordings. These can be any traffic footage you have captured (or downloaded from a legal source) for the corresponding Manhattan locations.

### How to set it up

1. Create the `videos/` directory:

    ```powershell
    # Windows
    mkdir videos
    ```

    ```bash
    # macOS / Linux
    mkdir videos
    ```

2. Copy your MP4 files into it, named `seg_a.mp4`, `seg_b.mp4`, and `seg_c.mp4`.

    > If you want to use different filenames or add more segments, edit the `SEGMENTS` list at the top of `compute_stats.py` — each entry just needs a `segment_id`, `name`, `lat`, `lon`, and `video_path`.

3. Run the script:

    ```bash
    python compute_stats.py
    ```

    Output: `segment_stats.csv`. The script samples every 10th frame up to 60 frames per clip, so even a 30-second clip works fine.

---

## Troubleshooting

| Symptom | Likely cause | Fix |
|---------|-------------|-----|
| `[WARN] Video not found for seg_a` | `videos/` folder or the MP4 file is missing | Create the folder and place your MP4 files inside it (see above) |
| `[ERROR] No segment stats produced` | All three video files are missing | Same as above |
| `FileNotFoundError: weights/yolov12s.pt` | Model weights not present | Copy `yolov12s.pt` into the `weights/` folder |
| Camera map is empty | `camera_stats.csv` has no valid lat/lon rows | Re-run `fetch_cameras.py` then `update_camera_stats.py` |
| LLM tab shows "Set OPENAI_API_KEY" | API key env var not set | Set the appropriate env var before running `streamlit run dashboard.py` |
| Dashboard shows stale scores | `camera_stats.csv` is old | Re-run `update_camera_stats.py` and restart the dashboard |
| `FlashAttention is not available` log line | Your GPU/CPU doesn't support Flash Attention | Safe to ignore — PyTorch falls back to `scaled_dot_product_attention` automatically |
| Camera frames fail to load in the explorer | Camera is offline or the NYC DOT URL changed | Try a different camera; re-run `fetch_cameras.py` to refresh URLs |
