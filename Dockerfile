# Multi-stage build for the Emergency Routing dashboard.
#
# Two things keep this image small. First, inference runs on ONNX Runtime
# rather than PyTorch, so torch (728 MB measured inside the earlier image)
# never gets installed. Second, pip and its build dependencies run in a
# throwaway builder stage; the runtime stage copies only the finished
# site-packages directory, so compilers and wheel caches stay behind.
#
# Build:  docker build -t emergency-routing .
# Run:    docker run -p 8501:8501 emergency-routing
#
# Never bake API keys in - pass them at run time:
#   docker run -p 8501:8501 \
#     -e LLM_PROVIDER=anthropic -e ANTHROPIC_API_KEY=sk-ant-... emergency-routing

# ---------- builder ----------------------------------------------------------
FROM python:3.11-slim AS builder

# build-essential is needed by any package without a prebuilt wheel for this
# platform. It is ~200 MB and stays in this stage only.
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install into a self-contained prefix that is easy to copy out wholesale.
COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt


# ---------- runtime ----------------------------------------------------------
FROM python:3.11-slim

# libglib2.0-0 is the one system library OpenCV still needs at runtime. The
# headless OpenCV build drops the GUI dependencies (libgl1 and friends) that
# the desktop build pulls in, which the container has no use for.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --shell /bin/bash app

COPY --from=builder /install /usr/local

WORKDIR /app

# Application code, the ONNX model, the pre-computed camera scores and the
# cached OSM street graph. Everything the app needs to start with no network
# access.
COPY routing/ routing/
COPY web/ web/
COPY weights/yolov12m.onnx weights/
COPY data/ data/
COPY *.py ./
COPY manhattan_cameras.csv camera_stats.csv ./

# Run as a non-root user. Nothing here needs root, and a container process
# that cannot write outside its own files is one less thing to worry about.
RUN chown -R app:app /app
USER app

# PORT is read at runtime so the same image works on any host: Cloud Run and
# Fly set $PORT, Hugging Face Spaces routes to app_port in its README (8000
# here), and it defaults to 8000 locally.
ENV PORT=8000
# A writable HOME for the non-root user, so any library that writes a cache
# (osmnx, matplotlib) does not fail on hosts like HF Spaces.
ENV HOME=/tmp
EXPOSE 8000

# Lets Docker and orchestrators tell "starting" apart from "wedged".
HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
    CMD python -c "import os,urllib.request,sys; p=os.environ.get('PORT','8000'); \
sys.exit(0 if urllib.request.urlopen(f'http://localhost:{p}/api/health', timeout=4).status==200 else 1)"

# FastAPI app (app.py). Shell form so ${PORT} expands at runtime.
CMD uvicorn app:app --host 0.0.0.0 --port ${PORT:-8000}
