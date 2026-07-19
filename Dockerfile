# Container for the Emergency Routing / traffic monitoring dashboard.
#
# Build:  docker build -t emergency-routing .
# Run:    docker run -p 8501:8501 emergency-routing
# Then open http://localhost:8501
#
# The image runs on CPU (no CUDA) so it works on any Linux host.  Live YOLO
# inference is slower than on a GPU but perfectly usable for single frames;
# the offline vision mode and the benchmark need no network at all.

FROM python:3.11-slim

# git      - needed because ultralytics is installed from a git ref
# libgl1 / libglib2.0-0 - runtime libraries OpenCV needs on slim images
RUN apt-get update \
    && apt-get install -y --no-install-recommends git libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install CPU-only torch first so the pinned versions in requirements.txt
# are already satisfied and pip never downloads the multi-GB CUDA wheels.
RUN pip install --no-cache-dir torch==2.9.1 torchvision==0.24.1 \
        --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code, model weights, the pre-computed camera data and
# the cached OSM street network so the dashboard works fully offline.
COPY routing/ routing/
COPY weights/ weights/
COPY data/ data/
COPY .streamlit/ .streamlit/
COPY *.py ./
COPY manhattan_cameras.csv camera_stats.csv segment_stats.csv ./

EXPOSE 8501

# Headless mode: no browser auto-open, listen on all interfaces.
CMD ["python", "-m", "streamlit", "run", "dashboard.py", \
     "--server.headless", "true", "--server.address", "0.0.0.0", \
     "--server.port", "8501"]
