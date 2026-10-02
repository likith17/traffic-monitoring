---
title: Emergency Routing
emoji: 🚑
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 8000
pinned: false
license: mit
---

# Emergency Routing for Smart Response

Vision-confirmed emergency routing on live NYC traffic cameras. A FastAPI
backend loads a YOLOv12 detector (ONNX Runtime) and the real Manhattan street
graph (10,893 intersections) once at startup, scores ~370 live DOT cameras for
congestion, and plans routes with A* / Dijkstra behind a camera-based vision
gate that reroutes around blockages. The map is a Leaflet front-end.

Built by Likith Podalakuru. Source: https://github.com/likith17/traffic-monitoring

**Note:** this is a research demo. Incident detection and the "ask" features need
a vision model key set as a Space secret (`ANTHROPIC_API_KEY`, `LLM_PROVIDER=anthropic`);
without one the app runs routing and congestion with a graceful fallback.
