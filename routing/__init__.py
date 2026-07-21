# The routing package: everything needed to plan emergency routes on a
# congestion-aware road graph built from the Manhattan camera data.
#
# Modules:
#   detect      - YOLOv12 inference via ONNX Runtime (no PyTorch at runtime)
#   streets     - real OSM drive network, cached as GraphML
#   graph       - builds the road network and attaches camera congestion
#   planners    - classical Dijkstra / A* shortest-path planners
#   rl_agent    - tabular Q-learning agent trained under traffic uncertainty
#   vision_gate - YOLO check that a route is actually passable before dispatch
#   explain     - plain-English route justification (LLM with offline fallback)
#   geo         - path -> lat/lon polylines for public map rendering
#   map_view    - Folium Google-Maps-like dual-route visualisation
#   external_route - Google Directions / OSRM comparison baseline
#   simulate    - benchmark of the full pipeline vs a static baseline
