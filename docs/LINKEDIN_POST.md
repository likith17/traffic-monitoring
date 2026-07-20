# LinkedIn post — Emergency Routing for Smart Response

Copy below into LinkedIn. Attach images from `docs/assets/` in this order:
1. `01-landing.png` (or `demo.gif` as the first media if LinkedIn accepts GIFs)
2. `02-route-compare.png`
3. `03-route-map.png`
4. `06-yolo-live.png`

Put the repo link in the **first comment** (LinkedIn suppresses reach on posts with external links) — block at the bottom.

---

## Post (ready to paste)

This is my first post here, so a small confession to start.

I've spent the last couple of years mostly consuming — courses, papers, other people's project write-ups. Somewhere along the way I realized I could explain how production ML systems work without ever having shipped one end to end. That gap bothered me. So this year I'm flipping it: less consuming, more creating. This post is the first receipt.

I built **Emergency Routing for Smart Response** — a Manhattan emergency router that can *see* the road ahead.

Google Maps is great at typical traffic. But it cannot see a blocked intersection on a live traffic camera. For emergency dispatch, that blind spot costs minutes.

What it does:
• Plans on real OpenStreetMap streets, not a toy grid
• Scores 373 live NYC DOT cameras with YOLOv12
• Looks ahead at the cameras on the route before the vehicle reaches them
• Recalculates mid-drive when a camera reports a blockage — the same UX as a wrong-turn reroute, driven by vision instead of GPS probes
• Runs A*, Dijkstra, and corridor Q-learning on every request and shows the fastest arrival
• Overlays OSRM / Google Directions and re-scores *their* path in *our* congestion model, so the comparison is honest

Measured over 30 simulated emergencies with hidden road blockages:

Static shortest path → 28.1 min mean response
A* + vision gate → 22.5 min (−19.9%)
Corridor Q-learning + vision → 22.9 min (−18.7%)

Honestly, the part that taught me the most wasn't the ML. It was the constraints:

→ Tabular RL can't explore a 10k-node city graph in demo time, so the agent trains only in a corridor around the A* spine
→ Synthetic grids look fine until a route cuts straight through the Hudson — so: real OSM street graph, cached
→ Nominatim rate-limits type-ahead search, so local landmark suggestions come first
→ A benchmark should measure what you ship — the RL in the benchmark is the exact corridor RL in the app

Stack: YOLOv12 · NetworkX / OSMnx · Streamlit + Folium · Docker. Repo in the comments.

If you build ML systems that have to survive contact with the real world, I'd genuinely love to compare notes. And if you're earlier in the journey than me: build the thing. It teaches faster than watching ever did.

#MachineLearning #ComputerVision #ReinforcementLearning #MLOps #OpenStreetMap

---

## First comment (paste after posting)

Repo + screenshots + demo GIF:
https://github.com/likith17/traffic-monitoring

The four decisions I'd defend in an interview:
1. Fixed street topology, live conditions as edge-weight multipliers (the pattern production navigators use)
2. Corridor Q-learning instead of city-wide RL (tractability beats purity in a live demo)
3. En-route recalculation from the vehicle's current node, driven by camera vision
4. Never lie with baselines — external routes are re-scored inside our own congestion model

---

## Optional shorter variant (if you prefer a punchier post)

First post here. New rule for myself: stop consuming, start creating.

First receipt: a Manhattan emergency router that **sees** the road ahead.

YOLOv12 on 373 DOT cameras → congestion weights on a real OSM street graph → A* / Dijkstra / corridor RL race on every request → mid-drive recalculation when a camera reports a blockage → honest side-by-side vs OSRM/Google.

Result over 30 blocked-road simulations: **~20% faster** than static shortest path (28.1 → 22.5 min).

The hard parts were constraints — RL latency, rate-limited geocoding, honest baselines — not the YOLO call.

Repo in the comments.

---

## Posting checklist

- [ ] Attach 3–4 images (landing, metrics, dual-route map, YOLO frame) **or** `demo.gif`
- [ ] Repo link in the FIRST COMMENT, not the post body
- [ ] Confirm the GitHub URL shows the new README (push main first)
- [ ] 5–8 hashtags max — no hashtag walls
- [ ] Do **not** lead with "achieved X% accuracy" — lead with problem + constraint + impact
