# LinkedIn post — Emergency Routing for Smart Response

Copy below into LinkedIn. Attach images from `docs/assets/` in this order:
1. `01-landing.png` (or the `demo.gif` as the first media if LinkedIn accepts GIFs)
2. `02-route-compare.png`
3. `03-route-map.png`
4. `06-yolo-live.png`

Suggested first comment (link + longer context): paste the “First comment” block at the bottom.

---

## Post (ready to paste)

Google Maps is great at typical traffic.

It still cannot see a blocked intersection on a live traffic camera.

I built **Emergency Routing for Smart Response** for Manhattan to close that gap:

• Plan on **real OpenStreetMap streets** (not a toy grid)
• Score **373 NYC DOT cameras** with **YOLOv12**
• **Look ahead** at cameras before the vehicle enters them
• **Recalculate mid-drive** when a camera reports a blockage — same UX as a wrong-turn reroute, driven by vision
• Run **A\*, Dijkstra, and corridor Q-learning** every time, show the fastest arrival, and put the full comparison below the map
• Overlay **OSRM / Google Directions** and re-score *their* path in *our* congestion model so the comparison is honest

**What changed when we measured it** (30 simulated emergencies with hidden blockages):

Static shortest path → **26.9 min** mean  
A\* + vision gate → **21.8 min** (−18.8%)  
Q-learning + vision → **22.5 min** (−16.4%)

The interesting part was not picking a fancier algorithm.

It was the **constraints**:

→ Full-graph tabular RL on ~10k nodes is too slow for a live demo → train Q-learning only in a corridor around the A\* spine  
→ Synthetic grids look fine until the route cuts through water → switch to a cached OSM street graph with curved geometry  
→ Nominatim rate-limits type-ahead → landmarks + camera-name suggestions first  
→ “Simulate a blockage” only makes sense against stored scores → hide it in live YOLO mode  

Stack: YOLOv12 · NetworkX / OSMnx · Streamlit + Folium · Docker.

Happy to walk through the trade-offs if you are hiring for ML systems that have to ship under real constraints — not just train a model in a notebook.

#MachineLearning #ComputerVision #ReinforcementLearning #MLOps #UrbanTech #OpenStreetMap #YOLO #Streamlit

---

## First comment (paste after posting)

Repo + screenshots + demo GIF:  
https://github.com/likith17/traffic-monitoring

Demo walkthrough (GIF in the README):  
https://github.com/likith17/traffic-monitoring#emergency-routing-for-smart-response

What I would talk about in an interview for this project:
1. Why we kept topology fixed and put congestion on edge weights  
2. Why corridor RL instead of city-wide Q-learning  
3. How en-route recalculation mirrors consumer navigation but uses vision  
4. How we avoid lying with baselines (re-score external routes in our model)

---

## Optional shorter variant (if you prefer a punchier post)

Built a Manhattan emergency router that **sees** the road ahead.

YOLOv12 on 373 DOT cameras → congestion weights on a real OSM street graph → A\* / Dijkstra / RL all run → best ETA shown → mid-drive recalculation when a camera reports a blockage → side-by-side vs OSRM/Google.

Result on 30 blocked-road simulations: **~19% faster** than static shortest path.

The hard parts were constraints (RL latency, rate-limited geocoding, honest baselines) — not the YOLO call itself.

Repo: https://github.com/likith17/traffic-monitoring

---

## Posting checklist

- [ ] Attach 3–4 images (landing, metrics, dual-route map, YOLO frame) **or** `demo.gif`
- [ ] Confirm the GitHub branch/URL in the first comment matches what you pushed
- [ ] Tag carefully — avoid spammy hashtag walls; 6–8 is enough
- [ ] Do **not** lead with “achieved X% accuracy” — lead with the problem + constraint + impact (that is what hiring managers actually read)
