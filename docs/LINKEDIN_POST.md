# LinkedIn post — Emergency Routing for Smart Response

Media, in this order: `01-landing.png`, `02-route-compare.png`, `03-route-map.png`, `06-yolo-live.png` — or lead with `demo.gif` if you prefer motion.

Put the repo link in the **first comment**, not the post body. LinkedIn suppresses reach on posts containing external links.

---

## Post (ready to paste)

This is my first post here, and I want to start it honestly.

For the last couple of years I've mostly been consuming — courses, papers, other people's write-ups. At some point I realised I could describe how a production ML system works without ever having shipped one end to end. That gap bothered me enough to do something about it. So: less consuming, more building. This is the first thing I have to show for it.

**Emergency Routing for Smart Response** — a Manhattan emergency router that checks live traffic cameras before it commits to a street.

Ordinary navigation optimises for *typical* traffic. It cannot see that the intersection two blocks ahead is blocked right now. New York already runs hundreds of public traffic cameras showing exactly that, and nothing in the dispatch loop looks at them.

So I wired them in. YOLOv12 scores ~370 live NYC DOT camera feeds, those scores become edge weights on the real OpenStreetMap street network (10,893 intersections), and the router refuses to send a vehicle through an intersection a camera says is impassable — recalculating mid-drive when one turns out to be, the same way your phone reroutes after a wrong turn.

Measured over 30 simulated emergencies with blockages the planner cannot see in advance:

Static shortest path → 27.0 min average response
A* + camera check → 21.4 min (−20.8%)
Corridor Q-learning + camera check → 21.9 min (−18.9%)

The part I did not expect to be the lesson: almost everything interesting was a constraint, not an algorithm.

→ Tabular RL cannot explore a 10,000-node city graph in the seconds a demo has. It never converged. Fix: train only in a corridor around the A* route, which is roughly what hierarchical routing engines do.

→ My Docker image was 3.27 GB. I measured before optimising instead of guessing, and found PyTorch was 728 MB of it — for a container that only ever runs forward passes. Exporting the model to ONNX and running inference on ONNX Runtime, plus a multi-stage build, took it to 1.5 GB.

→ That swap meant rewriting the model's pre- and post-processing by hand, which could have been wrong in a way nothing crashes on — it would just quietly shift every congestion score. So I wrote a test that feeds both backends identical input and compares raw output tensors. They agree to 0.0012. That test exists because I did not trust myself, and I think that instinct is the actual skill.

→ My own benchmark was measuring code that doesn't ship. It trained full-graph RL while the app used the corridor version. The numbers were real and described a different system. Caught it by reading the two files side by side.

→ Type-ahead search took 6.96 seconds to type one word. I assumed the CSV re-read was to blame, measured it, and it was 3 ms. The real cause was a rate limiter calling sleep() on the UI thread. Same input now takes 0.01 s. Measure before you optimise — I got that wrong and the measurement corrected me.

One thing I've left as a loss: the RL agent is slightly *behind* A* here, and I said so in the README rather than tuning until it won. On a static snapshot A* is provably optimal, so there's nothing to beat. A real result you can explain beats a better number you can't.

I built this with Claude as a pair — and I'll be specific rather than vague about it, because "used AI" has stopped meaning anything. It wrote a lot of the implementation while I drove the direction, and the real gain was less in typing speed than in the debugging: it was the one that insisted on measuring the image before optimising it, that flagged my Dockerfile was about to bake an API key into the layers, and that worked out the "virtualization not supported" error was a missing Windows feature rather than a BIOS setting. Weeks of work compressed into days. The judgement calls were still mine to make; I just got to make more of them per day.

Stack: YOLOv12 · ONNX Runtime · NetworkX / OSMnx · Streamlit · Folium · Docker

Repo in the comments — the README documents every design decision, including the ones I reversed and why.

If you build ML systems that have to survive contact with the real world, I'd genuinely like to compare notes. And if you're earlier on than me: build the thing. It taught me more in a few weeks than the previous year of reading did.

#MachineLearning #ComputerVision #ReinforcementLearning #MLOps #OpenStreetMap

---

## First comment (paste immediately after posting)

Repo, screenshots and the full write-up:
https://github.com/likith17/traffic-monitoring

Four decisions I'd defend in an interview:

1. Fixed street topology with live conditions as edge-weight multipliers — the pattern production navigators use, because rebuilding a 10k-node graph per request would dominate response time.
2. Corridor Q-learning instead of city-wide RL — tractability beats purity when the demo has to answer in seconds.
3. ONNX Runtime instead of PyTorch in the container — inference doesn't need autograd, and 728 MB is a lot to pay for machinery you never call.
4. Never lie with baselines — Google/OSRM routes are re-scored inside my own congestion model, because their ETA answers a different question than mine.

---

## Shorter variant (if you want a punchier post)

First post here. New rule: stop consuming, start building.

First result — a Manhattan emergency router that **sees** the road ahead.

YOLOv12 on ~370 live traffic cameras → congestion weights on the real OSM street graph → A* / Dijkstra / corridor RL race on every request → mid-drive recalculation when a camera reports a blockage → honest side-by-side against OSRM.

Over 30 simulated emergencies: **27.0 → 21.4 min, 20.8% faster** than static shortest-path dispatch.

Hardest parts weren't the models. They were constraints: RL that won't converge on a 10k-node graph, a 3.27 GB Docker image that turned out to be 728 MB of PyTorch doing nothing, and my own benchmark quietly measuring code that doesn't ship.

Built with Claude as a pair programmer — it caught the Dockerfile about to bake my API key into the image, which is the kind of mistake you only notice once it's public.

Repo in the comments.

---

## Before posting

- [ ] Repo link in the FIRST COMMENT, not the post body
- [ ] 3–4 images attached, or `demo.gif` as the lead
- [ ] Confirm the GitHub page shows the current README
- [ ] 5 hashtags is enough — no walls
- [ ] Do not lead with "achieved X% accuracy". Lead with problem → constraint → measured impact
