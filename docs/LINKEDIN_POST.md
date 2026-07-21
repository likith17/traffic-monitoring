# LinkedIn post — Emergency Routing for Smart Response

Media, in this order: `01-landing.png`, `02-route-compare.png`, `03-route-map.png`, `06-yolo-live.png`. Or lead with `demo.gif` if you want motion.

Put the repo link in the **first comment**, not in the post body. LinkedIn reduces reach on posts that contain external links.

---

## Post (ready to paste)

This is my first post here, so let me start it honestly.

For the last couple of years I have mostly been consuming. Courses, papers, other people's write-ups. At some point I realised I could explain how a production ML system works without ever having shipped one myself. That gap bothered me. So this year the rule is simple: less consuming, more building. This is the first thing I have to show for it.

**Emergency Routing for Smart Response.** It is a Manhattan emergency router that checks live traffic cameras before it sends a vehicle down a street.

Normal navigation apps optimise for typical traffic. They cannot see that the intersection two blocks ahead is blocked right now. New York already runs hundreds of public traffic cameras that show exactly this, and nothing in the dispatch loop actually looks at them.

So I connected them. YOLOv12 scores around 370 live NYC DOT camera feeds. Those scores become travel time weights on the real OpenStreetMap street network, which is 10,893 intersections. The router then refuses to send a vehicle through an intersection that a camera says is blocked, and recalculates mid drive when one turns out to be, the same way your phone reroutes after a wrong turn.

Measured over 30 simulated emergencies, with blockages the planner cannot see in advance:

Static shortest path: 27.0 min average response
A* plus camera check: 21.4 min, 20.8% faster
Corridor Q-learning plus camera check: 21.9 min, 18.9% faster

What surprised me is that almost none of the interesting work was about algorithms. It was about constraints.

1. Tabular RL cannot explore a 10,000 node city graph in the few seconds a demo has. It simply never converged. The fix was to train only inside a corridor around the A* route, which is close to what real hierarchical routing engines do.

2. My Docker image was 3.27 GB. Instead of guessing what was big, I measured it, and PyTorch alone was 728 MB. That is a lot of weight for a container that only ever runs the model forward. Exporting to ONNX and running inference on ONNX Runtime, plus a multi stage build, brought it down to 1.5 GB.

3. That swap meant rewriting the model's pre and post processing myself. This is code that can be wrong without crashing anything. It would just quietly shift every congestion score, and then the routing, and then the benchmark. So I wrote a test that feeds both versions the exact same input and compares the raw output numbers. They match to 0.0012. That test exists because I did not trust my own code, and I now think that instinct is the actual skill.

4. My own benchmark was measuring code that does not ship. It was training full graph RL while the app used the corridor version. The numbers were real, they just described a different system. I found it by reading the two files side by side.

5. The search box took 6.96 seconds to type one word. I assumed the CSV file being re-read was the reason, measured it, and it was 3 ms. The real cause was a rate limiter calling sleep() on the UI thread. Same input now takes 0.01 seconds. Measure first. My guess was wrong and only the measurement told me so.

One result I left in as a loss: the RL agent is slightly worse than A* here, and I wrote that in the README instead of tuning it until it won. On a fixed snapshot A* is already provably optimal, so there is nothing for RL to beat. A real result I can explain is worth more than a better number I cannot.

Now the part I want to be specific about, because "used AI" has stopped meaning anything.

I built this with Claude Code as a pair programmer, mostly Opus 4.8 and Fable 5, switching models depending on the task. It wrote a large share of the implementation while I decided direction, scope and what counted as done. The speed difference was real. Work that would have taken me weeks took days.

But it was not hands off, and it was wrong often enough that I want to say so plainly.

It called a function `load_street_graph` when the actual name in my code was `build_street_graph`, and the test failed immediately. It set the pass threshold on the ONNX comparison test at 10%, then the result came back at exactly 10.0%, which is not a pass, it is a coincidence sitting on the line. It guessed the search slowness was a CSV read, and the measurement said 3 ms. It gave me setup instructions that mixed TOML file syntax with PowerShell commands, so I pasted a line that PowerShell could not run at all. It suggested a Docker proxy setting as a fix that turned out not to be the cause.

Every one of those got caught the same way: by running it, measuring it, or reading the actual output instead of believing the explanation.

So the honest takeaway is not "AI writes your code now". It is that an LLM is a fast, confident, and genuinely useful collaborator that will also state something wrong in exactly the same tone it states something right. You cannot hear the difference. You have to verify. Run the code, check the numbers, read the file yourself. The tool made me much faster at producing work, it did not make me any less responsible for whether the work is correct.

Stack: YOLOv12, ONNX Runtime, NetworkX, OSMnx, Streamlit, Folium, Docker

Repo is in the comments. The README documents every design decision, including the ones I reversed and why.

If you build ML systems that have to survive contact with the real world, I would genuinely like to compare notes. And if you are earlier on than me: just build the thing. It taught me more in a few weeks than the previous year of reading did.

#MachineLearning #ComputerVision #ReinforcementLearning #MLOps #OpenStreetMap

---

## First comment (paste right after posting)

Repo, screenshots and the full write-up:
https://github.com/likith17/traffic-monitoring

Four decisions I would defend in an interview:

1. Fixed street topology, live conditions applied as edge weight multipliers. This is the pattern real navigators use, because rebuilding a 10k node graph on every request would dominate the response time.
2. Corridor Q-learning instead of city wide RL. Tractability beats purity when the demo has to answer in seconds.
3. ONNX Runtime instead of PyTorch inside the container. Inference does not need autograd, and 728 MB is a lot to pay for machinery you never call.
4. No dishonest baselines. Google and OSRM routes get re-scored inside my own congestion model, because their ETA answers a different question than mine does.

---

## Shorter variant (if you want something punchier)

First post here. New rule: stop consuming, start building.

First result is a Manhattan emergency router that actually sees the road ahead.

YOLOv12 on around 370 live traffic cameras, feeding congestion weights into the real OpenStreetMap street graph. A*, Dijkstra and corridor RL all run on every request, and the route recalculates mid drive when a camera reports a blockage.

Over 30 simulated emergencies: 27.0 min down to 21.4 min, so 20.8% faster than static shortest path dispatch.

The hard parts were not the models. They were constraints. RL that will not converge on a 10k node graph. A 3.27 GB Docker image that turned out to be 728 MB of PyTorch doing nothing. My own benchmark quietly measuring code that does not ship.

Built with Claude Code (Opus 4.8 and Fable 5) as a pair programmer. Genuinely fast, and genuinely wrong sometimes: wrong function names, a test threshold set so loosely that a borderline result passed, a performance guess that measurement disproved. Useful tool, but verify everything it hands you.

Repo in the comments.

---

## Before posting

- [ ] Repo link goes in the FIRST COMMENT, not the post body
- [ ] Attach 3 or 4 images, or lead with `demo.gif`
- [ ] Check the GitHub page is showing the current README
- [ ] 5 hashtags is enough
- [ ] Do not open with "achieved X% accuracy". Open with problem, constraint, measured result
