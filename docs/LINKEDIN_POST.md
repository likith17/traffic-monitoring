# LinkedIn post — Emergency Routing for Smart Response

LinkedIn caps a post at 3,000 characters. The version below is 2,912, so it fits
with room to spare. The longer draft is kept at the bottom for a blog post or a
portfolio page, where there is no limit.

Media, in this order: `01-landing.png`, `02-route-compare.png`, `03-route-map.png`,
`06-yolo-live.png`. Or lead with `demo.gif`.

Repo link goes in the **first comment**, not the post body. LinkedIn reduces reach
on posts containing external links.

---

## Post (2,912 characters, ready to paste)

Do you ever get the feeling that you have consumed enough, and it is time you actually built something?

That hit me a few months ago. For two years it was courses, papers, other people's write-ups. Then I realised I could explain how a production ML system works without ever having shipped one. So I stopped reading about it and built this.

Emergency Routing for Smart Response. A Manhattan emergency router that checks live traffic cameras before sending a vehicle down a street.

Navigation apps optimise for typical traffic. They cannot see that the intersection two blocks ahead is blocked right now. New York runs hundreds of public cameras showing exactly that, and nothing in dispatch looks at them.

So I connected them. YOLOv12 scores 370 live NYC DOT feeds. Those scores become travel time weights on the real OpenStreetMap network (10,893 intersections). The router refuses to enter an intersection a camera says is blocked, and recalculates mid drive when one turns out to be.

Over 30 simulated emergencies, with blockages the planner cannot see in advance:

Static shortest path: 27.0 min
A* plus camera check: 21.4 min (20.8% faster)

Almost none of the hard work was algorithms. It was constraints.

Tabular RL never converged on a 10,000 node graph, so I trained it only inside a corridor around the A* route. My Docker image was 3.27 GB, and measuring it (instead of guessing) showed PyTorch was 728 MB of that, for a container that only runs the model forward. Moving to ONNX Runtime got it to 1.5 GB.

Worst bug: my own benchmark was measuring code that does not ship. Real numbers, different system.

On the AI part, since "used AI" has stopped meaning anything.

I built this with Claude Code, mostly Opus 4.8 and Fable 5. It wrote much of the implementation while I set direction and decided what counted as done. Weeks of work became days.

It was also wrong often. It called a function load_street_graph when my code said build_street_graph. It set a test threshold at 10%, and the result came back at exactly 10.0%, which is not a pass, it is a coincidence on the line. It blamed my slow search on a CSV read that measured 3 ms. It mixed TOML syntax into PowerShell commands, so I pasted a line that could not run.

Each one got caught by running it, measuring it, or reading the output instead of believing the explanation.

So the takeaway is not that AI writes your code now. It is that an LLM states wrong answers in exactly the same confident tone as right ones. You cannot hear the difference. You have to verify. It made me much faster, not less responsible for whether the work is correct.

Repo in the comments. The README documents every decision, including the ones I reversed.

If you build ML systems that meet the real world, I would like to compare notes. And if you are earlier on than me: just build the thing.

#MachineLearning #ComputerVision #MLOps #OpenStreetMap

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
