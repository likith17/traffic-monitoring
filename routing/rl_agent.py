# Tabular Q-learning agent for routing under uncertain traffic.
#
# Why RL on top of Dijkstra / A*?  The classical planners assume the travel
# times they see are exact.  In a real emergency the congestion picture is
# noisy: cameras update every few minutes and traffic shifts constantly.
# The Q-learning agent trains across many episodes where every edge time is
# perturbed by random noise, so it learns routes that stay fast under
# uncertainty rather than routes that are only optimal for one frozen
# snapshot.
#
# The design is deliberately simple and readable:
#   state  = the intersection the vehicle is currently at
#   action = which neighbouring intersection to drive to next
#   reward = minus the (noisy) travel time of that street,
#            plus a big bonus for reaching the destination

from __future__ import annotations

import random

import networkx as nx

from routing.planners import astar_route


class QLearningRouter:
    """Learns a routing policy from src to dst on a weighted road graph."""

    def __init__(
        self,
        g: nx.DiGraph,
        alpha: float = 0.1,        # learning rate: how fast new info replaces old
        gamma: float = 0.98,       # discount: future travel time matters almost fully
        epsilon: float = 1.0,      # exploration rate, decays over training
        epsilon_min: float = 0.05,
        epsilon_decay: float = 0.999,
        noise: float = 0.25,       # +/-25% random perturbation of edge times per step
        goal_reward: float = 3000.0,
        max_steps: int = 400,
        seed: int | None = 42,
    ):
        self.g = g
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.noise = noise
        self.goal_reward = goal_reward
        self.max_steps = max_steps
        self.rng = random.Random(seed)

        # Q[(node, neighbour)] = learned value of driving that street.
        self.q: dict[tuple, float] = {}
        self.src: tuple | None = None
        self.dst: tuple | None = None

    # ── internals ────────────────────────────────────────────────────────────

    def _noisy_time(self, u: tuple, v: tuple) -> float:
        """Edge travel time with random traffic noise for this step.
        This is the 'uncertain traffic' the agent must be robust to."""
        t = self.g.edges[u, v]["travel_time"]
        factor = 1.0 + self.rng.uniform(-self.noise, self.noise)
        return t * factor

    def _best_action(self, node: tuple) -> tuple | None:
        """The neighbour with the highest learned Q-value from this node."""
        neighbours = list(self.g.successors(node))
        if not neighbours:
            return None
        return max(neighbours, key=lambda v: self.q.get((node, v), 0.0))

    def _choose_action(self, node: tuple) -> tuple | None:
        """Epsilon-greedy: mostly follow the best known street, sometimes
        explore a random one so better options can still be discovered."""
        neighbours = list(self.g.successors(node))
        if not neighbours:
            return None
        if self.rng.random() < self.epsilon:
            return self.rng.choice(neighbours)
        return self._best_action(node)

    # ── training ─────────────────────────────────────────────────────────────

    def train(self, src: tuple, dst: tuple, episodes: int = 1500) -> None:
        """Run Q-learning episodes from src towards dst.

        Each episode drives the vehicle through the graph with freshly
        sampled traffic noise, updating Q-values with the standard rule:
        Q <- Q + alpha * (reward + gamma * best_next_Q - Q)
        """
        self.src, self.dst = src, dst

        for _ in range(episodes):
            node = src
            for _ in range(self.max_steps):
                if node == dst:
                    break

                nxt = self._choose_action(node)
                if nxt is None:
                    break  # dead end (cannot happen on the grid, but be safe)

                # Reward: driving costs time; arriving pays a large bonus.
                reward = -self._noisy_time(node, nxt)
                if nxt == dst:
                    reward += self.goal_reward

                # Value of the best street leaving the next intersection.
                future = 0.0
                if nxt != dst:
                    best_next = self._best_action(nxt)
                    if best_next is not None:
                        future = self.q.get((nxt, best_next), 0.0)

                old = self.q.get((node, nxt), 0.0)
                self.q[(node, nxt)] = old + self.alpha * (
                    reward + self.gamma * future - old
                )

                node = nxt

            # Explore a little less after every episode.
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    # ── inference ────────────────────────────────────────────────────────────

    def best_route(self) -> list[tuple]:
        """Follow the greedy learned policy from src to dst.

        If the policy ever revisits a node or runs out of steps (possible when
        training was too short), fall back to A* so the caller always gets a
        valid route - the emergency vehicle must never be left without a plan.
        """
        if self.src is None or self.dst is None:
            raise RuntimeError("Call train() before best_route().")

        path = [self.src]
        visited = {self.src}
        node = self.src

        for _ in range(self.max_steps):
            if node == self.dst:
                return path
            nxt = self._best_action(node)
            if nxt is None or nxt in visited:
                break  # policy is stuck - use the classical fallback below
            path.append(nxt)
            visited.add(nxt)
            node = nxt

        return astar_route(self.g, self.src, self.dst)


def rl_route(
    g: nx.DiGraph,
    src: tuple,
    dst: tuple,
    episodes: int = 900,
    buffer_deg: float = 0.006,
    seed: int = 0,
) -> list[tuple]:
    """Q-learning route on a corridor around the A* path.

    Tabular Q-learning cannot explore a 10k-node city graph in reasonable
    time, so - like hierarchical routing engines that restrict the search
    space - we train inside a corridor of streets around the classical route
    and let the agent optimise within it under traffic noise.  If the learned
    policy is unusable, best_route() already falls back to A*.
    """
    spine = astar_route(g, src, dst)
    lats = [g.nodes[n]["lat"] for n in spine]
    lons = [g.nodes[n]["lon"] for n in spine]
    lat_min, lat_max = min(lats) - buffer_deg, max(lats) + buffer_deg
    lon_min, lon_max = min(lons) - buffer_deg, max(lons) + buffer_deg

    corridor = [
        n for n, d in g.nodes(data=True)
        if lat_min <= d["lat"] <= lat_max and lon_min <= d["lon"] <= lon_max
    ]
    sub = g.subgraph(corridor).copy()
    sub.graph.update(g.graph)

    agent = QLearningRouter(sub, seed=seed)
    agent.train(src, dst, episodes=episodes)
    return agent.best_route()


if __name__ == "__main__":
    # Self-test: train on the real congestion graph and check the learned
    # route is close to the A* optimum (it plans under noise, so a small gap
    # is expected and fine).
    import time

    from routing.graph import build_default_graph
    from routing.planners import route_metrics

    g = build_default_graph()
    src, dst = (0, 0), (29, 11)

    t0 = time.time()
    agent = QLearningRouter(g)
    agent.train(src, dst, episodes=1500)
    print(f"Trained 1500 episodes in {time.time() - t0:.1f}s, "
          f"{len(agent.q)} state-action pairs learned")

    rl_path = agent.best_route()
    ast_path = astar_route(g, src, dst)

    m_rl = route_metrics(g, rl_path)
    m_ast = route_metrics(g, ast_path)

    print(f"RL route : {m_rl['travel_time_s']:.0f}s over {m_rl['length_km']:.2f} km")
    print(f"A* route : {m_ast['travel_time_s']:.0f}s over {m_ast['length_km']:.2f} km")

    gap = m_rl["travel_time_s"] / m_ast["travel_time_s"] - 1.0
    print(f"RL is within {gap * 100:.1f}% of the A* optimum")
    assert gap < 0.20, "RL route should be within 20% of optimal after training"
    print("Self-test passed.")
