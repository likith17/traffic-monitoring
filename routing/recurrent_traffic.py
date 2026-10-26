# Phase 4, step 2: congestion with a recurring, forecastable structure.
#
# Step 1 (dynamic_traffic.py + dynamic_benchmark.py) made congestion change while
# you drive, but it scattered the jams at random. Random jams cannot be
# anticipated: there is nothing to learn from history, so a planner that knows
# "the forecast" has no edge over one that simply reacts to what it sees. That is
# why step 1 came back honest-but-flat - reactive replanning captured most of what
# was winnable, and a predictive method had no structure to exploit.
#
# Real city traffic is not random. It recurs: the same corridors clog at the same
# times of day, every day, with some variation. That recurrence is exactly what a
# forecast is: the expected conditions learned from history. This module builds
# that structure so the central Phase 4 question becomes answerable:
#
#   Once congestion is predictable, does routing against the forecast beat
#   reacting to current conditions - and how much of the gap to a perfect-
#   foresight oracle does it close?
#
# The design separates what is knowable from what is not:
#   - A fixed RECURRING PATTERN (the same corridors and peak times every episode).
#     This is the "history" a forecast is built from, so a predictive planner is
#     allowed to know it exactly.
#   - A per-episode REALIZATION of that pattern: each recurring jam is jittered in
#     timing and magnitude, and a few unforecastable SURPRISE incidents are added.
#     This is the traffic the vehicle actually drives through.
#
# A predictive planner routes on the pattern (the forecast); the oracle routes on
# the realization (perfect foresight); the gap between them is precisely the
# unpredictable part. That makes the result interpretable rather than a single
# opaque number.

from __future__ import annotations

import random
from dataclasses import dataclass

import networkx as nx

from routing.dynamic_traffic import CongestionEvent, random_events


@dataclass
class RecurrentCorridor:
    """A jam that forms in the same place at the same time of day, every day.

    This is the stable, forecastable unit. `expected_event` is what history says
    will happen (the forecast); `realize` is one noisy day's version of it (what
    actually happens). build_frac/fade of the surrounding window are fixed so the
    shape is the same triangular build-and-fade as a one-off event.
    """

    center: tuple
    t_peak: float          # scheduled peak time (the daily rhythm)
    peak_points: float     # typical severity at the centre
    radius_m: float
    window_s: float        # total active duration, centred on the peak

    def _event(self, t_peak: float, peak_points: float, radius_m: float) -> CongestionEvent:
        half = self.window_s / 2.0
        return CongestionEvent(
            center=self.center,
            t_start=t_peak - half,
            t_peak=t_peak,
            t_end=t_peak + half,
            peak_points=peak_points,
            radius_m=radius_m,
        )

    def expected_event(self) -> CongestionEvent:
        """The forecast: the corridor at its scheduled time and typical severity."""
        return self._event(self.t_peak, self.peak_points, self.radius_m)

    def realize(
        self, rng: random.Random, timing_jitter_s: float, mag_jitter: float
    ) -> CongestionEvent:
        """One day's actual version: the same jam, shifted and scaled a little.

        timing_jitter_s shifts the peak earlier/later; mag_jitter scales the
        severity. The jam is still clearly the same recurring one - that is what
        makes the forecast useful - but not identical, so perfect foresight still
        beats the forecast by the size of the noise.
        """
        t_peak = max(0.0, self.t_peak + rng.uniform(-timing_jitter_s, timing_jitter_s))
        peak = self.peak_points * (1.0 + rng.uniform(-mag_jitter, mag_jitter))
        return self._event(t_peak, max(0.0, peak), self.radius_m)


def recurring_pattern(
    g: nx.DiGraph,
    n_corridors: int = 10,
    horizon_s: float = 6000.0,
    peak_points: float = 60.0,
    radius_m: float = 3500.0,
    seed: int = 20260101,
) -> list[RecurrentCorridor]:
    """The city's stable congestion rhythm: fixed across episodes.

    Defaults are calibrated to the coarse city-wide routing grid, where adjacent
    intersections sit ~2 km apart and a cross-town trip runs ~80 minutes: the
    radius spans a few blocks, the horizon covers a whole trip, and the peaks are
    spread so that most corridors are QUIET at departure and bloom mid-trip. That
    last point is what makes the forecast matter - a jam already present at
    departure is handled by plain A*; the forecast only earns its keep on jams
    that are not there yet when you leave but will be when you arrive.

    Centres are drawn from interior intersections (not camera nodes, matching the
    benchmark's origin/destination pool). Seeded with a FIXED default so the
    pattern is the same every episode - it is the history, not the weather of one
    day. Spreading peaks deterministically keeps the rhythm genuinely learnable.
    """
    rng = random.Random(seed)
    interior = [n for n in g.nodes if "cam_id" not in g.nodes[n]]
    centers = rng.sample(interior, min(n_corridors, len(interior)))
    corridors: list[RecurrentCorridor] = []
    n = len(centers)
    for i, c in enumerate(centers):
        # Peaks march from 15% to 90% of the horizon, so early corridors bloom
        # while a vehicle is still near the start and late ones bloom only once it
        # has driven well into the city.
        frac = 0.15 + 0.75 * (i / max(n - 1, 1))
        corridors.append(RecurrentCorridor(
            center=c,
            t_peak=horizon_s * frac,
            peak_points=peak_points * rng.uniform(0.8, 1.2),
            radius_m=radius_m * rng.uniform(0.85, 1.15),
            window_s=horizon_s * 0.30,
        ))
    return corridors


def forecast_events(pattern: list[RecurrentCorridor]) -> list[CongestionEvent]:
    """The forecast a predictive planner routes on: every corridor at expectation."""
    return [c.expected_event() for c in pattern]


def realize_events(
    pattern: list[RecurrentCorridor],
    rng: random.Random,
    *,
    timing_jitter_s: float = 300.0,
    mag_jitter: float = 0.35,
    n_surprises: int = 3,
    surprise_points: float = 50.0,
    surprise_radius_m: float = 3000.0,
    horizon_s: float = 6000.0,
    g: nx.DiGraph | None = None,
) -> list[CongestionEvent]:
    """One episode's real congestion: the jittered pattern plus surprise incidents.

    The jittered recurring jams are forecastable to within the noise; the surprise
    incidents are not in the pattern at all, so no forecast can see them - only
    reacting (adaptive) or perfect foresight (oracle) can. Their presence is what
    keeps a realistic gap open above the predictive planner.
    """
    events = [c.realize(rng, timing_jitter_s, mag_jitter) for c in pattern]
    if n_surprises and g is not None:
        events += random_events(
            g, rng, n_surprises, horizon_s,
            peak_points=surprise_points, radius_m=surprise_radius_m,
        )
    return events


if __name__ == "__main__":
    # Self-test: the forecast and a realization describe the SAME jams (the
    # predictive planner's knowledge is sound), but a realization is not identical
    # to the forecast (so perfect foresight can still do better).
    from routing.graph import build_default_graph

    g = build_default_graph()
    pattern = recurring_pattern(g, n_corridors=5)
    assert len(pattern) == 5

    # The pattern is stable: rebuilding it gives the same corridors.
    again = recurring_pattern(g, n_corridors=5)
    assert [c.center for c in pattern] == [c.center for c in again], "pattern must be fixed"

    fc = forecast_events(pattern)
    assert len(fc) == 5 and all(e.peak_points > 0 for e in fc)

    rng = random.Random(7)
    actual = realize_events(pattern, rng, n_surprises=2, g=g)
    assert len(actual) == 7, "5 recurring + 2 surprise events"

    # Each recurring jam is centred where the forecast said, but shifted in time.
    shifted = sum(1 for e, c in zip(actual, pattern) if abs(e.t_peak - c.t_peak) > 1e-6)
    assert shifted >= 1, "realization should jitter at least one peak off schedule"
    assert all(e.center == c.center for e, c in zip(actual[:5], pattern)), \
        "recurring jams keep their place; only timing/magnitude vary"

    print(f"Pattern corridors : {len(pattern)}  (centres fixed across episodes)")
    print(f"Forecast events   : {len(fc)}")
    print(f"Realized events   : {len(actual)}  (incl. 2 unforecastable surprises)")
    print("recurrent_traffic.py self-test OK")
