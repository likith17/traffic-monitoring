# Phase 4, step 4: a congestion forecast learned from the city's own history.
#
# The predictive planner has been routing against a hand-built recurring pattern.
# That was fine for proving the idea, but the real goal is to anticipate the
# city's actual rhythm, learned from data rather than assumed. The hourly camera
# logger has recorded a congestion score for each camera for weeks, which is
# exactly the history a forecast needs.
#
# This module turns that log into a time-of-day congestion profile for each
# camera: what score to expect at a given hour. A camera with few readings in some
# hour is shrunk toward the city-wide average for that hour, so a thin slice of
# data does not produce a wild estimate (empirical-Bayes style shrinkage). The
# forecast is then usable two ways:
#   LearnedForecast.score(camera_id, hour)  - the expected score, for analysis
#   LearnedTraffic(graph, forecast, start_hour) - a drop-in for DynamicTraffic so
#       the existing route planners can route against the learned forecast
#
# Before trusting it, cross_validate() checks that the per-camera hourly profile
# predicts held-out readings better than two simpler baselines (one flat average,
# and one city-wide hourly average). If it does not beat them, the learned
# forecast is not worth using, and that is worth knowing.

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from routing.graph import CONGESTION_PER_POINT, haversine_m

LOG_PATH = "congestion_log.csv"
# Approximate New York local time. The log is UTC; the exact offset shifts by an
# hour at the daylight-saving boundary, which does not matter for an hour-of-day
# congestion profile.
UTC_OFFSET_H = -5
HOURS = 24


def load_log(path: str = LOG_PATH) -> pd.DataFrame:
    """Read the logger output and add a local hour-of-day column."""
    d = pd.read_csv(path)
    t = pd.to_datetime(d["timestamp_utc"], utc=True)
    d = d.assign(hour=((t.dt.hour + UTC_OFFSET_H) % HOURS).astype(int))
    d["score"] = pd.to_numeric(d["score"], errors="coerce")
    return d.dropna(subset=["score", "camera_id", "hour"])


class LearnedForecast:
    """Expected congestion score per camera and hour, learned from the log.

    shrink_k sets how many city-wide-average readings a camera's own data must
    outweigh before its estimate is trusted on its own. With k readings of weight,
    a camera seen k times in an hour is weighted equally against the city-wide
    hourly mean, which keeps sparse slices sane.
    """

    def __init__(self, df: pd.DataFrame, shrink_k: float = 5.0):
        self.grand = float(df["score"].mean())
        self.hourly = df.groupby("hour")["score"].mean().to_dict()
        agg = df.groupby(["camera_id", "hour"])["score"].agg(["mean", "count"])
        self.k = shrink_k
        self._cam_hour: dict[tuple, tuple[float, int]] = {
            (str(cid), int(h)): (float(r["mean"]), int(r["count"]))
            for (cid, h), r in agg.iterrows()
        }

    def score(self, camera_id: str, hour: int) -> float:
        """Shrunk expected score for a camera at an hour, with sane fallbacks."""
        hour = int(hour) % HOURS
        base = self.hourly.get(hour, self.grand)   # city-wide mean for this hour
        cam = self._cam_hour.get((str(camera_id), hour))
        if cam is None:
            return base
        mean, count = cam
        return (count * mean + self.k * base) / (count + self.k)


def cross_validate(df: pd.DataFrame, folds: int = 5, shrink_k: float = 5.0,
                   seed: int = 0) -> dict:
    """Does the camera-hour profile beat simpler baselines on held-out readings?

    Reports mean absolute error for three predictors, trained on each fold's
    training split and scored on its held-out split:
      grand   - one flat average score for everything
      hourly  - the city-wide average for the reading's hour
      profile - the shrunk per-camera, per-hour estimate
    If profile does not beat hourly, the per-camera detail is not earning its keep.
    """
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(df))
    parts = np.array_split(idx, folds)
    errs = {"grand": [], "hourly": [], "profile": []}
    for i in range(folds):
        test = df.iloc[parts[i]]
        train = df.iloc[np.concatenate([parts[j] for j in range(folds) if j != i])]
        fc = LearnedForecast(train, shrink_k=shrink_k)
        grand = float(train["score"].mean())
        hourly = train.groupby("hour")["score"].mean().to_dict()
        for _, row in test.iterrows():
            y = row["score"]
            h = int(row["hour"])
            errs["grand"].append(abs(y - grand))
            errs["hourly"].append(abs(y - hourly.get(h, grand)))
            errs["profile"].append(abs(y - fc.score(str(row["camera_id"]), h)))
    return {k: float(np.mean(v)) for k, v in errs.items()}


class LearnedTraffic:
    """A time-varying congestion field driven by the learned forecast.

    Exposes the same edge_time(u, v, t) / g / _pos surface as DynamicTraffic, so
    the A* planners in dynamic_traffic.py route against it unchanged. The clock t
    is seconds after departure; it maps to an hour of day via start_hour, so a
    trip that begins at 08:00 and runs two hours sees the 08:00, 09:00 and 10:00
    profiles in turn. Each edge takes the forecast of its nearest camera.
    """

    def __init__(self, g, forecast: LearnedForecast, start_hour: float = 8.0,
                 influence_m: float = 400.0):
        self.g = g
        self.fc = forecast
        self.start_hour = float(start_hour)
        self._pos = {n: (d["lat"], d["lon"]) for n, d in g.nodes(data=True)}
        cams = [(str(d["cam_id"]), self._pos[n])
                for n, d in g.nodes(data=True) if "cam_id" in d]
        # Precompute, for each edge, the nearest camera and whether it is close
        # enough to apply (edges far from every camera just use free-flow).
        self._edge_cam: dict[tuple, str | None] = {}
        for u, v in g.edges():
            mlat = (self._pos[u][0] + self._pos[v][0]) / 2.0
            mlon = (self._pos[u][1] + self._pos[v][1]) / 2.0
            best, bestd = None, influence_m
            for cid, (clat, clon) in cams:
                d = haversine_m(mlat, mlon, clat, clon)
                if d < bestd:
                    best, bestd = cid, d
            self._edge_cam[(u, v)] = best

    def _hour_at(self, t: float) -> int:
        return int((self.start_hour + t / 3600.0)) % HOURS

    def edge_multiplier(self, u: tuple, v: tuple, t: float) -> float:
        cid = self._edge_cam.get((u, v))
        if cid is None:
            return 1.0
        return 1.0 + CONGESTION_PER_POINT * self.fc.score(cid, self._hour_at(t))

    def edge_time(self, u: tuple, v: tuple, t: float) -> float:
        return self.g.edges[u, v]["base_time"] * self.edge_multiplier(u, v, t)


def main() -> None:
    ap = argparse.ArgumentParser(description="Learn and validate a congestion forecast from the log")
    ap.add_argument("--log", default=LOG_PATH)
    ap.add_argument("--shrink", type=float, default=5.0)
    args = ap.parse_args()

    df = load_log(args.log)
    print(f"Loaded {len(df):,} readings, {df['camera_id'].nunique()} cameras, "
          f"{df['hour'].nunique()} hours represented.")

    cv = cross_validate(df, shrink_k=args.shrink)
    print("\n5-fold mean absolute error (lower is better):")
    print(f"  flat average        : {cv['grand']:.3f}")
    print(f"  city-wide hourly    : {cv['hourly']:.3f}   "
          f"({100*(1-cv['hourly']/cv['grand']):+.1f}% vs flat)")
    print(f"  per-camera hourly   : {cv['profile']:.3f}   "
          f"({100*(1-cv['profile']/cv['hourly']):+.1f}% vs city-wide hourly)")
    if cv["profile"] < cv["hourly"] < cv["grand"]:
        print("\nThe learned per-camera time-of-day forecast has skill: it predicts")
        print("held-out congestion better than either a flat or a city-wide average.")
    else:
        print("\nThe per-camera detail did not beat the simpler baseline on held-out")
        print("data, so the forecast should stay coarse until more data accumulates.")


if __name__ == "__main__":
    main()
