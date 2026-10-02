# Phase 2A: does the congestion signal actually track measured speed, and which
# feature does it best? This is the analysis that decides whether a learned
# congestion model is worth fitting - run on the data the hourly logger has
# accumulated (congestion_log.csv).
#
# It answers three questions honestly:
#   1. How well does each feature correlate with measured speed (Pearson +
#      Spearman)? Negative is expected: more congestion, less speed.
#   2. Do the Phase 1 flow features (queue, distinct vehicles, pixel speed) track
#      speed better than the raw count score alone - the open hypothesis from the
#      earlier go/no-go check?
#   3. Does a simple linear fit on all features beat the single-feature baseline
#      (measured by leave-one-out R^2, which is honest on a small sample)?
#
# The sample is small, so this is a direction-finder, not a finished model. It
# prints the data size up front and refuses to overclaim.
#
# Run:  python -m routing.congestion_fit

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

LOG = "congestion_log.csv"
FEATURES = ["score", "queue_length", "unique_vehicles", "mean_density", "mean_speed_px"]
TARGET = "speed_mph"


def _loo_r2(X: np.ndarray, y: np.ndarray) -> float:
    """Leave-one-out R^2 for an ordinary least-squares fit with intercept.

    Honest on small samples: each point is predicted by a model trained on all
    the others, so it cannot reward overfitting. R^2 <= 0 means the fit is no
    better than predicting the mean.
    """
    n = len(y)
    if n < 5:
        return float("nan")
    Xi = np.column_stack([X, np.ones(n)])
    preds = np.empty(n)
    for i in range(n):
        mask = np.arange(n) != i
        beta, *_ = np.linalg.lstsq(Xi[mask], y[mask], rcond=None)
        preds[i] = Xi[i] @ beta
    ss_res = float(np.sum((y - preds) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")


def main() -> None:
    df = pd.read_csv(LOG)
    df = df[df[TARGET] > 0].copy()  # 0 mph means "sensor no reading", not stopped
    n = len(df)
    slices = df["timestamp_utc"].nunique()
    print(f"Data: {n} rows, {slices} time-slices, {df['camera_id'].nunique()} cameras.")
    if n < 20:
        print("WARNING: very small sample - read everything below as a direction,\n"
              "not a trustworthy model. Keep the logger running to strengthen it.")

    y = df[TARGET].to_numpy(float)

    print("\n1. Correlation of each feature with measured speed "
          "(expect negative):")
    print(f"   {'feature':16} {'Pearson':>9} {'Spearman':>9} {'p(spear)':>9}")
    usable_feats = []
    for f in FEATURES:
        x = df[f].to_numpy(float)
        if np.allclose(x, x[0]):
            print(f"   {f:16} {'(constant - no signal)':>29}")
            continue
        pr, _ = stats.pearsonr(x, y)
        sr, sp = stats.spearmanr(x, y)
        print(f"   {f:16} {pr:+9.2f} {sr:+9.2f} {sp:9.3f}")
        usable_feats.append(f)

    # 2. Flow features vs the raw count score.
    print("\n2. Raw count score vs Phase-1 flow features (leave-one-out R^2):")
    score_r2 = _loo_r2(df[["score"]].to_numpy(float), y)
    print(f"   score alone                 : R^2 = {score_r2:+.2f}")
    flow = [f for f in ("queue_length", "unique_vehicles", "mean_speed_px") if f in usable_feats]
    if flow:
        flow_r2 = _loo_r2(df[flow].to_numpy(float), y)
        print(f"   flow features {str(flow):27}: R^2 = {flow_r2:+.2f}")
    else:
        print("   flow features are constant/zero in this log - they carry no signal")
        print("   here (the still-image tracker rarely confirms tracks). Noted.")

    # 3. All usable features together.
    if len(usable_feats) >= 2:
        all_r2 = _loo_r2(df[usable_feats].to_numpy(float), y)
        print(f"\n3. All usable features together : R^2 = {all_r2:+.2f}")

    print("\nVerdict:")
    best = score_r2 if score_r2 is not None else float("nan")
    if np.isnan(best) or best <= 0.05:
        print("  No feature yet predicts speed better than the mean on this sample.")
        print("  Do NOT fit a learned model on this - keep collecting. The congestion")
        print("  weights stay as they are until the data earns a change.")
    else:
        print("  There is a usable signal; a learned/recalibrated scorer is justified.")
        print("  Next: fit on more data and validate before replacing the heuristic.")


if __name__ == "__main__":
    main()
