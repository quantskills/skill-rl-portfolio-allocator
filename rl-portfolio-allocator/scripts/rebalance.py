from __future__ import annotations

import numpy as np
import pandas as pd


def weekly_decision_indices(dates) -> np.ndarray:
    index = pd.DatetimeIndex(pd.to_datetime(dates))
    iso = index.isocalendar()
    keys = list(zip(iso.year.to_numpy(), iso.week.to_numpy()))
    return np.asarray([i for i, key in enumerate(keys)
                       if i == 0 or key != keys[i - 1]], dtype=int)


def buffered_long_short(scores, suspended, prev_w, long_entry, long_exit,
                        short_entry, short_exit):
    scores = np.asarray(scores, dtype=float)
    suspended = np.asarray(suspended, dtype=bool)
    prev_w = np.asarray(prev_w, dtype=float)
    eligible = np.flatnonzero(~suspended)
    ranked = eligible[np.argsort(-scores[eligible], kind="stable")]
    rank = {int(i): pos + 1 for pos, i in enumerate(ranked)}
    keep_long = [int(i) for i in eligible
                 if prev_w[i] > 0 and rank[int(i)] <= long_exit]
    keep_short = [int(i) for i in eligible
                  if prev_w[i] < 0 and rank[int(i)] > len(ranked) - short_exit]
    longs = keep_long + [int(i) for i in ranked if i not in keep_long]
    shorts = keep_short + [int(i) for i in ranked[::-1] if i not in keep_short]
    return np.asarray(longs[:long_entry + len(keep_long)], dtype=int), np.asarray(
        sorted(shorts[:short_entry + len(keep_short)]), dtype=int)


def project_turnover(prev, target, frozen, budget, long_cap, short_cap):
    previous = np.asarray(prev, dtype=float)
    desired = np.asarray(target, dtype=float).copy()
    frozen = np.asarray(frozen, dtype=bool)
    if not (np.isfinite(previous).all() and np.isfinite(desired).all()):
        raise ValueError("non-finite weight supplied to turnover projection")
    if previous.shape != desired.shape or frozen.shape != previous.shape:
        raise ValueError("weight and frozen shapes must match")
    desired[frozen] = previous[frozen]
    desired[previous * desired < 0] = 0.0
    frozen_long = np.clip(previous[frozen], 0, None).sum()
    frozen_short = np.clip(-previous[frozen], 0, None).sum()
    if frozen_long > long_cap or frozen_short > short_cap:
        raise ValueError("frozen positions already exceed notional cap")
    tradable = ~frozen
    positive = tradable & (desired > 0)
    negative = tradable & (desired < 0)
    long_available = long_cap - frozen_long
    short_available = short_cap - frozen_short
    long_sum = desired[positive].sum()
    short_sum = (-desired[negative]).sum()
    if long_sum > long_available and long_sum > 0:
        desired[positive] *= long_available / long_sum
    if short_sum > short_available and short_sum > 0:
        desired[negative] *= short_available / short_sum
    delta = desired - previous
    turnover = np.abs(delta).sum()
    if turnover > budget and turnover > 0:
        desired = previous + delta * (budget / turnover)
    if not np.isfinite(desired).all():
        raise ValueError("turnover projection produced non-finite weights")
    if np.abs(desired - previous).sum() > budget + 1e-9:
        raise ValueError("turnover projection exceeded budget")
    if np.clip(desired, 0, None).sum() > long_cap + 1e-9:
        raise ValueError("turnover projection exceeded long cap")
    if np.clip(-desired, 0, None).sum() > short_cap + 1e-9:
        raise ValueError("turnover projection exceeded short cap")
    if not np.array_equal(desired[frozen], previous[frozen]):
        raise ValueError("turnover projection changed frozen positions")
    return desired
