"""三种基线:等权(1/N)、纯多头 TopN、静态因子等权。均含成本、含 t+1 结算。"""
from __future__ import annotations
import numpy as np
import pandas as pd

from scripts.config import FACTOR_NAMES, K
from scripts.costs import total_costs
from scripts.portfolio import composite_score, select_long_short, target_weights, freeze_suspended


def _iter_dates(features_df, start, end):
    df = features_df.copy()
    df["trade_date"] = pd.to_datetime(df["trade_date"])
    df = df[(df["trade_date"] >= pd.Timestamp(start)) & (df["trade_date"] <= pd.Timestamp(end))]
    df = df.sort_values(["trade_date", "symbol"]).reset_index(drop=True)
    return df, sorted(df["trade_date"].unique())


def _panel(df):
    symbols = sorted(df["symbol"].unique())
    idx_map = {s: i for i, s in enumerate(symbols)}
    n = len(symbols)
    F_by, ret_by, susp_by = {}, {}, {}
    for d, g in df.groupby("trade_date"):
        F = np.zeros((n, K)); r = np.zeros(n); s = np.ones(n, dtype=bool)
        for _, row in g.iterrows():
            i = idx_map[row["symbol"]]
            F[i] = [row[fn] for fn in FACTOR_NAMES]
            r[i] = 0.0 if pd.isna(row["ret_1d"]) else float(row["ret_1d"])
            s[i] = bool(row["is_suspended"])
        F_by[d] = F; ret_by[d] = r; susp_by[d] = s
    return symbols, n, F_by, ret_by, susp_by


def _step_returns(prev_w, target_w, ret_next, cfg):
    costs = total_costs(prev_w, target_w, cfg)
    gross = float(target_w @ ret_next)
    return gross - costs["total"]


def equal_weight_rollout(features_df, cfg, start, end) -> np.ndarray:
    df, dates = _iter_dates(features_df, start, end)
    symbols, n, F_by, ret_by, susp_by = _panel(df)
    prev = np.zeros(n)
    out = []
    for i in range(len(dates) - 1):
        susp = susp_by[dates[i]]
        active = ~susp
        w = np.zeros(n)
        if active.sum() > 0:
            w[active] = 1.0 / active.sum()
        w = freeze_suspended(w, prev, susp)
        r_next = ret_by[dates[i + 1]]
        out.append(_step_returns(prev, w, r_next, cfg))
        prev = w
    return np.asarray(out)


def long_only_topn_rollout(features_df, cfg, start, end, static_factor_w: np.ndarray) -> np.ndarray:
    df, dates = _iter_dates(features_df, start, end)
    symbols, n, F_by, ret_by, susp_by = _panel(df)
    prev = np.zeros(n)
    out = []
    for i in range(len(dates) - 1):
        F = F_by[dates[i]]; susp = susp_by[dates[i]]
        scores = composite_score(F, static_factor_w)
        long_idx, _ = select_long_short(scores, susp, cfg["top_n"], bottom_m=0)
        w = target_weights(scores, long_idx, np.array([], dtype=int), cfg["long_notional"], 0.0)
        w = freeze_suspended(w, prev, susp)
        r_next = ret_by[dates[i + 1]]
        out.append(_step_returns(prev, w, r_next, cfg))
        prev = w
    return np.asarray(out)


def static_factor_equal_rollout(features_df, cfg, start, end) -> np.ndarray:
    static_w = np.ones(K) / K
    df, dates = _iter_dates(features_df, start, end)
    symbols, n, F_by, ret_by, susp_by = _panel(df)
    prev = np.zeros(n)
    out = []
    for i in range(len(dates) - 1):
        F = F_by[dates[i]]; susp = susp_by[dates[i]]
        scores = composite_score(F, static_w)
        long_idx, short_idx = select_long_short(scores, susp, cfg["top_n"], cfg["bottom_m"])
        w = target_weights(scores, long_idx, short_idx, cfg["long_notional"], cfg["short_notional_cap"])
        w = freeze_suspended(w, prev, susp)
        r_next = ret_by[dates[i + 1]]
        out.append(_step_returns(prev, w, r_next, cfg))
        prev = w
    return np.asarray(out)
