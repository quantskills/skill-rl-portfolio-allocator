"""三种基线:等权(1/N)、纯多头 TopN、静态因子等权。均含成本、含 t+1 结算。"""
from __future__ import annotations
import numpy as np
import pandas as pd

from scripts.config import FACTOR_NAMES, K
from scripts.portfolio import composite_score, target_weights, freeze_suspended
from scripts.rebalance import buffered_long_short, project_turnover, weekly_decision_indices
from scripts.env import extract_settle_holding_period


def _l1_normalize(values: np.ndarray, *, zero_error: bool) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if values.shape != (K,) or not np.isfinite(values).all():
        raise ValueError("factor weights must be finite and have one entry per factor")
    norm = float(np.abs(values).sum())
    if norm == 0.0:
        if zero_error:
            raise ValueError("factor weights are all zero")
        return np.zeros(K)
    return values / norm


def fit_static_factor_weights(factor_returns: pd.DataFrame) -> np.ndarray:
    """Fit frozen factor weights using only complete training observations."""
    sample = factor_returns.loc[:, FACTOR_NAMES].replace([np.inf, -np.inf], np.nan).dropna()
    if len(sample) < 60:
        raise ValueError("at least 60 complete factor-return observations are required")
    values = sample.to_numpy(dtype=float)
    mean = values.mean(axis=0)
    covariance = np.cov(values, rowvar=False, ddof=1)
    ridge = 1e-6 * max(float(np.trace(covariance)) / K, 1.0)
    weights = np.linalg.pinv(covariance + ridge * np.eye(K)) @ mean
    return _l1_normalize(weights, zero_error=True)


def rolling_ic_weights(state_row: pd.Series) -> np.ndarray:
    """Convert the current causal state row's trailing IC means to L1 weights."""
    values = np.asarray(
        [state_row[f"{factor}_ic_mean_20"] for factor in FACTOR_NAMES], dtype=float
    )
    return _l1_normalize(values, zero_error=False)


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


def _weekly_rollout(features_df, cfg, start, end, target_for_date):
    df, dates = _iter_dates(features_df, start, end)
    symbols, n, F_by, ret_by, susp_by = _panel(df)
    decision_indices = weekly_decision_indices(dates)
    prev = np.zeros(n)
    out = []
    for position, index in enumerate(decision_indices):
        if index >= len(dates) - 1:
            continue
        d = dates[index]
        next_index = (
            decision_indices[position + 1]
            if position + 1 < len(decision_indices)
            else len(dates) - 1
        )
        F = F_by[d]
        susp = susp_by[d]
        target = target_for_date(d, F, susp, prev)
        target = freeze_suspended(target, prev, susp)
        target = project_turnover(
            prev, target, susp, cfg["turnover_budget"],
            cfg["long_notional"], cfg["short_notional_cap"],
        )
        settlement_dates = dates[index + 1 : next_index + 1]
        settled = extract_settle_holding_period(
            prev, target, settlement_dates, ret_by, cfg,
        )
        out.extend(settled["daily_net_rets"])
        prev = target
    return np.asarray(out)


def _weekly_factor_rollout(features_df, cfg, start, end, weight_for_date, short_enabled=True):
    def target_for_date(date, F, susp, prev):
        factor_w = weight_for_date(date)
        scores = composite_score(F, factor_w)
        long_idx, short_idx = buffered_long_short(
            scores, susp, prev,
            cfg["top_n"], cfg.get("long_exit", cfg["top_n"]),
            cfg["bottom_m"] if short_enabled else 0,
            cfg.get("short_exit", cfg["bottom_m"]) if short_enabled else 0,
        )
        return target_weights(
            scores, long_idx, short_idx,
            cfg["long_notional"], cfg["short_notional_cap"] if short_enabled else 0.0,
        )

    return _weekly_rollout(features_df, cfg, start, end, target_for_date)


def equal_weight_rollout(features_df, cfg, start, end) -> np.ndarray:
    def target_for_date(date, F, susp, prev):
        active = ~susp
        w = np.zeros(len(susp))
        if active.sum() > 0:
            w[active] = 1.0 / active.sum()
        return w

    return _weekly_rollout(features_df, cfg, start, end, target_for_date)


def long_only_topn_rollout(features_df, cfg, start, end, static_factor_w: np.ndarray) -> np.ndarray:
    static_factor_w = _l1_normalize(static_factor_w, zero_error=True)
    return _weekly_factor_rollout(
        features_df, cfg, start, end, lambda _: static_factor_w, short_enabled=False
    )


def static_factor_equal_rollout(features_df, cfg, start, end) -> np.ndarray:
    static_w = np.ones(K) / K
    return _weekly_factor_rollout(features_df, cfg, start, end, lambda _: static_w)


def static_factor_optimized_rollout(
    features_df, cfg, start, end, train_factor_returns: pd.DataFrame
) -> np.ndarray:
    static_w = fit_static_factor_weights(train_factor_returns)
    return _weekly_factor_rollout(features_df, cfg, start, end, lambda _: static_w)


def rolling_ic_rollout(features_df, market_state_df, cfg, start, end) -> np.ndarray:
    state = market_state_df.copy()
    if "trade_date" in state.columns:
        state["trade_date"] = pd.to_datetime(state["trade_date"])
        state = state.set_index("trade_date")
    else:
        state.index = pd.to_datetime(state.index)
    state = state.sort_index()

    def weight_for_date(date):
        if date not in state.index:
            raise ValueError(f"missing causal market state for decision date {date}")
        return rolling_ic_weights(state.loc[date])

    return _weekly_factor_rollout(features_df, cfg, start, end, weight_for_date)
