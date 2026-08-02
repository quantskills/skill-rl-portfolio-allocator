"""Training-only metrics for fold-local factor selection.

This module deliberately stops at metrics and percentile scoring.  Selection,
redundancy control, and artifact writing belong to the following task.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np
import pandas as pd

from scripts import costs
from scripts.config import get_config
from scripts.env import extract_settle_holding_period
from scripts.rebalance import weekly_decision_indices


@dataclass(frozen=True)
class SelectionThresholds:
    min_coverage: float = 0.98
    min_symbols: int = 100
    min_dates: int = 500


_CORE_COLUMNS = {"trade_date", "symbol", "ret_1d"}
_COST_KEYS = ("commission", "stamp_tax", "impact", "borrow", "total")
_SCORE_FIELDS = (
    "mean_ic",
    "icir",
    "sign_consistency",
    "net_factor_sharpe",
    "doubled_cost_sharpe",
    "regime_stability",
    "factor_turnover",
    "tail_loss",
)


def _as_thresholds(cfg: dict[str, Any] | None) -> SelectionThresholds:
    cfg = cfg or {}
    value = cfg.get("selection_thresholds", cfg.get("thresholds"))
    if value is None:
        value = cfg
    if isinstance(value, SelectionThresholds):
        return value
    return SelectionThresholds(
        min_coverage=float(value.get("min_coverage", 0.98)),
        min_symbols=int(value.get("min_symbols", 100)),
        min_dates=int(value.get("min_dates", 500)),
    )


def _training_panel(
    panel: pd.DataFrame,
    factor_names: Iterable[str],
    train_start: Any,
    train_end: Any,
) -> tuple[pd.DataFrame, list[str], pd.Timestamp, pd.Timestamp]:
    if not isinstance(panel, pd.DataFrame):
        raise TypeError("panel must be a pandas DataFrame")
    names = list(factor_names)
    if not names or len(set(names)) != len(names) or not all(isinstance(x, str) for x in names):
        raise ValueError("factor_names must be a non-empty sequence of unique strings")
    missing = _CORE_COLUMNS - set(panel.columns)
    missing_factors = [name for name in names if name not in panel.columns]
    if missing:
        raise ValueError(f"panel schema missing columns: {sorted(missing)}")
    if missing_factors:
        raise ValueError(f"unknown factor columns: {missing_factors}")
    if panel.columns.duplicated().any():
        raise ValueError("panel schema contains duplicate columns")

    try:
        start = pd.Timestamp(train_start)
        end = pd.Timestamp(train_end)
    except (TypeError, ValueError) as exc:
        raise ValueError("train_start and train_end must be valid dates") from exc
    if pd.isna(start) or pd.isna(end) or start > end:
        raise ValueError("train_start must be on or before train_end")

    out = panel.copy()
    parsed_dates = pd.to_datetime(out["trade_date"], errors="coerce", format="mixed")
    in_train = parsed_dates.notna() & (parsed_dates >= start) & (parsed_dates <= end)
    out = out.loc[in_train].copy()
    out["trade_date"] = parsed_dates.loc[in_train]
    if out.empty:
        raise ValueError("training interval contains no valid rows")
    empty_symbols = out["symbol"].isna() | out["symbol"].astype(str).str.strip().eq("")
    if empty_symbols.any():
        raise ValueError("symbol contains null or empty values")
    if out.duplicated(["trade_date", "symbol"]).any():
        raise ValueError("duplicate (trade_date, symbol) keys")

    numeric = ["ret_1d", *names]
    for column in numeric:
        out[column] = pd.to_numeric(out[column], errors="coerce")
    if "is_suspended" not in out.columns:
        out["is_suspended"] = False
    else:
        out["is_suspended"] = out["is_suspended"].fillna(False).astype(bool)

    # The first training date therefore cannot borrow a factor observation
    # from outside the fold.
    out = out.sort_values(["symbol", "trade_date"], kind="mergesort").reset_index(drop=True)
    return out, names, start, end


def _finite(values: pd.Series) -> np.ndarray:
    return np.isfinite(values.to_numpy(dtype=float, na_value=np.nan))


def _spearman(left: pd.Series, right: pd.Series) -> float:
    x = left.to_numpy(dtype=float, na_value=np.nan)
    y = right.to_numpy(dtype=float, na_value=np.nan)
    valid = np.isfinite(x) & np.isfinite(y)
    if valid.sum() < 2:
        return np.nan
    x = x[valid]
    y = y[valid]
    if np.std(x) <= 0.0 or np.std(y) <= 0.0:
        return np.nan
    return float(pd.Series(x).corr(pd.Series(y), method="spearman"))


def _weekly_sharpe(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    series = np.asarray(values, dtype=float)
    deviation = float(np.std(series, ddof=1))
    if not np.isfinite(deviation) or deviation <= 0.0:
        return 0.0
    wealth = float(np.prod(1.0 + series))
    if not np.isfinite(wealth) or wealth <= 0.0:
        return 0.0
    annual_return = float(wealth ** (52.0 / len(series)) - 1.0)
    return annual_return / (deviation * np.sqrt(52.0))


def _empty_costs() -> dict[str, float]:
    return {key: 0.0 for key in _COST_KEYS}


def _cost_config(cfg: dict[str, Any] | None) -> dict[str, Any]:
    base = get_config()
    if cfg:
        for key in (
            "commission_bps",
            "stamp_tax_bps",
            "impact_bps",
            "borrow_rate_annual",
            "trading_days_per_year",
        ):
            if key in cfg:
                base[key] = cfg[key]
    return base


def _freeze_factor_target(
    raw_target: np.ndarray,
    previous_target: np.ndarray,
    suspended: np.ndarray,
) -> np.ndarray:
    """Freeze suspended holdings while preserving the factor 50/50 notional.

    ``portfolio.freeze_suspended`` also enforces the environment's 0.30 short
    cap.  Factor diagnostics require a symmetric 0.50/0.50 portfolio, so this
    is the same freeze rule with that environment-specific cap omitted.
    """
    target = np.asarray(raw_target, dtype=float).copy()
    previous = np.asarray(previous_target, dtype=float)
    frozen = np.asarray(suspended, dtype=bool)
    target[frozen] = previous[frozen]
    for positive in (True, False):
        side = target > 0.0 if positive else target < 0.0
        frozen_side = side & frozen
        free_side = side & ~frozen
        target_notional = 0.5
        held = float(np.clip(target[frozen_side], 0.0, None).sum()) if positive else float(np.clip(-target[frozen_side], 0.0, None).sum())
        available = target_notional - held
        free = float(np.clip(target[free_side], 0.0, None).sum()) if positive else float(np.clip(-target[free_side], 0.0, None).sum())
        if available < -1e-9:
            raise ValueError("suspended factor holdings exceed 50/50 notional")
        if free > 0.0:
            target[free_side] *= available / free
    return target


def _weekly_factor_series(
    df: pd.DataFrame,
    factor: str,
    direction: int,
    cfg: dict[str, Any] | None,
) -> dict[str, Any]:
    lagged = f"__{factor}_lag"
    df = df.copy()
    df[lagged] = df.groupby("symbol", sort=False)[factor].shift(1)
    dates = pd.DatetimeIndex(sorted(df["trade_date"].unique()))
    decision_indices = weekly_decision_indices(dates)
    cost_cfg = _cost_config(cfg)
    universe = sorted(df["symbol"].unique())
    previous_target = np.zeros(len(universe), dtype=float)
    gross_returns: list[float] = []
    net_returns: list[float] = []
    weekly_turnover: list[float] = []
    weekly_costs: list[dict[str, float]] = []
    weekly_decision_dates: list[pd.Timestamp] = []
    weekly_settlement_dates: list[list[pd.Timestamp]] = []
    weekly_signal_symbol_counts: list[int] = []
    weekly_long_notionals: list[float] = []
    weekly_short_notionals: list[float] = []
    weekly_target_weights: list[list[float]] = []
    weekly_failure_reasons: list[dict[str, Any]] = []
    all_decision_dates = [pd.Timestamp(dates[index]) for index in decision_indices]

    def record_failure(decision_date: pd.Timestamp | None, reason: str) -> None:
        date_value = None if decision_date is None or pd.isna(decision_date) else pd.Timestamp(decision_date).date().isoformat()
        weekly_failure_reasons.append({"decision_date": date_value, "reason": reason})

    for position, decision_index in enumerate(decision_indices):
        decision_date = pd.Timestamp(dates[decision_index])
        has_next_decision = position + 1 < len(decision_indices)
        next_index = decision_indices[position + 1] if has_next_decision else len(dates)
        settlement_end = next_index + 1 if has_next_decision else len(dates)
        settlement_dates = list(pd.to_datetime(dates[decision_index + 1 : settlement_end]))
        if not settlement_dates:
            # No settlement after the first/walk-forward boundary decision is
            # warmup or fold-end boundary, not a bad weekly observation.
            continue

        decision_rows = df.loc[df["trade_date"] == decision_date].copy()
        signal_values = decision_rows.loc[
            ~decision_rows["is_suspended"] & _finite(decision_rows[lagged]),
            ["symbol", lagged],
        ]
        signal_values = signal_values.drop_duplicates("symbol", keep="first")
        if not len(signal_values):
            # The first decision has no in-fold lag by construction.
            continue
        if len(signal_values) < 10:
            record_failure(decision_date, "short_week")
            continue
        signal_values = signal_values.assign(_effective_signal=direction * signal_values[lagged])
        ranked = signal_values.sort_values(["_effective_signal", "symbol"], kind="mergesort")
        side = len(ranked) // 2
        if side < 5:
            record_failure(decision_date, "short_week")
            continue
        raw_target = np.zeros(len(universe), dtype=float)
        symbol_index = {symbol: index for index, symbol in enumerate(universe)}
        for symbol in ranked.iloc[:side]["symbol"]:
            raw_target[symbol_index[symbol]] = -0.5 / side
        for symbol in ranked.iloc[-side:]["symbol"]:
            raw_target[symbol_index[symbol]] = 0.5 / side
        decision_by_symbol = decision_rows.set_index("symbol").reindex(universe)
        suspended_at_decision = decision_by_symbol["is_suspended"].to_numpy(dtype=bool)
        target = _freeze_factor_target(raw_target, previous_target, suspended_at_decision)

        returns_by_date: dict[pd.Timestamp, np.ndarray] = {}
        invalid_week = False
        for settlement_date in settlement_dates:
            day = df.loc[df["trade_date"] == settlement_date].set_index("symbol").reindex(universe)
            if day["ret_1d"].isna().any() or not np.isfinite(day["ret_1d"].to_numpy(dtype=float)).all():
                invalid_week = True
                break
            values = day["ret_1d"].to_numpy(dtype=float)
            values[day["is_suspended"].to_numpy(dtype=bool)] = 0.0
            returns_by_date[pd.Timestamp(settlement_date)] = values
        if invalid_week:
            record_failure(decision_date, "invalid_week")
            continue

        try:
            settled = extract_settle_holding_period(
                previous_target,
                target,
                settlement_dates,
                returns_by_date,
                cost_cfg,
            )
        except (KeyError, TypeError, ValueError, FloatingPointError):
            record_failure(decision_date, "invalid_week")
            continue

        gross = float(settled["gross_ret"])
        net = float(settled["net_ret"])
        gross_returns.append(gross)
        net_returns.append(net)
        weekly_turnover.append(float(np.abs(target - previous_target).sum()))
        weekly_costs.append({key: float(settled["costs"][key]) for key in _COST_KEYS})
        weekly_decision_dates.append(decision_date)
        weekly_settlement_dates.append([pd.Timestamp(value) for value in settlement_dates])
        weekly_signal_symbol_counts.append(int(len(ranked["symbol"].unique())))
        weekly_long_notionals.append(float(np.clip(target, 0.0, None).sum()))
        weekly_short_notionals.append(float(np.clip(-target, 0.0, None).sum()))
        weekly_target_weights.append(target.tolist())
        previous_target = target

    total_costs = _empty_costs()
    for item in weekly_costs:
        for key in _COST_KEYS:
            total_costs[key] += item.get(key, 0.0)
    doubled_costs = {key: 2.0 * value for key, value in total_costs.items()}
    weekly_doubled_costs = [
        {key: 2.0 * value for key, value in item.items()}
        for item in weekly_costs
    ]
    doubled_returns = [
        gross - 2.0 * (gross - net)
        for gross, net in zip(gross_returns, net_returns)
    ]
    if not weekly_decision_dates:
        record_failure(None, "no_valid_weekly_data")
    return {
        "weekly_gross_returns": gross_returns,
        "weekly_net_returns": net_returns,
        "weekly_doubled_cost_returns": doubled_returns,
        "weekly_turnover": weekly_turnover,
        "weekly_costs": weekly_costs,
        "weekly_doubled_costs": weekly_doubled_costs,
        "weekly_decision_dates": [date.date().isoformat() for date in weekly_decision_dates],
        "decision_dates": [date.date().isoformat() for date in all_decision_dates],
        "weekly_settlement_dates": [
            [date.date().isoformat() for date in dates]
            for dates in weekly_settlement_dates
        ],
        "weekly_signal_symbol_counts": weekly_signal_symbol_counts,
        "weekly_long_notionals": weekly_long_notionals,
        "weekly_short_notionals": weekly_short_notionals,
        "weekly_target_weights": weekly_target_weights,
        "weekly_failure_reasons": weekly_failure_reasons,
        "costs": total_costs,
        "doubled_costs": doubled_costs,
        "gross_factor_sharpe": _weekly_sharpe(gross_returns),
        "net_factor_sharpe": _weekly_sharpe(net_returns),
        "doubled_cost_sharpe": _weekly_sharpe(doubled_returns),
        "factor_turnover": float(np.mean(weekly_turnover) * 52.0) if weekly_turnover else 0.0,
    }


def _regime_metrics(
    df: pd.DataFrame,
    ic_by_date: list[tuple[pd.Timestamp, float]],
    direction: int,
) -> tuple[dict[str, float | None], dict[str, int], float]:
    daily = []
    for date, group in df.groupby("trade_date", sort=True):
        values = pd.to_numeric(
            group.loc[~group["is_suspended"], "ret_1d"], errors="coerce"
        )
        values = values.replace([np.inf, -np.inf], np.nan).dropna()
        daily.append((pd.Timestamp(date), float(values.median()) if len(values) else np.nan))
    market = pd.Series(
        {date: value for date, value in daily}, dtype=float
    ).sort_index()
    rolling_vol = market.rolling(20, min_periods=2).std()
    return_threshold = float(market.median()) if market.notna().any() else np.nan
    vol_threshold = float(rolling_vol.median()) if rolling_vol.notna().any() else np.nan
    regimes = {
        "bull": market >= return_threshold,
        "bear": market < return_threshold,
        "high_vol": rolling_vol >= vol_threshold,
        "low_vol": rolling_vol < vol_threshold,
    }
    ic_series = pd.Series(
        {pd.Timestamp(date): float(value) for date, value in ic_by_date}, dtype=float
    )
    scores: dict[str, float | None] = {}
    observations: dict[str, int] = {}
    for name, mask in regimes.items():
        selected = ic_series[mask.reindex(ic_series.index).fillna(False)]
        observations[name] = int(len(selected))
        scores[name] = (
            float(np.mean(direction * selected.to_numpy(dtype=float) > 0.0))
            if len(selected)
            else None
        )
    available = [value for value in scores.values() if value is not None]
    stability = float(np.mean(available)) if available else 0.0
    return scores, observations, stability


def _factor_metrics(
    df: pd.DataFrame,
    factor: str,
    thresholds: SelectionThresholds,
    cfg: dict[str, Any] | None,
) -> dict[str, Any]:
    lagged = f"__{factor}_lag"
    df = df.copy()
    df[lagged] = df.groupby("symbol", sort=False)[factor].shift(1)
    eligible = ~df["is_suspended"]
    factor_finite = _finite(df[factor])
    ret_finite = _finite(df["ret_1d"])
    base_count = int(eligible.sum())
    coverage = float((eligible.to_numpy() & factor_finite & ret_finite).sum() / base_count) if base_count else 0.0
    had_nonfinite = bool((eligible.to_numpy() & ~(factor_finite & ret_finite)).any())

    daily_ics: list[float] = []
    ic_by_date: list[tuple[pd.Timestamp, float]] = []
    scored_dates: list[pd.Timestamp] = []
    eligible_symbols_by_date: list[int] = []
    date_variances: list[float] = []
    first_training_date = pd.Timestamp(df["trade_date"].min()) if len(df) else None
    for date, group in df.groupby("trade_date", sort=True):
        valid = (
            ~group["is_suspended"].to_numpy(dtype=bool)
            & _finite(group[lagged])
            & _finite(group["ret_1d"])
        )
        sample = group.loc[valid, [lagged, "ret_1d", "symbol"]]
        if not len(sample):
            has_lag_observation = group[lagged].notna().any()
            if not has_lag_observation and pd.Timestamp(date) == first_training_date:
                continue
            scored_dates.append(pd.Timestamp(date))
            eligible_symbols_by_date.append(0)
            date_variances.append(0.0)
            continue
        scored_dates.append(pd.Timestamp(date))
        eligible_symbols_by_date.append(int(sample["symbol"].nunique()))
        variance = float(np.var(sample[lagged].to_numpy(dtype=float)))
        date_variances.append(variance)
        if len(sample) < 10:
            continue
        ic = _spearman(sample[lagged], sample["ret_1d"])
        if np.isfinite(ic):
            daily_ics.append(ic)
            ic_by_date.append((pd.Timestamp(date), ic))

    ic_values = np.asarray(daily_ics, dtype=float)
    mean_ic = float(np.mean(ic_values)) if len(ic_values) else 0.0
    direction = -1 if mean_ic < 0.0 else 1
    oriented_mean_ic = float(direction * mean_ic)
    ic_std = float(np.std(ic_values, ddof=1)) if len(ic_values) > 1 else 0.0
    icir = float(mean_ic / ic_std) if ic_std > 0.0 else 0.0
    positive_ic_rate = float(np.mean(direction * ic_values > 0.0)) if len(ic_values) else 0.0
    if ic_by_date:
        yearly_ic = pd.DataFrame(ic_by_date, columns=["trade_date", "ic"])
        yearly_means = yearly_ic.assign(year=yearly_ic["trade_date"].dt.year).groupby("year")["ic"].mean()
        sign_consistency = float(np.mean(direction * yearly_means.to_numpy() > 0.0)) if len(yearly_means) else 0.0
    else:
        sign_consistency = 0.0

    weekly = _weekly_factor_series(df, factor, direction, cfg)
    regime_scores, regime_observations, regime_stability = _regime_metrics(df, ic_by_date, direction)
    net_returns = weekly["weekly_net_returns"]
    worst_count = max(1, int(np.ceil(0.05 * len(net_returns)))) if net_returns else 0
    worst = float(np.mean(np.sort(net_returns)[:worst_count])) if worst_count else 0.0
    tail_loss = max(0.0, -worst)

    failure_reasons: list[str] = []
    def add_failure(reason: str) -> None:
        if reason not in failure_reasons:
            failure_reasons.append(reason)

    if coverage < thresholds.min_coverage:
        add_failure("coverage")
    symbols = min(eligible_symbols_by_date) if eligible_symbols_by_date else 0
    dates = len(daily_ics)
    if any(count < thresholds.min_symbols for count in eligible_symbols_by_date):
        add_failure("insufficient_symbols_on_scored_date")
        add_failure("symbols")
    if dates < thresholds.min_dates:
        add_failure("dates")
    if not date_variances or any(value <= 0.0 or not np.isfinite(value) for value in date_variances):
        add_failure("non_degenerate_cross_sectional_variance")
    if had_nonfinite:
        add_failure("non_finite_values")
    if not daily_ics:
        add_failure("no_valid_training_data")
    for item in weekly["weekly_failure_reasons"]:
        add_failure(item["reason"])
    if not weekly["weekly_net_returns"]:
        add_failure("weekly_data")
    if len(weekly["weekly_net_returns"]) < 2:
        add_failure("insufficient_weekly_observations")
    if any(value is None for value in regime_scores.values()):
        add_failure("insufficient_regime_data")

    return {
        "name": factor,
        "mean_ic": mean_ic,
        "icir": icir,
        "sign_consistency": sign_consistency,
        "positive_ic_rate": positive_ic_rate,
        "coverage": coverage,
        "symbols": int(symbols),
        "dates": int(dates),
        "scored_dates": [date.date().isoformat() for date in scored_dates],
        "eligible_symbols_by_date": eligible_symbols_by_date,
        "direction": direction,
        "oriented_mean_ic": oriented_mean_ic,
        "weekly_gross_returns": weekly["weekly_gross_returns"],
        "weekly_net_returns": weekly["weekly_net_returns"],
        "weekly_doubled_cost_returns": weekly["weekly_doubled_cost_returns"],
        "gross_returns": weekly["weekly_gross_returns"],
        "net_returns": weekly["weekly_net_returns"],
        "doubled_cost_returns": weekly["weekly_doubled_cost_returns"],
        "weekly_turnover": weekly["weekly_turnover"],
        "gross_factor_sharpe": weekly["gross_factor_sharpe"],
        "net_factor_sharpe": weekly["net_factor_sharpe"],
        "doubled_cost_sharpe": weekly["doubled_cost_sharpe"],
        "factor_turnover": weekly["factor_turnover"],
        "gross_sharpe": weekly["gross_factor_sharpe"],
        "net_sharpe": weekly["net_factor_sharpe"],
        "weekly_gross_sharpe": weekly["gross_factor_sharpe"],
        "weekly_net_sharpe": weekly["net_factor_sharpe"],
        "weekly_doubled_cost_sharpe": weekly["doubled_cost_sharpe"],
        "turnover": weekly["factor_turnover"],
        "annualized_turnover": weekly["factor_turnover"],
        "costs": weekly["costs"],
        "doubled_costs": weekly["doubled_costs"],
        "weekly_costs": weekly["weekly_costs"],
        "weekly_doubled_costs": weekly["weekly_doubled_costs"],
        "weekly_decision_dates": weekly["weekly_decision_dates"],
        "decision_dates": weekly["decision_dates"],
        "weekly_settlement_dates": weekly["weekly_settlement_dates"],
        "weekly_signal_symbol_counts": weekly["weekly_signal_symbol_counts"],
        "weekly_long_notionals": weekly["weekly_long_notionals"],
        "weekly_short_notionals": weekly["weekly_short_notionals"],
        "weekly_target_weights": weekly["weekly_target_weights"],
        "weekly_failure_reasons": weekly["weekly_failure_reasons"],
        "tail_loss": tail_loss,
        "worst_five_pct_return": worst,
        "regime_scores": regime_scores,
        "regime_observations": regime_observations,
        "regime_stability": regime_stability,
        "failure_reasons": failure_reasons,
        "passed": not failure_reasons,
    }


def compute_factor_metrics(
    panel: pd.DataFrame,
    factor_names: Iterable[str],
    train_start: Any,
    train_end: Any,
    cfg: dict[str, Any] | None,
) -> dict[str, dict[str, Any]]:
    """Compute causal metrics using only the inclusive training interval.

    The panel is sliced before per-symbol lagging, so no observation outside
    the fold can influence either IC or the weekly factor portfolios.
    """
    thresholds = _as_thresholds(cfg)
    training, names, _, _ = _training_panel(panel, factor_names, train_start, train_end)
    return {name: _factor_metrics(training, name, thresholds, cfg) for name in names}


def _percentile(values: dict[str, float]) -> dict[str, float]:
    if not values:
        return {}
    ordered = sorted(values.items(), key=lambda item: (item[1], item[0]))
    n = len(ordered)
    if n == 1:
        return {ordered[0][0]: 0.5}
    result: dict[str, float] = {}
    index = 0
    while index < n:
        end = index + 1
        while end < n and ordered[end][1] == ordered[index][1]:
            end += 1
        percentile = ((index + end - 1) / 2.0) / (n - 1)
        for position in range(index, end):
            result[ordered[position][0]] = float(percentile)
        index = end
    return result


def percentile_scores(metrics: dict[str, dict[str, Any]]) -> dict[str, float]:
    """Return deterministic within-fold composite percentile scores.

    Candidates that failed a hard gate or lack any scoring field are omitted;
    no value is imputed from another fold or from validation/test data.
    """
    if not isinstance(metrics, dict):
        raise TypeError("metrics must be a dictionary keyed by factor name")
    eligible: dict[str, dict[str, float]] = {}
    for name in sorted(metrics):
        item = metrics[name]
        if not isinstance(item, dict) or item.get("passed", not item.get("failure_reasons")) is not True:
            continue
        values: dict[str, float] = {}
        valid = True
        for field in _SCORE_FIELDS:
            try:
                value = float(item[field])
            except (KeyError, TypeError, ValueError):
                valid = False
                break
            if not np.isfinite(value):
                valid = False
                break
            if field in {"mean_ic", "icir"}:
                value = abs(value)
            values[field] = value
        if valid:
            eligible[name] = values
    if not eligible:
        return {}

    percentiles: dict[str, dict[str, float]] = {}
    for field in _SCORE_FIELDS:
        raw = {name: values[field] for name, values in eligible.items()}
        percentiles[field] = _percentile(raw)
    scores: dict[str, float] = {}
    for name in sorted(eligible):
        p = percentiles
        scores[name] = float(
            0.30 * p["mean_ic"][name]
            + 0.20 * p["icir"][name]
            + 0.15 * p["sign_consistency"][name]
            + 0.15 * p["net_factor_sharpe"][name]
            + 0.10 * p["doubled_cost_sharpe"][name]
            + 0.10 * p["regime_stability"][name]
            - 0.10 * p["factor_turnover"][name]
            - 0.05 * p["tail_loss"][name]
        )
    return scores
