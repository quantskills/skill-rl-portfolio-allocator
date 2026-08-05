"""Strictly causal market and factor state construction."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from scripts import costs
from scripts.config import get_config

MARKET_STATE_SCHEMA_VERSION = "market-state-v2"


def distribution_shift_report(
    train: pd.DataFrame,
    other: pd.DataFrame,
    fields: list[str] | tuple[str, ...],
    limit: float = 5,
) -> dict:
    details: dict[str, dict[str, float | bool]] = {}
    for field in fields:
        train_values = pd.to_numeric(train[field], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
        other_values = pd.to_numeric(other[field], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
        if train_values.empty or other_values.empty:
            shift = float("inf")
        else:
            scale = float(train_values.std(ddof=1)) / np.sqrt(len(train_values))
            shift = abs(float(other_values.mean() - train_values.mean())) / scale if scale > 0 else 0.0
        details[field] = {"standardized_mean_shift": shift, "passed": bool(shift <= limit)}
    return {"fields": details, "limit": limit, "passed": bool(details) and all(item["passed"] for item in details.values())}


def _require_columns(frame: pd.DataFrame, required: set[str], name: str) -> None:
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{name} missing columns: {sorted(missing)}")


def _factor_names(cfg: dict | None, factor_names) -> tuple[str, ...]:
    resolved = tuple(factor_names if factor_names is not None else (cfg or get_config())["factor_names"])
    if not resolved:
        raise ValueError("factor_names must be non-empty")
    return resolved


def _sorted_features(features: pd.DataFrame, factor_names: tuple[str, ...]) -> pd.DataFrame:
    _require_columns(features, {"trade_date", "symbol", "ret_1d", *factor_names}, "features")
    out = features.copy()
    out["trade_date"] = pd.to_datetime(out["trade_date"])
    return out.sort_values(["symbol", "trade_date"], kind="mergesort")


def compute_daily_factor_ic(features: pd.DataFrame, factor_names) -> pd.DataFrame:
    """Return daily Spearman IC using factor at t-1 and return at t."""
    names = _factor_names(None, factor_names)
    df = _sorted_features(features, names)
    for factor in names:
        df[f"_{factor}_lag"] = df.groupby("symbol", sort=False)[factor].shift(1)
    dates = sorted(df["trade_date"].dropna().unique())
    ic_cols: dict[str, list | pd.Series] = {"trade_date": dates}
    for factor in names:
        lagged = f"_{factor}_lag"

        def daily_ic(group: pd.DataFrame) -> float:
            sample = group[[lagged, "ret_1d"]].replace([np.inf, -np.inf], np.nan).dropna()
            if len(sample) < 10:
                return np.nan
            return float(sample[lagged].corr(sample["ret_1d"], method="spearman"))

        values = df.groupby("trade_date", sort=True)[[lagged, "ret_1d"]].apply(daily_ic)
        ic_cols[f"{factor}_ic"] = pd.Series(dates).map(values)
    return pd.DataFrame(ic_cols)


def compute_daily_factor_returns(
    features: pd.DataFrame, cfg: dict | None = None, factor_names=None
) -> pd.DataFrame:
    """Compute causal 50/50 long-short returns for every factor."""
    cfg = cfg or get_config()
    names = _factor_names(cfg, factor_names)
    df = _sorted_features(features, names)
    if "is_suspended" not in df:
        df["is_suspended"] = False
    for factor in names:
        df[f"_{factor}_lag"] = df.groupby("symbol", sort=False)[factor].shift(1)
    rows: list[dict] = []
    for date, group in df.groupby("trade_date", sort=True):
        row: dict = {"trade_date": date}
        eligible_base = group.loc[~group["is_suspended"].fillna(False)]
        for factor in names:
            cols = [f"_{factor}_lag", "ret_1d"]
            eligible = eligible_base[cols].replace([np.inf, -np.inf], np.nan).dropna()
            if len(eligible) < 10:
                row[f"{factor}_factor_ret"] = np.nan
                continue
            ranked = eligible.sort_values(cols[0], kind="mergesort")
            n_side = len(ranked) // 2
            if n_side < 5:
                row[f"{factor}_factor_ret"] = np.nan
                continue
            short = ranked.iloc[:n_side]["ret_1d"].mean()
            long = ranked.iloc[-n_side:]["ret_1d"].mean()
            target = np.zeros(len(ranked), dtype=float)
            target[-n_side:] = 0.5 / n_side
            target[:n_side] = -0.5 / n_side
            zero = np.zeros_like(target)
            trading_cost = costs.total_costs(zero, target, cfg)["total"]
            row[f"{factor}_factor_ret"] = float(0.5 * long - 0.5 * short - trading_cost)
        rows.append(row)
    return pd.DataFrame(rows)


def _rolling_drawdown(returns: pd.Series, window: int = 60) -> pd.Series:
    wealth = (1.0 + returns.fillna(0.0)).cumprod()
    peak = wealth.rolling(window, min_periods=window).max()
    return wealth / peak - 1.0


def _expanding_percentile(window: np.ndarray) -> float:
    """当前值在历史(不含当前)中的分位,因果、跨 regime 可比。"""
    current = window[-1]
    history = window[:-1]
    history = history[np.isfinite(history)]
    if not np.isfinite(current) or history.size == 0:
        return np.nan
    return float((history <= current).mean())


def rolling_mean_factor_corr(
    factor_returns: pd.DataFrame, window: int, factor_names
) -> pd.Series:
    """Return the mean pairwise correlation over a trailing return window."""
    names = _factor_names(None, factor_names)
    _require_columns(factor_returns, {"trade_date", *[f"{f}_factor_ret" for f in names]}, "factor_returns")
    df = factor_returns.copy()
    df["trade_date"] = pd.to_datetime(df["trade_date"])
    df = df.sort_values("trade_date").set_index("trade_date")
    columns = [f"{factor}_factor_ret" for factor in names]

    values: list[float] = []
    for end in range(len(df)):
        sample = df[columns].iloc[max(0, end + 1 - window) : end + 1]
        if len(sample) < window:
            values.append(np.nan)
            continue
        sample = sample.replace([np.inf, -np.inf], np.nan).dropna()
        if len(sample) < window:
            values.append(np.nan)
            continue
        if len(columns) == 1:
            values.append(1.0)
            continue
        corr = sample.corr().to_numpy()
        upper = corr[np.triu_indices(len(columns), 1)]
        values.append(float(np.nanmean(upper)) if np.isfinite(upper).any() else np.nan)
    return pd.Series(values, index=df.index, dtype=float)


def _market_features(index_returns: pd.DataFrame) -> pd.DataFrame:
    _require_columns(index_returns, {"trade_date", "ret"}, "index_returns")
    idx = index_returns.copy()
    idx["trade_date"] = pd.to_datetime(idx["trade_date"])
    idx = idx.sort_values("trade_date").drop_duplicates("trade_date").set_index("trade_date")
    ret = idx["ret"].replace([np.inf, -np.inf], np.nan)
    out = pd.DataFrame(index=idx.index)
    out["market_ret_20"] = (1.0 + ret).rolling(20, min_periods=20).apply(np.prod, raw=True) - 1.0
    out["market_ret_60"] = (1.0 + ret).rolling(60, min_periods=60).apply(np.prod, raw=True) - 1.0
    out["market_vol_20"] = ret.rolling(20, min_periods=20).std()
    out["market_vol_60"] = ret.rolling(60, min_periods=60).std()
    out["market_drawdown_60"] = _rolling_drawdown(ret, 60)
    expanding_vol = ret.expanding(min_periods=2).std()
    out["market_vol_regime"] = out["market_vol_20"] / expanding_vol.replace(0.0, np.nan)
    out["market_vol_percentile_20"] = out["market_vol_20"].expanding(min_periods=60).apply(
        _expanding_percentile, raw=True,
    )
    nav = (1.0 + ret.fillna(0.0)).cumprod()
    out["market_drawdown_ath"] = nav / nav.cummax() - 1.0
    return out.reset_index()


def build_market_state(
    features: pd.DataFrame, index_returns: pd.DataFrame, cfg: dict | None = None,
    factor_names=None,
) -> pd.DataFrame:
    cfg = cfg or get_config()
    names = _factor_names(cfg, factor_names)
    market = _market_features(index_returns)

    # Build IC-derived columns in a dict and create a single DataFrame to avoid
    # the fragmentation caused by assigning columns one-by-one in a loop.
    ic_raw = compute_daily_factor_ic(features, names)
    ic_cols: dict[str, pd.Series] = {"trade_date": ic_raw["trade_date"]}
    for factor in names:
        s = ic_raw[f"{factor}_ic"]
        ic_cols[f"{factor}_ic"] = s
        ic_cols[f"{factor}_ic_mean_20"] = s.rolling(20, min_periods=20).mean()
        ic_cols[f"{factor}_ic_mean_60"] = s.rolling(60, min_periods=60).mean()
        ic_cols[f"{factor}_icir_20"] = (
            s.rolling(20, min_periods=20).mean()
            / s.rolling(20, min_periods=20).std()
        )
        ic_cols[f"{factor}_ic_positive_20"] = (
            s.rolling(20, min_periods=20).apply(lambda x: np.mean(x > 0), raw=True)
        )
    ic = pd.DataFrame(ic_cols)

    # Same dict-based approach for factor return columns.
    fr_raw = compute_daily_factor_returns(features, cfg, names)
    fr_cols: dict[str, pd.Series] = {"trade_date": fr_raw["trade_date"]}
    for factor in names:
        col = f"{factor}_factor_ret"
        s = fr_raw[col]
        fr_cols[col] = s
        fr_cols[f"{factor}_factor_ret_20"] = s.rolling(20, min_periods=20).mean()
        fr_cols[f"{factor}_factor_ret_60"] = s.rolling(60, min_periods=60).mean()
        fr_cols[f"{factor}_factor_vol_20"] = s.rolling(20, min_periods=20).std()
        fr_cols[f"{factor}_factor_vol_60"] = s.rolling(60, min_periods=60).std()
    fr_cols["factor_corr_20"] = (
        rolling_mean_factor_corr(fr_raw, 20, names)
        .reindex(pd.to_datetime(fr_raw["trade_date"]))
        .to_numpy()
    )
    fr_cols["factor_corr_60"] = (
        rolling_mean_factor_corr(fr_raw, 60, names)
        .reindex(pd.to_datetime(fr_raw["trade_date"]))
        .to_numpy()
    )
    factor_returns = pd.DataFrame(fr_cols)

    state = market.merge(ic, on="trade_date", how="inner").merge(factor_returns, on="trade_date", how="inner")
    state = state.drop(columns=[f"{factor}_factor_ret" for factor in names])
    state = state.sort_values("trade_date").reset_index(drop=True)
    state["schema_version"] = MARKET_STATE_SCHEMA_VERSION
    numeric = state.select_dtypes(include=[np.number]).columns
    state[numeric] = state[numeric].replace([np.inf, -np.inf], np.nan)
    return state


def state_quality_report(state: pd.DataFrame) -> dict:
    fields = {}
    for column in state.columns:
        if column in {"trade_date", "schema_version"}:
            continue
        values = pd.to_numeric(state[column], errors="coerce")
        finite = np.isfinite(values.to_numpy())
        nonnull_rate = float(values.notna().mean())
        finite_rate = float(finite.mean()) if len(values) else 0.0
        std = float(values.std()) if values.notna().any() else 0.0
        fields[column] = {
            "nonnull_rate": nonnull_rate,
            "finite_rate": finite_rate,
            "zero_rate": float((values.fillna(0.0) == 0.0).mean()),
            "std": std,
            "passed": finite_rate > 0.95 and std > 0.0,
        }
    passed = bool(fields) and all(item["passed"] for item in fields.values())
    return {"schema_version": MARKET_STATE_SCHEMA_VERSION, "fields": fields, "passed": passed}


def main() -> None:
    root = Path(__file__).resolve().parent.parent
    features_path = root / "data" / "features.parquet"
    index_path = root / "data" / "index_returns.parquet"
    if not features_path.exists() or not index_path.exists():
        raise FileNotFoundError(f"required input missing: {features_path} and/or {index_path}")
    state = build_market_state(pd.read_parquet(features_path), pd.read_parquet(index_path), get_config())
    report = state_quality_report(state)
    if not report["passed"]:
        raise RuntimeError("market state quality check failed: " + json.dumps(report, ensure_ascii=False))
    output = root / "data" / "market_state.parquet"
    quality = root / "artifacts" / "state" / "market_state_quality.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    quality.parent.mkdir(parents=True, exist_ok=True)
    state.to_parquet(output, index=False)
    quality.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
