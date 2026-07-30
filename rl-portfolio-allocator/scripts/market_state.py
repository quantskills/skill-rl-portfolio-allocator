"""Strictly causal market and factor state construction."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from scripts import costs
from scripts.config import FACTOR_NAMES, get_config

MARKET_STATE_SCHEMA_VERSION = "market-state-v1"


def _require_columns(frame: pd.DataFrame, required: set[str], name: str) -> None:
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{name} missing columns: {sorted(missing)}")


def _sorted_features(features: pd.DataFrame) -> pd.DataFrame:
    _require_columns(features, {"trade_date", "symbol", "ret_1d", *FACTOR_NAMES}, "features")
    out = features.copy()
    out["trade_date"] = pd.to_datetime(out["trade_date"])
    return out.sort_values(["symbol", "trade_date"], kind="mergesort")


def compute_daily_factor_ic(features: pd.DataFrame) -> pd.DataFrame:
    """Return daily Spearman IC using factor at t-1 and return at t."""
    df = _sorted_features(features)
    for factor in FACTOR_NAMES:
        df[f"_{factor}_lag"] = df.groupby("symbol", sort=False)[factor].shift(1)
    dates = sorted(df["trade_date"].dropna().unique())
    result = pd.DataFrame({"trade_date": dates})
    for factor in FACTOR_NAMES:
        lagged = f"_{factor}_lag"

        def daily_ic(group: pd.DataFrame) -> float:
            sample = group[[lagged, "ret_1d"]].replace([np.inf, -np.inf], np.nan).dropna()
            if len(sample) < 10:
                return np.nan
            return float(sample[lagged].corr(sample["ret_1d"], method="spearman"))

        values = df.groupby("trade_date", sort=True)[[lagged, "ret_1d"]].apply(daily_ic)
        result[f"{factor}_ic"] = result["trade_date"].map(values)
    return result


def compute_daily_factor_returns(features: pd.DataFrame, cfg: dict | None = None) -> pd.DataFrame:
    """Compute causal 50/50 long-short returns for every factor."""
    cfg = cfg or get_config()
    df = _sorted_features(features)
    if "is_suspended" not in df:
        df["is_suspended"] = False
    for factor in FACTOR_NAMES:
        df[f"_{factor}_lag"] = df.groupby("symbol", sort=False)[factor].shift(1)
    rows: list[dict] = []
    for date, group in df.groupby("trade_date", sort=True):
        row: dict = {"trade_date": date}
        eligible_base = group.loc[~group["is_suspended"].fillna(False)]
        for factor in FACTOR_NAMES:
            cols = [f"_{factor}_lag", "ret_1d"]
            eligible = eligible_base[cols].replace([np.inf, -np.inf], np.nan).dropna()
            if len(eligible) < 10:
                row[f"{factor}_return"] = np.nan
                continue
            ranked = eligible.sort_values(cols[0], kind="mergesort")
            n_side = len(ranked) // 2
            if n_side < 5:
                row[f"{factor}_return"] = np.nan
                continue
            short = ranked.iloc[:n_side]["ret_1d"].mean()
            long = ranked.iloc[-n_side:]["ret_1d"].mean()
            target = np.zeros(len(ranked), dtype=float)
            target[-n_side:] = 0.5 / n_side
            target[:n_side] = -0.5 / n_side
            zero = np.zeros_like(target)
            trading_cost = costs.total_costs(zero, target, cfg)["total"]
            row[f"{factor}_return"] = float(0.5 * long - 0.5 * short - trading_cost)
        rows.append(row)
    return pd.DataFrame(rows)


def _rolling_drawdown(returns: pd.Series, window: int = 60) -> pd.Series:
    wealth = (1.0 + returns.fillna(0.0)).cumprod()
    peak = wealth.rolling(window, min_periods=window).max()
    return wealth / peak - 1.0


def rolling_mean_factor_corr(features: pd.DataFrame, window: int) -> pd.Series:
    df = _sorted_features(features)
    daily: list[tuple[pd.Timestamp, float]] = []
    for date, group in df.groupby("trade_date", sort=True):
        corr = group[FACTOR_NAMES].corr()
        upper = corr.to_numpy()[np.triu_indices(len(FACTOR_NAMES), 1)]
        daily.append((date, float(np.nanmean(upper)) if np.isfinite(upper).any() else np.nan))
    series = pd.Series(dict(daily), dtype=float).sort_index()
    return series.rolling(window, min_periods=window).mean()


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
    return out.reset_index()


def build_market_state(
    features: pd.DataFrame, index_returns: pd.DataFrame, cfg: dict | None = None
) -> pd.DataFrame:
    cfg = cfg or get_config()
    market = _market_features(index_returns)
    ic = compute_daily_factor_ic(features)
    for factor in FACTOR_NAMES:
        ic[f"{factor}_ic_mean_20"] = ic[f"{factor}_ic"].rolling(20, min_periods=20).mean()
        ic[f"{factor}_ic_mean_60"] = ic[f"{factor}_ic"].rolling(60, min_periods=60).mean()
        ic[f"{factor}_icir_20"] = ic[f"{factor}_ic"].rolling(20, min_periods=20).mean() / ic[f"{factor}_ic"].rolling(20, min_periods=20).std()
        ic[f"{factor}_ic_positive_20"] = ic[f"{factor}_ic"].rolling(20, min_periods=20).apply(lambda x: np.mean(x > 0), raw=True)
    factor_returns = compute_daily_factor_returns(features, cfg)
    for factor in FACTOR_NAMES:
        col = f"{factor}_return"
        factor_returns[f"{factor}_factor_ret_20"] = factor_returns[col].rolling(20, min_periods=20).mean()
        factor_returns[f"{factor}_factor_ret_60"] = factor_returns[col].rolling(60, min_periods=60).mean()
        factor_returns[f"{factor}_factor_vol_20"] = factor_returns[col].rolling(20, min_periods=20).std()
        factor_returns[f"{factor}_factor_vol_60"] = factor_returns[col].rolling(60, min_periods=60).std()
    factor_returns["factor_corr_20"] = rolling_mean_factor_corr(features, 20).reindex(pd.to_datetime(factor_returns["trade_date"])).to_numpy()
    factor_returns["factor_corr_60"] = rolling_mean_factor_corr(features, 60).reindex(pd.to_datetime(factor_returns["trade_date"])).to_numpy()
    state = market.merge(ic, on="trade_date", how="outer").merge(factor_returns, on="trade_date", how="outer")
    state = state.drop(columns=[f"{factor}_return" for factor in FACTOR_NAMES])
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
