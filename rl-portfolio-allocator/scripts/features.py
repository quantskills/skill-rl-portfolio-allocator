"""从 panda_data 加载 CSI300 成分与后复权量价,产生 K 因子横截面 z-score 表。

因子(K=6,与 config.FACTOR_NAMES 严格对齐):
  mom_20       ln(close_t / close_{t-20})
  reversal_5   -ln(close_t / close_{t-5})
  vol_20       std(ret_1d, 20)
  turnover_20  mean(volume / shares_float_proxy, 20)
  amihud_20    mean(|ret_1d| / (amount + eps), 20)
  ret_skew_60  skew(ret_1d, 60)

t 日只用 ≤ t 的量价 → 严格 shift(1) 或 rolling().last() 结构。
每日横截面 z-score(减均值除标准差)再 clip 到 [-3, 3]。
"""
from __future__ import annotations
import os
import pathlib
from typing import Optional
import numpy as np
import pandas as pd

from scripts.config import FACTOR_NAMES, get_config

_EPS = 1e-12


def _ln_return(close: pd.Series, n: int) -> pd.Series:
    return np.log(close / close.shift(n))


def _compute_single_symbol(g: pd.DataFrame) -> pd.DataFrame:
    g = g.sort_values("trade_date").copy()
    close = g["close"]
    volume = g["volume"]
    amount = g["amount"]
    ret_1d = close.pct_change()
    g["ret_1d"] = ret_1d
    g["mom_20"] = _ln_return(close, 20)
    g["reversal_5"] = -_ln_return(close, 5)
    g["vol_20"] = ret_1d.rolling(20).std()
    g["turnover_20"] = (volume / volume.rolling(252).mean()).rolling(20).mean()
    g["amihud_20"] = (ret_1d.abs() / (amount + _EPS)).rolling(20).mean()
    g["ret_skew_60"] = ret_1d.rolling(60).skew()
    return g


def _cross_sectional_zscore(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        grp = out.groupby("trade_date")[c]
        mu = grp.transform("mean")
        sd = grp.transform("std")
        z = (out[c] - mu) / (sd + _EPS)
        out[c] = z.clip(-3.0, 3.0)
    return out


def compute_factors(prices: pd.DataFrame) -> pd.DataFrame:
    """输入长表 prices(trade_date, symbol, OHLCV+amount+is_suspended),输出因子长表。"""
    required = {"trade_date", "symbol", "open", "high", "low", "close", "volume", "amount"}
    missing = required - set(prices.columns)
    if missing:
        raise ValueError(f"prices missing columns: {missing}")
    if "is_suspended" not in prices.columns:
        prices = prices.assign(is_suspended=False)
    df = prices.groupby("symbol", group_keys=False).apply(_compute_single_symbol)
    df = _cross_sectional_zscore(df, FACTOR_NAMES)
    keep = ["trade_date", "symbol", "ret_1d", "is_suspended", *FACTOR_NAMES]
    return df[keep].reset_index(drop=True)


def load_universe(start: str, end: Optional[str]) -> pd.DataFrame:
    import panda_data
    return panda_data.get_index_component("000300.SH", start_date=start, end_date=end)


def load_prices(symbols: list[str], start: str, end: Optional[str]) -> pd.DataFrame:
    import panda_data
    df = panda_data.get_stock_daily_post(
        symbols=symbols, start_date=start, end_date=end
    )
    if "is_suspended" not in df.columns:
        df["is_suspended"] = df["volume"].fillna(0) == 0
    return df


def save_features(df: pd.DataFrame, path: pathlib.Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)


def load_features(path: pathlib.Path) -> pd.DataFrame:
    return pd.read_parquet(path)


def main() -> None:
    cfg = get_config()
    start, end = cfg["start_date"], cfg["end_date"]
    universe = load_universe(start, end)
    symbols = sorted(universe["symbol"].unique().tolist())
    prices = load_prices(symbols, start, end)
    feats = compute_factors(prices)
    out = pathlib.Path(__file__).resolve().parent.parent / "data" / "features.parquet"
    save_features(feats, out)
    print(
        f"features saved: {out}  rows={len(feats)}  "
        f"dates={feats['trade_date'].min()}..{feats['trade_date'].max()}  "
        f"symbols={feats['symbol'].nunique()}  factors={len(FACTOR_NAMES)}"
    )


if __name__ == "__main__":
    main()
