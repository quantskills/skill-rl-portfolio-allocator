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
from scripts.factor_cache import BASE_COLUMNS, write_factor_cache
from scripts.factor_catalog import FACTOR_CATALOG
from scripts.factor_compute import compute_factor_panel

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
    symbol_frames = [
        _compute_single_symbol(group)
        for _, group in prices.groupby("symbol", sort=True)
    ]
    df = pd.concat(symbol_frames, axis=0) if symbol_frames else prices.copy()
    df = _cross_sectional_zscore(df, FACTOR_NAMES)
    keep = ["trade_date", "symbol", "ret_1d", "is_suspended", *FACTOR_NAMES]
    return df[keep].reset_index(drop=True)


def _date_chunks(start: str, end: str, max_years: int = 4) -> list[tuple[str, str]]:
    """将长时间区间拆分为不超过 max_years 的子区间（API 限制最多 5 年）。"""
    from datetime import date, timedelta
    s = date.fromisoformat(start)
    e = date.fromisoformat(end)
    chunks = []
    while s <= e:
        chunk_end = min(s.replace(year=s.year + max_years) - timedelta(days=1), e)
        chunks.append((s.isoformat(), chunk_end.isoformat()))
        s = chunk_end + timedelta(days=1)
    return chunks


def load_universe(start: str, end: Optional[str]) -> pd.DataFrame:
    import panda_data
    end = end or "2024-12-31"
    chunks = _date_chunks(start, end)
    dfs = []
    for cs, ce in chunks:
        df = panda_data.get_stock_daily_post(
            symbol="",
            start_date=cs.replace("-", ""),
            end_date=ce.replace("-", ""),
            indicator="000300"
        )
        if not df.empty:
            dfs.append(df[["symbol"]].drop_duplicates())
    if not dfs:
        raise ValueError("沪深300成分数据为空，请检查网络连接和 panda_data 服务")
    return pd.concat(dfs).drop_duplicates().reset_index(drop=True)


def load_prices(symbols: list[str], start: str, end: Optional[str]) -> pd.DataFrame:
    import panda_data
    end = end or "2024-12-31"
    chunks = _date_chunks(start, end)
    dfs = []
    for cs, ce in chunks:
        print(f"  loading prices {cs} ~ {ce} ...")
        df = panda_data.get_stock_daily_post(
            symbol="",
            start_date=cs.replace("-", ""),
            end_date=ce.replace("-", ""),
            indicator="000300"
        )
        if not df.empty:
            dfs.append(df)
    if not dfs:
        raise ValueError("价格数据为空")
    df = pd.concat(dfs, ignore_index=True).drop_duplicates(subset=["symbol", "date"])
    # 重命名日期列和其他必要字段
    df = df.rename(columns={"date": "trade_date"})
    # 确保有 amount 列（如果没有就用 close * volume 估算）
    if "amount" not in df.columns:
        df["amount"] = df["close"] * df["volume"]
    if "is_suspended" not in df.columns:
        df["is_suspended"] = df["volume"].fillna(0) == 0
    return df


def save_features(df: pd.DataFrame, path: pathlib.Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)


def load_features(path: pathlib.Path) -> pd.DataFrame:
    return pd.read_parquet(path)


def main() -> None:
    import panda_data
    cfg = get_config()

    # Initialize panda_data
    username = cfg.get("panda_username")
    password = cfg.get("panda_password")
    if not username or not password:
        raise ValueError("PANDA_DATA_USERNAME and PANDA_DATA_PASSWORD must be set")

    panda_data.init_token(username, password)

    start, end = cfg["start_date"], cfg["end_date"]
    universe = load_universe(start, end)
    symbols = sorted(universe["symbol"].unique().tolist())
    prices = load_prices(symbols, start, end)
    prices["trade_date"] = pd.to_datetime(prices["trade_date"])

    full_panel = compute_factor_panel(prices)
    factor_names = [spec.name for spec in FACTOR_CATALOG]
    full_panel = full_panel.loc[:, [*BASE_COLUMNS, *factor_names]]
    factor_root = pathlib.Path(__file__).resolve().parent.parent / "data" / "factors"
    write_factor_cache(full_panel, factor_root)
    print(f"factor cache saved: {factor_root}  rows={len(full_panel)}  factors={len(factor_names)}")

    feats = compute_factors(prices)

    # Filter out early dates where we don't have enough history for turnover_20 calculation
    # turnover_20 needs 20-day rolling window + 252-day average, so need ~270 days of history
    feats["trade_date"] = pd.to_datetime(feats["trade_date"])
    min_date = pd.to_datetime(start) + pd.Timedelta(days=270)
    feats = feats[feats["trade_date"] >= min_date].reset_index(drop=True)

    # Forward fill remaining NaN in factors (from individual stock calculations)
    for col in FACTOR_NAMES:
        if col in feats.columns:
            feats[col] = feats.groupby("symbol")[col].ffill()

    out = pathlib.Path(__file__).resolve().parent.parent / "data" / "features.parquet"
    save_features(feats, out)
    print(
        f"features saved: {out}  rows={len(feats)}  "
        f"dates={feats['trade_date'].min()}..{feats['trade_date'].max()}  "
        f"symbols={feats['symbol'].nunique()}  factors={len(FACTOR_NAMES)}"
    )

    # Generate index_returns (CSI300 daily returns)
    print("  loading CSI300 index daily data ...")
    idx_chunks = _date_chunks(start, end)
    idx_dfs = []
    for cs, ce in idx_chunks:
        idf = panda_data.get_index_daily(
            symbol="000300.SH",
            start_date=cs.replace("-", ""),
            end_date=ce.replace("-", ""),
        )
        if not idf.empty:
            idx_dfs.append(idf)
    if idx_dfs:
        idx_df = pd.concat(idx_dfs, ignore_index=True).drop_duplicates(subset=["date"]).sort_values("date")
        idx_df["trade_date"] = pd.to_datetime(idx_df["date"], format="%Y%m%d")
        idx_df["ret"] = idx_df["close"].pct_change()
        idx_out = out.parent / "index_returns.parquet"
        idx_df[["trade_date", "ret"]].dropna().to_parquet(idx_out, index=False)
        print(f"  index_returns saved: {idx_out}  rows={len(idx_df)-1}  "
              f"dates={idx_df['trade_date'].min().date()}..{idx_df['trade_date'].max().date()}")
    else:
        print("  WARNING: failed to load CSI300 index data")


if __name__ == "__main__":
    main()
