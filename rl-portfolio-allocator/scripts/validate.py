"""规范性校验:schema / 权重上限 / 无未来函数。不通过不得进入生产。"""
from __future__ import annotations
import argparse
import json
import pathlib
import sys
import pandas as pd

from scripts.config import get_config, STRATEGY_ID, DATA_VERSION

REQUIRED = [
    "trade_date", "symbol", "weight", "side", "factor_weights",
    "composite_score", "strategy_id", "data_version", "update_time",
]
_TOL = 1e-6


def validate_schema(df: pd.DataFrame) -> list:
    errs = []
    for c in REQUIRED:
        if c not in df.columns:
            errs.append(f"missing column: {c}")
    if errs:
        return errs
    if not (df["strategy_id"] == STRATEGY_ID).all():
        errs.append(f"strategy_id must all be {STRATEGY_ID!r}")
    if not (df["data_version"] == DATA_VERSION).all():
        errs.append(f"data_version must all be {DATA_VERSION!r}")
    if not df["side"].isin(["long", "short", "cash"]).all():
        errs.append("side must be in {long, short, cash}")
    for i, fw in enumerate(df["factor_weights"]):
        try:
            json.loads(fw)
        except Exception:
            errs.append(f"row {i}: factor_weights is not valid JSON")
            break
    return errs


def validate_weights(df: pd.DataFrame, cfg: dict) -> list:
    errs = []
    for date, g in df.groupby("trade_date"):
        long_sum = g.loc[g["side"] == "long", "weight"].sum()
        short_sum = -g.loc[g["side"] == "short", "weight"].sum()
        if long_sum > cfg["long_notional"] + _TOL:
            errs.append(f"{date}: long notional {long_sum:.4f} > cap {cfg['long_notional']}")
        if short_sum > cfg["short_notional_cap"] + _TOL:
            errs.append(f"{date}: short notional {short_sum:.4f} > cap {cfg['short_notional_cap']}")
        stock_sum = float(g.loc[g["side"].isin(["long", "short"]), "weight"].sum())
        cash_sum = float(g.loc[g["side"] == "cash", "weight"].sum())
        if abs(stock_sum + cash_sum - 1.0) > _TOL:
            errs.append(f"{date}: cash identity stock + cash weights must equal 1")
        if not g["weight"].apply(lambda x: x == x and abs(x) < 1e9).all():
            errs.append(f"{date}: non-finite weight detected")
    return errs


def validate_no_future(df: pd.DataFrame) -> list:
    errs = []
    for _, row in df.iterrows():
        td = pd.Timestamp(row["trade_date"]).normalize()
        try:
            ut = pd.Timestamp(row["update_time"]).tz_localize(None) if pd.Timestamp(row["update_time"]).tzinfo else pd.Timestamp(row["update_time"])
        except Exception:
            errs.append(f"row {row['symbol']}@{td}: bad update_time {row['update_time']!r}")
            continue
        if ut.normalize() < td:
            errs.append(f"{row['symbol']}@{td}: update_time {ut} earlier than trade_date {td} (future function)")
    return errs


def run_all(path: str, cfg: dict) -> tuple:
    df = pd.read_parquet(path)
    errs = validate_schema(df) + validate_weights(df, cfg) + validate_no_future(df)
    return (len(errs) == 0), errs


def main() -> None:
    cfg = get_config()
    root = pathlib.Path(__file__).resolve().parent.parent
    default_path = root.parent / "rl-portfolio-allocator-production" / "data" / "allocations.parquet"
    p = argparse.ArgumentParser()
    p.add_argument("--path", default=str(default_path))
    args = p.parse_args()
    ok, errs = run_all(args.path, cfg)
    if ok:
        print(f"[OK] {args.path} validates")
        sys.exit(0)
    print(f"[FAIL] {args.path}")
    for e in errs:
        print(f"  - {e}")
    sys.exit(1)


if __name__ == "__main__":
    main()
