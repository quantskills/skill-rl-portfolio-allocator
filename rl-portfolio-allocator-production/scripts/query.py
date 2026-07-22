"""只读查询已落盘的 RL 组合持仓。绝不重训、不联网。"""
from __future__ import annotations
import argparse
import pathlib
import pandas as pd

_DEFAULT_PATH = pathlib.Path(__file__).resolve().parent.parent / "data" / "allocations.parquet"


def load_allocations(path=None) -> pd.DataFrame:
    p = pathlib.Path(path) if path else _DEFAULT_PATH
    if not p.exists():
        raise FileNotFoundError(f"allocations parquet not found: {p}. "
                                f"Run `../rl-portfolio-allocator/scripts/allocate.py --retrain` first.")
    return pd.read_parquet(p)


def get_latest(df: pd.DataFrame) -> pd.DataFrame:
    latest_date = df["trade_date"].max()
    return df[df["trade_date"] == latest_date].reset_index(drop=True)


def get_range(df: pd.DataFrame, start, end) -> pd.DataFrame:
    s = pd.Timestamp(start); e = pd.Timestamp(end)
    m = (df["trade_date"] >= s) & (df["trade_date"] <= e)
    return df.loc[m].reset_index(drop=True)


def main() -> None:
    p = argparse.ArgumentParser()
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--latest", action="store_true")
    g.add_argument("--range", nargs=2, metavar=("START", "END"))
    p.add_argument("--path", default=None)
    args = p.parse_args()
    df = load_allocations(args.path)
    out = get_latest(df) if args.latest else get_range(df, args.range[0], args.range[1])
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
