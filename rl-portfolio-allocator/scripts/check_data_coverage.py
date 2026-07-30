"""数据覆盖前置检查(设计 §7.0):派生 fold 表并断言每段被必需数据源完整覆盖。

用途:在任何 walk-forward 研究开始前运行,确认 features 与 index 的实际覆盖区间,
据此生成/校验 fold 边界。index 覆盖不足的年份不得进入任何 fold 的状态构造。
"""
from __future__ import annotations
import argparse
import json
import pathlib
from dataclasses import dataclass, asdict

import pandas as pd

# market_state 中依赖真实 CSI300 指数收益的字段只能在 index 覆盖区间内严格因果构造。
# 因此凡包含这些字段的段(train/val/test),都必须落在 index 与 features 的交集内。
INDEX_DEPENDENT_FIELDS = ("market_vol_20", "market_vol_60", "market_drawdown", "vol_regime")

# 滚动窗口 warm-up:最长回看窗口(60 日)需要在有效起点之前先积累数据。
WARMUP_TRADING_DAYS = 60


@dataclass
class Coverage:
    source: str
    start: str
    end: str
    n_days: int


@dataclass
class Fold:
    fold: int
    train: tuple[str, str]
    val: tuple[str, str]
    test: tuple[str, str]


@dataclass
class SegmentCheck:
    fold: int
    segment: str
    start: str
    end: str
    covered: bool
    reason: str


def _dates(df: pd.DataFrame, col: str = "trade_date") -> pd.Series:
    if col not in df.columns:
        raise ValueError(f"missing date column: {col}")
    dates = pd.to_datetime(df[col], errors="coerce").dropna()
    dates = dates.sort_values().drop_duplicates().reset_index(drop=True)
    if dates.empty:
        raise ValueError(f"empty date source: {col}")
    return dates


def coverage_of(df: pd.DataFrame, source: str, col: str = "trade_date") -> Coverage:
    d = _dates(df, col)
    return Coverage(source=source, start=str(d.iloc[0].date()),
                    end=str(d.iloc[-1].date()), n_days=int(d.nunique()))


def effective_start(index_dates: pd.Series, warmup: int = WARMUP_TRADING_DAYS) -> pd.Timestamp:
    """index 有效状态起点:index 首日之后预留 warmup 个交易日给滚动窗口。"""
    uniq = index_dates.drop_duplicates().reset_index(drop=True)
    i = min(warmup, len(uniq) - 1)
    return uniq.iloc[i]


def _covers(seg: tuple[str, str], lo: pd.Timestamp, hi: pd.Timestamp) -> tuple[bool, str]:
    s, e = pd.Timestamp(seg[0]), pd.Timestamp(seg[1])
    if s < lo:
        return False, f"段起点 {seg[0]} 早于数据有效起点 {lo.date()}"
    if e > hi:
        return False, f"段终点 {seg[1]} 晚于数据终点 {hi.date()}"
    return True, "ok"


def check_folds(
    feats: pd.DataFrame,
    index: pd.DataFrame,
    folds: list[Fold],
    warmup: int = WARMUP_TRADING_DAYS,
) -> dict:
    """校验每个 fold 的 train/val/test 段是否被 features 与 index 完整覆盖。

    index-依赖的 market_state 字段要求各段落在 [index_effective_start, index_end] 内;
    features 要求各段落在 [features_start, features_end] 内。取两者更严的交集。
    """
    fdates, idates = _dates(feats), _dates(index)
    feat_lo, feat_hi = fdates.iloc[0], fdates.iloc[-1]
    idx_eff_lo = effective_start(idates, warmup)
    idx_hi = idates.iloc[-1]

    # 交集下界取更晚者(index 有效起点通常更晚),上界取更早者。
    lo = max(feat_lo, idx_eff_lo)
    hi = min(feat_hi, idx_hi)

    checks: list[SegmentCheck] = []
    fold_ok: dict[int, bool] = {}
    for f in folds:
        ok_all = True
        for seg_name, seg in (("train", f.train), ("val", f.val), ("test", f.test)):
            covered, reason = _covers(seg, lo, hi)
            checks.append(SegmentCheck(f.fold, seg_name, seg[0], seg[1], covered, reason))
            ok_all = ok_all and covered
        fold_ok[f.fold] = ok_all

    usable = [fid for fid, ok in fold_ok.items() if ok]
    return {
        "features_coverage": asdict(coverage_of(feats, "features")),
        "index_coverage": asdict(coverage_of(index, "index_returns")),
        "index_effective_start": str(idx_eff_lo.date()),
        "usable_intersection": {"start": str(lo.date()), "end": str(hi.date())},
        "warmup_trading_days": warmup,
        "index_dependent_fields": list(INDEX_DEPENDENT_FIELDS),
        "segment_checks": [asdict(c) for c in checks],
        "fold_ok": fold_ok,
        "usable_folds": usable,
        "folds": [
            {
                "fold": f.fold,
                "train": list(f.train),
                "val": list(f.val),
                "test": list(f.test),
            }
            for f in folds
        ],
    }


def default_folds() -> list[Fold]:
    """设计 §7.1 的默认切分(仅示例;真实边界由本检查按实际交易日派生)。

    训练起点用 2010-05-01,已让过 index 有效起点(index 首日 + 60 日 warmup ≈ 2010-04)。
    """
    return [
        Fold(1, ("2010-05-01", "2016-12-31"), ("2017-01-01", "2018-12-31"), ("2019-01-01", "2020-12-31")),
        Fold(2, ("2010-05-01", "2020-12-31"), ("2021-01-01", "2021-12-31"), ("2022-01-01", "2022-12-31")),
        Fold(3, ("2010-05-01", "2022-12-31"), ("2023-01-01", "2023-12-31"), ("2024-01-01", "2024-12-31")),
    ]


def main() -> int:
    root = pathlib.Path(__file__).resolve().parent.parent
    p = argparse.ArgumentParser(description="数据覆盖前置检查 (§7.0)")
    p.add_argument("--features", default=str(root / "data" / "features.parquet"))
    p.add_argument("--index", default=str(root / "data" / "index_returns.parquet"))
    p.add_argument("--min-folds", type=int, default=3,
                   help="硬门槛所需最少可用 fold 数(默认 3)")
    p.add_argument("--warmup", type=int, default=WARMUP_TRADING_DAYS)
    p.add_argument("--json", default=None, help="报告落盘路径 (JSON)")
    args = p.parse_args()

    feats = pd.read_parquet(args.features)
    index = pd.read_parquet(args.index)
    report = check_folds(feats, index, default_folds(), warmup=args.warmup)

    print("=== 数据覆盖 ===")
    fc, ic = report["features_coverage"], report["index_coverage"]
    print(f"features       : {fc['start']} ~ {fc['end']}  ({fc['n_days']} 日)")
    print(f"index_returns  : {ic['start']} ~ {ic['end']}  ({ic['n_days']} 日)")
    print(f"index 有效起点 : {report['index_effective_start']}  (warmup={report['warmup_trading_days']}d)")
    ui = report["usable_intersection"]
    print(f"可用交集       : {ui['start']} ~ {ui['end']}")
    print("\n=== Fold 段覆盖 ===")
    for c in report["segment_checks"]:
        mark = "OK " if c["covered"] else "!! "
        print(f"  {mark}fold{c['fold']} {c['segment']:5s} {c['start']} ~ {c['end']}  {c['reason']}")
    print(f"\n可用 fold: {report['usable_folds']}  (需要 >= {args.min_folds})")

    if args.json:
        pathlib.Path(args.json).write_text(json.dumps(report, ensure_ascii=False, indent=2))
        print(f"报告已落盘: {args.json}")

    enough = len(report["usable_folds"]) >= args.min_folds
    if not enough:
        print("\n数据不足,无法评估:可用 fold 数少于门槛要求,不得用退化数据凑数。")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
