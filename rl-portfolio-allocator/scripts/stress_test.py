"""压力测试(用途①):四段前向,每段独立训练 [数据起点, 段前]。
数据不足则明确报告并跳过,不伪造替代。除全 test 段外额外报告"核心段"指标。"""
from __future__ import annotations
import argparse
import json
import pathlib
import sys
import numpy as np
import pandas as pd

if __package__ in (None, ""):
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from scripts.config import get_config
from scripts.backtest import run_backtest
from scripts.baselines import (
    equal_weight_rollout, long_only_topn_rollout, static_factor_equal_rollout
)
from scripts.metrics import metrics_pack
from scripts.config import K


STRESS_SEGMENTS = [
    {
        "name": "2008_gfc",
        "train_end": "2007-06-30",
        "test_start": "2007-07-01", "test_end": "2009-03-31",
        "core_start": "2008-09-01", "core_end": "2009-03-31",
        "required_min_years": 2,
    },
    {
        "name": "2015_ashare_crash",
        "train_end": "2015-05-31",
        "test_start": "2015-06-12", "test_end": "2015-09-30",
        "core_start": "2015-06-12", "core_end": "2015-09-30",
        "required_min_years": 3,
    },
    {
        "name": "2020_covid",
        "train_end": "2020-01-31",
        "test_start": "2020-02-19", "test_end": "2020-04-30",
        "core_start": "2020-02-19", "core_end": "2020-04-30",
        "required_min_years": 3,
    },
    {
        "name": "2022_double_kill",
        "train_end": "2021-12-31",
        "test_start": "2022-01-01", "test_end": "2022-12-31",
        "core_start": "2022-01-01", "core_end": "2022-12-31",
        "required_min_years": 3,
    },
]


def load_frozen_method(path: str) -> dict:
    method = json.loads(pathlib.Path(path).read_text(encoding="utf-8"))
    if "frozen_method" in method:
        method = method["frozen_method"]
    elif "method_by_fold" in method:
        method = method["method_by_fold"][sorted(method["method_by_fold"])[-1]]
    if not method.get("frozen_candidate"):
        raise ValueError("method artifact must contain frozen_candidate")
    return method


def apply_frozen_method(cfg: dict, method: dict, default_budget: int) -> tuple[dict, int]:
    """Apply frozen training/evaluation metadata before running stress segments."""
    applied = dict(cfg)
    for key in ("reward_variant", "buffer_variant", "schema_version"):
        if key in method:
            applied[key] = method[key]
    buffer_config = method.get("buffer_config")
    if buffer_config is None:
        raise ValueError("frozen method artifact must contain buffer_config")
    applied.update(buffer_config)
    budget = int(method.get("training_budget", default_budget))
    if budget <= 0:
        raise ValueError("training_budget must be positive")
    return applied, budget


def _has_enough(feats: pd.DataFrame, market_state: pd.DataFrame,
                train_end: str, min_years: int) -> tuple[bool, str, str]:
    d = pd.to_datetime(feats["trade_date"])
    state_dates = (pd.to_datetime(market_state["trade_date"]) if "trade_date" in market_state
                   else pd.to_datetime(market_state.index))
    common = pd.DatetimeIndex(d.unique()).intersection(state_dates.unique())
    if len(common) == 0:
        return False, "n/a", "no feature/market_state date intersection"
    data_start = common.min()
    train_end_ts = pd.Timestamp(train_end)
    if data_start >= train_end_ts:
        return False, str(data_start.date()), f"data_start {data_start.date()} >= train_end {train_end}"
    years = (train_end_ts - data_start).days / 365.25
    if years < min_years:
        return False, str(data_start.date()), f"only {years:.1f} years before {train_end}, need {min_years}"
    return True, str(data_start.date()), ""


def _core_metrics(features_df: pd.DataFrame, cfg: dict,
                   rl_daily_rets: np.ndarray, rl_dates: list,
                   core_start, core_end) -> dict:
    core_s = pd.Timestamp(core_start); core_e = pd.Timestamp(core_end)
    mask = np.array([core_s <= pd.Timestamp(d) <= core_e for d in rl_dates])
    rl_core = rl_daily_rets[mask] if mask.any() else np.asarray([])
    ew = equal_weight_rollout(features_df, cfg, core_start, core_end)
    lo = long_only_topn_rollout(features_df, cfg, core_start, core_end, np.ones(K) / K)
    sf = static_factor_equal_rollout(features_df, cfg, core_start, core_end)
    return {
        "rl": metrics_pack(rl_core, "rl_core"),
        "equal_weight": metrics_pack(ew, "equal_weight_core"),
        "long_only_topn": metrics_pack(lo, "long_only_topn_core"),
        "static_factor_equal": metrics_pack(sf, "static_factor_equal_core"),
    }


def run_all_stress(features_df: pd.DataFrame, market_state_df: pd.DataFrame,
                   cfg: dict, timesteps: int = 100_000, method: dict | None = None) -> list:
    if method is not None:
        cfg, timesteps = apply_frozen_method(cfg, method, timesteps)
    out = []
    for seg in STRESS_SEGMENTS:
        ok, data_start, reason = _has_enough(
            features_df, market_state_df, seg["train_end"], seg["required_min_years"]
        )
        if not ok:
            out.append({"name": seg["name"], "skipped": True, "reason": reason})
            continue
        res = run_backtest(
            features_df=features_df, market_state_df=market_state_df, cfg=cfg,
            train_start=data_start, train_end=seg["train_end"],
            test_start=seg["test_start"], test_end=seg["test_end"],
            timesteps=timesteps, seed=0,
        )
        core = _core_metrics(
            features_df, cfg,
            rl_daily_rets=res["rl_daily_rets"], rl_dates=res["rl_dates"],
            core_start=seg["core_start"], core_end=seg["core_end"],
        )
        record = {"name": seg["name"], "skipped": False, "core_metrics": core, **res}
        if method is not None:
            # This is an evaluation-only annotation. Candidate selection is
            # deliberately absent from the stress path.
            record["frozen_candidate"] = method["frozen_candidate"]
            record["report_type"] = "frozen_method_stress"
            record["method_config"] = {
                "reward_variant": cfg.get("reward_variant"),
                "buffer_variant": cfg.get("buffer_variant"),
                "schema_version": cfg.get("schema_version"),
                "training_budget": timesteps,
            }
        out.append(record)
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--timesteps", type=int, default=100_000)
    p.add_argument("--method", default=None,
                   help="frozen walk-forward method JSON; never reselects a candidate")
    p.add_argument("--report", default=None,
                   help="optional JSON path for the separate stress report")
    args = p.parse_args()
    cfg = get_config()
    root = pathlib.Path(__file__).resolve().parent.parent
    feats = pd.read_parquet(root / "data" / "features.parquet")
    market_state = pd.read_parquet(root / "data" / "market_state.parquet")
    method = load_frozen_method(args.method) if args.method else None
    results = run_all_stress(feats, market_state, cfg, timesteps=args.timesteps, method=method)
    if args.report:
        pathlib.Path(args.report).parent.mkdir(parents=True, exist_ok=True)
        pathlib.Path(args.report).write_text(
            json.dumps({"report_type": "frozen_method_stress" if method else "stress", "results": results},
                       default=str, indent=2), encoding="utf-8"
        )
    print("=== Stress Test ===")
    for r in results:
        if r.get("skipped"):
            print(f"[SKIP] {r['name']:20s}  reason={r['reason']}")
            continue
        m = r["metrics"]["rl"]; b = r["metrics"]["static_factor_equal"]
        cm = r["core_metrics"]["rl"]; cb = r["core_metrics"]["static_factor_equal"]
        print(f"{r['name']:20s}  [full test]")
        print(f"  RL:  ARR={m['arr']*100:7.2f}%  Sharpe={m['sharpe']:6.2f}  MDD={m['mdd']*100:7.2f}%  Calmar={m['calmar']:6.2f}")
        print(f"  SFE: ARR={b['arr']*100:7.2f}%  Sharpe={b['sharpe']:6.2f}  MDD={b['mdd']*100:7.2f}%  Calmar={b['calmar']:6.2f}")
        print(f"  [core]  RL:  ARR={cm['arr']*100:7.2f}%  Sharpe={cm['sharpe']:6.2f}  MDD={cm['mdd']*100:7.2f}%  Calmar={cm['calmar']:6.2f}")
        print(f"          SFE: ARR={cb['arr']*100:7.2f}%  Sharpe={cb['sharpe']:6.2f}  MDD={cb['mdd']*100:7.2f}%  Calmar={cb['calmar']:6.2f}")
        for w in r.get("warnings", []):
            print(f"  ! {w}")


if __name__ == "__main__":
    main()
