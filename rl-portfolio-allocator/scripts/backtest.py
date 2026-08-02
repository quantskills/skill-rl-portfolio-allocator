"""主回测(用途②):训早期→测样本外,RL vs 三基线,含成本、含诊断。"""
from __future__ import annotations
import argparse
import pathlib
from typing import Optional

import numpy as np
import pandas as pd

from scripts.config import get_config, FACTOR_NAMES, K
from scripts.env import PortfolioEnv, effective_range
from scripts.train import train_ppo, select_device
from scripts.market_state import compute_daily_factor_returns
from scripts.baselines import (
    equal_weight_rollout, long_only_topn_rollout, static_factor_equal_rollout,
    static_factor_optimized_rollout, rolling_ic_rollout,
)
from scripts.metrics import metrics_pack
from scripts.diagnostics import summarize_rollout, check_degeneracy


def run_ppo_rollout(model, env) -> tuple[np.ndarray, list, list]:
    obs, _ = env.reset(seed=0)
    infos, rets, dates, done = [], [], [], False
    while not done:
        act, _ = model.predict(obs, deterministic=True)
        obs, _, term, trunc, info = env.step(act)
        infos.append(info)
        rets.extend(info["daily_net_rets"])
        dates.extend(info["settlement_dates"])
        done = term or trunc
    return np.asarray(rets), infos, dates


def run_backtest(
    features_df: pd.DataFrame,
    market_state_df: pd.DataFrame,
    cfg: dict,
    train_start, train_end, test_start, test_end,
    timesteps: int = 100_000, seed: int = 0,
    save_path: Optional[str] = None,
    online_retrain_interval: Optional[int] = None,
) -> dict:
    train_env = PortfolioEnv(features_df, market_state_df, cfg, train_start, train_end)
    device = select_device(cfg["train_device"])
    model = train_ppo(train_env, total_timesteps=timesteps, seed=seed, device=device, save_path=save_path)

    test_env = PortfolioEnv(features_df, market_state_df, cfg, test_start, test_end)
    rl_rets, infos, rl_dates = run_ppo_rollout(model, test_env)

    train_features = features_df.copy()
    train_features["trade_date"] = pd.to_datetime(train_features["trade_date"])
    train_features = train_features[
        (train_features["trade_date"] >= pd.Timestamp(train_start))
        & (train_features["trade_date"] <= pd.Timestamp(train_end))
    ]
    train_factor_returns = compute_daily_factor_returns(train_features, cfg).rename(
        columns={f"{factor}_factor_ret": factor for factor in FACTOR_NAMES}
    )

    # Online retraining: if specified interval (e.g., 252 days = 1 year), retrain on rolling window
    if online_retrain_interval and online_retrain_interval > 0:
        test_dates = sorted(pd.to_datetime(test_env.dates).unique())
        for i in range(online_retrain_interval, len(test_dates)):
            retrain_end = test_dates[i]
            if i % online_retrain_interval == 0:  # Retrain every N trading days
                retrain_start = max(test_dates[0], retrain_end - pd.Timedelta(days=365*2))  # 2-year window
                retrain_env = PortfolioEnv(features_df, market_state_df, cfg, retrain_start, retrain_end)
                model = train_ppo(retrain_env, total_timesteps=timesteps // 2, seed=seed, device=device)  # Lighter retraining

    ew = equal_weight_rollout(features_df, cfg, test_start, test_end)
    lo = long_only_topn_rollout(features_df, cfg, test_start, test_end, np.ones(K) / K)
    sf = static_factor_equal_rollout(features_df, cfg, test_start, test_end)
    so = static_factor_optimized_rollout(
        features_df, cfg, test_start, test_end, train_factor_returns
    )
    ric = rolling_ic_rollout(features_df, market_state_df, cfg, test_start, test_end)

    m = {
        "rl": metrics_pack(rl_rets, "rl"),
        "equal_weight": metrics_pack(ew, "equal_weight"),
        "long_only_topn": metrics_pack(lo, "long_only_topn"),
        "static_factor_equal": metrics_pack(sf, "static_factor_equal"),
        "static_factor_optimized": metrics_pack(so, "static_factor_optimized"),
        "rolling_ic": metrics_pack(ric, "rolling_ic"),
    }
    diag = summarize_rollout(infos)
    warns = check_degeneracy(diag, cfg)

    return {
        "metrics": m, "diagnostics": diag, "warnings": warns,
        "research_ok": False,
        "research_status": "single_split_is_diagnostic_only",
        "rl_daily_rets": rl_rets, "rl_dates": rl_dates,
    }


def main() -> None:
    cfg = get_config()
    root = pathlib.Path(__file__).resolve().parent.parent
    feats = pd.read_parquet(root / "data" / "features.parquet")
    market_state = pd.read_parquet(root / "data" / "market_state.parquet")
    p = argparse.ArgumentParser()
    p.add_argument("--train-start", default=None)
    p.add_argument("--train-end", default=None)
    p.add_argument("--test-start", default=None)
    p.add_argument("--test-end", default=None)
    p.add_argument("--timesteps", type=int, default=200_000)
    p.add_argument("--train-ratio", type=float, default=0.7,
                   help="fraction of data dates used for training (default 0.7)")
    args = p.parse_args()

    # Auto-detect train/test split from actual data dates if not explicitly provided
    causal_start, causal_end = effective_range(
        feats, market_state, cfg["start_date"], cfg["end_date"] or "2099-12-31", cfg=cfg
    )
    dates = sorted(
        d for d in pd.to_datetime(feats["trade_date"]).unique()
        if causal_start <= d <= causal_end
    )
    if args.test_end is None:
        args.test_end = str(causal_end.date())
    if args.train_start is None:
        args.train_start = str(causal_start.date())
    if args.train_end is None or args.test_start is None:
        split_idx = int(len(dates) * args.train_ratio)
        split_date = dates[split_idx]
        if args.train_end is None:
            args.train_end = str((split_date - pd.Timedelta(days=1)).date())
        if args.test_start is None:
            args.test_start = str(split_date.date())

    print(f"effective range: {causal_start.date()} ~ {causal_end.date()}")
    print(f"Train: {args.train_start} ~ {args.train_end}  |  Test: {args.test_start} ~ {args.test_end}")

    res = run_backtest(
        feats, market_state, cfg, args.train_start, args.train_end, args.test_start, args.test_end,
        timesteps=args.timesteps, seed=0,
        save_path=str(root / "checkpoints" / "backtest.zip"),
    )
    print("=== Backtest metrics ===")
    for name, m in res["metrics"].items():
        print(f"{name:20s}  ARR={m['arr']*100:7.2f}%  Sharpe={m['sharpe']:6.2f}  "
              f"MDD={m['mdd']*100:7.2f}%  Calmar={m['calmar']:6.2f}")
    print(f"\nresearch_ok={res['research_ok']}")
    if res["warnings"]:
        for w in res["warnings"]:
            print(f"  ! {w}")


if __name__ == "__main__":
    main()
