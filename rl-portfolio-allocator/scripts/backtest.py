"""主回测(用途②):训早期→测样本外,RL vs 三基线,含成本、含诊断。"""
from __future__ import annotations
import argparse
import pathlib
from typing import Optional

import numpy as np
import pandas as pd

from scripts.config import get_config, FACTOR_NAMES, K
from scripts.env import PortfolioEnv
from scripts.train import train_ppo, select_device
from scripts.baselines import (
    equal_weight_rollout, long_only_topn_rollout, static_factor_equal_rollout
)
from scripts.metrics import metrics_pack
from scripts.diagnostics import summarize_rollout, check_degeneracy


def run_ppo_rollout(model, env) -> tuple[np.ndarray, list, list]:
    obs, _ = env.reset(seed=0)
    infos, rets, dates, done = [], [], [], False
    while not done:
        signal_date = env.dates[env.t]
        act, _ = model.predict(obs, deterministic=True)
        obs, _, term, trunc, info = env.step(act)
        infos.append(info); rets.append(info["net_ret"]); dates.append(signal_date)
        done = term or trunc
    return np.asarray(rets), infos, dates


def run_backtest(
    features_df: pd.DataFrame,
    cfg: dict,
    train_start, train_end, test_start, test_end,
    timesteps: int = 100_000, seed: int = 0,
    save_path: Optional[str] = None,
) -> dict:
    idx = pd.Series(np.zeros(1), index=pd.to_datetime([features_df["trade_date"].min()]))
    train_env = PortfolioEnv(features_df, idx, cfg, train_start, train_end)
    device = select_device(cfg["train_device"])
    model = train_ppo(train_env, total_timesteps=timesteps, seed=seed, device=device, save_path=save_path)

    test_env = PortfolioEnv(features_df, idx, cfg, test_start, test_end)
    rl_rets, infos, rl_dates = run_ppo_rollout(model, test_env)

    ew = equal_weight_rollout(features_df, cfg, test_start, test_end)
    lo = long_only_topn_rollout(features_df, cfg, test_start, test_end, np.ones(K) / K)
    sf = static_factor_equal_rollout(features_df, cfg, test_start, test_end)

    m = {
        "rl": metrics_pack(rl_rets, "rl"),
        "equal_weight": metrics_pack(ew, "equal_weight"),
        "long_only_topn": metrics_pack(lo, "long_only_topn"),
        "static_factor_equal": metrics_pack(sf, "static_factor_equal"),
    }
    diag = summarize_rollout(infos)
    warns = check_degeneracy(diag, cfg)

    research_ok = (m["rl"]["sharpe"] > m["static_factor_equal"]["sharpe"]
                   and m["rl"]["calmar"] > m["static_factor_equal"]["calmar"])
    return {
        "metrics": m, "diagnostics": diag, "warnings": warns,
        "research_ok": bool(research_ok),
        "rl_daily_rets": rl_rets, "rl_dates": rl_dates,
    }


def main() -> None:
    cfg = get_config()
    root = pathlib.Path(__file__).resolve().parent.parent
    feats = pd.read_parquet(root / "data" / "features.parquet")
    p = argparse.ArgumentParser()
    p.add_argument("--train-start", default="2010-01-01")
    p.add_argument("--train-end", default="2022-12-31")
    p.add_argument("--test-start", default="2023-01-01")
    p.add_argument("--test-end", default=cfg["end_date"] or "2099-12-31")
    p.add_argument("--timesteps", type=int, default=200_000)
    args = p.parse_args()

    res = run_backtest(
        feats, cfg, args.train_start, args.train_end, args.test_start, args.test_end,
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
