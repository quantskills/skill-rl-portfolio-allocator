"""生产模型:--retrain 用数据起点~最新日全部数据训 PPO;--infer-only 复用现有模型每日推理。
落盘持仓表到 rl-portfolio-allocator-production/data/allocations.parquet。"""
from __future__ import annotations
import argparse
import json
import pathlib
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from scripts.config import (
    get_config, FACTOR_NAMES, K, STRATEGY_ID, DATA_VERSION
)
from scripts.env import PortfolioEnv, effective_range
from scripts.train import train_ppo, load_ppo, select_device


def retrain_production(features_df: pd.DataFrame, cfg: dict, timesteps: int,
                        seed: int, checkpoint_path: str,
                        market_state_df: pd.DataFrame) -> str:
    dates = pd.to_datetime(features_df["trade_date"])
    start, end = effective_range(features_df, market_state_df, dates.min(), dates.max())
    print(f"effective range: {start.date()} ~ {end.date()}")
    env = PortfolioEnv(features_df, market_state_df, cfg, start, end)
    device = select_device(cfg["train_device"])
    train_ppo(env, total_timesteps=timesteps, seed=seed, device=device, save_path=checkpoint_path)
    return checkpoint_path


def infer_latest(features_df: pd.DataFrame, cfg: dict, model_path: str,
                 market_state_df: pd.DataFrame) -> pd.DataFrame:
    dates = pd.to_datetime(features_df["trade_date"]).unique()
    dates = sorted(dates)
    ctx_start = dates[max(0, len(dates) - 60)]
    end = dates[-1]
    ctx_start, end = effective_range(features_df, market_state_df, ctx_start, end)
    env = PortfolioEnv(features_df, market_state_df, cfg, ctx_start, end)
    model = load_ppo(model_path, env)

    obs, _ = env.reset(seed=0)
    last_info = None
    last_target_w = None
    last_symbols = env.symbols
    done = False
    while not done:
        act, _ = model.predict(obs, deterministic=True)
        obs, _, term, trunc, info = env.step(act)
        last_info = info
        last_target_w = env.prev_stock_w.copy()
        done = term or trunc

    if last_info is None:
        raise RuntimeError("env produced no step; features_df too short")

    factor_w = np.asarray(last_info["factor_w"], dtype=float)
    scores = env._F_by_date[env.dates[env.t - 1]] @ factor_w

    # Normalize weights to respect caps (HARD CONSTRAINT - no exceptions)
    weights = last_target_w.copy()
    long_sum = float(np.clip(weights, 0, None).sum())
    short_sum = float(np.clip(-weights, 0, None).sum())
    long_cap = cfg["long_notional"]
    short_cap = cfg["short_notional_cap"]

    # Scale down if exceeds (always enforce, never allow violation)
    if long_sum > long_cap * 1.0001:  # 0.01% tolerance only
        scale_factor = long_cap / long_sum
        weights = weights * (weights > 0) * scale_factor + weights * (weights <= 0)
    if short_sum > short_cap * 1.0001:  # 0.01% tolerance only
        scale_factor = short_cap / short_sum
        weights = weights * (weights < 0) * scale_factor + weights * (weights >= 0)

    # Verify constraint
    final_long = float(np.clip(weights, 0, None).sum())
    final_short = float(np.clip(-weights, 0, None).sum())
    if final_long > long_cap * 1.001 or final_short > short_cap * 1.001:
        raise RuntimeError(f"Weight constraint FAILED: long={final_long}, short={final_short}")

    now = datetime.now(timezone.utc).isoformat()
    rows = []
    fw_json = json.dumps(dict(zip(FACTOR_NAMES, factor_w.tolist())))
    trade_date = pd.Timestamp(end).normalize()
    for i, s in enumerate(last_symbols):
        w = float(weights[i])
        if abs(w) < 1e-9:
            continue
        rows.append({
            "trade_date": trade_date, "symbol": s, "weight": w,
            "side": "long" if w > 0 else "short",
            "factor_weights": fw_json,
            "composite_score": float(scores[i]),
            "strategy_id": STRATEGY_ID, "data_version": DATA_VERSION,
            "update_time": now,
        })
    cash = 1.0 - float(np.clip(weights, 0, None).sum()) - float(np.clip(-weights, 0, None).sum())
    if abs(cash) > 1e-9:
        rows.append({
            "trade_date": trade_date, "symbol": "CASH", "weight": cash, "side": "cash",
            "factor_weights": fw_json, "composite_score": 0.0,
            "strategy_id": STRATEGY_ID, "data_version": DATA_VERSION, "update_time": now,
        })
    return pd.DataFrame(rows)


def save_allocations(df: pd.DataFrame, path: str) -> None:
    p = pathlib.Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    if p.exists():
        old = pd.read_parquet(p)
        # 一个 trade_date 的持仓是一个整体组合(多空各有 cap),必须整体覆盖:
        # 丢弃旧文件里与新数据同 trade_date 的所有行,避免上一批被淘汰的
        # symbol 残留、破坏组合加总约束。
        new_dates = set(df["trade_date"].unique())
        old = old[~old["trade_date"].isin(new_dates)]
        combined = pd.concat([old, df], ignore_index=True)
    else:
        combined = df
    combined.to_parquet(p, index=False)


def main() -> None:
    cfg = get_config()
    root = pathlib.Path(__file__).resolve().parent.parent
    feats_path = root / "data" / "features.parquet"
    market_state_path = root / "data" / "market_state.parquet"
    ckpt = root / "checkpoints" / "production.zip"
    out_path = root.parent / "rl-portfolio-allocator-production" / "data" / "allocations.parquet"

    p = argparse.ArgumentParser()
    grp = p.add_mutually_exclusive_group(required=True)
    grp.add_argument("--retrain", action="store_true", help="用数据起点~最新日全部数据重训生产模型")
    grp.add_argument("--infer-only", action="store_true", help="复用现有生产模型仅推理当日持仓")
    p.add_argument("--timesteps", type=int, default=200_000)
    args = p.parse_args()

    feats = pd.read_parquet(feats_path)
    market_state = pd.read_parquet(market_state_path)
    if args.retrain:
        retrain_production(
            feats, cfg, args.timesteps, seed=0,
            checkpoint_path=str(ckpt), market_state_df=market_state,
        )
        print(f"production checkpoint saved: {ckpt}")
    if not ckpt.exists():
        raise SystemExit(f"no production checkpoint at {ckpt}; run --retrain first")
    allocations = infer_latest(
        feats, cfg, model_path=str(ckpt), market_state_df=market_state
    )
    save_allocations(allocations, str(out_path))
    print(f"allocations saved: {out_path}  rows={len(allocations)}")


if __name__ == "__main__":
    main()
