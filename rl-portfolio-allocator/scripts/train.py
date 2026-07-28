"""SB3 PPO 训练核心。被 backtest/stress_test/allocate 复用。"""
from __future__ import annotations
import os
import pathlib
from typing import Optional

import numpy as np


def select_device(pref: str) -> str:
    """Select compute device. MPS is excluded for PPO MlpPolicy due to known
    float32 numerical instability that causes NaN in policy parameters."""
    if pref == "cpu":
        return "cpu"
    if pref == "cuda":
        return "cuda"
    # MPS (Apple Silicon) causes NaN in SB3 PPO MlpPolicy — always use CPU.
    # See: https://github.com/DLR-RM/stable-baselines3/issues/1245
    if pref == "mps":
        return "cpu"
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
    except Exception:
        pass
    return "cpu"


def train_ppo(env, total_timesteps: int, seed: int = 0, device: str = "auto",
              save_path: Optional[str] = None, eval_env=None,
              eval_freq: int = 10_000, n_eval_episodes: int = 1,
              patience: Optional[int] = None):
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv
    from stable_baselines3.common.monitor import Monitor

    def _mk():
        return Monitor(env)

    vec = DummyVecEnv([_mk])
    dev = select_device(device)
    model = PPO(
        "MlpPolicy", vec, verbose=0, seed=seed, device=dev,
        n_steps=1024, batch_size=256, learning_rate=3e-4,
        gamma=0.99, gae_lambda=0.95, clip_range=0.2, ent_coef=0.01,
    )

    callback = None
    if eval_env is not None:
        from stable_baselines3.common.callbacks import (
            EvalCallback, StopTrainingOnNoModelImprovement,
        )
        eval_vec = DummyVecEnv([lambda: Monitor(eval_env)])
        stop_cb = (StopTrainingOnNoModelImprovement(
            max_no_improvement_evals=patience, min_evals=patience, verbose=0)
            if patience else None)
        callback = EvalCallback(
            eval_vec, eval_freq=eval_freq, n_eval_episodes=n_eval_episodes,
            deterministic=True, verbose=0, callback_after_eval=stop_cb,
        )

    model.learn(total_timesteps=total_timesteps, callback=callback)
    if save_path:
        pathlib.Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        model.save(save_path)
    return model


def load_ppo(path: str, env):
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv
    from stable_baselines3.common.monitor import Monitor
    vec = DummyVecEnv([lambda: Monitor(env)])
    return PPO.load(path, env=vec)


def main() -> None:
    import argparse
    from scripts.config import get_config
    from scripts.env import make_env
    cfg = get_config()
    root = pathlib.Path(__file__).resolve().parent.parent
    features_path = root / "data" / "features.parquet"
    index_path = root / "data" / "index_returns.parquet"
    ckpt = root / "checkpoints" / "smoke.zip"

    ap = argparse.ArgumentParser()
    ap.add_argument("--timesteps", type=int, default=5000,
                    help="训练步数;默认 5000 为快速自检(smoke)")
    args = ap.parse_args()

    env = make_env(str(features_path), str(index_path), cfg, cfg["start_date"], cfg["end_date"] or "2099-12-31")
    device = select_device(cfg["train_device"])
    print(f"train device: {device}")
    model = train_ppo(env, total_timesteps=args.timesteps, seed=0, device=device, save_path=str(ckpt))
    print(f"checkpoint saved: {ckpt}  (timesteps={args.timesteps})")


if __name__ == "__main__":
    main()
