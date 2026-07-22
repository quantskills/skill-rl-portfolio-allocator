"""SB3 PPO 训练核心。被 backtest/stress_test/allocate 复用。"""
from __future__ import annotations
import os
import pathlib
from typing import Optional

import numpy as np


def select_device(pref: str) -> str:
    if pref in ("cpu", "cuda", "mps"):
        return pref
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
    except Exception:
        pass
    return "cpu"


def train_ppo(env, total_timesteps: int, seed: int = 0, device: str = "auto",
              save_path: Optional[str] = None):
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
    model.learn(total_timesteps=total_timesteps)
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
    from scripts.config import get_config
    from scripts.env import make_env
    cfg = get_config()
    root = pathlib.Path(__file__).resolve().parent.parent
    features_path = root / "data" / "features.parquet"
    index_path = root / "data" / "index_returns.parquet"
    ckpt = root / "checkpoints" / "smoke.zip"
    env = make_env(str(features_path), str(index_path), cfg, cfg["start_date"], cfg["end_date"] or "2099-12-31")
    device = select_device(cfg["train_device"])
    print(f"train device: {device}")
    model = train_ppo(env, total_timesteps=5000, seed=0, device=device, save_path=str(ckpt))
    print(f"smoke checkpoint saved: {ckpt}")


if __name__ == "__main__":
    main()
