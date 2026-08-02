"""SB3 PPO 训练核心。被 backtest/stress_test/allocate 复用。"""
from __future__ import annotations
import os
import json
import pathlib
from typing import Optional

import numpy as np
import pandas as pd


def artifact_paths(root, fold: int, seed: int, candidate: str,
                   schema_version: str) -> dict[str, pathlib.Path]:
    """Return the four stable paths for one fold/seed/candidate run."""
    root = pathlib.Path(root)
    stem = f"fold{fold}_seed{seed}"
    return {
        "model": root / f"{stem}_{candidate}.zip",
        "scaler": root / f"{stem}_scaler.json",
        "log": root / f"{stem}_{candidate}_training.jsonl",
        "metadata": root / f"{stem}_{candidate}_metadata.json",
    }


def _json_number(value, default=0.0):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return default
    return value if np.isfinite(value) else default


def _sharpe(rets) -> float:
    from scripts.metrics import sharpe
    return float(sharpe(np.asarray(rets, dtype=float)))


def _eval_daily_net_rets(env, model) -> list[float]:
    obs, _ = env.reset()
    rets = []
    while True:
        action, _ = model.predict(obs, deterministic=True)
        obs, _, terminated, truncated, info = env.step(action)
        rets.extend(float(x) for x in info.get("daily_net_rets", ()))
        if terminated or truncated:
            return rets


from stable_baselines3.common.callbacks import BaseCallback


class TrainingMetricsCallback(BaseCallback):
    def __init__(self, log_path, verbose=0):
        super().__init__(verbose)
        self.log_path = pathlib.Path(log_path)
        self._actions = []

    def _on_training_start(self):
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.log_path.write_text("", encoding="utf-8")

    def _on_step(self) -> bool:
        actions = self.locals.get("actions")
        if actions is not None:
            self._actions.append(np.asarray(actions, dtype=float).reshape(-1))
        return True

    def _on_rollout_end(self):
        values = self.logger.name_to_value
        actions = np.concatenate(self._actions) if self._actions else np.array([])
        ep_rewards = [x["r"] for x in self.model.ep_info_buffer]
        record = {
            "timesteps": int(self.num_timesteps),
            "rollout_ep_rew_mean": _json_number(np.mean(ep_rewards)) if ep_rewards else 0.0,
            "policy_gradient_loss": _json_number(values.get("train/policy_gradient_loss")),
            "value_loss": _json_number(values.get("train/value_loss")),
            "entropy_loss": _json_number(values.get("train/entropy_loss")),
            "approx_kl": _json_number(values.get("train/approx_kl")),
            "clip_fraction": _json_number(values.get("train/clip_fraction")),
            "explained_variance": _json_number(values.get("train/explained_variance")),
            "action_mean": _json_number(np.mean(actions)) if actions.size else 0.0,
            "action_std": _json_number(np.std(actions)) if actions.size else 0.0,
            "action_saturation": _json_number(np.mean(np.abs(actions) >= 0.999)) if actions.size else 0.0,
        }
        with self.log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
        self._actions.clear()

    def _on_training_end(self):
        if not self.log_path.exists() or not self.log_path.read_text(encoding="utf-8").strip():
            self._on_rollout_end()


class ValidationSharpeCallback(BaseCallback):
    def __init__(self, eval_env, eval_freq=10000, patience=5,
                 best_model_path=None, log_path=None, verbose=0):
        super().__init__(verbose)
        self.eval_env = eval_env
        self.eval_freq = int(eval_freq)
        self.patience = int(patience)
        self.best_model_path = pathlib.Path(best_model_path) if best_model_path else None
        self.log_path = pathlib.Path(log_path) if log_path else None
        self.best_eval_metric = float("-inf")
        self.no_improvement_evals = 0
        self.deterministic = True
        self.eval_history = []

    def _evaluate(self):
        rets = _eval_daily_net_rets(self.eval_env, self.model)
        metric = _sharpe(rets)
        self.eval_history.append(metric)
        if metric > self.best_eval_metric:
            self.best_eval_metric = metric
            self.no_improvement_evals = 0
            if self.best_model_path is not None:
                self.best_model_path.parent.mkdir(parents=True, exist_ok=True)
                self.model.save(str(self.best_model_path.with_suffix("")))
        else:
            self.no_improvement_evals += 1
            if self.no_improvement_evals >= self.patience:
                self.model.stop_training = True
        if self.log_path is not None:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            record = {
                "timesteps": int(self.num_timesteps),
                "validation_net_sharpe": metric,
                "best_score": self.best_eval_metric,
            }
            with self.log_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, sort_keys=True) + "\n")

    def _on_step(self) -> bool:
        if self.num_timesteps and self.num_timesteps % self.eval_freq == 0:
            self._evaluate()
        return True

    def _on_training_end(self):
        if not self.eval_history:
            self._evaluate()


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
              patience: Optional[int] = None, callback=None,
              training_log_path=None):
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

    callbacks = []
    if training_log_path is not None:
        callbacks.append(TrainingMetricsCallback(training_log_path))
    if callback is not None:
        callbacks.append(callback)
    if eval_env is not None and callback is None:
        callbacks.append(ValidationSharpeCallback(
            eval_env, eval_freq=eval_freq, patience=patience or 5,
            best_model_path=(pathlib.Path(save_path).with_name(
                pathlib.Path(save_path).stem + "_best.zip") if save_path else None)))

    callback_arg = callbacks if len(callbacks) > 1 else (callbacks[0] if callbacks else None)
    model.learn(total_timesteps=total_timesteps, callback=callback_arg)
    validation_callback = next((x for x in callbacks if isinstance(x, ValidationSharpeCallback)), None)
    if validation_callback is not None and validation_callback.best_model_path is not None:
        best_path = validation_callback.best_model_path
        if best_path.exists():
            model = PPO.load(str(best_path), env=vec)
    if save_path:
        pathlib.Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        model.save(save_path)
    return model


def train_candidates(*, root, fold, seed, candidates, schema_version,
                      raw_train_env, eval_env, total_timesteps, device="auto",
                      train_range=None, val_range=None, reward_variant=None,
                      buffer_variant=None):
    """Fit one scaler, then train each candidate against that frozen scaler."""
    from scripts.config import FACTOR_NAMES
    from scripts.observation import collect_training_observations, ObservationScaler
    from scripts.state import STATE_SCHEMA_VERSION, state_fields

    if schema_version != STATE_SCHEMA_VERSION:
        raise ValueError("schema_version must match the state schema")
    observations = collect_training_observations(raw_train_env, seed=seed)
    fields = tuple(state_fields(FACTOR_NAMES))
    if observations.shape[1] != len(fields):
        raise ValueError("raw training observations do not match the state schema")
    scaler = ObservationScaler.fit(observations, STATE_SCHEMA_VERSION, fields)
    results = {}
    for candidate in candidates:
        paths = artifact_paths(root, fold, seed, candidate, schema_version)
        paths["scaler"].parent.mkdir(parents=True, exist_ok=True)
        scaler.save(paths["scaler"])
        raw_train_env.observation_scaler = scaler
        eval_env.observation_scaler = scaler
        validation = ValidationSharpeCallback(
            eval_env, eval_freq=10000, patience=5,
            best_model_path=paths["model"], log_path=paths["log"])
        train_ppo(raw_train_env, total_timesteps=total_timesteps, seed=seed,
                  device=device, save_path=str(paths["model"]), callback=validation,
                  training_log_path=paths["log"])
        metadata = {
            "fold": fold, "seed": seed, "schema_version": schema_version,
            "scaler_path": str(paths["scaler"]), "train_range": train_range,
            "val_range": val_range, "reward_variant": reward_variant,
            "buffer_variant": buffer_variant, "total_timesteps": total_timesteps,
            "best_eval_metric": validation.best_eval_metric,
        }
        paths["metadata"].write_text(json.dumps(metadata, indent=2, default=str), encoding="utf-8")
        results[candidate] = paths
    return results


def load_ppo(path: str, env):
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv
    from stable_baselines3.common.monitor import Monitor
    vec = DummyVecEnv([lambda: Monitor(env)])
    return PPO.load(path, env=vec)


def main() -> None:
    import argparse
    from scripts.config import get_config
    from scripts.env import effective_range, make_env
    cfg = get_config()
    root = pathlib.Path(__file__).resolve().parent.parent
    features_path = root / "data" / "features.parquet"
    market_state_path = root / "data" / "market_state.parquet"
    ckpt = root / "checkpoints" / "smoke.zip"

    ap = argparse.ArgumentParser()
    ap.add_argument("--timesteps", type=int, default=5000,
                    help="训练步数;默认 5000 为快速自检(smoke)")
    args = ap.parse_args()

    feats = pd.read_parquet(features_path)
    market_state = pd.read_parquet(market_state_path)
    start, end = effective_range(
        feats, market_state, cfg["start_date"], cfg["end_date"] or "2099-12-31"
    )
    print(f"effective range: {start.date()} ~ {end.date()}")
    env = make_env(str(features_path), str(market_state_path), cfg, start, end)
    device = select_device(cfg["train_device"])
    print(f"train device: {device}")
    model = train_ppo(env, total_timesteps=args.timesteps, seed=0, device=device, save_path=str(ckpt))
    print(f"checkpoint saved: {ckpt}  (timesteps={args.timesteps})")


if __name__ == "__main__":
    main()
