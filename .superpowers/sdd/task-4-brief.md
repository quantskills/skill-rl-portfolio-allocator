# Task 4: train_ppo - Add Optional EvalCallback Early-Stopping (Backward-Compatible)

**Objective:** Add optional early-stopping via EvalCallback to allow training termination on eval-set performance plateau, while preserving backward compatibility (no `eval_env` = no callback, behaves identically to current).

## Scope

- **Files to modify:** `rl-portfolio-allocator/scripts/train.py` (function `train_ppo`, lines ~30-50)
- **Test location:** Create `rl-portfolio-allocator/tests/test_train.py` (new file)

## Exact Changes Required

### New train_ppo Signature

Replace the current `train_ppo` signature with:

```python
def train_ppo(env, total_timesteps: int, seed: int = 0, device: str = "auto",
              save_path: Optional[str] = None, eval_env=None,
              eval_freq: int = 10_000, n_eval_episodes: int = 1,
              patience: Optional[int] = None):
```

**New parameters:**
- `eval_env=None` — optional eval environment; if None, no early-stopping
- `eval_freq: int = 10_000` — eval every N timesteps
- `n_eval_episodes: int = 1` — evaluate for N episodes
- `patience: Optional[int] = None` — stop after N evals with no improvement; if None, no patience-based stopping

### New Function Body

```python
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
```

**Changes from current:**
1. Add 4 new optional parameters (eval_env=None, eval_freq, n_eval_episodes, patience)
2. Create EvalCallback and StopTrainingOnNoModelImprovement only if eval_env is not None
3. Pass callback to model.learn()
4. If eval_env is None, callback remains None and model.learn() behaves exactly as before

## Test Coverage

Write `tests/test_train.py` with:

```python
import inspect
import numpy as np
import pandas as pd
from scripts.config import get_config, FACTOR_NAMES, K
from scripts.env import PortfolioEnv
from scripts.train import train_ppo


def _toy_env():
    cfg = get_config()
    dates = pd.date_range("2020-01-01", periods=8, freq="D")
    rows = []
    for d in dates:
        for i in range(40):
            row = {"trade_date": d, "symbol": f"S{i:03d}",
                   "ret_1d": 0.01 if i % 2 else -0.01, "is_suspended": False}
            for fn in FACTOR_NAMES:
                row[fn] = float((i % 5) - 2)
            rows.append(row)
    feats = pd.DataFrame(rows)
    idx = pd.Series(np.zeros(1), index=[feats["trade_date"].min()])
    return PortfolioEnv(feats, idx, cfg, feats["trade_date"].min(), feats["trade_date"].max())


def test_train_ppo_accepts_eval_kwargs():
    sig = inspect.signature(train_ppo)
    for p in ("eval_env", "eval_freq", "n_eval_episodes", "patience"):
        assert p in sig.parameters, f"missing param {p}"


def test_train_ppo_backward_compatible():
    # No eval_env, minimal steps, should complete without error
    model = train_ppo(_toy_env(), total_timesteps=64, seed=0, device="cpu")
    assert model is not None
```

Both tests must PASS after implementation.

## Interface Contract

**Consumes:**
- PPO model creation (unchanged)
- stable-baselines3 callbacks (EvalCallback, StopTrainingOnNoModelImprovement)

**Produces:**
- New signature with optional eval_env, eval_freq, n_eval_episodes, patience
- When eval_env=None: callback=None, model.learn() behaves identically to current
- When eval_env provided: EvalCallback monitors eval performance, optionally stops on plateau

**Constraints:**
- Backward compatibility: calling train_ppo(env, 5000) must work exactly as before
- Hard constraint on patience: if patience provided but eval_env is None, patience is ignored (no error)
- No changes to PPO hyperparameters (n_steps, batch_size, learning_rate, etc.)
- callback=None path must be identical to current implementation

