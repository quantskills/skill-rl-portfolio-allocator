import inspect
import numpy as np
import pandas as pd
from scripts.config import get_config, FACTOR_NAMES, K
from scripts.env import PortfolioEnv
from scripts.train import train_ppo
from scripts.state import exogenous_fields


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
    market_state = pd.DataFrame(0.1, index=dates, columns=exogenous_fields(FACTOR_NAMES))
    return PortfolioEnv(feats, market_state, cfg, dates.min(), dates.max())


def test_train_ppo_accepts_eval_kwargs():
    sig = inspect.signature(train_ppo)
    for p in ("eval_env", "eval_freq", "n_eval_episodes", "patience"):
        assert p in sig.parameters, f"missing param {p}"


def test_train_ppo_backward_compatible():
    # No eval_env, minimal steps, should complete without error
    model = train_ppo(_toy_env(), total_timesteps=64, seed=0, device="cpu")
    assert model is not None
