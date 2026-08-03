import numpy as np
import pandas as pd
import pytest
from scripts.config import get_config, FACTOR_NAMES, K
from scripts.env import PortfolioEnv
from scripts.state import exogenous_fields


def _toy_features():
    dates = pd.date_range("2020-01-01", periods=6, freq="D")
    rows = []
    for d in dates:
        for i in range(40):
            row = {"trade_date": d, "symbol": f"S{i:03d}",
                   "ret_1d": 0.01 if i % 2 else -0.01, "is_suspended": False}
            for fn in FACTOR_NAMES:
                row[fn] = float((i % 5) - 2)
            rows.append(row)
    return pd.DataFrame(rows)


def _toy_market_state(dates):
    return pd.DataFrame(0.1, index=pd.DatetimeIndex(dates), columns=exogenous_fields(FACTOR_NAMES))


@pytest.mark.parametrize("variant", ["none", "low", "medium", "constrained"])
def test_step_uses_default_reward_without_dsr_parts(variant):
    cfg = get_config()
    cfg["reward_variant"] = variant
    feats = _toy_features()
    market_state = _toy_market_state(feats["trade_date"].unique())
    env = PortfolioEnv(feats, market_state, cfg, feats["trade_date"].min(), feats["trade_date"].max())
    env.reset(seed=0)
    _, reward, _, _, info = env.step(np.zeros(K, dtype=np.float32))
    assert "scaled_net_return" in info["reward_parts"]
    assert "dsr" not in info["reward_parts"]
    assert "dsr" not in info
    assert "dsr_metric" in info["diagnostics"]
    assert -5.0 <= reward <= 5.0
    if variant == "constrained":
        assert "dual_lambda" in info["reward_parts"]
        assert "recovery_credit" in info["reward_parts"]
        assert "downside_vol_penalty" in info["reward_parts"]


def test_reset_initializes_previous_drawdown_and_toy_rollout_is_bounded():
    cfg = get_config()
    feats = _toy_features()
    market_state = _toy_market_state(feats["trade_date"].unique())
    env = PortfolioEnv(feats, market_state, cfg, feats["trade_date"].min(), feats["trade_date"].max())
    env.reset(seed=0)
    assert env.prev_drawdown == 0.0
    rewards = []
    for _ in range(len(env.dates)):
        _, reward, terminated, _, info = env.step(np.zeros(K, dtype=np.float32))
        rewards.append(reward)
        assert "dsr" not in info["reward_parts"]
        if terminated:
            break
    assert all(-5.0 <= reward <= 5.0 for reward in rewards)


def test_duplicate_market_state_dates_are_rejected():
    cfg = get_config()
    feats = _toy_features()
    market_state = _toy_market_state(feats["trade_date"].unique())
    market_state = pd.concat([market_state, market_state.iloc[[0]]])
    with pytest.raises(ValueError, match="duplicate market state dates"):
        PortfolioEnv(feats, market_state, cfg, feats["trade_date"].min(), feats["trade_date"].max())
