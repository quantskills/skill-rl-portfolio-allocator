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


def test_step_info_has_ret_term():
    cfg = get_config()
    feats = _toy_features()
    market_state = _toy_market_state(feats["trade_date"].unique())
    env = PortfolioEnv(feats, market_state, cfg, feats["trade_date"].min(), feats["trade_date"].max())
    env.reset(seed=0)
    _, reward, _, _, info = env.step(np.zeros(K, dtype=np.float32))
    assert "ret_term" in info["reward_parts"]
    # ret_term should equal reward_ret_weight * net_ret
    assert abs(info["reward_parts"]["ret_term"]
               - cfg["reward_ret_weight"] * info["net_ret"]) < 1e-12


def test_duplicate_market_state_dates_are_rejected():
    cfg = get_config()
    feats = _toy_features()
    market_state = _toy_market_state(feats["trade_date"].unique())
    market_state = pd.concat([market_state, market_state.iloc[[0]]])
    with pytest.raises(ValueError, match="duplicate market state dates"):
        PortfolioEnv(feats, market_state, cfg, feats["trade_date"].min(), feats["trade_date"].max())
