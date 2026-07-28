import numpy as np
import pandas as pd
from scripts.config import get_config, FACTOR_NAMES, K
from scripts.env import PortfolioEnv


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


def test_step_info_has_ret_term():
    cfg = get_config()
    feats = _toy_features()
    idx = pd.Series(np.zeros(1), index=[feats["trade_date"].min()])
    env = PortfolioEnv(feats, idx, cfg, feats["trade_date"].min(), feats["trade_date"].max())
    env.reset(seed=0)
    _, reward, _, _, info = env.step(np.zeros(K, dtype=np.float32))
    assert "ret_term" in info["reward_parts"]
    # ret_term should equal reward_ret_weight * net_ret
    assert abs(info["reward_parts"]["ret_term"]
               - cfg["reward_ret_weight"] * info["net_ret"]) < 1e-12
