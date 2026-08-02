import numpy as np
import pandas as pd
import pytest

from scripts.config import FACTOR_NAMES, get_config
from scripts.env import PortfolioEnv
from scripts.state import STATE_SCHEMA_VERSION, exogenous_fields, state_dim


def dynamic_inputs(k=3):
    names = list(FACTOR_NAMES[:k])
    dates = pd.date_range("2024-01-01", periods=15, freq="D")
    rows = []
    for date in dates:
        for i in range(4):
            rows.append({
                "trade_date": date,
                "symbol": f"S{i}",
                "ret_1d": 0.01 * (i + 1),
                "is_suspended": False,
                **{name: float(i + 1) for name in names},
            })
    features = pd.DataFrame(rows)
    market_state = pd.DataFrame(
        0.1, index=dates, columns=exogenous_fields(names)
    )
    cfg = get_config()
    cfg.update({
        "factor_names": names,
        "k": len(names),
        "top_n": 1,
        "bottom_m": 0,
        "short_notional_cap": 0.0,
        "turnover_budget": 2.0,
        "ema_alpha": 1.0,
    })
    return features, market_state, cfg, names


def test_environment_uses_explicit_factor_names():
    features, market_state, cfg, names = dynamic_inputs(k=3)
    env = PortfolioEnv(
        features,
        market_state,
        cfg,
        features.trade_date.min(),
        features.trade_date.max(),
    )

    obs, _ = env.reset(seed=0)

    assert env.factor_names == tuple(names)
    assert env.action_space.shape == (3,)
    assert obs.shape == (state_dim(names),)
    _, _, _, _, info = env.step(np.zeros(3, dtype=np.float32))
    assert len(info["factor_w"]) == 3


def test_environment_rejects_factor_dimension_mismatch():
    features, market_state, cfg, _ = dynamic_inputs(k=3)
    cfg["k"] = 2

    with pytest.raises(ValueError, match="factor_names and k"):
        PortfolioEnv(
            features,
            market_state,
            cfg,
            features.trade_date.min(),
            features.trade_date.max(),
        )


def test_environment_rejects_empty_factor_names():
    features, market_state, cfg, _ = dynamic_inputs(k=3)
    cfg.update({"factor_names": [], "k": 0})

    with pytest.raises(ValueError, match="non-empty"):
        PortfolioEnv(
            features,
            market_state,
            cfg,
            features.trade_date.min(),
            features.trade_date.max(),
        )


def test_get_config_returns_independent_factor_name_list():
    cfg = get_config()
    cfg["factor_names"].append("temporary_factor")

    assert "temporary_factor" not in get_config()["factor_names"]


def test_state_schema_and_dimension_require_explicit_factor_names():
    assert STATE_SCHEMA_VERSION == "state-v2-dynamic-factors"
    with pytest.raises(TypeError):
        state_dim(3)
