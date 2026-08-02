import numpy as np
import pandas as pd
import pytest

from scripts.config import FACTOR_NAMES, K, get_config
from scripts.env import PortfolioEnv, extract_settle_holding_period
from scripts.state import exogenous_fields


def _toy_data(periods=15):
    dates = pd.date_range("2024-01-01", periods=periods, freq="D")
    rows = []
    for d in dates:
        for i in range(4):
            rows.append({
                "trade_date": d, "symbol": f"S{i}", "ret_1d": 0.01 * (i + 1),
                "is_suspended": False, **{name: float(i + 1) for name in FACTOR_NAMES},
            })
    features = pd.DataFrame(rows)
    state = pd.DataFrame(0.1, index=dates, columns=exogenous_fields(FACTOR_NAMES))
    return features, state


def test_env_rejects_data_without_a_settleable_period():
    features, state = _toy_data(periods=1)
    with pytest.raises(ValueError, match="at least two all_dates"):
        PortfolioEnv(features, state, get_config(), features.trade_date.min(), features.trade_date.max())


def test_weekly_env_exposes_decision_dates_and_compounds_each_settlement_day():
    features, state = _toy_data()
    cfg = get_config()
    cfg.update({"top_n": 1, "bottom_m": 1, "short_notional_cap": 0.0,
                "turnover_budget": 2.0, "ema_alpha": 1.0})
    env = PortfolioEnv(features, state, cfg, features.trade_date.min(), features.trade_date.max())

    assert len(env.all_dates) == 15
    assert env.decision_dates == list(pd.to_datetime(["2024-01-01", "2024-01-08", "2024-01-15"]))
    env.reset(seed=0)
    _, reward, terminated, _, info = env.step(np.zeros(K, dtype=np.float32))

    assert not terminated
    assert len(info["settlement_dates"]) == 7
    assert info["settlement_dates"][0] == pd.Timestamp("2024-01-02")
    assert info["settlement_dates"][-1] == pd.Timestamp("2024-01-08")
    assert len(info["daily_net_rets"]) == 7
    assert info["net_ret"] == np.prod(1.0 + np.asarray(info["daily_net_rets"])) - 1.0
    assert reward != info["net_ret"]


def test_final_week_settles_through_data_end_and_terminates_without_empty_transition():
    features, state = _toy_data()
    cfg = get_config()
    cfg.update({"top_n": 1, "bottom_m": 0, "short_notional_cap": 0.0,
                "turnover_budget": 2.0, "ema_alpha": 1.0})
    env = PortfolioEnv(features, state, cfg, features.trade_date.min(), features.trade_date.max())
    env.reset(seed=0)
    env.step(np.zeros(K, dtype=np.float32))
    _, _, terminated, _, info = env.step(np.zeros(K, dtype=np.float32))

    assert terminated
    assert info["settlement_dates"] == list(pd.date_range("2024-01-09", "2024-01-15"))
    assert len(info["daily_net_rets"]) == 7


def test_settlement_helper_charges_trade_cost_once_and_borrow_daily():
    cfg = get_config()
    cfg.update({"commission_bps": 100.0, "stamp_tax_bps": 0.0, "impact_bps": 0.0,
                "borrow_rate_annual": 0.252, "trading_days_per_year": 252})
    dates = list(pd.date_range("2024-01-02", periods=3, freq="D"))
    result = extract_settle_holding_period(
        np.zeros(2), np.array([0.0, -1.0]), dates,
        {date: np.zeros(2) for date in dates}, cfg,
    )

    assert result["costs"]["commission"] == 0.01
    assert result["costs"]["borrow"] == 3 * 0.001
    assert result["costs"]["total"] == result["costs"]["commission"] + result["costs"]["borrow"]
    assert len(result["daily_net_rets"]) == 3


def test_rollout_requires_weekly_daily_settlement_fields():
    from scripts.backtest import run_ppo_rollout

    class Model:
        def predict(self, obs, deterministic):
            return np.zeros(K), None

    class Env:
        decision_dates = [pd.Timestamp("2024-01-01")]
        t = 0

        def reset(self, seed):
            return np.zeros(1), {}

        def step(self, action):
            return np.zeros(1), 0.0, True, False, {"net_ret": 0.1}

    with pytest.raises(KeyError):
        run_ppo_rollout(Model(), Env())
