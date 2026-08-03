import numpy as np
import pandas as pd
import pytest

from scripts.config import FACTOR_NAMES, K, get_config
from scripts.env import PortfolioEnv, effective_range, extract_settle_holding_period
from scripts.state import exogenous_fields, state_dim


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


class _StubScaler:
    def __init__(self, dim, threshold):
        self.mean = np.zeros(dim)
        self.scale = np.ones(dim)
        self.ood_shift_threshold = threshold

    def transform(self, obs):
        return np.asarray(obs, dtype=np.float32)


def test_ood_mix_blends_factor_weights_toward_uniform():
    features, state = _toy_data(periods=15)
    cfg = get_config()
    dim = state_dim(tuple(cfg["factor_names"]))
    env = PortfolioEnv(features, state, cfg,
                       features["trade_date"].min(), features["trade_date"].max(),
                       observation_scaler=_StubScaler(dim, threshold=1e-9))
    env.ood_mix_enabled = True
    env.reset(seed=0)
    k = len(cfg["factor_names"])
    # 非对称动作,使未混合的 RL 权重与等权不同
    action = np.linspace(1.0, -1.0, k, dtype=np.float32)
    _, _, _, _, info = env.step(action)
    assert info["ood_alpha"] == pytest.approx(1.0)
    assert info["factor_w"] == pytest.approx([1.0 / k] * k)
    # prev_factor_w 保存未混合的 RL 权重,不被兜底污染
    assert not np.allclose(env.prev_factor_w, np.ones(k) / k)


def test_ood_mix_disabled_by_default_leaves_weights_untouched():
    features, state = _toy_data(periods=15)
    cfg = get_config()
    dim = state_dim(tuple(cfg["factor_names"]))
    env = PortfolioEnv(features, state, cfg,
                       features["trade_date"].min(), features["trade_date"].max(),
                       observation_scaler=_StubScaler(dim, threshold=1e-9))
    env.reset(seed=0)
    k = len(cfg["factor_names"])
    _, _, _, _, info = env.step(np.ones(k, dtype=np.float32))
    assert info["ood_alpha"] == 0.0


def test_env_rejects_data_without_a_settleable_period():
    features, state = _toy_data(periods=1)
    with pytest.raises(ValueError, match="at least two all_dates"):
        PortfolioEnv(features, state, get_config(), features.trade_date.min(), features.trade_date.max())


def test_effective_range_skips_non_finite_market_state_warmup():
    features, state = _toy_data()
    state.iloc[:3, 0] = np.nan

    start, end = effective_range(
        features, state, features.trade_date.min(), features.trade_date.max()
    )

    assert start == features.trade_date.unique()[3]
    assert end == features.trade_date.max()


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


def test_episode_start_weights_oversample_crisis_segments():
    from scripts.env import CRISIS_SEGMENTS, episode_start_weights
    dates = [pd.Timestamp("2015-01-05"), pd.Timestamp("2015-07-06"),
             pd.Timestamp("2018-06-04"), pd.Timestamp("2019-01-07"),
             pd.Timestamp("2020-03-02"), pd.Timestamp("2022-06-06")]
    weights = episode_start_weights(dates)
    assert weights.tolist() == [1.0, 3.0, 3.0, 1.0, 3.0, 3.0]
    assert CRISIS_SEGMENTS == (
        ("2015-06-01", "2015-09-30"),
        ("2018-01-01", "2018-12-31"),
        ("2020-02-01", "2020-04-30"),
        ("2022-01-01", "2022-12-31"),
    )


def _randomized_env(cfg_overrides=None):
    features, state = _toy_data(periods=15)
    cfg = get_config()
    cfg.update({"episode_min_weeks": 1, "episode_max_weeks": 2,
                "crisis_oversample_weight": 3.0, "episode_randomization": True})
    cfg.update(cfg_overrides or {})
    return PortfolioEnv(features, state, cfg,
                        features["trade_date"].min(), features["trade_date"].max())


def test_randomized_reset_is_reproducible_per_seed():
    env = _randomized_env()
    env.reset(seed=7)
    first = (env.t, env.episode_end)
    env.reset(seed=7)
    assert (env.t, env.episode_end) == first
    assert 0 <= env.t < env.episode_end <= len(env.decision_dates) - 1
    assert env.episode_end - env.t <= 2  # episode_max_weeks


def test_randomized_episode_terminates_at_sampled_end():
    env = _randomized_env()
    env.reset(seed=3)
    expected_steps = env.episode_end - env.t
    k = len(env.cfg["factor_names"])
    steps, done = 0, False
    while not done:
        _, _, term, trunc, _ = env.step(np.zeros(k, dtype=np.float32))
        steps += 1
        done = term or trunc
    assert steps == expected_steps


def test_short_window_falls_back_to_full_episode():
    env = _randomized_env({"episode_min_weeks": 52, "episode_max_weeks": 156})
    env.reset(seed=0)
    assert env.t == 0
    assert env.episode_end == len(env.decision_dates) - 1


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
