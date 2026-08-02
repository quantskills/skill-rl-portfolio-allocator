import numpy as np
import pandas as pd
import pytest

from scripts.baselines import (
    equal_weight_rollout,
    fit_static_factor_weights,
    rolling_ic_weights,
    static_factor_equal_rollout,
)
from scripts.config import FACTOR_NAMES, K, get_config
from scripts.state import exogenous_fields


def _features(periods=17, symbols=4):
    dates = pd.date_range("2024-01-01", periods=periods, freq="D")
    rows = []
    for d in dates:
        for i in range(symbols):
            rows.append({
                "trade_date": d,
                "symbol": f"S{i}",
                "ret_1d": 0.01 * (i + 1),
                "is_suspended": False,
                **{factor: float(i + 1) for factor in FACTOR_NAMES},
            })
    return pd.DataFrame(rows)


def _baseline_cfg():
    cfg = get_config()
    cfg.update({"top_n": 1, "bottom_m": 1, "turnover_budget": 2.0})
    return cfg


def test_fit_static_factor_weights_uses_training_columns_and_is_l1_normalized():
    values = np.arange(60 * K, dtype=float).reshape(60, K)
    factor_returns = pd.DataFrame(values, columns=FACTOR_NAMES)

    weights = fit_static_factor_weights(factor_returns, FACTOR_NAMES)
    factor_returns.iloc[:, 0] = -999.0

    assert weights.shape == (K,)
    assert np.sum(np.abs(weights)) == pytest.approx(1.0)
    assert np.all(np.isfinite(weights))


def test_fit_static_factor_weights_requires_60_complete_observations():
    factor_returns = pd.DataFrame(np.ones((59, K)), columns=FACTOR_NAMES)

    with pytest.raises(ValueError, match="60"):
        fit_static_factor_weights(factor_returns, FACTOR_NAMES)


def test_fit_static_factor_weights_rejects_zero_solution():
    factor_returns = pd.DataFrame(np.zeros((60, K)), columns=FACTOR_NAMES)

    with pytest.raises(ValueError, match="zero"):
        fit_static_factor_weights(factor_returns, FACTOR_NAMES)


def test_rolling_ic_weights_reads_current_state_and_is_l1_normalized():
    row = pd.Series({f"{factor}_ic_mean_20": i - 2 for i, factor in enumerate(FACTOR_NAMES)})

    weights = rolling_ic_weights(row, FACTOR_NAMES)

    assert np.sum(np.abs(weights)) == pytest.approx(1.0)
    assert np.allclose(weights, np.asarray([-2, -1, 0, 1, 2, 3]) / 9.0)


def test_rolling_ic_weights_returns_zero_for_zero_signal():
    row = pd.Series({f"{factor}_ic_mean_20": 0.0 for factor in FACTOR_NAMES})

    assert np.array_equal(rolling_ic_weights(row, FACTOR_NAMES), np.zeros(K))


def test_factor_rollout_settles_final_decision_period_for_17_dates():
    features = _features()
    result = static_factor_equal_rollout(
        features, _baseline_cfg(), features.trade_date.min(), features.trade_date.max()
    )

    assert len(result) == 16


def test_equal_factor_baseline_dimension_matches_config():
    names = list(FACTOR_NAMES[:3])
    cfg = _baseline_cfg()
    cfg["factor_names"] = names
    cfg["k"] = len(names)
    features = _features().drop(columns=[name for name in FACTOR_NAMES if name not in names])

    result = static_factor_equal_rollout(
        features, cfg, features.trade_date.min(), features.trade_date.max()
    )

    assert result.ndim == 1
    assert len(result) == 16


def test_static_weight_helpers_use_explicit_factor_names():
    names = list(FACTOR_NAMES[:3])
    returns = pd.DataFrame(np.arange(180, dtype=float).reshape(60, 3), columns=names)
    row = pd.Series({f"{name}_ic_mean_20": i + 1.0 for i, name in enumerate(names)})

    fitted = fit_static_factor_weights(returns, names)
    rolling = rolling_ic_weights(row, names)

    assert fitted.shape == (3,)
    assert rolling.shape == (3,)


def test_equal_weight_rollout_uses_weekly_settlement_length():
    features = _features()
    result = equal_weight_rollout(
        features, _baseline_cfg(), features.trade_date.min(), features.trade_date.max()
    )

    assert len(result) == 16


def test_backtest_calls_optimized_and_rolling_baselines(monkeypatch):
    import scripts.backtest as backtest

    calls = []

    class DummyEnv:
        dates = list(pd.date_range("2024-01-01", periods=3))

        def reset(self, seed=0):
            return np.zeros(1), {}

        def step(self, action):
            return np.zeros(1), 0.0, True, False, {
                "daily_net_rets": [0.0], "settlement_dates": [self.dates[1]]
            }

    class DummyModel:
        def predict(self, obs, deterministic=True):
            return np.zeros(K), None

    def record(name, value):
        def rollout(*args, **kwargs):
            calls.append((name, args, kwargs))
            return np.zeros(1)
        return rollout

    monkeypatch.setattr(backtest, "PortfolioEnv", lambda *args, **kwargs: DummyEnv())
    monkeypatch.setattr(backtest, "select_device", lambda device: "cpu")
    monkeypatch.setattr(backtest, "train_ppo", lambda *args, **kwargs: DummyModel())
    monkeypatch.setattr(backtest, "equal_weight_rollout", record("equal", None))
    monkeypatch.setattr(backtest, "long_only_topn_rollout", record("long_only", None))
    monkeypatch.setattr(backtest, "static_factor_equal_rollout", record("equal_factor", None))
    monkeypatch.setattr(backtest, "static_factor_optimized_rollout", record("optimized", None))
    monkeypatch.setattr(backtest, "rolling_ic_rollout", record("rolling_ic", None))
    monkeypatch.setattr(backtest, "metrics_pack", lambda rets, name: {"name": name})
    monkeypatch.setattr(backtest, "summarize_rollout", lambda infos: {})
    monkeypatch.setattr(backtest, "check_degeneracy", lambda diag, cfg: [])

    features = _features(periods=3, symbols=10)
    state = pd.DataFrame(0.1, index=pd.date_range("2024-01-01", periods=3),
                         columns=exogenous_fields(FACTOR_NAMES))
    result = backtest.run_backtest(
        features, state, _baseline_cfg(),
        "2024-01-01", "2024-01-01", "2024-01-02", "2024-01-03",
        timesteps=1,
    )

    assert {call[0] for call in calls} == {
        "equal", "long_only", "equal_factor", "optimized", "rolling_ic"
    }
    assert result["research_ok"] is False
