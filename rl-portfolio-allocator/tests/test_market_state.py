import numpy as np
import pandas as pd
import pytest

import scripts.market_state as market_state
from scripts.config import FACTOR_NAMES, get_config
from scripts.market_state import (
    MARKET_STATE_SCHEMA_VERSION,
    build_market_state,
    compute_daily_factor_ic,
)


@pytest.fixture
def synthetic_inputs():
    dates = pd.bdate_range("2020-01-01", periods=90)
    symbols = [f"S{i:02d}" for i in range(20)]
    rows = []
    for date_idx, date in enumerate(dates):
        for symbol_idx, symbol in enumerate(symbols):
            row = {
                "trade_date": date,
                "symbol": symbol,
                "ret_1d": float(date_idx - 1 + symbol_idx / 1000),
                "is_suspended": False,
            }
            for factor in FACTOR_NAMES:
                row[factor] = float(symbol_idx + date_idx / 1000)
            row["mom_20"] = float(symbol_idx)
            rows.append(row)
    features = pd.DataFrame(rows)
    index_returns = pd.DataFrame({"trade_date": dates, "ret": np.linspace(-0.01, 0.01, len(dates))})
    return features, index_returns


def test_daily_ic_uses_previous_factor_and_current_return(synthetic_inputs):
    features, _ = synthetic_inputs
    ic = compute_daily_factor_ic(features, FACTOR_NAMES)
    assert ic["mom_20_ic"].dropna().iloc[-1] > 0.99


def test_market_state_contains_only_requested_factor_fields(synthetic_inputs):
    features, index_returns = synthetic_inputs
    names = list(FACTOR_NAMES[:3])
    cfg = get_config()
    cfg["factor_names"] = names
    cfg["k"] = len(names)

    state = build_market_state(features, index_returns, cfg, factor_names=names)

    for name in names:
        assert f"{name}_ic_mean_20" in state
        assert f"{name}_factor_ret_20" in state
    omitted = next(name for name in FACTOR_NAMES if name not in names)
    assert not any(column.startswith(f"{omitted}_") for column in state.columns)


def test_future_mutation_does_not_change_past_state(synthetic_inputs):
    features, index_returns = synthetic_inputs
    cutoff = features["trade_date"].sort_values().unique()[60]
    before = build_market_state(features, index_returns, get_config())
    mutated = features.copy()
    mutated.loc[mutated["trade_date"] > cutoff, "ret_1d"] += 1000.0
    after = build_market_state(mutated, index_returns, get_config())
    cols = [c for c in before.columns if c != "trade_date"]
    left = before.loc[before["trade_date"] <= cutoff, cols].reset_index(drop=True)
    right = after.loc[after["trade_date"] <= cutoff, cols].reset_index(drop=True)
    pd.testing.assert_frame_equal(left, right, check_exact=True)


def test_schema_version_is_explicit():
    assert MARKET_STATE_SCHEMA_VERSION == "market-state-v2"


def test_market_state_schema_names_and_warmup(synthetic_inputs):
    features, index_returns = synthetic_inputs
    state = build_market_state(features, index_returns, get_config())

    expected_market = {
        "market_ret_20",
        "market_ret_60",
        "market_vol_20",
        "market_vol_60",
        "market_drawdown_60",
        "market_vol_regime",
    }
    assert expected_market <= set(state.columns)
    for factor in FACTOR_NAMES:
        assert {
            f"{factor}_factor_ret_20",
            f"{factor}_factor_ret_60",
            f"{factor}_factor_vol_20",
            f"{factor}_factor_vol_60",
        } <= set(state.columns)
        assert f"{factor}_return_mean_20" not in state.columns
        assert f"{factor}_return_vol_20" not in state.columns

    assert state["market_ret_20"].iloc[:19].isna().all()
    assert state["market_ret_20"].iloc[19:].notna().all()
    assert state["market_vol_20"].iloc[:19].isna().all()
    assert state["market_vol_60"].iloc[:59].isna().all()
    assert state["market_drawdown_60"].iloc[:59].isna().all()
    assert state["market_vol_regime"].iloc[:19].isna().all()
    assert state["mom_20_factor_ret_20"].iloc[:19].isna().all()
    assert state["mom_20_factor_ret_60"].iloc[:59].isna().all()


def test_market_state_is_limited_to_index_coverage(synthetic_inputs):
    features, index_returns = synthetic_inputs
    index_returns = index_returns.iloc[20:].reset_index(drop=True)

    state = build_market_state(features, index_returns, get_config())

    assert state["trade_date"].tolist() == index_returns["trade_date"].tolist()


def test_factor_corr_uses_daily_factor_returns_not_feature_cross_section(synthetic_inputs, monkeypatch):
    features, index_returns = synthetic_inputs
    rng = np.random.default_rng(7)
    features = features.copy()
    for factor in FACTOR_NAMES:
        features[factor] = rng.normal(size=len(features))

    dates = index_returns["trade_date"]
    common_returns = np.sin(np.arange(len(dates)) / 5.0)
    daily_returns = {"trade_date": dates}
    for factor in FACTOR_NAMES:
        daily_returns[f"{factor}_factor_ret"] = common_returns
    monkeypatch.setattr(
        market_state,
        "compute_daily_factor_returns",
        lambda features, cfg, factor_names=None: pd.DataFrame(daily_returns),
    )

    state = market_state.build_market_state(features, index_returns, get_config())
    assert state["factor_corr_20"].dropna().iloc[-1] > 0.99
    assert state["factor_corr_60"].dropna().iloc[-1] > 0.99


def test_single_factor_market_state_has_usable_factor_corr(synthetic_inputs):
    features, index_returns = synthetic_inputs
    cfg = get_config()
    names = [FACTOR_NAMES[0]]
    cfg["factor_names"] = names
    cfg["k"] = 1

    state = build_market_state(features, index_returns, cfg, factor_names=names)

    first_valid_20 = state["factor_corr_20"].first_valid_index()
    first_valid_60 = state["factor_corr_60"].first_valid_index()

    assert first_valid_20 == 20
    assert state["factor_corr_20"].iloc[:first_valid_20].isna().all()
    assert state["factor_corr_20"].iloc[first_valid_20:].eq(1.0).all()
    assert first_valid_60 == 60
    assert state["factor_corr_60"].iloc[:first_valid_60].isna().all()
    assert state["factor_corr_60"].iloc[first_valid_60:].eq(1.0).all()


def test_market_vol_percentile_is_causal_bounded_and_warms_up(synthetic_inputs):
    features, index_returns = synthetic_inputs
    state = build_market_state(features, index_returns, get_config())
    percentile = state["market_vol_percentile_20"]
    # vol_20 第 20 个观测起有效(index 19),expanding min_periods=60 → 首个有效值在 index 78
    assert percentile.iloc[:78].isna().all()
    assert percentile.iloc[78:].notna().all()
    valid = percentile.dropna()
    assert ((valid >= 0.0) & (valid <= 1.0)).all()


def test_market_drawdown_ath_matches_manual_cummax(synthetic_inputs):
    features, index_returns = synthetic_inputs
    state = build_market_state(features, index_returns, get_config())
    ret = index_returns.set_index("trade_date")["ret"].reindex(state["trade_date"])
    nav = (1.0 + ret.fillna(0.0)).cumprod()
    expected = (nav / nav.cummax() - 1.0).reset_index(drop=True)
    pd.testing.assert_series_equal(
        state["market_drawdown_ath"], expected, check_names=False,
    )
    assert (state["market_drawdown_ath"] <= 1e-12).all()


def test_new_fields_are_causal(synthetic_inputs):
    features, index_returns = synthetic_inputs
    cutoff = pd.Timestamp(features["trade_date"].unique()[60])
    left = build_market_state(features, index_returns, get_config())
    mutated_features = features.copy()
    mutated_index = index_returns.copy()
    mutated_features.loc[mutated_features["trade_date"] > cutoff, "ret_1d"] += 1000.0
    mutated_index.loc[mutated_index["trade_date"] > cutoff, "ret"] += 1000.0
    right = build_market_state(mutated_features, mutated_index, get_config())
    cols = ["market_vol_percentile_20", "market_drawdown_ath"]
    pd.testing.assert_frame_equal(
        left.loc[left["trade_date"] <= cutoff, cols].reset_index(drop=True),
        right.loc[right["trade_date"] <= cutoff, cols].reset_index(drop=True),
        check_exact=True,
    )
