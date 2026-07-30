import numpy as np
import pandas as pd
import pytest

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
    ic = compute_daily_factor_ic(features)
    assert ic["mom_20_ic"].dropna().iloc[-1] > 0.99


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
    assert MARKET_STATE_SCHEMA_VERSION == "market-state-v1"
