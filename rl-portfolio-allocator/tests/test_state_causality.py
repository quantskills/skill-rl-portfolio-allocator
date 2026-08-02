import numpy as np
import pandas as pd
import pytest

from scripts.config import FACTOR_NAMES
from scripts.market_state import distribution_shift_report
from scripts.state import (
    BASE_MARKET_FIELDS,
    StateBuilder,
    exogenous_fields,
    state_dim,
    state_fields,
)


def _market_state(dates):
    columns = exogenous_fields(FACTOR_NAMES)
    return pd.DataFrame(1.0, index=pd.DatetimeIndex(dates), columns=columns)


def test_state_fields_are_versioned_and_unique_without_factor_ic_0():
    fields = state_fields(FACTOR_NAMES)
    assert "factor_ic_0" not in fields
    assert len(fields) == len(set(fields))
    assert fields[: len(BASE_MARKET_FIELDS)] == list(BASE_MARKET_FIELDS)


def test_state_dim_accepts_factor_name_sequence():
    assert state_dim(FACTOR_NAMES) == len(state_fields(FACTOR_NAMES))


def test_series_market_state_is_rejected():
    with pytest.raises((TypeError, ValueError)):
        StateBuilder({}, pd.Series([1.0], index=[pd.Timestamp("2020-01-01")]))


def test_nan_in_one_panel_factor_does_not_zero_other_exposures():
    date = pd.Timestamp("2020-01-02")
    panel = pd.DataFrame(
        [[np.nan, 2.0], [1.0, 3.0]],
        index=["A", "B"],
        columns=FACTOR_NAMES[:2],
    )
    builder = StateBuilder({date: panel}, _market_state([date]))

    result = builder.build(date, np.array([0.5, 0.5]), FACTOR_NAMES,
                           np.zeros(len(FACTOR_NAMES)), cash=1.0)

    assert np.isfinite(result).all()
    assert result[builder.field_index("exposure_mom_20")] == pytest.approx(0.5)
    assert result[builder.field_index("exposure_reversal_5")] == pytest.approx(2.5)


def test_missing_market_state_date_raises_key_error():
    date = pd.Timestamp("2020-01-02")
    builder = StateBuilder({}, _market_state([pd.Timestamp("2020-01-01")]))
    with pytest.raises(KeyError, match="market state"):
        builder.build(date, np.zeros(2), FACTOR_NAMES, np.zeros(len(FACTOR_NAMES)), 1.0)


def test_distribution_shift_report_flags_severe_drift():
    train = pd.DataFrame({"x": np.arange(100, dtype=float)})
    test = pd.DataFrame({"x": np.arange(100, dtype=float) + 100.0})
    report = distribution_shift_report(train, test, fields=("x",))
    assert report["fields"]["x"]["standardized_mean_shift"] > 5
    assert report["passed"] is False
