from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from scripts.factor_catalog import FACTOR_CATALOG
from scripts.factor_compute import (
    EPS,
    compute_raw_factors,
    compute_factor_panel,
    downside_std,
    log_return,
    rolling_corr,
    safe_div,
    upside_std,
)


@pytest.fixture
def price_frame() -> pd.DataFrame:
    rows = np.arange(320, dtype=float)
    daily_returns = (
        0.0012 * np.sin(rows / 5.0)
        + 0.0008 * np.cos(rows / 13.0)
        + 0.00025 * ((rows.astype(int) % 9) - 4)
    )
    close = 100.0 * np.exp(np.cumsum(daily_returns))
    open_ = close * (1.0 + 0.0015 * np.sin(rows / 7.0))
    spread = 0.004 + 0.0004 * (rows.astype(int) % 6)
    high = np.maximum(open_, close) * (1.0 + spread)
    low = np.minimum(open_, close) * (1.0 - spread)
    volume = 1_000_000.0 * (
        1.0 + 0.18 * np.sin(rows / 11.0) + 0.01 * (rows.astype(int) % 17)
    )
    amount = close * volume * (1.0 + 0.03 * np.cos(rows / 19.0))

    frame = pd.DataFrame(
        {
            "trade_date": pd.date_range("2020-01-01", periods=len(rows)),
            "symbol": "TEST",
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
            "amount": amount,
        }
    )
    assert (frame[["open", "high", "low", "close", "volume", "amount"]] > 0).all().all()
    assert frame["close"].pct_change().nunique() > 100
    return frame


def test_price_family_formulas_match_manual_values(price_frame: pd.DataFrame) -> None:
    raw = compute_raw_factors(price_frame)
    row = raw.iloc[-1]
    close = price_frame["close"]

    assert row["mom_20"] == pytest.approx(np.log(close.iloc[-1] / close.iloc[-21]))
    assert row["reversal_5"] == pytest.approx(-np.log(close.iloc[-1] / close.iloc[-6]))
    assert row["ma_gap_5_20"] == pytest.approx(
        close.iloc[-5:].mean() / close.iloc[-20:].mean() - 1.0
    )
    assert row["range_position_20"] == pytest.approx(
        (close.iloc[-1] - price_frame["low"].iloc[-20:].min())
        / (price_frame["high"].iloc[-20:].max() - price_frame["low"].iloc[-20:].min())
    )


def test_range_formulas_match_manual_values(price_frame: pd.DataFrame) -> None:
    raw = compute_raw_factors(price_frame)
    row = raw.iloc[-1]
    high = price_frame["high"]
    low = price_frame["low"]
    close = price_frame["close"]
    previous_close = close.shift(1)
    true_range = pd.concat(
        [high - low, (high - previous_close).abs(), (low - previous_close).abs()],
        axis=1,
    ).max(axis=1)
    expected_parkinson = np.sqrt(
        np.log(high.iloc[-20:] / low.iloc[-20:]).pow(2).mean() / (4.0 * np.log(2.0))
    )

    assert row["atr_14"] == pytest.approx(true_range.iloc[-14:].mean())
    assert row["parkinson_vol_20"] == pytest.approx(expected_parkinson)
    assert row["range_mean_5"] == pytest.approx(
        ((high - low) / close).iloc[-5:].mean()
    )


def test_atr_propagates_intraday_nan_but_keeps_first_window_usable(
    price_frame: pd.DataFrame,
) -> None:
    with_nan = price_frame.copy()
    with_nan.loc[10, "high"] = np.nan
    raw = compute_raw_factors(with_nan)

    assert pd.notna(raw.loc[4, "atr_5"])
    assert raw.loc[10:14, "atr_5"].isna().all()
    assert pd.notna(raw.loc[15, "atr_5"])


def test_raw_computation_emits_all_catalog_names(
    price_frame: pd.DataFrame,
) -> None:
    raw = compute_raw_factors(price_frame)
    expected = [spec.name for spec in FACTOR_CATALOG]

    assert list(raw.columns[-100:]) == expected
    assert set(expected) <= set(raw.columns)


def test_raw_factor_columns_are_float32_and_warmup_stays_nan(
    price_frame: pd.DataFrame,
) -> None:
    raw = compute_raw_factors(price_frame)
    factor_names = [spec.name for spec in FACTOR_CATALOG]

    assert all(raw[name].dtype == np.dtype("float32") for name in factor_names)
    for spec in FACTOR_CATALOG:
        pre_window = raw[spec.name].iloc[: max(spec.lookback - 1, 0)]
        assert pre_window.isna().all(), (
            f"{spec.name} emitted before its catalog lookback window"
        )
    assert pd.notna(raw.loc[0, "intraday_ret_1"])
    assert raw.loc[0, [name for name in factor_names if name != "intraday_ret_1"]].isna().all()
    assert pd.isna(raw.loc[19, "zero_return_ratio_20"])
    assert pd.isna(raw.loc[19, "up_down_volume_ratio_20"])
    assert pd.isna(raw.loc[251, "mom_252"])
    assert pd.notna(raw.loc[252, "mom_252"])
    assert pd.isna(raw.loc[18, "parkinson_vol_20"])
    assert pd.notna(raw.loc[19, "parkinson_vol_20"])


def test_raw_computation_is_causal_under_future_mutation(
    price_frame: pd.DataFrame,
) -> None:
    baseline = compute_raw_factors(price_frame)
    mutated_frame = price_frame.copy()
    mutated_frame.loc[300, ["open", "high", "low", "close", "volume", "amount"]] *= 7.0
    mutated = compute_raw_factors(mutated_frame)
    factor_names = [spec.name for spec in FACTOR_CATALOG]

    np.testing.assert_allclose(
        baseline.loc[260, factor_names].to_numpy(dtype=float),
        mutated.loc[260, factor_names].to_numpy(dtype=float),
        equal_nan=True,
    )


def test_raw_computation_requires_catalog_declared_columns(
    price_frame: pd.DataFrame,
) -> None:
    missing_high = price_frame.drop(columns=["high"])

    with pytest.raises(ValueError, match="high"):
        compute_raw_factors(missing_high)


def test_raw_computation_does_not_require_columns_unused_by_first_forty(
    price_frame: pd.DataFrame,
) -> None:
    without_open = price_frame.drop(columns=["open"])
    raw = compute_raw_factors(without_open, catalog=FACTOR_CATALOG[:40])

    assert set(spec.name for spec in FACTOR_CATALOG[:40]) <= set(raw.columns)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("family", "not_canonical"),
        ("lookback", 999),
        ("required_columns", ("volume",)),
    ],
)
def test_custom_catalog_rejects_noncanonical_metadata(
    price_frame: pd.DataFrame,
    field: str,
    value: object,
) -> None:
    noncanonical = replace(FACTOR_CATALOG[0], **{field: value})

    with pytest.raises(ValueError, match="canonical metadata"):
        compute_raw_factors(price_frame, catalog=(noncanonical,))


def test_custom_catalog_subset_computes_only_selected_dependencies(
    price_frame: pd.DataFrame,
) -> None:
    mom_only = price_frame[["trade_date", "symbol", "close"]]
    raw = compute_raw_factors(mom_only, catalog=(FACTOR_CATALOG[2],))

    assert list(raw.columns) == ["trade_date", "symbol", "close", "mom_20"]
    assert raw["mom_20"].dtype == np.dtype("float32")


def test_range_only_subset_computes_without_close(
    price_frame: pd.DataFrame,
) -> None:
    range_only = price_frame[["trade_date", "symbol", "high", "low"]]
    raw = compute_raw_factors(range_only, catalog=(FACTOR_CATALOG[30],))
    high = price_frame["high"]
    low = price_frame["low"]
    expected = np.sqrt(
        np.log(high.iloc[-10:] / low.iloc[-10:]).pow(2).mean()
        / (4.0 * np.log(2.0))
    )

    assert list(raw.columns) == [
        "trade_date", "symbol", "high", "low", "parkinson_vol_10"
    ]
    assert raw["parkinson_vol_10"].dtype == np.dtype("float32")
    assert raw.iloc[-1]["parkinson_vol_10"] == pytest.approx(expected)


def test_raw_computation_isolated_per_symbol(
    price_frame: pd.DataFrame,
) -> None:
    other_symbol = price_frame.copy()
    other_symbol["symbol"] = "OTHER"
    other_symbol[["open", "high", "low", "close"]] *= 1.7
    other_symbol["volume"] *= 2.0
    other_symbol["amount"] *= 3.4
    combined = pd.concat([price_frame, other_symbol], ignore_index=True)
    factor_names = [spec.name for spec in FACTOR_CATALOG]

    combined_raw = compute_raw_factors(combined)
    expected_first = compute_raw_factors(price_frame)
    expected_second = compute_raw_factors(other_symbol)

    np.testing.assert_allclose(
        combined_raw.loc[combined_raw["symbol"] == "TEST", factor_names]
        .to_numpy(dtype=float),
        expected_first[factor_names].to_numpy(dtype=float),
        equal_nan=True,
    )
    np.testing.assert_allclose(
        combined_raw.loc[combined_raw["symbol"] == "OTHER", factor_names]
        .to_numpy(dtype=float),
        expected_second[factor_names].to_numpy(dtype=float),
        equal_nan=True,
    )
    assert combined_raw.loc[combined_raw["symbol"] == "TEST", "mom_20"].iloc[:20].isna().all()
    assert combined_raw.loc[combined_raw["symbol"] == "OTHER", "mom_20"].iloc[:20].isna().all()


def test_trade_date_sorting_is_causal_and_maps_back_to_input_order(
    price_frame: pd.DataFrame,
) -> None:
    shuffled = price_frame.sample(frac=1.0, random_state=17)
    factor_names = [spec.name for spec in FACTOR_CATALOG[:40]]

    shuffled_raw = compute_raw_factors(shuffled)
    ordered_raw = compute_raw_factors(price_frame)
    actual = shuffled_raw.sort_values("trade_date").reset_index(drop=True)
    expected = ordered_raw.sort_values("trade_date").reset_index(drop=True)

    np.testing.assert_allclose(
        actual[factor_names].to_numpy(dtype=float),
        expected[factor_names].to_numpy(dtype=float),
        equal_nan=True,
    )
    assert shuffled_raw.index.equals(shuffled.index)


def test_without_trade_date_preserves_input_row_order_behavior(
    price_frame: pd.DataFrame,
) -> None:
    without_date = price_frame.drop(columns=["trade_date"])
    expected = compute_raw_factors(price_frame)
    actual = compute_raw_factors(without_date)
    factor_names = [spec.name for spec in FACTOR_CATALOG[:40]]

    np.testing.assert_allclose(
        actual[factor_names].to_numpy(dtype=float),
        expected[factor_names].to_numpy(dtype=float),
        equal_nan=True,
    )


def test_causal_helpers_use_epsilon_and_full_windows() -> None:
    numerator = pd.Series([1.0, 1.0, 2.0])
    denominator = pd.Series([0.0, EPS, 2.0])
    assert safe_div(numerator, denominator).isna().iloc[:2].all()
    assert log_return(pd.Series([2.0, 4.0, 8.0]), 1).iloc[-1] == pytest.approx(np.log(2.0))

    left = pd.Series([1.0, 2.0, 3.0, 4.0])
    right = pd.Series([4.0, 3.0, 2.0, 1.0])
    assert pd.isna(rolling_corr(left, right, 3).iloc[1])
    assert rolling_corr(left, right, 3).iloc[-1] == pytest.approx(-1.0)

    assert downside_std(pd.Series([-1.0, -2.0, -3.0]), 3).iloc[-1] == pytest.approx(1.0)
    assert upside_std(pd.Series([1.0, 2.0, 3.0]), 3).iloc[-1] == pytest.approx(1.0)

    mixed_returns = pd.Series([-1.0, 1.0, -2.0, 2.0])
    assert downside_std(mixed_returns, 2).isna().all()
    assert upside_std(mixed_returns, 2).isna().all()


def test_invalid_prices_and_divisors_produce_nan_not_infinity(
    price_frame: pd.DataFrame,
) -> None:
    quotient = safe_div(
        pd.Series([1.0, np.inf, 1.0, 1.0]),
        pd.Series([1.0, 1.0, 0.0, np.inf]),
    )
    assert quotient.iloc[0] == pytest.approx(1.0)
    assert quotient.iloc[1:].isna().all()

    invalid_prices = pd.Series([100.0, 0.0, -1.0, np.inf, 100.0])
    returns = log_return(invalid_prices, 1)
    assert returns.iloc[1:].isna().all()
    assert not np.isinf(returns.to_numpy()).any()

    dirty = price_frame.copy()
    dirty.loc[5, "close"] = 0.0
    dirty.loc[6, "close"] = np.inf
    dirty.loc[7, "high"] = np.inf
    dirty.loc[8, "low"] = -1.0
    raw = compute_raw_factors(dirty)
    factor_names = [spec.name for spec in FACTOR_CATALOG]

    assert not np.isinf(raw[factor_names].to_numpy(dtype=float)).any()


def test_float32_output_fail_closes_large_finite_atr_values(
    price_frame: pd.DataFrame,
) -> None:
    huge_range = price_frame.copy()
    huge_range["high"] = 1e308
    huge_range["low"] = 1.0
    huge_range["close"] = 1.0
    huge_range["open"] = 1.0
    raw = compute_raw_factors(huge_range)
    factor_names = [spec.name for spec in FACTOR_CATALOG]

    assert all(raw[name].dtype == np.dtype("float32") for name in factor_names)
    assert not np.isinf(raw[factor_names].to_numpy(dtype=float)).any()
    assert pd.isna(raw.loc[4, "atr_5"])


@pytest.fixture
def multi_symbol_prices(price_frame: pd.DataFrame) -> pd.DataFrame:
    other = price_frame.copy()
    other["symbol"] = "OTHER"
    other[["open", "high", "low", "close"]] *= 1.35
    other["volume"] *= 0.7
    other["amount"] *= 0.95
    combined = pd.concat([price_frame, other], ignore_index=True)
    combined["ret_1d"] = combined.groupby("symbol", sort=False)["close"].pct_change()
    combined["is_suspended"] = False
    return combined.sample(frac=1.0, random_state=41).reset_index(drop=True)


def test_task3_raw_default_emits_all_catalog_names_and_float32(
    price_frame: pd.DataFrame,
) -> None:
    raw = compute_raw_factors(price_frame)
    factor_names = [spec.name for spec in FACTOR_CATALOG]

    assert list(raw.columns[-100:]) == factor_names
    assert all(raw[name].dtype == np.dtype("float32") for name in factor_names)


def test_task3_volume_liquidity_and_turnover_formulas(
    price_frame: pd.DataFrame,
) -> None:
    raw = compute_raw_factors(price_frame)
    last = raw.iloc[-1]
    volume = price_frame["volume"]
    amount = price_frame["amount"]
    returns = price_frame["close"].pct_change()
    turnover = volume / volume.rolling(252, min_periods=252).mean()
    amihud = (returns.abs() / (amount + EPS)).rolling(20, min_periods=20).mean()

    assert last["volume_ratio_5_20"] == pytest.approx(
        volume.iloc[-5:].mean() / volume.iloc[-20:].mean()
    )
    assert last["volume_zscore_20"] == pytest.approx(
        (volume.iloc[-1] - volume.iloc[-20:].mean()) / volume.iloc[-20:].std()
    )
    assert last["volume_volatility_20"] == pytest.approx(
        volume.pct_change().iloc[-20:].std()
    )
    assert last["turnover_20"] == pytest.approx(
        turnover.iloc[-20:].mean()
    )
    assert last["amihud_20"] == pytest.approx(amihud.iloc[-1])


def test_task3_price_volume_candle_and_tail_formulas(
    price_frame: pd.DataFrame,
) -> None:
    raw = compute_raw_factors(price_frame)
    last = raw.iloc[-1]
    close = price_frame["close"]
    high = price_frame["high"]
    low = price_frame["low"]
    open_ = price_frame["open"]
    returns = close.pct_change()
    adjacent_cov = returns.rolling(20, min_periods=20).cov(returns.shift(1)).iloc[-1]
    obv = (np.sign(returns).fillna(0.0) * price_frame["volume"]).cumsum()
    ad = ((2.0 * close - high - low) / (high - low) * price_frame["volume"])
    overnight_log = np.log(open_ / close.shift(1))
    intraday_log = np.log(close / open_)
    q = returns.iloc[-20:].quantile(0.05)

    assert last["roll_spread_20"] == pytest.approx(
        2.0 * np.sqrt(max(-adjacent_cov, 0.0))
    )
    assert last["obv_mom_20"] == pytest.approx(obv.iloc[-1] - obv.iloc[-21])
    assert last["accumulation_distribution_20"] == pytest.approx(ad.iloc[-20:].sum())
    assert last["overnight_ret_1"] == pytest.approx(
        open_.iloc[-1] / close.iloc[-2] - 1.0
    )
    assert last["overnight_ret_20"] == pytest.approx(overnight_log.iloc[-20:].sum())
    assert last["intraday_ret_20"] == pytest.approx(intraday_log.iloc[-20:].sum())
    assert last["body_ratio_20"] == pytest.approx(
        (abs(close - open_) / (high - low)).iloc[-20:].mean()
    )
    assert last["var_5pct_20"] == pytest.approx(q)
    assert last["cvar_5pct_20"] == pytest.approx(returns.iloc[-20:][returns.iloc[-20:] <= q].mean())
    assert last["max_ret_20"] == pytest.approx(returns.iloc[-20:].max())
    assert last["min_ret_20"] == pytest.approx(returns.iloc[-20:].min())


def test_flat_candle_structure_uses_eps_range_instead_of_nan(
    price_frame: pd.DataFrame,
) -> None:
    flat = price_frame.copy()
    flat.loc[300:, ["open", "high", "low", "close"]] = 100.0
    raw = compute_raw_factors(flat)

    last = raw.iloc[-1]
    assert last["upper_shadow_20"] == pytest.approx(0.0)
    assert last["lower_shadow_20"] == pytest.approx(0.0)
    assert last["body_ratio_20"] == pytest.approx(0.0)
    assert last["close_location_20"] == pytest.approx(0.0)


def test_task3_raw_full_catalog_preserves_subset_dependency_and_metadata_contract(
    price_frame: pd.DataFrame,
) -> None:
    volume_only = price_frame[["trade_date", "symbol", "volume"]]
    raw = compute_raw_factors(volume_only, catalog=(FACTOR_CATALOG[46],))

    assert list(raw.columns) == ["trade_date", "symbol", "volume", "volume_zscore_20"]
    assert raw["volume_zscore_20"].dtype == np.dtype("float32")


def test_task3_panel_is_causal_sorted_and_cross_sectionally_clipped(
    multi_symbol_prices: pd.DataFrame,
) -> None:
    cutoff = multi_symbol_prices["trade_date"].sort_values().iloc[180]
    before = compute_factor_panel(multi_symbol_prices)
    changed = multi_symbol_prices.copy()
    changed.loc[changed["trade_date"] > cutoff, ["open", "high", "low", "close", "volume", "amount"]] *= 10.0
    after = compute_factor_panel(changed)

    historical_before = before[before["trade_date"] <= cutoff].reset_index(drop=True)
    historical_after = after[after["trade_date"] <= cutoff].reset_index(drop=True)
    pd.testing.assert_frame_equal(historical_before, historical_after)
    assert after[[spec.name for spec in FACTOR_CATALOG]].dtypes.eq(np.dtype("float32")).all()
    values = after[[spec.name for spec in FACTOR_CATALOG]].to_numpy(dtype=float)
    assert not np.isinf(values).any()
    assert np.nanmax(values) <= 3.0
    assert np.nanmin(values) >= -3.0
    assert after[["trade_date", "symbol"]].equals(
        after[["trade_date", "symbol"]].sort_values(["trade_date", "symbol"]).reset_index(drop=True)
    )


def test_task3_panel_keeps_base_columns_without_cross_symbol_fill(
    multi_symbol_prices: pd.DataFrame,
) -> None:
    panel = compute_factor_panel(multi_symbol_prices)
    assert {"trade_date", "symbol", "ret_1d", "is_suspended"} <= set(panel.columns)
    first_dates = panel.groupby("symbol", sort=False)["trade_date"].min()
    for symbol, first_date in first_dates.items():
        row = panel[(panel["symbol"] == symbol) & (panel["trade_date"] == first_date)].iloc[0]
        assert pd.isna(row["mom_5"])


def test_short_history_default_catalog_distribution_is_warning_free(
    price_frame: pd.DataFrame,
) -> None:
    short = price_frame.iloc[:1].copy()
    short["close"] = np.nan

    raw = compute_raw_factors(short)
    factor_names = [spec.name for spec in FACTOR_CATALOG]

    assert list(raw.columns[-100:]) == factor_names
    assert raw[factor_names].isna().all().all()


def test_panel_winsorizes_each_date_before_standardizing(
    price_frame: pd.DataFrame,
) -> None:
    frames = []
    rows = np.arange(len(price_frame), dtype=float)
    for idx in range(5):
        frame = price_frame.copy()
        frame["symbol"] = f"S{idx}"
        scale = np.exp((idx - 2) * rows * 0.002)
        frame[["open", "high", "low", "close"]] *= scale[:, None]
        frame["amount"] = frame["close"] * frame["volume"]
        frames.append(frame)
    prices = pd.concat(frames, ignore_index=True)

    raw = compute_raw_factors(prices)
    date = price_frame["trade_date"].iloc[-1]
    raw_values = raw.loc[raw["trade_date"] == date].sort_values("symbol")["mom_5"]
    lower = raw_values.quantile(0.01)
    upper = raw_values.quantile(0.99)
    winsorized = raw_values.clip(lower=lower, upper=upper)
    expected = ((winsorized - winsorized.mean()) / winsorized.std()).clip(-3.0, 3.0)

    panel = compute_factor_panel(prices)
    actual = panel.loc[panel["trade_date"] == date].sort_values("symbol")["mom_5"]

    np.testing.assert_allclose(
        actual.to_numpy(), expected.to_numpy(), rtol=1e-6, atol=1e-6
    )


def test_panel_constant_and_single_sample_cross_sections_are_nan(
    price_frame: pd.DataFrame,
    multi_symbol_prices: pd.DataFrame,
) -> None:
    date = price_frame["trade_date"].iloc[-1]
    constant_panel = compute_factor_panel(multi_symbol_prices)
    constant_values = constant_panel.loc[constant_panel["trade_date"] == date, "mom_5"]
    assert constant_values.isna().all()

    single_panel = compute_factor_panel(price_frame)
    single_value = single_panel.loc[single_panel["trade_date"] == date, "mom_5"]
    assert single_value.isna().all()


def test_panel_cleans_nonfinite_ret_1d_after_zero_to_positive_recovery(
    price_frame: pd.DataFrame,
) -> None:
    dirty = price_frame.iloc[:40].copy()
    dirty.loc[5, "close"] = 0.0
    dirty.loc[6, "close"] = 100.0

    panel = compute_factor_panel(dirty)
    numeric = panel.select_dtypes(include=[np.number]).to_numpy(dtype=float)
    recovery_date = dirty["trade_date"].iloc[6]

    assert not np.isinf(numeric).any()
    assert pd.isna(panel.loc[panel["trade_date"] == recovery_date, "ret_1d"]).all()
