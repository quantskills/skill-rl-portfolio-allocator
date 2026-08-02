"""Causal raw computation for the first forty catalog factors.

This module deliberately stops before cross-sectional normalization and factor
selection.  The public ``catalog`` argument is the extension point for later
factor families; only catalog entries with an implemented formula are accepted.
"""
from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from scripts.factor_catalog import FACTOR_CATALOG, FactorSpec


EPS = 1e-12
SUPPORTED_FACTOR_CATALOG: tuple[FactorSpec, ...] = FACTOR_CATALOG[:40]
_SUPPORTED_FACTOR_NAMES = frozenset(spec.name for spec in SUPPORTED_FACTOR_CATALOG)
_CANONICAL_FACTOR_SPECS = {
    spec.name: spec for spec in SUPPORTED_FACTOR_CATALOG
}


def _positive_finite(series: pd.Series) -> pd.Series:
    return series.where(np.isfinite(series) & (series > EPS))


def safe_div(num: pd.Series, den: pd.Series) -> pd.Series:
    """Divide while treating zero and near-zero denominators as missing."""
    numerator = num.where(np.isfinite(num))
    denominator = _positive_finite(den)
    quotient = numerator / denominator
    return quotient.where(np.isfinite(quotient))


def log_return(close: pd.Series, window: int) -> pd.Series:
    """Return the causal log return over ``window`` observations."""
    valid_close = _positive_finite(close)
    result = np.log(safe_div(valid_close, valid_close.shift(window)))
    return result.where(np.isfinite(result))


def rolling_corr(left: pd.Series, right: pd.Series, window: int) -> pd.Series:
    """Calculate a full-window trailing Pearson correlation."""
    return left.rolling(window, min_periods=window).corr(right)


def downside_std(rets: pd.Series, window: int) -> pd.Series:
    """Calculate trailing standard deviation of negative returns only."""
    return rets.where(rets < 0).rolling(window, min_periods=window).std()


def upside_std(rets: pd.Series, window: int) -> pd.Series:
    """Calculate trailing standard deviation of positive returns only."""
    return rets.where(rets > 0).rolling(window, min_periods=window).std()


def _rolling_mean(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window, min_periods=window).mean()


def _rolling_std(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window, min_periods=window).std()


def _rolling_min(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window, min_periods=window).min()


def _rolling_max(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window, min_periods=window).max()


def _single_symbol_factors(
    frame: pd.DataFrame,
    selected_catalog: Sequence[FactorSpec],
) -> pd.DataFrame:
    """Compute the supported factors for one chronologically ordered symbol."""
    selected_required_columns = {
        column
        for spec in selected_catalog
        for column in spec.required_columns
    }
    if "close" in selected_required_columns:
        close = _positive_finite(frame["close"])
    else:
        close = pd.Series(np.nan, index=frame.index, dtype=float)

    returns = log_return(close, 1)
    mom_5 = log_return(close, 5)
    mom_20 = log_return(close, 20)
    mom_60 = log_return(close, 60)

    close_ma_5 = _rolling_mean(close, 5)
    close_ma_20 = _rolling_mean(close, 20)
    close_ma_60 = _rolling_mean(close, 60)
    close_ma_120 = _rolling_mean(close, 120)

    factors: dict[str, pd.Series] = {
        # Trend and momentum.
        "mom_5": mom_5,
        "mom_10": log_return(close, 10),
        "mom_20": mom_20,
        "mom_60": mom_60,
        "mom_120": log_return(close, 120),
        "mom_252": log_return(close, 252),
        "ma_gap_5_20": safe_div(close_ma_5, close_ma_20) - 1.0,
        "ma_gap_20_60": safe_div(close_ma_20, close_ma_60) - 1.0,
        "ma_gap_60_120": safe_div(close_ma_60, close_ma_120) - 1.0,
        "mom_accel_20_60": mom_20 - mom_60 / 3.0,
        # Short-term reversal.
        "reversal_1": -returns,
        "reversal_2": -log_return(close, 2),
        "reversal_3": -log_return(close, 3),
        "reversal_5": -mom_5,
        "reversal_10": -log_return(close, 10),
        "reversal_20": -mom_20,
        "reversal_from_high_20": safe_div(close, _rolling_max(close, 20)) - 1.0,
        "reversal_from_low_20": -(safe_div(close, _rolling_min(close, 20)) - 1.0),
        "return_autocorr_20": rolling_corr(returns, returns.shift(1), 20),
        "short_vs_medium_reversal": -(mom_5 - mom_20 / 4.0),
        # Volatility and downside risk.
        "vol_5": _rolling_std(returns, 5),
        "vol_10": _rolling_std(returns, 10),
        "vol_20": _rolling_std(returns, 20),
        "vol_60": _rolling_std(returns, 60),
    }

    downside_20 = downside_std(returns, 20)
    downside_60 = downside_std(returns, 60)
    upside_20 = upside_std(returns, 20)
    vol_5 = factors["vol_5"]
    vol_20 = factors["vol_20"]
    vol_60 = factors["vol_60"]
    factors.update(
        {
            "downside_vol_20": downside_20,
            "downside_vol_60": downside_60,
            "upside_vol_20": upside_20,
            "semivol_ratio_20": safe_div(downside_20, upside_20),
            "vol_ratio_5_20": safe_div(vol_5, vol_20),
            "vol_ratio_20_60": safe_div(vol_20, vol_60),
        }
    )

    if "high" in selected_required_columns or "low" in selected_required_columns:
        high = _positive_finite(frame["high"])
        low = _positive_finite(frame["low"])
        # True range uses the first bar's high-low when no previous close exists.
        previous_close = close.shift(1)
        high_low = high - low
        true_range = pd.concat(
            [high_low, (high - previous_close).abs(), (low - previous_close).abs()],
            axis=1,
        ).max(axis=1, skipna=False)
        if not true_range.empty:
            true_range.iloc[0] = high_low.iloc[0]
        intraday_range = safe_div(high - low, close)
        log_high_low = np.log(safe_div(high, low))

        def parkinson(window: int) -> pd.Series:
            return np.sqrt(
                _rolling_mean(log_high_low.pow(2), window) / (4.0 * np.log(2.0))
            )

        def atr(window: int) -> pd.Series:
            return _rolling_mean(true_range, window)

        range_mean_5 = _rolling_mean(intraday_range, 5)
        range_mean_20 = _rolling_mean(intraday_range, 20)
        factors.update(
            {
                # Range and intraday dispersion.
                "parkinson_vol_10": parkinson(10),
                "parkinson_vol_20": parkinson(20),
                "parkinson_vol_60": parkinson(60),
                "atr_5": atr(5),
                "atr_14": atr(14),
                "atr_20": atr(20),
                "range_mean_5": range_mean_5,
                "range_mean_20": range_mean_20,
                "range_expansion": safe_div(range_mean_5, range_mean_20),
                "range_position_20": safe_div(
                    close - _rolling_min(low, 20),
                    _rolling_max(high, 20) - _rolling_min(low, 20),
                ),
            }
        )

    return pd.DataFrame(
        {spec.name: factors[spec.name] for spec in selected_catalog},
        index=frame.index,
    )


def _validate_catalog(catalog: Sequence[FactorSpec]) -> tuple[FactorSpec, ...]:
    selected = tuple(catalog)
    if not selected:
        raise ValueError("factor catalog must not be empty")
    unsupported = [
        spec.name for spec in selected if spec.name not in _SUPPORTED_FACTOR_NAMES
    ]
    if unsupported:
        raise NotImplementedError(
            "raw formulas are not implemented for catalog factors: "
            + ", ".join(unsupported)
        )
    noncanonical = [
        spec.name
        for spec in selected
        if spec != _CANONICAL_FACTOR_SPECS[spec.name]
    ]
    if noncanonical:
        raise ValueError(
            "custom catalog factors must match canonical metadata: "
            + ", ".join(noncanonical)
        )
    names = [spec.name for spec in selected]
    if len(set(names)) != len(names):
        raise ValueError("factor catalog names must be unique")
    return selected


def _validate_raw_columns(frame: pd.DataFrame, catalog: Sequence[FactorSpec]) -> None:
    required = {
        column for spec in catalog for column in spec.required_columns
    }
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(
            "raw frame is missing catalog-required columns: " + ", ".join(missing)
        )


def compute_raw_factors(
    frame: pd.DataFrame,
    catalog: Sequence[FactorSpec] | None = None,
) -> pd.DataFrame:
    """Append causal raw factors to ``frame`` in catalog order.

    The input must already be ordered chronologically within each symbol.  If a
    ``symbol`` column exists, every symbol is computed independently; otherwise
    the frame is treated as one symbol.  The default catalog is the first forty
    entries because later families are implemented by subsequent tasks.
    """
    selected_catalog = _validate_catalog(
        SUPPORTED_FACTOR_CATALOG if catalog is None else catalog
    )
    _validate_raw_columns(frame, selected_catalog)
    factor_names = [spec.name for spec in selected_catalog]

    work = frame.reset_index(drop=True)
    factor_values = pd.DataFrame(
        np.nan, index=work.index, columns=factor_names, dtype=np.float64
    )
    if "symbol" in work.columns:
        groups = work.groupby("symbol", sort=False, dropna=False)
    else:
        groups = [(None, work)]

    for _, group in groups:
        if "trade_date" in group.columns:
            group = group.sort_values("trade_date", kind="mergesort")
        computed = _single_symbol_factors(group, selected_catalog)
        factor_values.loc[group.index, factor_names] = computed.loc[
            :, factor_names
        ].to_numpy()

    result = frame.drop(columns=factor_names, errors="ignore").copy()
    with np.errstate(over="ignore", invalid="ignore"):
        factor_values_float32 = factor_values.to_numpy(dtype=np.float32)
    factor_values_float32[~np.isfinite(factor_values_float32)] = np.nan
    for position, name in enumerate(factor_names):
        result[name] = factor_values_float32[:, position]
    return result


__all__ = [
    "EPS",
    "SUPPORTED_FACTOR_CATALOG",
    "compute_raw_factors",
    "downside_std",
    "log_return",
    "rolling_corr",
    "safe_div",
    "upside_std",
]
