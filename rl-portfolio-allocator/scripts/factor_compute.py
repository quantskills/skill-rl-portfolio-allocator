"""Causal computation for the catalog's 100 OHLCV factors."""
from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from scripts.factor_catalog import FACTOR_CATALOG, FactorSpec


EPS = 1e-12
SUPPORTED_FACTOR_CATALOG: tuple[FactorSpec, ...] = FACTOR_CATALOG
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


def _rolling_sum(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window, min_periods=window).sum()


def _rolling_skew(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window, min_periods=window).apply(
        lambda values: pd.Series(values).skew(), raw=True
    )


def _rolling_kurt(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window, min_periods=window).apply(
        lambda values: pd.Series(values).kurt(), raw=True
    )


def _finite(series: pd.Series) -> pd.Series:
    return series.where(np.isfinite(series))


def _pct_return(close: pd.Series) -> pd.Series:
    return safe_div(close, close.shift(1)) - 1.0


def _rolling_rank(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window, min_periods=window).apply(
        lambda values: pd.Series(values).rank(pct=True).iloc[-1], raw=True
    )


def _cvar(series: pd.Series, window: int) -> pd.Series:
    def tail_mean(values: np.ndarray) -> float:
        quantile = np.quantile(values, 0.05)
        tail = values[values <= quantile]
        return float(tail.mean()) if tail.size else np.nan

    return series.rolling(window, min_periods=window).apply(tail_mean, raw=True)


def _single_symbol_factors(
    frame: pd.DataFrame,
    selected_catalog: Sequence[FactorSpec],
) -> pd.DataFrame:
    """Compute selected factors for one chronologically ordered symbol."""
    selected_required_columns = {
        column
        for spec in selected_catalog
        for column in spec.required_columns
    }
    def column(name: str) -> pd.Series:
        if name in selected_required_columns:
            return _finite(frame[name])
        return pd.Series(np.nan, index=frame.index, dtype=float)

    close = _positive_finite(column("close"))
    high = _positive_finite(column("high"))
    low = _positive_finite(column("low"))
    open_ = _positive_finite(column("open"))
    volume = _finite(column("volume"))
    amount = _finite(column("amount"))

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

    # Volume trend.
    volume_mean_5 = _rolling_mean(volume, 5)
    volume_mean_20 = _rolling_mean(volume, 20)
    volume_mean_60 = _rolling_mean(volume, 60)
    volume_mean_252 = _rolling_mean(volume, 252)
    volume_returns = _pct_return(volume)
    factors.update(
        {
            "volume_ratio_5_20": safe_div(volume_mean_5, volume_mean_20),
            "volume_ratio_20_60": safe_div(volume_mean_20, volume_mean_60),
            "volume_ratio_60_252": safe_div(volume_mean_60, volume_mean_252),
            "volume_mom_5": np.log(safe_div(volume_mean_5, volume_mean_20)),
            "volume_mom_20": np.log(safe_div(volume_mean_20, volume_mean_60)),
            "volume_mom_60": np.log(safe_div(volume_mean_60, volume_mean_252)),
            "volume_zscore_20": safe_div(
                volume - volume_mean_20, _rolling_std(volume, 20)
            ),
            "volume_zscore_60": safe_div(
                volume - volume_mean_60, _rolling_std(volume, 60)
            ),
            "volume_volatility_20": _rolling_std(volume_returns, 20),
            "volume_persistence_20": rolling_corr(volume, volume.shift(1), 20),
        }
    )

    # Turnover/activity proxy: volume relative to its trailing 252-day mean.
    turnover_proxy = safe_div(volume, volume_mean_252)
    turnover_5 = _rolling_mean(turnover_proxy, 5)
    turnover_10 = _rolling_mean(turnover_proxy, 10)
    turnover_20 = _rolling_mean(turnover_proxy, 20)
    turnover_60 = _rolling_mean(turnover_proxy, 60)
    turnover_std_20 = _rolling_std(turnover_proxy, 20)
    turnover_std_60 = _rolling_std(turnover_proxy, 60)
    factors.update(
        {
            "turnover_5": turnover_5,
            "turnover_10": turnover_10,
            "turnover_20": turnover_20,
            "turnover_60": turnover_60,
            "turnover_std_20": turnover_std_20,
            "turnover_std_60": turnover_std_60,
            "turnover_cv_20": safe_div(turnover_std_20, turnover_20),
            "turnover_shock_5_60": safe_div(turnover_5, turnover_60),
            "turnover_change_20": turnover_5 - turnover_20,
            "active_days_ratio_20": (volume > EPS).where(volume.notna()).rolling(
                20, min_periods=20
            ).mean(),
        }
    )

    # Liquidity and price impact.
    pct_returns = _pct_return(close)
    impact = safe_div(pct_returns.abs(), amount + EPS)
    amihud_5 = _rolling_mean(impact, 5)
    amihud_10 = _rolling_mean(impact, 10)
    amihud_20 = _rolling_mean(impact, 20)
    amihud_60 = _rolling_mean(impact, 60)
    adjacent_cov = pct_returns.rolling(20, min_periods=20).cov(
        pct_returns.shift(1)
    )
    roll_spread = 2.0 * np.sqrt((-adjacent_cov).clip(lower=0.0))
    factors.update(
        {
            "amihud_5": amihud_5,
            "amihud_10": amihud_10,
            "amihud_20": amihud_20,
            "amihud_60": amihud_60,
            "amihud_change_5_20": safe_div(amihud_5, amihud_20) - 1.0,
            "amihud_vol_20": _rolling_std(amihud_20, 20),
            "inverse_amount_20": safe_div(
                pd.Series(1.0, index=frame.index), _rolling_mean(amount, 20)
            ),
            "amount_zscore_20": safe_div(
                amount - _rolling_mean(amount, 20), _rolling_std(amount, 20)
            ),
            "zero_return_ratio_20": (pct_returns == 0.0).where(
                pct_returns.notna()
            ).rolling(20, min_periods=20).mean(),
            "roll_spread_20": roll_spread,
        }
    )

    # Price-volume relationship.
    obv_increment = pct_returns.fillna(0.0).apply(np.sign) * volume
    obv = obv_increment.cumsum()
    up_volume = volume.where(pct_returns > 0.0, 0.0).where(pct_returns.notna())
    down_volume = volume.where(pct_returns < 0.0, 0.0).where(pct_returns.notna())
    ad_line = safe_div(2.0 * close - high - low, high - low) * volume
    factors.update(
        {
            "ret_volume_corr_10": rolling_corr(pct_returns, volume, 10),
            "ret_volume_corr_20": rolling_corr(pct_returns, volume, 20),
            "ret_volume_corr_60": rolling_corr(pct_returns, volume, 60),
            "absret_volume_corr_20": rolling_corr(pct_returns.abs(), volume, 20),
            "price_volume_rank_div_20": _rolling_rank(close, 20)
            - _rolling_rank(volume, 20),
            "obv_mom_10": obv - obv.shift(10),
            "obv_mom_20": obv - obv.shift(20),
            "obv_mom_60": obv - obv.shift(60),
            "up_down_volume_ratio_20": safe_div(
                _rolling_sum(up_volume, 20), _rolling_sum(down_volume, 20)
            ),
            "accumulation_distribution_20": _rolling_sum(ad_line, 20),
        }
    )

    # Overnight/intraday and candle structure.
    overnight_1 = safe_div(open_, close.shift(1)) - 1.0
    intraday_1 = safe_div(close, open_) - 1.0
    overnight_log = np.log(safe_div(open_, close.shift(1)))
    intraday_log = np.log(safe_div(close, open_))
    candle_range = (high - low).clip(lower=EPS)
    upper_shadow = (
        high - pd.concat([open_, close], axis=1).max(axis=1)
    ) / candle_range
    lower_shadow = (
        pd.concat([open_, close], axis=1).min(axis=1) - low
    ) / candle_range
    body_ratio = (close - open_).abs() / candle_range
    close_location = (close - low) / candle_range
    factors.update(
        {
            "overnight_ret_1": overnight_1,
            "overnight_ret_5": overnight_log.rolling(5, min_periods=5).sum(),
            "overnight_ret_20": overnight_log.rolling(20, min_periods=20).sum(),
            "intraday_ret_1": intraday_1,
            "intraday_ret_5": intraday_log.rolling(5, min_periods=5).sum(),
            "intraday_ret_20": intraday_log.rolling(20, min_periods=20).sum(),
            "upper_shadow_20": _rolling_mean(upper_shadow, 20),
            "lower_shadow_20": _rolling_mean(lower_shadow, 20),
            "body_ratio_20": _rolling_mean(body_ratio, 20),
            "close_location_20": _rolling_mean(close_location, 20),
        }
    )

    # Distribution shape and tail risk use one-day percentage returns.  Avoid
    # invoking pandas' all-NaN rolling reducers for dependency-only subsets.
    distribution_names = {
        "ret_skew_20", "ret_skew_60", "ret_kurt_20", "ret_kurt_60",
        "var_5pct_20", "var_5pct_60", "cvar_5pct_20", "cvar_5pct_60",
        "max_ret_20", "min_ret_20",
    }
    if distribution_names.intersection(spec.name for spec in selected_catalog):
        distribution_factors = {
            "ret_skew_20": _rolling_skew(pct_returns, 20),
            "ret_skew_60": _rolling_skew(pct_returns, 60),
            "ret_kurt_20": _rolling_kurt(pct_returns, 20),
            "ret_kurt_60": _rolling_kurt(pct_returns, 60),
            "var_5pct_20": pct_returns.rolling(20, min_periods=20).quantile(0.05),
            "var_5pct_60": pct_returns.rolling(60, min_periods=60).quantile(0.05),
            "cvar_5pct_20": _cvar(pct_returns, 20),
            "cvar_5pct_60": _cvar(pct_returns, 60),
            "max_ret_20": pct_returns.rolling(20, min_periods=20).max(),
            "min_ret_20": pct_returns.rolling(20, min_periods=20).min(),
        }
    else:
        distribution_factors = {
            name: pd.Series(np.nan, index=frame.index, dtype=float)
            for name in distribution_names
        }
    factors.update(distribution_factors)

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
    the frame is treated as one symbol.  The default catalog is the complete
    canonical 100-factor catalog; pass a canonical subset for family-batched
    computation.
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
        with np.errstate(all="ignore"):
            computed = _single_symbol_factors(group, selected_catalog)
        factor_values.loc[group.index, factor_names] = computed.loc[:, factor_names].to_numpy()

    base = frame.drop(columns=factor_names, errors="ignore").copy()
    with np.errstate(over="ignore", invalid="ignore"):
        factor_values_float32 = factor_values.to_numpy(dtype=np.float32)
    factor_values_float32[~np.isfinite(factor_values_float32)] = np.nan
    factor_frame = pd.DataFrame(
        factor_values_float32,
        index=frame.index,
        columns=factor_names,
    )
    return pd.concat([base, factor_frame], axis=1)


def compute_factor_panel(prices: pd.DataFrame) -> pd.DataFrame:
    """Compute and cross-sectionally normalize all catalog factors.

    Rows are ordered by symbol and trade date for causal per-symbol formulas,
    then returned by trade date and symbol for downstream panel consumers.
    """
    required = {
        "trade_date", "symbol", "open", "high", "low", "close", "volume", "amount"
    }
    missing = sorted(required.difference(prices.columns))
    if missing:
        raise ValueError("prices missing columns: " + ", ".join(missing))

    ordered = prices.sort_values(["symbol", "trade_date"], kind="mergesort").reset_index(drop=True)
    if "is_suspended" not in ordered.columns:
        ordered = ordered.assign(is_suspended=False)
    if "ret_1d" not in ordered.columns:
        ordered["ret_1d"] = _finite(
            ordered.groupby("symbol", sort=False)["close"].pct_change(
                fill_method=None
            )
        )
    else:
        ordered["ret_1d"] = _finite(ordered["ret_1d"])

    raw = compute_raw_factors(ordered)
    factor_names = [spec.name for spec in FACTOR_CATALOG]
    for name in factor_names:
        grouped = raw.groupby("trade_date", sort=False)[name]
        quantiles = grouped.quantile([0.01, 0.99]).unstack()
        lower = raw["trade_date"].map(quantiles[0.01])
        upper = raw["trade_date"].map(quantiles[0.99])
        winsorized = raw[name].clip(lower=lower, upper=upper)
        winsorized_grouped = winsorized.groupby(raw["trade_date"], sort=False)
        mean = winsorized_grouped.transform("mean")
        std = winsorized_grouped.transform("std")
        with np.errstate(all="ignore"):
            normalized = ((winsorized - mean) / std).where(std > 0.0)
        raw[name] = normalized.clip(-3.0, 3.0).astype(np.float32)
        raw[name] = raw[name].where(np.isfinite(raw[name]))

    numeric_columns = raw.select_dtypes(include=[np.number]).columns
    for column in numeric_columns:
        values = raw[column].to_numpy(copy=True)
        values[~np.isfinite(values)] = np.nan
        raw[column] = values
    for name in factor_names:
        raw[name] = raw[name].astype(np.float32)

    return raw.sort_values(["trade_date", "symbol"], kind="mergesort").reset_index(drop=True)


__all__ = [
    "EPS",
    "SUPPORTED_FACTOR_CATALOG",
    "compute_factor_panel",
    "compute_raw_factors",
    "downside_std",
    "log_return",
    "rolling_corr",
    "safe_div",
    "upside_std",
]
