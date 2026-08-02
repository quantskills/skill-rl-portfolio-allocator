"""Immutable metadata for the complete candidate factor catalog."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json


CATALOG_VERSION = "factor-catalog-v2"
_ALLOWED_REQUIRED_COLUMNS = frozenset({
    "trade_date",
    "symbol",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "amount",
    "ret_1d",
    "is_suspended",
})


@dataclass(frozen=True)
class FactorSpec:
    name: str
    family: str
    lookback: int
    required_columns: tuple[str, ...]
    version: str = "v1"

    def __post_init__(self) -> None:
        if isinstance(self.required_columns, str):
            raise TypeError("required_columns must be an iterable of strings")
        try:
            required_columns = tuple(self.required_columns)
        except TypeError as exc:
            raise TypeError(
                "required_columns must be an iterable of strings"
            ) from exc
        if not all(isinstance(column, str) for column in required_columns):
            raise TypeError("required_columns must contain only strings")
        object.__setattr__(self, "required_columns", required_columns)


def validate_catalog(catalog: tuple[FactorSpec, ...]) -> None:
    """Raise when a catalog does not satisfy the immutable catalog contract."""
    if len(catalog) != 100:
        raise ValueError("factor catalog must contain exactly 100 factors")
    names = [spec.name for spec in catalog]
    if len(set(names)) != len(names):
        raise ValueError("factor catalog names must be unique")
    for spec in catalog:
        if (not isinstance(spec.lookback, int)
                or isinstance(spec.lookback, bool)
                or spec.lookback <= 0):
            raise ValueError(
                f"factor {spec.name} lookback must be a positive integer"
            )
        if not isinstance(spec.required_columns, tuple):
            raise ValueError(
                f"factor {spec.name} required_columns must be a tuple"
            )
        if not spec.required_columns:
            raise ValueError(
                f"factor {spec.name} required_columns must be a non-empty tuple"
            )
        invalid_columns = [
            column for column in spec.required_columns
            if not isinstance(column, str) or column not in _ALLOWED_REQUIRED_COLUMNS
        ]
        if invalid_columns:
            raise ValueError(
                f"factor {spec.name} required_columns must contain only allowed "
                f"schema fields; invalid: {invalid_columns}"
            )
    counts = {family: sum(spec.family == family for spec in catalog)
              for family in {spec.family for spec in catalog}}
    if len(counts) != 10 or set(counts.values()) != {10}:
        raise ValueError("factor catalog must contain ten families of ten")


def catalog_hash(catalog: tuple[FactorSpec, ...]) -> str:
    """Return a stable hash of the ordered factor metadata."""
    payload = json.dumps([asdict(spec) for spec in catalog],
                         sort_keys=True, separators=(",", ":")).encode()
    return "sha256:" + hashlib.sha256(payload).hexdigest()


FACTOR_CATALOG: tuple[FactorSpec, ...] = (
    # Trend and momentum
    FactorSpec("mom_5", "momentum", 5, ("close",)),
    FactorSpec("mom_10", "momentum", 10, ("close",)),
    FactorSpec("mom_20", "momentum", 20, ("close",)),
    FactorSpec("mom_60", "momentum", 60, ("close",)),
    FactorSpec("mom_120", "momentum", 120, ("close",)),
    FactorSpec("mom_252", "momentum", 252, ("close",)),
    FactorSpec("ma_gap_5_20", "momentum", 20, ("close",)),
    FactorSpec("ma_gap_20_60", "momentum", 60, ("close",)),
    FactorSpec("ma_gap_60_120", "momentum", 120, ("close",)),
    FactorSpec("mom_accel_20_60", "momentum", 60, ("close",)),

    # Short-term reversal
    FactorSpec("reversal_1", "reversal", 1, ("close",)),
    FactorSpec("reversal_2", "reversal", 2, ("close",)),
    FactorSpec("reversal_3", "reversal", 3, ("close",)),
    FactorSpec("reversal_5", "reversal", 5, ("close",)),
    FactorSpec("reversal_10", "reversal", 10, ("close",)),
    FactorSpec("reversal_20", "reversal", 20, ("close",)),
    FactorSpec("reversal_from_high_20", "reversal", 20, ("close",)),
    FactorSpec("reversal_from_low_20", "reversal", 20, ("close",)),
    FactorSpec("return_autocorr_20", "reversal", 20, ("close",)),
    FactorSpec("short_vs_medium_reversal", "reversal", 20, ("close",)),

    # Volatility and downside risk
    FactorSpec("vol_5", "volatility", 5, ("close",)),
    FactorSpec("vol_10", "volatility", 10, ("close",)),
    FactorSpec("vol_20", "volatility", 20, ("close",)),
    FactorSpec("vol_60", "volatility", 60, ("close",)),
    FactorSpec("downside_vol_20", "volatility", 20, ("close",)),
    FactorSpec("downside_vol_60", "volatility", 60, ("close",)),
    FactorSpec("upside_vol_20", "volatility", 20, ("close",)),
    FactorSpec("semivol_ratio_20", "volatility", 20, ("close",)),
    FactorSpec("vol_ratio_5_20", "volatility", 20, ("close",)),
    FactorSpec("vol_ratio_20_60", "volatility", 60, ("close",)),

    # Range and intraday dispersion
    FactorSpec("parkinson_vol_10", "range", 10, ("high", "low")),
    FactorSpec("parkinson_vol_20", "range", 20, ("high", "low")),
    FactorSpec("parkinson_vol_60", "range", 60, ("high", "low")),
    FactorSpec("atr_5", "range", 5, ("close", "high", "low")),
    FactorSpec("atr_14", "range", 14, ("close", "high", "low")),
    FactorSpec("atr_20", "range", 20, ("close", "high", "low")),
    FactorSpec("range_mean_5", "range", 5, ("close", "high", "low")),
    FactorSpec("range_mean_20", "range", 20, ("close", "high", "low")),
    FactorSpec("range_expansion", "range", 20, ("close", "high", "low")),
    FactorSpec("range_position_20", "range", 20, ("close", "high", "low")),

    # Volume trend
    FactorSpec("volume_ratio_5_20", "volume", 20, ("volume",)),
    FactorSpec("volume_ratio_20_60", "volume", 60, ("volume",)),
    FactorSpec("volume_ratio_60_252", "volume", 252, ("volume",)),
    FactorSpec("volume_mom_5", "volume", 20, ("volume",)),
    FactorSpec("volume_mom_20", "volume", 60, ("volume",)),
    FactorSpec("volume_mom_60", "volume", 252, ("volume",)),
    FactorSpec("volume_zscore_20", "volume", 20, ("volume",)),
    FactorSpec("volume_zscore_60", "volume", 60, ("volume",)),
    FactorSpec("volume_volatility_20", "volume", 20, ("volume",)),
    FactorSpec("volume_persistence_20", "volume", 20, ("volume",)),

    # Turnover and activity; turnover uses volume / trailing 252-day volume.
    FactorSpec("turnover_5", "turnover", 252, ("volume",)),
    FactorSpec("turnover_10", "turnover", 252, ("volume",)),
    FactorSpec("turnover_20", "turnover", 252, ("volume",)),
    FactorSpec("turnover_60", "turnover", 252, ("volume",)),
    FactorSpec("turnover_std_20", "turnover", 252, ("volume",)),
    FactorSpec("turnover_std_60", "turnover", 252, ("volume",)),
    FactorSpec("turnover_cv_20", "turnover", 252, ("volume",)),
    FactorSpec("turnover_shock_5_60", "turnover", 252, ("volume",)),
    FactorSpec("turnover_change_20", "turnover", 252, ("volume",)),
    FactorSpec("active_days_ratio_20", "turnover", 20, ("volume",)),

    # Liquidity and price impact
    FactorSpec("amihud_5", "liquidity", 5, ("amount", "close")),
    FactorSpec("amihud_10", "liquidity", 10, ("amount", "close")),
    FactorSpec("amihud_20", "liquidity", 20, ("amount", "close")),
    FactorSpec("amihud_60", "liquidity", 60, ("amount", "close")),
    FactorSpec("amihud_change_5_20", "liquidity", 20, ("amount", "close")),
    FactorSpec("amihud_vol_20", "liquidity", 20, ("amount", "close")),
    FactorSpec("inverse_amount_20", "liquidity", 20, ("amount",)),
    FactorSpec("amount_zscore_20", "liquidity", 20, ("amount",)),
    FactorSpec("zero_return_ratio_20", "liquidity", 20, ("close",)),
    FactorSpec("roll_spread_20", "liquidity", 20, ("close",)),

    # Price-volume relationship
    FactorSpec("ret_volume_corr_10", "price_volume", 10, ("close", "volume")),
    FactorSpec("ret_volume_corr_20", "price_volume", 20, ("close", "volume")),
    FactorSpec("ret_volume_corr_60", "price_volume", 60, ("close", "volume")),
    FactorSpec("absret_volume_corr_20", "price_volume", 20, ("close", "volume")),
    FactorSpec("price_volume_rank_div_20", "price_volume", 20, ("close", "volume")),
    FactorSpec("obv_mom_10", "price_volume", 10, ("close", "volume")),
    FactorSpec("obv_mom_20", "price_volume", 20, ("close", "volume")),
    FactorSpec("obv_mom_60", "price_volume", 60, ("close", "volume")),
    FactorSpec("up_down_volume_ratio_20", "price_volume", 20, ("close", "volume")),
    FactorSpec("accumulation_distribution_20", "price_volume", 20,
               ("close", "high", "low", "volume")),

    # Overnight and candle structure
    FactorSpec("overnight_ret_1", "candle", 1, ("close", "open")),
    FactorSpec("overnight_ret_5", "candle", 5, ("close", "open")),
    FactorSpec("overnight_ret_20", "candle", 20, ("close", "open")),
    FactorSpec("intraday_ret_1", "candle", 1, ("close", "open")),
    FactorSpec("intraday_ret_5", "candle", 5, ("close", "open")),
    FactorSpec("intraday_ret_20", "candle", 20, ("close", "open")),
    FactorSpec("upper_shadow_20", "candle", 20, ("close", "high", "low", "open")),
    FactorSpec("lower_shadow_20", "candle", 20, ("close", "high", "low", "open")),
    FactorSpec("body_ratio_20", "candle", 20, ("close", "high", "low", "open")),
    FactorSpec("close_location_20", "candle", 20, ("close", "high", "low")),

    # Distribution shape and tail risk
    FactorSpec("ret_skew_20", "distribution", 20, ("close",)),
    FactorSpec("ret_skew_60", "distribution", 60, ("close",)),
    FactorSpec("ret_kurt_20", "distribution", 20, ("close",)),
    FactorSpec("ret_kurt_60", "distribution", 60, ("close",)),
    FactorSpec("var_5pct_20", "distribution", 20, ("close",)),
    FactorSpec("var_5pct_60", "distribution", 60, ("close",)),
    FactorSpec("cvar_5pct_20", "distribution", 20, ("close",)),
    FactorSpec("cvar_5pct_60", "distribution", 60, ("close",)),
    FactorSpec("max_ret_20", "distribution", 20, ("close",)),
    FactorSpec("min_ret_20", "distribution", 20, ("close",)),
)


validate_catalog(FACTOR_CATALOG)


__all__ = [
    "CATALOG_VERSION",
    "FACTOR_CATALOG",
    "FactorSpec",
    "catalog_hash",
    "validate_catalog",
]
