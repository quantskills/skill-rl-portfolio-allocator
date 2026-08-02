from dataclasses import FrozenInstanceError
from dataclasses import replace

import pytest

from scripts.config import CONTROL_FACTOR_NAMES, FACTOR_NAMES, K
from scripts.factor_catalog import (
    CATALOG_VERSION,
    FACTOR_CATALOG,
    FactorSpec,
    catalog_hash,
    validate_catalog,
)


EXPECTED_NAMES = (
    "mom_5", "mom_10", "mom_20", "mom_60", "mom_120", "mom_252",
    "ma_gap_5_20", "ma_gap_20_60", "ma_gap_60_120", "mom_accel_20_60",
    "reversal_1", "reversal_2", "reversal_3", "reversal_5", "reversal_10",
    "reversal_20", "reversal_from_high_20", "reversal_from_low_20",
    "return_autocorr_20", "short_vs_medium_reversal",
    "vol_5", "vol_10", "vol_20", "vol_60", "downside_vol_20",
    "downside_vol_60", "upside_vol_20", "semivol_ratio_20", "vol_ratio_5_20",
    "vol_ratio_20_60",
    "parkinson_vol_10", "parkinson_vol_20", "parkinson_vol_60", "atr_5",
    "atr_14", "atr_20", "range_mean_5", "range_mean_20", "range_expansion",
    "range_position_20",
    "volume_ratio_5_20", "volume_ratio_20_60", "volume_ratio_60_252",
    "volume_mom_5", "volume_mom_20", "volume_mom_60", "volume_zscore_20",
    "volume_zscore_60", "volume_volatility_20", "volume_persistence_20",
    "turnover_5", "turnover_10", "turnover_20", "turnover_60", "turnover_std_20",
    "turnover_std_60", "turnover_cv_20", "turnover_shock_5_60",
    "turnover_change_20", "active_days_ratio_20",
    "amihud_5", "amihud_10", "amihud_20", "amihud_60", "amihud_change_5_20",
    "amihud_vol_20", "inverse_amount_20", "amount_zscore_20", "zero_return_ratio_20",
    "roll_spread_20",
    "ret_volume_corr_10", "ret_volume_corr_20", "ret_volume_corr_60",
    "absret_volume_corr_20", "price_volume_rank_div_20", "obv_mom_10", "obv_mom_20",
    "obv_mom_60", "up_down_volume_ratio_20", "accumulation_distribution_20",
    "overnight_ret_1", "overnight_ret_5", "overnight_ret_20", "intraday_ret_1",
    "intraday_ret_5", "intraday_ret_20", "upper_shadow_20", "lower_shadow_20",
    "body_ratio_20", "close_location_20",
    "ret_skew_20", "ret_skew_60", "ret_kurt_20", "ret_kurt_60", "var_5pct_20",
    "var_5pct_60", "cvar_5pct_20", "cvar_5pct_60", "max_ret_20", "min_ret_20",
)


def test_catalog_has_exactly_ten_families_of_ten_unique_factors():
    assert CATALOG_VERSION == "factor-catalog-v2"
    assert len(FACTOR_CATALOG) == 100
    assert tuple(spec.name for spec in FACTOR_CATALOG) == EXPECTED_NAMES
    assert len({spec.name for spec in FACTOR_CATALOG}) == 100
    families = {spec.family for spec in FACTOR_CATALOG}
    assert len(families) == 10
    assert {family: sum(s.family == family for s in FACTOR_CATALOG)
            for family in families} == {family: 10 for family in families}


def test_existing_six_are_present_once():
    existing = {"mom_20", "reversal_5", "vol_20", "turnover_20",
                "amihud_20", "ret_skew_60"}
    names = [spec.name for spec in FACTOR_CATALOG]
    assert existing <= set(names)
    assert all(names.count(name) == 1 for name in existing)


def test_catalog_hash_is_order_sensitive_and_stable():
    assert catalog_hash(FACTOR_CATALOG) == catalog_hash(FACTOR_CATALOG)
    assert catalog_hash(tuple(reversed(FACTOR_CATALOG))) != catalog_hash(FACTOR_CATALOG)
    assert catalog_hash(FACTOR_CATALOG).startswith("sha256:")


def test_factor_spec_is_frozen_and_has_public_shape():
    spec = FactorSpec("x", "family", 5, ("close",))
    assert spec.version == "v1"
    with pytest.raises(FrozenInstanceError):
        spec.name = "changed"


def test_factor_spec_normalizes_required_columns_and_detaches_input_list():
    required_columns = ["close"]
    spec = FactorSpec("x", "family", 5, required_columns)
    before = catalog_hash((spec,))

    assert spec.required_columns == ("close",)
    assert isinstance(spec.required_columns, tuple)
    with pytest.raises(AttributeError):
        spec.required_columns.append("volume")

    required_columns.append("volume")
    assert spec.required_columns == ("close",)
    assert catalog_hash((spec,)) == before


@pytest.mark.parametrize("required_columns", [["close", ["volume"]], [["close"]]])
def test_factor_spec_rejects_non_string_required_columns(required_columns):
    with pytest.raises((TypeError, ValueError), match="required_columns"):
        FactorSpec("x", "family", 5, required_columns)


def test_validate_catalog_accepts_design_raw_return_column():
    catalog = list(FACTOR_CATALOG)
    catalog[0] = replace(catalog[0], required_columns=("ret_1d",))
    validate_catalog(tuple(catalog))


@pytest.mark.parametrize(
    ("catalog", "message"),
    [
        (FACTOR_CATALOG[:-1], "exactly 100"),
        (FACTOR_CATALOG[:-1] + (FACTOR_CATALOG[0],), "unique"),
        (tuple(FactorSpec(f"factor_{i}", "one_family", 1, ("close",))
               for i in range(100)), "ten families"),
    ],
)
def test_validate_catalog_rejects_invalid_catalogs(catalog, message):
    with pytest.raises(ValueError, match=message):
        validate_catalog(catalog)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("lookback", 0, "lookback.*positive integer"),
        ("lookback", -1, "lookback.*positive integer"),
        ("lookback", 1.5, "lookback.*positive integer"),
        ("lookback", True, "lookback.*positive integer"),
        ("required_columns", (), "required_columns.*non-empty tuple"),
        ("required_columns", ("unknown_column",), "required_columns.*allowed"),
    ],
)
def test_validate_catalog_rejects_invalid_factor_metadata(field, value, message):
    catalog = list(FACTOR_CATALOG)
    catalog[0] = replace(catalog[0], **{field: value})
    with pytest.raises(ValueError, match=message):
        validate_catalog(tuple(catalog))


def test_validate_catalog_rejects_mutable_required_columns_even_if_bypassed():
    spec = replace(FACTOR_CATALOG[0])
    object.__setattr__(spec, "required_columns", ["close"])
    catalog = list(FACTOR_CATALOG)
    catalog[0] = spec
    with pytest.raises(ValueError, match="required_columns.*tuple"):
        validate_catalog(tuple(catalog))


def test_config_keeps_six_factor_control_compatibility():
    expected = [
        "mom_20", "reversal_5", "vol_20",
        "turnover_20", "amihud_20", "ret_skew_60",
    ]
    assert CONTROL_FACTOR_NAMES == expected
    assert FACTOR_NAMES is CONTROL_FACTOR_NAMES
    assert K == len(CONTROL_FACTOR_NAMES) == 6
