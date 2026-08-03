import numpy as np
import pytest

from scripts.config import get_config
from scripts.stress_test import _has_enough, apply_frozen_method
from scripts.state import STATE_SCHEMA_VERSION
from tests.test_weekly_env import _toy_data


def test_stress_coverage_starts_after_market_state_warmup():
    features, state = _toy_data(periods=800)
    state.iloc[:60, 0] = np.nan
    state = state.reset_index(names="trade_date")

    ok, data_start, reason = _has_enough(
        features, state, str(features.trade_date.max().date()), min_years=1
    )

    assert ok, reason
    assert data_start == str(features.trade_date.unique()[60].date())


def test_stress_coverage_uses_selected_factor_names():
    features, state = _toy_data(periods=800)
    state = state.reset_index(names="trade_date")
    selected_factor = "reversal_5"
    omitted_default_factor = "mom_20"
    state.loc[:59, f"{omitted_default_factor}_ic_mean_20"] = np.nan

    ok, data_start, reason = _has_enough(
        features,
        state,
        str(features.trade_date.max().date()),
        min_years=1,
        factor_names=[selected_factor],
    )

    assert ok, reason
    assert data_start == str(features.trade_date.unique()[0].date())
    assert not np.isnan(state.loc[0, f"{selected_factor}_ic_mean_20"])


def test_frozen_method_supplies_factor_metadata_to_stress_config():
    cfg = get_config()
    names = list(cfg["factor_names"][:3])
    method = {
        "frozen_candidate": "candidate-a",
        "buffer_config": {},
        "factor_catalog_version": "catalog-v1",
        "factor_catalog_hash": "sha256:catalog",
        "selected_factors": names,
        "factor_directions": [-1, 1, -1],
        "selection_run_id": "selection-42",
        "fold": 2,
        "state_schema_version": STATE_SCHEMA_VERSION,
        "training_budget": 10,
    }

    applied, budget = apply_frozen_method(cfg, method, default_budget=1)

    assert applied["factor_names"] == names
    assert applied["factor_directions"] == [-1, 1, -1]
    assert applied["k"] == 3
    assert budget == 10


def test_stress_frozen_method_rejects_incomplete_factor_contract():
    with pytest.raises(ValueError, match="complete factor contract fields"):
        apply_frozen_method(
            get_config(),
            {"frozen_candidate": "candidate-a", "buffer_config": {}},
            default_budget=1,
        )


def test_stress_frozen_method_supports_legacy_factor_names_alias():
    names = list(get_config()["factor_names"][:2])
    method = {
        "frozen_candidate": "candidate-a",
        "buffer_config": {},
        "factor_catalog_version": "catalog-v1",
        "factor_catalog_hash": "sha256:catalog",
        "factor_names": names,
        "factor_directions": [1, -1],
        "selection_run_id": "selection-42",
        "fold": 2,
        "schema_version": STATE_SCHEMA_VERSION,
    }

    applied, _ = apply_frozen_method(get_config(), method, default_budget=1)

    assert applied["factor_names"] == names
    assert applied["factor_directions"] == [1, -1]
