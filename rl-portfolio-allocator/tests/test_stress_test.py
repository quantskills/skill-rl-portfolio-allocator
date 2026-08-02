import numpy as np

from scripts.config import get_config
from scripts.stress_test import _has_enough, apply_frozen_method
from tests.test_weekly_env import _toy_data


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
        "factor_names": names,
        "training_budget": 10,
    }

    applied, budget = apply_frozen_method(cfg, method, default_budget=1)

    assert applied["factor_names"] == names
    assert applied["k"] == 3
    assert budget == 10
