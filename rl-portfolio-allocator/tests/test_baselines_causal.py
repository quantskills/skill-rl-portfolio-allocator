import numpy as np
import pandas as pd
import pytest

from scripts.baselines import fit_static_factor_weights, rolling_ic_weights
from scripts.config import FACTOR_NAMES, K


def test_fit_static_factor_weights_uses_training_columns_and_is_l1_normalized():
    values = np.arange(60 * K, dtype=float).reshape(60, K)
    factor_returns = pd.DataFrame(values, columns=FACTOR_NAMES)

    weights = fit_static_factor_weights(factor_returns)
    factor_returns.iloc[:, 0] = -999.0

    assert weights.shape == (K,)
    assert np.sum(np.abs(weights)) == pytest.approx(1.0)
    assert np.all(np.isfinite(weights))


def test_fit_static_factor_weights_requires_60_complete_observations():
    factor_returns = pd.DataFrame(np.ones((59, K)), columns=FACTOR_NAMES)

    with pytest.raises(ValueError, match="60"):
        fit_static_factor_weights(factor_returns)


def test_fit_static_factor_weights_rejects_zero_solution():
    factor_returns = pd.DataFrame(np.zeros((60, K)), columns=FACTOR_NAMES)

    with pytest.raises(ValueError, match="zero"):
        fit_static_factor_weights(factor_returns)


def test_rolling_ic_weights_reads_current_state_and_is_l1_normalized():
    row = pd.Series({f"{factor}_ic_mean_20": i - 2 for i, factor in enumerate(FACTOR_NAMES)})

    weights = rolling_ic_weights(row)

    assert np.sum(np.abs(weights)) == pytest.approx(1.0)
    assert np.allclose(weights, np.asarray([-2, -1, 0, 1, 2, 3]) / 9.0)


def test_rolling_ic_weights_returns_zero_for_zero_signal():
    row = pd.Series({f"{factor}_ic_mean_20": 0.0 for factor in FACTOR_NAMES})

    assert np.array_equal(rolling_ic_weights(row), np.zeros(K))
