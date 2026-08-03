import numpy as np
import pytest

from scripts.metrics import metrics_pack


def test_metrics_pack_accepts_weekly_grouped_daily_returns():
    grouped_returns = [[0.01, -0.02], [0.03, 0.0]]

    metrics = metrics_pack(grouped_returns, "test")

    assert metrics["mdd"] == pytest.approx(-0.02)
    assert metrics["cumret"] == pytest.approx(
        np.prod(1.0 + np.array([0.01, -0.02, 0.03, 0.0])) - 1.0
    )
