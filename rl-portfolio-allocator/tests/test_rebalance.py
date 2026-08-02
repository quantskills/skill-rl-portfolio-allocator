import numpy as np
import pytest

from scripts.rebalance import (
    buffered_long_short,
    project_turnover,
    weekly_decision_indices,
)


def test_weekly_decision_indices_selects_first_trading_date_per_iso_week():
    dates = np.array(["2024-01-02", "2024-01-03", "2024-01-08", "2024-01-15"])

    assert weekly_decision_indices(dates).tolist() == [0, 2, 3]


def test_buffered_long_short_keeps_existing_positions_inside_rank_buffer():
    scores = np.array([10.0, 9.0, 8.0, 7.0, 6.0, 5.0])
    suspended = np.zeros(6, dtype=bool)
    prev_w = np.array([0.1, 0.0, 0.0, 0.0, 0.0, -0.1])

    longs, shorts = buffered_long_short(
        scores, suspended, prev_w,
        long_entry=2, long_exit=3,
        short_entry=2, short_exit=3,
    )

    assert longs.tolist() == [0, 1, 2]
    assert shorts.tolist() == [3, 4, 5]


def test_buffered_long_short_does_not_select_suspended_names():
    scores = np.array([10.0, 9.0, 8.0, 7.0])
    suspended = np.array([False, True, False, False])
    prev_w = np.zeros(4)

    longs, shorts = buffered_long_short(
        scores, suspended, prev_w,
        long_entry=2, long_exit=2,
        short_entry=1, short_exit=1,
    )

    assert longs.tolist() == [0, 2]
    assert shorts.tolist() == [3]


def test_project_turnover_preserves_frozen_and_respects_caps_and_budget():
    prev = np.array([0.4, -0.2, 0.0, 0.0])
    target = np.array([0.1, -0.4, 0.8, -0.8])
    frozen = np.array([True, False, False, False])

    out = project_turnover(prev, target, frozen, budget=0.3, long_cap=0.8, short_cap=0.5)

    assert out[0] == prev[0]
    assert np.clip(out, 0, None).sum() <= 0.8 + 1e-12
    assert np.clip(-out, 0, None).sum() <= 0.5 + 1e-12
    assert np.abs(out - prev).sum() <= 0.3 + 1e-12


def test_project_turnover_closes_before_reversing():
    prev = np.array([0.4, -0.4])
    target = np.array([-0.5, 0.5])

    out = project_turnover(prev, target, np.zeros(2, dtype=bool), budget=2.0,
                           long_cap=1.0, short_cap=1.0)

    assert out.tolist() == [0.0, 0.0]


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_project_turnover_rejects_nonfinite_inputs(bad):
    with pytest.raises(ValueError):
        project_turnover(np.array([0.0]), np.array([bad]), np.array([False]),
                         budget=1.0, long_cap=1.0, short_cap=1.0)
