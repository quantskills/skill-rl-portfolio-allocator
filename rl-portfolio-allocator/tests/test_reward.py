import numpy as np
import pytest

from scripts.config import get_config
from scripts.diagnostics import reward_quality_report
from scripts.reward import compose_reward, reward_coefficients


def _parts(net_ret, prev_drawdown=0.0, drawdown=0.1, turnover=0.8, hhi_val=0.05,
           variant="medium"):
    cfg = get_config()
    cfg["reward_variant"] = variant
    return compose_reward(net_ret, prev_drawdown, drawdown, turnover, hhi_val, cfg)


def test_reward_coefficients_are_explicit():
    assert reward_coefficients("none") == (0.0, 0.0, 0.0)
    assert reward_coefficients("low") == (0.5, 0.5, 0.05)
    assert reward_coefficients("medium") == (1.0, 1.0, 0.10)


def test_reward_parts_have_new_contract_without_dsr():
    total, parts = _parts(0.005)
    for key in ("scaled_net_return", "incremental_drawdown_penalty", "turnover_penalty",
                "concentration_penalty", "total"):
        assert key in parts, f"missing {key}"
    assert "dsr" not in parts
    subtotal = (parts["scaled_net_return"] + parts["incremental_drawdown_penalty"]
                + parts["turnover_penalty"] + parts["concentration_penalty"])
    assert abs(subtotal - parts["total"]) < 1e-12
    assert abs(total - parts["total"]) < 1e-12


def test_scaled_return_and_incremental_excess_penalties():
    _, parts = _parts(0.005, prev_drawdown=0.02, drawdown=0.05,
                      turnover=0.30, hhi_val=0.05, variant="low")
    assert parts["scaled_net_return"] == 0.5
    assert parts["incremental_drawdown_penalty"] == pytest.approx(-0.015)
    assert parts["turnover_penalty"] == pytest.approx(-0.005)
    assert parts["concentration_penalty"] == pytest.approx(-0.01)


def test_positive_return_raises_reward():
    lo, _ = _parts(0.002)
    hi, _ = _parts(0.010)
    assert hi > lo


def test_drawdown_does_not_repeat_when_it_does_not_increase():
    _, parts = _parts(0.0, prev_drawdown=0.1, drawdown=0.1)
    assert parts["incremental_drawdown_penalty"] == 0.0


def test_reward_is_bounded_by_clip():
    total, _ = _parts(1.0, drawdown=10.0, turnover=10.0, hhi_val=10.0)
    assert total == 5.0
    total, _ = _parts(-1.0, drawdown=10.0, turnover=10.0, hhi_val=10.0)
    assert total == -5.0


def test_reward_quality_contract():
    rewards = np.linspace(-1.0, 1.0, 1000)
    report = reward_quality_report(rewards)
    assert report["std_in_range"]
    assert report["abs_q999"] <= 5.0
    assert report["max_abs_share"] <= 0.01
    assert report["passed"]
