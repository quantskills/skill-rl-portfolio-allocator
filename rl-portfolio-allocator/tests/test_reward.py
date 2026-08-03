import numpy as np
import pytest

from scripts.config import get_config
from scripts.diagnostics import reward_quality_report
from scripts.reward import (
    DualState,
    compose_constrained_reward,
    compose_reward,
    reward_coefficients,
)


def _parts(net_ret, prev_drawdown=0.0, drawdown=0.1, turnover=0.8, hhi_val=0.05,
           variant="medium"):
    cfg = get_config()
    cfg["reward_variant"] = variant
    return compose_reward(net_ret, prev_drawdown, drawdown, turnover, hhi_val, cfg)


def test_reward_coefficients_are_explicit():
    assert reward_coefficients("none") == (0.0, 0.0, 0.0)
    assert reward_coefficients("gentle") == (0.10, 0.20, 0.05)
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


def test_dual_lambda_rises_while_episode_mdd_exceeds_target():
    dual = DualState()
    lam1 = dual.update(0.15, target_mdd=0.10, lr_dual=0.05)
    assert lam1 == pytest.approx(0.0025)
    # episode_mdd 是粘性最大值:本步回撤回落仍按 0.15 更新
    lam2 = dual.update(0.12, target_mdd=0.10, lr_dual=0.05)
    assert lam2 == pytest.approx(0.005)
    assert dual.episode_mdd == pytest.approx(0.15)


def test_dual_lambda_decays_to_zero_below_target():
    dual = DualState(lam=0.4, episode_mdd=0.05)
    lam = dual.update(0.03, target_mdd=0.10, lr_dual=0.05)
    assert lam == pytest.approx(0.3975)  # 0.4 + 0.05 * (0.05 - 0.10)
    for _ in range(200):
        lam = dual.update(0.0, target_mdd=0.10, lr_dual=0.05)
    assert lam == 0.0


def _constrained_parts(net_ret, prev_drawdown=0.0, drawdown=0.1, turnover=0.8,
                       hhi_val=0.05, lam=0.5):
    cfg = get_config()
    cfg["reward_variant"] = "constrained"
    return compose_constrained_reward(
        net_ret, prev_drawdown, drawdown, turnover, hhi_val, lam, cfg,
    )


def test_constrained_recovery_credit_rewards_drawdown_repair():
    _, parts = _constrained_parts(0.0, prev_drawdown=0.10, drawdown=0.04, lam=0.5)
    assert parts["recovery_credit"] == pytest.approx(0.1 * 0.06)
    assert parts["incremental_drawdown_penalty"] == 0.0


def test_constrained_no_recovery_credit_when_drawdown_deepens():
    _, parts = _constrained_parts(0.0, prev_drawdown=0.02, drawdown=0.08, lam=0.5)
    assert parts["recovery_credit"] == 0.0
    assert parts["incremental_drawdown_penalty"] == pytest.approx(-0.5 * 0.06)


def test_constrained_downside_semivariance_penalizes_only_losses():
    _, loss_parts = _constrained_parts(-0.02)
    assert loss_parts["downside_vol_penalty"] == pytest.approx(-0.2 * 0.0004)
    _, gain_parts = _constrained_parts(0.02)
    assert gain_parts["downside_vol_penalty"] == 0.0


def test_constrained_reward_parts_contract_and_clip():
    total, parts = _constrained_parts(10.0)  # 巨大收益触发 clip
    assert total == 5.0
    assert set(parts) == {
        "scaled_net_return", "incremental_drawdown_penalty", "recovery_credit",
        "downside_vol_penalty", "turnover_penalty", "concentration_penalty",
        "dual_lambda", "total",
    }
    assert parts["dual_lambda"] == pytest.approx(0.5)
    assert "dsr" not in parts
