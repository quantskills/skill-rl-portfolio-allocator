import pytest

from scripts.config import get_config


def test_reward_defaults():
    cfg = get_config()
    assert cfg["reward_variant"] == "low"
    assert cfg["reward_scale"] == 100.0
    assert cfg["reward_clip"] == 5.0
    assert cfg["hhi_target"] == 0.03
    assert cfg["turnover_budget"] == 0.20
    assert cfg["lambda_drawdown"] == 0.5
    assert cfg["lambda_concentration"] == 0.5
    assert cfg["lambda_turnover"] == 0.05


@pytest.mark.parametrize("variant", ["none", "gentle", "low", "medium", "legacy_dsr", "constrained"])
def test_reward_variant_from_environment(monkeypatch, variant):
    monkeypatch.setenv("REWARD_VARIANT", variant)
    assert get_config()["reward_variant"] == variant


def test_constrained_variant_accepted(monkeypatch):
    monkeypatch.setenv("REWARD_VARIANT", "constrained")
    assert get_config()["reward_variant"] == "constrained"


def test_unknown_reward_variant_is_rejected(monkeypatch):
    monkeypatch.setenv("REWARD_VARIANT", "unknown")
    with pytest.raises(ValueError, match="reward variant"):
        get_config()


def test_reward_ret_weight_default():
    assert get_config()["reward_ret_weight"] == 1.0


def test_episode_randomization_defaults():
    cfg = get_config()
    assert cfg["episode_min_weeks"] == 52
    assert cfg["episode_max_weeks"] == 156
    assert cfg["crisis_oversample_weight"] == 3.0
    assert cfg["ent_coef"] == 0.02


def test_episode_randomization_env_overrides(monkeypatch):
    monkeypatch.setenv("RLPA_EPISODE_MIN_WEEKS", "26")
    monkeypatch.setenv("RLPA_EPISODE_MAX_WEEKS", "104")
    monkeypatch.setenv("RLPA_CRISIS_OVERSAMPLE_WEIGHT", "5.0")
    monkeypatch.setenv("RLPA_ENT_COEF", "0.01")
    cfg = get_config()
    assert cfg["episode_min_weeks"] == 26
    assert cfg["episode_max_weeks"] == 104
    assert cfg["crisis_oversample_weight"] == 5.0
    assert cfg["ent_coef"] == 0.01
