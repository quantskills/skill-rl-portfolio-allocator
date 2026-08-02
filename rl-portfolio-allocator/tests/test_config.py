import pytest

from scripts.config import get_config


def test_reward_defaults():
    cfg = get_config()
    assert cfg["reward_scale"] == 100.0
    assert cfg["reward_clip"] == 5.0
    assert cfg["hhi_target"] == 0.03
    assert cfg["turnover_budget"] == 0.20
    assert cfg["lambda_drawdown"] == 0.5
    assert cfg["lambda_concentration"] == 0.5
    assert cfg["lambda_turnover"] == 0.05


@pytest.mark.parametrize("variant", ["none", "low", "medium", "legacy_dsr"])
def test_reward_variant_from_environment(monkeypatch, variant):
    monkeypatch.setenv("REWARD_VARIANT", variant)
    assert get_config()["reward_variant"] == variant


def test_unknown_reward_variant_is_rejected(monkeypatch):
    monkeypatch.setenv("REWARD_VARIANT", "unknown")
    with pytest.raises(ValueError, match="reward variant"):
        get_config()


def test_reward_ret_weight_default():
    assert get_config()["reward_ret_weight"] == 1.0
