from scripts.config import get_config


def test_lambdas_rescaled():
    cfg = get_config()
    assert cfg["lambda_drawdown"] == 0.005
    assert cfg["lambda_turnover"] == 0.002
    assert cfg["lambda_concentration"] == 0.02


def test_reward_ret_weight_default():
    cfg = get_config()
    assert cfg["reward_ret_weight"] == 1.0
