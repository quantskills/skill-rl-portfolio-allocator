from scripts.config import get_config
from scripts.reward import compose_reward


def _parts(net_ret, drawdown=0.1, turnover=0.8, hhi_val=0.05):
    cfg = get_config()
    total, parts = compose_reward(
        dsr_delta=0.0018, drawdown=drawdown, turnover=turnover,
        hhi_val=hhi_val, cfg=cfg, net_ret=net_ret,
        long_notional=1.0, short_notional=0.3, long_cap=1.0, short_cap=0.3,
    )
    return total, parts


def test_reward_parts_complete():
    total, parts = _parts(0.005)
    for k in ("ret_term", "dsr", "drawdown_penalty", "turnover_penalty",
              "concentration_penalty", "constraint_penalty", "total"):
        assert k in parts, f"missing {k}"
    s = (parts["ret_term"] + parts["dsr"] + parts["drawdown_penalty"]
         + parts["turnover_penalty"] + parts["concentration_penalty"]
         + parts["constraint_penalty"])
    assert abs(s - parts["total"]) < 1e-12
    assert abs(total - parts["total"]) < 1e-12


def test_return_term_not_dominated_by_penalties():
    # 典型单步: 净收益 +0.5%, 换手 0.8, 回撤 10%
    total, parts = _parts(0.005)
    penalties = abs(parts["drawdown_penalty"] + parts["turnover_penalty"]
                    + parts["concentration_penalty"])
    assert abs(parts["ret_term"]) > 0
    # 修复前 penalty/return ≈ 26x; 要求降到 ≤ 5x
    assert penalties / abs(parts["ret_term"]) <= 5.0, (
        f"penalties {penalties} dominate ret_term {parts['ret_term']}")


def test_positive_return_raises_reward():
    lo, _ = _parts(0.002)
    hi, _ = _parts(0.010)
    assert hi > lo
