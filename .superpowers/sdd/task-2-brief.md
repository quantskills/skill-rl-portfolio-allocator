# Task 2: Reward - Add net_ret Return Term to compose_reward

**Objective:** Inject a direct net-return reward term into `compose_reward` so the agent directly optimizes returns, not just penalties.

## Scope

- **Files to modify:** `rl-portfolio-allocator/scripts/reward.py` (function signature and body, lines ~40-63)
- **Test location:** Create `rl-portfolio-allocator/tests/test_reward.py` (new file)

## Exact Changes Required

### New compose_reward Signature

Replace the current function signature with:

```python
def compose_reward(
    dsr_delta: float, drawdown: float, turnover: float, hhi_val: float, cfg: dict,
    net_ret: float,
    long_notional: float = 0.0, short_notional: float = 0.0,
    long_cap: float = 1.0, short_cap: float = 0.3,
) -> tuple[float, dict]:
```

**Key change:** `net_ret` added as a new **required positional parameter** after `cfg` (before optional keyword args).

### New Function Body

```python
def compose_reward(...) -> tuple[float, dict]:
    ret_term = cfg["reward_ret_weight"] * net_ret
    dd_pen = -cfg["lambda_drawdown"] * max(0.0, drawdown)
    to_pen = -cfg["lambda_turnover"] * turnover
    conc_pen = -cfg["lambda_concentration"] * hhi_val

    constraint_pen = 0.0
    if long_notional > long_cap * 1.01:
        constraint_pen -= 1.0 * (long_notional - long_cap)
    if short_notional > short_cap * 1.01:
        constraint_pen -= 1.0 * (short_notional - short_cap)

    total = ret_term + dsr_delta + dd_pen + to_pen + conc_pen + constraint_pen
    return total, {
        "ret_term": ret_term,
        "dsr": dsr_delta,
        "drawdown_penalty": dd_pen,
        "turnover_penalty": to_pen,
        "concentration_penalty": conc_pen,
        "constraint_penalty": constraint_pen,
        "total": total,
    }
```

**Changes from current:**
1. Add `ret_term = cfg["reward_ret_weight"] * net_ret` at top
2. Add `ret_term` to parts dict (new key)
3. Change `total` calculation to include `ret_term` first
4. DSR delta stays second in summation

## Test Coverage

Write `tests/test_reward.py` with:

```python
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
```

All 3 tests must PASS after implementation.

## Interface Contract

**Consumes (from Task 1):**
- `cfg["reward_ret_weight"]` (default 1.0)
- `cfg["lambda_drawdown"]` (now 0.005)
- `cfg["lambda_turnover"]` (now 0.002)
- `cfg["lambda_concentration"]` (still 0.02)

**Produces:**
- New parts dict key: `"ret_term"`
- Reward total now includes return term prominently
- Penalties remain negative values
- Hard constraints still apply unchanged

**Constraints:**
- `dsr_delta` is added after `ret_term` (not replaced)
- No changes to DSRState, hhi(), or constraint_penalty logic
- Return summation order: ret_term + dsr_delta + penalties + constraint_pen

