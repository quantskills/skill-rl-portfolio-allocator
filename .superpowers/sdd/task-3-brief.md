# Task 3: Env - Wire net_ret into Reward Composition

**Objective:** Thread `net_ret` (already calculated in env.step) through to `compose_reward` call so the new return term is passed the correct net profit/loss.

## Scope

- **Files to modify:** `rl-portfolio-allocator/scripts/env.py` (env.step method, around line 137-141)
- **Test location:** Create `rl-portfolio-allocator/tests/test_env_reward_wiring.py` (new file)

## Context

In `env.py:step()`, the variable `net` is already computed at line 122 as:
```python
net = gross - costs["total"]  # net profit/loss for this step
```

This is passed to `self.dsr.update(net, ...)` at line 135, but currently NOT passed to `compose_reward()`.

## Exact Changes Required

### Update compose_reward Call

At `scripts/env.py:137-141`, change from:

```python
        reward, parts = compose_reward(
            dsr_delta, drawdown, turnover, hhi_v, self.cfg,
            long_notional=long_notional, short_notional=short_notional,
            long_cap=self.cfg["long_notional"], short_cap=self.cfg["short_notional_cap"]
        )
```

To:

```python
        reward, parts = compose_reward(
            dsr_delta, drawdown, turnover, hhi_v, self.cfg,
            net_ret=net,
            long_notional=long_notional, short_notional=short_notional,
            long_cap=self.cfg["long_notional"], short_cap=self.cfg["short_notional_cap"]
        )
```

**Only change:** Add `net_ret=net` as a keyword argument after the positional `cfg` argument.

## Test Coverage

Write `tests/test_env_reward_wiring.py` with:

```python
import numpy as np
import pandas as pd
from scripts.config import get_config, FACTOR_NAMES, K
from scripts.env import PortfolioEnv


def _toy_features():
    dates = pd.date_range("2020-01-01", periods=6, freq="D")
    rows = []
    for d in dates:
        for i in range(40):
            row = {"trade_date": d, "symbol": f"S{i:03d}",
                   "ret_1d": 0.01 if i % 2 else -0.01, "is_suspended": False}
            for fn in FACTOR_NAMES:
                row[fn] = float((i % 5) - 2)
            rows.append(row)
    return pd.DataFrame(rows)


def test_step_info_has_ret_term():
    cfg = get_config()
    feats = _toy_features()
    idx = pd.Series(np.zeros(1), index=[feats["trade_date"].min()])
    env = PortfolioEnv(feats, idx, cfg, feats["trade_date"].min(), feats["trade_date"].max())
    env.reset(seed=0)
    _, reward, _, _, info = env.step(np.zeros(K, dtype=np.float32))
    assert "ret_term" in info["reward_parts"]
    # ret_term should equal reward_ret_weight * net_ret
    assert abs(info["reward_parts"]["ret_term"]
               - cfg["reward_ret_weight"] * info["net_ret"]) < 1e-12
```

The test must PASS after implementation.

## Interface Contract

**Consumes (from Task 2):**
- New `compose_reward()` signature that accepts `net_ret` as a required keyword argument

**Produces:**
- env.step() passes `net` to `compose_reward()` as `net_ret=net`
- info["reward_parts"] dict now contains "ret_term" key
- Wiring confirmed: ret_term in reward breakdown matches calculation

**Constraints:**
- No changes to cost calculations, net computation, or state tracking
- env.step() return signature unchanged
- Hard constraints on notional still apply
- DSR state update unchanged

