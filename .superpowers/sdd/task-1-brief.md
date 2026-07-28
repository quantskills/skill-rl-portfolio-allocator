# Task 1: Config - Add reward_ret_weight + Rescale Lambdas

**Objective:** Add `reward_ret_weight` config parameter (default 1.0) and rescale reward penalty lambdas to encourage net-return optimization instead of risk-aversion bias.

## Scope

- **Files to modify:** `rl-portfolio-allocator/scripts/config.py` (lines 29-31 for constants, 52-54 for get_config return dict)
- **Test location:** Create `rl-portfolio-allocator/tests/test_config.py` (new file)

## Exact Changes Required

### Lambda Rescaling

Update these constants in `scripts/config.py:29-31`:

```python
LAMBDA_DRAWDOWN: float = 0.005      # was 0.05 (10x reduction)
LAMBDA_TURNOVER: float = 0.002      # was 0.02 (10x reduction)
LAMBDA_CONCENTRATION: float = 0.02  # unchanged
```

### Add reward_ret_weight to get_config()

In `get_config()` return dict (around line 52-54 where other lambda keys live), add:

```python
"reward_ret_weight": float(os.environ.get("REWARD_RET_WEIGHT", "1.0")),
```

Default must be `1.0` (from environment variable or hardcoded).

## Test Coverage

Write `tests/test_config.py` with:

```python
from scripts.config import get_config

def test_lambdas_rescaled():
    cfg = get_config()
    assert cfg["lambda_drawdown"] == 0.005
    assert cfg["lambda_turnover"] == 0.002
    assert cfg["lambda_concentration"] == 0.02

def test_reward_ret_weight_default():
    cfg = get_config()
    assert cfg["reward_ret_weight"] == 1.0
```

Both tests must PASS after implementation.

## Interface Contract

**Produces:** `get_config()` dict with:
- `"lambda_drawdown": 0.005` (rescaled)
- `"lambda_turnover": 0.002` (rescaled)
- `"reward_ret_weight": 1.0` (new, default, readable from env)

**Constraints:**
- No other config values changed
- Environment variable `REWARD_RET_WEIGHT` overrides default if set
- Hard constraint: `lambda_concentration` stays 0.02

