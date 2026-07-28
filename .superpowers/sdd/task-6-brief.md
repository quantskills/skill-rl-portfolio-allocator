# Task 6: Full Regression + 20k Short Training Validation

**Objective:** Run full unit test suite to ensure all Tasks 1-5 changes integrate correctly, then do a 20k-step training run to verify the reward rescaling improves return-signal dominance (penalties no longer 26x larger than returns).

## Scope

- **No new files to create** — uses existing tests + one diagnostic command
- **Test location:** All tests in `rl-portfolio-allocator/tests/`

## Exact Steps Required

### Step 1: Full Unit Test Suite

Run: `cd rl-portfolio-allocator && python -m pytest -q`

Expected: All tests PASS (including those from Tasks 1-5).

### Step 2: 20k Short Training + Reward Diagnostics

Run the following diagnostic script (in `rl-portfolio-allocator/` directory):

```bash
python -c "
import pandas as pd, numpy as np
from scripts.config import get_config, FACTOR_NAMES
from scripts.env import PortfolioEnv
from scripts.train import train_ppo
from scripts.backtest import run_ppo_rollout
from scripts.diagnostics import summarize_rollout, check_degeneracy
cfg=get_config()
feats=pd.read_parquet('data/features.parquet')
dates=sorted(pd.to_datetime(feats['trade_date']).unique())
split=dates[int(len(dates)*0.7)]
idx=pd.Series(np.zeros(1),index=pd.to_datetime([feats['trade_date'].min()]))
tr=PortfolioEnv(feats,idx,cfg,dates[0],split-pd.Timedelta(days=1))
model=train_ppo(tr,total_timesteps=20000,seed=0,device='cpu')
te=PortfolioEnv(feats,idx,cfg,split,dates[-1])
rets,infos,_=run_ppo_rollout(model,te)
s=summarize_rollout(infos)
rb=s['reward_breakdown']
print('reward_breakdown:',{k:round(v,6) for k,v in rb.items()})
print('mean daily net ret', round(float(np.mean(rets)),6))
print('degeneracy:', check_degeneracy(s,cfg))
" 2>&1 | grep -vE "Warning|warn|deprecat"
```

Expected output:
- `reward_breakdown:` dict with `ret_term` (if `summarize_rollout` was updated) or at minimum penalty values significantly smaller than pre-fix (~0.003 vs ~0.03)
- `mean daily net ret` should NOT be a large negative value (≥ −0.0005 is acceptable; positive is ideal)
- `degeneracy:` should be False or an acceptable status

### Step 3: Allocations Still Valid

Run: `cd rl-portfolio-allocator && python -m scripts.validate`

Expected: Output contains `[OK]` marker indicating allocations still meet constraints.

### Step 4: Commit (conditional)

If any diagnostic files or scripts needed tweaks in Step 2, commit those:

```bash
git add -A && git commit -m "test: full regression after reward rescale" --allow-empty

# (with Co-Authored-By if changes were made)
```

If no changes, --allow-empty allows a "no-op" commit to mark the stage.

## Interface Contract

**Consumes (all Tasks 1-5):**
- Config with rescaled lambdas and reward_ret_weight
- compose_reward with net_ret term
- env.step() wiring net_ret
- train_ppo with optional EvalCallback
- train.py main() with --timesteps

**Produces:**
- All unit tests PASS
- 20k training completes without error
- Reward breakdown shows return term non-dominated by penalties
- Allocations remain valid

**Constraints:**
- No changes to selection logic, cost model, or action space
- Penalties must drop 10x from pre-fix (0.05→0.005, 0.02→0.002)
- Hard constraints (notional caps) still enforced
- Mean daily return should improve (or at worst stay similar)

