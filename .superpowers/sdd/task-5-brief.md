# Task 5: train.py main() - Add --timesteps Flag; Pipeline 200k

**Objective:** Add `--timesteps` CLI argument to train.py (default 5000 for smoke) and update run_pipeline.sh to call training with 200k steps for production.

## Scope

- **Files to modify:**
  - `rl-portfolio-allocator/scripts/train.py` (main() function, lines ~61-74)
  - `run_pipeline.sh` (run_train function, around line 59-63)
- **Test location:** Manual verification (no pytest needed)

## Exact Changes Required

### Update train.py main()

Replace the current `main()` function with:

```python
def main() -> None:
    import argparse
    from scripts.config import get_config
    from scripts.env import make_env
    cfg = get_config()
    root = pathlib.Path(__file__).resolve().parent.parent
    features_path = root / "data" / "features.parquet"
    index_path = root / "data" / "index_returns.parquet"
    ckpt = root / "checkpoints" / "smoke.zip"

    ap = argparse.ArgumentParser()
    ap.add_argument("--timesteps", type=int, default=5000,
                    help="训练步数;默认 5000 为快速自检(smoke)")
    args = ap.parse_args()

    env = make_env(str(features_path), str(index_path), cfg,
                   cfg["start_date"], cfg["end_date"] or "2099-12-31")
    device = select_device(cfg["train_device"])
    print(f"train device: {device}")
    model = train_ppo(env, total_timesteps=args.timesteps, seed=0,
                      device=device, save_path=str(ckpt))
    print(f"checkpoint saved: {ckpt}  (timesteps={args.timesteps})")
```

**Changes:**
1. Add `import argparse`
2. Create ArgumentParser with `--timesteps` (default 5000)
3. Use `args.timesteps` instead of hardcoded 5000
4. Update print to show timesteps used

### Update run_pipeline.sh

In the `run_train()` function (approximately line 59-63), change the training command from:

```bash
    PYTHONPATH="$WORK_DIR:$PYTHONPATH" python -m scripts.train
```

To:

```bash
    PYTHONPATH="$WORK_DIR:$PYTHONPATH" python -m scripts.train --timesteps 200000
```

**Only change:** Add `--timesteps 200000` to the command line.

## Verification Steps

### Step 1: Verify argparse works

Run: `cd rl-portfolio-allocator && python -m scripts.train --help 2>&1 | grep -A1 timesteps`

Expected output should show:
```
--timesteps TIMESTEPS   训练步数;默认 5000 为快速自检(smoke)
```

### Step 2: Syntax check run_pipeline.sh

Run: `bash -n run_pipeline.sh && echo OK`

Expected: `OK` (no syntax errors)

## Test Coverage

No pytest needed. Manual verification confirms:
- `python -m scripts.train --help` shows `--timesteps`
- `python -m scripts.train --timesteps 100` works without error
- `python -m scripts.train` (no args) defaults to 5000
- `bash -n run_pipeline.sh` validates script syntax

## Interface Contract

**Consumes (from Task 4):**
- Updated `train_ppo()` signature (unchanged in Task 5)

**Produces:**
- CLI entry point `python -m scripts.train --timesteps N` where N overrides default 5000
- Pipeline calls train.py with --timesteps 200000 for production training

**Constraints:**
- Default 5000 must be preserved (smoke tests still work)
- --timesteps must be optional (--help should work)
- No changes to train_ppo(), env creation, or model checkpointing
- run_pipeline.sh must remain syntactically valid bash

