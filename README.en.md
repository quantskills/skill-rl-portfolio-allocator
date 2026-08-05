# skill-rl-portfolio-allocator

PPO reinforcement learning dynamic factor-weight allocator for CSI300
constituents with embedded transaction costs and risk-penalized reward,
validated through walk-forward research gates.

This repository is research tooling only. It is not investment advice, does not
promise returns, and is not an official endorsement by QuantSkills, Panda data,
Codex, Claude Code, Cursor, Hermes or OpenClaw.

## Runtime Entries

| Runtime | Entry |
|---|---|
| Codex | `AGENTS.md` |
| Claude Code | `CLAUDE.md` |
| Hermes | `HERMES.md` |
| OpenClaw | `OPENCLAW.md` |

The root `SKILL.md` is the canonical skill entry.

## Structure

```text
SKILL.md
skill.json
AGENTS.md
CLAUDE.md
HERMES.md
OPENCLAW.md
run_pipeline.sh
rl-portfolio-allocator/
  SKILL.md
  pyproject.toml
  conftest.py
  scripts/               # 25+ modules: features, train, backtest, stress_test, walk_forward, etc.
  tests/                  # 31 pytest test files
  references/
    factor-selection.md
    dsr_derivation.md
  checkpoints/
rl-portfolio-allocator-production/
  SKILL.md
  scripts/
    query.py
```

## What It Does

The RL agent learns a K-dimensional factor weight vector (not individual stock
weights) on a weekly rebalance schedule. A 100-factor candidate pool (10
families × 10 causal OHLCV factors, `factor-catalog-v2`) feeds into per-fold
train-only factor selection, freezing 20 factors for validation and testing.

- **Control group** `control_6f`: six fixed factors.
- **Candidate group** `candidate_20f`: fold-selected 20 factors.
- **Action transform**: `tanh` → L1 normalization (`Σ|wᵢ|=1`) → EMA smoothing
  (`α=0.5`).
- **Portfolio**: composite score → Top-30 long + Bottom-15 short (max 30%
  notional short).
- **Reward** (default `variant=low`): `100·net_return − 0.5·max(0, Δdrawdown) −
  0.05·max(0, turnover−0.2) − 0.5·max(0, HHI−0.03)`, clipped ±5. Multiple
  variants (`none`, `low`, `gentle`, `constrained`, `legacy_dsr`) are supported.
- **Costs**: commission 3bps + stamp duty 10bps (sell side) + impact 5bps ×
  turnover + short borrow 8% annualized.
- **Suspension handling**: `is_suspended` flag + `freeze_suspended` logic.

## Data Source

Panda data SDK. No local files are read as authoritative input.

| Variable | Required | Notes |
|---|---:|---|
| `PANDA_DATA_USERNAME` | yes | Panda data account name |
| `PANDA_DATA_PASSWORD` | yes | Panda data password |
| `PANDA_DATA_START_DATE` | no | default `2004-01-01` |
| `PANDA_DATA_END_DATE` | no | default `2024-12-31` |
| `TRAIN_DEVICE` | no | `auto` / `cuda` / `mps` / `cpu` |
| `RLPA_REWARD_CANDIDATES` | no | reward variant list for validation |
| `RLPA_SELECTION_TARGET_COUNT` | no | factor selection target count |
| `RLPA_LAMBDA_DRAWDOWN` | no | drawdown penalty coefficient |

## Commands

The standard entry point is the root `run_pipeline.sh`:

```bash
# Smoke test: wiring validation only, research_ok forced false
bash run_pipeline.sh --research-smoke

# Full research: 3 folds × 5 seeds, writes approval only if gates pass
RLPA_RUN_ID=<run-id> bash run_pipeline.sh --research-full

# Publish: validate selected factor bundle, retrain production model
bash run_pipeline.sh --publish --approval \
  rl-portfolio-allocator/artifacts/walk_forward/<run_id>/approval.json
```

## Research Gates

| Gate | Threshold | Description |
|---|---|---|
| `median_oos_sharpe_gain` | ≥ 0.1 | Median out-of-sample Sharpe gain over baseline |
| `positive_excess_folds` | ≥ 2 | Number of folds with positive excess |
| `candidate_cost_2x_oos_sharpe` | ≥ 0 | OOS Sharpe still positive after 2× costs |
| `candidate_annualized_turnover` | ≤ 12 | Annualized turnover cap |
| `candidate_stress_mdd_excess` | ≥ 0.05 | Stress test MDD excess |
| `candidate_stress_calmar_excess` | ≥ 0 | Stress test Calmar excess |
| `candidate_stress_long_exposure_util` | ≥ 0.5 | Long exposure utilization |
| `complete_paired_evidence` | true | Control and candidate both fully executed |

All gates must pass (`research_ok=true`) before publishing. If gates fail,
publication is blocked — this is by design.

## Stress Test Scenarios

| Scenario | Description |
|---|---|
| 2015 A-share crash | 2015 A-share market crash |
| 2020 COVID | COVID-19 market shock |
| 2022 double kill | Simultaneous equity and bond drawdown |
| 2008 GFC | Global financial crisis (skipped if data starts after 2007) |

## Output Fields (Allocation Table)

`trade_date`, `symbol`, `weight`, `side`, `factor_weights` (JSON),
`composite_score`, `strategy_id` (`RLPA`), `data_version` (`real-v1`),
`update_time`.

## Acceptance Criteria

- `--research-smoke`: all 31 pytest tests pass, `research_ok` forced false
- `--research-full`: 3-fold walk-forward completes, `comparison.json` and
  `gates.json` generated
- All 8 research gates pass
- All 4 stress test scenarios completed (explicit SKIP when data insufficient)
- Publication blocked when `research_ok=false`

## Production Queries

Trading agents use the `rl-portfolio-allocator-production` skill to read
allocation data, filtering by `strategy_id == "RLPA"`. No retraining or
backtesting is performed at query time.

Production query results are research data only and must not be presented as
trading recommendations, return predictions, or deterministic signals.

## Dependencies

- Python 3.10+
- panda-data
- stable-baselines3 ≥ 2.0
- gymnasium
- pandas
- numpy
- pyarrow
- pytest

## License

GPL-3.0-only. See `LICENSE`.
