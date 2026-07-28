# RL 奖励重构 + 训练量修复 — 设计文档

**Skill 名**: `skill-rl-portfolio-allocator`
**日期**: 2026-07-28
**类型**: bugfix / 策略修复（承接主设计 `2026-07-22-rl-portfolio-allocator-design.md`）

---

## 1. 背景与根因

跑通 `run_pipeline.sh --all` 后，RL 策略回测 ARR=−8.94% / Sharpe=−0.39，跑输唯一正收益的等权基准（+6.44%），压力测试全线巨亏。系统化排查（含数据取证）确认根因**不是模型能力不足，而是两点**：

1. **奖励被惩罚项碾压（核心）**：`reward.py` 里唯一奖励收益的是差分 Sharpe 增量 `dsr_delta`，天生是无量纲小量（训练实测均值 ≈ +0.0018）；而 `λ_drawdown=0.05` 与 `λ_turnover=0.02` 产生的惩罚合计 ≈ −0.047。**penalty/return ≈ 26×**，PPO 梯度 96% 来自"少回撤少换手"，agent 学到的最优解是躺平避险，收益自然为负。
2. **训练量不足**：`run_pipeline.sh --all` 的训练步骤调用 `train.py main()`，其中硬编码 `total_timesteps=5000`（smoke），生产模型实际只训了 5000 步。

因子信号本身有效（`reversal_5` 次日 Rank IC=+0.041，纯多空 gross 年化 +23.5%），故本轮不动因子。

## 2. 目标

让 PPO 优化**收益**而非避险，并给足训练步数。penalty/return 从 26× 降到 O(1) 量级。

## 3. 改动清单

### 3.1 奖励重构

**`scripts/config.py`**
- 新增 `REWARD_RET_WEIGHT`（env 变量 `REWARD_RET_WEIGHT`，默认 `1.0`），进入 `get_config()` 返回 `reward_ret_weight`。
- `LAMBDA_DRAWDOWN`: 0.05 → **0.005**
- `LAMBDA_TURNOVER`: 0.02 → **0.002**
- `LAMBDA_CONCENTRATION` 保持 0.02（本就极小，非主因）。

**`scripts/reward.py` — `compose_reward`**
- 新增入参 `net_ret: float`。
- 新增主收益项 `ret_term = cfg["reward_ret_weight"] * net_ret`。
- `total = ret_term + dsr_delta + dd_pen + to_pen + conc_pen + constraint_pen`。
- `parts` 字典新增 `"ret_term"` 字段；其余字段保留（向后兼容 diagnostics）。
- `dsr_delta` 保留（作为风险调整平滑项），`constraint_penalty` 硬约束逻辑不动。

**`scripts/env.py` — `step`**
- 调用 `compose_reward(...)` 时传入 `net_ret=net`。

### 3.2 训练量 + 早停

**`scripts/train.py` — `train_ppo`**
- 新增可选参数 `eval_env=None`, `eval_freq=10_000`, `n_eval_episodes=1`, `patience=None`。
- 当 `eval_env` 提供时，构造 SB3 `EvalCallback`（`StopTrainingOnNoModelImprovement` 作为 patience 早停），传入 `model.learn(callback=...)`。
- 不传 `eval_env` 时行为与现状完全一致（向后兼容）。
- `main()` 的 5000-step smoke **保留**，仅作快速自检。

**`scripts/train.py` — `main()`**
- 新增 `--timesteps`（默认 `5000` 保持 smoke 语义），使 pipeline 可覆盖为 200k。

**`run_pipeline.sh` — `run_train`（`--all`、`--quick` 路径）**
- `run_train` 改为 `python -m scripts.train --timesteps 200000`（不再走 5000-step smoke 默认值）。
- 由于 backtest 步骤已独立用默认 200k 训练+回测，`--all` 端到端至少两处走 ≥200k 训练。

### 3.3 信号

本轮不改。

## 4. 验证（TDD）

**`tests/test_reward.py`（新增）**
- `test_return_term_not_dominated_by_penalties`：给定典型单步（net_ret=+0.005, turnover=0.8, drawdown=0.1, hhi=0.05，用新 lambda），断言 `|penalty 总和| / |ret_term| ≤ 5`（不再是 26×）。
- `test_reward_parts_complete`：断言 `parts` 含 `ret_term/dsr/drawdown_penalty/turnover_penalty/concentration_penalty/constraint_penalty/total`，且 `total` 等于各项之和。
- `test_positive_return_raises_reward`：net_ret 越大 total 越大（收益项方向正确）。

**`tests/test_train.py`（新增，轻量）**
- `test_train_ppo_backward_compatible`：不传 eval_env 时 `train_ppo` 正常返回 model（用极小 timesteps + 小环境）。

**回归验证（手动跑一次）**
- 短训练 20k，打印 `reward_breakdown`：确认 `ret_term` 与 penalty 同量级、mean daily net ret 不再持续为负；`check_degeneracy` 无致命告警。

**已有守护**
- `tests/test_save_allocations.py`（本会话已加）继续通过。
- `python -m scripts.validate` 仍 `[OK]`。

## 5. 非目标

- 不改因子集/因子窗口。
- 不改选股逻辑（Top-N/Bottom-M）、成本模型、硬约束缩放。
- 不做超参数网格搜索——仅把奖励尺度与训练量调到合理区间。
