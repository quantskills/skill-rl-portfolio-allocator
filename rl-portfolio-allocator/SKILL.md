---
name: rl-portfolio-allocator
description: Use when training or backtesting a PPO factor-weight allocator on CSI300 with embedded costs and risk-penalized reward.
license: GPL-3.0-only
tags: [quant, rl, portfolio, ppo, ashare]
---

# RL 组合权重优化器(研究与训练)

## 因子逻辑
- RL 学的是 **K 维因子权重**(K 由方法产物决定),不是股票权重。
- **100 候选 → 20 选中**:候选库为 10 族 × 10 共 100 个因果 OHLCV 因子(`factor_catalog.py`,版本 `factor-catalog-v2`);每个 walk-forward fold 在训练区间内做训练-only 选择,冻结 20 个因子(含方向)用于该 fold 的验证与测试。
- 对照组 `control_6f` 固定为六因子:动量20/反转5/波动率20/换手率20/流动性20/收益率偏度60;候选组 `candidate_20f` 使用 fold 内选出的 20 因子。
- 详见 `references/factor-selection.md`(缓存布局、选择阈值、目录地图、gates、发布要求)。
- 动作变换:`tanh` → L1 归一化(`Σ|wᵢ|=1`,可正可负)→ EMA 平滑(`α=0.5`)。
- 综合得分 `s = F_t · w̃_t` → Top-N 做多(N=30) + Bottom-M 做空(M=15,名义上限 30%)。
- Reward(默认 variant=low)= 100·净收益 − 0.5·max(0, Δ回撤) − 0.05·max(0, turnover−0.2) − 0.5·max(0, HHI−0.03),clip ±5;variant=constrained 时回撤系数为对偶变量 λ_t(随 episode MDD 相对 TARGET_MDD=0.10 自适应),另加恢复信用 0.1·max(0, −Δ回撤) 与下行半方差项 0.2·min(0, 净收益)²;legacy_dsr 变体保留旧 DSR 公式。
- 成本:佣金 3bps + 印花税 10bps(卖出侧) + 冲击 5bps × turnover + 融券年化 8%(按日折算)。

## 三类训练用途(不可混用)
| 用途 | 脚本 | 训练窗口 |
|---|---|---|
| 压力测试 | `stress_test.py` | 各压力段之前 |
| 主回测 | `backtest.py` | 早期(如 2010–2022),测样本外 2023+ |
| 生产模型 | `allocate.py --retrain` | 数据起点 ~ 最新日,**无样本外尾巴** |

## 环境变量
`PANDA_DATA_USERNAME` / `PANDA_DATA_PASSWORD`(必填)、`PANDA_DATA_START_DATE` / `PANDA_DATA_END_DATE`、`RL_ALGO=ppo`、`REWARD_TYPE=sharpe|sortino`、`RETRAIN_CADENCE=monthly|quarterly`、`TRAIN_DEVICE=auto|cuda|mps|cpu`。

## 运行顺序

标准入口是仓库根目录的 `run_pipeline.sh`(研究流程:原始数据 + 因子缓存 → 数据覆盖 → pytest → fold-local control/candidate walk-forward → 对照 gates → 冻结 stress):

```bash
bash run_pipeline.sh --research-smoke   # 冒烟:验证契约,research_ok 强制 false,不写 approval
RLPA_RUN_ID=<run-id> bash run_pipeline.sh --research-full
                                        # 正式:3 folds × 5 seeds,仅 research_ok=true 写 approval
bash run_pipeline.sh --publish --approval rl-portfolio-allocator/artifacts/walk_forward/<run_id>/approval.json
                                        # 发布:校验选中因子 bundle 后重训生产模型
```

产物:`data/factors/`(族分区缓存)、`artifacts/walk_forward/<run_id>/{control_6f,candidate_20f}/`、`comparison.json`、`gates.json`、`approval.json`,目录地图见 `references/factor-selection.md`。

## 产物字段(持仓表)
`trade_date, symbol, weight, side, factor_weights(JSON), composite_score, strategy_id='RLPA', data_version='real-v1', update_time`。
