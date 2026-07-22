---
name: rl-portfolio-allocator
description: Use when training or backtesting a PPO factor-weight allocator on CSI300 with embedded costs and DSR reward.
license: GPL-3.0-only
tags: [quant, rl, portfolio, ppo, ashare]
---

# RL 组合权重优化器(研究与训练)

## 因子逻辑
- RL 学的是 **K 维因子权重**(默认 K=6:动量20/反转5/波动率20/换手率20/流动性20/收益率偏度60),不是股票权重。
- 动作变换:`tanh` → L1 归一化(`Σ|wᵢ|=1`,可正可负)→ EMA 平滑(`α=0.5`)。
- 综合得分 `s = F_t · w̃_t` → Top-N 做多(N=30) + Bottom-M 做空(M=15,名义上限 30%)。
- Reward = 差分 Sharpe DSR(扣全成本净收益)− `λ_dd·max(0,drawdown)` − `λ_to·turnover` − `λ_conc·HHI`。
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
```bash
python scripts/features.py
python scripts/train.py
python scripts/backtest.py
python scripts/stress_test.py
python scripts/allocate.py --retrain     # or --infer-only
python scripts/validate.py
```

## 产物字段(持仓表)
`trade_date, symbol, weight, side, factor_weights(JSON), composite_score, strategy_id='RLPA', data_version='real-v1', update_time`。
