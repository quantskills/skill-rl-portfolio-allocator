# skill-rl-portfolio-allocator

PPO 学 K 维因子权重 → CSI300 多空组合。Reward = 差分 Sharpe(DSR)+ 回撤/换手/集中度惩罚;成本(佣金/印花税/融券/冲击)内嵌 env。

- 研究/训练:见 `rl-portfolio-allocator/SKILL.md`
- 只读生产查询:见 `rl-portfolio-allocator-production/SKILL.md`
- 设计文档:`docs/superpowers/specs/2026-07-22-rl-portfolio-allocator-design.md`

## 快速开始

```bash
export PANDA_DATA_USERNAME=<your>
export PANDA_DATA_PASSWORD=<your>
cd rl-portfolio-allocator
python scripts/features.py
python scripts/train.py
python scripts/backtest.py
python scripts/stress_test.py
python scripts/allocate.py --retrain
python scripts/validate.py
```

## 已实现能力(与设计 §8 验收标准对照)

| 验收项 | 由哪个 Task/脚本承担 |
|---|---|
| 无未来函数 | T3 feature + T9 env(`ret_source=t_plus_1`) + T15 stress + T17 validate |
| DSR + 惩罚项 reward,禁 cumret | T6 reward + T9 env |
| K 维因子权重动作(tanh+L1+EMA) | T5 action_transform + T9 env |
| 训练诊断(退化检测) | T11 diagnostics + T14 backtest |
| 成本内嵌 env | T4 costs + T9 env |
| 四段前向压力测试 + 数据不足跳过 | T15 stress_test |
| 三基线对比 | T13 baselines + T14 backtest |
| panda_data 数据源、不提交凭据 | T3 features + .gitignore |
| 停牌处理 | T3 (`is_suspended` 列) + T8 `freeze_suspended` + T9 env |
| 三类训练不可混用 | T14 backtest / T15 stress / T16 allocate 独立入口 |
| 生产模型无样本外尾巴 + 每日推理 | T16 allocate `--retrain` / `--infer-only` |
| 双模式(只读生产查询) | T18 rl-portfolio-allocator-production |

## 故障排查

| 症状 | 可能原因 | 对策 |
|---|---|---|
| `ModuleNotFoundError: stable_baselines3` | 未安装 | `pip install "stable-baselines3>=2.0" gymnasium` |
| `PANDA_DATA_USERNAME not set` | 环境变量未导出 | `export PANDA_DATA_USERNAME=... PANDA_DATA_PASSWORD=...` |
| `no production checkpoint at .../production.zip` | 只调 `--infer-only` 未先训练 | `python scripts/allocate.py --retrain` |
| `[SKIP] 2008_crisis` | panda_data 起点晚于 2007 | 正常,设计允许跳过,不伪造 |
| Sharpe 为 0 / agent 躺平 | 惩罚系数过大 | 参照 T11 warnings,调小 `LAMBDA_*` |
| `train device: cpu` 而不是 gpu | 未检测到 CUDA/MPS | 设置 `TRAIN_DEVICE=cuda` 强制或安装对应 torch |
