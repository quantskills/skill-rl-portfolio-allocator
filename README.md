# skill-rl-portfolio-allocator

PPO 学 K 维因子权重 → CSI300 多空组合。研究系统使用严格因果的 market state、bounded net-return reward、周频 action、净成本和多 fold/seed walk-forward gate。研究 gate 未通过时不允许发布生产产物。

- 研究/训练:见 `rl-portfolio-allocator/SKILL.md`
- 只读生产查询:见 `rl-portfolio-allocator-production/SKILL.md`
- 设计文档:`docs/superpowers/specs/2026-07-22-rl-portfolio-allocator-design.md`

## 快速开始

```bash
export PANDA_DATA_USERNAME=<your>
export PANDA_DATA_PASSWORD=<your>
./run_pipeline.sh --research-smoke
# full OOS research (可能较慢)
./run_pipeline.sh --research-full
# 只有 full gate 通过后，显式 approval 才能发布
./run_pipeline.sh --publish --approval rl-portfolio-allocator/artifacts/walk_forward/<run-id>/approval.json
```

## 研究与发布边界

- smoke 只验证 wiring，`publishable=false`，不会生成生产 approval。
- full 研究只在验证集选 candidate；测试集每个 fold/seed 只运行一次。
- `research_ok=false` 是合法研究结果，不能通过调阈值或手工 approval 绕过。
- 生产 retrain 没有 OOS 尾巴，且必须由 `approval.json` 解锁；发布后仍需 `validate`。

## 已实现能力

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
