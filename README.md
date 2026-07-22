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
