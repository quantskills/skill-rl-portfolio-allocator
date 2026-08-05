# skill-rl-portfolio-allocator

PPO 强化学习动态因子权重分配器，面向 CSI300 成分股，内嵌交易成本与风险惩罚 reward，通过 walk-forward 研究关口验证。

本仓库仅提供研究与验证工具，不构成投资建议，不承诺收益，也不代表 QuantSkills、Panda data、Codex、Claude Code、Cursor、Hermes 或 OpenClaw 的官方背书。

## 社区合规状态

| 项目 | 状态 |
|------|------|
| 仓库类型 | Skill |
| 仓库名 | `skill-rl-portfolio-allocator` |
| 根入口 | `SKILL.md` |
| 中文说明 | `README.md` |
| 英文说明 | `README.en.md` |
| 许可证 | `GPL-3.0-only`，见 `LICENSE` 与 `skill.json` |
| Codex 入口 | `AGENTS.md` |
| Claude Code 入口 | `CLAUDE.md` |
| Hermes 入口 | `HERMES.md` |
| OpenClaw 入口 | `OPENCLAW.md` |

## 目录结构

```
SKILL.md                                            根 skill 入口
skill.json                                          QuantSkills 元数据
AGENTS.md                                           Codex 入口
CLAUDE.md                                           Claude Code 入口
HERMES.md                                           Hermes 入口
OPENCLAW.md                                         OpenClaw 入口
README.en.md                                        英文说明
LICENSE                                             GPL-3.0-only 许可声明
run_pipeline.sh                                     研究/发布统一入口

rl-portfolio-allocator/                             研究训练产物
  SKILL.md                                          研究 skill 说明
  run_pipeline.sh                                   研究子入口
  pyproject.toml                                    Python 项目配置
  conftest.py                                       pytest 共享 fixtures
  scripts/
    features.py          因子计算与特征生成
    factor_catalog.py    100 候选因子族定义
    factor_cache.py      因子缓存读写
    factor_compute.py    因子值计算
    factor_selection.py  Walk-forward fold 内因子选择
    market_state.py      因果市场状态构建
    state.py             环境状态编码
    observation.py       观测空间定义
    action_transform.py  tanh + L1 + EMA 动作变换
    reward.py            DSR + 惩罚项 reward 设计
    env.py               Gymnasium 周频交易环境
    costs.py             交易成本（佣金/印花税/冲击/融券）
    portfolio.py         多空组合构建
    rebalance.py         周度调仓逻辑
    train.py             PPO 训练入口
    backtest.py          主回测
    baselines.py         等权/波动率倒数/Markowitz 基线
    stress_test.py       四段历史压力测试
    walk_forward.py      多 fold/seed walk-forward 编排
    research_gates.py    研究关口判定
    metrics.py           绩效指标计算
    diagnostics.py       训练退化诊断
    validate.py          产物校验
    allocate.py          生产模型训练/推理
    check_data_coverage.py 数据覆盖检查
    config.py            配置与环境变量读取
  tests/                  31 个 pytest 测试文件
  references/
    factor-selection.md  因子选择方案与目录地图
    dsr_derivation.md    DSR reward 推导
  checkpoints/           模型 checkpoint 目录
  data/                  数据目录（不提交，见 .gitignore）
  artifacts/             运行产物目录（不提交，见 .gitignore）

rl-portfolio-allocator-production/                  生产查询（只读）
  SKILL.md                                          生产 skill 说明
  conftest.py                                       pytest fixtures
  scripts/
    query.py            生产 allocation 查询
  tests/
    __init__.py

README.md                                           本文件
```

## 核心逻辑

- **RL 目标**：PPO 学习 K 维因子权重向量（非个股权重），每周输出一次动作。
- **因子池**：10 族 × 10 共 100 个因果 OHLCV 因子（`factor-catalog-v2`），每个 walk-forward fold 在训练区间内做训练-only 选择，冻结 20 个因子用于该 fold 的验证与测试。
- **对照组** `control_6f`：固定六因子（动量20/反转5/波动率20/换手率20/流动性20/收益率偏度60）。
- **候选组** `candidate_20f`：fold 内选出的 20 因子。
- **动作变换**：`tanh` → L1 归一化（`Σ|wᵢ|=1`，可正可负）→ EMA 平滑（`α=0.5`）。
- **组合构建**：综合得分 `s = F_t · w̃_t` → Top-30 做多 + Bottom-15 做空（名义上限 30%）。
- **Reward**（默认 `variant=low`）：`100·净收益 − 0.5·max(0, Δ回撤) − 0.05·max(0, turnover−0.2) − 0.5·max(0, HHI−0.03)`，clip ±5。支持多个 reward 变体（`none`/`low`/`gentle`/`constrained`/`legacy_dsr`），通过 `RLPA_REWARD_CANDIDATES` 控制验证搜索空间。
- **成本内嵌**：佣金 3bps + 印花税 10bps（卖出侧）+ 冲击 5bps × turnover + 融券年化 8%（按日折算）。
- **停牌处理**：`is_suspended` 列标记 + `freeze_suspended` 冻结持仓。

## 数据源

使用 Panda data SDK，不读取本地表格作为正式输入。

| 接口 | 用途 |
|------|------|
| `get_a_share_daily` | A 股日线行情（OHLCV） |
| `get_csi300_members` | CSI300 成分股及权重 |

| 环境变量 | 说明 |
|----------|------|
| `PANDA_DATA_USERNAME` | 必填 |
| `PANDA_DATA_PASSWORD` | 必填 |
| `PANDA_DATA_START_DATE` | 可选，默认 `2004-01-01` |
| `PANDA_DATA_END_DATE` | 可选，默认 `2024-12-31` |
| `RL_ALGO` | 可选，RL 算法，默认 `ppo` |
| `TRAIN_DEVICE` | 可选，训练设备 `auto`/`cuda`/`mps`/`cpu` |
| `RLPA_REWARD_CANDIDATES` | 可选，reward 变体候选列表 |
| `RLPA_SELECTION_TARGET_COUNT` | 可选，因子选择目标数量 |
| `RLPA_LAMBDA_DRAWDOWN` | 可选，回撤惩罚系数 |

## 运行方式

标准入口为仓库根目录的 `run_pipeline.sh`：

```bash
# 冒烟测试：验证契约，research_ok 强制 false，不写 approval
bash run_pipeline.sh --research-smoke

# 正式研究：3 folds × 5 seeds，仅 research_ok=true 时写 approval
RLPA_RUN_ID=<run-id> bash run_pipeline.sh --research-full

# 发布：校验选中因子 bundle 后重训生产模型
bash run_pipeline.sh --publish --approval rl-portfolio-allocator/artifacts/walk_forward/<run_id>/approval.json
```

产物目录：`rl-portfolio-allocator/artifacts/walk_forward/<run_id>/`，包含 `control_6f/`、`candidate_20f/`、`comparison.json`、`gates.json`、`approval.json`。

## 研究关口

| 关口 | 阈值 | 说明 |
|------|------|------|
| `median_oos_sharpe_gain` | ≥ 0.1 | 样本外 Sharpe 中位数超额 |
| `positive_excess_folds` | ≥ 2 | 超额 fold 数 |
| `candidate_cost_2x_oos_sharpe` | ≥ 0 | 2 倍成本后 OOS Sharpe 仍为正 |
| `candidate_annualized_turnover` | ≤ 12 | 年化换手率上限 |
| `candidate_stress_mdd_excess` | ≥ 0.05 | 压力测试 MDD 超额 |
| `candidate_stress_calmar_excess` | ≥ 0 | 压力测试 Calmar 超额 |
| `candidate_stress_long_exposure_util` | ≥ 0.5 | 多头敞口利用率 |
| `complete_paired_evidence` | true | 对照组与候选组均已完整运行 |

全部关口通过（`research_ok=true`）后方可 publish。关口失败时 `publish` 被阻止。

## 压力测试场景

| 场景 | 说明 |
|------|------|
| 2015 A-share crash | 2015 年 A 股暴跌 |
| 2020 COVID | 新冠疫情冲击 |
| 2022 double kill | 股债双杀 |
| 2008 GFC | 全球金融危机（数据起点晚于 2007 则跳过） |

## 验收标准

- `run_pipeline.sh --research-smoke` 全部 pytest 通过（31 个测试），research_ok 强制 false
- `run_pipeline.sh --research-full` 3 folds walk-forward 完整运行，生成 comparison.json 和 gates.json
- 全部 8 项研究关口通过
- 压力测试四场景均完成（数据不足时显式 SKIP，不伪造）
- 不通过研究关口不得发布

## 生产查询

交易 agent 使用 `rl-portfolio-allocator-production` skill 读取 allocation 数据，按 `strategy_id == "RLPA"` 筛选，不在调用时重新训练或回测。

生产查询结果只能作为研究数据读取，不应表述为交易建议、收益预测或确定性信号。

## 依赖

- Python 3.10+
- panda-data
- stable-baselines3 ≥ 2.0
- gymnasium
- pandas
- numpy
- pyarrow
- pytest

## 许可证

本仓库使用 `GPL-3.0-only`。见 `LICENSE`。
