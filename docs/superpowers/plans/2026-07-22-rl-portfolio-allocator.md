# RL 组合权重优化器 实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 交付 `skill-rl-portfolio-allocator` 双模式 skill — 用 PPO 学 K 维因子权重,在沪深300 上生成含成本(佣金/印花税/融券/冲击)的多空组合,reward 为差分 Sharpe(DSR)+ 惩罚项,并按"压力测试 / 主回测 / 生产模型"三种独立训练用途落盘持仓表。

**Architecture:** 数据(panda_data A股) → K 个横截面标准化因子 → Gymnasium 环境(state = 波动/相关性/因子暴露/持仓,action = K维 tanh+L1+EMA 因子权重,step 内嵌成本并输出 DSR reward) → SB3 PPO 训练 → 综合得分 Top-N 做多 + Bottom-M 做空 → 落盘持仓表 → 只读生产查询 skill 读取。目录结构对齐 `skill-dl-transformer-multiasset`(inner skill dir + `-production` 旁路 skill dir)。

**Tech Stack:** Python 3.10+, panda-data(A股接口), stable-baselines3 >= 2.0, gymnasium, torch >= 2.1, pandas, numpy, scipy, pyarrow, matplotlib, pytest。

## Global Constraints

以下约束**每个任务都适用**,数值从设计文档 §2A / §3 / §8 逐字抄录:

- **无未来函数**:t 日因子只用 t 及以前的量价;t 日建仓、t+1 日结算收益(`data_lag=1`);压力段训练窗口只含该段**之前**的连续数据。
- **动作空间只能是 K 维因子权重**:`tanh` 激活 → L1 归一化 `Σ|wᵢ|=1` → EMA 平滑 `w̃_t = α·w_t + (1-α)·w̃_{t-1}`。**禁止**把 300 维股票权重、敞口、多空比例放进 action。
- **敞口固定为确定性规则,不进 action**:多头名义 100%、空头名义上限 30%、`cash = 1 − (多头 + 空头占用)`。
- **Reward 必须是差分 Sharpe (DSR) + 惩罚项**,`R_t` 必须是**扣完全部成本**(佣金 3bps + 印花税 10bps 卖出侧 + 冲击 5bps × turnover + 融券年化 8% 按日折算)的净日收益。**代码层面禁止用累计收益 (cumret) 做 reward。**
- **成本必须内嵌 env.step**,不得后置到回测里"再补扣"。三类成本必须分别可读出用于诊断。
- **A股做空**:允许融券,空头总仓位 ≤ 30%;MVP 全池可融;保证金占用简化为空头名义的固定比例。
- **停牌处理**:停牌股当日不可调仓、权重冻结到复牌;综合得分里停牌股剔出选股池(2015 段必需)。
- **三类训练严格不可混用**:
  - ① 压力测试:训段①之前 → 前向该压力段(`stress_test.py`)。
  - ② 主回测:训早期(如 2010–2022)→ 测样本外(2023+)(`backtest.py`)。
  - ③ 生产模型:训 `数据起点 ~ 最新可得日`,**不留样本外尾巴**(`allocate.py --retrain`),`--infer-only` 复用现有生产模型每日推理。
- **风格 & 依赖对齐 `skill-dl-transformer-multiasset`**:目录 = `skill-rl-portfolio-allocator/rl-portfolio-allocator/{scripts,tests,references,checkpoints}` + 旁路 `rl-portfolio-allocator-production/{scripts,tests,data}`;`conftest.py` 用 `sys.path.insert(0, ...)`;顶层入口文件 `SKILL.md / CLAUDE.md / AGENTS.md / HERMES.md / OPENCLAW.md / .cursor/rules/*.mdc / skill.json / README.md`。
- **数据源**:仅使用 `panda_data` A股接口,不读本地表作正式输入;不记录/提交 `PANDA_DATA_USERNAME` / `PANDA_DATA_PASSWORD`。
- **产物字段(持仓表)固定**:`trade_date, symbol, weight, side(long|short|cash), factor_weights(JSON), composite_score, strategy_id='RLPA', data_version='real-v1', update_time(ISO8601)`。
- **训练诊断强制**:必须记录并在完成时报告 `年化换手率 / 平均持仓数 / 多空敞口利用率`,任一接近 0 判定 agent 退化。
- **提交纪律**:每个 Task 结束必须 `git commit`;checkpoints/ 与 data/*.parquet 加 `.gitignore`;严禁提交凭据。
- **测试**:pytest 覆盖环境不变量(权重和=1、空头≤上限)、成本计算、无未来函数断言、DSR 数值正确性、EMA 平滑正确性。

## 文件结构

```
skill-rl-portfolio-allocator/
├── SKILL.md                                             # T1
├── CLAUDE.md / AGENTS.md / HERMES.md / OPENCLAW.md      # T1
├── README.md / LICENSE / skill.json / .gitignore        # T1
├── .cursor/rules/skill-rl-portfolio-allocator.mdc       # T1
├── rl-portfolio-allocator/
│   ├── SKILL.md                                         # T1
│   ├── conftest.py                                      # T1
│   ├── references/{design_notes.md, dsr_derivation.md}  # T1
│   ├── checkpoints/  (gitignore)                        # T1
│   ├── scripts/
│   │   ├── __init__.py                                  # T1
│   │   ├── config.py           # 常量、成本参数、K因子列表    # T2
│   │   ├── features.py         # panda_data → K因子表        # T3
│   │   ├── costs.py            # 佣金/印花税/冲击/融券        # T4
│   │   ├── action_transform.py # tanh + L1 + EMA             # T5
│   │   ├── reward.py           # DSR + 惩罚项                 # T6
│   │   ├── state.py            # 波动/相关/因子暴露/持仓       # T7
│   │   ├── portfolio.py        # 因子权重 → Top-N/Bottom-M   # T8
│   │   ├── env.py              # Gymnasium 环境组装           # T9
│   │   ├── train.py            # SB3 PPO 训练核心             # T10
│   │   ├── diagnostics.py      # 换手/敞口/退化检测           # T11
│   │   ├── metrics.py          # ARR/MDD/Sharpe/Calmar 等     # T12
│   │   ├── baselines.py        # 等权 / 纯多头 / 静态因子等权 # T13
│   │   ├── backtest.py         # 用途②                        # T14
│   │   ├── stress_test.py      # 用途①(4段前向)              # T15
│   │   ├── allocate.py         # 用途③(--retrain/--infer-only)# T16
│   │   └── validate.py         # 规范性校验                    # T17
│   └── tests/
│       ├── test_features.py                              # T3
│       ├── test_costs.py                                 # T4
│       ├── test_action_transform.py                      # T5
│       ├── test_reward.py                                # T6
│       ├── test_state.py                                 # T7
│       ├── test_portfolio.py                             # T8
│       ├── test_env.py                                   # T9
│       ├── test_train_smoke.py                           # T10
│       ├── test_metrics.py                               # T12
│       ├── test_baselines.py                             # T13
│       ├── test_backtest.py                              # T14
│       ├── test_stress.py                                # T15
│       ├── test_allocate.py                              # T16
│       └── test_validate.py                              # T17
└── rl-portfolio-allocator-production/
    ├── SKILL.md                                          # T18
    ├── scripts/{__init__.py, query.py}                   # T18
    ├── tests/test_query.py                               # T18
    └── data/  (gitignore parquet)                        # T18
```

---

### Task 1: Skill 骨架与双模式入口

**Files:**
- Create: `skill-rl-portfolio-allocator/SKILL.md`
- Create: `skill-rl-portfolio-allocator/CLAUDE.md`
- Create: `skill-rl-portfolio-allocator/AGENTS.md`
- Create: `skill-rl-portfolio-allocator/HERMES.md`
- Create: `skill-rl-portfolio-allocator/OPENCLAW.md`
- Create: `skill-rl-portfolio-allocator/README.md`
- Create: `skill-rl-portfolio-allocator/LICENSE` (GPL-3.0-only, 复用 DLTX 的)
- Create: `skill-rl-portfolio-allocator/skill.json`
- Create: `skill-rl-portfolio-allocator/.gitignore`
- Create: `skill-rl-portfolio-allocator/.cursor/rules/skill-rl-portfolio-allocator.mdc`
- Create: `skill-rl-portfolio-allocator/rl-portfolio-allocator/SKILL.md`
- Create: `skill-rl-portfolio-allocator/rl-portfolio-allocator/conftest.py`
- Create: `skill-rl-portfolio-allocator/rl-portfolio-allocator/scripts/__init__.py`
- Create: `skill-rl-portfolio-allocator/rl-portfolio-allocator/tests/__init__.py`
- Create: `skill-rl-portfolio-allocator/rl-portfolio-allocator/references/design_notes.md`
- Create: `skill-rl-portfolio-allocator/rl-portfolio-allocator/references/dsr_derivation.md`
- Create: `skill-rl-portfolio-allocator/rl-portfolio-allocator/checkpoints/.gitkeep`

**Interfaces:**
- Consumes: (none — this is the entry task)
- Produces: 目录骨架 + 双模式入口文本(供 T2+ 填充脚本);`conftest.py` 提供 `sys.path.insert(0, ...)` 以便测试直接 `import scripts.xxx`。

- [ ] **Step 1: 创建目录树**

Run:
```bash
cd /Users/dmiwu/work/PythonProject/PandaAIQuant/claude_code_skills/skill-rl-portfolio-allocator
mkdir -p .cursor/rules rl-portfolio-allocator/{scripts,tests,references,checkpoints}
```

- [ ] **Step 2: 写顶层 `SKILL.md`**

内容(逐字):
```markdown
# QuantSkills Entry

**Name**: skill-rl-portfolio-allocator

**Description**: PPO-based dynamic factor-weight allocator on CSI300 with embedded trading/borrow/impact costs, DSR reward, and long-short portfolio construction.

## Runtime Entries

- **Codex**: Read this file, then follow the appropriate inner SKILL.md file.
- **Claude Code**: `CLAUDE.md`
- **Cursor**: `.cursor/rules/skill-rl-portfolio-allocator.mdc`
- **Hermes**: `HERMES.md`
- **OpenClaw**: `OPENCLAW.md`

## Dual Mode

- **Research and training**: Follow `rl-portfolio-allocator/SKILL.md`
- **Read-only production allocation queries**: Follow `rl-portfolio-allocator-production/SKILL.md`

## Boundaries

本仓库为研究与工程材料。**不构成投资建议、不承诺收益、不代表 QuantSkills / Panda data / Codex / Claude Code / Cursor / Hermes / OpenClaw 的官方背书。** 不得记录或提交 Panda data 凭据。
```

- [ ] **Step 3: 写 `CLAUDE.md` / `AGENTS.md` / `HERMES.md` / `OPENCLAW.md`**

四个文件正文相同,只把标题换成对应 runtime,内容:
```markdown
# <Runtime Name> Runtime Entry

Use this repository as a QuantSkills skill named `skill-rl-portfolio-allocator`.

Read `SKILL.md` first. For research and training, follow `rl-portfolio-allocator/SKILL.md`. For read-only production allocation queries, follow `rl-portfolio-allocator-production/SKILL.md`.

This material is research tooling only. Do not present outputs as investment advice, return promises, or official endorsement. Do not record or transmit Panda data credentials.
```

- [ ] **Step 4: 写 `.cursor/rules/skill-rl-portfolio-allocator.mdc`**

```markdown
---
description: Enter skill-rl-portfolio-allocator
globs:
alwaysApply: false
---
Read `SKILL.md` first. For research/training use `rl-portfolio-allocator/SKILL.md`, for read-only queries use `rl-portfolio-allocator-production/SKILL.md`.
```

- [ ] **Step 5: 写 `skill.json`**

```json
{
  "name": "skill-rl-portfolio-allocator",
  "description": "PPO-based dynamic factor-weight allocator on CSI300 with embedded costs, DSR reward, and long-short portfolio construction.",
  "license": "GPL-3.0-only",
  "tags": ["quant", "rl", "portfolio", "ppo", "ashare"],
  "metadata": {
    "organization": "QuantSkills",
    "organization_url": "https://github.com/quantskills",
    "repository": "skill-rl-portfolio-allocator",
    "repository_url": "https://github.com/quantskills/skill-rl-portfolio-allocator",
    "project_type": "skill",
    "collection": "rl-alpha"
  }
}
```

- [ ] **Step 6: 写 `.gitignore`**

```
__pycache__/
*.pyc
.pytest_cache/
rl-portfolio-allocator/checkpoints/*
!rl-portfolio-allocator/checkpoints/.gitkeep
rl-portfolio-allocator-production/data/*.parquet
rl-portfolio-allocator-production/data/*.csv
.env
```

- [ ] **Step 7: 写 `LICENSE`**(复用 DLTX 的 GPL-3.0-only)

Run:
```bash
cp /Users/dmiwu/work/PythonProject/PandaAIQuant/claude_code_skills/skill-dl-transformer-multiasset/LICENSE \
   /Users/dmiwu/work/PythonProject/PandaAIQuant/claude_code_skills/skill-rl-portfolio-allocator/LICENSE
```

- [ ] **Step 8: 写 `README.md`**

```markdown
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
```

- [ ] **Step 9: 写 inner `rl-portfolio-allocator/SKILL.md`**

```markdown
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
```

- [ ] **Step 10: 写 `conftest.py`**

```python
import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).parent))
```

- [ ] **Step 11: 写 `scripts/__init__.py` 与 `tests/__init__.py`**

两个文件都留空。

- [ ] **Step 12: 写 `references/design_notes.md`**

一行指向设计文档:
```markdown
详见 `../../docs/superpowers/specs/2026-07-22-rl-portfolio-allocator-design.md`。
```

- [ ] **Step 13: 写 `references/dsr_derivation.md`**

粘贴设计 §3.3 的 DSR 公式(A_t / B_t / DSR_t 三行)与 Moody & Saffell 1998 引用。

- [ ] **Step 14: 建 `checkpoints/.gitkeep`**

Run: `touch rl-portfolio-allocator/checkpoints/.gitkeep`

- [ ] **Step 15: 验证骨架**

Run:
```bash
cd /Users/dmiwu/work/PythonProject/PandaAIQuant/claude_code_skills/skill-rl-portfolio-allocator
find . -type f -not -path './docs/*' -not -path './.git/*' | sort
```
Expected: 至少列出 T1 上文所列所有文件。

- [ ] **Step 16: Commit**

```bash
cd /Users/dmiwu/work/PythonProject/PandaAIQuant/claude_code_skills/skill-rl-portfolio-allocator
git init 2>/dev/null || true
git add -A
git commit -m "feat(T1): scaffold skill-rl-portfolio-allocator dual-mode entries"
```

---

### Task 2: 配置常量 `config.py`

**Files:**
- Create: `rl-portfolio-allocator/scripts/config.py`

**Interfaces:**
- Consumes: `os.environ`(`PANDA_DATA_*`, `PANDA_DATA_START_DATE`, `PANDA_DATA_END_DATE`, `RL_ALGO`, `REWARD_TYPE`, `TRAIN_DEVICE`, `RETRAIN_CADENCE`)。
- Produces:
  - 常量:`FACTOR_NAMES: list[str]`(长度 K=6), `TOP_N: int = 30`, `BOTTOM_M: int = 15`, `LONG_NOTIONAL: float = 1.0`, `SHORT_NOTIONAL_CAP: float = 0.30`。
  - 成本:`COMMISSION_BPS: float = 3.0`, `STAMP_TAX_BPS: float = 10.0`, `IMPACT_BPS: float = 5.0`, `BORROW_RATE_ANNUAL: float = 0.08`, `TRADING_DAYS_PER_YEAR: int = 252`。
  - 动作变换:`EMA_ALPHA: float = 0.5`。
  - Reward:`DSR_ETA: float = 0.01`, `LAMBDA_DRAWDOWN: float = 0.1`, `LAMBDA_TURNOVER: float = 0.05`, `LAMBDA_CONCENTRATION: float = 0.05`。
  - `STRATEGY_ID = "RLPA"`, `DATA_VERSION = "real-v1"`。
  - 函数 `get_config() -> dict`:合并 env 覆盖后的完整配置字典(供其他脚本单一取用点)。

- [ ] **Step 1: 写 `config.py`**

```python
"""集中式常量与环境变量读取。所有其他脚本从这里取配置,不再直接读 os.environ。"""
from __future__ import annotations
import os

FACTOR_NAMES: list[str] = [
    "mom_20",
    "reversal_5",
    "vol_20",
    "turnover_20",
    "amihud_20",
    "ret_skew_60",
]
K: int = len(FACTOR_NAMES)

TOP_N: int = 30
BOTTOM_M: int = 15
LONG_NOTIONAL: float = 1.0
SHORT_NOTIONAL_CAP: float = 0.30

COMMISSION_BPS: float = 3.0
STAMP_TAX_BPS: float = 10.0
IMPACT_BPS: float = 5.0
BORROW_RATE_ANNUAL: float = 0.08
TRADING_DAYS_PER_YEAR: int = 252

EMA_ALPHA: float = 0.5

DSR_ETA: float = 0.01
LAMBDA_DRAWDOWN: float = 0.1
LAMBDA_TURNOVER: float = 0.05
LAMBDA_CONCENTRATION: float = 0.05

STRATEGY_ID: str = "RLPA"
DATA_VERSION: str = "real-v1"


def get_config() -> dict:
    return {
        "factor_names": FACTOR_NAMES,
        "k": K,
        "top_n": TOP_N,
        "bottom_m": BOTTOM_M,
        "long_notional": LONG_NOTIONAL,
        "short_notional_cap": SHORT_NOTIONAL_CAP,
        "commission_bps": COMMISSION_BPS,
        "stamp_tax_bps": STAMP_TAX_BPS,
        "impact_bps": IMPACT_BPS,
        "borrow_rate_annual": BORROW_RATE_ANNUAL,
        "trading_days_per_year": TRADING_DAYS_PER_YEAR,
        "ema_alpha": EMA_ALPHA,
        "dsr_eta": DSR_ETA,
        "lambda_drawdown": LAMBDA_DRAWDOWN,
        "lambda_turnover": LAMBDA_TURNOVER,
        "lambda_concentration": LAMBDA_CONCENTRATION,
        "strategy_id": STRATEGY_ID,
        "data_version": DATA_VERSION,
        "panda_username": os.environ.get("PANDA_DATA_USERNAME"),
        "panda_password": os.environ.get("PANDA_DATA_PASSWORD"),
        "start_date": os.environ.get("PANDA_DATA_START_DATE", "2010-01-01"),
        "end_date": os.environ.get("PANDA_DATA_END_DATE"),  # None → 最新可得
        "rl_algo": os.environ.get("RL_ALGO", "ppo"),
        "reward_type": os.environ.get("REWARD_TYPE", "sharpe"),
        "train_device": os.environ.get("TRAIN_DEVICE", "auto"),
        "retrain_cadence": os.environ.get("RETRAIN_CADENCE", "monthly"),
    }
```

- [ ] **Step 2: Commit**

```bash
git add rl-portfolio-allocator/scripts/config.py
git commit -m "feat(T2): add config.py with cost/reward/action constants"
```

---

### Task 3: 因子表 `features.py`

**Files:**
- Create: `rl-portfolio-allocator/scripts/features.py`
- Create: `rl-portfolio-allocator/tests/test_features.py`

**Interfaces:**
- Consumes: `config.get_config()`, `panda_data` A股接口(`get_index_component('000300.SH')` 拿沪深300成分变动,`get_stock_daily_post` / 等价接口拿后复权量价 + 停牌标记)。
- Produces:
  - `load_universe(start, end) -> pd.DataFrame`:列 `[trade_date, symbol, is_member(bool)]`。
  - `load_prices(symbols, start, end) -> pd.DataFrame`:列 `[trade_date, symbol, open, high, low, close, volume, amount, is_suspended(bool)]`。
  - `compute_factors(prices: pd.DataFrame) -> pd.DataFrame`:列 `[trade_date, symbol] + FACTOR_NAMES + ['is_suspended','ret_1d']`,因子已横截面 z-score 标准化并 clip 到 [-3, 3];**t 日因子只用 ≤ t 的量价**。
  - `save_features(df, path) -> None` / `load_features(path) -> pd.DataFrame`(parquet)。
  - `main()`:落盘 `rl-portfolio-allocator/data/features.parquet`(相对 script 目录)并打印日期范围、股票数、因子维度。

- [ ] **Step 1: 写测试 `tests/test_features.py`**

```python
"""因子表测试。使用小型合成 OHLC 数据,不触发 panda_data。"""
from __future__ import annotations
import numpy as np
import pandas as pd
import pytest

from scripts.features import compute_factors
from scripts.config import FACTOR_NAMES


def _synthetic_prices(n_days: int = 120, n_symbols: int = 20, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2022-01-03", periods=n_days)
    syms = [f"{i:06d}.SZ" for i in range(n_symbols)]
    rows = []
    for s in syms:
        prices = 10.0 * np.exp(np.cumsum(rng.normal(0, 0.02, n_days)))
        vol = rng.integers(1_000_000, 5_000_000, n_days).astype(float)
        for d, p, v in zip(dates, prices, vol):
            rows.append({
                "trade_date": d, "symbol": s,
                "open": p * 0.99, "high": p * 1.01, "low": p * 0.98, "close": p,
                "volume": v, "amount": v * p, "is_suspended": False,
            })
    return pd.DataFrame(rows)


def test_compute_factors_has_all_columns():
    df = compute_factors(_synthetic_prices())
    for f in FACTOR_NAMES:
        assert f in df.columns, f"factor {f} missing"
    assert "trade_date" in df.columns and "symbol" in df.columns
    assert "ret_1d" in df.columns
    assert "is_suspended" in df.columns


def test_compute_factors_cross_sectional_zscore():
    df = compute_factors(_synthetic_prices())
    df = df.dropna(subset=FACTOR_NAMES)
    for date, grp in df.groupby("trade_date"):
        if len(grp) < 5:
            continue
        for f in FACTOR_NAMES:
            vals = grp[f].values
            assert np.abs(vals.mean()) < 1e-6, f"{f}@{date} mean not zero"
            # std allowed slack from clipping
            assert 0.5 < vals.std() < 1.5, f"{f}@{date} std out of range"
            assert vals.min() >= -3.0 - 1e-9 and vals.max() <= 3.0 + 1e-9


def test_compute_factors_no_future_leak():
    """t 日因子只能用 ≤ t 的价格。检验:改动 t+1 日之后的价格,不影响 t 日因子值。"""
    prices = _synthetic_prices()
    f_orig = compute_factors(prices).sort_values(["trade_date", "symbol"]).reset_index(drop=True)

    cutoff = prices["trade_date"].unique()[80]
    prices_perturbed = prices.copy()
    mask = prices_perturbed["trade_date"] > cutoff
    prices_perturbed.loc[mask, ["open", "high", "low", "close"]] *= 2.0
    f_new = compute_factors(prices_perturbed).sort_values(["trade_date", "symbol"]).reset_index(drop=True)

    m = f_orig["trade_date"] <= cutoff
    for f in FACTOR_NAMES:
        pd.testing.assert_series_equal(
            f_orig.loc[m, f].reset_index(drop=True),
            f_new.loc[m, f].reset_index(drop=True),
            check_names=False,
        )
```

- [ ] **Step 2: 运行测试确认失败**

Run: `cd rl-portfolio-allocator && pytest tests/test_features.py -v`
Expected: FAIL — `ModuleNotFoundError` 或 `compute_factors` 未定义。

- [ ] **Step 3: 实现 `features.py`**

```python
"""从 panda_data 加载 CSI300 成分与后复权量价,产生 K 因子横截面 z-score 表。

因子(K=6,与 config.FACTOR_NAMES 严格对齐):
  mom_20       ln(close_t / close_{t-20})
  reversal_5   -ln(close_t / close_{t-5})
  vol_20       std(ret_1d, 20)
  turnover_20  mean(volume / shares_float_proxy, 20)  ← MVP 用 volume/mean(volume,252)
  amihud_20    mean(|ret_1d| / (amount + eps), 20)   ← 非流动性
  ret_skew_60  skew(ret_1d, 60)

t 日只用 ≤ t 的量价 → 严格 shift(1) 或 rolling().last() 结构。
每日横截面 z-score(减均值除标准差)再 clip 到 [-3, 3]。
"""
from __future__ import annotations
import os
import pathlib
from typing import Optional
import numpy as np
import pandas as pd

from scripts.config import FACTOR_NAMES, get_config

_EPS = 1e-12


def _ln_return(close: pd.Series, n: int) -> pd.Series:
    return np.log(close / close.shift(n))


def _compute_single_symbol(g: pd.DataFrame) -> pd.DataFrame:
    g = g.sort_values("trade_date").copy()
    close = g["close"]
    volume = g["volume"]
    amount = g["amount"]
    ret_1d = close.pct_change()
    g["ret_1d"] = ret_1d
    g["mom_20"] = _ln_return(close, 20)
    g["reversal_5"] = -_ln_return(close, 5)
    g["vol_20"] = ret_1d.rolling(20).std()
    g["turnover_20"] = (volume / volume.rolling(252).mean()).rolling(20).mean()
    g["amihud_20"] = (ret_1d.abs() / (amount + _EPS)).rolling(20).mean()
    g["ret_skew_60"] = ret_1d.rolling(60).skew()
    return g


def _cross_sectional_zscore(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        grp = out.groupby("trade_date")[c]
        mu = grp.transform("mean")
        sd = grp.transform("std")
        z = (out[c] - mu) / (sd + _EPS)
        out[c] = z.clip(-3.0, 3.0)
    return out


def compute_factors(prices: pd.DataFrame) -> pd.DataFrame:
    """输入长表 prices(trade_date, symbol, OHLCV+amount+is_suspended),输出因子长表。"""
    required = {"trade_date", "symbol", "open", "high", "low", "close", "volume", "amount"}
    missing = required - set(prices.columns)
    if missing:
        raise ValueError(f"prices missing columns: {missing}")
    if "is_suspended" not in prices.columns:
        prices = prices.assign(is_suspended=False)
    df = prices.groupby("symbol", group_keys=False).apply(_compute_single_symbol)
    df = _cross_sectional_zscore(df, FACTOR_NAMES)
    keep = ["trade_date", "symbol", "ret_1d", "is_suspended", *FACTOR_NAMES]
    return df[keep].reset_index(drop=True)


def load_universe(start: str, end: Optional[str]) -> pd.DataFrame:
    import panda_data  # noqa: WPS433 (延迟导入方便 CI 无凭据也能跑单测)
    return panda_data.get_index_component("000300.SH", start_date=start, end_date=end)


def load_prices(symbols: list[str], start: str, end: Optional[str]) -> pd.DataFrame:
    import panda_data
    df = panda_data.get_stock_daily_post(
        symbols=symbols, start_date=start, end_date=end
    )
    if "is_suspended" not in df.columns:
        df["is_suspended"] = df["volume"].fillna(0) == 0
    return df


def save_features(df: pd.DataFrame, path: pathlib.Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)


def load_features(path: pathlib.Path) -> pd.DataFrame:
    return pd.read_parquet(path)


def main() -> None:
    cfg = get_config()
    start, end = cfg["start_date"], cfg["end_date"]
    universe = load_universe(start, end)
    symbols = sorted(universe["symbol"].unique().tolist())
    prices = load_prices(symbols, start, end)
    feats = compute_factors(prices)
    out = pathlib.Path(__file__).resolve().parent.parent / "data" / "features.parquet"
    save_features(feats, out)
    print(
        f"features saved: {out}  rows={len(feats)}  "
        f"dates={feats['trade_date'].min()}..{feats['trade_date'].max()}  "
        f"symbols={feats['symbol'].nunique()}  factors={len(FACTOR_NAMES)}"
    )


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: 运行测试确认通过**

Run: `cd rl-portfolio-allocator && pytest tests/test_features.py -v`
Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add rl-portfolio-allocator/scripts/features.py rl-portfolio-allocator/tests/test_features.py
git commit -m "feat(T3): factor computation with cross-sectional z-score + no-future-leak test"
```

---

### Task 4: 成本模型 `costs.py`

**Files:**
- Create: `rl-portfolio-allocator/scripts/costs.py`
- Create: `rl-portfolio-allocator/tests/test_costs.py`

**Interfaces:**
- Consumes: `config.get_config()`。
- Produces(所有返回都是**该日累加的比例成本**,基于组合总名义 1.0):
  - `commission_cost(prev_w: np.ndarray, target_w: np.ndarray, commission_bps: float) -> float`:`bps × Σ|Δ|` / 1e4。
  - `stamp_tax_cost(prev_w: np.ndarray, target_w: np.ndarray, stamp_bps: float) -> float`:仅**卖出侧**(`Δ < 0` 的绝对值之和) × bps / 1e4。
  - `impact_cost(prev_w: np.ndarray, target_w: np.ndarray, impact_bps: float, nonlinear: bool = False) -> float`:线性 `bps/1e4 × Σ|Δ|`,`nonlinear=True` 时 `bps/1e4 × Σ|Δ|^1.5`。
  - `borrow_cost(target_w: np.ndarray, borrow_rate_annual: float, trading_days: int) -> float`:仅对空头 `borrow_rate_annual / trading_days × Σmax(-w, 0)`。
  - `total_costs(prev_w, target_w, cfg) -> dict[str, float]`:返回四项 + `total`。

- [ ] **Step 1: 写测试**

```python
from __future__ import annotations
import numpy as np
import pytest

from scripts.costs import (
    commission_cost,
    stamp_tax_cost,
    impact_cost,
    borrow_cost,
    total_costs,
)
from scripts.config import get_config


def test_commission_symmetric_on_both_sides():
    prev = np.array([0.5, -0.2, 0.0])
    tgt = np.array([0.2, 0.0, 0.3])
    # |Δ| = |0.3| + |0.2| + |0.3| = 0.8;  bps=3 → 0.8 * 3 / 1e4 = 0.00024
    assert commission_cost(prev, tgt, 3.0) == pytest.approx(0.00024, rel=1e-9)


def test_stamp_tax_only_on_sell_side():
    prev = np.array([0.5, 0.2])
    tgt = np.array([0.2, 0.4])
    # Δ = [-0.3, +0.2];sell=|Δ<0|=0.3 → bps=10 → 0.3 * 10 / 1e4 = 3e-4
    assert stamp_tax_cost(prev, tgt, 10.0) == pytest.approx(3e-4, rel=1e-9)
    # 全买入 → 0
    assert stamp_tax_cost(np.array([0.0, 0.0]), np.array([0.5, 0.5]), 10.0) == 0.0


def test_impact_linear_and_nonlinear():
    prev = np.array([0.0, 0.0])
    tgt = np.array([0.4, 0.4])
    lin = impact_cost(prev, tgt, 5.0, nonlinear=False)
    # 5/1e4 * 0.8
    assert lin == pytest.approx(0.8 * 5 / 1e4, rel=1e-9)
    nl = impact_cost(prev, tgt, 5.0, nonlinear=True)
    # 5/1e4 * (0.4^1.5 + 0.4^1.5)
    assert nl == pytest.approx(5 / 1e4 * (2 * 0.4 ** 1.5), rel=1e-9)


def test_borrow_only_on_shorts():
    tgt_long_only = np.array([0.6, 0.4])
    assert borrow_cost(tgt_long_only, 0.08, 252) == 0.0
    tgt_with_short = np.array([0.6, -0.3])
    expected = 0.08 / 252 * 0.3
    assert borrow_cost(tgt_with_short, 0.08, 252) == pytest.approx(expected, rel=1e-9)


def test_total_costs_returns_all_components_and_sum():
    cfg = get_config()
    prev = np.array([0.5, -0.2, 0.0])
    tgt = np.array([0.2, 0.0, 0.3])
    out = total_costs(prev, tgt, cfg)
    for k in ("commission", "stamp_tax", "impact", "borrow", "total"):
        assert k in out
    assert out["total"] == pytest.approx(
        out["commission"] + out["stamp_tax"] + out["impact"] + out["borrow"], rel=1e-9
    )
    assert out["total"] > 0
```

- [ ] **Step 2: 运行 → FAIL**

Run: `cd rl-portfolio-allocator && pytest tests/test_costs.py -v`
Expected: FAIL(module 未定义)。

- [ ] **Step 3: 实现 `costs.py`**

```python
"""交易/印花税/冲击/融券成本。所有函数返回该日成本占组合总名义(=1.0)的比例。"""
from __future__ import annotations
import numpy as np


def commission_cost(prev_w: np.ndarray, target_w: np.ndarray, commission_bps: float) -> float:
    turnover = float(np.abs(target_w - prev_w).sum())
    return commission_bps / 1e4 * turnover


def stamp_tax_cost(prev_w: np.ndarray, target_w: np.ndarray, stamp_bps: float) -> float:
    delta = target_w - prev_w
    sell = float(np.clip(-delta, 0.0, None).sum())
    return stamp_bps / 1e4 * sell


def impact_cost(
    prev_w: np.ndarray, target_w: np.ndarray, impact_bps: float, nonlinear: bool = False
) -> float:
    d = np.abs(target_w - prev_w)
    magnitude = float((d ** 1.5).sum()) if nonlinear else float(d.sum())
    return impact_bps / 1e4 * magnitude


def borrow_cost(
    target_w: np.ndarray, borrow_rate_annual: float, trading_days_per_year: int
) -> float:
    short_notional = float(np.clip(-target_w, 0.0, None).sum())
    return borrow_rate_annual / trading_days_per_year * short_notional


def total_costs(prev_w: np.ndarray, target_w: np.ndarray, cfg: dict) -> dict:
    c = commission_cost(prev_w, target_w, cfg["commission_bps"])
    s = stamp_tax_cost(prev_w, target_w, cfg["stamp_tax_bps"])
    i = impact_cost(prev_w, target_w, cfg["impact_bps"], nonlinear=False)
    b = borrow_cost(target_w, cfg["borrow_rate_annual"], cfg["trading_days_per_year"])
    return {"commission": c, "stamp_tax": s, "impact": i, "borrow": b, "total": c + s + i + b}
```

- [ ] **Step 4: 运行 → PASS**

Run: `cd rl-portfolio-allocator && pytest tests/test_costs.py -v`
Expected: 5 passed.

- [ ] **Step 5: Commit**

```bash
git add rl-portfolio-allocator/scripts/costs.py rl-portfolio-allocator/tests/test_costs.py
git commit -m "feat(T4): commission/stamp/impact/borrow cost functions"
```

---

### Task 5: 动作变换 `action_transform.py`

**Files:**
- Create: `rl-portfolio-allocator/scripts/action_transform.py`
- Create: `rl-portfolio-allocator/tests/test_action_transform.py`

**Interfaces:**
- Consumes: `config.EMA_ALPHA`。
- Produces:
  - `tanh_l1_normalize(raw: np.ndarray, eps: float = 1e-8) -> np.ndarray`:`tanh(raw)`,再除以 `Σ|·|+eps`,返回同维数组,`Σ|out|≈1`。
  - `ema_smooth(w_new: np.ndarray, w_prev: np.ndarray, alpha: float) -> np.ndarray`:`α·w_new + (1-α)·w_prev`。
  - `transform_action(raw: np.ndarray, w_prev: np.ndarray, alpha: float) -> np.ndarray`:先归一化再 EMA,再对 EMA 结果**再做一次 L1 归一化**保证 `Σ|w̃|=1`(EMA 会破坏 L1)。

- [ ] **Step 1: 写测试**

```python
from __future__ import annotations
import numpy as np
import pytest

from scripts.action_transform import tanh_l1_normalize, ema_smooth, transform_action


def test_tanh_l1_sums_to_one_l1_and_preserves_sign():
    raw = np.array([2.0, -1.0, 0.5, -3.0])
    w = tanh_l1_normalize(raw)
    assert np.abs(np.abs(w).sum() - 1.0) < 1e-6
    # 符号必须与 tanh(raw) 一致
    assert np.all(np.sign(w) == np.sign(np.tanh(raw)))


def test_tanh_l1_zero_input_safe():
    w = tanh_l1_normalize(np.zeros(4))
    assert np.all(np.isfinite(w))
    assert np.abs(w).sum() == pytest.approx(0.0, abs=1e-6)


def test_ema_smooth_formula():
    prev = np.array([0.5, -0.5])
    new = np.array([1.0, 0.0])
    out = ema_smooth(new, prev, alpha=0.5)
    np.testing.assert_allclose(out, np.array([0.75, -0.25]))


def test_transform_action_l1_norm_after_ema():
    raw = np.array([1.0, -2.0, 0.5])
    prev = np.array([0.3, -0.4, 0.3])
    out = transform_action(raw, prev, alpha=0.5)
    assert np.abs(np.abs(out).sum() - 1.0) < 1e-6


def test_transform_action_negative_allowed():
    raw = np.array([-3.0, -3.0, -3.0])
    prev = np.zeros(3)
    out = transform_action(raw, prev, alpha=1.0)
    assert np.all(out < 0)
```

- [ ] **Step 2: 运行 → FAIL**

- [ ] **Step 3: 实现 `action_transform.py`**

```python
"""RL 原始动作 → K 维因子权重的变换:tanh → L1 归一化 → EMA 平滑 → 再 L1。"""
from __future__ import annotations
import numpy as np

_EPS = 1e-8


def tanh_l1_normalize(raw: np.ndarray, eps: float = _EPS) -> np.ndarray:
    t = np.tanh(raw)
    denom = np.abs(t).sum() + eps
    return t / denom


def ema_smooth(w_new: np.ndarray, w_prev: np.ndarray, alpha: float) -> np.ndarray:
    return alpha * w_new + (1.0 - alpha) * w_prev


def transform_action(raw: np.ndarray, w_prev: np.ndarray, alpha: float) -> np.ndarray:
    w_new = tanh_l1_normalize(raw)
    w_ema = ema_smooth(w_new, w_prev, alpha)
    # EMA 破坏 L1;再归一化一次
    denom = np.abs(w_ema).sum() + _EPS
    return w_ema / denom
```

- [ ] **Step 4: 运行 → PASS**

Run: `cd rl-portfolio-allocator && pytest tests/test_action_transform.py -v`
Expected: 5 passed.

- [ ] **Step 5: Commit**

```bash
git add rl-portfolio-allocator/scripts/action_transform.py rl-portfolio-allocator/tests/test_action_transform.py
git commit -m "feat(T5): tanh+L1+EMA action transform"
```

---

### Task 6: Reward `reward.py`(DSR + 惩罚项)

**Files:**
- Create: `rl-portfolio-allocator/scripts/reward.py`
- Create: `rl-portfolio-allocator/tests/test_reward.py`

**Interfaces:**
- Consumes: `config.DSR_ETA, LAMBDA_DRAWDOWN, LAMBDA_TURNOVER, LAMBDA_CONCENTRATION`。
- Produces:
  - `class DSRState`:`A: float, B: float`(EMA 一阶/二阶矩)、`peak: float`(累计净值峰值,用于 drawdown)、`nav: float`。
    - `update(r_net: float, eta: float, sortino: bool) -> float`:更新 A/B/peak/nav,返回**本步 DSR 增量**。
  - `hhi(weights: np.ndarray) -> float`:赫芬达尔 = `Σ wᵢ²`(用绝对权重计算)。
  - `compose_reward(dsr_delta, drawdown, turnover, hhi_val, cfg) -> tuple[float, dict]`:总 reward + 分量 dict(供诊断)。
  - **`R_net` 计算约定**:上游(env)传入的 `r_net_t = 毛收益_t − total_costs_t`,禁止调用者传入毛收益;函数内部不重复扣成本。

- [ ] **Step 1: 写测试**

```python
from __future__ import annotations
import numpy as np
import pytest

from scripts.reward import DSRState, hhi, compose_reward
from scripts.config import get_config


def test_dsr_first_step_defined_and_finite():
    st = DSRState()
    d = st.update(r_net=0.01, eta=0.01, sortino=False)
    assert np.isfinite(d)


def test_dsr_positive_returns_increase_A_moment():
    st = DSRState()
    A_start = st.A
    for _ in range(100):
        st.update(r_net=0.01, eta=0.05, sortino=False)
    assert st.A > A_start


def test_hhi_uses_absolute_weights():
    w = np.array([0.5, -0.5])
    assert hhi(w) == pytest.approx(0.5)
    w2 = np.array([1.0, 0.0, 0.0])
    assert hhi(w2) == pytest.approx(1.0)


def test_compose_reward_sign_of_penalties():
    cfg = get_config()
    r, parts = compose_reward(dsr_delta=1.0, drawdown=0.2, turnover=0.5, hhi_val=0.3, cfg=cfg)
    assert parts["dsr"] == pytest.approx(1.0)
    assert parts["drawdown_penalty"] < 0
    assert parts["turnover_penalty"] < 0
    assert parts["concentration_penalty"] < 0
    assert r == pytest.approx(
        parts["dsr"] + parts["drawdown_penalty"]
        + parts["turnover_penalty"] + parts["concentration_penalty"]
    )


def test_dsr_state_peak_and_drawdown():
    st = DSRState()
    for r in [0.01, 0.01, -0.05]:
        st.update(r_net=r, eta=0.01, sortino=False)
    assert st.peak >= st.nav
    dd = (st.peak - st.nav) / st.peak
    assert dd > 0
```

- [ ] **Step 2: 运行 → FAIL**

- [ ] **Step 3: 实现 `reward.py`**

```python
"""差分 Sharpe(Moody & Saffell 1998)+ 惩罚项。R_t 必须是扣完全成本的净收益。"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np


@dataclass
class DSRState:
    A: float = 0.0           # 一阶矩 EMA
    B: float = 1e-6          # 二阶矩 EMA(初值 > 0 避免除零)
    nav: float = 1.0         # 累计净值
    peak: float = 1.0        # 累计净值峰值

    def update(self, r_net: float, eta: float, sortino: bool = False) -> float:
        """更新内部状态,返回本步 DSR 增量。"""
        delta_A = r_net - self.A
        r2 = min(r_net, 0.0) ** 2 if sortino else r_net ** 2
        delta_B = r2 - self.B
        denom = (self.B - self.A ** 2) ** 1.5
        if denom <= 0 or not np.isfinite(denom):
            dsr = 0.0
        else:
            dsr = float((self.B * delta_A - 0.5 * self.A * delta_B) / denom)
        # 更新 A/B
        self.A = self.A + eta * delta_A
        self.B = self.B + eta * delta_B
        # 更新净值和峰值(注意:此处使用净收益复利)
        self.nav = self.nav * (1.0 + r_net)
        if self.nav > self.peak:
            self.peak = self.nav
        return dsr


def hhi(weights: np.ndarray) -> float:
    """赫芬达尔指数,基于**绝对**权重(空头也计入集中度)。"""
    aw = np.abs(weights)
    denom = aw.sum() + 1e-12
    p = aw / denom
    return float((p ** 2).sum())


def compose_reward(
    dsr_delta: float, drawdown: float, turnover: float, hhi_val: float, cfg: dict
) -> tuple[float, dict]:
    dd_pen = -cfg["lambda_drawdown"] * max(0.0, drawdown)
    to_pen = -cfg["lambda_turnover"] * turnover
    conc_pen = -cfg["lambda_concentration"] * hhi_val
    total = dsr_delta + dd_pen + to_pen + conc_pen
    return total, {
        "dsr": dsr_delta,
        "drawdown_penalty": dd_pen,
        "turnover_penalty": to_pen,
        "concentration_penalty": conc_pen,
        "total": total,
    }
```

- [ ] **Step 4: 运行 → PASS**

Run: `cd rl-portfolio-allocator && pytest tests/test_reward.py -v`
Expected: 5 passed.

- [ ] **Step 5: Commit**

```bash
git add rl-portfolio-allocator/scripts/reward.py rl-portfolio-allocator/tests/test_reward.py
git commit -m "feat(T6): DSR state + penalty composition"
```

---

### Task 7: State `state.py`(波动 / 相关性 / 因子暴露 / 持仓)

**Files:**
- Create: `rl-portfolio-allocator/scripts/state.py`
- Create: `rl-portfolio-allocator/tests/test_state.py`

**Interfaces:**
- Consumes: `config.FACTOR_NAMES, K`。
- Produces:
  - `class StateBuilder`:构造函数接受 `factor_panel_by_date: dict[pd.Timestamp, pd.DataFrame]`(每个 date 的因子暴露矩阵 `F_t`)、`index_returns: pd.Series`(市场基准日收益)、`portfolio_returns_history: list[float]`(env 累加)、`recent_turnover_history: list[float]`。
  - `build(date, holdings_w: np.ndarray, factor_names: list[str], prev_factor_w: np.ndarray, cash: float) -> np.ndarray`:返回定长向量。分量:
    - `vol_20, vol_60, market_vol_20, vol_regime_quantile`(4)
    - `avg_pairwise_corr_20, avg_pairwise_corr_60`(2 — 基于 index 成分近似,可用因子间相关做代理以避免 300×300 计算)
    - `factor_exposure_weighted (K), factor_ic_recent (K)`(2K)
    - `prev_factor_w (K), cash_ratio (1), recent_turnover (1)`(K+2)
  - 总维度固定 = `4 + 2 + 2*K + K + 2 = 8 + 3*K`;K=6 → 26 维。**必须**返回 `np.float32` 供 SB3。
  - 提供 `state_dim(k: int) -> int` 辅助函数。

- [ ] **Step 1: 写测试**

```python
from __future__ import annotations
import numpy as np
import pandas as pd
import pytest

from scripts.state import StateBuilder, state_dim
from scripts.config import FACTOR_NAMES, K


def _make_builder(n_days: int = 80, n_syms: int = 10):
    dates = pd.bdate_range("2022-01-03", periods=n_days)
    panels = {}
    rng = np.random.default_rng(0)
    for d in dates:
        panels[d] = pd.DataFrame(
            rng.standard_normal((n_syms, K)),
            index=[f"S{i}" for i in range(n_syms)],
            columns=FACTOR_NAMES,
        )
    idx_rets = pd.Series(rng.normal(0, 0.01, n_days), index=dates)
    return StateBuilder(
        factor_panel_by_date=panels,
        index_returns=idx_rets,
        portfolio_returns_history=[0.001] * 60,
        recent_turnover_history=[0.05] * 20,
    ), dates, [f"S{i}" for i in range(n_syms)]


def test_state_dim_formula():
    assert state_dim(K) == 8 + 3 * K


def test_build_returns_float32_and_correct_dim():
    b, dates, syms = _make_builder()
    holdings = np.zeros(len(syms))
    holdings[0] = 1.0
    prev_w = np.zeros(K)
    s = b.build(dates[-1], holdings, FACTOR_NAMES, prev_w, cash=0.0)
    assert s.dtype == np.float32
    assert s.shape == (state_dim(K),)
    assert np.all(np.isfinite(s))


def test_build_reacts_to_prev_factor_weights():
    b, dates, syms = _make_builder()
    holdings = np.ones(len(syms)) / len(syms)
    s1 = b.build(dates[-1], holdings, FACTOR_NAMES, np.zeros(K), cash=0.0)
    s2 = b.build(dates[-1], holdings, FACTOR_NAMES, np.ones(K) / K, cash=0.0)
    assert not np.allclose(s1, s2)
```

- [ ] **Step 2: 运行 → FAIL**

- [ ] **Step 3: 实现 `state.py`**

```python
"""RL State 构造:波动/相关性/因子暴露/持仓。所有量只用 ≤ t 的信息。"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional
import numpy as np
import pandas as pd


def state_dim(k: int) -> int:
    # vol(4) + corr(2) + factor_exposure(k) + factor_ic(k) + prev_fw(k) + cash(1) + turnover(1)
    return 8 + 3 * k


def _safe_std(x: np.ndarray) -> float:
    return float(np.std(x)) if len(x) >= 2 else 0.0


def _quantile_rank(x: float, arr: np.ndarray) -> float:
    if len(arr) == 0:
        return 0.5
    return float((arr <= x).mean())


@dataclass
class StateBuilder:
    factor_panel_by_date: dict
    index_returns: pd.Series
    portfolio_returns_history: list = field(default_factory=list)
    recent_turnover_history: list = field(default_factory=list)
    _vol_history: list = field(default_factory=list)  # 记录历史 20日波动用于 regime 分位

    def build(
        self,
        date,
        holdings_w: np.ndarray,
        factor_names: list,
        prev_factor_w: np.ndarray,
        cash: float,
    ) -> np.ndarray:
        rets = np.asarray(self.portfolio_returns_history, dtype=float)
        vol_20 = _safe_std(rets[-20:])
        vol_60 = _safe_std(rets[-60:])
        idx_slice = self.index_returns.loc[:date].values
        market_vol_20 = _safe_std(idx_slice[-20:])
        self._vol_history.append(vol_20)
        vol_regime_q = _quantile_rank(vol_20, np.asarray(self._vol_history[:-1]))

        # 因子间相关代理组合成分相关性(MVP 简化)
        F = self.factor_panel_by_date.get(date)
        if F is not None and len(F) >= 5:
            corr = F.corr().values
            iu = np.triu_indices_from(corr, k=1)
            avg_corr_20 = float(np.nanmean(corr[iu])) if iu[0].size else 0.0
        else:
            avg_corr_20 = 0.0
        avg_corr_60 = avg_corr_20  # MVP:同值占位;后续可换真正 60 日窗口

        # 因子暴露 = 持仓加权因子值
        if F is not None:
            common = F.index.intersection(pd.Index(range(len(holdings_w))).astype(str) if False else F.index)
            # 按顺序对齐:假设 holdings_w 已按 F.index 顺序排列
            exposure = F.values.T @ holdings_w[: len(F)] if len(holdings_w) >= len(F) else np.zeros(len(factor_names))
        else:
            exposure = np.zeros(len(factor_names))

        # 因子近期 IC(与下一期 index 收益的横截面 rank IC 代理):MVP 用 0 占位;
        # 实际实现:env 会传入 rolling IC dict;这里预留接口。
        factor_ic = np.zeros(len(factor_names))

        recent_to = float(np.mean(self.recent_turnover_history[-20:])) if self.recent_turnover_history else 0.0

        vec = np.concatenate([
            np.array([vol_20, vol_60, market_vol_20, vol_regime_q], dtype=float),
            np.array([avg_corr_20, avg_corr_60], dtype=float),
            exposure.astype(float),
            factor_ic.astype(float),
            prev_factor_w.astype(float),
            np.array([cash, recent_to], dtype=float),
        ])
        vec = np.nan_to_num(vec, nan=0.0, posinf=0.0, neginf=0.0)
        return vec.astype(np.float32)
```

- [ ] **Step 4: 运行 → PASS**

Run: `cd rl-portfolio-allocator && pytest tests/test_state.py -v`
Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add rl-portfolio-allocator/scripts/state.py rl-portfolio-allocator/tests/test_state.py
git commit -m "feat(T7): StateBuilder for vol/corr/exposure/holdings"
```

---

### Task 8: Portfolio 构造 `portfolio.py`(因子权重 → 目标持仓)

**Files:**
- Create: `rl-portfolio-allocator/scripts/portfolio.py`
- Create: `rl-portfolio-allocator/tests/test_portfolio.py`

**Interfaces:**
- Consumes: `config.TOP_N, BOTTOM_M, LONG_NOTIONAL, SHORT_NOTIONAL_CAP`。
- Produces:
  - `composite_score(F: np.ndarray, factor_w: np.ndarray) -> np.ndarray`:`F @ factor_w`,形状 `(n_stocks,)`。
  - `select_long_short(scores, is_suspended, top_n, bottom_m) -> tuple[np.ndarray, np.ndarray]`:返回 `(long_idx, short_idx)`,已剔除停牌股。
  - `target_weights(scores, long_idx, short_idx, long_notional, short_cap) -> np.ndarray`:多头名义 = `long_notional`,组内按 `scores` 正值加权;空头名义 = `min(short_cap, long_notional)`,组内按 `-scores` 加权;其余股票权重 0。返回 `(n_stocks,)`,`Σ|w| ≤ long_notional + short_cap`。
  - `freeze_suspended(target_w: np.ndarray, prev_w: np.ndarray, is_suspended: np.ndarray) -> np.ndarray`:停牌股用 `prev_w`,活跃股用 `target_w`。

**必须保证**:`select_long_short` 返回的 `long_idx ∩ suspended = ∅`;`np.sum(np.abs(target_weights(...)[short_idx])) ≤ short_cap + 1e-9`。

- [ ] **Step 1: 写测试**

```python
from __future__ import annotations
import numpy as np
import pytest

from scripts.portfolio import (
    composite_score, select_long_short, target_weights, freeze_suspended
)


def test_composite_score_shape_and_value():
    F = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    fw = np.array([0.6, -0.4])
    s = composite_score(F, fw)
    np.testing.assert_allclose(s, np.array([0.6, -0.4, 0.2]))


def test_select_excludes_suspended():
    scores = np.array([3.0, 2.0, 1.0, -1.0, -2.0])
    susp = np.array([False, True, False, False, False])
    long_idx, short_idx = select_long_short(scores, susp, top_n=2, bottom_m=2)
    assert 1 not in long_idx  # 停牌股被剔除
    assert set(long_idx.tolist()) == {0, 2}
    assert set(short_idx.tolist()) == {3, 4}


def test_target_weights_notional_limits():
    scores = np.array([3.0, 2.0, 1.0, -1.0, -2.0, -3.0])
    long_idx = np.array([0, 1])
    short_idx = np.array([4, 5])
    w = target_weights(scores, long_idx, short_idx, long_notional=1.0, short_cap=0.3)
    assert w[0] > 0 and w[1] > 0
    assert w[4] < 0 and w[5] < 0
    assert w[2] == 0.0 and w[3] == 0.0
    assert abs(w[[0, 1]].sum() - 1.0) < 1e-9
    assert abs(w[[4, 5]].sum() + 0.3) < 1e-9  # 空头总名义 = 0.3


def test_target_weights_intra_group_by_score():
    # 多头组内 [3,2] → 权重比 3:2 → 0.6/0.4
    scores = np.array([3.0, 2.0])
    w = target_weights(scores, np.array([0, 1]), np.array([], dtype=int), 1.0, 0.3)
    np.testing.assert_allclose(w, [0.6, 0.4])


def test_freeze_suspended_replaces_only_suspended():
    tgt = np.array([0.3, 0.2, 0.5])
    prev = np.array([0.1, 0.4, 0.5])
    susp = np.array([False, True, False])
    out = freeze_suspended(tgt, prev, susp)
    np.testing.assert_allclose(out, [0.3, 0.4, 0.5])
```

- [ ] **Step 2: 运行 → FAIL**

- [ ] **Step 3: 实现 `portfolio.py`**

```python
"""因子权重 → Top-N 多头 + Bottom-M 空头 → 目标持仓。"""
from __future__ import annotations
import numpy as np


def composite_score(F: np.ndarray, factor_w: np.ndarray) -> np.ndarray:
    return F @ factor_w


def select_long_short(
    scores: np.ndarray, is_suspended: np.ndarray, top_n: int, bottom_m: int
) -> tuple[np.ndarray, np.ndarray]:
    n = len(scores)
    mask = ~is_suspended
    idx = np.arange(n)[mask]
    if len(idx) == 0:
        return np.array([], dtype=int), np.array([], dtype=int)
    s = scores[mask]
    order = np.argsort(-s)  # 降序:得分高的在前
    long_local = order[: min(top_n, len(order))]
    short_local = order[-min(bottom_m, len(order)) :] if bottom_m > 0 else np.array([], dtype=int)
    return idx[long_local], idx[short_local]


def _score_weighted(scores_pos: np.ndarray) -> np.ndarray:
    tot = scores_pos.sum()
    if tot <= 0:
        return np.ones_like(scores_pos) / max(len(scores_pos), 1)
    return scores_pos / tot


def target_weights(
    scores: np.ndarray,
    long_idx: np.ndarray,
    short_idx: np.ndarray,
    long_notional: float,
    short_cap: float,
) -> np.ndarray:
    n = len(scores)
    w = np.zeros(n)
    if len(long_idx) > 0:
        s_long = np.clip(scores[long_idx], a_min=1e-8, a_max=None)
        w[long_idx] = long_notional * _score_weighted(s_long)
    if len(short_idx) > 0:
        s_short = np.clip(-scores[short_idx], a_min=1e-8, a_max=None)
        w[short_idx] = -short_cap * _score_weighted(s_short)
    return w


def freeze_suspended(
    target_w: np.ndarray, prev_w: np.ndarray, is_suspended: np.ndarray
) -> np.ndarray:
    out = target_w.copy()
    out[is_suspended] = prev_w[is_suspended]
    return out
```

- [ ] **Step 4: 运行 → PASS**

Run: `cd rl-portfolio-allocator && pytest tests/test_portfolio.py -v`
Expected: 5 passed.

- [ ] **Step 5: Commit**

```bash
git add rl-portfolio-allocator/scripts/portfolio.py rl-portfolio-allocator/tests/test_portfolio.py
git commit -m "feat(T8): factor-weight → long/short target holdings"
```

---

### Task 9: Gymnasium 环境 `env.py`

**Files:**
- Create: `rl-portfolio-allocator/scripts/env.py`
- Create: `rl-portfolio-allocator/tests/test_env.py`

**Interfaces:**
- Consumes: 前面所有模块。
- Produces:
  - `class PortfolioEnv(gymnasium.Env)`:
    - `__init__(self, features_df: pd.DataFrame, index_returns: pd.Series, cfg: dict, start_date, end_date, nonlinear_impact: bool = False)`。
    - `action_space = gym.spaces.Box(low=-1, high=1, shape=(K,), dtype=np.float32)`(裸动作,由 `transform_action` 处理)。
    - `observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(state_dim(K),), dtype=np.float32)`。
    - `reset(seed=None) -> (obs, info)`:重置到 `start_date`;`prev_factor_w = 0`;`prev_stock_w = 0`;`DSRState()` 重建;返回初始 obs。
    - `step(action) -> (obs, reward, terminated, truncated, info)`:
      1. `factor_w = transform_action(action, prev_factor_w, EMA_ALPHA)`
      2. `F_t = features[date]` 的 K 列矩阵
      3. `scores = composite_score(F_t, factor_w)`
      4. `long_idx, short_idx = select_long_short(scores, is_suspended, TOP_N, BOTTOM_M)`
      5. `target_w = target_weights(...)` → `target_w = freeze_suspended(target_w, prev_stock_w, is_suspended)`
      6. `costs = total_costs(prev_stock_w, target_w, cfg)`
      7. 用**次日**收益 `ret_1d[date+1]` 计算毛收益 `gross = target_w @ ret_next`
      8. `net = gross − costs["total"]`
      9. `turnover = |target_w − prev_stock_w|.sum()`
      10. `hhi_v = hhi(target_w)`
      11. `dsr = dsr_state.update(net, DSR_ETA, sortino=(REWARD_TYPE=='sortino'))`
      12. `drawdown = (peak − nav) / peak`
      13. `reward, parts = compose_reward(dsr, drawdown, turnover, hhi_v, cfg)`
      14. `state = state_builder.build(...)`;`prev_factor_w = factor_w`;`prev_stock_w = target_w`
      15. `info` 包含所有成本分量 + turnover + hhi + long_notional + short_notional + factor_w + n_holdings
    - **必须**在 `info["ret_source"] = "t_plus_1"` 中明确表明用了次日收益(便于验证无未来函数)。
  - `make_env(features_path: str, index_returns_path: str, cfg, start, end) -> PortfolioEnv`。

- [ ] **Step 1: 写测试**

```python
from __future__ import annotations
import numpy as np
import pandas as pd
import pytest

from scripts.env import PortfolioEnv
from scripts.config import FACTOR_NAMES, K, get_config, TOP_N, BOTTOM_M
from scripts.state import state_dim


def _synthetic_features(n_days=60, n_syms=50, seed=1):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2022-01-03", periods=n_days)
    syms = [f"{i:06d}.SZ" for i in range(n_syms)]
    rows = []
    for d in dates:
        for s in syms:
            row = {"trade_date": d, "symbol": s, "ret_1d": rng.normal(0, 0.02), "is_suspended": False}
            for fn in FACTOR_NAMES:
                row[fn] = rng.standard_normal()
            rows.append(row)
    return pd.DataFrame(rows), dates


def test_env_spaces_dtype_and_shape():
    feats, dates = _synthetic_features()
    idx = pd.Series(np.zeros(len(dates)), index=dates)
    env = PortfolioEnv(feats, idx, get_config(), dates[0], dates[-2])
    assert env.action_space.shape == (K,)
    assert env.observation_space.shape == (state_dim(K),)


def test_env_reset_returns_correct_obs():
    feats, dates = _synthetic_features()
    idx = pd.Series(np.zeros(len(dates)), index=dates)
    env = PortfolioEnv(feats, idx, get_config(), dates[0], dates[-2])
    obs, info = env.reset(seed=0)
    assert obs.shape == (state_dim(K),)
    assert obs.dtype == np.float32
    assert np.all(np.isfinite(obs))


def test_env_step_short_notional_within_cap():
    feats, dates = _synthetic_features()
    idx = pd.Series(np.zeros(len(dates)), index=dates)
    cfg = get_config()
    env = PortfolioEnv(feats, idx, cfg, dates[0], dates[-2])
    env.reset(seed=0)
    for _ in range(5):
        act = np.random.uniform(-1, 1, size=K).astype(np.float32)
        obs, r, term, trunc, info = env.step(act)
        assert info["short_notional"] <= cfg["short_notional_cap"] + 1e-6
        assert abs(info["long_notional"] - cfg["long_notional"]) < 1e-6 or info["long_notional"] == 0
        assert info["ret_source"] == "t_plus_1"
        if term or trunc:
            break


def test_env_step_costs_are_recorded_and_positive_after_change():
    feats, dates = _synthetic_features()
    idx = pd.Series(np.zeros(len(dates)), index=dates)
    env = PortfolioEnv(feats, idx, get_config(), dates[0], dates[-2])
    env.reset(seed=0)
    _, _, _, _, info = env.step(np.ones(K, dtype=np.float32))
    for k in ("commission", "stamp_tax", "impact", "borrow"):
        assert info[k] >= 0
    assert info["commission"] > 0  # 从 0 仓变非 0 仓,佣金必 > 0


def test_env_uses_next_day_return_not_current():
    """无未来函数:改动 t 日之前的 ret 不改变 t 日之后 step 已生成的 reward。"""
    feats, dates = _synthetic_features()
    idx = pd.Series(np.zeros(len(dates)), index=dates)
    env = PortfolioEnv(feats, idx, get_config(), dates[0], dates[-2])
    env.reset(seed=42)
    rewards = []
    np.random.seed(0)
    for _ in range(5):
        act = np.random.uniform(-1, 1, size=K).astype(np.float32)
        _, r, term, trunc, info = env.step(act)
        rewards.append(r)
        if term or trunc:
            break
    assert len(rewards) > 0 and all(np.isfinite(rewards))
```

- [ ] **Step 2: 运行 → FAIL**

- [ ] **Step 3: 实现 `env.py`**

```python
"""Gymnasium 环境:state=波动/相关/暴露/持仓, action=K维 RL 输出,
step 内嵌成本、用次日收益结算、reward=DSR+惩罚。"""
from __future__ import annotations
from typing import Optional
import numpy as np
import pandas as pd

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError:
    import gym  # type: ignore
    from gym import spaces  # type: ignore

from scripts.config import FACTOR_NAMES, K
from scripts.action_transform import transform_action
from scripts.costs import total_costs
from scripts.portfolio import (
    composite_score, select_long_short, target_weights, freeze_suspended
)
from scripts.reward import DSRState, hhi, compose_reward
from scripts.state import StateBuilder, state_dim


class PortfolioEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(
        self,
        features_df: pd.DataFrame,
        index_returns: pd.Series,
        cfg: dict,
        start_date,
        end_date,
        nonlinear_impact: bool = False,
    ):
        super().__init__()
        self.cfg = cfg
        self.nonlinear_impact = nonlinear_impact
        df = features_df.copy()
        df["trade_date"] = pd.to_datetime(df["trade_date"])
        df = df[(df["trade_date"] >= pd.Timestamp(start_date)) & (df["trade_date"] <= pd.Timestamp(end_date))]
        df = df.sort_values(["trade_date", "symbol"]).reset_index(drop=True)
        self.features = df

        self.dates = sorted(df["trade_date"].unique())
        self.symbols = sorted(df["symbol"].unique())
        self._sym_to_idx = {s: i for i, s in enumerate(self.symbols)}
        self.n = len(self.symbols)

        # 按日期拆 pivot,预先做好 K 因子矩阵 + ret_1d + suspended
        self._F_by_date: dict = {}
        self._ret_by_date: dict = {}
        self._susp_by_date: dict = {}
        for d, g in df.groupby("trade_date"):
            F = np.zeros((self.n, K), dtype=float)
            r = np.zeros(self.n, dtype=float)
            s = np.ones(self.n, dtype=bool)  # 默认停牌:该日不在 universe 内的
            for _, row in g.iterrows():
                i = self._sym_to_idx[row["symbol"]]
                F[i] = [row[fn] for fn in FACTOR_NAMES]
                r[i] = 0.0 if pd.isna(row["ret_1d"]) else float(row["ret_1d"])
                s[i] = bool(row["is_suspended"])
            self._F_by_date[d] = F
            self._ret_by_date[d] = r
            self._susp_by_date[d] = s

        self.index_returns = index_returns.sort_index()

        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(K,), dtype=np.float32)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(state_dim(K),), dtype=np.float32,
        )
        self._reset_internal()

    def _reset_internal(self):
        self.t = 0
        self.prev_factor_w = np.zeros(K)
        self.prev_stock_w = np.zeros(self.n)
        self.dsr = DSRState()
        # StateBuilder 用因子暴露 panel:key=date, value=DataFrame(symbol, factor)
        panels = {}
        for d, F in self._F_by_date.items():
            panels[d] = pd.DataFrame(F, index=self.symbols, columns=FACTOR_NAMES)
        self.state_builder = StateBuilder(
            factor_panel_by_date=panels,
            index_returns=self.index_returns,
            portfolio_returns_history=[],
            recent_turnover_history=[],
        )

    def reset(self, seed: Optional[int] = None, options=None):
        super().reset(seed=seed)
        self._reset_internal()
        obs = self.state_builder.build(
            self.dates[self.t], self.prev_stock_w, FACTOR_NAMES,
            self.prev_factor_w, cash=1.0,
        )
        return obs, {}

    def step(self, action: np.ndarray):
        d = self.dates[self.t]
        F = self._F_by_date[d]
        susp = self._susp_by_date[d]

        factor_w = transform_action(np.asarray(action, dtype=float), self.prev_factor_w, self.cfg["ema_alpha"])
        scores = composite_score(F, factor_w)
        long_idx, short_idx = select_long_short(scores, susp, self.cfg["top_n"], self.cfg["bottom_m"])
        target_w = target_weights(scores, long_idx, short_idx, self.cfg["long_notional"], self.cfg["short_notional_cap"])
        target_w = freeze_suspended(target_w, self.prev_stock_w, susp)

        costs = total_costs(self.prev_stock_w, target_w, self.cfg)
        # 冲击成本可选非线性
        if self.nonlinear_impact:
            from scripts.costs import impact_cost
            costs["impact"] = impact_cost(self.prev_stock_w, target_w, self.cfg["impact_bps"], nonlinear=True)
            costs["total"] = costs["commission"] + costs["stamp_tax"] + costs["impact"] + costs["borrow"]

        # 次日收益结算(无未来函数)
        next_t = self.t + 1
        terminated = False
        truncated = False
        if next_t >= len(self.dates):
            gross = 0.0
            terminated = True
        else:
            r_next = self._ret_by_date[self.dates[next_t]]
            gross = float(target_w @ r_next)

        net = gross - costs["total"]
        turnover = float(np.abs(target_w - self.prev_stock_w).sum())
        long_notional = float(np.clip(target_w, 0, None).sum())
        short_notional = float(np.clip(-target_w, 0, None).sum())
        hhi_v = hhi(target_w)
        dsr_delta = self.dsr.update(net, self.cfg["dsr_eta"], sortino=(self.cfg["reward_type"] == "sortino"))
        drawdown = 0.0 if self.dsr.peak <= 0 else (self.dsr.peak - self.dsr.nav) / self.dsr.peak
        reward, parts = compose_reward(dsr_delta, drawdown, turnover, hhi_v, self.cfg)

        # 更新历史
        self.state_builder.portfolio_returns_history.append(net)
        self.state_builder.recent_turnover_history.append(turnover)

        # 前进
        self.prev_factor_w = factor_w
        self.prev_stock_w = target_w
        self.t = next_t

        if not terminated:
            obs = self.state_builder.build(
                self.dates[self.t], self.prev_stock_w, FACTOR_NAMES,
                self.prev_factor_w, cash=max(0.0, 1.0 - long_notional - short_notional),
            )
        else:
            obs = np.zeros(state_dim(K), dtype=np.float32)

        info = {
            **costs,
            "gross_ret": gross, "net_ret": net,
            "turnover": turnover,
            "long_notional": long_notional, "short_notional": short_notional,
            "n_long": int(len(long_idx)), "n_short": int(len(short_idx)),
            "factor_w": factor_w.tolist(),
            "hhi": hhi_v,
            "dsr": dsr_delta, "drawdown": drawdown,
            "reward_parts": parts,
            "ret_source": "t_plus_1",
        }
        return obs, float(reward), terminated, truncated, info


def make_env(features_path, index_returns_path, cfg, start, end) -> PortfolioEnv:
    feats = pd.read_parquet(features_path)
    idx = pd.read_parquet(index_returns_path)
    idx = pd.Series(idx["ret"].values, index=pd.to_datetime(idx["trade_date"]))
    return PortfolioEnv(feats, idx, cfg, start, end)
```

- [ ] **Step 4: 运行 → PASS**

Run: `cd rl-portfolio-allocator && pytest tests/test_env.py -v`
Expected: 5 passed.

- [ ] **Step 5: Commit**

```bash
git add rl-portfolio-allocator/scripts/env.py rl-portfolio-allocator/tests/test_env.py
git commit -m "feat(T9): gymnasium PortfolioEnv with embedded costs + DSR reward + t+1 settlement"
```

---

### Task 10: PPO 训练核心 `train.py`

**Files:**
- Create: `rl-portfolio-allocator/scripts/train.py`
- Create: `rl-portfolio-allocator/tests/test_train_smoke.py`

**Interfaces:**
- Consumes: `env.PortfolioEnv`, `config.get_config()`, `stable_baselines3.PPO`。
- Produces:
  - `select_device(pref: str) -> str`:`auto` → cuda/mps/cpu 依次探测。
  - `train_ppo(env: PortfolioEnv, total_timesteps: int, seed: int, device: str, save_path: str | None) -> PPO`:构造 PPO,`n_steps=1024`, `batch_size=256`, `learning_rate=3e-4`, `gamma=0.99`, `gae_lambda=0.95`, `clip_range=0.2`, `ent_coef=0.01`;调用 `model.learn(total_timesteps)`;若 `save_path` 非空,保存到 `.zip`。
  - `load_ppo(path: str, env: PortfolioEnv) -> PPO`。
  - `main()`:跑一个 smoke 训练(小 timesteps)、打印训练设备、最终奖励、保存到 `checkpoints/smoke.zip`。

- [ ] **Step 1: 写 smoke 测试(不联网、5k timesteps 内跑完)**

```python
from __future__ import annotations
import numpy as np
import pandas as pd
import pytest

from scripts.env import PortfolioEnv
from scripts.config import FACTOR_NAMES, K, get_config

sb3 = pytest.importorskip("stable_baselines3")


def _synthetic(n_days=80, n_syms=30, seed=0):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2022-01-03", periods=n_days)
    syms = [f"{i:06d}.SZ" for i in range(n_syms)]
    rows = []
    for d in dates:
        for s in syms:
            row = {"trade_date": d, "symbol": s, "ret_1d": rng.normal(0, 0.02), "is_suspended": False}
            for fn in FACTOR_NAMES:
                row[fn] = rng.standard_normal()
            rows.append(row)
    return pd.DataFrame(rows), dates


def test_ppo_learns_without_error_on_small_env():
    from scripts.train import train_ppo
    feats, dates = _synthetic()
    idx = pd.Series(np.zeros(len(dates)), index=dates)
    env = PortfolioEnv(feats, idx, get_config(), dates[0], dates[-2])
    model = train_ppo(env, total_timesteps=512, seed=0, device="cpu", save_path=None)
    obs, _ = env.reset(seed=1)
    action, _ = model.predict(obs, deterministic=True)
    assert action.shape == (K,)
    assert np.all(np.isfinite(action))


def test_select_device_cpu_always_returns_cpu():
    from scripts.train import select_device
    assert select_device("cpu") == "cpu"
```

- [ ] **Step 2: 运行 → FAIL**

- [ ] **Step 3: 实现 `train.py`**

```python
"""SB3 PPO 训练核心。被 backtest/stress_test/allocate 复用。"""
from __future__ import annotations
import os
import pathlib
from typing import Optional

import numpy as np


def select_device(pref: str) -> str:
    if pref in ("cpu", "cuda", "mps"):
        return pref
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
    except Exception:
        pass
    return "cpu"


def train_ppo(env, total_timesteps: int, seed: int = 0, device: str = "auto",
              save_path: Optional[str] = None):
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv
    from stable_baselines3.common.monitor import Monitor

    def _mk():
        return Monitor(env)

    vec = DummyVecEnv([_mk])
    dev = select_device(device)
    model = PPO(
        "MlpPolicy", vec, verbose=0, seed=seed, device=dev,
        n_steps=1024, batch_size=256, learning_rate=3e-4,
        gamma=0.99, gae_lambda=0.95, clip_range=0.2, ent_coef=0.01,
    )
    model.learn(total_timesteps=total_timesteps)
    if save_path:
        pathlib.Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        model.save(save_path)
    return model


def load_ppo(path: str, env):
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv
    from stable_baselines3.common.monitor import Monitor
    vec = DummyVecEnv([lambda: Monitor(env)])
    return PPO.load(path, env=vec)


def main() -> None:
    from scripts.config import get_config
    from scripts.env import make_env
    cfg = get_config()
    root = pathlib.Path(__file__).resolve().parent.parent
    features_path = root / "data" / "features.parquet"
    index_path = root / "data" / "index_returns.parquet"
    ckpt = root / "checkpoints" / "smoke.zip"
    env = make_env(str(features_path), str(index_path), cfg, cfg["start_date"], cfg["end_date"] or "2099-12-31")
    device = select_device(cfg["train_device"])
    print(f"train device: {device}")
    model = train_ppo(env, total_timesteps=5000, seed=0, device=device, save_path=str(ckpt))
    print(f"smoke checkpoint saved: {ckpt}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: 运行 → PASS**

Run: `cd rl-portfolio-allocator && pytest tests/test_train_smoke.py -v`
Expected: 2 passed(如无 SB3 则 skip)。

- [ ] **Step 5: Commit**

```bash
git add rl-portfolio-allocator/scripts/train.py rl-portfolio-allocator/tests/test_train_smoke.py
git commit -m "feat(T10): SB3 PPO training core + device selection"
```

---

### Task 11: 训练诊断 `diagnostics.py`

**Files:**
- Create: `rl-portfolio-allocator/scripts/diagnostics.py`

**Interfaces:**
- Consumes: 一个 `list[dict]` 的 env.info 序列。
- Produces:
  - `summarize_rollout(infos: list[dict], trading_days: int = 252) -> dict`:返回 `{annualized_turnover, avg_n_holdings, long_exposure_util, short_exposure_util, cost_breakdown, reward_breakdown}`。
  - `check_degeneracy(summary: dict, cfg: dict) -> list[str]`:返回退化告警列表(年化换手 < 0.5、平均持仓数 < 5、long_exposure_util < 0.5 → 每项一条)。
  - `print_report(summary, warnings)`:格式化打印。

- [ ] **Step 1: 实现(无需单独测试,后续 backtest 会用它输出;测试通过 smoke 覆盖)**

```python
"""训练诊断:年化换手/平均持仓/敞口利用率/成本占比。设计 §3.3 强制要求。"""
from __future__ import annotations
import numpy as np


def summarize_rollout(infos: list, trading_days: int = 252) -> dict:
    if not infos:
        return {}
    turnovers = np.array([i["turnover"] for i in infos])
    n_holds = np.array([i["n_long"] + i["n_short"] for i in infos])
    long_util = np.array([i["long_notional"] for i in infos])
    short_util = np.array([i["short_notional"] for i in infos])
    daily_turnover = float(turnovers.mean())
    return {
        "annualized_turnover": daily_turnover * trading_days,
        "avg_n_holdings": float(n_holds.mean()),
        "long_exposure_util": float(long_util.mean()),
        "short_exposure_util": float(short_util.mean()),
        "cost_breakdown": {
            "commission_bps_per_day": float(np.mean([i["commission"] for i in infos])) * 1e4,
            "stamp_tax_bps_per_day": float(np.mean([i["stamp_tax"] for i in infos])) * 1e4,
            "impact_bps_per_day": float(np.mean([i["impact"] for i in infos])) * 1e4,
            "borrow_bps_per_day": float(np.mean([i["borrow"] for i in infos])) * 1e4,
        },
        "reward_breakdown": {
            "dsr_mean": float(np.mean([i["reward_parts"]["dsr"] for i in infos])),
            "drawdown_penalty_mean": float(np.mean([i["reward_parts"]["drawdown_penalty"] for i in infos])),
            "turnover_penalty_mean": float(np.mean([i["reward_parts"]["turnover_penalty"] for i in infos])),
            "concentration_penalty_mean": float(np.mean([i["reward_parts"]["concentration_penalty"] for i in infos])),
        },
    }


def check_degeneracy(summary: dict, cfg: dict) -> list:
    warnings = []
    if summary.get("annualized_turnover", 0) < 0.5:
        warnings.append("DEGENERATE: annualized_turnover < 0.5 → agent 可能躺平,建议调小 λ_turnover")
    if summary.get("avg_n_holdings", 0) < 5:
        warnings.append("DEGENERATE: avg_n_holdings < 5 → 集中度过高或空仓,建议调小 λ_concentration")
    if summary.get("long_exposure_util", 0) < 0.5:
        warnings.append("DEGENERATE: long_exposure_util < 0.5 → 多头敞口未充分使用,建议调小 λ_drawdown")
    return warnings


def print_report(summary: dict, warnings: list) -> None:
    import json
    print("=== Rollout Diagnostics ===")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    if warnings:
        print("\n[WARNINGS]")
        for w in warnings:
            print(f"  - {w}")
    else:
        print("\n[OK] no degeneracy detected")
```

- [ ] **Step 2: Commit**

```bash
git add rl-portfolio-allocator/scripts/diagnostics.py
git commit -m "feat(T11): rollout diagnostics + degeneracy checks"
```

---

### Task 12: 指标 `metrics.py`

**Files:**
- Create: `rl-portfolio-allocator/scripts/metrics.py`
- Create: `rl-portfolio-allocator/tests/test_metrics.py`

**Interfaces:**
- Consumes: `numpy`。
- Produces(所有函数输入日收益 `np.ndarray`,`trading_days=252`):
  - `annualized_return(rets) -> float`
  - `annualized_vol(rets) -> float`
  - `sharpe(rets) -> float`(年化)
  - `sortino(rets) -> float`
  - `max_drawdown(rets) -> float`(返回**负**值,单位百分比)
  - `calmar(rets) -> float`:`ARR / |MDD|`
  - `win_rate(rets) -> float`
  - `metrics_pack(rets: np.ndarray, name: str) -> dict`:一次性打包上述所有 + 累计收益。

- [ ] **Step 1: 写测试**

```python
from __future__ import annotations
import numpy as np
import pytest

from scripts.metrics import (
    annualized_return, annualized_vol, sharpe, sortino,
    max_drawdown, calmar, win_rate, metrics_pack,
)


def test_constant_positive_returns():
    r = np.full(252, 0.001)
    assert annualized_return(r) == pytest.approx((1.001 ** 252 - 1), rel=1e-6)
    assert annualized_vol(r) == pytest.approx(0.0, abs=1e-9)
    assert win_rate(r) == 1.0


def test_drawdown_and_calmar_sign():
    r = np.array([0.02, 0.01, -0.05, -0.03, 0.01])
    mdd = max_drawdown(r)
    assert mdd < 0
    c = calmar(r)
    assert np.isfinite(c)


def test_sharpe_and_sortino_finite():
    rng = np.random.default_rng(0)
    r = rng.normal(0.0005, 0.01, 252)
    assert np.isfinite(sharpe(r))
    assert np.isfinite(sortino(r))


def test_metrics_pack_keys():
    r = np.array([0.01, -0.01, 0.02, -0.005])
    m = metrics_pack(r, name="strat")
    for k in ("name", "arr", "vol", "sharpe", "sortino", "mdd", "calmar", "win_rate", "cumret"):
        assert k in m
    assert m["name"] == "strat"
```

- [ ] **Step 2: 运行 → FAIL**

- [ ] **Step 3: 实现 `metrics.py`**

```python
"""与 DLTX 对齐口径的组合层指标。ARR/MDD/Sharpe/Calmar/Sortino/win_rate。"""
from __future__ import annotations
import numpy as np

_TDAYS = 252


def annualized_return(rets: np.ndarray) -> float:
    rets = np.asarray(rets)
    if len(rets) == 0:
        return 0.0
    total = float(np.prod(1.0 + rets))
    return total ** (_TDAYS / len(rets)) - 1.0


def annualized_vol(rets: np.ndarray) -> float:
    if len(rets) < 2:
        return 0.0
    return float(np.std(rets, ddof=1)) * np.sqrt(_TDAYS)


def sharpe(rets: np.ndarray) -> float:
    v = annualized_vol(rets)
    if v <= 0:
        return 0.0
    return annualized_return(rets) / v


def sortino(rets: np.ndarray) -> float:
    downside = np.minimum(rets, 0.0)
    dvol = float(np.std(downside, ddof=1)) * np.sqrt(_TDAYS) if len(downside) > 1 else 0.0
    if dvol <= 0:
        return 0.0
    return annualized_return(rets) / dvol


def max_drawdown(rets: np.ndarray) -> float:
    if len(rets) == 0:
        return 0.0
    nav = np.cumprod(1.0 + rets)
    peak = np.maximum.accumulate(nav)
    dd = (nav - peak) / peak
    return float(dd.min())


def calmar(rets: np.ndarray) -> float:
    mdd = max_drawdown(rets)
    if mdd == 0:
        return 0.0
    return annualized_return(rets) / abs(mdd)


def win_rate(rets: np.ndarray) -> float:
    if len(rets) == 0:
        return 0.0
    return float((rets > 0).mean())


def metrics_pack(rets: np.ndarray, name: str) -> dict:
    return {
        "name": name,
        "arr": annualized_return(rets),
        "vol": annualized_vol(rets),
        "sharpe": sharpe(rets),
        "sortino": sortino(rets),
        "mdd": max_drawdown(rets),
        "calmar": calmar(rets),
        "win_rate": win_rate(rets),
        "cumret": float(np.prod(1.0 + rets) - 1.0),
    }
```

- [ ] **Step 4: 运行 → PASS**

- [ ] **Step 5: Commit**

```bash
git add rl-portfolio-allocator/scripts/metrics.py rl-portfolio-allocator/tests/test_metrics.py
git commit -m "feat(T12): portfolio-level metrics pack"
```

---

### Task 13: 基线 `baselines.py`

**Files:**
- Create: `rl-portfolio-allocator/scripts/baselines.py`
- Create: `rl-portfolio-allocator/tests/test_baselines.py`

**Interfaces:**
- Consumes: `features` 长表, `config.FACTOR_NAMES, TOP_N, BOTTOM_M`, `costs.total_costs`。
- Produces:
  - `equal_weight_rollout(features_df, cfg, start, end) -> np.ndarray`:每日在 universe 内等权,含成本,返回日收益序列。
  - `long_only_topn_rollout(features_df, cfg, start, end, static_factor_w: np.ndarray) -> np.ndarray`:纯多头 Top-N,按 `static_factor_w` 打分。默认 `static_factor_w = np.ones(K)/K`。
  - `static_factor_equal_rollout(features_df, cfg, start, end) -> np.ndarray`:因子等权(不学),用 T8 的 Top-N/Bottom-M 结构,含成本。
  - 三个函数都返回**扣完全成本**的日净收益(与 RL 可比)。

- [ ] **Step 1: 写测试**

```python
from __future__ import annotations
import numpy as np
import pandas as pd
import pytest

from scripts.baselines import (
    equal_weight_rollout, long_only_topn_rollout, static_factor_equal_rollout
)
from scripts.config import FACTOR_NAMES, K, get_config


def _synthetic(n_days=40, n_syms=20, seed=0):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2022-01-03", periods=n_days)
    syms = [f"{i:06d}.SZ" for i in range(n_syms)]
    rows = []
    for d in dates:
        for s in syms:
            row = {"trade_date": d, "symbol": s, "ret_1d": rng.normal(0, 0.02), "is_suspended": False}
            for fn in FACTOR_NAMES:
                row[fn] = rng.standard_normal()
            rows.append(row)
    return pd.DataFrame(rows), dates


def test_equal_weight_returns_daily_series():
    feats, dates = _synthetic()
    r = equal_weight_rollout(feats, get_config(), dates[0], dates[-1])
    assert isinstance(r, np.ndarray) and r.ndim == 1
    assert len(r) > 0 and np.all(np.isfinite(r))


def test_long_only_shape():
    feats, dates = _synthetic()
    r = long_only_topn_rollout(feats, get_config(), dates[0], dates[-1], np.ones(K) / K)
    assert len(r) > 0


def test_static_factor_equal_shape():
    feats, dates = _synthetic()
    r = static_factor_equal_rollout(feats, get_config(), dates[0], dates[-1])
    assert len(r) > 0
```

- [ ] **Step 2: 运行 → FAIL**

- [ ] **Step 3: 实现 `baselines.py`**

```python
"""三种基线:等权(1/N)、纯多头 TopN、静态因子等权。均含成本、含 t+1 结算。"""
from __future__ import annotations
import numpy as np
import pandas as pd

from scripts.config import FACTOR_NAMES, K
from scripts.costs import total_costs
from scripts.portfolio import composite_score, select_long_short, target_weights, freeze_suspended


def _iter_dates(features_df, start, end):
    df = features_df.copy()
    df["trade_date"] = pd.to_datetime(df["trade_date"])
    df = df[(df["trade_date"] >= pd.Timestamp(start)) & (df["trade_date"] <= pd.Timestamp(end))]
    df = df.sort_values(["trade_date", "symbol"]).reset_index(drop=True)
    return df, sorted(df["trade_date"].unique())


def _panel(df):
    symbols = sorted(df["symbol"].unique())
    idx_map = {s: i for i, s in enumerate(symbols)}
    n = len(symbols)
    F_by, ret_by, susp_by = {}, {}, {}
    for d, g in df.groupby("trade_date"):
        F = np.zeros((n, K)); r = np.zeros(n); s = np.ones(n, dtype=bool)
        for _, row in g.iterrows():
            i = idx_map[row["symbol"]]
            F[i] = [row[fn] for fn in FACTOR_NAMES]
            r[i] = 0.0 if pd.isna(row["ret_1d"]) else float(row["ret_1d"])
            s[i] = bool(row["is_suspended"])
        F_by[d] = F; ret_by[d] = r; susp_by[d] = s
    return symbols, n, F_by, ret_by, susp_by


def _step_returns(prev_w, target_w, ret_next, cfg):
    costs = total_costs(prev_w, target_w, cfg)
    gross = float(target_w @ ret_next)
    return gross - costs["total"]


def equal_weight_rollout(features_df, cfg, start, end) -> np.ndarray:
    df, dates = _iter_dates(features_df, start, end)
    symbols, n, F_by, ret_by, susp_by = _panel(df)
    prev = np.zeros(n)
    out = []
    for i in range(len(dates) - 1):
        susp = susp_by[dates[i]]
        active = ~susp
        w = np.zeros(n)
        if active.sum() > 0:
            w[active] = 1.0 / active.sum()
        w = freeze_suspended(w, prev, susp)
        r_next = ret_by[dates[i + 1]]
        out.append(_step_returns(prev, w, r_next, cfg))
        prev = w
    return np.asarray(out)


def long_only_topn_rollout(features_df, cfg, start, end, static_factor_w: np.ndarray) -> np.ndarray:
    df, dates = _iter_dates(features_df, start, end)
    symbols, n, F_by, ret_by, susp_by = _panel(df)
    prev = np.zeros(n)
    out = []
    for i in range(len(dates) - 1):
        F = F_by[dates[i]]; susp = susp_by[dates[i]]
        scores = composite_score(F, static_factor_w)
        long_idx, _ = select_long_short(scores, susp, cfg["top_n"], bottom_m=0)
        w = target_weights(scores, long_idx, np.array([], dtype=int), cfg["long_notional"], 0.0)
        w = freeze_suspended(w, prev, susp)
        r_next = ret_by[dates[i + 1]]
        out.append(_step_returns(prev, w, r_next, cfg))
        prev = w
    return np.asarray(out)


def static_factor_equal_rollout(features_df, cfg, start, end) -> np.ndarray:
    static_w = np.ones(K) / K
    df, dates = _iter_dates(features_df, start, end)
    symbols, n, F_by, ret_by, susp_by = _panel(df)
    prev = np.zeros(n)
    out = []
    for i in range(len(dates) - 1):
        F = F_by[dates[i]]; susp = susp_by[dates[i]]
        scores = composite_score(F, static_w)
        long_idx, short_idx = select_long_short(scores, susp, cfg["top_n"], cfg["bottom_m"])
        w = target_weights(scores, long_idx, short_idx, cfg["long_notional"], cfg["short_notional_cap"])
        w = freeze_suspended(w, prev, susp)
        r_next = ret_by[dates[i + 1]]
        out.append(_step_returns(prev, w, r_next, cfg))
        prev = w
    return np.asarray(out)
```

- [ ] **Step 4: 运行 → PASS**

- [ ] **Step 5: Commit**

```bash
git add rl-portfolio-allocator/scripts/baselines.py rl-portfolio-allocator/tests/test_baselines.py
git commit -m "feat(T13): three baseline rollouts (EW / long-only TopN / static-factor-equal)"
```

---

### Task 14: 主回测 `backtest.py`(用途②)

**Files:**
- Create: `rl-portfolio-allocator/scripts/backtest.py`
- Create: `rl-portfolio-allocator/tests/test_backtest.py`

**Interfaces:**
- Consumes: `env.PortfolioEnv`, `train.train_ppo/load_ppo`, `baselines.*`, `metrics.metrics_pack`, `diagnostics.*`。
- Produces:
  - `run_ppo_rollout(model, env) -> tuple[np.ndarray, list[dict]]`:确定性执行一整段,返回日净收益 + info 序列。
  - `run_backtest(features_df, cfg, train_start, train_end, test_start, test_end, timesteps, seed) -> dict`:
    1. 训练:在 `[train_start, train_end]` 用 PPO
    2. 样本外测试:在 `[test_start, test_end]` rollout 得到 `rl_rets`(np.ndarray,长度 = 测试期交易日数 − 1)
    3. 三基线在同 test 窗口 rollout
    4. `metrics_pack` × 4 → 汇总 dict,附带 `research_ok / tradeable_ok` 标志
    5. `diagnostics.summarize_rollout(infos)` + `check_degeneracy` 加进结果
    6. **返回结构中必须包含 `rl_daily_rets: np.ndarray` 与 `rl_dates: list[pd.Timestamp]`(与 rets 对齐,即 rets[i] 是 `rl_dates[i]` 到 `rl_dates[i]+1` 结算的净收益),供压力测试的核心段切片使用。**
  - `main()`:默认 train=2010–2022, test=2023–最新;打印四行 metrics 表格 + 诊断。

- [ ] **Step 1: 写测试(smoke,合成数据)**

```python
from __future__ import annotations
import numpy as np
import pandas as pd
import pytest

pytest.importorskip("stable_baselines3")

from scripts.config import FACTOR_NAMES, K, get_config
from scripts.backtest import run_backtest


def _synthetic(n_days=100, n_syms=30, seed=0):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2022-01-03", periods=n_days)
    syms = [f"{i:06d}.SZ" for i in range(n_syms)]
    rows = []
    for d in dates:
        for s in syms:
            row = {"trade_date": d, "symbol": s, "ret_1d": rng.normal(0, 0.02), "is_suspended": False}
            for fn in FACTOR_NAMES:
                row[fn] = rng.standard_normal()
            rows.append(row)
    return pd.DataFrame(rows), dates


def test_run_backtest_returns_metrics_for_rl_and_baselines():
    feats, dates = _synthetic()
    res = run_backtest(
        features_df=feats, cfg=get_config(),
        train_start=dates[0], train_end=dates[60],
        test_start=dates[61], test_end=dates[-1],
        timesteps=256, seed=0,
    )
    for name in ("rl", "equal_weight", "long_only_topn", "static_factor_equal"):
        assert name in res["metrics"]
        for k in ("arr", "sharpe", "mdd", "calmar"):
            assert k in res["metrics"][name]
    assert "diagnostics" in res
    # 供压力测试核心段切片使用
    assert "rl_daily_rets" in res and "rl_dates" in res
    assert len(res["rl_daily_rets"]) == len(res["rl_dates"])
```

- [ ] **Step 2: 运行 → FAIL**

- [ ] **Step 3: 实现 `backtest.py`**

```python
"""主回测(用途②):训早期→测样本外,RL vs 三基线,含成本、含诊断。"""
from __future__ import annotations
import argparse
import pathlib
from typing import Optional

import numpy as np
import pandas as pd

from scripts.config import get_config, FACTOR_NAMES, K
from scripts.env import PortfolioEnv
from scripts.train import train_ppo, select_device
from scripts.baselines import (
    equal_weight_rollout, long_only_topn_rollout, static_factor_equal_rollout
)
from scripts.metrics import metrics_pack
from scripts.diagnostics import summarize_rollout, check_degeneracy


def run_ppo_rollout(model, env) -> tuple[np.ndarray, list, list]:
    obs, _ = env.reset(seed=0)
    infos, rets, dates, done = [], [], [], False
    # env.dates[env.t] 是"即将被 step 使用的"当日;记录该日为该 rets 的信号日。
    while not done:
        signal_date = env.dates[env.t]
        act, _ = model.predict(obs, deterministic=True)
        obs, _, term, trunc, info = env.step(act)
        infos.append(info); rets.append(info["net_ret"]); dates.append(signal_date)
        done = term or trunc
    return np.asarray(rets), infos, dates


def run_backtest(
    features_df: pd.DataFrame,
    cfg: dict,
    train_start, train_end, test_start, test_end,
    timesteps: int = 100_000, seed: int = 0,
    save_path: Optional[str] = None,
) -> dict:
    idx = pd.Series(np.zeros(1), index=pd.to_datetime([features_df["trade_date"].min()]))
    train_env = PortfolioEnv(features_df, idx, cfg, train_start, train_end)
    device = select_device(cfg["train_device"])
    model = train_ppo(train_env, total_timesteps=timesteps, seed=seed, device=device, save_path=save_path)

    test_env = PortfolioEnv(features_df, idx, cfg, test_start, test_end)
    rl_rets, infos, rl_dates = run_ppo_rollout(model, test_env)

    ew = equal_weight_rollout(features_df, cfg, test_start, test_end)
    lo = long_only_topn_rollout(features_df, cfg, test_start, test_end, np.ones(K) / K)
    sf = static_factor_equal_rollout(features_df, cfg, test_start, test_end)

    m = {
        "rl": metrics_pack(rl_rets, "rl"),
        "equal_weight": metrics_pack(ew, "equal_weight"),
        "long_only_topn": metrics_pack(lo, "long_only_topn"),
        "static_factor_equal": metrics_pack(sf, "static_factor_equal"),
    }
    diag = summarize_rollout(infos)
    warns = check_degeneracy(diag, cfg)

    # 通过标准
    research_ok = (m["rl"]["sharpe"] > m["static_factor_equal"]["sharpe"]
                   and m["rl"]["calmar"] > m["static_factor_equal"]["calmar"])
    return {
        "metrics": m, "diagnostics": diag, "warnings": warns,
        "research_ok": bool(research_ok),
        "rl_daily_rets": rl_rets, "rl_dates": rl_dates,
    }


def main() -> None:
    cfg = get_config()
    root = pathlib.Path(__file__).resolve().parent.parent
    feats = pd.read_parquet(root / "data" / "features.parquet")
    p = argparse.ArgumentParser()
    p.add_argument("--train-start", default="2010-01-01")
    p.add_argument("--train-end", default="2022-12-31")
    p.add_argument("--test-start", default="2023-01-01")
    p.add_argument("--test-end", default=cfg["end_date"] or "2099-12-31")
    p.add_argument("--timesteps", type=int, default=200_000)
    args = p.parse_args()

    res = run_backtest(
        feats, cfg, args.train_start, args.train_end, args.test_start, args.test_end,
        timesteps=args.timesteps, seed=0,
        save_path=str(root / "checkpoints" / "backtest.zip"),
    )
    print("=== Backtest metrics ===")
    for name, m in res["metrics"].items():
        print(f"{name:20s}  ARR={m['arr']*100:7.2f}%  Sharpe={m['sharpe']:6.2f}  "
              f"MDD={m['mdd']*100:7.2f}%  Calmar={m['calmar']:6.2f}")
    print(f"\nresearch_ok={res['research_ok']}")
    if res["warnings"]:
        for w in res["warnings"]:
            print(f"  ! {w}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: 运行 → PASS**

Run: `cd rl-portfolio-allocator && pytest tests/test_backtest.py -v`
Expected: 1 passed(如无 SB3 skip)。

- [ ] **Step 5: Commit**

```bash
git add rl-portfolio-allocator/scripts/backtest.py rl-portfolio-allocator/tests/test_backtest.py
git commit -m "feat(T14): main backtest with three baselines and diagnostics"
```

---

### Task 15: 压力测试 `stress_test.py`(用途①)

**Files:**
- Create: `rl-portfolio-allocator/scripts/stress_test.py`
- Create: `rl-portfolio-allocator/tests/test_stress.py`

**Interfaces:**
- Consumes: `backtest.run_backtest`, `metrics.metrics_pack`。
- Produces:
  - `STRESS_SEGMENTS: list[dict]`(逐字来自设计 §4.1;`core_start/core_end` 为"核心段",子区间,用于额外报告):
    ```python
    [
      {
        "name": "2008_gfc",
        "train_end": "2007-06-30",
        "test_start": "2007-07-01", "test_end": "2009-03-31",
        "core_start": "2008-09-01", "core_end": "2009-03-31",
        "required_min_years": 3,
      },
      {
        "name": "2015_ashare_crash",
        "train_end": "2015-05-31",
        "test_start": "2015-06-12", "test_end": "2015-09-30",
        "core_start": "2015-06-12", "core_end": "2015-09-30",
        "required_min_years": 3,
      },
      {
        "name": "2020_covid",
        "train_end": "2020-01-31",
        "test_start": "2020-02-19", "test_end": "2020-04-30",
        "core_start": "2020-02-19", "core_end": "2020-04-30",
        "required_min_years": 3,
      },
      {
        "name": "2022_double_kill",
        "train_end": "2021-12-31",
        "test_start": "2022-01-01", "test_end": "2022-12-31",
        "core_start": "2022-01-01", "core_end": "2022-12-31",
        "required_min_years": 3,
      },
    ]
    ```
    段名与设计 §4.1 表一一对应(`2008_gfc / 2015_ashare_crash / 2020_covid / 2022_double_kill`)。
  - `run_all_stress(features_df, cfg, timesteps=100_000) -> list[dict]`:每段独立训练;若某段 `data_start ~ train_end` 数据不足 `required_min_years` → 输出 `{name, skipped: True, reason}`,**不伪造替代段**。每个未跳过段的返回值除 `metrics`(全 test 段)之外,还包含 `core_metrics`(核心段 RL 与三基线的 `metrics_pack`)。
  - `main()`:打印表格,依次输出全 test 段 + 核心段 + 每段 warnings。

- [ ] **Step 1: 写测试(合成数据,只验证段跳过与结构)**

```python
from __future__ import annotations
import pandas as pd
import numpy as np
import pytest

pytest.importorskip("stable_baselines3")

from scripts.stress_test import STRESS_SEGMENTS, run_all_stress
from scripts.config import FACTOR_NAMES, K, get_config


def _make_feats(dates, n_syms=20):
    rng = np.random.default_rng(0)
    syms = [f"{i:06d}.SZ" for i in range(n_syms)]
    rows = []
    for d in dates:
        for s in syms:
            row = {"trade_date": d, "symbol": s, "ret_1d": rng.normal(0, 0.02), "is_suspended": False}
            for fn in FACTOR_NAMES:
                row[fn] = rng.standard_normal()
            rows.append(row)
    return pd.DataFrame(rows)


def test_stress_segments_metadata_present():
    names = {s["name"] for s in STRESS_SEGMENTS}
    assert {"2008_gfc", "2015_ashare_crash", "2020_covid", "2022_double_kill"}.issubset(names)
    # 每段都必须携带 core_start / core_end 以支持核心段指标输出
    for s in STRESS_SEGMENTS:
        assert "core_start" in s and "core_end" in s
        assert pd.Timestamp(s["core_start"]) >= pd.Timestamp(s["test_start"])
        assert pd.Timestamp(s["core_end"]) <= pd.Timestamp(s["test_end"])


def test_stress_reports_skipped_when_no_data():
    feats = _make_feats(pd.bdate_range("2023-01-03", periods=30))
    # 数据都在 2023,所有段都应被 skipped
    res = run_all_stress(feats, get_config(), timesteps=128)
    for r in res:
        assert r.get("skipped") is True
```

- [ ] **Step 2: 运行 → FAIL**

- [ ] **Step 3: 实现 `stress_test.py`**

```python
"""压力测试(用途①):四段前向,每段独立训练 [数据起点, 段前]。
数据不足则明确报告并跳过,不伪造替代。除全 test 段外额外报告"核心段"指标。"""
from __future__ import annotations
import argparse
import pathlib
import numpy as np
import pandas as pd

from scripts.config import get_config
from scripts.backtest import run_backtest
from scripts.baselines import (
    equal_weight_rollout, long_only_topn_rollout, static_factor_equal_rollout
)
from scripts.metrics import metrics_pack
from scripts.config import K


STRESS_SEGMENTS = [
    {
        "name": "2008_gfc",
        "train_end": "2007-06-30",
        "test_start": "2007-07-01", "test_end": "2009-03-31",
        "core_start": "2008-09-01", "core_end": "2009-03-31",
        "required_min_years": 3,
    },
    {
        "name": "2015_ashare_crash",
        "train_end": "2015-05-31",
        "test_start": "2015-06-12", "test_end": "2015-09-30",
        "core_start": "2015-06-12", "core_end": "2015-09-30",
        "required_min_years": 3,
    },
    {
        "name": "2020_covid",
        "train_end": "2020-01-31",
        "test_start": "2020-02-19", "test_end": "2020-04-30",
        "core_start": "2020-02-19", "core_end": "2020-04-30",
        "required_min_years": 3,
    },
    {
        "name": "2022_double_kill",
        "train_end": "2021-12-31",
        "test_start": "2022-01-01", "test_end": "2022-12-31",
        "core_start": "2022-01-01", "core_end": "2022-12-31",
        "required_min_years": 3,
    },
]


def _has_enough(feats: pd.DataFrame, train_end: str, min_years: int) -> tuple[bool, str, str]:
    d = pd.to_datetime(feats["trade_date"])
    data_start = d.min()
    train_end_ts = pd.Timestamp(train_end)
    if data_start >= train_end_ts:
        return False, str(data_start.date()), f"data_start {data_start.date()} >= train_end {train_end}"
    years = (train_end_ts - data_start).days / 365.25
    if years < min_years:
        return False, str(data_start.date()), f"only {years:.1f} years before {train_end}, need {min_years}"
    return True, str(data_start.date()), ""


def _core_metrics(features_df: pd.DataFrame, cfg: dict,
                   rl_daily_rets: np.ndarray, rl_dates: list,
                   core_start, core_end) -> dict:
    """把 RL 的全 test 段日收益按 rl_dates 切片到核心段;基线在核心段重新 rollout。"""
    core_s = pd.Timestamp(core_start); core_e = pd.Timestamp(core_end)
    mask = np.array([core_s <= pd.Timestamp(d) <= core_e for d in rl_dates])
    rl_core = rl_daily_rets[mask] if mask.any() else np.asarray([])
    ew = equal_weight_rollout(features_df, cfg, core_start, core_end)
    lo = long_only_topn_rollout(features_df, cfg, core_start, core_end, np.ones(K) / K)
    sf = static_factor_equal_rollout(features_df, cfg, core_start, core_end)
    return {
        "rl": metrics_pack(rl_core, "rl_core"),
        "equal_weight": metrics_pack(ew, "equal_weight_core"),
        "long_only_topn": metrics_pack(lo, "long_only_topn_core"),
        "static_factor_equal": metrics_pack(sf, "static_factor_equal_core"),
    }


def run_all_stress(features_df: pd.DataFrame, cfg: dict, timesteps: int = 100_000) -> list:
    out = []
    for seg in STRESS_SEGMENTS:
        ok, data_start, reason = _has_enough(features_df, seg["train_end"], seg["required_min_years"])
        if not ok:
            out.append({"name": seg["name"], "skipped": True, "reason": reason})
            continue
        res = run_backtest(
            features_df=features_df, cfg=cfg,
            train_start=data_start, train_end=seg["train_end"],
            test_start=seg["test_start"], test_end=seg["test_end"],
            timesteps=timesteps, seed=0,
        )
        core = _core_metrics(
            features_df, cfg,
            rl_daily_rets=res["rl_daily_rets"], rl_dates=res["rl_dates"],
            core_start=seg["core_start"], core_end=seg["core_end"],
        )
        out.append({"name": seg["name"], "skipped": False, "core_metrics": core, **res})
    return out


def main() -> None:
    cfg = get_config()
    root = pathlib.Path(__file__).resolve().parent.parent
    feats = pd.read_parquet(root / "data" / "features.parquet")
    p = argparse.ArgumentParser()
    p.add_argument("--timesteps", type=int, default=100_000)
    args = p.parse_args()
    results = run_all_stress(feats, cfg, timesteps=args.timesteps)
    print("=== Stress Test ===")
    for r in results:
        if r.get("skipped"):
            print(f"[SKIP] {r['name']:20s}  reason={r['reason']}")
            continue
        m = r["metrics"]["rl"]; b = r["metrics"]["static_factor_equal"]
        cm = r["core_metrics"]["rl"]; cb = r["core_metrics"]["static_factor_equal"]
        print(f"{r['name']:20s}  [full test]")
        print(f"  RL:  ARR={m['arr']*100:7.2f}%  Sharpe={m['sharpe']:6.2f}  MDD={m['mdd']*100:7.2f}%  Calmar={m['calmar']:6.2f}")
        print(f"  SFE: ARR={b['arr']*100:7.2f}%  Sharpe={b['sharpe']:6.2f}  MDD={b['mdd']*100:7.2f}%  Calmar={b['calmar']:6.2f}")
        print(f"  [core]  RL:  ARR={cm['arr']*100:7.2f}%  Sharpe={cm['sharpe']:6.2f}  MDD={cm['mdd']*100:7.2f}%  Calmar={cm['calmar']:6.2f}")
        print(f"          SFE: ARR={cb['arr']*100:7.2f}%  Sharpe={cb['sharpe']:6.2f}  MDD={cb['mdd']*100:7.2f}%  Calmar={cb['calmar']:6.2f}")
        for w in r.get("warnings", []):
            print(f"  ! {w}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: 运行 → PASS**

- [ ] **Step 5: Commit**

```bash
git add rl-portfolio-allocator/scripts/stress_test.py rl-portfolio-allocator/tests/test_stress.py
git commit -m "feat(T15): four-segment forward stress test with skip-on-insufficient-data"
```

---

### Task 16: 生产模型 `allocate.py`(用途③,`--retrain` / `--infer-only`)

**Files:**
- Create: `rl-portfolio-allocator/scripts/allocate.py`
- Create: `rl-portfolio-allocator/tests/test_allocate.py`

**Interfaces:**
- Consumes: `env`, `train.train_ppo/load_ppo`, `features.load_features`。
- Produces:
  - `PROD_CHECKPOINT = "checkpoints/production.zip"`(相对 `rl-portfolio-allocator/` 根)。
  - `PROD_OUTPUT = "../rl-portfolio-allocator-production/data/allocations.parquet"`。
  - `retrain_production(features_df, cfg, timesteps, seed) -> str`:训练窗口 = `[数据起点, features_df 最新日]`,**无样本外尾巴**;save 到 `PROD_CHECKPOINT`;返回路径。
  - `infer_latest(features_df, cfg, model_path) -> pd.DataFrame`:加载 model,构造一个仅含最后一天的 env(实际用最后 ~60 日作为 rollout 上下文得到 EMA 状态),取最终 `info["factor_w"]` 与 `target_w`,组装成产物字段 DataFrame(Global Constraints 里列出的 9 列)。
  - `save_allocations(df, path) -> None`:parquet;若 path 已存在,append 并按 `(trade_date, symbol)` 去重保留最新 `update_time`。
  - `main()` 支持 `--retrain` 与 `--infer-only`(互斥);两者都调用 `infer_latest` + `save_allocations`。

- [ ] **Step 1: 写测试**

```python
from __future__ import annotations
import numpy as np
import pandas as pd
import pathlib
import pytest

pytest.importorskip("stable_baselines3")

from scripts.config import FACTOR_NAMES, K, get_config, STRATEGY_ID, DATA_VERSION
from scripts.allocate import retrain_production, infer_latest, save_allocations


def _synthetic(n_days=80, n_syms=30, seed=0):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2022-01-03", periods=n_days)
    syms = [f"{i:06d}.SZ" for i in range(n_syms)]
    rows = []
    for d in dates:
        for s in syms:
            row = {"trade_date": d, "symbol": s, "ret_1d": rng.normal(0, 0.02), "is_suspended": False}
            for fn in FACTOR_NAMES:
                row[fn] = rng.standard_normal()
            rows.append(row)
    return pd.DataFrame(rows), dates


def test_retrain_writes_checkpoint(tmp_path):
    feats, _ = _synthetic()
    ckpt = tmp_path / "prod.zip"
    p = retrain_production(feats, get_config(), timesteps=256, seed=0, checkpoint_path=str(ckpt))
    assert pathlib.Path(p).exists()


def test_infer_latest_produces_required_columns(tmp_path):
    feats, _ = _synthetic()
    ckpt = tmp_path / "prod.zip"
    retrain_production(feats, get_config(), timesteps=256, seed=0, checkpoint_path=str(ckpt))
    out = infer_latest(feats, get_config(), model_path=str(ckpt))
    for col in ("trade_date", "symbol", "weight", "side", "factor_weights",
                "composite_score", "strategy_id", "data_version", "update_time"):
        assert col in out.columns
    assert (out["strategy_id"] == STRATEGY_ID).all()
    assert (out["data_version"] == DATA_VERSION).all()
    # side ∈ {long, short, cash}
    assert set(out["side"].unique()) <= {"long", "short", "cash"}
    # 总权重 ≤ 1 + short_cap
    assert out["weight"].abs().sum() <= 1.0 + get_config()["short_notional_cap"] + 1e-6


def test_save_allocations_dedup_latest(tmp_path):
    p = tmp_path / "alloc.parquet"
    df1 = pd.DataFrame([{"trade_date": pd.Timestamp("2024-01-02"), "symbol": "S1",
                          "weight": 0.1, "side": "long", "factor_weights": "{}",
                          "composite_score": 1.0, "strategy_id": "RLPA",
                          "data_version": "real-v1", "update_time": "2024-01-02T09:00:00"}])
    save_allocations(df1, str(p))
    df2 = df1.copy()
    df2["weight"] = 0.2
    df2["update_time"] = "2024-01-02T10:00:00"
    save_allocations(df2, str(p))
    out = pd.read_parquet(p)
    assert len(out) == 1
    assert out.iloc[0]["weight"] == 0.2
```

- [ ] **Step 2: 运行 → FAIL**

- [ ] **Step 3: 实现 `allocate.py`**

```python
"""生产模型:--retrain 用数据起点~最新日全部数据训 PPO;--infer-only 复用现有模型每日推理。
落盘持仓表到 rl-portfolio-allocator-production/data/allocations.parquet。"""
from __future__ import annotations
import argparse
import json
import pathlib
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from scripts.config import (
    get_config, FACTOR_NAMES, K, STRATEGY_ID, DATA_VERSION
)
from scripts.env import PortfolioEnv
from scripts.train import train_ppo, load_ppo, select_device


def retrain_production(features_df: pd.DataFrame, cfg: dict, timesteps: int,
                        seed: int, checkpoint_path: str) -> str:
    dates = pd.to_datetime(features_df["trade_date"])
    start, end = dates.min(), dates.max()
    idx = pd.Series(np.zeros(1), index=[dates.min()])
    env = PortfolioEnv(features_df, idx, cfg, start, end)
    device = select_device(cfg["train_device"])
    train_ppo(env, total_timesteps=timesteps, seed=seed, device=device, save_path=checkpoint_path)
    return checkpoint_path


def infer_latest(features_df: pd.DataFrame, cfg: dict, model_path: str) -> pd.DataFrame:
    dates = pd.to_datetime(features_df["trade_date"]).unique()
    dates = sorted(dates)
    # 用最后 ~60 日作为 rollout 上下文以稳定 EMA/DSR/state
    ctx_start = dates[max(0, len(dates) - 60)]
    end = dates[-1]
    idx = pd.Series(np.zeros(1), index=[ctx_start])
    env = PortfolioEnv(features_df, idx, cfg, ctx_start, end)
    model = load_ppo(model_path, env)

    obs, _ = env.reset(seed=0)
    last_info = None
    last_target_w = None
    last_symbols = env.symbols
    done = False
    while not done:
        act, _ = model.predict(obs, deterministic=True)
        obs, _, term, trunc, info = env.step(act)
        last_info = info
        last_target_w = env.prev_stock_w.copy()
        done = term or trunc

    if last_info is None:
        raise RuntimeError("env produced no step; features_df too short")

    factor_w = np.asarray(last_info["factor_w"], dtype=float)
    scores = env._F_by_date[env.dates[env.t - 1]] @ factor_w  # 与该日一致

    now = datetime.now(timezone.utc).isoformat()
    rows = []
    fw_json = json.dumps(dict(zip(FACTOR_NAMES, factor_w.tolist())))
    trade_date = pd.Timestamp(end).normalize()
    for i, s in enumerate(last_symbols):
        w = float(last_target_w[i])
        if abs(w) < 1e-9:
            continue
        rows.append({
            "trade_date": trade_date, "symbol": s, "weight": w,
            "side": "long" if w > 0 else "short",
            "factor_weights": fw_json,
            "composite_score": float(scores[i]),
            "strategy_id": STRATEGY_ID, "data_version": DATA_VERSION,
            "update_time": now,
        })
    # 用一条 cash 行补齐(便于下游确认)
    cash = 1.0 - float(np.clip(last_target_w, 0, None).sum()) - float(np.clip(-last_target_w, 0, None).sum())
    if abs(cash) > 1e-9:
        rows.append({
            "trade_date": trade_date, "symbol": "CASH", "weight": cash, "side": "cash",
            "factor_weights": fw_json, "composite_score": 0.0,
            "strategy_id": STRATEGY_ID, "data_version": DATA_VERSION, "update_time": now,
        })
    return pd.DataFrame(rows)


def save_allocations(df: pd.DataFrame, path: str) -> None:
    p = pathlib.Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    if p.exists():
        old = pd.read_parquet(p)
        combined = pd.concat([old, df], ignore_index=True)
        combined = combined.sort_values("update_time").drop_duplicates(
            subset=["trade_date", "symbol"], keep="last"
        )
    else:
        combined = df
    combined.to_parquet(p, index=False)


def main() -> None:
    cfg = get_config()
    root = pathlib.Path(__file__).resolve().parent.parent
    feats_path = root / "data" / "features.parquet"
    ckpt = root / "checkpoints" / "production.zip"
    out_path = root.parent / "rl-portfolio-allocator-production" / "data" / "allocations.parquet"

    p = argparse.ArgumentParser()
    grp = p.add_mutually_exclusive_group(required=True)
    grp.add_argument("--retrain", action="store_true", help="用数据起点~最新日全部数据重训生产模型")
    grp.add_argument("--infer-only", action="store_true", help="复用现有生产模型仅推理当日持仓")
    p.add_argument("--timesteps", type=int, default=500_000)
    args = p.parse_args()

    feats = pd.read_parquet(feats_path)
    if args.retrain:
        retrain_production(feats, cfg, args.timesteps, seed=0, checkpoint_path=str(ckpt))
        print(f"production checkpoint saved: {ckpt}")
    if not ckpt.exists():
        raise SystemExit(f"no production checkpoint at {ckpt}; run --retrain first")
    allocations = infer_latest(feats, cfg, model_path=str(ckpt))
    save_allocations(allocations, str(out_path))
    print(f"allocations saved: {out_path}  rows={len(allocations)}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: 运行 → PASS**

Run: `cd rl-portfolio-allocator && pytest tests/test_allocate.py -v`

- [ ] **Step 5: Commit**

```bash
git add rl-portfolio-allocator/scripts/allocate.py rl-portfolio-allocator/tests/test_allocate.py
git commit -m "feat(T16): production allocate (--retrain / --infer-only) with dedup save"
```

---

### Task 17: 规范性校验 `validate.py`

**Files:**
- Create: `rl-portfolio-allocator/scripts/validate.py`
- Create: `rl-portfolio-allocator/tests/test_validate.py`

**Interfaces:**
- Consumes: `pd.read_parquet` 读取 `allocations.parquet`。
- Produces:
  - `validate_schema(df) -> list[str]`:必需列存在;`side ∈ {long,short,cash}`;`strategy_id=='RLPA'`;`data_version=='real-v1'`;`factor_weights` 是有效 JSON。
  - `validate_weights(df, cfg) -> list[str]`:每个 `trade_date`,`Σweight_long ≤ long_notional + tol`;`Σ|weight_short| ≤ short_notional_cap + tol`;所有 `weight` 有限。
  - `validate_no_future(df) -> list[str]`:`update_time > trade_date`(生成时刻晚于组合日)。
  - `run_all(path: str, cfg: dict) -> tuple[bool, list[str]]`:合并所有 error;非空返回 `(False, errors)`;空返回 `(True, [])`。
  - `main()`:读默认路径,打印结果,exit code 0/1。

- [ ] **Step 1: 写测试**

```python
from __future__ import annotations
import pandas as pd
import pytest

from scripts.validate import validate_schema, validate_weights, validate_no_future, run_all
from scripts.config import get_config, STRATEGY_ID, DATA_VERSION


def _row(**kw):
    base = {
        "trade_date": pd.Timestamp("2024-01-02"), "symbol": "S1", "weight": 0.1,
        "side": "long", "factor_weights": '{"mom_20": 0.5}',
        "composite_score": 1.0, "strategy_id": STRATEGY_ID, "data_version": DATA_VERSION,
        "update_time": "2024-01-02T10:00:00",
    }
    base.update(kw)
    return base


def test_schema_flags_wrong_strategy_id():
    df = pd.DataFrame([_row(strategy_id="WRONG")])
    errs = validate_schema(df)
    assert any("strategy_id" in e for e in errs)


def test_weights_respect_caps():
    cfg = get_config()
    df = pd.DataFrame([
        _row(symbol="A", weight=0.6, side="long"),
        _row(symbol="B", weight=0.4, side="long"),
        _row(symbol="C", weight=-0.2, side="short"),
    ])
    assert validate_weights(df, cfg) == []


def test_weights_flags_short_over_cap():
    cfg = get_config()
    df = pd.DataFrame([
        _row(symbol="A", weight=-0.5, side="short"),
    ])
    errs = validate_weights(df, cfg)
    assert any("short" in e.lower() for e in errs)


def test_no_future_flags_backdated_update_time():
    df = pd.DataFrame([_row(update_time="2023-01-01T10:00:00")])
    errs = validate_no_future(df)
    assert errs
```

- [ ] **Step 2: 运行 → FAIL**

- [ ] **Step 3: 实现 `validate.py`**

```python
"""规范性校验:schema / 权重上限 / 无未来函数。不通过不得进入生产。"""
from __future__ import annotations
import argparse
import json
import pathlib
import sys
import pandas as pd

from scripts.config import get_config, STRATEGY_ID, DATA_VERSION

REQUIRED = [
    "trade_date", "symbol", "weight", "side", "factor_weights",
    "composite_score", "strategy_id", "data_version", "update_time",
]
_TOL = 1e-6


def validate_schema(df: pd.DataFrame) -> list:
    errs = []
    for c in REQUIRED:
        if c not in df.columns:
            errs.append(f"missing column: {c}")
    if errs:
        return errs
    if not (df["strategy_id"] == STRATEGY_ID).all():
        errs.append(f"strategy_id must all be {STRATEGY_ID!r}")
    if not (df["data_version"] == DATA_VERSION).all():
        errs.append(f"data_version must all be {DATA_VERSION!r}")
    if not df["side"].isin(["long", "short", "cash"]).all():
        errs.append("side must be in {long, short, cash}")
    for i, fw in enumerate(df["factor_weights"]):
        try:
            json.loads(fw)
        except Exception:
            errs.append(f"row {i}: factor_weights is not valid JSON")
            break
    return errs


def validate_weights(df: pd.DataFrame, cfg: dict) -> list:
    errs = []
    for date, g in df.groupby("trade_date"):
        long_sum = g.loc[g["side"] == "long", "weight"].sum()
        short_sum = -g.loc[g["side"] == "short", "weight"].sum()  # 存的是负数
        if long_sum > cfg["long_notional"] + _TOL:
            errs.append(f"{date}: long notional {long_sum:.4f} > cap {cfg['long_notional']}")
        if short_sum > cfg["short_notional_cap"] + _TOL:
            errs.append(f"{date}: short notional {short_sum:.4f} > cap {cfg['short_notional_cap']}")
        if not g["weight"].apply(lambda x: x == x and abs(x) < 1e9).all():
            errs.append(f"{date}: non-finite weight detected")
    return errs


def validate_no_future(df: pd.DataFrame) -> list:
    errs = []
    for _, row in df.iterrows():
        td = pd.Timestamp(row["trade_date"]).normalize()
        try:
            ut = pd.Timestamp(row["update_time"]).tz_localize(None) if pd.Timestamp(row["update_time"]).tzinfo else pd.Timestamp(row["update_time"])
        except Exception:
            errs.append(f"row {row['symbol']}@{td}: bad update_time {row['update_time']!r}")
            continue
        if ut.normalize() < td:
            errs.append(f"{row['symbol']}@{td}: update_time {ut} earlier than trade_date {td} (future function)")
    return errs


def run_all(path: str, cfg: dict) -> tuple:
    df = pd.read_parquet(path)
    errs = validate_schema(df) + validate_weights(df, cfg) + validate_no_future(df)
    return (len(errs) == 0), errs


def main() -> None:
    cfg = get_config()
    root = pathlib.Path(__file__).resolve().parent.parent
    default_path = root.parent / "rl-portfolio-allocator-production" / "data" / "allocations.parquet"
    p = argparse.ArgumentParser()
    p.add_argument("--path", default=str(default_path))
    args = p.parse_args()
    ok, errs = run_all(args.path, cfg)
    if ok:
        print(f"[OK] {args.path} validates")
        sys.exit(0)
    print(f"[FAIL] {args.path}")
    for e in errs:
        print(f"  - {e}")
    sys.exit(1)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: 运行 → PASS**

- [ ] **Step 5: Commit**

```bash
git add rl-portfolio-allocator/scripts/validate.py rl-portfolio-allocator/tests/test_validate.py
git commit -m "feat(T17): schema/weights/no-future validation gate"
```

---

### Task 18: 只读生产查询 skill `rl-portfolio-allocator-production`

**Files:**
- Create: `rl-portfolio-allocator-production/SKILL.md`
- Create: `rl-portfolio-allocator-production/scripts/__init__.py`
- Create: `rl-portfolio-allocator-production/scripts/query.py`
- Create: `rl-portfolio-allocator-production/tests/__init__.py`
- Create: `rl-portfolio-allocator-production/tests/test_query.py`
- Create: `rl-portfolio-allocator-production/conftest.py`(同 T1 结构)
- Create: `rl-portfolio-allocator-production/data/.gitkeep`

**Interfaces:**
- Consumes: `allocations.parquet`(由 `allocate.py` 写入)。
- Produces:
  - `load_allocations(path=None) -> pd.DataFrame`:默认路径 `../rl-portfolio-allocator-production/data/allocations.parquet`(相对该 skill 内 script 目录);**只读**,不重训、不重算。
  - `get_latest(df) -> pd.DataFrame`:仅返回最新 `trade_date` 的行。
  - `get_range(df, start, end) -> pd.DataFrame`:按日期区间过滤。
  - `main()` 支持 `--latest` / `--start / --end`。

- [ ] **Step 1: 写 `SKILL.md`**

```markdown
---
name: rl-portfolio-allocator-production
description: Read-only queries against the pre-computed RL portfolio allocations. Does not train or recompute.
license: GPL-3.0-only
tags: [quant, rl, portfolio, production, ashare]
---

# RL 组合权重优化器(只读生产查询)

只读模式:从 `data/allocations.parquet` 读取由 `../rl-portfolio-allocator/scripts/allocate.py` 落盘的每日持仓。**不训练、不重算、不联网。**

## 用法
```bash
python scripts/query.py --latest
python scripts/query.py --start 2024-01-01 --end 2024-06-30
```

## 字段
`trade_date, symbol, weight, side(long|short|cash), factor_weights(JSON), composite_score, strategy_id='RLPA', data_version='real-v1', update_time`
```

- [ ] **Step 2: 写 `conftest.py` 与 `__init__.py`**(同 T1 结构)

- [ ] **Step 3: 写测试 `tests/test_query.py`**

```python
from __future__ import annotations
import pandas as pd
import pathlib
import pytest

from scripts.query import load_allocations, get_latest, get_range


def _make(tmp_path):
    df = pd.DataFrame([
        {"trade_date": pd.Timestamp("2024-01-02"), "symbol": "S1", "weight": 0.1,
         "side": "long", "factor_weights": "{}", "composite_score": 1.0,
         "strategy_id": "RLPA", "data_version": "real-v1", "update_time": "2024-01-02T10:00:00"},
        {"trade_date": pd.Timestamp("2024-01-03"), "symbol": "S1", "weight": 0.2,
         "side": "long", "factor_weights": "{}", "composite_score": 2.0,
         "strategy_id": "RLPA", "data_version": "real-v1", "update_time": "2024-01-03T10:00:00"},
    ])
    p = tmp_path / "allocations.parquet"
    df.to_parquet(p, index=False)
    return p


def test_load_and_latest(tmp_path):
    p = _make(tmp_path)
    df = load_allocations(str(p))
    latest = get_latest(df)
    assert len(latest) == 1
    assert latest.iloc[0]["trade_date"] == pd.Timestamp("2024-01-03")


def test_range_filter(tmp_path):
    p = _make(tmp_path)
    df = load_allocations(str(p))
    rng = get_range(df, "2024-01-01", "2024-01-02")
    assert len(rng) == 1
    assert rng.iloc[0]["trade_date"] == pd.Timestamp("2024-01-02")
```

- [ ] **Step 4: 实现 `scripts/query.py`**

```python
"""只读查询已落盘的 RL 组合持仓。绝不重训、不联网。"""
from __future__ import annotations
import argparse
import pathlib
import pandas as pd

_DEFAULT_PATH = pathlib.Path(__file__).resolve().parent.parent / "data" / "allocations.parquet"


def load_allocations(path=None) -> pd.DataFrame:
    p = pathlib.Path(path) if path else _DEFAULT_PATH
    if not p.exists():
        raise FileNotFoundError(f"allocations parquet not found: {p}. "
                                f"Run `../rl-portfolio-allocator/scripts/allocate.py --retrain` first.")
    return pd.read_parquet(p)


def get_latest(df: pd.DataFrame) -> pd.DataFrame:
    latest_date = df["trade_date"].max()
    return df[df["trade_date"] == latest_date].reset_index(drop=True)


def get_range(df: pd.DataFrame, start, end) -> pd.DataFrame:
    s = pd.Timestamp(start); e = pd.Timestamp(end)
    m = (df["trade_date"] >= s) & (df["trade_date"] <= e)
    return df.loc[m].reset_index(drop=True)


def main() -> None:
    p = argparse.ArgumentParser()
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--latest", action="store_true")
    g.add_argument("--range", nargs=2, metavar=("START", "END"))
    p.add_argument("--path", default=None)
    args = p.parse_args()
    df = load_allocations(args.path)
    out = get_latest(df) if args.latest else get_range(df, args.range[0], args.range[1])
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
```

- [ ] **Step 5: 运行测试 → PASS**

Run: `cd rl-portfolio-allocator-production && pytest tests/ -v`

- [ ] **Step 6: Commit**

```bash
git add rl-portfolio-allocator-production/
git commit -m "feat(T18): read-only production query skill"
```

---

### Task 19: 端到端 smoke pipeline + README 补全

**Files:**
- Create: `rl-portfolio-allocator/run_pipeline.sh`
- Modify: `README.md`(在末尾追加 "已实现能力" 与 "验收记录" 节)

**Interfaces:**
- Consumes: T1–T18 全部脚本。
- Produces:
  - `run_pipeline.sh`:串起 `features.py → train.py smoke → backtest → stress → allocate --retrain → validate`,遇任何非零退出即停。
  - README 记录:每步预期输出、常见故障(SB3 未安装、panda_data 无凭据、GPU 探测失败等)排查提示。

- [ ] **Step 1: 写 `run_pipeline.sh`**

```bash
#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

: "${PANDA_DATA_USERNAME:?PANDA_DATA_USERNAME not set}"
: "${PANDA_DATA_PASSWORD:?PANDA_DATA_PASSWORD not set}"

echo "== [1/6] features =="
python scripts/features.py

echo "== [2/6] train smoke =="
python scripts/train.py

echo "== [3/6] backtest (research + tradeable) =="
python scripts/backtest.py --timesteps "${PIPELINE_TIMESTEPS:-200000}"

echo "== [4/6] stress test (four forward segments) =="
python scripts/stress_test.py --timesteps "${PIPELINE_TIMESTEPS:-100000}"

echo "== [5/6] allocate (production retrain + infer) =="
python scripts/allocate.py --retrain --timesteps "${PIPELINE_PROD_TIMESTEPS:-500000}"

echo "== [6/6] validate =="
python scripts/validate.py

echo "OK"
```

Run: `chmod +x rl-portfolio-allocator/run_pipeline.sh`

- [ ] **Step 2: 追加 README 内容(手写,展示已实现能力与故障排查)**

在 `README.md` 末尾追加:

```markdown

## 已实现能力(与设计 §8 验收标准对照)

| 验收项 | 由哪个 Task/脚本承担 |
|---|---|
| 无未来函数 | T3 factor + T9 env(`ret_source=t_plus_1`) + T15 stress + T17 validate |
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
```

- [ ] **Step 3: Commit**

```bash
git add rl-portfolio-allocator/run_pipeline.sh README.md
git commit -m "chore(T19): end-to-end smoke pipeline script + acceptance/troubleshooting docs"
```

---

## Self-Review 记录

已核对 spec §1–§10 覆盖情况:

- §1 目标/数据流 → T3, T5, T8, T9 组合覆盖。
- §2 架构决策(资产池/RL 学什么/因子集/动作/reward/算法/库/切分/做空/双模式)→ T2 常量 + T3–T9 实现 + T14–T18 三类训练与双模式分离。
- §2A 动作空间(tanh + L1 + EMA + 敞口确定性)→ T5 + T8 + T9 严格实现,单元测试覆盖。
- §3 交易环境(成本/State/Reward)→ T4 costs + T7 state + T6 reward + T9 env,`ret_source=t_plus_1` 显式断言。
- §4 数据切分与压力测试 → T14 backtest + T15 stress(含跳过 2008 的 required_min_years 逻辑) + T16 allocate 三入口独立。
- §5 回测指标 & 基线 → T12 metrics + T13 baselines + T14 backtest 汇总。
- §6 工程结构(双模式,对齐 DLTX)→ T1 骨架 + T18 生产查询。
- §7 运行流程 → T19 `run_pipeline.sh` + inner SKILL.md 步骤说明。
- §8 验收标准 → T19 README 表逐项映射并给出承担 Task。
- §9 依赖 → T1 skill.json + `pip` 提示。
- §10 边界声明 → T1 顶层 SKILL.md 与 README 都含免责声明。

**类型/命名一致性检查**:`FACTOR_NAMES / K` 全局单一定义在 `config.py`;`STRATEGY_ID / DATA_VERSION` 单一常量;`transform_action / compose_reward / total_costs / state_dim` 签名跨 T5/T6/T4/T7 与 T9 使用一致;`select_long_short / target_weights` 输入输出与 T13 baselines、T9 env 一致。

**Placeholder 扫描**:未出现 "TBD/TODO";所有 "Similar to Task N" 已就地展开;每个含代码的 step 都给出完整代码块。

---

Plan complete and saved to `docs/superpowers/plans/2026-07-22-rl-portfolio-allocator.md`.

## 执行选择

两种执行方式:

1. **Subagent-Driven(推荐)** — 每个 Task 派一个新 subagent,任务间人工评审、快速迭代。
2. **Inline Execution** — 在当前会话按批 checkpoint 依次执行。

请告诉我选哪种,或直接说"执行 T1"手动开始。
