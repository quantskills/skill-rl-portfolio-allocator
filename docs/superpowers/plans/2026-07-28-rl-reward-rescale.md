# RL 奖励重构 + 训练量修复 实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 让 PPO 优化收益而非躺平避险——奖励里加入直接净收益项、降低回撤/换手惩罚量级，并把生产训练步数从 5000 提到 200k。

**Architecture:** 改 `compose_reward` 增加 `w_ret*net_ret` 主收益项并从 `env.step` 传入 `net`；`config.py` 下调两个 lambda 并新增 `reward_ret_weight`；`train_ppo` 加可选 `EvalCallback` 早停（向后兼容）；`train.py main()` + `run_pipeline.sh` 走 200k。

**Tech Stack:** Python, NumPy, Pandas, Gymnasium, stable-baselines3 (PPO), pytest。

## Global Constraints

- 工作目录: `/Users/dmiwu/work/PythonProject/PandaAIQuant/claude_code_skills/skill-rl-portfolio-allocator`
- 所有 python/pytest 命令在 `rl-portfolio-allocator/` 子目录下运行（`conftest.py` 已把该目录加入 sys.path，import 形如 `from scripts.reward import ...`）。
- 硬约束逻辑（`constraint_penalty`、notional 缩放）不得改动。
- 不改因子集、选股逻辑、成本模型。
- `train_ppo` 不传 `eval_env` 时行为必须与现状完全一致（向后兼容）。
- `reward_ret_weight` 默认 `1.0`；`λ_drawdown` 0.05→0.005；`λ_turnover` 0.02→0.002；`λ_concentration` 保持 0.02。

---

### Task 1: config 新增 reward_ret_weight 并下调 lambda

**Files:**
- Modify: `rl-portfolio-allocator/scripts/config.py:29-31`（lambda 常量）、`:52-54`（get_config 返回）、新增 `reward_ret_weight` 读取

**Interfaces:**
- Produces: `get_config()` 返回 dict 新增键 `"reward_ret_weight": float`（默认 1.0）；`"lambda_drawdown"=0.005`、`"lambda_turnover"=0.002`。

- [ ] **Step 1: 写失败测试**

创建 `rl-portfolio-allocator/tests/test_config.py`：

```python
from scripts.config import get_config


def test_lambdas_rescaled():
    cfg = get_config()
    assert cfg["lambda_drawdown"] == 0.005
    assert cfg["lambda_turnover"] == 0.002
    assert cfg["lambda_concentration"] == 0.02


def test_reward_ret_weight_default():
    cfg = get_config()
    assert cfg["reward_ret_weight"] == 1.0
```

- [ ] **Step 2: 运行测试确认失败**

Run: `cd rl-portfolio-allocator && python -m pytest tests/test_config.py -q`
Expected: FAIL（`lambda_drawdown` 仍为 0.05；`reward_ret_weight` KeyError）

- [ ] **Step 3: 改 config.py**

`scripts/config.py:29-31` 改为：

```python
LAMBDA_DRAWDOWN: float = 0.005
LAMBDA_TURNOVER: float = 0.002
LAMBDA_CONCENTRATION: float = 0.02
```

在 `get_config()` 返回 dict 中（`"reward_type"` 行附近）新增一行：

```python
        "reward_ret_weight": float(os.environ.get("REWARD_RET_WEIGHT", "1.0")),
```

- [ ] **Step 4: 运行测试确认通过**

Run: `cd rl-portfolio-allocator && python -m pytest tests/test_config.py -q`
Expected: PASS（2 passed）

- [ ] **Step 5: 提交**

```bash
git add rl-portfolio-allocator/scripts/config.py rl-portfolio-allocator/tests/test_config.py
git commit -m "feat: rescale reward lambdas, add reward_ret_weight

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 2: compose_reward 加入 net_ret 主收益项

**Files:**
- Modify: `rl-portfolio-allocator/scripts/reward.py:40-63`
- Test: `rl-portfolio-allocator/tests/test_reward.py`（新增）

**Interfaces:**
- Consumes: `cfg["reward_ret_weight"]`（Task 1）、`cfg["lambda_drawdown"]`、`cfg["lambda_turnover"]`、`cfg["lambda_concentration"]`。
- Produces: `compose_reward(dsr_delta, drawdown, turnover, hhi_val, cfg, net_ret, long_notional=0.0, short_notional=0.0, long_cap=1.0, short_cap=0.3) -> tuple[float, dict]`；`parts` 新增 `"ret_term"`；`total` = 各项之和。

- [ ] **Step 1: 写失败测试**

创建 `rl-portfolio-allocator/tests/test_reward.py`：

```python
from scripts.config import get_config
from scripts.reward import compose_reward


def _parts(net_ret, drawdown=0.1, turnover=0.8, hhi_val=0.05):
    cfg = get_config()
    total, parts = compose_reward(
        dsr_delta=0.0018, drawdown=drawdown, turnover=turnover,
        hhi_val=hhi_val, cfg=cfg, net_ret=net_ret,
        long_notional=1.0, short_notional=0.3, long_cap=1.0, short_cap=0.3,
    )
    return total, parts


def test_reward_parts_complete():
    total, parts = _parts(0.005)
    for k in ("ret_term", "dsr", "drawdown_penalty", "turnover_penalty",
              "concentration_penalty", "constraint_penalty", "total"):
        assert k in parts, f"missing {k}"
    s = (parts["ret_term"] + parts["dsr"] + parts["drawdown_penalty"]
         + parts["turnover_penalty"] + parts["concentration_penalty"]
         + parts["constraint_penalty"])
    assert abs(s - parts["total"]) < 1e-12
    assert abs(total - parts["total"]) < 1e-12


def test_return_term_not_dominated_by_penalties():
    # 典型单步:净收益 +0.5%,换手 0.8,回撤 10%
    total, parts = _parts(0.005)
    penalties = abs(parts["drawdown_penalty"] + parts["turnover_penalty"]
                    + parts["concentration_penalty"])
    assert abs(parts["ret_term"]) > 0
    # 修复前 penalty/return ≈ 26x;要求降到 ≤ 5x
    assert penalties / abs(parts["ret_term"]) <= 5.0, (
        f"penalties {penalties} dominate ret_term {parts['ret_term']}")


def test_positive_return_raises_reward():
    lo, _ = _parts(0.002)
    hi, _ = _parts(0.010)
    assert hi > lo
```

- [ ] **Step 2: 运行测试确认失败**

Run: `cd rl-portfolio-allocator && python -m pytest tests/test_reward.py -q`
Expected: FAIL（`compose_reward()` 缺 `net_ret` 参数 → TypeError）

- [ ] **Step 3: 改 reward.py**

`scripts/reward.py` 的 `compose_reward` 签名改为（在 `cfg` 后新增必填 `net_ret`）：

```python
def compose_reward(
    dsr_delta: float, drawdown: float, turnover: float, hhi_val: float, cfg: dict,
    net_ret: float,
    long_notional: float = 0.0, short_notional: float = 0.0,
    long_cap: float = 1.0, short_cap: float = 0.3,
) -> tuple[float, dict]:
    ret_term = cfg["reward_ret_weight"] * net_ret
    dd_pen = -cfg["lambda_drawdown"] * max(0.0, drawdown)
    to_pen = -cfg["lambda_turnover"] * turnover
    conc_pen = -cfg["lambda_concentration"] * hhi_val

    constraint_pen = 0.0
    if long_notional > long_cap * 1.01:
        constraint_pen -= 1.0 * (long_notional - long_cap)
    if short_notional > short_cap * 1.01:
        constraint_pen -= 1.0 * (short_notional - short_cap)

    total = ret_term + dsr_delta + dd_pen + to_pen + conc_pen + constraint_pen
    return total, {
        "ret_term": ret_term,
        "dsr": dsr_delta,
        "drawdown_penalty": dd_pen,
        "turnover_penalty": to_pen,
        "concentration_penalty": conc_pen,
        "constraint_penalty": constraint_pen,
        "total": total,
    }
```

- [ ] **Step 4: 运行测试确认通过**

Run: `cd rl-portfolio-allocator && python -m pytest tests/test_reward.py -q`
Expected: PASS（3 passed）

- [ ] **Step 5: 提交**

```bash
git add rl-portfolio-allocator/scripts/reward.py rl-portfolio-allocator/tests/test_reward.py
git commit -m "feat: add net_ret return term to compose_reward

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 3: env.step 传入 net_ret

**Files:**
- Modify: `rl-portfolio-allocator/scripts/env.py:137-141`

**Interfaces:**
- Consumes: `compose_reward(..., net_ret=net)`（Task 2 的新签名）；`net` 已在 `env.py:122` 定义为 `net = gross - costs["total"]`。

- [ ] **Step 1: 写失败测试**

创建 `rl-portfolio-allocator/tests/test_env_reward_wiring.py`：

```python
import numpy as np
import pandas as pd
from scripts.config import get_config, FACTOR_NAMES, K
from scripts.env import PortfolioEnv


def _toy_features():
    dates = pd.date_range("2020-01-01", periods=6, freq="D")
    rows = []
    for d in dates:
        for i in range(40):
            row = {"trade_date": d, "symbol": f"S{i:03d}",
                   "ret_1d": 0.01 if i % 2 else -0.01, "is_suspended": False}
            for fn in FACTOR_NAMES:
                row[fn] = float((i % 5) - 2)
            rows.append(row)
    return pd.DataFrame(rows)


def test_step_info_has_ret_term():
    cfg = get_config()
    feats = _toy_features()
    idx = pd.Series(np.zeros(1), index=[feats["trade_date"].min()])
    env = PortfolioEnv(feats, idx, cfg, feats["trade_date"].min(), feats["trade_date"].max())
    env.reset(seed=0)
    _, reward, _, _, info = env.step(np.zeros(K, dtype=np.float32))
    assert "ret_term" in info["reward_parts"]
    # ret_term 应等于 reward_ret_weight * net_ret
    assert abs(info["reward_parts"]["ret_term"]
               - cfg["reward_ret_weight"] * info["net_ret"]) < 1e-12
```

- [ ] **Step 2: 运行测试确认失败**

Run: `cd rl-portfolio-allocator && python -m pytest tests/test_env_reward_wiring.py -q`
Expected: FAIL（`compose_reward()` 未传 `net_ret` → TypeError；env.py 尚未更新）

- [ ] **Step 3: 改 env.py**

`scripts/env.py:137-141` 的 `compose_reward(...)` 调用改为传入 `net_ret=net`：

```python
        reward, parts = compose_reward(
            dsr_delta, drawdown, turnover, hhi_v, self.cfg,
            net_ret=net,
            long_notional=long_notional, short_notional=short_notional,
            long_cap=self.cfg["long_notional"], short_cap=self.cfg["short_notional_cap"]
        )
```

- [ ] **Step 4: 运行测试确认通过**

Run: `cd rl-portfolio-allocator && python -m pytest tests/test_env_reward_wiring.py -q`
Expected: PASS（1 passed）

- [ ] **Step 5: 提交**

```bash
git add rl-portfolio-allocator/scripts/env.py rl-portfolio-allocator/tests/test_env_reward_wiring.py
git commit -m "feat: wire net_ret into env reward composition

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 4: train_ppo 加可选 EvalCallback 早停（向后兼容）

**Files:**
- Modify: `rl-portfolio-allocator/scripts/train.py:30-50`
- Test: `rl-portfolio-allocator/tests/test_train.py`（新增）

**Interfaces:**
- Produces: `train_ppo(env, total_timesteps, seed=0, device="auto", save_path=None, eval_env=None, eval_freq=10_000, n_eval_episodes=1, patience=None)`；不传 `eval_env` 时行为不变。

- [ ] **Step 1: 写失败测试**

创建 `rl-portfolio-allocator/tests/test_train.py`：

```python
import inspect
import numpy as np
import pandas as pd
from scripts.config import get_config, FACTOR_NAMES, K
from scripts.env import PortfolioEnv
from scripts.train import train_ppo


def _toy_env():
    cfg = get_config()
    dates = pd.date_range("2020-01-01", periods=8, freq="D")
    rows = []
    for d in dates:
        for i in range(40):
            row = {"trade_date": d, "symbol": f"S{i:03d}",
                   "ret_1d": 0.01 if i % 2 else -0.01, "is_suspended": False}
            for fn in FACTOR_NAMES:
                row[fn] = float((i % 5) - 2)
            rows.append(row)
    feats = pd.DataFrame(rows)
    idx = pd.Series(np.zeros(1), index=[feats["trade_date"].min()])
    return PortfolioEnv(feats, idx, cfg, feats["trade_date"].min(), feats["trade_date"].max())


def test_train_ppo_accepts_eval_kwargs():
    sig = inspect.signature(train_ppo)
    for p in ("eval_env", "eval_freq", "n_eval_episodes", "patience"):
        assert p in sig.parameters, f"missing param {p}"


def test_train_ppo_backward_compatible():
    # 不传 eval_env,极小步数,应正常返回 model
    model = train_ppo(_toy_env(), total_timesteps=64, seed=0, device="cpu")
    assert model is not None
```

- [ ] **Step 2: 运行测试确认失败**

Run: `cd rl-portfolio-allocator && python -m pytest tests/test_train.py -q`
Expected: FAIL（`test_train_ppo_accepts_eval_kwargs` 断言缺参数）

- [ ] **Step 3: 改 train.py**

`scripts/train.py` 的 `train_ppo` 整体替换为：

```python
def train_ppo(env, total_timesteps: int, seed: int = 0, device: str = "auto",
              save_path: Optional[str] = None, eval_env=None,
              eval_freq: int = 10_000, n_eval_episodes: int = 1,
              patience: Optional[int] = None):
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

    callback = None
    if eval_env is not None:
        from stable_baselines3.common.callbacks import (
            EvalCallback, StopTrainingOnNoModelImprovement,
        )
        eval_vec = DummyVecEnv([lambda: Monitor(eval_env)])
        stop_cb = (StopTrainingOnNoModelImprovement(
            max_no_improvement_evals=patience, min_evals=patience, verbose=0)
            if patience else None)
        callback = EvalCallback(
            eval_vec, eval_freq=eval_freq, n_eval_episodes=n_eval_episodes,
            deterministic=True, verbose=0, callback_after_eval=stop_cb,
        )

    model.learn(total_timesteps=total_timesteps, callback=callback)
    if save_path:
        pathlib.Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        model.save(save_path)
    return model
```

- [ ] **Step 4: 运行测试确认通过**

Run: `cd rl-portfolio-allocator && python -m pytest tests/test_train.py -q`
Expected: PASS（2 passed）

- [ ] **Step 5: 提交**

```bash
git add rl-portfolio-allocator/scripts/train.py rl-portfolio-allocator/tests/test_train.py
git commit -m "feat: add optional EvalCallback early-stopping to train_ppo

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 5: train.py main() 加 --timesteps；pipeline 走 200k

**Files:**
- Modify: `rl-portfolio-allocator/scripts/train.py:61-74`（main）
- Modify: `run_pipeline.sh`（`run_train` 函数体，约第 59-63 行）

**Interfaces:**
- Consumes: `train_ppo`（Task 4）。
- Produces: `python -m scripts.train --timesteps N` 可覆盖训练步数（默认 5000 保持 smoke）。

- [ ] **Step 1: 改 train.py main()**

`scripts/train.py` 的 `main()` 改为（加 argparse `--timesteps`，默认 5000）：

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

- [ ] **Step 2: 手动验证 argparse 生效（不实际训练）**

Run: `cd rl-portfolio-allocator && python -m scripts.train --help 2>&1 | grep -A1 timesteps`
Expected: 输出含 `--timesteps` 及其 help 文本。

- [ ] **Step 3: 改 run_pipeline.sh 的 run_train**

`run_pipeline.sh` 中 `run_train()` 函数体里的训练命令行改为：

```bash
    PYTHONPATH="$WORK_DIR:$PYTHONPATH" python -m scripts.train --timesteps 200000
```

- [ ] **Step 4: 语法校验**

Run: `bash -n run_pipeline.sh && echo OK`
Expected: `OK`（无语法错误）

- [ ] **Step 5: 提交**

```bash
git add rl-portfolio-allocator/scripts/train.py run_pipeline.sh
git commit -m "feat: --timesteps flag for train.py; pipeline trains 200k not smoke

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 6: 全量回归 + 20k 短训练验证收益项主导

**Files:**
- 无新增；运行既有测试 + 一次性诊断脚本。

**Interfaces:**
- Consumes: Task 1-5 全部改动。

- [ ] **Step 1: 跑全部单元测试**

Run: `cd rl-portfolio-allocator && python -m pytest -q`
Expected: 全部 PASS（含本会话早先的 `test_save_allocations.py`）。

- [ ] **Step 2: 20k 短训练确认 reward 分解与净收益方向**

Run（在 `rl-portfolio-allocator/` 下）：

```bash
python -c "
import pandas as pd, numpy as np
from scripts.config import get_config, FACTOR_NAMES
from scripts.env import PortfolioEnv
from scripts.train import train_ppo
from scripts.backtest import run_ppo_rollout
from scripts.diagnostics import summarize_rollout, check_degeneracy
cfg=get_config()
feats=pd.read_parquet('data/features.parquet')
dates=sorted(pd.to_datetime(feats['trade_date']).unique())
split=dates[int(len(dates)*0.7)]
idx=pd.Series(np.zeros(1),index=pd.to_datetime([feats['trade_date'].min()]))
tr=PortfolioEnv(feats,idx,cfg,dates[0],split-pd.Timedelta(days=1))
model=train_ppo(tr,total_timesteps=20000,seed=0,device='cpu')
te=PortfolioEnv(feats,idx,cfg,split,dates[-1])
rets,infos,_=run_ppo_rollout(model,te)
s=summarize_rollout(infos)
rb=s['reward_breakdown']
print('reward_breakdown:',{k:round(v,6) for k,v in rb.items()})
print('mean daily net ret', round(float(np.mean(rets)),6))
print('degeneracy:', check_degeneracy(s,cfg))
" 2>&1 | grep -vE "Warning|warn|deprecat"
```

Expected: `reward_breakdown` 中若含 `ret_term` 则其绝对量级 ≥ 各 penalty；`mean daily net ret` 不再是明显负值（≥ 约 −0.0005，理想为正）。注意 `summarize_rollout` 若未汇总 `ret_term` 则至少 penalty 均值应显著小于修复前（drawdown_penalty 从 ~−0.03 降到 ~−0.003）。

- [ ] **Step 3: 校验 allocations 仍合规**

Run: `cd rl-portfolio-allocator && python -m scripts.validate`
Expected: `[OK] ... validates`

- [ ] **Step 4: 提交（若 Step 2 有需要的 diagnostics 小改则一并提交，否则跳过）**

```bash
git add -A && git commit -m "test: full regression after reward rescale" --allow-empty

# 附 Co-Authored-By
```

---

## Self-Review

**Spec coverage:**
- §3.1 奖励重构 → Task 1（config）+ Task 2（compose_reward）+ Task 3（env 传参）✓
- §3.2 训练量+早停 → Task 4（EvalCallback）+ Task 5（--timesteps / pipeline 200k）✓
- §3.3 信号不动 → 无任务 ✓
- §4 验证 → Task 2/3/4 的 TDD + Task 6 回归 ✓

**Placeholder scan:** 无 TBD/TODO；所有代码步骤含完整代码。Task 6 Step 4 的 diagnostics 小改标注为条件性，非占位。

**Type consistency:** `compose_reward` 新签名 `net_ret` 位置在 Task 2 定义、Task 3 以 `net_ret=net` 关键字调用一致；`train_ppo` 新参数在 Task 4 定义、Task 5 未用新参数（仅 total_timesteps）无冲突；`reward_ret_weight` 键在 Task 1 产出、Task 2 消费一致。

**注意事项（供实现者）：** `summarize_rollout`（diagnostics.py）目前不汇总 `ret_term`，Task 6 Step 2 依赖 penalty 均值下降来验证。若希望 diagnostics 直接显示 ret_term，可在 Task 6 顺带给 `reward_breakdown` 加一行 `"ret_term_mean"`——属可选增强，非阻塞。
