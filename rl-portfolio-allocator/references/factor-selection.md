# 100 因子候选库与 fold 内选择(研究手册)

本文档描述 RLPA 的「100 候选因子 → 每个 walk-forward fold 选出 20 个」工作流,
包括因果计算、族缓存、训练-only 选择、control/candidate 对照 gate、以及生产发布要求。

## 1. 候选库:100 因子 vs 选中 20 因子

- 候选目录定义在 `scripts/factor_catalog.py`,版本 `factor-catalog-v2`,
  共 **10 族 × 10 因子 = 100 个**,导入时即校验数量、唯一性与族结构;
  `catalog_hash()` 对有序目录做 sha256,任何目录变化都会使 hash 变化。
- 十个族:`momentum`、`reversal`、`volatility`、`range`、`volume`、`turnover`、
  `liquidity`、`price_volume`、`candle`、`distribution`。
  既有六因子(`mom_20, reversal_5, vol_20, turnover_20, amihud_20, ret_skew_60`)
  全部在目录内,作为 **control_6f 对照组**(`config.CONTROL_FACTOR_NAMES`)。
- 原始公式在 `scripts/factor_compute.py`,全部为因果( trailing )计算;
  面板按 symbol/date 排序后逐 symbol 计算,再按交易日做横截面 z-score,
  裁剪到 ±3,统一 float32。不跨 symbol 前向填充。

## 2. 族分区缓存(内存/磁盘影响)

`scripts/factor_cache.py` 将面板按族写入 `data/factors/`:

```
data/factors/
├── catalog.json          # 目录版本 + sha256 manifest
├── base.parquet          # trade_date, symbol, ret_1d, is_suspended
├── momentum.parquet      # 每族一个文件,列均为 float32
├── reversal.parquet
└── ... (共 10 个族 parquet)
```

- 写入采用临时目录 + 校验(行数、键唯一性、dtype、目录 hash)+ 原子 rename。
- **磁盘**:100 因子全量缓存;**内存**:每个 fold 只物化其选中的 20 列
  (`materialize_selected_panel`,按选中顺序、乘上方向 ±1),避免一次性加载全目录。
- 选择产物与因子缓存分离:缓存是候选数据,选择结果是方法产物。

## 3. Fold-local、训练-only 选择(无泄漏)

`scripts/factor_selection.py` 在每个 fold 的**训练区间内**计算因子指标并选择:

- 因子按 symbol 滞后一个交易日再与 `ret_1d` 相关(IC、ICIR、符号一致性、IC 为正比例);
  另构建周频 50/50 多空组合的净 Sharpe(使用与环境一致的停牌与成本函数,含双倍成本序列)。
- 硬 gate(不满足即记 `failure_reasons`,不参与选择):
  覆盖率 `min_coverage=0.98`、每日合格 symbol 数 `min_symbols=100`、
  有效交易日数 `min_dates=500`。
- 打分阈值按 **level 0→4 顺序放宽**(mean_ic 0.015→不限制、icir 0.10→不限制、
  sign_consistency 0.60→不限制、family_cap 3→不限制、相关性上限 0.80→不限制),
  每次放宽都记录在 `relaxation_log`;硬 gate 不满足且候选不足 20 个时直接报错。
- 冗余控制:同族数量上限 + 因子收益/横截面相关性上限,防止选中近似重复因子。
- 确定性:得分降序、目录顺序打破平局;每个 fold 恰好选出 **20 个**因子(含方向 ±1)。

## 4. 输出目录地图

```
data/factors/                                   # 族分区候选缓存(见 §2)
artifacts/state/data_coverage.json              # 数据覆盖前置检查
artifacts/walk_forward/<run_id>/
├── candidate_20f/
│   ├── selection/fold*/                        # candidates.parquet, selected_factors.json,
│   │                                           # factor_metrics.json, correlation_matrix.parquet,
│   │                                           # relaxation_log.json, selection_report.json
│   ├── features/  state/                       # 冻结的 20 因子特征与 market_state
│   └── <fold/seed 训练、validation、test、stress 产物>
├── control_6f/                                 # 同构的六因子对照分支
├── comparison.json                             # 成对证据汇总(3 folds × 5 seeds)
├── gates.json                                  # evaluate_candidate_gates 结果
├── summary.json                                # 含 publishable 标记
└── approval.json                               # 仅 full 运行且 research_ok=true 时写出
```

## 5. Smoke vs Full research

| | `--research-smoke`(`--all` 别名) | `--research-full` |
|---|---|---|
| 目的 | 快速验证流水线契约与产物结构 | 正式研究结论 |
| fold/seed | 缩减 | 恰好 3 folds × 5 seeds |
| gates.json | **强制 `research_ok=false`** | 按真实指标判定 |
| approval.json | **绝不写出** | 仅 `research_ok=true` 时写出 |
| 发布 | 不可发布 | approval 可供 `--publish` 使用 |

两者都会完整写出 control_6f / candidate_20f 产物与 comparison.json。
full 运行建议显式指定 run id:`RLPA_RUN_ID=<run-id> bash run_pipeline.sh --research-full`。

## 6. Control-vs-Candidate 研究 gates

`scripts/research_gates.py: evaluate_candidate_gates(comparison)` 全部为硬检查,
任一失败即 `research_ok=false`(fail closed):

1. **成对证据完整**:candidate_20f 与 control_6f 各 15 行(3 folds × 5 seeds),
   fold/seed 配对一致,指标有限,stress 产物带 sha256;
2. candidate 中位 OOS Sharpe − control ≥ **0.10**;
3. 正超额收益 fold 数 ≥ **2**;
4. candidate 双倍成本 OOS Sharpe > **0**;
5. candidate 年化换手率 ≤ **12**;
6. candidate stress MDD 相对 control 恶化 ≤ **0.05**。

## 7. 生产 approval 与发布要求

- 训练产物(model metadata 与 scaler JSON)持久化 **factor contract**
  (`train.FACTOR_CONTRACT_FIELDS`):`factor_catalog_version`、`factor_catalog_hash`、
  `selected_factors`(有序)、`factor_directions`(按选中顺序)、`selection_run_id`、
  `fold`、`state_schema_version`。加载、测试、配置、候选校验、发布路径都会校验,
  任一字段不匹配即拒绝。
- `approval.json` 绑定 method、gates、comparison 三份证据的相对路径与 sha256;
  `allocate.py` 在发布前重算 hash 并重新评估 `evaluate_candidate_gates`,证据不过期才放行。
- 发布命令:`bash run_pipeline.sh --publish --approval rl-portfolio-allocator/artifacts/walk_forward/<run_id>/approval.json`。
  `--publish` 先复制并校验选中因子 bundle,再重训生产模型;
  生产推理只从 approved method 读取有序选中因子并物化这些列。
- smoke 产物永不可发布;不要放宽任何已有 approval 检查。
