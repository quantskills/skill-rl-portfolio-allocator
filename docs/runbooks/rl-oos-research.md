# RLPA 样本外研究运行手册

本手册描述研究与生产发布的边界。研究 gate 是 fail-closed 的：任何缺失数据、失败测试、成本敏感性失败或未通过的 OOS gate，都意味着本次结果不可发布。

## 标准运行

```bash
cd rl-portfolio-allocator
RLPA_RUN_ID=20260730T120000Z
python -m scripts.features
python -m scripts.market_state
python -m scripts.check_data_coverage \
  --json artifacts/state/data_coverage.json
python -m pytest -q
python -m scripts.walk_forward --smoke
python -m scripts.walk_forward --full
python -m scripts.allocate --retrain \
  --approval "artifacts/walk_forward/${RLPA_RUN_ID}/approval.json"
python -m scripts.validate
```

推荐使用根目录包装器，它会固定顺序并在 gate 失败时立即停止：

```bash
./run_pipeline.sh --research-smoke
./run_pipeline.sh --research-full
./run_pipeline.sh --publish \
  --approval rl-portfolio-allocator/artifacts/walk_forward/<run-id>/approval.json
```

`--all` 等同于 research smoke，不会自动执行生产覆盖。full 研究在 CPU 上可能耗时较长。approval 必须来自通过的 full run；smoke 明确 `publishable=false`，不会生成生产 approval。

## 纪律与诊断

- 测试集只运行一次：candidate、reward、buffer、scaler 和 benchmark 选择只能使用训练/验证段。
- 先检查 `artifacts/state/data_coverage.json`、`market_state_quality.json`，再看 `artifacts/walk_forward/<run-id>/summary.json`、`research_summary.json` 和 `gates.json`。
- reward 诊断查看训练 JSONL 的 `reward_quality`、`scaled_net_return`、drawdown、turnover 和 concentration 分项；state 诊断查看 schema、warmup、nonnull/finite rate 及漂移报告。
- turnover/cost 诊断查看每个 fold/seed 的 `annualized_turnover`、`cost_1x`、`cost_2x` 和 baseline 对照；seed 诊断查看 `metrics.jsonl` 与 summary 的聚合结果。
- 只要 gate 失败，就保留研究产物用于诊断，不修改阈值、不凑 fold、不发布。

## 恢复上一个生产版本

发布前保留当前 production 目录及其 approval、checkpoint、scaler 和 allocations 的完整副本。新 approval 未通过或 `validate` 失败时，不替换当前版本；恢复时将最后一个已验证版本的完整目录原子替换回 production 目录，然后重新运行：

```bash
python -m scripts.validate --path ../rl-portfolio-allocator-production/data/allocations.parquet
```

恢复动作必须记录版本、approval 路径、校验结果和操作者；不要直接覆盖单个 parquet 或 checkpoint。
