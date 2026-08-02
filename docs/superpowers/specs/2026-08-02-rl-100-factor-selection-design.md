# RL Portfolio Allocator: 100-Factor Candidate Selection Design

## Status

Approved conversational design. This document specifies a research-only extension of the CSI300 PPO factor-weight allocator. It does not authorize production publication and does not constitute investment advice.

## Objective

Expand the current six OHLCV-derived factors into a catalog of exactly 100 price-volume candidates. For each walk-forward fold, use only that fold's training interval to select and orient exactly 20 factors. Freeze the factor identities, order, directions, reward variant, and rank-buffer variant before validation and out-of-sample testing.

The change is successful only if the selected-20 candidate method outperforms the original six-factor control under the full three-fold, five-seed, cost-aware research protocol. Merely generating more factors is not success.

## Constraints

- Inputs are limited to existing OHLCV, amount, suspension, symbol, and trade-date data.
- The catalog contains 100 factors total, including the existing six; 94 factors are new.
- Every fold selects exactly 20 factors.
- Selection is fold-local and training-only.
- If strict gates yield fewer than 20 factors, thresholds are relaxed in a deterministic sequence until 20 are selected.
- Causality, finite-value coverage, sample-length, and schema checks are never relaxed.
- Smoke runs validate plumbing only and can never approve publication.
- Full research remains fail-closed and requires three folds and five seeds.

## Why the Current PPO Is Weak

The current smoke result is not a reliable PPO performance estimate because it uses one fold, one seed, and 128 timesteps. It is a pipeline test rather than a converged experiment.

The current factors also show weak and unstable predictive content. In the observed data, five-day reversal was positive in 2010–2022 but lost its direction in 2023–2024. Most other factors had small or negative IC. The selected smoke strategy had positive 2024 OOS results but annualized turnover of about 50.4 versus a research ceiling of 12, and it lost money in the 2015, 2020, and 2022 stress periods.

Adding factors therefore addresses signal breadth only. It must be combined with training-only selection, redundancy control, cost-aware evaluation, and an unchanged six-factor benchmark.

## Factor Catalog

All factors at date `t` use data no later than `t`. Holdings settle from the next trading date. Values are cross-sectionally winsorized, standardized, and clipped to `[-3, 3]`. Divisions use a documented epsilon. Windows require their full causal history; missing warm-up values are not backfilled from the future.

Notation:

- `r_n = log(close_t / close_{t-n})`
- `MA_n` is the n-day closing-price mean.
- `AVG(x,n)` and `STD(x,n)` are trailing rolling statistics.
- OHLCV uses open, high, low, close, and volume; amount is traded amount.

### Trend and momentum

1. `mom_5`
2. `mom_10`
3. `mom_20` (existing)
4. `mom_60`
5. `mom_120`
6. `mom_252`
7. `ma_gap_5_20 = MA_5 / MA_20 - 1`
8. `ma_gap_20_60 = MA_20 / MA_60 - 1`
9. `ma_gap_60_120 = MA_60 / MA_120 - 1`
10. `mom_accel_20_60 = mom_20 - mom_60 / 3`

### Short-term reversal

11. `reversal_1 = -r_1`
12. `reversal_2 = -r_2`
13. `reversal_3 = -r_3`
14. `reversal_5` (existing)
15. `reversal_10 = -r_10`
16. `reversal_20 = -r_20`
17. `reversal_from_high_20 = close / rolling_max(close,20) - 1`
18. `reversal_from_low_20 = -(close / rolling_min(close,20) - 1)`
19. `return_autocorr_20`, lag-one return autocorrelation
20. `short_vs_medium_reversal = -(mom_5 - mom_20 / 4)`

### Volatility and downside risk

21. `vol_5`
22. `vol_10`
23. `vol_20` (existing)
24. `vol_60`
25. `downside_vol_20`
26. `downside_vol_60`
27. `upside_vol_20`
28. `semivol_ratio_20 = downside_vol_20 / upside_vol_20`
29. `vol_ratio_5_20 = vol_5 / vol_20`
30. `vol_ratio_20_60 = vol_20 / vol_60`

### Range and intraday dispersion

31. `parkinson_vol_10`
32. `parkinson_vol_20`
33. `parkinson_vol_60`
34. `atr_5`
35. `atr_14`
36. `atr_20`
37. `range_mean_5 = AVG((high-low)/close,5)`
38. `range_mean_20`
39. `range_expansion = range_mean_5 / range_mean_20`
40. `range_position_20 = (close-rolling_min(low,20))/(rolling_max(high,20)-rolling_min(low,20))`

### Volume trend

41. `volume_ratio_5_20`
42. `volume_ratio_20_60`
43. `volume_ratio_60_252`
44. `volume_mom_5 = log(AVG(volume,5)/AVG(volume,20))`
45. `volume_mom_20 = log(AVG(volume,20)/AVG(volume,60))`
46. `volume_mom_60 = log(AVG(volume,60)/AVG(volume,252))`
47. `volume_zscore_20`
48. `volume_zscore_60`
49. `volume_volatility_20`
50. `volume_persistence_20`, lag-one volume autocorrelation

### Turnover and activity

Because exact float shares are not available, turnover remains a volume-relative-to-252-day-average proxy.

51. `turnover_5`
52. `turnover_10`
53. `turnover_20` (existing)
54. `turnover_60`
55. `turnover_std_20`
56. `turnover_std_60`
57. `turnover_cv_20 = turnover_std_20 / AVG(turnover,20)`
58. `turnover_shock_5_60 = turnover_5 / turnover_60`
59. `turnover_change_20 = turnover_5 - turnover_20`
60. `active_days_ratio_20`

### Liquidity and price impact

61. `amihud_5`
62. `amihud_10`
63. `amihud_20` (existing)
64. `amihud_60`
65. `amihud_change_5_20`
66. `amihud_vol_20`
67. `inverse_amount_20 = 1 / AVG(amount,20)`
68. `amount_zscore_20`
69. `zero_return_ratio_20`
70. `roll_spread_20`, estimated from negative adjacent-return covariance

### Price-volume relationship

71. `ret_volume_corr_10`
72. `ret_volume_corr_20`
73. `ret_volume_corr_60`
74. `absret_volume_corr_20`
75. `price_volume_rank_div_20`
76. `obv_mom_10`
77. `obv_mom_20`
78. `obv_mom_60`
79. `up_down_volume_ratio_20`
80. `accumulation_distribution_20`

### Overnight and candle structure

81. `overnight_ret_1 = open_t / close_{t-1} - 1`
82. `overnight_ret_5`
83. `overnight_ret_20`
84. `intraday_ret_1 = close_t / open_t - 1`
85. `intraday_ret_5`
86. `intraday_ret_20`
87. `upper_shadow_20`
88. `lower_shadow_20`
89. `body_ratio_20 = AVG(abs(close-open)/(high-low),20)`
90. `close_location_20 = AVG((close-low)/(high-low),20)`

### Distribution shape and tail risk

91. `ret_skew_20`
92. `ret_skew_60` (existing)
93. `ret_kurt_20`
94. `ret_kurt_60`
95. `var_5pct_20`
96. `var_5pct_60`
97. `cvar_5pct_20`
98. `cvar_5pct_60`
99. `max_ret_20`
100. `min_ret_20`

## Fold-Local Selection

For each fold, selection uses only the training interval. Validation and test data cannot affect factor metrics, direction, thresholds, correlation filtering, or membership.

Each factor is evaluated on:

- mean daily Spearman IC against next-day return;
- ICIR;
- yearly sign consistency;
- direction-adjusted positive-IC rate;
- weekly long-short net factor Sharpe under configured costs;
- doubled-cost Sharpe;
- annualized factor turnover;
- finite coverage;
- worst-five-percent average return;
- stability across bull, bear, high-volatility, and low-volatility regimes.

The training-period mean-IC sign orients each factor. A negative-IC factor is multiplied by `-1` before PPO use. Direction is frozen through validation and test.

### Composite score

Metrics are converted to within-fold percentiles. The score is:

```text
  30% * abs(mean_ic)
+ 20% * abs(icir)
+ 15% * sign_consistency
+ 15% * net_factor_sharpe
+ 10% * doubled_cost_sharpe
+ 10% * regime_stability
- 10% * factor_turnover
-  5% * tail_loss
```

### Non-relaxable gates

- causal computation;
- at least 98% finite coverage in the eligible training panel;
- at least 100 eligible symbols per scored date;
- non-degenerate cross-sectional variance;
- at least 500 valid training dates;
- no infinity, future fill, or cross-symbol fill;
- explicit failure records rather than silent zero substitution.

### Strict selection

- `abs(mean_ic) >= 0.015`
- `abs(icir) >= 0.10`
- yearly sign consistency at least 60%
- direction-adjusted positive-IC rate at least 52%
- net factor Sharpe above zero
- pairwise redundancy correlation no greater than 0.80
- no more than three selected factors per family

Redundancy uses both factor-return time-series correlation and the median daily cross-sectional Spearman correlation. A higher-scoring factor wins a conflict.

### Deterministic relaxation

If fewer than 20 factors pass, apply these levels in order and record every admission:

1. Reduce `abs(mean_ic)` to 0.010, `abs(icir)` to 0.07, sign consistency to 55%, and allow net factor Sharpe down to -0.10.
2. Raise the family cap to four and correlation ceiling to 0.85.
3. Reduce `abs(mean_ic)` to 0.005 and raise the correlation ceiling to 0.90.
4. Fill remaining slots by composite score from factors satisfying all non-relaxable gates.

The result is exactly 20 factors or an explicit fold failure if fewer than 20 candidates satisfy non-relaxable data-quality gates.

## Architecture

Add:

- `scripts/factor_catalog.py`: immutable `FactorSpec` definitions, family membership, lookbacks, dependencies, versions, and catalog hash.
- `scripts/factor_compute.py`: causal family-batched calculations without selection policy.
- `scripts/factor_selection.py`: fold-local metrics, direction, score, quota, redundancy, relaxation, and reports.

Runtime code must no longer depend on a global six-factor `K`. `factor_names` and `k` come from the selected-factor artifact and flow through action spaces, observations, composite scoring, scalers, checkpoints, and allocation output.

Each fold has a distinct model, so folds may select different factor identities while all retain a 20-dimensional action space.

## Storage and Memory

Do not materialize all 100 float64 columns in one always-loaded DataFrame. Store factor values as float32 and partition by family:

```text
data/factors/
  catalog.json
  base.parquet
  momentum.parquet
  reversal.parquet
  volatility.parquet
  range.parquet
  volume.parquet
  turnover.parquet
  liquidity.parquet
  price_volume.parquet
  candle.parquet
  distribution.parquet
```

`base.parquet` contains trade date, symbol, next-return source, and suspension status. Selection reads families incrementally. After selection, materialize only the 20 chosen factors for PPO:

```text
data/fold_features/<run_id>/fold<N>.parquet
data/fold_market_state/<run_id>/fold<N>.parquet
```

Market state is split conceptually into base market regime fields and selected-factor rolling fields. Only selected factors contribute IC, return, volatility, and correlation state to PPO.

## Checkpoint Contract

Every model and scaler records:

- catalog version and hash;
- ordered selected-factor names;
- frozen directions;
- selection run ID and fold;
- state schema version;
- training range and budget.

Inference fails closed if name, order, direction, hash, or schema differs.

## Research Pipeline

```text
download OHLCV
-> compute and cache 100 candidates
-> coverage and quality checks
-> fold-local training-only factor selection
-> fold-specific 20-factor feature/state artifacts
-> reward validation
-> buffer validation
-> freeze factors/reward/buffer
-> one OOS test per seed
-> research gates
-> frozen-method stress tests
```

Smoke remains one fold, one seed, and 128 timesteps. It verifies mechanics only. Full research uses three folds, five seeds, and 100,000 timesteps per configured training run.

## Six-Factor Control

Every full experiment compares:

- `control_6f`: the existing six factors;
- `candidate_20f`: the fold-selected 20 factors.

They share folds, seeds, budgets, reward candidates, buffer candidates, costs, and test intervals.

Candidate-specific gates require:

- median OOS Sharpe at least 0.10 above the control;
- positive cost-adjusted excess return in at least two of three folds;
- positive doubled-cost OOS Sharpe;
- annualized turnover no greater than 12;
- no material stress-test maximum-drawdown degradation;
- complete three-fold, five-seed coverage.

Failure preserves the six-factor method and prevents candidate approval.

## Artifacts

```text
artifacts/factor_selection/<run_id>/fold<N>/
  candidates.parquet
  selected_factors.json
  factor_metrics.json
  correlation_matrix.parquet
  relaxation_log.json
  selection_report.json

artifacts/walk_forward/<run_id>/
  control_6f/
  candidate_20f/
  comparison.json
  research_summary.json
  gates.json
  stress.json
  approval.json  # full passing runs only
```

Repeated smoke runs may overwrite the smoke namespace. Full runs use unique or explicitly supplied run IDs.

## Error Handling

- Catalog count, duplicate name, family count, dependency, or hash mismatch fails immediately.
- Missing raw fields fail the affected catalog build; they are not imputed with fabricated values.
- A factor calculation failure is recorded with factor, family, fold, and reason.
- Selection reports every relaxation level and admitted factor.
- Insufficient non-relaxable candidates fail the fold.
- Missing selection artifacts or checkpoint metadata prevent validation, testing, allocation, and publication.
- Production publication remains atomic and approval-controlled.

## Verification

Tests must cover:

- exactly 100 unique names and ten factors per family;
- hand-calculated examples for every formula class;
- future price mutations do not alter historical factors;
- validation/test mutations do not alter training selection;
- factor direction is training-only and frozen;
- correlation filtering and family quotas;
- deterministic relaxation order and exact 20-factor output;
- fold artifact isolation;
- dynamic 20-dimensional action and observation schemas;
- checkpoint rejection on factor order, direction, catalog hash, or schema mismatch;
- continued operation of the six-factor control;
- failure on missing, infinite, degenerate, or corrupted data;
- smoke end-to-end generation;
- full control-versus-candidate gate aggregation.

## Acceptance Criteria

Implementation acceptance requires all automated tests and a successful smoke run that generates 100 candidates, selects exactly 20 for Fold 3, trains and reloads a matching checkpoint, and writes all factor-selection and walk-forward reports.

Research acceptance requires a full three-fold, five-seed run satisfying the candidate gates and outperforming the six-factor control. Until then, the 100-factor extension remains research-only and cannot produce a production approval.
