# RL 100-Factor Selection Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a causal 100-factor OHLCV candidate library, select exactly 20 factors inside each walk-forward training fold, and compare the selected-20 PPO against the existing six-factor control without weakening publication gates.

**Architecture:** Factor definitions and computations are separated from fold-local selection policy. Candidate data is cached by family as float32, while each fold materializes only its selected 20 factors and matching market state. Runtime components receive ordered factor names from method artifacts instead of importing a global six-factor dimension.

**Tech Stack:** Python 3.10, pandas, NumPy, PyArrow/Parquet, Gymnasium, Stable-Baselines3 PPO, pytest, JSON research artifacts.

---

## Scope and milestone boundaries

Implement in a dedicated git worktree. Preserve the current dirty working tree and do not copy generated `data/` or `artifacts/` into commits.

The work has four sequential milestones:

1. Catalog and family-batched factor computation.
2. Training-only selection and fold artifacts.
3. Dynamic factor runtime and checkpoint contracts.
4. Control-versus-candidate walk-forward, gates, stress, and pipeline integration.

Every task below ends in a focused commit. Do not begin a task while the preceding task's targeted tests are red.

## File map

### New files

- `rl-portfolio-allocator/scripts/factor_catalog.py`: immutable catalog metadata, validation, and stable hash.
- `rl-portfolio-allocator/scripts/factor_compute.py`: causal raw factor formulas, cross-sectional normalization, and family computation.
- `rl-portfolio-allocator/scripts/factor_cache.py`: family-partitioned Parquet writer/reader and selected-panel materialization.
- `rl-portfolio-allocator/scripts/factor_selection.py`: training-only metrics, scoring, redundancy control, relaxation, and artifact output.
- `rl-portfolio-allocator/tests/test_factor_catalog.py`: catalog count/family/hash contract.
- `rl-portfolio-allocator/tests/test_factor_compute.py`: formula, normalization, dtype, and causality tests.
- `rl-portfolio-allocator/tests/test_factor_cache.py`: storage schema and selected-panel tests.
- `rl-portfolio-allocator/tests/test_factor_selection.py`: metrics, leakage, quota, correlation, relaxation, and artifacts.
- `rl-portfolio-allocator/tests/test_dynamic_factors.py`: dynamic action/observation/baseline integration.
- `rl-portfolio-allocator/tests/test_factor_checkpoint_contract.py`: metadata mismatch rejection.
- `rl-portfolio-allocator/tests/test_factor_pipeline.py`: smoke orchestration and control/candidate artifact isolation.

### Modified files

- `rl-portfolio-allocator/scripts/config.py`: retain `CONTROL_FACTOR_NAMES`; make runtime names config-driven.
- `rl-portfolio-allocator/scripts/features.py`: download raw inputs and delegate 100-factor cache generation.
- `rl-portfolio-allocator/scripts/state.py`: bump state schema and continue accepting explicit names.
- `rl-portfolio-allocator/scripts/market_state.py`: accept explicit factor names and directions.
- `rl-portfolio-allocator/scripts/env.py`: derive `factor_names` and `k` from config.
- `rl-portfolio-allocator/scripts/baselines.py`: remove module-global `K` and factor list.
- `rl-portfolio-allocator/scripts/backtest.py`: pass explicit names through baselines and market state.
- `rl-portfolio-allocator/scripts/train.py`: persist ordered factor metadata with scaler/model artifacts.
- `rl-portfolio-allocator/scripts/walk_forward.py`: run fold-local selection and separate control/candidate methods.
- `rl-portfolio-allocator/scripts/research_gates.py`: add candidate-versus-control gates.
- `rl-portfolio-allocator/scripts/stress_test.py`: use frozen method factor metadata.
- `rl-portfolio-allocator/scripts/allocate.py`: validate and apply approved selected factors.
- `rl-portfolio-allocator/run_pipeline.sh`: expose factor build/selection in research flow.
- Existing tests importing `K`/`FACTOR_NAMES`: use explicit fixture factor names where runtime behavior is under test.

## Milestone 1: Catalog and computation

### Task 1: Establish the immutable 100-factor catalog

**Files:**
- Create: `rl-portfolio-allocator/scripts/factor_catalog.py`
- Create: `rl-portfolio-allocator/tests/test_factor_catalog.py`
- Modify: `rl-portfolio-allocator/scripts/config.py`

- [ ] **Step 1: Write catalog contract tests**

```python
from scripts.factor_catalog import CATALOG_VERSION, FACTOR_CATALOG, catalog_hash


def test_catalog_has_exactly_ten_families_of_ten_unique_factors():
    assert CATALOG_VERSION == "factor-catalog-v2"
    assert len(FACTOR_CATALOG) == 100
    assert len({spec.name for spec in FACTOR_CATALOG}) == 100
    families = {spec.family for spec in FACTOR_CATALOG}
    assert len(families) == 10
    assert {family: sum(s.family == family for s in FACTOR_CATALOG)
            for family in families} == {family: 10 for family in families}


def test_existing_six_are_present_once():
    existing = {"mom_20", "reversal_5", "vol_20", "turnover_20",
                "amihud_20", "ret_skew_60"}
    assert existing <= {spec.name for spec in FACTOR_CATALOG}


def test_catalog_hash_is_order_sensitive_and_stable():
    assert catalog_hash(FACTOR_CATALOG) == catalog_hash(FACTOR_CATALOG)
    assert catalog_hash(tuple(reversed(FACTOR_CATALOG))) != catalog_hash(FACTOR_CATALOG)
```

- [ ] **Step 2: Run the tests and verify RED**

Run:

```bash
cd rl-portfolio-allocator
python -m pytest -q tests/test_factor_catalog.py
```

Expected: import failure for `scripts.factor_catalog`.

- [ ] **Step 3: Implement catalog types, validation, and hash**

Use this public API:

```python
from dataclasses import asdict, dataclass
import hashlib
import json

CATALOG_VERSION = "factor-catalog-v2"


@dataclass(frozen=True)
class FactorSpec:
    name: str
    family: str
    lookback: int
    required_columns: tuple[str, ...]
    version: str = "v1"


def validate_catalog(catalog: tuple[FactorSpec, ...]) -> None:
    if len(catalog) != 100:
        raise ValueError("factor catalog must contain exactly 100 factors")
    names = [spec.name for spec in catalog]
    if len(set(names)) != len(names):
        raise ValueError("factor catalog names must be unique")
    counts = {family: sum(spec.family == family for spec in catalog)
              for family in {spec.family for spec in catalog}}
    if len(counts) != 10 or set(counts.values()) != {10}:
        raise ValueError("factor catalog must contain ten families of ten")


def catalog_hash(catalog: tuple[FactorSpec, ...]) -> str:
    payload = json.dumps([asdict(spec) for spec in catalog],
                         sort_keys=True, separators=(",", ":")).encode()
    return "sha256:" + hashlib.sha256(payload).hexdigest()
```

Declare all 100 names, families, maximum lookbacks, and raw dependencies exactly as listed in the approved design. Call `validate_catalog(FACTOR_CATALOG)` at import time.

In `config.py`, rename the static list to:

```python
CONTROL_FACTOR_NAMES = [
    "mom_20", "reversal_5", "vol_20",
    "turnover_20", "amihud_20", "ret_skew_60",
]
FACTOR_NAMES = CONTROL_FACTOR_NAMES  # compatibility during migration
K = len(CONTROL_FACTOR_NAMES)
```

- [ ] **Step 4: Run catalog and config tests**

Run:

```bash
python -m pytest -q tests/test_factor_catalog.py tests/test_config.py
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add rl-portfolio-allocator/scripts/factor_catalog.py \
  rl-portfolio-allocator/scripts/config.py \
  rl-portfolio-allocator/tests/test_factor_catalog.py
git commit -m "feat: define immutable 100-factor catalog"
```

### Task 2: Implement price, trend, reversal, volatility, and range families

**Files:**
- Create: `rl-portfolio-allocator/scripts/factor_compute.py`
- Create: `rl-portfolio-allocator/tests/test_factor_compute.py`

- [ ] **Step 1: Write deterministic formula tests**

Create a 320-row single-symbol OHLCV fixture with strictly positive prices and varying returns. Test representative formulas and all output names:

```python
def test_price_family_formulas_match_manual_values(price_frame):
    raw = compute_raw_factors(price_frame)
    row = raw.iloc[-1]
    close = price_frame["close"]
    assert row["mom_20"] == pytest.approx(np.log(close.iloc[-1] / close.iloc[-21]))
    assert row["reversal_5"] == pytest.approx(-np.log(close.iloc[-1] / close.iloc[-6]))
    assert row["ma_gap_5_20"] == pytest.approx(
        close.iloc[-5:].mean() / close.iloc[-20:].mean() - 1.0
    )
    assert row["range_position_20"] == pytest.approx(
        (close.iloc[-1] - price_frame["low"].iloc[-20:].min()) /
        (price_frame["high"].iloc[-20:].max() - price_frame["low"].iloc[-20:].min())
    )


def test_raw_computation_emits_first_four_families(price_frame):
    raw = compute_raw_factors(price_frame)
    expected = {spec.name for spec in FACTOR_CATALOG
                if spec.family in {"momentum", "reversal", "volatility", "range"}}
    assert expected <= set(raw.columns)
```

- [ ] **Step 2: Run the tests and verify RED**

Run `python -m pytest -q tests/test_factor_compute.py`.

Expected: import failure for `compute_raw_factors`.

- [ ] **Step 3: Implement shared causal helpers**

Implement explicit helpers in `factor_compute.py`:

```python
EPS = 1e-12


def safe_div(num, den):
    return num / den.replace(0.0, np.nan)


def log_return(close, window):
    return np.log(safe_div(close, close.shift(window)))


def rolling_corr(left, right, window):
    return left.rolling(window, min_periods=window).corr(right)


def downside_std(rets, window):
    return rets.where(rets < 0).rolling(window, min_periods=window).std()


def upside_std(rets, window):
    return rets.where(rets > 0).rolling(window, min_periods=window).std()
```

Implement all factors 1–40 from the design in one per-symbol pass. Use full-window `min_periods`; implement ATR from the maximum of high-low, absolute high-previous-close, and absolute low-previous-close; implement Parkinson volatility as the square root of rolling mean squared log high/low divided by `4*log(2)`.

- [ ] **Step 4: Run representative formula tests**

Run `python -m pytest -q tests/test_factor_compute.py -k 'price or raw'`.

Expected: PASS for factors 1–40.

- [ ] **Step 5: Commit**

```bash
git add rl-portfolio-allocator/scripts/factor_compute.py \
  rl-portfolio-allocator/tests/test_factor_compute.py
git commit -m "feat: compute causal price and volatility factors"
```

### Task 3: Implement volume, liquidity, candle, and distribution families

**Files:**
- Modify: `rl-portfolio-allocator/scripts/factor_compute.py`
- Modify: `rl-portfolio-allocator/tests/test_factor_compute.py`

- [ ] **Step 1: Add failing formula-class tests**

Add assertions covering factors 41–100:

```python
def test_volume_liquidity_and_candle_formulas(price_frame):
    raw = compute_raw_factors(price_frame)
    last = raw.iloc[-1]
    volume = price_frame["volume"]
    amount = price_frame["amount"]
    assert last["volume_ratio_5_20"] == pytest.approx(
        volume.iloc[-5:].mean() / volume.iloc[-20:].mean())
    assert last["inverse_amount_20"] == pytest.approx(1.0 / amount.iloc[-20:].mean())
    assert last["overnight_ret_1"] == pytest.approx(
        price_frame["open"].iloc[-1] / price_frame["close"].iloc[-2] - 1.0)
    assert last["intraday_ret_1"] == pytest.approx(
        price_frame["close"].iloc[-1] / price_frame["open"].iloc[-1] - 1.0)


def test_tail_statistics_use_trailing_returns(price_frame):
    raw = compute_raw_factors(price_frame)
    rets = price_frame["close"].pct_change().iloc[-20:].dropna()
    assert raw["max_ret_20"].iloc[-1] == pytest.approx(rets.max())
    assert raw["min_ret_20"].iloc[-1] == pytest.approx(rets.min())


def test_raw_computation_emits_every_catalog_name(price_frame):
    raw = compute_raw_factors(price_frame)
    assert {spec.name for spec in FACTOR_CATALOG} <= set(raw.columns)
```

- [ ] **Step 2: Verify RED**

Run `python -m pytest -q tests/test_factor_compute.py`.

Expected: missing factors 41–100.

- [ ] **Step 3: Implement factors 41–100**

Use the approved formulas with these unambiguous conventions:

- `volume_zscore_n = (volume - AVG(volume,n)) / STD(volume,n)`.
- `volume_volatility_20 = STD(pct_change(volume),20)`.
- turnover proxy is `volume / AVG(volume,252)`; windowed turnover is its trailing mean.
- Amihud is `AVG(abs(ret_1d)/(amount+EPS),n)`.
- Roll spread is `2*sqrt(max(-cov(ret_t,ret_{t-1}),0))` over 20 days.
- OBV increments are `sign(ret_1d)*volume`; momentum is the n-day OBV difference.
- accumulation/distribution uses `((2*close-high-low)/(high-low))*volume`, summed over 20 days.
- aggregated overnight/intraday 5/20 factors are trailing sums of their one-day log returns.
- shadows and candle bodies divide by `max(high-low, EPS)`.
- VaR is the rolling 5th percentile; CVaR is the rolling mean of returns at or below that trailing quantile.

- [ ] **Step 4: Add causal mutation and dtype tests**

```python
def test_future_mutation_does_not_change_historical_factors(multi_symbol_prices):
    cutoff = pd.Timestamp("2021-06-30")
    before = compute_factor_panel(multi_symbol_prices)
    changed = multi_symbol_prices.copy()
    changed.loc[changed.trade_date > cutoff, ["open", "high", "low", "close",
                                               "volume", "amount"]] *= 10
    after = compute_factor_panel(changed)
    pd.testing.assert_frame_equal(
        before[before.trade_date <= cutoff].reset_index(drop=True),
        after[after.trade_date <= cutoff].reset_index(drop=True),
    )


def test_normalized_factor_columns_are_float32_and_clipped(multi_symbol_prices):
    panel = compute_factor_panel(multi_symbol_prices)
    for spec in FACTOR_CATALOG:
        assert panel[spec.name].dtype == np.float32
        assert panel[spec.name].dropna().between(-3.0, 3.0).all()
```

Implement `compute_factor_panel(prices)` by sorting by symbol/date, applying `compute_raw_factors` per symbol, then cross-sectionally z-scoring each catalog field. Do not forward-fill across symbols.

- [ ] **Step 5: Run and commit**

Run `python -m pytest -q tests/test_factor_compute.py`.

Expected: PASS.

```bash
git add rl-portfolio-allocator/scripts/factor_compute.py \
  rl-portfolio-allocator/tests/test_factor_compute.py
git commit -m "feat: complete 100 causal OHLCV factors"
```

### Task 4: Add family-partitioned factor caching

**Files:**
- Create: `rl-portfolio-allocator/scripts/factor_cache.py`
- Create: `rl-portfolio-allocator/tests/test_factor_cache.py`
- Modify: `rl-portfolio-allocator/scripts/features.py`

- [ ] **Step 1: Write cache contract tests**

```python
def test_write_cache_partitions_catalog_by_family(tmp_path, factor_panel):
    manifest = write_factor_cache(factor_panel, tmp_path)
    assert (tmp_path / "catalog.json").exists()
    assert (tmp_path / "base.parquet").exists()
    assert len(list(tmp_path.glob("*.parquet"))) == 11
    assert manifest["catalog_hash"].startswith("sha256:")


def test_materialize_selected_panel_preserves_order_and_direction(tmp_path, factor_panel):
    write_factor_cache(factor_panel, tmp_path)
    selected = [
        {"name": "vol_20", "direction": -1},
        {"name": "mom_20", "direction": 1},
    ]
    out = materialize_selected_panel(tmp_path, selected)
    assert list(out.columns[-2:]) == ["vol_20", "mom_20"]
    expected = -factor_panel["vol_20"]
    np.testing.assert_allclose(out["vol_20"], expected, equal_nan=True)
```

- [ ] **Step 2: Verify RED**

Run `python -m pytest -q tests/test_factor_cache.py`.

- [ ] **Step 3: Implement atomic cache writer and reader**

Public API:

- `BASE_COLUMNS = ("trade_date", "symbol", "ret_1d", "is_suspended")`
- `write_factor_cache(panel: pd.DataFrame, root: pathlib.Path) -> dict`
- `load_family(root: pathlib.Path, family: str) -> pd.DataFrame`
- `materialize_selected_panel(root: pathlib.Path, selected: list[dict]) -> pd.DataFrame`

Write to a sibling temporary directory, validate row counts, keys, float32 dtypes, catalog version/hash, then rename into place. Reject duplicate `(trade_date,symbol)` keys and unknown selected names.

Update `features.main()` to compute the catalog panel and write `data/factors/`. Keep `data/features.parquet` as the six-factor compatibility panel until Milestone 4 changes the pipeline.

- [ ] **Step 4: Run cache and legacy feature tests**

Run:

```bash
python -m pytest -q tests/test_factor_cache.py tests/test_config.py tests/test_market_state.py
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add rl-portfolio-allocator/scripts/factor_cache.py \
  rl-portfolio-allocator/scripts/features.py \
  rl-portfolio-allocator/tests/test_factor_cache.py
git commit -m "feat: cache factor candidates by family"
```

## Milestone 2: Training-only factor selection

### Task 5: Compute fold-local factor metrics

**Files:**
- Create: `rl-portfolio-allocator/scripts/factor_selection.py`
- Create: `rl-portfolio-allocator/tests/test_factor_selection.py`

- [ ] **Step 1: Write metric and leakage tests**

```python
def test_metrics_use_only_training_dates(candidate_panel):
    metrics = compute_factor_metrics(candidate_panel, ["f1"],
                                     "2018-01-01", "2020-12-31", test_cfg())
    mutated = candidate_panel.copy()
    mutated.loc[mutated.trade_date > "2020-12-31", ["f1", "ret_1d"]] *= -100
    changed = compute_factor_metrics(mutated, ["f1"],
                                     "2018-01-01", "2020-12-31", test_cfg())
    assert metrics == changed


def test_negative_ic_orients_factor_positive(candidate_panel):
    metrics = compute_factor_metrics(candidate_panel, ["negative_predictor"],
                                     "2018-01-01", "2020-12-31", test_cfg())
    assert metrics["negative_predictor"]["direction"] == -1
    assert metrics["negative_predictor"]["oriented_mean_ic"] > 0
```

- [ ] **Step 2: Verify RED**

Run `python -m pytest -q tests/test_factor_selection.py -k metrics`.

- [ ] **Step 3: Implement metric API**

```python
@dataclass(frozen=True)
class SelectionThresholds:
    min_coverage: float = 0.98
    min_symbols: int = 100
    min_dates: int = 500
```

Expose `compute_factor_metrics(panel, factor_names, train_start, train_end, cfg) -> dict`
and `percentile_scores(metrics: dict) -> dict`.

Lag factors one trading date by symbol before correlating with `ret_1d`. Build weekly 50/50 long-short returns with the same suspension and cost functions used by the environment. Compute the doubled-cost series from gross and net returns. Record every hard-gate failure in `failure_reasons`.

- [ ] **Step 4: Test metrics**

Run `python -m pytest -q tests/test_factor_selection.py -k metrics`.

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add rl-portfolio-allocator/scripts/factor_selection.py \
  rl-portfolio-allocator/tests/test_factor_selection.py
git commit -m "feat: score factors inside training folds"
```

### Task 6: Implement deterministic selection, relaxation, and artifacts

**Files:**
- Modify: `rl-portfolio-allocator/scripts/factor_selection.py`
- Modify: `rl-portfolio-allocator/tests/test_factor_selection.py`

- [ ] **Step 1: Write quota, redundancy, and relaxation tests**

```python
def test_selector_caps_family_and_rejects_correlated_duplicate(metric_fixture):
    result = select_factors(metric_fixture, target_count=20)
    families = Counter(item["family"] for item in result.selected)
    assert max(families.values()) <= result.final_family_cap
    assert not ({"mom_5", "mom_10"} <= {x["name"] for x in result.selected})


def test_selector_relaxes_in_documented_order(relaxation_fixture):
    result = select_factors(relaxation_fixture, target_count=20)
    assert len(result.selected) == 20
    assert [event["level"] for event in result.relaxation_log] == [0, 1, 2, 3, 4]


def test_selector_fails_when_hard_valid_candidates_are_insufficient(invalid_fixture):
    with pytest.raises(ValueError, match="fewer than 20 hard-valid factors"):
        select_factors(invalid_fixture, target_count=20)
```

- [ ] **Step 2: Verify RED**

Run `python -m pytest -q tests/test_factor_selection.py -k 'selector or relaxes'`.

- [ ] **Step 3: Implement selector and artifact writer**

Public result contract:

```python
@dataclass(frozen=True)
class SelectionResult:
    selected: tuple[dict, ...]
    relaxation_log: tuple[dict, ...]
    final_family_cap: int
    final_correlation_ceiling: float
    catalog_hash: str
```

Expose `select_factors(metrics, return_corr, cross_section_corr, target_count=20)
-> SelectionResult` and `write_selection_artifacts(result, metrics, correlations,
output_dir, fold, train_range) -> None`.

Use score-descending then catalog-order tie breaking. Apply levels 0–4 exactly as specified. Preserve the selected order in JSON. Write `candidates.parquet`, `selected_factors.json`, `factor_metrics.json`, `correlation_matrix.parquet`, `relaxation_log.json`, and `selection_report.json` atomically.

- [ ] **Step 4: Run all selection tests**

Run `python -m pytest -q tests/test_factor_selection.py`.

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add rl-portfolio-allocator/scripts/factor_selection.py \
  rl-portfolio-allocator/tests/test_factor_selection.py
git commit -m "feat: select 20 diverse factors per fold"
```

## Milestone 3: Dynamic factor runtime

### Task 7: Make environment and state dimensions config-driven

**Files:**
- Modify: `rl-portfolio-allocator/scripts/config.py`
- Modify: `rl-portfolio-allocator/scripts/env.py`
- Modify: `rl-portfolio-allocator/scripts/state.py`
- Create: `rl-portfolio-allocator/tests/test_dynamic_factors.py`
- Modify: existing environment/state tests

- [ ] **Step 1: Write dynamic-dimension tests**

```python
def test_environment_uses_explicit_factor_names(dynamic_inputs):
    features, market_state, cfg, names = dynamic_inputs(k=3)
    env = PortfolioEnv(features, market_state, cfg,
                       features.trade_date.min(), features.trade_date.max())
    obs, _ = env.reset(seed=0)
    assert env.factor_names == tuple(names)
    assert env.action_space.shape == (3,)
    assert obs.shape == (state_dim(names),)
    _, _, _, _, info = env.step(np.zeros(3, dtype=np.float32))
    assert len(info["factor_w"]) == 3
```

- [ ] **Step 2: Verify RED**

Run `python -m pytest -q tests/test_dynamic_factors.py`.

- [ ] **Step 3: Replace runtime globals**

In `get_config()`, copy the selected list rather than sharing a mutable global:

```python
"factor_names": list(CONTROL_FACTOR_NAMES),
"k": len(CONTROL_FACTOR_NAMES),
```

In `PortfolioEnv.__init__`:

```python
self.factor_names = tuple(cfg["factor_names"])
self.k = len(self.factor_names)
if self.k == 0 or cfg.get("k", self.k) != self.k:
    raise ValueError("factor_names and k must define the same non-empty dimension")
```

Replace every runtime `FACTOR_NAMES` and `K` in `env.py` with these fields. Bump `STATE_SCHEMA_VERSION` to `state-v2-dynamic-factors` and require explicit names for state-field construction.

- [ ] **Step 4: Run environment, state, reward, and observation tests**

Run:

```bash
python -m pytest -q tests/test_dynamic_factors.py tests/test_weekly_env.py \
  tests/test_state_causality.py tests/test_env_reward_wiring.py tests/test_observation.py
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add rl-portfolio-allocator/scripts/config.py \
  rl-portfolio-allocator/scripts/env.py rl-portfolio-allocator/scripts/state.py \
  rl-portfolio-allocator/tests/test_dynamic_factors.py \
  rl-portfolio-allocator/tests/test_weekly_env.py \
  rl-portfolio-allocator/tests/test_state_causality.py \
  rl-portfolio-allocator/tests/test_env_reward_wiring.py \
  rl-portfolio-allocator/tests/test_observation.py
git commit -m "refactor: make PPO factor dimensions explicit"
```

### Task 8: Make market state and baselines factor-list aware

**Files:**
- Modify: `rl-portfolio-allocator/scripts/market_state.py`
- Modify: `rl-portfolio-allocator/scripts/baselines.py`
- Modify: `rl-portfolio-allocator/scripts/backtest.py`
- Modify: `rl-portfolio-allocator/scripts/stress_test.py`
- Modify: related tests

- [ ] **Step 1: Write explicit-list tests**

```python
def test_market_state_contains_only_requested_factor_fields(dynamic_inputs):
    features, index_returns, cfg, names = dynamic_inputs(k=3, index=True)
    state = build_market_state(features, index_returns, cfg, factor_names=names)
    assert all(f"{name}_ic_mean_20" in state for name in names)
    assert "mom_20_ic_mean_20" not in state unless "mom_20" in names


def test_equal_factor_baseline_dimension_matches_config(dynamic_inputs):
    features, market_state, cfg, names = dynamic_inputs(k=3)
    result = static_factor_equal_rollout(features, cfg, "2020-01-01", "2020-12-31")
    assert result.ndim == 1
```

- [ ] **Step 2: Verify RED**

Run targeted market-state and baseline tests.

- [ ] **Step 3: Thread explicit names through APIs**

Change signatures to `compute_daily_factor_ic(features, factor_names)`,
`compute_daily_factor_returns(features, cfg=None, factor_names=None)`,
`rolling_mean_factor_corr(factor_returns, window, factor_names)`, and
`build_market_state(features, index_returns, cfg=None, factor_names=None)`.

Default `factor_names` from `cfg["factor_names"]`, never from module globals in runtime paths. In baselines derive `names = tuple(cfg["factor_names"])` and `k = len(names)`. Pass names through backtest and stress functions.

- [ ] **Step 4: Run related suites**

Run:

```bash
python -m pytest -q tests/test_market_state.py tests/test_baselines_causal.py \
  tests/test_cost_sensitivity.py tests/test_stress_test.py
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add rl-portfolio-allocator/scripts/market_state.py \
  rl-portfolio-allocator/scripts/baselines.py \
  rl-portfolio-allocator/scripts/backtest.py \
  rl-portfolio-allocator/scripts/stress_test.py \
  rl-portfolio-allocator/tests/test_market_state.py \
  rl-portfolio-allocator/tests/test_baselines_causal.py \
  rl-portfolio-allocator/tests/test_cost_sensitivity.py \
  rl-portfolio-allocator/tests/test_stress_test.py
git commit -m "refactor: build state and baselines for selected factors"
```

### Task 9: Enforce selected-factor checkpoint and scaler metadata

**Files:**
- Modify: `rl-portfolio-allocator/scripts/train.py`
- Modify: `rl-portfolio-allocator/scripts/observation.py`
- Modify: `rl-portfolio-allocator/scripts/allocate.py`
- Create: `rl-portfolio-allocator/tests/test_factor_checkpoint_contract.py`
- Modify: training/production artifact tests

- [ ] **Step 1: Write mismatch rejection tests**

```python
@pytest.mark.parametrize("field,value", [
    ("selected_factors", ["b", "a"]),
    ("factor_directions", {"a": -1, "b": 1}),
    ("factor_catalog_hash", "sha256:wrong"),
    ("state_schema_version", "wrong"),
])
def test_checkpoint_contract_rejects_mismatch(tmp_path, valid_contract, field, value):
    actual = dict(valid_contract)
    actual[field] = value
    with pytest.raises(ValueError, match="factor checkpoint contract mismatch"):
        validate_factor_contract(actual, valid_contract)
```

- [ ] **Step 2: Verify RED**

Run `python -m pytest -q tests/test_factor_checkpoint_contract.py`.

- [ ] **Step 3: Implement one shared contract validator**

Define in `train.py`:

```python
FACTOR_CONTRACT_FIELDS = (
    "factor_catalog_version", "factor_catalog_hash", "selected_factors",
    "factor_directions", "selection_run_id", "fold", "state_schema_version",
)


def validate_factor_contract(actual: dict, expected: dict) -> None:
    mismatches = [key for key in FACTOR_CONTRACT_FIELDS
                  if actual.get(key) != expected.get(key)]
    if mismatches:
        raise ValueError("factor checkpoint contract mismatch: " + ", ".join(mismatches))
```

Persist the same contract in model metadata and scaler JSON. Require it in load, test, allocation, candidate validation, and publish paths. Serialize factor directions in selected-factor order.

- [ ] **Step 4: Run artifact and production-gate tests**

Run:

```bash
python -m pytest -q tests/test_factor_checkpoint_contract.py \
  tests/test_training_artifacts.py tests/test_production_gate.py \
  tests/test_allocate_candidate.py tests/test_observation.py
```

Expected: PASS.

- [ ] **Step 5: Commit**

Stage and commit the exact modified production-contract and test files:

```bash
git add rl-portfolio-allocator/scripts/train.py \
  rl-portfolio-allocator/scripts/observation.py \
  rl-portfolio-allocator/scripts/allocate.py \
  rl-portfolio-allocator/tests/test_factor_checkpoint_contract.py \
  rl-portfolio-allocator/tests/test_training_artifacts.py \
  rl-portfolio-allocator/tests/test_production_gate.py \
  rl-portfolio-allocator/tests/test_allocate_candidate.py \
  rl-portfolio-allocator/tests/test_observation.py
git commit -m "feat: bind checkpoints to selected factor contracts"
```

## Milestone 4: Walk-forward comparison and pipeline

### Task 10: Integrate fold-local selection into walk-forward

**Files:**
- Modify: `rl-portfolio-allocator/scripts/walk_forward.py`
- Create: `rl-portfolio-allocator/tests/test_factor_pipeline.py`
- Modify: `rl-portfolio-allocator/tests/test_walk_forward.py`

- [ ] **Step 1: Write orchestration and leakage tests**

```python
def test_walk_forward_selects_before_validation_and_freezes_for_test(tmp_path):
    events = []
    result = run_walk_forward(
        folds=[toy_fold()], output_root=tmp_path, smoke=True,
        selector=recording_selector(events), trainer=recording_trainer(events),
        tester=recording_tester(events), candidate_cache=toy_cache(),
    )
    assert events[0][0] == "select"
    selected = result["candidate_20f"]["method_by_fold"]["3"]["selected_factors"]
    assert all(event[-1] == selected for event in events if event[0] in {"train", "test"})


def test_future_mutation_does_not_change_fold_selection(tmp_path, candidate_cache):
    first = select_walk_forward_fold(candidate_cache, toy_fold(), tmp_path / "a")
    candidate_cache.mutate_after("2022-12-31")
    second = select_walk_forward_fold(candidate_cache, toy_fold(), tmp_path / "b")
    assert first.selected == second.selected
```

- [ ] **Step 2: Verify RED**

Run the two tests above.

- [ ] **Step 3: Add fold data preparation**

Implement:

```python
def prepare_fold_factors(*, fold, cache_root, index_returns,
                         selection_root, feature_root, state_root, cfg):
    """Select on fold.train, then materialize frozen features/state through fold.test."""
```

The returned object contains ordered names, directions, contract, feature path, market-state path, and selection artifact path. `run_walk_forward` invokes it before reward candidates and passes the same object unchanged to validation and test.

- [ ] **Step 4: Run walk-forward tests**

Run:

```bash
python -m pytest -q tests/test_factor_pipeline.py tests/test_walk_forward.py
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add rl-portfolio-allocator/scripts/walk_forward.py \
  rl-portfolio-allocator/tests/test_factor_pipeline.py \
  rl-portfolio-allocator/tests/test_walk_forward.py
git commit -m "feat: select and freeze factors per walk-forward fold"
```

### Task 11: Add six-factor control versus selected-20 candidate experiments

**Files:**
- Modify: `rl-portfolio-allocator/scripts/walk_forward.py`
- Modify: `rl-portfolio-allocator/scripts/research_gates.py`
- Modify: `rl-portfolio-allocator/tests/test_factor_pipeline.py`
- Modify: `rl-portfolio-allocator/tests/test_research_gates.py`

- [ ] **Step 1: Write comparison gate tests**

```python
def test_candidate_comparison_gates_can_pass():
    report = evaluate_candidate_gates({
        "candidate_median_oos_sharpe": 0.70,
        "control_median_oos_sharpe": 0.55,
        "positive_excess_return_folds": 2,
        "total_folds": 3,
        "candidate_cost_2x_oos_sharpe": 0.20,
        "candidate_annualized_turnover": 10.0,
        "candidate_stress_mdd": -0.25,
        "control_stress_mdd": -0.24,
        "fold_count": 3,
        "seed_count": 5,
    })
    assert report["research_ok"] is True


def test_candidate_gate_fails_when_sharpe_gain_is_below_point_one():
    summary = passing_candidate_summary()
    summary["candidate_median_oos_sharpe"] = 0.64
    summary["control_median_oos_sharpe"] = 0.55
    assert evaluate_candidate_gates(summary)["research_ok"] is False
```

- [ ] **Step 2: Verify RED**

Run candidate gate tests.

- [ ] **Step 3: Implement paired research branches**

For every fold/seed run the existing six-factor control and selected-20 candidate with identical budget and cost configuration. Write:

```text
artifacts/walk_forward/<run_id>/control_6f/<fold-and-seed-artifacts>
artifacts/walk_forward/<run_id>/candidate_20f/<fold-and-seed-artifacts>
artifacts/walk_forward/<run_id>/comparison.json
```

Add `evaluate_candidate_gates(summary)` with hard checks for Sharpe gain `>= 0.10`, at least two positive-excess folds, doubled-cost Sharpe `> 0`, turnover `<= 12`, stress MDD no worse than control by more than 0.05, and exactly three folds/five seeds.

Smoke writes these artifacts but forces `research_ok=false` and never writes approval.

- [ ] **Step 4: Run comparison and gate tests**

Run:

```bash
python -m pytest -q tests/test_factor_pipeline.py tests/test_walk_forward.py \
  tests/test_research_gates.py tests/test_production_gate.py
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add rl-portfolio-allocator/scripts/walk_forward.py \
  rl-portfolio-allocator/scripts/research_gates.py \
  rl-portfolio-allocator/tests/test_factor_pipeline.py \
  rl-portfolio-allocator/tests/test_research_gates.py
git commit -m "feat: gate selected factors against six-factor control"
```

### Task 12: Integrate stress, approval, publication, and shell pipeline

**Files:**
- Modify: `rl-portfolio-allocator/scripts/stress_test.py`
- Modify: `rl-portfolio-allocator/scripts/allocate.py`
- Modify: `rl-portfolio-allocator/run_pipeline.sh`
- Modify: `rl-portfolio-allocator/tests/test_production_gate.py`
- Modify: `rl-portfolio-allocator/tests/test_allocate_candidate.py`
- Modify: `rl-portfolio-allocator/tests/test_walk_forward.py`

- [ ] **Step 1: Write end-to-end contract tests**

```python
def test_smoke_never_writes_factor_approval(tmp_path, smoke_dependencies):
    run_walk_forward(output_root=tmp_path, smoke=True, **smoke_dependencies)
    assert not (tmp_path / "smoke" / "approval.json").exists()


def test_publish_requires_selected_factor_bundle(tmp_path, passing_approval):
    passing_approval["factor_selection_path"] = "missing.json"
    with pytest.raises((FileNotFoundError, ValueError)):
        load_research_approval(write_json(tmp_path / "approval.json", passing_approval))
```

- [ ] **Step 2: Verify RED**

Run targeted production tests.

- [ ] **Step 3: Complete pipeline wiring**

Change research order to:

```text
raw data and factor cache
-> data coverage
-> tests
-> fold-local control/candidate walk-forward
-> comparison gates
-> frozen control/candidate stress
```

Full passing approval includes relative paths and hashes for selected-factor artifacts. `--publish` copies and validates the selection bundle before retraining. Production inference loads the ordered selected factors from the approved method and materializes only those columns.

Keep `--all` as a non-publishing smoke alias. Do not relax existing approval checks.

- [ ] **Step 4: Run all automated tests**

Run:

```bash
cd rl-portfolio-allocator
python -m pytest -q
```

Expected: all tests pass with zero failures.

- [ ] **Step 5: Run a clean smoke with real cached data**

Run:

```bash
bash ../run_pipeline.sh --research-smoke
```

Expected:

- factor cache manifest reports 100 factors and ten families;
- Fold 3 selection reports exactly 20 frozen factors;
- both `control_6f` and `candidate_20f` produce validation/test artifacts;
- factor contract validates when checkpoints reload;
- `gates.json` remains non-publishable in smoke mode;
- `stress.json` is written;
- command exits zero despite expected smoke gate warning.

- [ ] **Step 6: Inspect generated artifacts without committing them**

Run:

```bash
python - <<'PY'
import json
from pathlib import Path

root = Path("artifacts/walk_forward/smoke")
selection = json.loads(next(Path("artifacts/factor_selection/smoke").glob("fold*/selected_factors.json")).read_text())
summary = json.loads((root / "summary.json").read_text())
assert len(selection["selected"]) == 20
assert {"control_6f", "candidate_20f"} <= set(summary)
assert summary["publishable"] is False
print("smoke factor selection and comparison artifacts verified")
PY
```

Expected: printed verification line and exit zero.

- [ ] **Step 7: Commit pipeline integration**

```bash
git add rl-portfolio-allocator/scripts/stress_test.py \
  rl-portfolio-allocator/scripts/allocate.py \
  rl-portfolio-allocator/run_pipeline.sh \
  rl-portfolio-allocator/tests/test_production_gate.py \
  rl-portfolio-allocator/tests/test_allocate_candidate.py \
  rl-portfolio-allocator/tests/test_walk_forward.py
git commit -m "feat: integrate selected-factor research pipeline"
```

### Task 13: Documentation, migration check, and final verification

**Files:**
- Modify: `rl-portfolio-allocator/SKILL.md`
- Create: `rl-portfolio-allocator/references/factor-selection.md`
- Verify: all source, tests, and artifacts contracts

- [ ] **Step 1: Document user-facing behavior**

Document:

- 100 candidates versus 20 selected factors;
- fold-local training-only selection;
- smoke versus full research meaning;
- output directory map;
- control-versus-candidate gates;
- production approval requirements;
- expected memory/disk implications of family caches.

- [ ] **Step 2: Run schema and global-dimension scans**

Run:

```bash
rg -n '\bFACTOR_NAMES\b|\bK\b' rl-portfolio-allocator/scripts
```

Expected: remaining references are catalog/control compatibility declarations or explicitly justified non-runtime uses. Runtime environment, baselines, market state, training, stress, and allocation must use config/method factor names.

- [ ] **Step 3: Run final tests and static checks**

Run:

```bash
cd rl-portfolio-allocator
python -m pytest -q
python -m compileall -q scripts tests
git diff --check
```

Expected: tests and compileall exit zero; `git diff --check` prints nothing.

- [ ] **Step 4: Verify no generated data is staged**

Run:

```bash
git status --short
git diff --cached --name-only
```

Expected: no paths below `rl-portfolio-allocator/data/` or `rl-portfolio-allocator/artifacts/` are staged.

- [ ] **Step 5: Commit documentation**

```bash
git add rl-portfolio-allocator/SKILL.md rl-portfolio-allocator/references/factor-selection.md
git commit -m "docs: explain selected-factor research workflow"
```

## Full research handoff

Do not automatically start the full run after implementation. It requires substantially more time than smoke and uses three folds, five seeds, two methods, reward candidates, buffer candidates, and 100,000 timesteps per training run.

After smoke passes, report the command and estimated workload to the user:

```bash
RLPA_RUN_ID=<explicit-run-id> bash run_pipeline.sh --research-full
```

Only a full run whose `gates.json` has `research_ok=true` may write an approval. Publication remains a separate explicit command:

```bash
bash run_pipeline.sh --publish --approval \
  rl-portfolio-allocator/artifacts/walk_forward/<run-id>/approval.json
```
