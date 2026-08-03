"""Leakage-safe walk-forward orchestration for the research-only OOS report."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib
import shutil
import sys
import tempfile
import uuid
from collections import defaultdict
from statistics import median
from typing import Callable

import numpy as np
import pandas as pd

if __package__ in (None, ""):
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from scripts.check_data_coverage import Fold, check_folds, default_folds
from scripts.factor_catalog import CATALOG_VERSION, FACTOR_CATALOG, catalog_hash
from scripts.config import CONTROL_FACTOR_NAMES
from scripts.research_gates import evaluate_candidate_gates, evaluate_research_gates
from scripts.state import STATE_SCHEMA_VERSION, state_fields
from scripts.market_state import build_market_state


class _FrozenDict(dict):
    """A recursively immutable dict that remains compatible with dict consumers."""

    def __init__(self, values=()):
        dict.__init__(self)
        for key, value in dict(values).items():
            dict.__setitem__(self, key, _freeze(value))

    @staticmethod
    def _immutable(*args, **kwargs):
        raise TypeError("frozen walk-forward inputs cannot be modified")

    __setitem__ = _immutable
    __delitem__ = _immutable
    clear = _immutable
    pop = _immutable
    popitem = _immutable
    setdefault = _immutable
    update = _immutable
    __ior__ = _immutable


class _FrozenList(list):
    """List-compatible immutable sequence for existing config consumers."""

    def __init__(self, values=()):
        list.__init__(self, (_freeze(value) for value in values))

    @staticmethod
    def _immutable(*args, **kwargs):
        raise TypeError("frozen walk-forward inputs cannot be modified")

    __setitem__ = _immutable
    __delitem__ = _immutable
    __iadd__ = _immutable
    __imul__ = _immutable
    append = _immutable
    clear = _immutable
    extend = _immutable
    insert = _immutable
    pop = _immutable
    remove = _immutable
    reverse = _immutable
    sort = _immutable


def _freeze(value):
    """Return an independent immutable value suitable for runtime config."""
    if isinstance(value, _FrozenDict):
        return value
    if isinstance(value, _FrozenList):
        return value
    if value is None or isinstance(value, (bool, int, float, str, bytes, pathlib.Path, np.generic)):
        return value
    if isinstance(value, np.ndarray):
        if value.dtype.hasobject:
            raise TypeError(
                "unsupported mutable walk-forward config value: numpy.ndarray with object dtype"
            )
        frozen = value.copy()
        frozen.flags.writeable = False
        return frozen
    if isinstance(value, dict):
        return _FrozenDict(value)
    if isinstance(value, list):
        return _FrozenList(value)
    if isinstance(value, tuple):
        return tuple(_freeze(item) for item in value)
    if isinstance(value, set):
        return frozenset(_freeze(item) for item in value)
    raise TypeError(
        "unsupported mutable walk-forward config value: "
        f"{type(value).__module__}.{type(value).__qualname__}"
    )


class _MaterializedFoldInputs:
    """Keep canonical fold panels private while exposing immutable metadata."""

    def __init__(self, factor_bundle: dict, features: pd.DataFrame, market_state: pd.DataFrame):
        if not isinstance(features, pd.DataFrame) or not isinstance(market_state, pd.DataFrame):
            raise TypeError("prepared fold features and market_state must be pandas DataFrames")
        metadata_fields = (
            "fold", "names", "directions", "factor_contract", "feature_path",
            "market_state_path", "selection_artifact_path",
        )
        missing = set(metadata_fields) - set(factor_bundle)
        if missing:
            raise ValueError(f"prepared fold metadata missing fields: {sorted(missing)}")
        self.factor_bundle = _freeze({key: factor_bundle[key] for key in metadata_fields})
        self._features = features.copy(deep=True)
        self._market_state = market_state.copy(deep=True)


def frozen_method_id(method: dict) -> str:
    payload = json.dumps(method, sort_keys=True, separators=(",", ":")).encode()
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def artifact_id(path: pathlib.Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def write_approval(run_root: pathlib.Path, method: dict, gate_report: dict,
                   run_id: str, created_at: str, *, run_mode: str = "full",
                   fold_count: int | None = None, seed_count: int | None = None,
                   selection_artifact_path=None):
    if gate_report.get("research_ok") is not True:
        return None
    from scripts.train import require_factor_contract

    require_factor_contract(method, context="approved method")
    if selection_artifact_path is None:
        raise ValueError("approval requires the persisted selected-factor artifact")
    run_root.mkdir(parents=True, exist_ok=True)
    method_path = run_root / "method.json"
    gates_path = run_root / "gates.json"
    comparison_path = run_root / "comparison.json"
    if not comparison_path.is_file():
        raise FileNotFoundError("approval requires comparison.json evidence")
    comparison_id = artifact_id(comparison_path)
    if gate_report.get("comparison_path") != "comparison.json" or gate_report.get("comparison_id") != comparison_id:
        raise ValueError("approval gates must bind comparison.json evidence")
    selection_path = pathlib.Path(selection_artifact_path)
    if not selection_path.is_file():
        raise FileNotFoundError("approval requires persisted selected factors: "
                                f"{selection_path}")
    try:
        selection_reference = selection_path.resolve().relative_to(run_root.resolve())
    except ValueError:
        raise ValueError("selected-factor artifact must live inside the run root") from None
    method_path.write_text(json.dumps(_jsonable(method), indent=2, sort_keys=True), encoding="utf-8")
    gates_path.write_text(json.dumps(_jsonable(gate_report), indent=2, sort_keys=True), encoding="utf-8")
    approval = {
        "research_ok": True,
        "schema_version": method.get("schema_version", STATE_SCHEMA_VERSION),
        "method_id": frozen_method_id(method), "method_path": "method.json",
        "gates_path": "gates.json", "run_id": run_id, "created_at": created_at,
        "comparison_path": "comparison.json", "comparison_id": comparison_id,
        "factor_selection_path": str(selection_reference),
        "factor_selection_id": artifact_id(selection_path),
        "run_mode": run_mode, "fold_count": fold_count, "seed_count": seed_count,
    }
    path = run_root / "approval.json"
    path.write_text(json.dumps(approval, indent=2, sort_keys=True), encoding="utf-8")
    return path


SEEDS = (0, 1, 2, 3, 4)
DEFAULT_REWARD_CANDIDATES = ("none", "gentle", "low", "constrained", "legacy_dsr")


def reward_candidates() -> tuple[str, ...]:
    """奖励候选集,可用 RLPA_REWARD_CANDIDATES 环境变量覆盖(逗号分隔)。"""
    raw = os.environ.get("RLPA_REWARD_CANDIDATES")
    if not raw:
        return DEFAULT_REWARD_CANDIDATES
    return tuple(name.strip() for name in raw.split(",") if name.strip())
BUFFER_CANDIDATES = ("tight", "default", "wide")
BUFFER_CONFIGS = {
    "tight": {"long_entry": 30, "long_exit": 40, "short_entry": 15, "short_exit": 25},
    "default": {"long_entry": 30, "long_exit": 45, "short_entry": 15, "short_exit": 30},
    "wide": {"long_entry": 30, "long_exit": 60, "short_entry": 15, "short_exit": 45},
}


def select_candidate_on_validation(rows):
    """Select the highest-median candidate without inspecting test rows."""
    scores = defaultdict(list)
    for row in rows:
        scores[row["candidate"]].append(float(row["val_sharpe"]))
    if not scores:
        raise ValueError("validation rows must not be empty")
    return max(scores, key=lambda candidate: (median(scores[candidate]), -list(scores).index(candidate)))


def _jsonable(value):
    if hasattr(value, "tolist"):
        return _jsonable(value.tolist())
    if isinstance(value, pathlib.Path):
        return str(value)
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    return value


def _write(path: pathlib.Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(payload), ensure_ascii=False, indent=2), encoding="utf-8")


def _invoke_trainer(trainer: Callable, **kwargs) -> dict:
    result = trainer(**kwargs)
    if "val_sharpe" not in result:
        raise ValueError("trainer result must contain val_sharpe")
    return result


def _checkpoint_factor_contract(
    cfg: dict | None,
    *,
    fold: int | None = None,
    factor_contract: dict | None = None,
    run_id: str | None = None,
) -> dict:
    """Return the explicit fold contract used by training and testing.

    Walk-forward may receive the selected-factor records directly, through a
    nested ``factor_contract`` or through a fold-indexed contract map.  It must
    never invent directions or selection identity: the resulting method is a
    publishable artifact and therefore requires all canonical contract fields.
    """
    cfg = dict(cfg or {})
    source: dict = {}
    by_fold = cfg.get("factor_contract_by_fold")
    if isinstance(by_fold, dict) and fold is not None:
        fold_source = by_fold.get(str(fold), by_fold.get(fold))
        if isinstance(fold_source, dict):
            source.update(fold_source)
    nested = cfg.get("factor_contract")
    if isinstance(nested, dict):
        source.update(nested)
    source.update({
        key: cfg[key]
        for key in (
            "factor_catalog_version", "factor_catalog_hash", "catalog_version",
            "catalog_hash", "selected_factors", "factor_names", "factor_directions",
            "selection_run_id", "fold", "state_schema_version", "schema_version",
        )
        if key in cfg
    })
    if factor_contract is not None:
        source.update(factor_contract)

    selected = source.get("selected_factors")
    directions = source.get("factor_directions")
    if isinstance(selected, (list, tuple)) and selected and isinstance(selected[0], dict):
        records = list(selected)
        source["selected_factors"] = [record.get("name") for record in records]
        if directions is None:
            source["factor_directions"] = [record.get("direction") for record in records]
    if "factor_catalog_version" not in source and "catalog_version" in source:
        source["factor_catalog_version"] = source["catalog_version"]
    if "factor_catalog_hash" not in source and "catalog_hash" in source:
        source["factor_catalog_hash"] = source["catalog_hash"]
    if "selection_run_id" not in source:
        source["selection_run_id"] = run_id
    source["fold"] = fold if fold is not None else source.get("fold")
    from scripts.train import require_factor_contract

    return require_factor_contract(source, context="walk-forward factor contract")


def _cache_panel(candidate_cache, cache_root, selected):
    if candidate_cache is None:
        from scripts.factor_cache import materialize_selected_panel
        return materialize_selected_panel(cache_root, selected)
    materialize = getattr(candidate_cache, "materialize_selected_panel", candidate_cache)
    return materialize(cache_root, selected)


def _selected_records(selection):
    selected = getattr(selection, "selected", selection)
    records = [dict(item) for item in selected]
    if not records or any(set(("name", "direction")) - set(item) for item in records):
        raise ValueError("factor selection must return ordered name/direction records")
    return records


def _default_fold_selector(*, panel, fold, cfg):
    from scripts.factor_selection import (
        compute_factor_metrics, select_factors, write_selection_artifacts,
    )

    names = [spec.name for spec in FACTOR_CATALOG]
    train = panel.loc[
        (pd.to_datetime(panel["trade_date"]) >= pd.Timestamp(fold.train[0]))
        & (pd.to_datetime(panel["trade_date"]) <= pd.Timestamp(fold.train[1]))
    ].copy()
    metrics = compute_factor_metrics(panel, names, *fold.train, cfg)
    correlations = train.loc[:, names].corr().fillna(0.0)
    result = select_factors(
        metrics, correlations, correlations,
        target_count=int(cfg.get("selection_target_count", 20)),
    )
    return result, metrics, {"return_corr": correlations, "cross_section_corr": correlations}


_SELECTION_CACHE_VERSION = 1  # bump when factor metrics or selection logic changes


def _selection_cache_entry(cache_root, fold, cfg) -> pathlib.Path | None:
    """Content-keyed cache entry covering inputs, fold window, config, and code version."""
    try:
        catalog = json.loads(
            (pathlib.Path(cache_root) / "catalog.json").read_text(encoding="utf-8")
        )
        payload = {
            "version": _SELECTION_CACHE_VERSION,
            "catalog_hash": catalog_hash(FACTOR_CATALOG),
            "train": [str(fold.train[0]), str(fold.train[1])],
            "row_count": catalog["row_count"],
            "file_hashes": catalog["file_hashes"],
            "cfg": _jsonable(dict(cfg)),
        }
        canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    except (OSError, KeyError, TypeError, ValueError):
        return None
    key = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return pathlib.Path(cache_root).parent / "selection_cache" / key


def _read_selection_cache(cache_root, fold, cfg):
    """Return the cached selector output, or None on any miss or inconsistency."""
    entry = _selection_cache_entry(cache_root, fold, cfg)
    if entry is None:
        return None
    from scripts.factor_selection import SelectionResult

    try:
        selected_payload = json.loads((entry / "selected.json").read_text(encoding="utf-8"))
        metrics = json.loads((entry / "metrics.json").read_text(encoding="utf-8"))
        correlations = pd.read_parquet(entry / "correlations.parquet")
        result = SelectionResult(
            selected=tuple(dict(item) for item in selected_payload["selected"]),
            relaxation_log=tuple(dict(item) for item in selected_payload["relaxation_log"]),
            final_family_cap=int(selected_payload["final_family_cap"]),
            final_correlation_ceiling=float(selected_payload["final_correlation_ceiling"]),
            catalog_hash=str(selected_payload["catalog_hash"]),
        )
        if result.catalog_hash != catalog_hash(FACTOR_CATALOG):
            return None
        if not isinstance(metrics, dict) or not metrics:
            return None
    except (OSError, KeyError, TypeError, ValueError):
        return None
    return result, metrics, {"return_corr": correlations, "cross_section_corr": correlations}


def _write_selection_cache(cache_root, fold, cfg, selection) -> None:
    """Best-effort cache write; a failure only means the next run recomputes."""
    entry = _selection_cache_entry(cache_root, fold, cfg)
    if entry is None:
        return
    try:
        result, metrics, correlations = selection
        frame = correlations["return_corr"]
        if not isinstance(frame, pd.DataFrame):
            return
        root = entry.parent
        root.mkdir(parents=True, exist_ok=True)
        stage = pathlib.Path(tempfile.mkdtemp(prefix=f".{entry.name}.", dir=str(root)))
        try:
            _write(stage / "selected.json", {
                "selected": list(result.selected),
                "relaxation_log": list(result.relaxation_log),
                "final_family_cap": result.final_family_cap,
                "final_correlation_ceiling": result.final_correlation_ceiling,
                "catalog_hash": result.catalog_hash,
            })
            _write(stage / "metrics.json", metrics)
            frame.to_parquet(stage / "correlations.parquet", index=True)
            if entry.exists():
                shutil.rmtree(entry)
            stage.rename(entry)
        except Exception:
            shutil.rmtree(stage, ignore_errors=True)
            raise
    except Exception:  # cache writes must never break a run
        return


def prepare_fold_factors(*, fold, cache_root, index_returns, selection_root,
                         feature_root, state_root, cfg, overwrite_selection=False,
                         selector=None, candidate_cache=None,
                         expected_factor_count=None, factor_bundle_name="selected"):
    """Select on one fold's train interval and freeze inputs through its test end."""
    cfg = dict(cfg or {})
    forbidden_dependencies = {"selector", "candidate_cache"} & set(cfg)
    if forbidden_dependencies:
        names = ", ".join(sorted(forbidden_dependencies))
        raise ValueError(f"{names} must be passed explicitly to prepare_fold_factors")
    cfg = _FrozenDict(cfg)
    selector = selector or _default_fold_selector
    cacheable = selector is _default_fold_selector and candidate_cache is None and cache_root is not None
    selected = _read_selection_cache(cache_root, fold, cfg) if cacheable else None
    if selected is None:
        all_candidates = [{"name": spec.name, "direction": 1} for spec in FACTOR_CATALOG]
        full_panel = _cache_panel(candidate_cache, cache_root, all_candidates)
        dates = pd.to_datetime(full_panel["trade_date"])
        train_panel = full_panel.loc[
            (dates >= pd.Timestamp(fold.train[0])) & (dates <= pd.Timestamp(fold.train[1]) )
        ].copy()
        selected = selector(panel=train_panel, fold=fold, cfg=cfg)
        if cacheable:
            _write_selection_cache(cache_root, fold, cfg, selected)
    selection_details = None
    if isinstance(selected, tuple) and selected and not isinstance(selected[0], dict):
        selected, *selection_details = selected
    records = _selected_records(selected)
    if expected_factor_count is not None and len(records) != expected_factor_count:
        raise ValueError(
            f"{factor_bundle_name} factor selection must contain exactly "
            f"{expected_factor_count} ordered factors; got {len(records)}"
        )
    selection_dir = pathlib.Path(selection_root) / f"fold{fold.fold}"
    if overwrite_selection and selection_dir.exists():
        shutil.rmtree(selection_dir)
    selection_dir.mkdir(parents=True, exist_ok=True)
    selection_artifact = selection_dir / "selected_factors.json"
    if selection_details:
        from scripts.factor_selection import write_selection_artifacts
        metrics, correlations = selection_details
        write_selection_artifacts(selected, metrics, correlations, selection_dir, fold.fold, fold.train)
    else:
        _write(selection_artifact, {"fold": fold.fold, "train_range": fold.train,
                                    "selected_factors": records})

    frozen_features = _cache_panel(candidate_cache, cache_root, records)
    feature_dates = pd.to_datetime(frozen_features["trade_date"])
    frozen_features = frozen_features.loc[feature_dates <= pd.Timestamp(fold.test[1])].copy()
    frozen_state = build_market_state(
        frozen_features, index_returns, cfg, factor_names=[item["name"] for item in records]
    )
    state_dates = pd.to_datetime(frozen_state["trade_date"])
    frozen_state = frozen_state.loc[state_dates <= pd.Timestamp(fold.test[1])].copy()
    feature_path = pathlib.Path(feature_root) / f"fold{fold.fold}.parquet"
    market_state_path = pathlib.Path(state_root) / f"fold{fold.fold}.parquet"
    feature_path.parent.mkdir(parents=True, exist_ok=True)
    market_state_path.parent.mkdir(parents=True, exist_ok=True)
    frozen_features.to_parquet(feature_path, index=False)
    frozen_state.to_parquet(market_state_path, index=False)
    names = tuple(item["name"] for item in records)
    directions = tuple(int(item["direction"]) for item in records)
    contract_cfg = {
        **cfg,
        "factor_catalog_version": CATALOG_VERSION,
        "factor_catalog_hash": catalog_hash(FACTOR_CATALOG),
        "factor_names": list(names), "factor_directions": list(directions),
        "selection_run_id": f"fold-{fold.fold}",
        "state_schema_version": cfg.get("state_schema_version", STATE_SCHEMA_VERSION),
        "schema_version": cfg.get("schema_version", cfg.get("state_schema_version", STATE_SCHEMA_VERSION)),
    }
    contract = _checkpoint_factor_contract(contract_cfg, fold=fold.fold)
    factor_bundle = {
        "fold": fold.fold, "names": names, "directions": directions,
        "factor_contract": contract,
        "feature_path": feature_path, "market_state_path": market_state_path,
        "selection_artifact_path": selection_artifact,
    }
    return _MaterializedFoldInputs(factor_bundle, frozen_features, frozen_state)


def prepare_control_fold_factors(*, fold, cache_root, index_returns, selection_root,
                                 feature_root, state_root, cfg, candidate_cache=None):
    """Materialize the fixed six-factor control using the candidate run's settings."""
    configured_names = list(dict(cfg or {}).get("factor_names", ()))
    configured_directions = list(dict(cfg or {}).get("factor_directions", ()))
    direction_by_name = dict(zip(configured_names, configured_directions))
    selected = [
        {"name": name, "direction": int(direction_by_name.get(name, 1))}
        for name in CONTROL_FACTOR_NAMES
    ]
    return prepare_fold_factors(
        fold=fold, cache_root=cache_root, index_returns=index_returns,
        selection_root=selection_root, feature_root=feature_root, state_root=state_root,
        cfg=cfg, selector=lambda **_: selected, candidate_cache=candidate_cache,
        expected_factor_count=6, factor_bundle_name="control",
    )


def _fold_runtime_cfg(base_cfg: dict | None, factor_bundle: dict) -> dict:
    """Copy base config and bind it to one frozen fold factor contract."""
    contract = factor_bundle["factor_contract"]
    cfg = {
        key: value for key, value in dict(base_cfg or {}).items()
        if key not in {"selector", "candidate_cache"}
    }
    cfg.update({
        "factor_names": list(contract["selected_factors"]),
        "factor_directions": list(contract["factor_directions"]),
        "k": len(contract["selected_factors"]),
        "factor_contract": contract,
    })
    return _FrozenDict(cfg)


def _fold_input_copies(fold_inputs: _MaterializedFoldInputs) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return private mutable views of a fold's canonical materialized panels."""
    return fold_inputs._features.copy(deep=True), fold_inputs._market_state.copy(deep=True)


def aggregate_research_summary(test_rows, total_folds: int) -> dict:
    """Aggregate supplied per-test metrics without inventing missing evidence."""
    def median_field(name):
        values = [float(row[name]) for row in test_rows if row.get(name) is not None]
        return median(values) if values else None
    by_fold = defaultdict(list)
    for row in test_rows:
        by_fold[row.get("fold")].append(row)
    positive_folds = 0
    for rows in by_fold.values():
        excess = [float(row["excess_return"]) for row in rows if row.get("excess_return") is not None]
        if excess and median(excess) > 0:
            positive_folds += 1
    return {
        "combined_oos_arr": median_field("oos_arr"),
        "median_seed_oos_sharpe": median_field("oos_sharpe"),
        "strongest_baseline_sharpe": median_field("strongest_baseline_sharpe"),
        "positive_excess_return_folds": positive_folds,
        "total_folds": total_folds,
        "median_seed_excess_return": median_field("excess_return"),
        "oos_mdd": median_field("oos_mdd"),
        "strongest_baseline_mdd": median_field("strongest_baseline_mdd"),
        "annualized_turnover": median_field("annualized_turnover"),
        "cost_2x_oos_sharpe": median_field("cost_2x_oos_sharpe"),
        "no_leakage_tests_passed": all(row.get("no_leakage_tests_passed") is True for row in test_rows) if test_rows else None,
        "state_quality_tests_passed": all(row.get("state_quality_tests_passed") is True for row in test_rows) if test_rows else None,
    }


def aggregate_candidate_comparison(candidate_rows, control_rows, *, fold_count: int, seed_count: int) -> dict:
    """Summarize paired OOS evidence without filling in unavailable metrics."""
    del fold_count, seed_count  # Configured counts are not evidence of a complete paired run.

    def median_field(rows, name):
        values = []
        for row in rows:
            if not isinstance(row, dict) or row.get(name) is None:
                continue
            try:
                value = float(row[name])
            except (TypeError, ValueError):
                continue
            if np.isfinite(value):
                values.append(value)
        return median(values) if values else None

    control_by_key = {
        (row.get("fold"), row.get("seed")): row
        for row in control_rows
        if isinstance(row, dict) and "fold" in row and "seed" in row
    }
    positive_excess_folds = 0
    for fold in {row.get("fold") for row in candidate_rows if isinstance(row, dict) and "fold" in row}:
        excess = []
        for row in candidate_rows:
            if not isinstance(row, dict) or row.get("fold") != fold or row.get("seed") is None:
                continue
            control = control_by_key.get((row["fold"], row["seed"]), {})
            try:
                candidate_sharpe = float(row["oos_sharpe"])
                control_sharpe = float(control["oos_sharpe"])
            except (KeyError, TypeError, ValueError):
                continue
            if np.isfinite(candidate_sharpe) and np.isfinite(control_sharpe):
                excess.append(candidate_sharpe - control_sharpe)
        if excess and median(excess) > 0:
            positive_excess_folds += 1
    return {
        "candidate_median_oos_sharpe": median_field(candidate_rows, "oos_sharpe"),
        "control_median_oos_sharpe": median_field(control_rows, "oos_sharpe"),
        "positive_excess_folds": positive_excess_folds,
        "candidate_cost_2x_oos_sharpe": median_field(candidate_rows, "cost_2x_oos_sharpe"),
        "candidate_annualized_turnover": median_field(candidate_rows, "annualized_turnover"),
        "candidate_stress_mdd": median_field(candidate_rows, "stress_mdd"),
        "control_stress_mdd": median_field(control_rows, "stress_mdd"),
        "candidate_stress_calmar_excess": median_field(candidate_rows, "stress_calmar_excess"),
        "candidate_stress_long_exposure_util": median_field(candidate_rows, "stress_long_exposure_util"),
        "paired_evidence": {
            "candidate_20f": {"rows": candidate_rows},
            "control_6f": {"rows": control_rows},
        },
    }


def _frozen_method(*, factor_contract: dict, frozen_candidate: str,
                   reward_variant: str, buffer_variant: str,
                   training_budget: int) -> dict:
    return {
        **factor_contract,
        "frozen_candidate": frozen_candidate,
        "schema_version": factor_contract["state_schema_version"],
        "reward_variant": reward_variant,
        "buffer_variant": buffer_variant,
        "buffer_config": BUFFER_CONFIGS[buffer_variant],
        "training_budget": training_budget,
    }


def _default_stress_tester(**kwargs) -> list:
    """Run the real frozen-method stress suite for one branch/fold/seed pair."""
    from scripts.stress_test import run_all_stress

    return run_all_stress(
        kwargs["features_df"], kwargs["market_state_df"], kwargs["cfg"],
        timesteps=kwargs["timesteps"], method=kwargs["method"],
        seed=kwargs["seed"], checkpoint_path=kwargs.get("checkpoint_path"),
        branch=kwargs["branch"], fold=kwargs["fold"],
    )


def _stress_evidence(results: list) -> tuple[float | None, float | None, float | None, list[dict]]:
    """Return the worst actual RL stress evidence, never an OOS substitute.

    Worst = min across non-skipped segments. stress_calmar_excess is the RL
    calmar minus the static_factor_equal baseline calmar within the same
    segment; long_exposure_util comes from the rollout diagnostics.
    """
    mdds, calmar_excesses, utils = [], [], []
    evidence = []
    for result in results:
        record = {"name": result.get("name"), "skipped": bool(result.get("skipped"))}
        if result.get("skipped"):
            record["reason"] = result.get("reason")
        else:
            metrics = result.get("metrics", {})
            mdd = metrics.get("rl", {}).get("mdd")
            record["stress_mdd"] = mdd
            if mdd is not None:
                mdds.append(float(mdd))
            rl_calmar = metrics.get("rl", {}).get("calmar")
            sfe_calmar = metrics.get("static_factor_equal", {}).get("calmar")
            calmar_excess = (
                float(rl_calmar) - float(sfe_calmar)
                if rl_calmar is not None and sfe_calmar is not None
                else None
            )
            record["stress_calmar_excess"] = calmar_excess
            if calmar_excess is not None:
                calmar_excesses.append(calmar_excess)
            util = result.get("diagnostics", {}).get("long_exposure_util")
            record["long_exposure_util"] = util
            if util is not None:
                utils.append(float(util))
        evidence.append(record)
    return (
        min(mdds) if mdds else None,
        min(calmar_excesses) if calmar_excesses else None,
        min(utils) if utils else None,
        evidence,
    )


def run_walk_forward(*, folds=None, output_root, smoke=False, trainer=None, tester=None,
                     coverage_checker=None, features_df=None, index_df=None,
                     cfg=None, timesteps=None, run_id=None, selector=None,
                     candidate_cache=None, cache_root=None, index_returns=None,
                     stress_tester=None) -> dict:
    """Run validation-only candidate selection followed by one frozen test per seed."""
    cfg = dict(cfg or {})
    forbidden_dependencies = {"selector", "candidate_cache"} & set(cfg)
    if forbidden_dependencies:
        names = ", ".join(sorted(forbidden_dependencies))
        raise ValueError(f"{names} must be passed explicitly to run_walk_forward")
    root = pathlib.Path(output_root)
    run_id = "smoke" if smoke else (run_id or uuid.uuid4().hex)
    run_root = root / "smoke" if smoke else root / run_id
    runtime_cfg = _FrozenDict(cfg)
    if smoke:
        if run_root.exists():
            shutil.rmtree(run_root)
        run_root.mkdir(parents=True, exist_ok=False)
    else:
        try:
            run_root.mkdir(parents=True, exist_ok=False)
        except FileExistsError as exc:
            raise FileExistsError(
                f"full walk-forward run directory is immutable and already exists: {run_root}"
            ) from exc
    folds = list(folds or default_folds())
    if coverage_checker is not None:
        coverage_checker()
    elif features_df is not None and index_df is not None:
        report = check_folds(features_df, index_df, folds)
        if len(report["usable_folds"]) != len(folds):
            raise ValueError("data coverage check rejected one or more walk-forward folds")
    if trainer is None or tester is None:
        raise ValueError("trainer and tester dependency injections are required")
    if not smoke:
        # Full research is publishable only with fresh, real stress output.
        # Custom test/stress injections are smoke-only conveniences.
        stress_tester = _default_stress_tester
    elif stress_tester is None and tester is _default_tester:
        stress_tester = _default_stress_tester
    has_factor_cache = cache_root is not None or candidate_cache is not None
    if not has_factor_cache:
        raise ValueError(
            "fold-local factor cache/input is required; static cfg/data inputs cannot bypass selection/materialization"
        )

    selected_folds = [folds[-1]] if smoke else folds
    selected_seeds = (0,) if smoke else SEEDS
    candidate_root = run_root / "candidate_20f"
    control_root = run_root / "control_6f"
    validation_root = candidate_root / "validation"
    test_root = candidate_root / "test"
    validation_rows = []
    reward_results = {}
    selected_rewards = {}
    if index_returns is None:
        raise ValueError("index_returns is required for fold-local factor preparation")
    prepared_folds = {}
    for fold in selected_folds:
        prepared_folds[fold.fold] = prepare_fold_factors(
            fold=fold, cache_root=cache_root, index_returns=index_returns,
            selection_root=candidate_root / "selection", feature_root=candidate_root / "features",
            state_root=candidate_root / "state", cfg=runtime_cfg, overwrite_selection=smoke,
            selector=selector, candidate_cache=candidate_cache,
            expected_factor_count=int(runtime_cfg.get("selection_target_count", 20)),
            factor_bundle_name="candidate",
        )
    control_folds = {
        fold.fold: prepare_control_fold_factors(
            fold=fold, cache_root=cache_root, index_returns=index_returns,
            selection_root=control_root / "selection", feature_root=control_root / "features",
            state_root=control_root / "state", cfg=runtime_cfg, candidate_cache=candidate_cache,
        )
        for fold in selected_folds
    }
    fold_contracts = {
        fold.fold: prepared_folds[fold.fold].factor_bundle["factor_contract"]
        for fold in selected_folds
    }
    fold_runtime_cfgs = {
        fold.fold: _fold_runtime_cfg(runtime_cfg, prepared_folds[fold.fold].factor_bundle)
        for fold in selected_folds
    }
    control_contracts = {
        fold.fold: control_folds[fold.fold].factor_bundle["factor_contract"]
        for fold in selected_folds
    }
    control_runtime_cfgs = {
        fold.fold: _fold_runtime_cfg(runtime_cfg, control_folds[fold.fold].factor_bundle)
        for fold in selected_folds
    }

    def bind_result_contract(result: dict, expected: dict) -> dict:
        required = ("factor_contract", "checkpoint_path", "metadata_path", "scaler_path")
        missing = [key for key in required if not result.get(key)]
        if missing:
            raise ValueError("trainer result must contain " + ", ".join(missing))
        from scripts.observation import ObservationScaler
        from scripts.state import state_fields
        from scripts.train import require_factor_contract, validate_factor_contract

        actual = require_factor_contract(
            result["factor_contract"], context="trainer factor contract"
        )
        validate_factor_contract(actual, expected)
        checkpoint = pathlib.Path(result["checkpoint_path"])
        metadata = pathlib.Path(result["metadata_path"])
        scaler = pathlib.Path(result["scaler_path"])
        if not checkpoint.exists() or not metadata.exists() or not scaler.exists():
            raise ValueError("trainer result artifact paths must exist")
        metadata_contract = require_factor_contract(
            json.loads(metadata.read_text(encoding="utf-8")),
            context="trainer checkpoint metadata",
        )
        validate_factor_contract(metadata_contract, expected)
        ObservationScaler.load(
            scaler, expected_schema=expected["state_schema_version"],
            expected_fields=tuple(state_fields(expected["selected_factors"])),
            expected_factor_contract=expected,
        )
        result["factor_contract"] = actual
        return result

    # Phase 1: default rank buffer, reward ablation. Tests are impossible here.
    for fold in selected_folds:
        fold_inputs = prepared_folds[fold.fold]
        factor_bundle = fold_inputs.factor_bundle
        fold_cfg = fold_runtime_cfgs[fold.fold]
        for reward in reward_candidates():
            for seed in selected_seeds:
                fold_features, fold_state = _fold_input_copies(fold_inputs)
                result = _invoke_trainer(
                    trainer, stage="reward_ablation", fold=fold.fold, seed=seed,
                    candidate=reward, reward_variant=reward, buffer_variant="default",
                    buffer_config=BUFFER_CONFIGS["default"], train_range=fold.train,
                    val_range=fold.val, test_range=fold.test, timesteps=timesteps or (128 if smoke else 100_000),
                    cfg=fold_cfg, artifact_dir=validation_root / f"fold{fold.fold}" / reward / f"seed{seed}",
                    features_df=fold_features, market_state_df=fold_state,
                    factor_contract=fold_contracts[fold.fold],
                    factor_bundle=factor_bundle,
                    branch="candidate_20f",
                    run_id=run_id,
                )
                result = bind_result_contract(result, fold_contracts[fold.fold])
                row = {"fold": fold.fold, "seed": seed, "candidate": reward,
                       "val_sharpe": float(result["val_sharpe"]), "stage": "reward_ablation",
                       "trainer_result": _jsonable(result)}
                reward_results[(fold.fold, seed, reward)] = result
                validation_rows.append(row)
                _write(validation_root / f"fold{fold.fold}" / "reward" / f"{reward}_seed{seed}.json", row)
        selected_rewards[fold.fold] = select_candidate_on_validation(
            [row for row in validation_rows if row["fold"] == fold.fold]
        )

    # Phase 2: buffer ablation with the reward choice frozen by validation only.
    buffer_rows = []
    buffer_results = {}
    selected_buffers = {}
    for fold in selected_folds:
        fold_inputs = prepared_folds[fold.fold]
        factor_bundle = fold_inputs.factor_bundle
        fold_cfg = fold_runtime_cfgs[fold.fold]
        selected_reward = selected_rewards[fold.fold]
        for buffer in BUFFER_CANDIDATES:
            for seed in selected_seeds:
                fold_features, fold_state = _fold_input_copies(fold_inputs)
                result = _invoke_trainer(
                    trainer, stage="buffer_ablation", fold=fold.fold, seed=seed,
                    candidate=buffer, reward_variant=selected_reward, buffer_variant=buffer,
                    buffer_config=BUFFER_CONFIGS[buffer], train_range=fold.train,
                    val_range=fold.val, test_range=fold.test, timesteps=timesteps or (128 if smoke else 100_000),
                    cfg=fold_cfg, artifact_dir=validation_root / f"fold{fold.fold}" / buffer / f"seed{seed}",
                    features_df=fold_features, market_state_df=fold_state,
                    factor_contract=fold_contracts[fold.fold],
                    factor_bundle=factor_bundle,
                    branch="candidate_20f",
                    run_id=run_id,
                )
                result = bind_result_contract(result, fold_contracts[fold.fold])
                row = {"fold": fold.fold, "seed": seed, "candidate": buffer,
                       "val_sharpe": float(result["val_sharpe"]), "stage": "buffer_ablation",
                       "reward_variant": selected_reward, "trainer_result": _jsonable(result)}
                buffer_results[(fold.fold, seed, buffer)] = result
                buffer_rows.append(row)
                _write(validation_root / f"fold{fold.fold}" / "buffer" / f"{buffer}_seed{seed}.json", row)
        selected_buffers[fold.fold] = select_candidate_on_validation(
            [row for row in buffer_rows if row["fold"] == fold.fold]
        )

    # Phase 3: freeze the method, then evaluate each seed exactly once on test.
    candidate_test_rows = []
    control_test_rows = []
    for fold in selected_folds:
        fold_inputs = prepared_folds[fold.fold]
        factor_bundle = fold_inputs.factor_bundle
        fold_cfg = fold_runtime_cfgs[fold.fold]
        selected_reward = selected_rewards[fold.fold]
        selected_buffer = selected_buffers[fold.fold]
        frozen_candidate = f"{selected_reward}__{selected_buffer}"
        for seed in selected_seeds:
            validation_result = buffer_results[(fold.fold, seed, selected_buffer)]
            checkpoint_path = validation_result.get("checkpoint_path")
            fold_features, fold_state = _fold_input_copies(fold_inputs)
            result = tester(
                fold=fold.fold, seed=seed, candidate=frozen_candidate,
                reward_variant=selected_reward, buffer_variant=selected_buffer,
                buffer_config=BUFFER_CONFIGS[selected_buffer], train_range=fold.train,
                val_range=fold.val, test_range=fold.test, cfg=fold_cfg,
                artifact_dir=test_root / f"fold{fold.fold}" / frozen_candidate / f"seed{seed}",
                validation_result=validation_result, checkpoint_path=checkpoint_path,
                features_df=fold_features, market_state_df=fold_state,
                factor_contract=fold_contracts[fold.fold],
                factor_bundle=factor_bundle,
                branch="candidate_20f",
                run_id=run_id,
            )
            row = {"fold": fold.fold, "seed": seed, "candidate": frozen_candidate,
                   "branch": "candidate_20f", "reward_variant": selected_reward,
                   "buffer_variant": selected_buffer, "checkpoint_path": checkpoint_path,
                   **result}
            candidate_test_rows.append(row)
            _write(test_root / f"fold{fold.fold}" / frozen_candidate / f"seed{seed}.json", row)
            control_inputs = control_folds[fold.fold]
            control_bundle = control_inputs.factor_bundle
            control_cfg = control_runtime_cfgs[fold.fold]
            control_features, control_state = _fold_input_copies(control_inputs)
            control_validation = _invoke_trainer(
                trainer, stage="paired_control", fold=fold.fold, seed=seed,
                candidate=frozen_candidate, reward_variant=selected_reward,
                buffer_variant=selected_buffer, buffer_config=BUFFER_CONFIGS[selected_buffer],
                train_range=fold.train, val_range=fold.val, test_range=fold.test,
                timesteps=timesteps or (128 if smoke else 100_000), cfg=control_cfg,
                artifact_dir=control_root / "train" / f"fold{fold.fold}" / f"seed{seed}",
                features_df=control_features, market_state_df=control_state,
                factor_contract=control_contracts[fold.fold], factor_bundle=control_bundle,
                branch="control_6f", run_id=run_id,
            )
            control_validation = bind_result_contract(control_validation, control_contracts[fold.fold])
            control_features, control_state = _fold_input_copies(control_inputs)
            control_result = tester(
                fold=fold.fold, seed=seed, candidate=frozen_candidate,
                reward_variant=selected_reward, buffer_variant=selected_buffer,
                buffer_config=BUFFER_CONFIGS[selected_buffer], train_range=fold.train,
                val_range=fold.val, test_range=fold.test, cfg=control_cfg,
                artifact_dir=control_root / "test" / f"fold{fold.fold}" / f"seed{seed}",
                validation_result=control_validation,
                checkpoint_path=control_validation.get("checkpoint_path"),
                features_df=control_features, market_state_df=control_state,
                factor_contract=control_contracts[fold.fold], factor_bundle=control_bundle,
                branch="control_6f", run_id=run_id,
            )
            control_row = {"fold": fold.fold, "seed": seed, "candidate": frozen_candidate,
                           "branch": "control_6f", "reward_variant": selected_reward,
                           "buffer_variant": selected_buffer,
                           "checkpoint_path": control_validation.get("checkpoint_path"),
                           **control_result}
            control_test_rows.append(control_row)
            _write(control_root / "test" / f"fold{fold.fold}" / f"seed{seed}.json", control_row)

    summary = {
        "run_id": run_id, "publishable": False if smoke else False,
        "selected_reward": selected_rewards if not smoke else selected_rewards[selected_folds[0].fold],
        "selected_buffer": selected_buffers if not smoke else selected_buffers[selected_folds[0].fold],
        "frozen_candidate": {
            str(fold.fold): f"{selected_rewards[fold.fold]}__{selected_buffers[fold.fold]}"
            for fold in selected_folds
        }, "validation": validation_rows + buffer_rows,
        "test": candidate_test_rows + control_test_rows,
    }
    training_budget = timesteps or (128 if smoke else 100_000)
    method_by_fold = {
        str(fold.fold): _frozen_method(
            factor_contract=fold_contracts[fold.fold],
            frozen_candidate=f"{selected_rewards[fold.fold]}__{selected_buffers[fold.fold]}",
            reward_variant=selected_rewards[fold.fold],
            buffer_variant=selected_buffers[fold.fold],
            training_budget=training_budget,
        )
        for fold in selected_folds
    }
    if stress_tester is not None:
        for fold in selected_folds:
            frozen_candidate = f"{selected_rewards[fold.fold]}__{selected_buffers[fold.fold]}"
            branch_inputs = (
                ("candidate_20f", candidate_test_rows, fold_inputs, fold_runtime_cfgs[fold.fold],
                 method_by_fold[str(fold.fold)], candidate_root),
                ("control_6f", control_test_rows, control_folds[fold.fold], control_runtime_cfgs[fold.fold],
                 _frozen_method(
                     factor_contract=control_contracts[fold.fold], frozen_candidate=frozen_candidate,
                     reward_variant=selected_rewards[fold.fold], buffer_variant=selected_buffers[fold.fold],
                     training_budget=training_budget,
                 ), control_root),
            )
            for branch, rows, inputs, branch_cfg, method, branch_root in branch_inputs:
                for row in (row for row in rows if row["fold"] == fold.fold):
                    stress_features, stress_state = _fold_input_copies(inputs)
                    stress_results = stress_tester(
                        branch=branch, fold=fold.fold, seed=row["seed"],
                        checkpoint_path=row.get("checkpoint_path"),
                        features_df=stress_features, market_state_df=stress_state,
                        cfg=dict(branch_cfg), timesteps=training_budget, method=method,
                    )
                    stress_mdd, stress_calmar_excess, stress_util, stress_evidence = _stress_evidence(stress_results)
                    stress_path = branch_root / "stress" / f"fold{fold.fold}" / f"seed{row['seed']}.json"
                    _write(stress_path, {
                        "branch": branch, "fold": fold.fold, "seed": row["seed"],
                        "checkpoint_path": row.get("checkpoint_path"),
                        "stress_mdd": stress_mdd,
                        "stress_calmar_excess": stress_calmar_excess,
                        "stress_long_exposure_util": stress_util,
                        "segments": stress_evidence,
                    })
                    row["stress_mdd"] = stress_mdd
                    row["stress_calmar_excess"] = stress_calmar_excess
                    row["stress_long_exposure_util"] = stress_util
                    row["stress_artifact_path"] = str(stress_path.relative_to(run_root))
                    row["stress_artifact_sha256"] = artifact_id(stress_path)
                    if branch == "candidate_20f":
                        test_path = branch_root / "test" / f"fold{fold.fold}" / row["candidate"] / f"seed{row['seed']}.json"
                    else:
                        test_path = branch_root / "test" / f"fold{fold.fold}" / f"seed{row['seed']}.json"
                    _write(test_path, row)
    summary["method_by_fold"] = method_by_fold
    if smoke:
        summary["frozen_method"] = method_by_fold[str(selected_folds[0].fold)]
    research_summary = aggregate_research_summary(candidate_test_rows, len(selected_folds))
    comparison = aggregate_candidate_comparison(
        candidate_test_rows, control_test_rows,
        fold_count=len(selected_folds), seed_count=len(selected_seeds),
    )
    if smoke:
        research_summary["no_leakage_tests_passed"] = False
        research_summary["state_quality_tests_passed"] = False
    gate_report = evaluate_candidate_gates(comparison)
    if smoke:
        gate_report["research_ok"] = False
    _write(run_root / "research_summary.json", research_summary)
    _write(run_root / "comparison.json", comparison)
    gate_report["comparison_path"] = "comparison.json"
    gate_report["comparison_id"] = artifact_id(run_root / "comparison.json")
    _write(run_root / "gates.json", gate_report)
    if not smoke:
        approved_fold = selected_folds[0]
        method = method_by_fold.get(str(approved_fold.fold), {})
        write_approval(
            run_root, method, gate_report, run_id, "", run_mode="full",
            fold_count=len(selected_folds), seed_count=len(selected_seeds),
            selection_artifact_path=(
                prepared_folds[approved_fold.fold].factor_bundle["selection_artifact_path"]
            ),
        )
    summary["research_summary"] = research_summary
    summary["gates"] = gate_report
    summary["publishable"] = bool(not smoke and gate_report["research_ok"])
    _write(run_root / "summary.json", summary)
    return summary


def _default_trainer(**kwargs) -> dict:
    """Small production adapter; tests can replace it through injection."""
    if kwargs.get("features_df") is None or kwargs.get("market_state_df") is None:
        raise ValueError("default trainer requires features_df and market_state_df")
    from scripts.env import PortfolioEnv
    from scripts.metrics import sharpe
    from scripts.observation import ObservationScaler
    from scripts.train import load_ppo, select_device, train_candidates

    cfg = dict(kwargs.get("cfg") or {})
    cfg["reward_variant"] = kwargs["reward_variant"]
    cfg.update(kwargs["buffer_config"])
    factor_contract = _checkpoint_factor_contract(
        cfg, fold=kwargs["fold"], factor_contract=kwargs.get("factor_contract"),
        run_id=kwargs.get("run_id"),
    )
    raw_env = PortfolioEnv(kwargs["features_df"], kwargs["market_state_df"], cfg,
                           *kwargs["train_range"])
    raw_env.episode_randomization = True  # 仅训练 env;val/stress/backtest 保持全窗口确定性
    val_env = PortfolioEnv(kwargs["features_df"], kwargs["market_state_df"], cfg,
                           *kwargs["val_range"])
    artifact_dir = pathlib.Path(kwargs["artifact_dir"])
    paths = train_candidates(
        root=artifact_dir, fold=kwargs["fold"], seed=kwargs["seed"],
        candidates=(kwargs["candidate"],), schema_version=STATE_SCHEMA_VERSION,
        raw_train_env=raw_env, eval_env=val_env,
        total_timesteps=kwargs["timesteps"], device=select_device(cfg.get("train_device", "auto")),
        train_range=kwargs["train_range"], val_range=kwargs["val_range"],
        reward_variant=kwargs["reward_variant"], buffer_variant=kwargs["buffer_variant"],
        factor_contract=factor_contract,
    )[kwargs["candidate"]]
    scaler = ObservationScaler.load(
        paths["scaler"], expected_schema=STATE_SCHEMA_VERSION,
        expected_fields=tuple(state_fields(factor_contract["selected_factors"])),
        expected_factor_contract=factor_contract,
    )
    val_env.observation_scaler = scaler
    model = load_ppo(str(paths["model"]), val_env,
                     expected_factor_contract=factor_contract,
                     metadata_path=paths["metadata"])
    obs, _ = val_env.reset(seed=kwargs["seed"])
    returns = []
    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, _, terminated, truncated, info = val_env.step(action)
        returns.extend(info["daily_net_rets"])
        done = terminated or truncated
    return {
        "val_sharpe": float(sharpe(returns)),
        "checkpoint_path": str(paths["model"]),
        "metadata_path": str(paths["metadata"]),
        "scaler_path": str(paths["scaler"]),
        "factor_contract": factor_contract,
        "schema_version": STATE_SCHEMA_VERSION,
        "training_budget": kwargs["timesteps"],
    }


def _default_tester(**kwargs) -> dict:
    checkpoint_path = kwargs.get("checkpoint_path")
    if not checkpoint_path or not pathlib.Path(checkpoint_path).exists():
        raise ValueError("frozen validation checkpoint is required; tester will not retrain")
    from scripts.env import PortfolioEnv
    from scripts.metrics import metrics_pack
    from scripts.observation import ObservationScaler
    from scripts.state import state_fields
    from scripts.train import load_ppo
    cfg = dict(kwargs.get("cfg") or {})
    cfg["reward_variant"] = kwargs["reward_variant"]
    cfg.update(kwargs["buffer_config"])
    factor_contract = kwargs.get("validation_result", {}).get("factor_contract")
    if factor_contract is None:
        factor_contract = _checkpoint_factor_contract(
            cfg, fold=kwargs.get("fold"), run_id=kwargs.get("run_id")
        )
    metadata_path = kwargs.get("validation_result", {}).get("metadata_path")
    scaler_path = kwargs.get("validation_result", {}).get("scaler_path")
    if not scaler_path:
        raise ValueError("frozen validation scaler is required")
    scaler = ObservationScaler.load(
        scaler_path, expected_schema=factor_contract["state_schema_version"],
        expected_fields=tuple(state_fields(factor_contract["selected_factors"])),
        expected_factor_contract=factor_contract,
    )
    env = PortfolioEnv(kwargs["features_df"], kwargs["market_state_df"], cfg,
                       *kwargs["test_range"], observation_scaler=scaler)
    model = load_ppo(
        str(checkpoint_path),
        env,
        expected_factor_contract=factor_contract,
        metadata_path=metadata_path,
    )
    obs, _ = env.reset(seed=kwargs["seed"])
    returns = []
    gross_returns = []
    turnovers = []
    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, _, terminated, truncated, info = env.step(action)
        returns.extend(info["daily_net_rets"])
        gross_returns.extend(info.get("daily_gross_rets", ()))
        turnovers.append(float(info.get("turnover", 0.0)))
        done = terminated or truncated
    metrics = metrics_pack(returns, "test")
    gross = list(gross_returns)
    net = list(returns)
    doubled_cost_returns = [2.0 * n - g for n, g in zip(net, gross)]
    cost_2x = metrics_pack(doubled_cost_returns, "test_cost_2x")
    return {
        "test_sharpe": metrics["sharpe"],
        "oos_sharpe": metrics["sharpe"],
        "oos_arr": metrics["arr"],
        "oos_mdd": metrics["mdd"],
        "daily_returns": returns,
        "annualized_turnover": (sum(turnovers) / max(len(turnovers), 1)) * 252.0,
        "cost_2x_oos_sharpe": cost_2x["sharpe"],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--full", action="store_true")
    parser.add_argument("--output-root", default="artifacts/walk_forward")
    parser.add_argument("--run-id", default=None,
                        help="explicit full-run directory name for pipeline integration")
    parser.add_argument("--timesteps", type=int, default=None)
    args = parser.parse_args()
    if args.smoke and args.full:
        parser.error("choose only one of --smoke or --full")
    if not args.smoke and not args.full:
        parser.error("one of --smoke or --full is required")
    root = pathlib.Path(__file__).resolve().parent.parent
    features_path = root / "data" / "features.parquet"
    index_returns_path = root / "data" / "index_returns.parquet"
    factor_cache_root = root / "data" / "factors"
    if not features_path.exists() or not index_returns_path.exists():
        parser.error(f"required input missing: {features_path} and {index_returns_path}")
    import pandas as pd
    from scripts.config import get_config
    features = pd.read_parquet(features_path)
    index_returns = pd.read_parquet(index_returns_path)
    run_walk_forward(
        folds=default_folds(), output_root=root / args.output_root,
        smoke=args.smoke, trainer=_default_trainer, tester=_default_tester,
        features_df=features, index_df=index_returns, cfg=get_config(),
        timesteps=args.timesteps or (128 if args.smoke else 100_000),
        run_id=args.run_id, cache_root=factor_cache_root,
        index_returns=index_returns,
    )
    return 0


if __name__ == "__main__":
    main()
