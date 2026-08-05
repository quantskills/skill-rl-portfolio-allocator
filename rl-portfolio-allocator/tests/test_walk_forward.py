import json
import pathlib
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

from scripts.config import TRAIN_SEEDS
from scripts.walk_forward import (
    BUFFER_CONFIGS,
    BUFFER_CANDIDATES,
    DEFAULT_REWARD_CANDIDATES,
    SEEDS,
    default_folds,
    reward_candidates,
    run_walk_forward,
    select_candidate_on_validation,
)
from scripts.walk_forward import _default_tester, _checkpoint_factor_contract
from scripts.config import FACTOR_NAMES
from scripts.stress_test import apply_frozen_method, load_frozen_method
from scripts.state import STATE_SCHEMA_VERSION
from scripts.train import FACTOR_CONTRACT_FIELDS
from scripts.observation import ObservationScaler
from scripts.state import state_fields
from scripts.factor_selection import select_factors
from scripts.factor_catalog import FACTOR_CATALOG
import scripts.walk_forward as walk_forward


CANDIDATE_FACTOR_NAMES = [spec.name for spec in FACTOR_CATALOG[:20]]


def _contract_cfg(names=None):
    names = list(names or CANDIDATE_FACTOR_NAMES)
    return {
        "factor_catalog_version": "catalog-v1",
        "factor_catalog_hash": "sha256:catalog",
        "factor_names": names,
        "factor_directions": [(-1 if index % 2 else 1) for index, _ in enumerate(names)],
        "selection_run_id": "selection-42",
        "state_schema_version": STATE_SCHEMA_VERSION,
        "schema_version": STATE_SCHEMA_VERSION,
    }


def _trainer_artifacts(kwargs, **result):
    artifact_dir = kwargs["artifact_dir"]
    artifact_dir.mkdir(parents=True, exist_ok=True)
    contract = kwargs["factor_contract"]
    checkpoint = artifact_dir / "best.zip"
    metadata = artifact_dir / "best_metadata.json"
    scaler_path = artifact_dir / "scaler.json"
    checkpoint.write_bytes(b"checkpoint")
    fields = tuple(state_fields(contract["selected_factors"]))
    scaler = ObservationScaler(
        schema_version=contract["state_schema_version"], fields=fields,
        mean=(0.0,) * len(fields), scale=(1.0,) * len(fields),
    )
    scaler.save(scaler_path, factor_contract=contract)
    metadata.write_text(json.dumps({
        "schema_version": contract["state_schema_version"],
        "checkpoint_path": str(checkpoint), "scaler_path": str(scaler_path),
        **contract,
    }), encoding="utf-8")
    return {
        **result, "checkpoint_path": str(checkpoint),
        "metadata_path": str(metadata), "scaler_path": str(scaler_path),
        "factor_contract": contract,
    }


class _CandidateCache:
    def __init__(self, names=None):
        self.names = list(names or CANDIDATE_FACTOR_NAMES)
        self.panel = pd.DataFrame({
            "trade_date": [pd.Timestamp("2010-05-01")],
            "symbol": ["A"], "ret_1d": [0.0], "is_suspended": [False],
            **{name: [1.0] for name in self.names},
        })

    def materialize_selected_panel(self, root, selected):
        return self.panel.copy(deep=True)


class _MutableCfgMetadata:
    pass


def _select_configured_factors(*, cfg, **_):
    return [
        {"name": name, "direction": direction}
        for name, direction in zip(cfg["factor_names"], cfg["factor_directions"])
    ]


def _select_factors_with_artifacts(*, cfg, **_):
    names = cfg["factor_names"]
    metrics = {
        name: {
            "name": name, "family": "test", "catalog_order": index,
            "passed": True, "mean_ic": 0.02, "icir": 0.20,
            "sign_consistency": 0.80, "positive_ic_rate": 0.60,
            "net_factor_sharpe": 0.50, "doubled_cost_sharpe": 0.20,
            "regime_stability": 0.70, "factor_turnover": 0.20,
            "tail_loss": 0.10, "direction": direction, "coverage": 1.0,
            "symbols": 100, "dates": 500, "failure_reasons": [],
        }
        for index, (name, direction) in enumerate(
            zip(names, cfg["factor_directions"])
        )
    }
    correlations = pd.DataFrame(0.0, index=names, columns=names)
    for name in names:
        correlations.loc[name, name] = 1.0
    result = select_factors(metrics, correlations, correlations, target_count=len(names))
    return result, metrics, {
        "return_corr": correlations,
        "cross_section_corr": correlations,
    }


def _factor_inputs(tmp_path, names=None):
    return {
        "selector": _select_configured_factors,
        "candidate_cache": _CandidateCache(names),
        "cache_root": tmp_path / "factor-cache",
        "index_returns": pd.DataFrame(),
    }


def _mock_real_stress(monkeypatch):
    import scripts.stress_test as stress_test

    monkeypatch.setattr(
        stress_test, "run_all_stress",
        lambda *args, **kwargs: [{"skipped": False, "metrics": {"rl": {"mdd": -0.10}}}],
    )


@pytest.fixture(autouse=True)
def _simple_fold_market_state(monkeypatch):
    monkeypatch.setattr(
        walk_forward, "build_market_state",
        lambda features, index_returns, cfg, factor_names: pd.DataFrame({
            "trade_date": pd.to_datetime(features["trade_date"]), "state": 1.0,
        }),
    )


def test_constants_are_closed_and_buffer_configs_are_exact():
    assert SEEDS == TRAIN_SEEDS == (0,)
    assert DEFAULT_REWARD_CANDIDATES == ("none", "gentle", "low", "constrained", "legacy_dsr")
    assert reward_candidates() == DEFAULT_REWARD_CANDIDATES
    assert BUFFER_CANDIDATES == ("tight", "default", "wide")
    assert BUFFER_CONFIGS == {
        "tight": {"long_entry": 30, "long_exit": 40, "short_entry": 15, "short_exit": 25},
        "default": {"long_entry": 30, "long_exit": 45, "short_entry": 15, "short_exit": 30},
        "wide": {"long_entry": 30, "long_exit": 60, "short_entry": 15, "short_exit": 45},
    }


def test_reward_candidates_env_override(monkeypatch):
    monkeypatch.setenv("RLPA_REWARD_CANDIDATES", "none, low, gentle")
    assert reward_candidates() == ("none", "low", "gentle")


def test_validation_selects_candidate_by_group_median_descending():
    rows = [
        {"candidate": "low", "val_sharpe": 1.0},
        {"candidate": "low", "val_sharpe": 3.0},
        {"candidate": "medium", "val_sharpe": 2.0},
        {"candidate": "medium", "val_sharpe": 2.1},
        {"candidate": "none", "val_sharpe": 9.0},
        {"candidate": "none", "val_sharpe": -9.0},
    ]
    assert select_candidate_on_validation(rows) == "medium"


def test_default_folds_are_three():
    assert len(default_folds()) == 3
    assert [fold.fold for fold in default_folds()] == [1, 2, 3]


def test_walk_forward_orders_validation_then_frozen_tests_once(tmp_path):
    events = []

    def trainer(**kwargs):
        events.append(("train", kwargs["fold"], kwargs["stage"], kwargs["candidate"], kwargs["seed"]))
        score = {"none": 1.0, "low": 3.0, "medium": 2.0, "legacy_dsr": 0.0}.get(
            kwargs["reward_variant"], 0.0
        )
        return _trainer_artifacts(
            kwargs, val_sharpe=score, method={"candidate": kwargs["candidate"]}
        )

    def tester(**kwargs):
        events.append(("test", kwargs["fold"], kwargs["candidate"], kwargs["seed"]))
        return {"test_sharpe": 0.5}

    result = run_walk_forward(
        folds=default_folds(), output_root=tmp_path, smoke=True,
        trainer=trainer, tester=tester, coverage_checker=lambda: None,
        cfg=_contract_cfg(), **_factor_inputs(tmp_path),
    )
    assert result["publishable"] is False
    assert sum(e[0] == "test" for e in events) == 2
    first_test = next(i for i, event in enumerate(events) if event[0] == "test")
    assert all(event[0] == "train" for event in events[:first_test])
    assert all(event[2] == "low__tight" for event in events if event[0] == "test")
    assert all(events.count(event) == 2 for event in events if event[0] == "test")
    assert (tmp_path / "smoke" / "summary.json").exists()
    assert json.loads((tmp_path / "smoke" / "summary.json").read_text())["publishable"] is False


def test_walk_forward_selects_before_candidates_and_reuses_one_frozen_bundle(tmp_path):
    events = []
    base_cfg = _contract_cfg()

    def trainer(**kwargs):
        events.append(("train", kwargs["factor_bundle"]))
        assert set(kwargs["factor_bundle"]) == {
            "fold", "names", "directions", "factor_contract", "feature_path",
            "market_state_path", "selection_artifact_path",
        }
        assert "arbitrary_metadata" not in kwargs["factor_bundle"]
        assert kwargs["factor_contract"] is kwargs["factor_bundle"]["factor_contract"]
        assert isinstance(kwargs["features_df"], pd.DataFrame)
        expected_names = CANDIDATE_FACTOR_NAMES if kwargs["branch"] == "candidate_20f" else walk_forward.CONTROL_FACTOR_NAMES
        assert kwargs["cfg"]["factor_names"] == expected_names
        assert kwargs["cfg"]["k"] == len(expected_names)
        assert kwargs["cfg"]["factor_contract"] is kwargs["factor_bundle"]["factor_contract"]
        return _trainer_artifacts(kwargs, val_sharpe=1.0)

    def tester(**kwargs):
        events.append(("test", kwargs["factor_bundle"]))
        assert kwargs["factor_contract"] is kwargs["factor_bundle"]["factor_contract"]
        assert isinstance(kwargs["features_df"], pd.DataFrame)
        expected_names = CANDIDATE_FACTOR_NAMES if kwargs["branch"] == "candidate_20f" else walk_forward.CONTROL_FACTOR_NAMES
        assert kwargs["cfg"]["factor_names"] == expected_names
        assert kwargs["cfg"]["k"] == len(expected_names)
        assert kwargs["cfg"]["factor_contract"] is kwargs["factor_bundle"]["factor_contract"]
        return {"test_sharpe": 0.0}

    run_walk_forward(
        folds=[default_folds()[0]], output_root=tmp_path, smoke=True,
        trainer=trainer, tester=tester, coverage_checker=lambda: None,
        cfg=base_cfg, selector=_select_configured_factors,
        candidate_cache=_CandidateCache(),
        cache_root=tmp_path / "factor-cache", index_returns=pd.DataFrame(),
    )

    candidate_bundles = [bundle for _, bundle in events if bundle["names"] == tuple(CANDIDATE_FACTOR_NAMES)]
    assert {id(bundle) for bundle in candidate_bundles} == {id(candidate_bundles[0])}
    assert base_cfg["factor_names"] == CANDIDATE_FACTOR_NAMES
    assert "k" not in base_cfg


def test_walk_forward_keeps_fold_materialization_after_future_cache_mutation(tmp_path):
    cache = _CandidateCache()
    source_cache = cache.panel
    received = []

    def trainer(**kwargs):
        received.append(kwargs)
        assert kwargs["features_df"]["mom_20"].tolist() == [1.0]
        source_cache.loc[0, "mom_20"] = -999.0
        return _trainer_artifacts(kwargs, val_sharpe=1.0)

    def tester(**kwargs):
        received.append(kwargs)
        assert kwargs["features_df"]["mom_20"].tolist() == [1.0]
        return {"test_sharpe": 0.0}

    run_walk_forward(
        folds=[default_folds()[0]], output_root=tmp_path, smoke=True,
        trainer=trainer, tester=tester, coverage_checker=lambda: None,
        cfg=_contract_cfg(),
        selector=_select_configured_factors, candidate_cache=cache,
        cache_root=tmp_path / "factor-cache", index_returns=pd.DataFrame(),
    )

    assert source_cache["mom_20"].tolist() == [-999.0]
    candidate_received = [kwargs for kwargs in received if kwargs["branch"] == "candidate_20f"]
    assert {id(kwargs["factor_bundle"]) for kwargs in candidate_received} == {id(candidate_received[0]["factor_bundle"])}
    assert {id(kwargs["cfg"]) for kwargs in candidate_received} == {id(candidate_received[0]["cfg"])}


def test_walk_forward_isolates_canonical_panels_from_mutating_trainer(tmp_path):
    fold = default_folds()[0]
    cache = _CandidateCache()
    canonical_features = cache.panel

    def trainer(**kwargs):
        assert "features" not in kwargs["factor_bundle"]
        assert "market_state" not in kwargs["factor_bundle"]
        kwargs["features_df"].loc[0, "mom_20"] = -999.0
        kwargs["market_state_df"].loc[0, "mutated"] = True
        return _trainer_artifacts(kwargs, val_sharpe=1.0)

    def tester(**kwargs):
        assert "features" not in kwargs["factor_bundle"]
        assert "market_state" not in kwargs["factor_bundle"]
        assert kwargs["features_df"]["mom_20"].tolist() == [1.0]
        assert "mutated" not in kwargs["market_state_df"]
        return {"test_sharpe": 0.0}

    run_walk_forward(
        folds=[fold], output_root=tmp_path, smoke=True,
        trainer=trainer, tester=tester, coverage_checker=lambda: None,
        cfg=_contract_cfg(), selector=_select_configured_factors,
        candidate_cache=cache, cache_root=tmp_path / "factor-cache", index_returns=pd.DataFrame(),
    )

    assert canonical_features["mom_20"].tolist() == [1.0]
    assert "mutated" not in cache.panel


def test_walk_forward_rejects_static_factor_inputs_without_prepared_bundle(tmp_path):
    with pytest.raises(ValueError, match="fold-local factor cache"):
        run_walk_forward(
            folds=[default_folds()[0]], output_root=tmp_path, smoke=True,
            trainer=lambda **kwargs: _trainer_artifacts(kwargs, val_sharpe=1.0),
            tester=lambda **kwargs: {"test_sharpe": 0.0}, coverage_checker=lambda: None,
            features_df=pd.DataFrame(), index_df=pd.DataFrame(), cfg=_contract_cfg(),
        )


def test_walk_forward_rejects_injected_non_twenty_factor_candidate_before_branch_artifacts(tmp_path):
    calls = []

    with pytest.raises(ValueError, match="candidate factor selection must contain exactly 20 ordered factors; got 19"):
        run_walk_forward(
            folds=[default_folds()[0]], output_root=tmp_path, smoke=True,
            trainer=lambda **kwargs: calls.append(("trainer", kwargs)),
            tester=lambda **kwargs: calls.append(("tester", kwargs)),
            coverage_checker=lambda: None, cfg=_contract_cfg(),
            selector=lambda **_: [
                {"name": name, "direction": 1}
                for name in CANDIDATE_FACTOR_NAMES[:-1]
            ],
            candidate_cache=_CandidateCache(), cache_root=tmp_path / "factor-cache",
            index_returns=pd.DataFrame(),
        )

    assert calls == []
    assert not (tmp_path / "smoke" / "candidate_20f").exists()


def test_walk_forward_protects_shared_fold_bundle_and_runtime_cfg_from_mutating_trainer(tmp_path):
    base_cfg = _contract_cfg()
    observed = []

    def snapshot(kwargs):
        return (
            tuple(kwargs["factor_bundle"]["names"]),
            tuple(kwargs["factor_bundle"]["directions"]),
            tuple(kwargs["cfg"]["factor_names"]),
            tuple(kwargs["cfg"]["factor_directions"]),
            tuple(kwargs["cfg"]["factor_contract"]["selected_factors"]),
        )

    def trainer(**kwargs):
        if not observed:
            with pytest.raises(TypeError):
                kwargs["factor_bundle"]["names"] = ("corrupted",)
            with pytest.raises((AttributeError, TypeError)):
                kwargs["factor_bundle"]["factor_contract"]["selected_factors"].append("corrupted")
            with pytest.raises((AttributeError, TypeError)):
                kwargs["cfg"]["factor_names"].append("corrupted")
        observed.append(("train", id(kwargs["factor_bundle"]), id(kwargs["cfg"]), snapshot(kwargs)))
        return _trainer_artifacts(kwargs, val_sharpe=1.0)

    def tester(**kwargs):
        observed.append(("test", id(kwargs["factor_bundle"]), id(kwargs["cfg"]), snapshot(kwargs)))
        return {"test_sharpe": 0.0}

    run_walk_forward(
        folds=[default_folds()[0]], output_root=tmp_path, smoke=True,
        trainer=trainer, tester=tester, coverage_checker=lambda: None,
        cfg=base_cfg, selector=_select_configured_factors,
        candidate_cache=_CandidateCache(),
        cache_root=tmp_path / "factor-cache", index_returns=pd.DataFrame(),
    )

    expected = (
        tuple(CANDIDATE_FACTOR_NAMES), tuple(_contract_cfg()["factor_directions"]),
        tuple(CANDIDATE_FACTOR_NAMES), tuple(_contract_cfg()["factor_directions"]),
        tuple(CANDIDATE_FACTOR_NAMES),
    )
    candidate_records = [record for record in observed if record[3] == expected]
    assert {record[1] for record in candidate_records} == {candidate_records[0][1]}
    assert {record[2] for record in candidate_records} == {candidate_records[0][2]}
    assert base_cfg["factor_names"] == CANDIDATE_FACTOR_NAMES


def test_walk_forward_multifold_prepares_each_fold_before_its_validation_and_reuses_it_for_test(tmp_path, monkeypatch):
    monkeypatch.setattr(walk_forward, "SEEDS", (0,))
    _mock_real_stress(monkeypatch)
    events = []

    def selector(**kwargs):
        fold = kwargs["fold"].fold
        events.append(("prepared", fold))
        return [{"name": name, "direction": 1} for name in CANDIDATE_FACTOR_NAMES]

    def trainer(**kwargs):
        events.append(("validation", kwargs["fold"], id(kwargs["factor_bundle"])))
        return _trainer_artifacts(kwargs, val_sharpe=1.0)

    def tester(**kwargs):
        events.append(("test", kwargs["fold"], id(kwargs["factor_bundle"])))
        return {"test_sharpe": 0.0}

    run_walk_forward(
        folds=default_folds()[:2], output_root=tmp_path, smoke=False,
        trainer=trainer, tester=tester, coverage_checker=lambda: None,
        cfg=_contract_cfg(), selector=selector,
        candidate_cache=_CandidateCache(),
        cache_root=tmp_path / "factor-cache", index_returns=pd.DataFrame(), run_id="multifold",
    )

    for fold in (1, 2):
        prepared_at = next(i for i, event in enumerate(events) if event[:2] == ("prepared", fold))
        validation_at = next(i for i, event in enumerate(events) if event[:2] == ("validation", fold))
        test = next(event for event in events if event[:2] == ("test", fold))
        assert prepared_at < validation_at
        assert test[2] == next(event[2] for event in events if event[:2] == ("validation", fold))


def test_tester_receives_saved_validation_result_and_checkpoint(tmp_path):
    received = []

    def trainer(**kwargs):
        return _trainer_artifacts(kwargs, val_sharpe=1.0, best=True)

    def tester(**kwargs):
        received.append(kwargs)
        return {"test_sharpe": 0.0}

    run_walk_forward(
        folds=default_folds(), output_root=tmp_path, smoke=True,
        trainer=trainer, tester=tester, coverage_checker=lambda: None,
        cfg=_contract_cfg(), **_factor_inputs(tmp_path),
    )
    assert received
    assert received[0]["checkpoint_path"].endswith("best.zip")
    assert received[0]["validation_result"]["best"] is True
    assert received[0]["validation_result"]["checkpoint_path"] == received[0]["checkpoint_path"]


def test_frozen_method_applies_reward_buffer_schema_and_budget():
    cfg = {"reward_variant": "none", "schema_version": "old", "long_entry": 1}
    method = {
        "reward_variant": "medium", "buffer_variant": "tight",
        "buffer_config": {"long_entry": 30, "long_exit": 40, "short_entry": 15, "short_exit": 25},
        "factor_catalog_version": "catalog-v1",
        "factor_catalog_hash": "sha256:catalog",
        "selected_factors": ["mom_20", "vol_20"],
        "factor_directions": [1, -1],
        "selection_run_id": "selection-42",
        "fold": 2,
        "state_schema_version": "state-v1",
        "training_budget": 777,
    }
    applied_cfg, budget = apply_frozen_method(cfg, method, 100)
    assert applied_cfg["reward_variant"] == "medium"
    assert applied_cfg["buffer_variant"] == "tight"
    assert applied_cfg["long_exit"] == 40
    assert applied_cfg["schema_version"] == "state-v1"
    assert budget == 777


def test_direct_script_help_bootstraps_package_imports():
    for script in ("scripts/walk_forward.py", "scripts/stress_test.py"):
        completed = subprocess.run(
            [sys.executable, script, "--help"], capture_output=True, text=True
        )
        assert completed.returncode == 0, completed.stderr
        assert "usage" in completed.stdout.lower()


def test_cli_wires_fold_factor_inputs_into_walk_forward(monkeypatch, tmp_path):
    import scripts.walk_forward as walk_forward

    captured = {}
    index_returns = pd.DataFrame({"trade_date": [], "ret": []})

    def fake_read_parquet(path):
        return index_returns if str(path).endswith("index_returns.parquet") else pd.DataFrame()

    def fake_run_walk_forward(**kwargs):
        captured.update(kwargs)

    # Patch file existence checks so the test is isolated from disk state.
    _original_exists = pathlib.Path.exists
    def _fake_exists(self: pathlib.Path) -> bool:
        if self.name in ("features.parquet", "index_returns.parquet"):
            return True
        return _original_exists(self)
    monkeypatch.setattr(pathlib.Path, "exists", _fake_exists)

    monkeypatch.setattr(walk_forward.pd, "read_parquet", fake_read_parquet)
    monkeypatch.setattr(walk_forward, "run_walk_forward", fake_run_walk_forward)
    monkeypatch.setattr(sys, "argv", ["walk_forward.py", "--smoke"])

    assert walk_forward.main() == 0
    root = pathlib.Path(walk_forward.__file__).resolve().parent.parent
    assert captured["cache_root"] == root / "data" / "factors"
    assert captured["index_returns"] is index_returns


def test_summary_contains_stress_readable_frozen_method(tmp_path, monkeypatch):
    _mock_real_stress(monkeypatch)
    def trainer(**kwargs):
        return _trainer_artifacts(kwargs, val_sharpe=1.0)

    def tester(**kwargs):
        return {"test_sharpe": 0.0}

    smoke = run_walk_forward(
        folds=default_folds(), output_root=tmp_path, smoke=True,
        trainer=trainer, tester=tester, coverage_checker=lambda: None,
        cfg=_contract_cfg(), timesteps=128,
        **_factor_inputs(tmp_path),
    )
    smoke_method = smoke["frozen_method"]
    assert smoke_method["schema_version"] == STATE_SCHEMA_VERSION
    assert all(field in smoke_method for field in FACTOR_CONTRACT_FIELDS)
    assert smoke_method["selected_factors"] == _contract_cfg()["factor_names"]
    assert smoke_method["factor_directions"] == _contract_cfg()["factor_directions"]
    assert smoke_method["reward_variant"] == "none"
    assert smoke_method["buffer_variant"] == "tight"
    assert smoke_method["buffer_config"] == BUFFER_CONFIGS["tight"]
    assert smoke_method["training_budget"] == 128
    assert json.loads((tmp_path / "smoke" / "summary.json").read_text())["frozen_method"] == smoke_method
    loaded = load_frozen_method(str(tmp_path / "smoke" / "summary.json"))
    assert loaded["buffer_variant"] == "tight"
    assert loaded["training_budget"] == 128

    full = run_walk_forward(
        folds=default_folds(), output_root=tmp_path, smoke=False,
        trainer=trainer, tester=tester, coverage_checker=lambda: None,
        cfg=_contract_cfg(), timesteps=256,
        **_factor_inputs(tmp_path),
    )
    assert set(full["method_by_fold"]) == {"1", "2", "3"}
    assert all(method["training_budget"] == 256 for method in full["method_by_fold"].values())


def test_walk_forward_defaults_emitted_methods_to_current_state_schema(tmp_path):
    def trainer(**kwargs):
        return _trainer_artifacts(kwargs, val_sharpe=1.0)

    smoke = run_walk_forward(
        folds=default_folds(), output_root=tmp_path, smoke=True,
        trainer=trainer, tester=lambda **kwargs: {"test_sharpe": 0.0},
        coverage_checker=lambda: None, cfg=_contract_cfg(),
        **_factor_inputs(tmp_path),
    )

    assert smoke["frozen_method"]["schema_version"] == STATE_SCHEMA_VERSION


def test_full_runs_use_unique_run_directories_and_write_frozen_test_records(tmp_path, monkeypatch):
    _mock_real_stress(monkeypatch)
    calls = []

    def trainer(**kwargs):
        calls.append(("train", kwargs["fold"], kwargs["stage"], kwargs["candidate"], kwargs["seed"]))
        return _trainer_artifacts(
            kwargs, val_sharpe=1.0 if kwargs["candidate"] == "medium" else 0.0
        )

    def tester(**kwargs):
        calls.append(("test", kwargs["fold"], kwargs["candidate"], kwargs["seed"]))
        return {"test_sharpe": 1.0}

    first = run_walk_forward(
        folds=default_folds(), output_root=tmp_path, smoke=False,
        trainer=trainer, tester=tester, coverage_checker=lambda: None,
        cfg=_contract_cfg(), **_factor_inputs(tmp_path),
    )
    second = run_walk_forward(
        folds=default_folds(), output_root=tmp_path, smoke=False,
        trainer=trainer, tester=tester, coverage_checker=lambda: None,
        cfg=_contract_cfg(), **_factor_inputs(tmp_path),
    )
    assert first["run_id"] != second["run_id"]
    assert len(list(tmp_path.iterdir())) == 2
    assert len([c for c in calls if c[0] == "test"]) == 2 * 3 * len(SEEDS) * 2
    for run_id in (first["run_id"], second["run_id"]):
        assert len(list((tmp_path / run_id / "candidate_20f" / "test").rglob("*.json"))) == len(SEEDS) * 3
        assert len(list((tmp_path / run_id / "control_6f" / "test").rglob("*.json"))) == len(SEEDS) * 3


def test_full_run_accepts_explicit_run_id_and_writes_full_approval(tmp_path, monkeypatch):
    import scripts.stress_test as stress_test

    def trainer(**kwargs):
        return _trainer_artifacts(kwargs, val_sharpe=1.0)

    def tester(**kwargs):
        return {
            "test_sharpe": 1.0, "oos_sharpe": 1.0 if kwargs["branch"] == "candidate_20f" else 0.8, "oos_arr": 0.2,
            "oos_mdd": -0.1, "excess_return": 0.2,
            "strongest_baseline_sharpe": 0.1, "strongest_baseline_mdd": -0.2,
            "annualized_turnover": 1.0, "cost_2x_oos_sharpe": 0.5,
            "no_leakage_tests_passed": True, "state_quality_tests_passed": True,
        }

    monkeypatch.setattr(
        stress_test, "run_all_stress",
        lambda *args, **kwargs: [{
            "skipped": False,
            "metrics": {
                "rl": {"mdd": -0.10, "calmar": 0.9},
                "static_factor_equal": {"mdd": -0.11, "calmar": 0.8},
            },
            "diagnostics": {"long_exposure_util": 0.9},
        }],
    )
    result = run_walk_forward(
        folds=default_folds(), output_root=tmp_path, smoke=False,
        run_id="full-run-001", trainer=trainer, tester=tester,
        coverage_checker=lambda: None, cfg=_contract_cfg(),
        **_factor_inputs(tmp_path),
    )
    assert result["run_id"] == "full-run-001"
    approval = json.loads((tmp_path / "full-run-001" / "approval.json").read_text())
    assert approval["run_mode"] == "full"
    assert approval["fold_count"] == 3
    assert approval["seed_count"] == len(SEEDS)
    assert approval["comparison_path"] == "comparison.json"
    assert approval["comparison_id"] == walk_forward.artifact_id(
        tmp_path / "full-run-001" / "comparison.json"
    )
    assert approval["factor_selection_path"] == (
        "candidate_20f/selection/fold1/selected_factors.json"
    )
    assert approval["factor_selection_id"] == walk_forward.artifact_id(
        tmp_path / "full-run-001" / approval["factor_selection_path"]
    )


def test_walk_forward_writes_paired_branch_artifacts_and_comparison(tmp_path, monkeypatch):
    monkeypatch.setattr(walk_forward, "SEEDS", (0,))
    received = []

    def trainer(**kwargs):
        received.append((kwargs["branch"], kwargs["factor_contract"], kwargs["cfg"], kwargs["artifact_dir"]))
        return _trainer_artifacts(kwargs, val_sharpe=1.0)

    def tester(**kwargs):
        received.append((kwargs["branch"], kwargs["factor_contract"], kwargs["cfg"], kwargs["artifact_dir"]))
        return {
            "oos_sharpe": 0.6 if kwargs["branch"] == "candidate_20f" else 0.4,
            "oos_mdd": -0.20, "cost_2x_oos_sharpe": 0.1,
            "annualized_turnover": 8.0, "stress_mdd": -0.20,
        }

    result = run_walk_forward(
        folds=[default_folds()[0]], output_root=tmp_path, smoke=True,
        trainer=trainer, tester=tester, coverage_checker=lambda: None,
        cfg=_contract_cfg(), **_factor_inputs(tmp_path),
    )

    run_root = tmp_path / "smoke"
    assert (run_root / "control_6f" / "test" / "fold1" / "seed0.json").is_file()
    assert (run_root / "candidate_20f" / "test" / "fold1" / "none__tight" / "seed0.json").is_file()
    comparison = json.loads((run_root / "comparison.json").read_text())
    assert comparison["candidate_median_oos_sharpe"] == 0.6
    assert comparison["control_median_oos_sharpe"] == 0.4
    assert result["gates"]["research_ok"] is False
    assert not (run_root / "approval.json").exists()
    assert {branch for branch, *_ in received} == {"control_6f", "candidate_20f"}
    assert {
        len(contract["selected_factors"])
        for branch, contract, *_ in received if branch == "candidate_20f"
    } == {20}
    assert {
        len(contract["selected_factors"])
        for branch, contract, *_ in received if branch == "control_6f"
    } == {6}


def test_walk_forward_default_stress_runner_supplies_paired_branch_mdd(tmp_path, monkeypatch):
    import scripts.stress_test as stress_test

    calls = []

    def trainer(**kwargs):
        return _trainer_artifacts(kwargs, val_sharpe=1.0)

    def tester(**kwargs):
        return {
            "oos_sharpe": 0.6 if kwargs["branch"] == "candidate_20f" else 0.4,
            "oos_mdd": -0.02,
            "cost_2x_oos_sharpe": 0.1,
            "annualized_turnover": 8.0,
        }

    def run_all_stress(*args, method, **kwargs):
        calls.append(method["frozen_candidate"])
        return [{"skipped": False, "metrics": {"rl": {"mdd": -0.31 if len(method["selected_factors"]) == 20 else -0.27}}}]

    monkeypatch.setattr(stress_test, "run_all_stress", run_all_stress)
    monkeypatch.setattr(walk_forward, "_default_tester", tester)
    result = run_walk_forward(
        folds=[default_folds()[0]], output_root=tmp_path, smoke=True,
        trainer=trainer, tester=walk_forward._default_tester, coverage_checker=lambda: None,
        cfg=_contract_cfg(), **_factor_inputs(tmp_path),
    )

    comparison = json.loads((tmp_path / "smoke" / "comparison.json").read_text())
    assert calls == ["none__tight", "none__tight"]
    assert comparison["candidate_stress_mdd"] == -0.31
    assert comparison["control_stress_mdd"] == -0.27
    assert result["gates"]["research_ok"] is False


def test_full_custom_tester_cannot_publish_without_real_stress_output(tmp_path, monkeypatch):
    import scripts.stress_test as stress_test

    stress_calls = []

    def trainer(**kwargs):
        return _trainer_artifacts(kwargs, val_sharpe=1.0)

    def tester(**kwargs):
        return {
            "oos_sharpe": 0.6 if kwargs["branch"] == "candidate_20f" else 0.4,
            "oos_arr": 0.2, "oos_mdd": -0.1, "excess_return": 0.2,
            "strongest_baseline_sharpe": 0.1, "strongest_baseline_mdd": -0.2,
            "annualized_turnover": 1.0, "cost_2x_oos_sharpe": 0.5,
            "no_leakage_tests_passed": True, "state_quality_tests_passed": True,
            "stress_mdd": -0.01,  # A custom tester cannot forge real stress evidence.
        }

    def no_real_stress(*args, **kwargs):
        stress_calls.append(kwargs)
        return []

    monkeypatch.setattr(stress_test, "run_all_stress", no_real_stress)
    result = run_walk_forward(
        folds=default_folds(), output_root=tmp_path, smoke=False,
        trainer=trainer, tester=tester, coverage_checker=lambda: None,
        cfg=_contract_cfg(), **_factor_inputs(tmp_path),
    )

    assert len(stress_calls) == len(SEEDS) * 3 * 2
    assert result["gates"]["research_ok"] is False
    assert not (tmp_path / result["run_id"] / "approval.json").exists()


def test_full_stress_evidence_is_seed_specific_and_persisted_per_test_row(tmp_path, monkeypatch):
    import scripts.stress_test as stress_test

    stress_calls = []

    def trainer(**kwargs):
        return _trainer_artifacts(kwargs, val_sharpe=1.0)

    def tester(**kwargs):
        return {
            "oos_sharpe": 0.6 if kwargs["branch"] == "candidate_20f" else 0.4,
            "oos_arr": 0.2, "oos_mdd": -0.1, "excess_return": 0.2,
            "strongest_baseline_sharpe": 0.1, "strongest_baseline_mdd": -0.2,
            "annualized_turnover": 1.0, "cost_2x_oos_sharpe": 0.5,
            "no_leakage_tests_passed": True, "state_quality_tests_passed": True,
        }

    def run_all_stress(*args, seed, checkpoint_path, **kwargs):
        stress_calls.append((kwargs["branch"], kwargs["fold"], seed, checkpoint_path))
        offset = 0.01 if kwargs["branch"] == "candidate_20f" else 0.02
        return [{"name": "seeded", "skipped": False, "metrics": {"rl": {"mdd": -(seed + offset)}}}]

    monkeypatch.setattr(stress_test, "run_all_stress", run_all_stress)
    result = run_walk_forward(
        folds=default_folds(), output_root=tmp_path, smoke=False,
        trainer=trainer, tester=tester, coverage_checker=lambda: None,
        cfg=_contract_cfg(), **_factor_inputs(tmp_path),
    )

    assert len(stress_calls) == len(SEEDS) * 3 * 2
    assert {(branch, fold, seed) for branch, fold, seed, _ in stress_calls} == {
        (branch, fold, seed)
        for branch in ("candidate_20f", "control_6f")
        for fold in (1, 2, 3)
        for seed in SEEDS
    }
    run_root = tmp_path / result["run_id"]
    for row in result["test"]:
        expected = -(row["seed"] + (0.01 if row["branch"] == "candidate_20f" else 0.02))
        assert row["stress_mdd"] == pytest.approx(expected)
        artifact = run_root / row["stress_artifact_path"]
        assert artifact.is_file()
        assert json.loads(artifact.read_text())["stress_mdd"] == pytest.approx(expected)
        assert row["stress_artifact_sha256"] == walk_forward.artifact_id(artifact)


def test_full_run_rejects_existing_run_directory_without_writing(tmp_path):
    run_root = tmp_path / "full-run-001"
    run_root.mkdir()
    sentinel = run_root / "existing-evidence.json"
    sentinel.write_text('{"preserve": true}\n', encoding="utf-8")
    calls = []

    def trainer(**kwargs):
        calls.append(kwargs)
        return _trainer_artifacts(kwargs, val_sharpe=1.0)

    with pytest.raises(FileExistsError, match="immutable"):
        run_walk_forward(
            folds=default_folds(), output_root=tmp_path, smoke=False,
            run_id="full-run-001", trainer=trainer,
            tester=lambda **kwargs: {"test_sharpe": 0.0},
            coverage_checker=lambda: None, cfg=_contract_cfg(),
            **_factor_inputs(tmp_path),
        )

    assert calls == []
    assert sentinel.read_text(encoding="utf-8") == '{"preserve": true}\n'
    assert list(run_root.iterdir()) == [sentinel]


def test_full_run_reserves_explicit_run_root_before_workflow_callbacks(tmp_path, monkeypatch):
    _mock_real_stress(monkeypatch)
    run_root = tmp_path / "full-run-001"

    def coverage_checker():
        assert run_root.is_dir()

    def trainer(**kwargs):
        with pytest.raises(FileExistsError, match="immutable"):
            run_walk_forward(
                folds=default_folds(), output_root=tmp_path, smoke=False,
                run_id="full-run-001", trainer=lambda **_: None,
                tester=lambda **_: {"test_sharpe": 0.0},
                coverage_checker=lambda: None, cfg=_contract_cfg(),
                **_factor_inputs(tmp_path),
            )
        return _trainer_artifacts(kwargs, val_sharpe=1.0)

    run_walk_forward(
        folds=[default_folds()[0]], output_root=tmp_path, smoke=False,
        run_id="full-run-001", trainer=trainer,
        tester=lambda **_: {"test_sharpe": 0.0}, coverage_checker=coverage_checker,
        cfg=_contract_cfg(), **_factor_inputs(tmp_path),
    )


def test_repeated_smoke_replaces_existing_selection_artifacts(tmp_path, monkeypatch):
    import scripts.factor_selection as factor_selection

    def trainer(**kwargs):
        return _trainer_artifacts(kwargs, val_sharpe=1.0)

    run_args = {
        "folds": default_folds(), "output_root": tmp_path, "smoke": True,
        "trainer": trainer, "tester": lambda **kwargs: {"test_sharpe": 0.0},
        "coverage_checker": lambda: None, "cfg": _contract_cfg(),
        **_factor_inputs(tmp_path),
    }
    run_args["selector"] = _select_factors_with_artifacts
    real_writer = factor_selection.write_selection_artifacts

    def write_into_empty_fold(*args, **kwargs):
        output_dir = pathlib.Path(args[3])
        assert not any(output_dir.iterdir())
        return real_writer(*args, **kwargs)

    monkeypatch.setattr(factor_selection, "write_selection_artifacts", write_into_empty_fold)

    run_walk_forward(**run_args)
    run_walk_forward(**run_args)

    selection_dir = tmp_path / "smoke" / "candidate_20f" / "selection" / "fold3"
    assert (selection_dir / "selected_factors.json").is_file()
    assert (selection_dir / "selection_report.json").is_file()


def test_repeated_smoke_removes_stale_test_artifacts_from_prior_candidate(tmp_path):
    selected = {"reward": "low", "buffer": "tight"}

    def trainer(**kwargs):
        chosen = selected["reward"] if kwargs["stage"] == "reward_ablation" else selected["buffer"]
        return _trainer_artifacts(kwargs, val_sharpe=float(kwargs["candidate"] == chosen))

    run_args = {
        "folds": default_folds(), "output_root": tmp_path, "smoke": True,
        "trainer": trainer, "tester": lambda **kwargs: {"test_sharpe": 0.0},
        "coverage_checker": lambda: None, "cfg": _contract_cfg(),
        **_factor_inputs(tmp_path),
    }
    run_walk_forward(**run_args)
    stale_test = tmp_path / "smoke" / "candidate_20f" / "test" / "fold3" / "low__tight" / "seed0.json"
    assert stale_test.is_file()

    selected.update(reward="constrained", buffer="wide")
    run_walk_forward(**run_args)

    fresh_test = tmp_path / "smoke" / "candidate_20f" / "test" / "fold3" / "constrained__wide" / "seed0.json"
    assert fresh_test.is_file()
    assert not stale_test.exists()


@pytest.mark.parametrize(
    "unsafe_cfg_value",
    [pd.DataFrame({"unsafe": [1]}), _MutableCfgMetadata()],
    ids=["dataframe", "custom-object"],
)
def test_walk_forward_rejects_unsafe_mutable_runtime_cfg_values(tmp_path, unsafe_cfg_value):
    cfg = _contract_cfg()
    cfg["arbitrary_metadata"] = unsafe_cfg_value
    selector_calls = []
    trainer_calls = []

    def selector(**kwargs):
        selector_calls.append(kwargs)
        return _select_configured_factors(**kwargs)

    def trainer(**kwargs):
        trainer_calls.append(kwargs)
        return _trainer_artifacts(kwargs, val_sharpe=1.0)

    factor_inputs = _factor_inputs(tmp_path)
    factor_inputs["selector"] = selector
    with pytest.raises(TypeError, match="unsupported mutable walk-forward config value"):
        run_walk_forward(
            folds=[default_folds()[0]], output_root=tmp_path, smoke=True,
            trainer=trainer, tester=lambda **kwargs: {"test_sharpe": 0.0},
            coverage_checker=lambda: None, cfg=cfg, **factor_inputs,
        )

    assert selector_calls == []
    assert trainer_calls == []


@pytest.mark.parametrize("smoke", [True, False], ids=["smoke", "full"])
def test_walk_forward_validates_cfg_before_mutating_output_root(tmp_path, smoke):
    run_id = "retryable-full-run"
    run_root = tmp_path / ("smoke" if smoke else run_id)
    if smoke:
        run_root.mkdir()
        sentinel = run_root / "existing-evidence.json"
        sentinel.write_text('{"preserve": true}\n', encoding="utf-8")

    cfg = _contract_cfg()
    cfg["unsafe_metadata"] = _MutableCfgMetadata()
    with pytest.raises(TypeError, match="unsupported mutable walk-forward config value"):
        run_walk_forward(
            folds=[default_folds()[0]], output_root=tmp_path, smoke=smoke,
            run_id=run_id, trainer=lambda **_: None,
            tester=lambda **_: {"test_sharpe": 0.0}, coverage_checker=lambda: None,
            cfg=cfg, **_factor_inputs(tmp_path),
        )

    if smoke:
        assert sentinel.read_text(encoding="utf-8") == '{"preserve": true}\n'
        assert list(run_root.iterdir()) == [sentinel]
    else:
        assert not run_root.exists()


@pytest.mark.parametrize(
    "dependency_key, dependency",
    [("selector", lambda **_: []), ("candidate_cache", object())],
)
@pytest.mark.parametrize("smoke", [True, False], ids=["smoke", "full"])
def test_walk_forward_rejects_config_carried_dependencies_before_output_mutation(
        tmp_path, dependency_key, dependency, smoke):
    run_id = "unreserved-full-run"
    run_root = tmp_path / ("smoke" if smoke else run_id)
    if smoke:
        run_root.mkdir()
        sentinel = run_root / "existing-evidence.json"
        sentinel.write_text('{"preserve": true}\n', encoding="utf-8")

    selector_calls = []

    def selector(**kwargs):
        selector_calls.append(kwargs)
        return _select_configured_factors(**kwargs)

    cfg = _contract_cfg()
    cfg[dependency_key] = dependency
    factor_inputs = _factor_inputs(tmp_path)
    factor_inputs["selector"] = selector
    with pytest.raises(ValueError, match="must be passed explicitly"):
        run_walk_forward(
            folds=[default_folds()[0]], output_root=tmp_path, smoke=smoke,
            run_id=run_id, trainer=lambda **_: None,
            tester=lambda **_: {"test_sharpe": 0.0}, coverage_checker=lambda: None,
            cfg=cfg, **factor_inputs,
        )

    assert selector_calls == []
    if smoke:
        assert sentinel.read_text(encoding="utf-8") == '{"preserve": true}\n'
        assert list(run_root.iterdir()) == [sentinel]
    else:
        assert not run_root.exists()


def test_walk_forward_freezes_an_independent_readonly_numpy_runtime_cfg_value():
    value = np.array([1.0, 2.0])
    frozen = walk_forward._freeze(value)

    assert np.array_equal(frozen, value)
    assert frozen is not value
    assert frozen.flags.writeable is False
    value[0] = -1.0
    assert frozen.tolist() == [1.0, 2.0]


def test_default_tester_passes_checkpoint_contract_and_metadata_to_loader(tmp_path, monkeypatch):
    import scripts.env as env_module
    import scripts.metrics as metrics_module
    import scripts.train as train_module

    checkpoint = tmp_path / "best.zip"
    checkpoint.write_bytes(b"checkpoint")
    metadata = tmp_path / "best_metadata.json"
    names = ["mom_20", "vol_20"]
    expected = _checkpoint_factor_contract(_contract_cfg(names), fold=3)
    scaler_path = tmp_path / "scaler.json"
    fields = tuple(state_fields(names))
    ObservationScaler(
        schema_version=STATE_SCHEMA_VERSION, fields=fields,
        mean=(0.0,) * len(fields), scale=(1.0,) * len(fields),
    ).save(scaler_path, factor_contract=expected)
    captured = {}

    class FakeEnv:
        def reset(self, seed=0):
            return 0, {}

        def step(self, action):
            return 0, 0.0, True, False, {
                "daily_net_rets": [0.01],
                "daily_gross_rets": [0.02],
                "turnover": 0.1,
            }

    class FakeModel:
        def predict(self, obs, deterministic=True):
            return 0, None

    monkeypatch.setattr(env_module, "PortfolioEnv", lambda *args, **kwargs: FakeEnv())
    monkeypatch.setattr(metrics_module, "metrics_pack", lambda values, label: {
        "sharpe": 1.0, "arr": 0.1, "mdd": -0.1,
    })

    def fake_load(path, env, **kwargs):
        captured.update(kwargs)
        return FakeModel()

    monkeypatch.setattr(train_module, "load_ppo", fake_load)

    _default_tester(
        checkpoint_path=str(checkpoint),
        validation_result={
                "factor_contract": expected,
                "metadata_path": str(metadata),
                "scaler_path": str(scaler_path),
        },
        cfg={"factor_names": names},
        reward_variant="none",
        buffer_config={},
        test_range=("2024-01-01", "2024-01-02"),
        features_df=object(),
        market_state_df=object(),
        seed=0,
    )

    assert captured["expected_factor_contract"] == expected
    assert captured["metadata_path"] == str(metadata)


def test_walk_forward_explicitly_persists_default_directions():
    cfg = _contract_cfg()
    cfg.pop("factor_directions")

    with pytest.raises(ValueError, match="factor_directions"):
        _checkpoint_factor_contract(cfg, fold=2, run_id="selection-default")


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("selected_factors", ["vol_20", "mom_20", "reversal_5"],
         "selected_factors and factor_names"),
        ("schema_version", "state-v2", "state_schema_version and schema_version"),
    ],
)
def test_walk_forward_rejects_conflicting_factor_contract_aliases(field, value, message):
    cfg = _contract_cfg()
    cfg[field] = value

    with pytest.raises(ValueError, match=message):
        _checkpoint_factor_contract(cfg, fold=2, run_id="selection-42")


def test_walk_forward_rejects_trainer_result_without_factor_contract(tmp_path):
    def trainer(**kwargs):
        return {"val_sharpe": 1.0}

    with pytest.raises(ValueError, match="trainer result must contain factor_contract"):
        run_walk_forward(
            folds=default_folds(), output_root=tmp_path, smoke=True,
            trainer=trainer, tester=lambda **kwargs: {"test_sharpe": 0.0},
            coverage_checker=lambda: None, cfg=_contract_cfg(),
            **_factor_inputs(tmp_path),
        )


def test_walk_forward_rejects_trainer_result_without_artifact_evidence(tmp_path):
    def trainer(**kwargs):
        return {"val_sharpe": 1.0, "factor_contract": kwargs["factor_contract"]}

    with pytest.raises(ValueError, match="metadata_path"):
        run_walk_forward(
            folds=default_folds(), output_root=tmp_path, smoke=True,
            trainer=trainer, tester=lambda **kwargs: {"test_sharpe": 0.0},
            coverage_checker=lambda: None, cfg=_contract_cfg(),
            **_factor_inputs(tmp_path),
        )


def test_smoke_never_writes_factor_approval(tmp_path):
    def trainer(**kwargs):
        return _trainer_artifacts(kwargs, val_sharpe=1.0)

    def tester(**kwargs):
        return {"test_sharpe": 1.0}

    run_walk_forward(
        folds=[default_folds()[0]], output_root=tmp_path, smoke=True,
        trainer=trainer, tester=tester, coverage_checker=lambda: None,
        cfg=_contract_cfg(), **_factor_inputs(tmp_path),
    )

    assert not (tmp_path / "smoke" / "approval.json").exists()


def test_stress_evidence_extracts_worst_segment_metrics():
    from scripts.walk_forward import _stress_evidence
    results = [
        {"name": "2008_gfc", "skipped": True, "reason": "insufficient history"},
        {"name": "2015_ashare_crash", "skipped": False,
         "metrics": {"rl": {"mdd": -0.15, "calmar": 0.8},
                     "static_factor_equal": {"mdd": -0.16, "calmar": 0.7}},
         "diagnostics": {"long_exposure_util": 0.9}},
        {"name": "2020_covid", "skipped": False,
         "metrics": {"rl": {"mdd": -0.10, "calmar": 0.5},
                     "static_factor_equal": {"mdd": -0.08, "calmar": 0.6}},
         "diagnostics": {"long_exposure_util": 0.4}},
    ]
    mdd, calmar_excess, util, evidence = _stress_evidence(results)
    assert mdd == pytest.approx(-0.15)
    assert calmar_excess == pytest.approx(-0.1)   # 2020 段 0.5 − 0.6 最差
    assert util == pytest.approx(0.4)
    assert evidence[0] == {"name": "2008_gfc", "skipped": True, "reason": "insufficient history"}
    assert evidence[2]["stress_calmar_excess"] == pytest.approx(-0.1)
    assert evidence[2]["long_exposure_util"] == pytest.approx(0.4)


def test_stress_evidence_all_skipped_returns_none():
    from scripts.walk_forward import _stress_evidence
    results = [{"name": "2008_gfc", "skipped": True, "reason": "x"}]
    assert _stress_evidence(results)[:3] == (None, None, None)


def test_full_run_candidate_stress_uses_each_folds_own_inputs(tmp_path, monkeypatch):
    """Regression: the candidate stress branch must use fold N's own materialized
    inputs, never the stale ``fold_inputs`` loop variable (last fold's panels)."""
    fold_one_names = CANDIDATE_FACTOR_NAMES[:20]
    fold_two_names = CANDIDATE_FACTOR_NAMES[:19] + [FACTOR_CATALOG[20].name]
    names_by_fold = {1: fold_one_names, 2: fold_two_names}

    def selector(*, panel, fold, cfg):
        return [{"name": name, "direction": 1} for name in names_by_fold[fold.fold]]

    # Market state carries one distinctive return column per selected factor,
    # mirroring the real build_market_state output.
    def fake_build_market_state(features, index_returns, cfg, factor_names):
        return pd.DataFrame({
            "trade_date": pd.to_datetime(features["trade_date"]),
            **{f"{name}_factor_ret_20": 0.0 for name in factor_names},
        })

    monkeypatch.setattr(walk_forward, "build_market_state", fake_build_market_state)

    stress_calls = []

    def stub_stress_tester(**kwargs):
        stress_calls.append({
            "branch": kwargs["branch"], "fold": kwargs["fold"], "seed": kwargs["seed"],
            "market_state_columns": tuple(kwargs["market_state_df"].columns),
        })
        return [{
            "name": "stub_segment", "skipped": False,
            "metrics": {
                "rl": {"mdd": -0.10, "calmar": 0.9},
                "static_factor_equal": {"mdd": -0.11, "calmar": 0.8},
            },
            "diagnostics": {"long_exposure_util": 0.9},
        }]

    monkeypatch.setattr(walk_forward, "_default_stress_tester", stub_stress_tester)

    def trainer(**kwargs):
        return _trainer_artifacts(kwargs, val_sharpe=1.0)

    def tester(**kwargs):
        return {
            "test_sharpe": 1.0, "oos_sharpe": 1.0 if kwargs["branch"] == "candidate_20f" else 0.8, "oos_arr": 0.2,
            "oos_mdd": -0.1, "excess_return": 0.2,
            "strongest_baseline_sharpe": 0.1, "strongest_baseline_mdd": -0.2,
            "annualized_turnover": 1.0, "cost_2x_oos_sharpe": 0.5,
            "no_leakage_tests_passed": True, "state_quality_tests_passed": True,
        }

    all_names = CANDIDATE_FACTOR_NAMES + [FACTOR_CATALOG[20].name]
    run_walk_forward(
        folds=default_folds()[:2], output_root=tmp_path, smoke=False,
        run_id="stress-inputs-regression", trainer=trainer, tester=tester,
        coverage_checker=lambda: None, cfg=_contract_cfg(all_names),
        selector=selector, candidate_cache=_CandidateCache(all_names),
        cache_root=tmp_path / "factor-cache", index_returns=pd.DataFrame(),
    )

    for fold_no, names in names_by_fold.items():
        candidate_calls = [
            call for call in stress_calls
            if call["branch"] == "candidate_20f" and call["fold"] == fold_no
        ]
        assert candidate_calls, f"no candidate stress calls recorded for fold {fold_no}"
        distinctive = f"{names[-1]}_factor_ret_20"
        for call in candidate_calls:
            assert distinctive in call["market_state_columns"], (
                f"fold {fold_no} candidate stress received a market_state without "
                f"its own factor column {distinctive}: {call['market_state_columns']}"
            )
