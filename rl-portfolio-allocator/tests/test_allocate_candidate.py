import json
import sys

import pandas as pd
import numpy as np
import pytest

import scripts.allocate as allocate
from scripts.config import FACTOR_NAMES
from scripts.state import STATE_SCHEMA_VERSION, exogenous_fields, state_fields


def _factor_contract(selected_factors=None):
    names = list(selected_factors or FACTOR_NAMES)
    return {
        "factor_catalog_version": "catalog-v1",
        "factor_catalog_hash": "sha256:catalog",
        "selected_factors": names,
        "factor_directions": [(-1 if index % 2 else 1) for index, _ in enumerate(names)],
        "selection_run_id": "selection-42",
        "fold": 3,
        "state_schema_version": STATE_SCHEMA_VERSION,
    }


def _write_passing_approval(root, method):
    candidate_rows = []
    control_rows = []
    for branch, rows, sharpe, mdd in (
        ("candidate_20f", candidate_rows, 0.60, -0.25),
        ("control_6f", control_rows, 0.40, -0.24),
    ):
        for fold in range(1, 4):
            for seed in range(5):
                path = root / branch / "stress" / f"fold{fold}" / f"seed{seed}.json"
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(json.dumps({
                    "branch": branch, "fold": fold, "seed": seed, "stress_mdd": mdd,
                }), encoding="utf-8")
                rows.append({
                    "branch": branch, "fold": fold, "seed": seed,
                    "oos_sharpe": sharpe, "cost_2x_oos_sharpe": 0.12,
                    "annualized_turnover": 8.0, "stress_mdd": mdd,
                    "stress_calmar_excess": 0.02 if branch == "candidate_20f" else 0.01,
                    "stress_long_exposure_util": 0.8 if branch == "candidate_20f" else 0.75,
                    "stress_artifact_path": str(path.relative_to(root)),
                    "stress_artifact_sha256": allocate._file_id(path),
                })
    comparison = {
        "candidate_median_oos_sharpe": 0.60,
        "control_median_oos_sharpe": 0.40,
        "positive_excess_folds": 3,
        "candidate_cost_2x_oos_sharpe": 0.12,
        "candidate_annualized_turnover": 8.0,
        "candidate_stress_mdd": -0.25,
        "control_stress_mdd": -0.24,
        "candidate_stress_calmar_excess": 0.02,
        "candidate_stress_long_exposure_util": 0.8,
        "paired_evidence": {
            "candidate_20f": {"rows": candidate_rows},
            "control_6f": {"rows": control_rows},
        },
    }
    comparison_path = root / "comparison.json"
    comparison_path.write_text(json.dumps(comparison), encoding="utf-8")
    comparison_id = allocate._file_id(comparison_path)
    (root / "method.json").write_text(json.dumps(method), encoding="utf-8")
    selection_relative = "candidate_20f/selection/fold3/selected_factors.json"
    selection_path = root / selection_relative
    selection_path.parent.mkdir(parents=True, exist_ok=True)
    selection_path.write_text(json.dumps({
        "fold": 3,
        "selected_factors": [
            {"name": name, "direction": direction}
            for name, direction in zip(method["selected_factors"], method["factor_directions"])
        ],
    }), encoding="utf-8")
    (root / "gates.json").write_text(json.dumps({
        "research_ok": True, "comparison_path": "comparison.json",
        "comparison_id": comparison_id,
    }), encoding="utf-8")
    approval = {
        "research_ok": True, "run_mode": "full", "fold_count": 3, "seed_count": 5,
        "schema_version": method["schema_version"],
        "method_id": allocate.frozen_method_id(method), "method_path": "method.json",
        "gates_path": "gates.json", "comparison_path": "comparison.json",
        "comparison_id": comparison_id,
        "factor_selection_path": selection_relative,
        "factor_selection_id": allocate._file_id(selection_path),
    }
    approval_path = root / "approval.json"
    approval_path.write_text(json.dumps(approval), encoding="utf-8")
    return approval_path


def test_candidate_approval_bundle_preserves_referenced_paths(tmp_path):
    source = tmp_path / "research"
    candidate = tmp_path / "candidate"
    source.mkdir()
    method = {
        "schema_version": STATE_SCHEMA_VERSION,
        "method": "ppo",
        **_factor_contract(),
    }
    approval_path = _write_passing_approval(source, method)
    approval = json.loads(approval_path.read_text(encoding="utf-8"))

    allocate.copy_approval_bundle(approval_path, candidate)

    assert allocate.load_research_approval(candidate / "approval.json") == approval
    assert (candidate / "method.json").read_text(encoding="utf-8") == (
        source / "method.json"
    ).read_text(encoding="utf-8")
    assert (candidate / "gates.json").exists()
    assert (candidate / "comparison.json").exists()
    assert (candidate / "candidate_20f" / "stress" / "fold1" / "seed0.json").exists()
    assert (candidate / "control_6f" / "stress" / "fold3" / "seed4.json").exists()
    assert (candidate / "candidate_20f" / "selection" / "fold3" / "selected_factors.json").exists()


def test_retrain_candidate_routes_all_artifacts_away_from_production(
    tmp_path, monkeypatch
):
    candidate = tmp_path / "candidate"
    formal_checkpoint = tmp_path / "checkpoints" / "production.zip"
    formal_allocations = tmp_path / "production" / "allocations.parquet"
    approval = tmp_path / "approval.json"
    selected_factors = [FACTOR_NAMES[4], FACTOR_NAMES[0], FACTOR_NAMES[5]]
    method = {
        "schema_version": STATE_SCHEMA_VERSION,
        "method": "ppo",
        **_factor_contract(selected_factors),
    }
    approval = _write_passing_approval(tmp_path, method)

    monkeypatch.setattr(
        allocate, "load_approved_method",
        lambda path: (json.loads(approval.read_text()), method),
    )
    panels = {}

    def fake_panels(root, factor_contract, cfg):
        panels["factor_contract"] = factor_contract
        return (
            pd.DataFrame({"trade_date": [pd.Timestamp("2024-01-02")]}),
            pd.DataFrame(),
        )

    monkeypatch.setattr(allocate, "_load_production_panels", fake_panels)

    captured = {}

    def fake_retrain(features_df, cfg, *args, checkpoint_path, **kwargs):
        captured["retrain_cfg"] = dict(cfg)
        checkpoint = __import__("pathlib").Path(checkpoint_path)
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        checkpoint.write_bytes(b"candidate-checkpoint")
        contract = _factor_contract(selected_factors)
        (checkpoint.parent / "scaler.json").write_text(json.dumps({
            "schema_version": STATE_SCHEMA_VERSION,
            "fields": list(allocate.state_fields(selected_factors)),
            "mean": [0.0] * len(allocate.state_fields(selected_factors)),
            "scale": [1.0] * len(allocate.state_fields(selected_factors)),
            **contract,
        }), encoding="utf-8")
        (checkpoint.parent / "checkpoint_metadata.json").write_text(
            json.dumps({"schema_version": STATE_SCHEMA_VERSION, **contract}), encoding="utf-8"
        )
        return checkpoint_path

    monkeypatch.setattr(allocate, "retrain_production", fake_retrain)
    inferred = {}
    monkeypatch.setattr(
        allocate,
        "infer_latest",
        lambda *args, **kwargs: (
            captured.update({"infer_cfg": dict(args[1])})
            or
            inferred.update(kwargs)
            or pd.DataFrame([{
                "trade_date": pd.Timestamp("2024-01-02"),
                "side": "cash",
                "weight": 1.0,
            }])
        ),
    )
    monkeypatch.setattr(allocate, "save_allocations", lambda df, path: pd.DataFrame(df).to_parquet(path))
    monkeypatch.setattr(sys, "argv", [
        "allocate.py",
        "--retrain",
        "--timesteps",
        "1",
        "--approval",
        str(approval),
        "--candidate-dir",
        str(candidate),
    ])

    allocate.main()

    assert (candidate / "checkpoint.zip").read_bytes() == b"candidate-checkpoint"
    assert (candidate / "approval.json").read_text(encoding="utf-8") == approval.read_text(
        encoding="utf-8"
    )
    assert (candidate / "allocations.parquet").exists()
    assert inferred["observation_scaler"].schema_version == STATE_SCHEMA_VERSION
    assert captured["retrain_cfg"]["factor_names"] == selected_factors
    assert captured["retrain_cfg"]["k"] == len(selected_factors)
    assert captured["infer_cfg"]["factor_names"] == selected_factors
    assert captured["infer_cfg"]["k"] == len(selected_factors)
    assert panels["factor_contract"]["selected_factors"] == selected_factors
    assert not formal_checkpoint.exists()
    assert not formal_allocations.exists()


def test_retrain_production_fits_and_saves_independent_scaler(tmp_path, monkeypatch):
    class FakeEnv:
        def __init__(self, *args, **kwargs):
            self.observation_scaler = None

    class FakeScaler:
        def save(self, path, factor_contract=None):
            target = __import__("pathlib").Path(path)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(
                json.dumps({
                    "schema_version": STATE_SCHEMA_VERSION,
                    "fields": list(allocate.state_fields(FACTOR_NAMES)),
                    "mean": [0.0] * len(allocate.state_fields(FACTOR_NAMES)),
                    "scale": [1.0] * len(allocate.state_fields(FACTOR_NAMES)),
                    **(factor_contract or {}),
                }),
                encoding="utf-8",
            )

    fitted = FakeScaler()
    captured = {}
    monkeypatch.setattr(allocate, "PortfolioEnv", FakeEnv)
    monkeypatch.setattr(allocate, "select_device", lambda _: "cpu")
    monkeypatch.setattr(
        allocate,
        "fit_production_scaler",
        lambda env, seed, factor_names=None: fitted,
    )

    def fake_train(env, **kwargs):
        captured["scaler"] = env.observation_scaler
        __import__("pathlib").Path(kwargs["save_path"]).write_bytes(b"checkpoint")

    monkeypatch.setattr(allocate, "train_ppo", fake_train)
    features = pd.DataFrame({"trade_date": [pd.Timestamp("2024-01-01")]})
    market_state = pd.DataFrame([{
        "trade_date": pd.Timestamp("2024-01-01"),
        **{field: 0.1 for field in exogenous_fields(FACTOR_NAMES)},
    }])
    output = tmp_path / "candidate" / "checkpoint.zip"

    factor_contract = _factor_contract()
    result = allocate.retrain_production(
        features,
        {"train_device": "cpu", "factor_names": list(FACTOR_NAMES), "k": len(FACTOR_NAMES)},
        timesteps=1,
        seed=0,
        checkpoint_path=str(output),
        market_state_df=market_state,
        scaler_path=str(tmp_path / "candidate" / "scaler.json"),
        factor_contract=factor_contract,
    )

    assert result == str(output)
    assert captured["scaler"] is fitted
    scaler_payload = json.loads((tmp_path / "candidate" / "scaler.json").read_text(encoding="utf-8"))
    metadata_payload = json.loads(
        (tmp_path / "candidate" / "checkpoint_metadata.json").read_text(encoding="utf-8")
    )
    for field, value in factor_contract.items():
        assert scaler_payload[field] == value
        assert metadata_payload[field] == value


def test_retrain_and_infer_thread_dynamic_config_to_effective_range(tmp_path, monkeypatch):
    names = list(FACTOR_NAMES[:3])
    cfg = {"factor_names": names, "k": len(names), "train_device": "cpu"}
    features = pd.DataFrame({"trade_date": pd.date_range("2024-01-01", periods=2)})
    market_state = pd.DataFrame({"trade_date": pd.date_range("2024-01-01", periods=2)})
    seen = []

    def fake_range(features_df, market_state_df, start, end, cfg=None):
        seen.append(cfg)
        return pd.Timestamp(start), pd.Timestamp(end)

    class StopHere(Exception):
        pass

    monkeypatch.setattr(allocate, "effective_range", fake_range)
    monkeypatch.setattr(allocate, "PortfolioEnv", lambda *args, **kwargs: (_ for _ in ()).throw(StopHere()))

    with pytest.raises(StopHere):
        allocate.retrain_production(
            features, cfg, 1, 0, str(tmp_path / "checkpoint.zip"), market_state,
            factor_contract=_factor_contract(names),
        )
    with pytest.raises(StopHere):
        allocate.infer_latest(
            features, cfg, "checkpoint.zip", market_state,
            observation_scaler=__import__("scripts.observation", fromlist=["ObservationScaler"]).ObservationScaler(
                schema_version=STATE_SCHEMA_VERSION,
                fields=tuple(state_fields(names)),
                mean=(0.0,) * len(state_fields(names)),
                scale=(1.0,) * len(state_fields(names)),
            ),
            factor_contract=_factor_contract(names), metadata_path=tmp_path / "metadata.json",
        )

    assert seen == [cfg, cfg]


def test_load_production_scaler_rejects_partial_contract_for_subset_fields(tmp_path):
    selected_factors = list(FACTOR_NAMES[:3])
    path = tmp_path / "scaler.json"
    path.write_text(json.dumps({
        "schema_version": STATE_SCHEMA_VERSION,
        "fields": list(state_fields(selected_factors)),
        "mean": [0.0] * len(state_fields(selected_factors)),
        "scale": [1.0] * len(state_fields(selected_factors)),
        **_factor_contract(selected_factors),
    }), encoding="utf-8")

    with pytest.raises(ValueError, match="complete factor contract"):
        allocate.load_production_scaler(
            path,
            factor_names=selected_factors,
            factor_contract=allocate._factor_contract_from_payload({
                "schema_version": STATE_SCHEMA_VERSION,
                "factor_names": selected_factors,
            }),
        )


def test_allocation_rejects_config_and_contract_factor_disagreement():
    with pytest.raises(ValueError, match="disagree with config"):
        allocate._runtime_factor_names(
            {"factor_names": list(FACTOR_NAMES[:2])},
            {"factor_names": list(FACTOR_NAMES[2:4])},
        )


def test_load_production_scaler_rejects_fields_from_different_factor_order(tmp_path):
    selected_factors = list(FACTOR_NAMES[:3])
    path = tmp_path / "scaler.json"
    fields = state_fields(selected_factors)
    path.write_text(json.dumps({
        "schema_version": STATE_SCHEMA_VERSION,
        "fields": list(fields),
        "mean": [0.0] * len(fields),
        "scale": [1.0] * len(fields),
    }), encoding="utf-8")

    with pytest.raises(ValueError, match="disagree with scaler fields"):
        allocate.load_production_scaler(
            path,
            factor_names=list(FACTOR_NAMES[1:4]),
            factor_contract=_factor_contract(selected_factors),
        )


def test_infer_latest_labels_factor_weights_with_runtime_factor_order(monkeypatch):
    factor_names = [FACTOR_NAMES[3], FACTOR_NAMES[0], FACTOR_NAMES[5]]
    cfg = {
        "factor_names": factor_names,
        "long_notional": 1.0,
        "short_notional_cap": 0.3,
    }
    features = pd.DataFrame({"trade_date": pd.date_range("2024-01-01", periods=3)})
    market_state = pd.DataFrame({"trade_date": pd.date_range("2024-01-01", periods=3)})

    class FakeEnv:
        def __init__(self, *args, **kwargs):
            self.symbols = ["AAA"]
            self.prev_stock_w = np.array([0.25], dtype=float)
            self.dates = [pd.Timestamp("2024-01-01"), pd.Timestamp("2024-01-02")]
            self.t = 1
            self._F_by_date = {self.dates[0]: np.array([[1.0, 2.0, 3.0]])}
            self._done = False

        def reset(self, seed=0):
            return np.zeros(1, dtype=np.float32), {}

        def step(self, action):
            self.prev_stock_w = np.array([0.25], dtype=float)
            if self._done:
                raise AssertionError("step called after completion")
            self._done = True
            return (
                np.zeros(1, dtype=np.float32),
                0.0,
                True,
                False,
                {"factor_w": np.array([0.7, -0.2, 0.1], dtype=float)},
            )

    class FakeModel:
        def predict(self, obs, deterministic=True):
            return np.zeros(1, dtype=np.float32), None

    monkeypatch.setattr(allocate, "effective_range", lambda *args, **kwargs: (pd.Timestamp("2024-01-01"), pd.Timestamp("2024-01-03")))
    monkeypatch.setattr(allocate, "PortfolioEnv", FakeEnv)
    monkeypatch.setattr(allocate, "load_ppo", lambda *args, **kwargs: FakeModel())

    from scripts.observation import ObservationScaler
    observation_scaler = ObservationScaler(
        schema_version=STATE_SCHEMA_VERSION,
        fields=tuple(state_fields(factor_names)),
        mean=(0.0,) * len(state_fields(factor_names)),
        scale=(1.0,) * len(state_fields(factor_names)),
    )
    result = allocate.infer_latest(
        features, cfg, "checkpoint.zip", market_state,
        observation_scaler=observation_scaler,
        factor_contract=_factor_contract(factor_names), metadata_path="metadata.json",
    )

    factor_weights = json.loads(result.iloc[0]["factor_weights"])
    assert list(factor_weights.keys()) == factor_names
    assert factor_weights == {
        factor_names[0]: 0.7,
        factor_names[1]: -0.2,
        factor_names[2]: 0.1,
    }


def test_infer_latest_rejects_missing_observation_scaler():
    with pytest.raises(ValueError, match="validated observation scaler"):
        allocate.infer_latest(
            pd.DataFrame({"trade_date": pd.date_range("2024-01-01", periods=2)}),
            {"factor_names": list(FACTOR_NAMES[:2])}, "checkpoint.zip",
            pd.DataFrame(), factor_contract=_factor_contract(FACTOR_NAMES[:2]),
            metadata_path="metadata.json",
        )


def test_load_production_panels_materializes_only_approved_factors(tmp_path, monkeypatch):
    import scripts.factor_cache as factor_cache
    import scripts.market_state as market_state_module

    selected = [FACTOR_NAMES[4], FACTOR_NAMES[0]]
    contract = _factor_contract(selected)
    contract["factor_directions"] = [1, -1]
    captured = {}
    features = pd.DataFrame({"trade_date": [pd.Timestamp("2024-01-02")]})
    state = pd.DataFrame({"trade_date": [pd.Timestamp("2024-01-02")]})

    monkeypatch.setattr(
        factor_cache,
        "materialize_selected_panel",
        lambda root, records: captured.update({"root": root, "records": records}) or features,
    )
    monkeypatch.setattr(
        market_state_module,
        "build_market_state",
        lambda feats, index_returns, cfg, factor_names: (
            captured.update({"factor_names": factor_names}) or state
        ),
    )
    monkeypatch.setattr(
        allocate.pd,
        "read_parquet",
        lambda path: pd.DataFrame({"trade_date": [pd.Timestamp("2024-01-02")], "ret": [0.0]}),
    )

    out_features, out_state = allocate._load_production_panels(tmp_path, contract, {})

    assert out_features is features
    assert out_state is state
    assert captured["root"] == tmp_path / "data" / "factors"
    assert captured["records"] == [
        {"name": FACTOR_NAMES[4], "direction": 1},
        {"name": FACTOR_NAMES[0], "direction": -1},
    ]
    assert captured["factor_names"] == selected


def test_load_production_panels_uses_legacy_files_for_control_factors(tmp_path):
    contract = _factor_contract()
    (tmp_path / "data").mkdir()
    pd.DataFrame({"trade_date": [pd.Timestamp("2024-01-02")], "x": [1]}).to_parquet(
        tmp_path / "data" / "features.parquet", index=False
    )
    pd.DataFrame({"trade_date": [pd.Timestamp("2024-01-02")], "y": [2]}).to_parquet(
        tmp_path / "data" / "market_state.parquet", index=False
    )

    feats, market_state = allocate._load_production_panels(tmp_path, contract, {})

    assert list(feats.columns) == ["trade_date", "x"]
    assert list(market_state.columns) == ["trade_date", "y"]
