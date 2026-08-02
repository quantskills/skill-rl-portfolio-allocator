import json
import sys

import pandas as pd
import numpy as np

import scripts.allocate as allocate


def test_candidate_approval_bundle_preserves_referenced_paths(tmp_path):
    source = tmp_path / "research"
    candidate = tmp_path / "candidate"
    source.mkdir()
    method = {"schema_version": "state-v1", "method": "ppo"}
    (source / "method.json").write_text(json.dumps(method), encoding="utf-8")
    (source / "gates.json").write_text(json.dumps({"research_ok": True}), encoding="utf-8")
    approval = {
        "research_ok": True,
        "schema_version": "state-v1",
        "method_id": allocate.frozen_method_id(method),
        "method_path": "method.json",
        "gates_path": "gates.json",
    }
    approval_path = source / "approval.json"
    approval_path.write_text(json.dumps(approval), encoding="utf-8")

    allocate.copy_approval_bundle(approval_path, candidate)

    assert allocate.load_research_approval(candidate / "approval.json") == approval
    assert (candidate / "method.json").read_text(encoding="utf-8") == (
        source / "method.json"
    ).read_text(encoding="utf-8")
    assert (candidate / "gates.json").exists()


def test_retrain_candidate_routes_all_artifacts_away_from_production(
    tmp_path, monkeypatch
):
    candidate = tmp_path / "candidate"
    formal_checkpoint = tmp_path / "checkpoints" / "production.zip"
    formal_allocations = tmp_path / "production" / "allocations.parquet"
    approval = tmp_path / "approval.json"
    method = {"schema_version": "state-v1", "method": "ppo"}
    (tmp_path / "method.json").write_text(json.dumps(method), encoding="utf-8")
    (tmp_path / "gates.json").write_text(json.dumps({"research_ok": True}), encoding="utf-8")
    approval.write_text(json.dumps({
        "research_ok": True,
        "schema_version": "state-v1",
        "method_id": allocate.frozen_method_id(method),
        "method_path": "method.json",
        "gates_path": "gates.json",
    }), encoding="utf-8")

    monkeypatch.setattr(allocate, "load_research_approval", lambda path: {})
    monkeypatch.setattr(
        allocate.pd,
        "read_parquet",
        lambda path: pd.DataFrame({"trade_date": [pd.Timestamp("2024-01-02")]})
        if "features" in str(path)
        else pd.DataFrame(),
    )

    def fake_retrain(*args, checkpoint_path, **kwargs):
        checkpoint = __import__("pathlib").Path(checkpoint_path)
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        checkpoint.write_bytes(b"candidate-checkpoint")
        (checkpoint.parent / "scaler.json").write_text(json.dumps({
            "schema_version": "state-v1",
            "fields": list(allocate.state_fields(allocate.FACTOR_NAMES)),
            "mean": [0.0] * len(allocate.state_fields(allocate.FACTOR_NAMES)),
            "scale": [1.0] * len(allocate.state_fields(allocate.FACTOR_NAMES)),
        }), encoding="utf-8")
        (checkpoint.parent / "checkpoint_metadata.json").write_text(
            json.dumps({"schema_version": "state-v1"}), encoding="utf-8"
        )
        return checkpoint_path

    monkeypatch.setattr(allocate, "retrain_production", fake_retrain)
    inferred = {}
    monkeypatch.setattr(
        allocate,
        "infer_latest",
        lambda *args, **kwargs: (
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
    assert inferred["observation_scaler"].schema_version == "state-v1"
    assert not formal_checkpoint.exists()
    assert not formal_allocations.exists()


def test_retrain_production_fits_and_saves_independent_scaler(tmp_path, monkeypatch):
    class FakeEnv:
        def __init__(self, *args, **kwargs):
            self.observation_scaler = None

    class FakeScaler:
        def save(self, path):
            target = __import__("pathlib").Path(path)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text("scaler", encoding="utf-8")

    fitted = FakeScaler()
    captured = {}
    monkeypatch.setattr(allocate, "PortfolioEnv", FakeEnv)
    monkeypatch.setattr(allocate, "select_device", lambda _: "cpu")
    monkeypatch.setattr(allocate, "fit_production_scaler", lambda env, seed: fitted)

    def fake_train(env, **kwargs):
        captured["scaler"] = env.observation_scaler
        __import__("pathlib").Path(kwargs["save_path"]).write_bytes(b"checkpoint")

    monkeypatch.setattr(allocate, "train_ppo", fake_train)
    features = pd.DataFrame({"trade_date": [pd.Timestamp("2024-01-01")]})
    output = tmp_path / "candidate" / "checkpoint.zip"

    result = allocate.retrain_production(
        features,
        {"train_device": "cpu"},
        timesteps=1,
        seed=0,
        checkpoint_path=str(output),
        market_state_df=pd.DataFrame({"trade_date": [pd.Timestamp("2024-01-01")] }),
        scaler_path=str(tmp_path / "candidate" / "scaler.json"),
    )

    assert result == str(output)
    assert captured["scaler"] is fitted
    assert (tmp_path / "candidate" / "scaler.json").exists()
