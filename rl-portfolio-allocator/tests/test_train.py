import inspect
import json
import numpy as np
import pandas as pd
import pytest
from scripts.config import get_config, FACTOR_NAMES, K
from scripts.env import PortfolioEnv
from scripts.train import (
    FACTOR_CONTRACT_FIELDS,
    ValidationSharpeCallback,
    load_ppo,
    train_ppo,
)
from scripts.state import exogenous_fields
from scripts.state import STATE_SCHEMA_VERSION


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
    market_state = pd.DataFrame(0.1, index=dates, columns=exogenous_fields(FACTOR_NAMES))
    return PortfolioEnv(feats, market_state, cfg, dates.min(), dates.max())


def test_train_ppo_accepts_eval_kwargs():
    sig = inspect.signature(train_ppo)
    for p in ("eval_env", "eval_freq", "n_eval_episodes", "patience", "ent_coef"):
        assert p in sig.parameters, f"missing param {p}"


def test_train_ppo_ent_coef_defaults_to_env_cfg_and_explicit_override():
    model = train_ppo(_toy_env(), total_timesteps=64, seed=0, device="cpu")
    assert model.ent_coef == pytest.approx(0.02)
    model = train_ppo(_toy_env(), total_timesteps=64, seed=0, device="cpu", ent_coef=0.05)
    assert model.ent_coef == pytest.approx(0.05)


def test_train_ppo_backward_compatible():
    # No eval_env, minimal steps, should complete without error
    model = train_ppo(_toy_env(), total_timesteps=64, seed=0, device="cpu")
    assert model is not None


def test_train_ppo_saving_writes_contract_bound_metadata(tmp_path):
    contract = {
        "factor_catalog_version": "catalog-v1",
        "factor_catalog_hash": "sha256:catalog",
        "factor_names": list(FACTOR_NAMES),
        "factor_directions": [1] * len(FACTOR_NAMES),
        "selection_run_id": "selection-1",
        "fold": 1,
        "schema_version": STATE_SCHEMA_VERSION,
    }
    checkpoint = tmp_path / "checkpoint.zip"

    train_ppo(
        _toy_env(), total_timesteps=64, seed=0, device="cpu",
        save_path=str(checkpoint), factor_contract=contract,
    )

    metadata = json.loads((tmp_path / "checkpoint_metadata.json").read_text())
    assert checkpoint.exists()
    assert metadata["selected_factors"] == list(FACTOR_NAMES)
    assert metadata["state_schema_version"] == STATE_SCHEMA_VERSION


def test_factor_selected_best_checkpoint_writes_strict_loadable_contract_metadata(tmp_path):
    contract = {
        "factor_catalog_version": "catalog-v1",
        "factor_catalog_hash": "sha256:catalog",
        "selected_factors": list(FACTOR_NAMES),
        "factor_directions": [1] * len(FACTOR_NAMES),
        "selection_run_id": "selection-1",
        "fold": 1,
        "state_schema_version": STATE_SCHEMA_VERSION,
    }
    checkpoint = tmp_path / "checkpoint.zip"
    best_checkpoint = tmp_path / "checkpoint_best.zip"
    best_metadata = tmp_path / "checkpoint_best_metadata.json"
    env = _toy_env()

    train_ppo(
        env, total_timesteps=64, seed=0, device="cpu", save_path=str(checkpoint),
        eval_env=_toy_env(), factor_contract=contract,
    )

    metadata = json.loads(best_metadata.read_text())
    assert best_checkpoint.exists()
    assert {field: metadata[field] for field in FACTOR_CONTRACT_FIELDS} == contract
    assert load_ppo(
        str(best_checkpoint), _toy_env(), expected_factor_contract=contract,
        metadata_path=best_metadata,
    ) is not None


def test_validation_callback_rejects_best_checkpoint_without_complete_contract(tmp_path):
    with pytest.raises(ValueError, match="best checkpoint factor contract must be an object"):
        ValidationSharpeCallback(
            _toy_env(), best_model_path=tmp_path / "best.zip",
        )


def test_train_ppo_rejects_best_checkpoint_with_mismatched_metadata(tmp_path):
    contract = {
        "factor_catalog_version": "catalog-v1",
        "factor_catalog_hash": "sha256:catalog",
        "selected_factors": list(FACTOR_NAMES),
        "factor_directions": [1] * len(FACTOR_NAMES),
        "selection_run_id": "selection-1",
        "fold": 1,
        "state_schema_version": STATE_SCHEMA_VERSION,
    }
    best_checkpoint = tmp_path / "checkpoint_best.zip"
    callback = ValidationSharpeCallback(
        _toy_env(), best_model_path=best_checkpoint, factor_contract=contract,
    )
    evaluate = callback._evaluate

    def write_mismatched_metadata():
        evaluate()
        metadata_path = tmp_path / "checkpoint_best_metadata.json"
        metadata = json.loads(metadata_path.read_text())
        metadata["factor_catalog_hash"] = "sha256:mismatch"
        metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

    callback._evaluate = write_mismatched_metadata

    with pytest.raises(ValueError, match="factor checkpoint contract mismatch: factor_catalog_hash"):
        train_ppo(
            _toy_env(), total_timesteps=64, seed=0, device="cpu",
            callback=callback,
        )


def test_validation_callback_uses_daily_returns_and_requested_defaults():
    callback = ValidationSharpeCallback(_toy_env(), eval_freq=10000, patience=5)
    assert callback.eval_freq == 10000
    assert callback.patience == 5
    assert callback.deterministic is True


def test_training_log_records_dual_lambda_for_constrained_variant(tmp_path):
    env = _toy_env()
    env.cfg["reward_variant"] = "constrained"
    env.cfg.update({"target_mdd": 0.10, "dual_lr": 0.05,
                    "recovery_credit": 0.1, "downside_vol_coeff": 0.2})
    log_path = tmp_path / "train.jsonl"
    train_ppo(env, total_timesteps=64, seed=0, device="cpu",
              training_log_path=str(log_path))
    records = [json.loads(line) for line in log_path.read_text().splitlines() if line.strip()]
    assert any("dual_lambda_last" in record for record in records)
