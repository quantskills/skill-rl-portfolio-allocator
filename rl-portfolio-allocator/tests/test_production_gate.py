"""Production approval and publish gates are fail-closed and atomic."""
import json

import numpy as np
import pandas as pd
import pytest

import scripts.allocate as allocate
from scripts.allocate import atomic_publish, load_research_approval
from scripts.config import FACTOR_NAMES
from scripts.state import STATE_SCHEMA_VERSION, state_fields
from scripts.validate import validate_weights


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


def _write_approval(root, *, research_ok=True, method=None, gates_ok=True, evidence=True):
    method = method or {
        "schema_version": STATE_SCHEMA_VERSION,
        "method": "ppo",
        **_factor_contract(),
    }
    (root / "method.json").write_text(json.dumps(method), encoding="utf-8")
    selection_names = list(
        method.get("selected_factors") or method.get("factor_names") or FACTOR_NAMES
    )
    selection_directions = list(
        method.get("factor_directions")
        or [(-1 if index % 2 else 1) for index, _ in enumerate(selection_names)]
    )
    selection_relative = "candidate_20f/selection/fold3/selected_factors.json"
    selection_path = root / selection_relative
    selection_path.parent.mkdir(parents=True, exist_ok=True)
    selection_path.write_text(json.dumps({
        "fold": 3,
        "selected_factors": [
            {"name": name, "direction": direction}
            for name, direction in zip(selection_names, selection_directions)
        ],
    }), encoding="utf-8")
    selection_id = allocate._file_id(selection_path)
    if evidence:
        candidate_rows = []
        control_rows = []
        for branch, rows, sharpe, mdd in (
            ("candidate_20f", candidate_rows, 0.60, -0.25),
            ("control_6f", control_rows, 0.40, -0.24),
        ):
            for fold in range(1, 4):
                for seed in range(5):
                    relative = f"{branch}/stress/fold{fold}/seed{seed}.json"
                    (root / relative).parent.mkdir(parents=True, exist_ok=True)
                    (root / relative).write_text(json.dumps({
                        "branch": branch, "fold": fold, "seed": seed,
                        "stress_mdd": mdd,
                    }), encoding="utf-8")
                    stress_artifact_sha256 = allocate._file_id(root / relative)
                    rows.append({
                        "branch": branch, "fold": fold, "seed": seed,
                        "oos_sharpe": sharpe, "cost_2x_oos_sharpe": 0.12,
                        "annualized_turnover": 8.0, "stress_mdd": mdd,
                        "stress_calmar_excess": 0.02 if branch == "candidate_20f" else 0.01,
                        "stress_long_exposure_util": 0.8 if branch == "candidate_20f" else 0.75,
                        "stress_artifact_path": relative,
                        "stress_artifact_sha256": stress_artifact_sha256,
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
    else:
        comparison_id = None
    (root / "gates.json").write_text(json.dumps({
        "research_ok": gates_ok,
        **({"comparison_path": "comparison.json", "comparison_id": comparison_id} if evidence else {}),
    }), encoding="utf-8")
    approval = {
        "research_ok": research_ok,
        "run_mode": "full",
        "fold_count": 3,
        "seed_count": 5,
        "schema_version": method["schema_version"],
        "method_id": __import__("scripts.walk_forward", fromlist=["frozen_method_id"])
        .frozen_method_id(method),
        "method_path": "method.json",
        "gates_path": "gates.json",
        "factor_selection_path": selection_relative,
        "factor_selection_id": selection_id,
        **({"comparison_path": "comparison.json", "comparison_id": comparison_id} if evidence else {}),
    }
    path = root / "approval.json"
    path.write_text(json.dumps(approval), encoding="utf-8")
    return path


def test_load_research_approval_rejects_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError, match="approval"):
        load_research_approval(tmp_path / "approval.json")


def test_load_research_approval_rejects_failed_gate(tmp_path):
    path = _write_approval(tmp_path, research_ok=False)
    with pytest.raises(RuntimeError, match="research_ok"):
        load_research_approval(path)


def test_load_research_approval_rejects_smoke_run(tmp_path):
    path = _write_approval(tmp_path)
    data = json.loads(path.read_text())
    data["run_mode"] = "smoke"
    path.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(RuntimeError, match="full walk-forward"):
        load_research_approval(path)


def test_load_research_approval_rejects_incomplete_full_metadata(tmp_path):
    path = _write_approval(tmp_path)
    data = json.loads(path.read_text())
    data.pop("fold_count")
    path.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ValueError, match="missing fields|full-run metadata"):
        load_research_approval(path)


def test_load_research_approval_accepts_valid_referenced_artifacts(tmp_path):
    path = _write_approval(tmp_path)
    approval = load_research_approval(path)
    assert approval["method_id"].startswith("sha256:")
    assert approval["method_path"] == "method.json"


def test_load_research_approval_rejects_evidence_free_passing_approval(tmp_path):
    path = _write_approval(tmp_path, evidence=False)

    with pytest.raises((FileNotFoundError, ValueError, RuntimeError), match="comparison|evidence"):
        load_research_approval(path)


def test_load_research_approval_rejects_comparison_swapped_after_gate(tmp_path):
    path = _write_approval(tmp_path)
    comparison_path = tmp_path / "comparison.json"
    comparison = json.loads(comparison_path.read_text(encoding="utf-8"))
    comparison["candidate_median_oos_sharpe"] = 9.99
    comparison_path.write_text(json.dumps(comparison), encoding="utf-8")

    with pytest.raises(RuntimeError, match="comparison hash"):
        load_research_approval(path)


def test_load_research_approval_rejects_stress_artifact_mdd_mismatch(tmp_path):
    path = _write_approval(tmp_path)
    artifact = tmp_path / "candidate_20f" / "stress" / "fold1" / "seed0.json"
    persisted = json.loads(artifact.read_text(encoding="utf-8"))
    persisted["stress_mdd"] = -0.99
    artifact.write_text(json.dumps(persisted), encoding="utf-8")

    with pytest.raises(RuntimeError, match="MDD mismatch"):
        load_research_approval(path)


@pytest.mark.parametrize("field", ["segments", "checkpoint_path"])
def test_load_research_approval_rejects_stress_artifact_content_change_with_same_mdd(
        tmp_path, field):
    path = _write_approval(tmp_path)
    artifact = tmp_path / "candidate_20f" / "stress" / "fold1" / "seed0.json"
    persisted = json.loads(artifact.read_text(encoding="utf-8"))
    persisted[field] = ["changed"] if field == "segments" else "changed-checkpoint.zip"
    artifact.write_text(json.dumps(persisted), encoding="utf-8")

    with pytest.raises(RuntimeError, match="stress artifact hash mismatch"):
        load_research_approval(path)


def test_load_research_approval_rejects_missing_complete_factor_contract(tmp_path):
    path = _write_approval(
        tmp_path,
        method={"schema_version": STATE_SCHEMA_VERSION, "method": "ppo"},
    )
    with pytest.raises(ValueError, match="complete factor contract"):
        load_research_approval(path)


def test_load_research_approval_rejects_missing_factor_directions(tmp_path):
    method = {"schema_version": STATE_SCHEMA_VERSION, "method": "ppo", **_factor_contract()}
    method.pop("factor_directions")
    path = _write_approval(tmp_path, method=method)
    with pytest.raises(ValueError, match="factor_directions"):
        load_research_approval(path)


def test_load_research_approval_rejects_conflicting_factor_name_alias(tmp_path):
    method = {
        "schema_version": STATE_SCHEMA_VERSION,
        "method": "ppo",
        **_factor_contract(),
        "factor_names": list(reversed(FACTOR_NAMES)),
    }
    path = _write_approval(tmp_path, method=method)

    with pytest.raises(ValueError, match="selected_factors and factor_names"):
        load_research_approval(path)


def test_publish_requires_selected_factor_bundle(tmp_path):
    path = _write_approval(tmp_path)
    data = json.loads(path.read_text(encoding="utf-8"))
    data["factor_selection_path"] = "missing.json"
    path.write_text(json.dumps(data), encoding="utf-8")

    with pytest.raises((FileNotFoundError, ValueError)):
        load_research_approval(path)


def test_load_research_approval_rejects_selected_factor_hash_mismatch(tmp_path):
    path = _write_approval(tmp_path)
    selection = tmp_path / "candidate_20f" / "selection" / "fold3" / "selected_factors.json"
    payload = json.loads(selection.read_text(encoding="utf-8"))
    payload["notes"] = "tampered"
    selection.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(RuntimeError, match="selected-factor"):
        load_research_approval(path)


def test_load_research_approval_rejects_selected_factor_method_disagreement(tmp_path):
    path = _write_approval(tmp_path)
    selection = tmp_path / "candidate_20f" / "selection" / "fold3" / "selected_factors.json"
    payload = json.loads(selection.read_text(encoding="utf-8"))
    payload["selected_factors"] = list(reversed(payload["selected_factors"]))
    selection.write_text(json.dumps(payload), encoding="utf-8")
    data = json.loads(path.read_text(encoding="utf-8"))
    data["factor_selection_id"] = allocate._file_id(selection)
    path.write_text(json.dumps(data), encoding="utf-8")

    with pytest.raises(RuntimeError, match="selected-factor bundle"):
        load_research_approval(path)


def test_validate_weights_requires_stock_plus_cash_identity():
    df = pd.DataFrame([
        {"trade_date": pd.Timestamp("2024-01-01"), "side": "long", "weight": 0.6},
        {"trade_date": pd.Timestamp("2024-01-01"), "side": "short", "weight": -0.1},
        {"trade_date": pd.Timestamp("2024-01-01"), "side": "cash", "weight": 0.5},
    ])
    assert validate_weights(df, {"long_notional": 1.0, "short_notional_cap": 0.3}) == []

    invalid = df.copy()
    invalid.loc[2, "weight"] = 0.6
    assert any("cash identity" in error for error in validate_weights(
        invalid, {"long_notional": 1.0, "short_notional_cap": 0.3}
    ))


def test_failed_publish_does_not_replace_existing_formal_files(tmp_path):
    production = tmp_path / "production"
    candidate = tmp_path / "candidate"
    production.mkdir()
    candidate.mkdir()
    sentinel = production / "allocations.parquet"
    sentinel.write_bytes(b"sentinel")
    (candidate / "allocations.parquet").write_bytes(b"candidate")
    (candidate / "approval.json").write_text("{}", encoding="utf-8")
    (candidate / "scaler.json").write_text(
        json.dumps({"schema_version": "wrong", "fields": [], "mean": [], "scale": []}),
        encoding="utf-8",
    )
    (candidate / "checkpoint.zip").write_bytes(b"not-a-checkpoint")

    with pytest.raises((RuntimeError, ValueError)):
        atomic_publish(candidate, production)
    assert sentinel.read_bytes() == b"sentinel"


def test_atomic_publish_accepts_current_dynamic_state_schema(tmp_path, monkeypatch):
    candidate = tmp_path / "candidate"
    production = tmp_path / "production"
    candidate.mkdir()
    selected_factors = list(FACTOR_NAMES[:3])
    method = {
        "schema_version": STATE_SCHEMA_VERSION,
        "method": "ppo",
        **_factor_contract(selected_factors),
    }
    _write_approval(candidate, method=method)
    fields = list(state_fields(selected_factors))
    contract = _factor_contract(selected_factors)
    scaler = candidate / "scaler.json"
    scaler.write_text(json.dumps({
        "schema_version": STATE_SCHEMA_VERSION,
        "fields": fields,
        "mean": [0.0] * len(fields),
        "scale": [1.0] * len(fields),
        **contract,
    }), encoding="utf-8")
    checkpoint = candidate / "checkpoint.zip"
    checkpoint.write_bytes(b"checkpoint")
    (candidate / "allocations.parquet").write_bytes(b"allocations")
    approval = json.loads((candidate / "approval.json").read_text())
    (candidate / "checkpoint_metadata.json").write_text(json.dumps({
        "schema_version": STATE_SCHEMA_VERSION,
        "method_id": approval["method_id"],
        "checkpoint_id": allocate._file_id(checkpoint),
        "scaler_id": allocate._file_id(scaler),
        **contract,
    }), encoding="utf-8")
    monkeypatch.setattr(allocate, "run_all", lambda *args, **kwargs: (True, []))

    atomic_publish(candidate, production)

    assert (production / "checkpoint.zip").read_bytes() == b"checkpoint"
    assert load_research_approval(production / "approval.json")["method_path"] == "method.json"
    assert (production / "method.json").exists()
    assert (production / "gates.json").exists()


def test_atomic_publish_rejects_factor_contract_mismatch(tmp_path, monkeypatch):
    candidate = tmp_path / "candidate"
    production = tmp_path / "production"
    candidate.mkdir()
    method = {"schema_version": STATE_SCHEMA_VERSION, "method": "ppo", **_factor_contract()}
    _write_approval(candidate, method=method)
    fields = list(state_fields(FACTOR_NAMES))
    scaler = candidate / "scaler.json"
    scaler.write_text(json.dumps({
        "schema_version": STATE_SCHEMA_VERSION,
        "fields": fields,
        "mean": [0.0] * len(fields),
        "scale": [1.0] * len(fields),
        **_factor_contract(),
    }), encoding="utf-8")
    checkpoint = candidate / "checkpoint.zip"
    checkpoint.write_bytes(b"checkpoint")
    (candidate / "allocations.parquet").write_bytes(b"allocations")
    approval = json.loads((candidate / "approval.json").read_text())
    mismatched = _factor_contract()
    mismatched["factor_catalog_hash"] = "sha256:wrong"
    (candidate / "checkpoint_metadata.json").write_text(json.dumps({
        "schema_version": STATE_SCHEMA_VERSION,
        "method_id": approval["method_id"],
        "checkpoint_id": allocate._file_id(checkpoint),
        "scaler_id": allocate._file_id(scaler),
        **mismatched,
    }), encoding="utf-8")
    monkeypatch.setattr(allocate, "run_all", lambda *args, **kwargs: (True, []))

    with pytest.raises(ValueError, match="factor checkpoint contract mismatch"):
        atomic_publish(candidate, production)
