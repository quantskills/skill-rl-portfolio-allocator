"""Production approval and publish gates are fail-closed and atomic."""
import json

import numpy as np
import pandas as pd
import pytest

from scripts.allocate import atomic_publish, load_research_approval
from scripts.validate import validate_weights


def _write_approval(root, *, research_ok=True, method=None, gates_ok=True):
    method = method or {"schema_version": "state-v1", "method": "ppo"}
    (root / "method.json").write_text(json.dumps(method), encoding="utf-8")
    (root / "gates.json").write_text(
        json.dumps({"research_ok": gates_ok}), encoding="utf-8"
    )
    approval = {
        "research_ok": research_ok,
        "schema_version": method["schema_version"],
        "method_id": __import__("scripts.walk_forward", fromlist=["frozen_method_id"])
        .frozen_method_id(method),
        "method_path": "method.json",
        "gates_path": "gates.json",
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


def test_load_research_approval_accepts_valid_referenced_artifacts(tmp_path):
    path = _write_approval(tmp_path)
    approval = load_research_approval(path)
    assert approval["method_id"].startswith("sha256:")
    assert approval["method_path"] == "method.json"


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
