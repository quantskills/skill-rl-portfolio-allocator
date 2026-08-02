import json

import pytest

from scripts.config import get_config
from scripts.costs import scaled_cost_config
from scripts.research_gates import evaluate_research_gates
from scripts.walk_forward import frozen_method_id, write_approval


def test_scaled_cost_config_is_copy():
    cfg = get_config()
    doubled = scaled_cost_config(cfg, 2.0)
    assert doubled["commission_bps"] == cfg["commission_bps"] * 2
    assert doubled["stamp_tax_bps"] == cfg["stamp_tax_bps"] * 2
    assert doubled["impact_bps"] == cfg["impact_bps"] * 2
    assert doubled["borrow_rate_annual"] == cfg["borrow_rate_annual"] * 2
    assert cfg["commission_bps"] == 3.0


def test_scaled_cost_config_rejects_nonpositive():
    with pytest.raises(ValueError):
        scaled_cost_config(get_config(), 0)


def test_failed_gate_does_not_write_approval(tmp_path):
    method = {"reward_variant": "low", "buffer_variant": "default"}
    report = evaluate_research_gates({})
    assert write_approval(tmp_path, method, report, "smoke", "now") is None
    assert not (tmp_path / "approval.json").exists()


def test_passing_gate_writes_hashed_approval(tmp_path):
    method = {"reward_variant": "low", "buffer_variant": "default"}
    report = {"research_ok": True, "gates": []}
    path = write_approval(tmp_path, method, report, "run1", "now")
    assert path.exists()
    approval = json.loads(path.read_text())
    assert approval["method_id"] == frozen_method_id(method)
    assert json.loads((tmp_path / "gates.json").read_text())["research_ok"] is True
