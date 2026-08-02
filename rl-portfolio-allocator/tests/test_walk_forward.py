import json
import subprocess
import sys

import pytest

from scripts.walk_forward import (
    BUFFER_CONFIGS,
    BUFFER_CANDIDATES,
    REWARD_CANDIDATES,
    SEEDS,
    default_folds,
    run_walk_forward,
    select_candidate_on_validation,
)
from scripts.stress_test import apply_frozen_method, load_frozen_method
from scripts.state import STATE_SCHEMA_VERSION


def test_constants_are_closed_and_buffer_configs_are_exact():
    assert SEEDS == (0, 1, 2, 3, 4)
    assert REWARD_CANDIDATES == ("none", "low", "medium", "legacy_dsr")
    assert BUFFER_CANDIDATES == ("tight", "default", "wide")
    assert BUFFER_CONFIGS == {
        "tight": {"long_entry": 30, "long_exit": 40, "short_entry": 15, "short_exit": 25},
        "default": {"long_entry": 30, "long_exit": 45, "short_entry": 15, "short_exit": 30},
        "wide": {"long_entry": 30, "long_exit": 60, "short_entry": 15, "short_exit": 45},
    }


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
        return {"val_sharpe": score, "method": {"candidate": kwargs["candidate"]}}

    def tester(**kwargs):
        events.append(("test", kwargs["fold"], kwargs["candidate"], kwargs["seed"]))
        return {"test_sharpe": 0.5}

    result = run_walk_forward(
        folds=default_folds(), output_root=tmp_path, smoke=True,
        trainer=trainer, tester=tester, coverage_checker=lambda: None,
    )
    assert result["publishable"] is False
    assert sum(e[0] == "test" for e in events) == 1
    assert all(e[0] == "train" for e in events[:-1])
    first_test = next(i for i, event in enumerate(events) if event[0] == "test")
    assert all(event[0] == "train" for event in events[:first_test])
    assert all(event[2] == "low__tight" for event in events if event[0] == "test")
    assert all(events.count(event) == 1 for event in events if event[0] == "test")
    assert (tmp_path / "smoke" / "summary.json").exists()
    assert json.loads((tmp_path / "smoke" / "summary.json").read_text())["publishable"] is False


def test_tester_receives_saved_validation_result_and_checkpoint(tmp_path):
    received = []

    def trainer(**kwargs):
        checkpoint = kwargs["artifact_dir"] / "best.zip"
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        checkpoint.write_text("fake")
        return {"val_sharpe": 1.0, "checkpoint_path": str(checkpoint), "best": True}

    def tester(**kwargs):
        received.append(kwargs)
        return {"test_sharpe": 0.0}

    run_walk_forward(
        folds=default_folds(), output_root=tmp_path, smoke=True,
        trainer=trainer, tester=tester, coverage_checker=lambda: None,
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
        "schema_version": "state-v1", "training_budget": 777,
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


def test_summary_contains_stress_readable_frozen_method(tmp_path):
    def trainer(**kwargs):
        return {"val_sharpe": 1.0, "checkpoint_path": "/tmp/fake.zip"}

    def tester(**kwargs):
        return {"test_sharpe": 0.0}

    smoke = run_walk_forward(
        folds=default_folds(), output_root=tmp_path, smoke=True,
        trainer=trainer, tester=tester, coverage_checker=lambda: None,
        cfg={"schema_version": "state-v1"}, timesteps=128,
    )
    smoke_method = smoke["frozen_method"]
    assert smoke_method["schema_version"] == "state-v1"
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
        cfg={"schema_version": "state-v1"}, timesteps=256,
    )
    assert set(full["method_by_fold"]) == {"1", "2", "3"}
    assert all(method["training_budget"] == 256 for method in full["method_by_fold"].values())


def test_walk_forward_defaults_emitted_methods_to_current_state_schema(tmp_path):
    def trainer(**kwargs):
        return {"val_sharpe": 1.0, "checkpoint_path": "/tmp/fake.zip"}

    smoke = run_walk_forward(
        folds=default_folds(), output_root=tmp_path, smoke=True,
        trainer=trainer, tester=lambda **kwargs: {"test_sharpe": 0.0},
        coverage_checker=lambda: None,
    )

    assert smoke["frozen_method"]["schema_version"] == STATE_SCHEMA_VERSION


def test_full_runs_use_unique_run_directories_and_write_frozen_test_records(tmp_path):
    calls = []

    def trainer(**kwargs):
        calls.append(("train", kwargs["fold"], kwargs["stage"], kwargs["candidate"], kwargs["seed"]))
        return {"val_sharpe": 1.0 if kwargs["candidate"] == "medium" else 0.0}

    def tester(**kwargs):
        calls.append(("test", kwargs["fold"], kwargs["candidate"], kwargs["seed"]))
        return {"test_sharpe": 1.0}

    first = run_walk_forward(
        folds=default_folds(), output_root=tmp_path, smoke=False,
        trainer=trainer, tester=tester, coverage_checker=lambda: None,
    )
    second = run_walk_forward(
        folds=default_folds(), output_root=tmp_path, smoke=False,
        trainer=trainer, tester=tester, coverage_checker=lambda: None,
    )
    assert first["run_id"] != second["run_id"]
    assert len(list(tmp_path.iterdir())) == 2
    assert len([c for c in calls if c[0] == "test"]) == 30
    for run_id in (first["run_id"], second["run_id"]):
        test_files = list((tmp_path / run_id / "test").rglob("*.json"))
        assert len(test_files) == 15


def test_full_run_accepts_explicit_run_id_and_writes_full_approval(tmp_path):
    def trainer(**kwargs):
        return {"val_sharpe": 1.0, "checkpoint_path": "/tmp/fake.zip"}

    def tester(**kwargs):
        return {
            "test_sharpe": 1.0, "oos_sharpe": 1.0, "oos_arr": 0.2,
            "oos_mdd": -0.1, "excess_return": 0.2,
            "strongest_baseline_sharpe": 0.1, "strongest_baseline_mdd": -0.2,
            "annualized_turnover": 1.0, "cost_2x_oos_sharpe": 0.5,
            "no_leakage_tests_passed": True, "state_quality_tests_passed": True,
        }

    result = run_walk_forward(
        folds=default_folds(), output_root=tmp_path, smoke=False,
        run_id="full-run-001", trainer=trainer, tester=tester,
        coverage_checker=lambda: None, cfg={"schema_version": "state-v1"},
    )
    assert result["run_id"] == "full-run-001"
    approval = json.loads((tmp_path / "full-run-001" / "approval.json").read_text())
    assert approval["run_mode"] == "full"
    assert approval["fold_count"] == 3
    assert approval["seed_count"] == 5
