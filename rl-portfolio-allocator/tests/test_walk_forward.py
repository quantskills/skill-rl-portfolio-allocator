import json

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
    assert (tmp_path / "summary.json").exists()
    assert json.loads((tmp_path / "summary.json").read_text())["publishable"] is False


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
