from __future__ import annotations

import dataclasses
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scripts import factor_selection as factor_selection_module
from scripts.factor_selection import (
    SelectionResult,
    SelectionThresholds,
    compute_factor_metrics,
    percentile_scores,
    select_factors,
    write_selection_artifacts,
)


def _selection_metric(name, family, order, **overrides):
    metric = {
        "name": name,
        "family": family,
        "catalog_order": order,
        "passed": True,
        "mean_ic": 0.02,
        "icir": 0.20,
        "sign_consistency": 0.80,
        "positive_ic_rate": 0.60,
        "net_factor_sharpe": 0.50,
        "doubled_cost_sharpe": 0.20,
        "regime_stability": 0.70,
        "factor_turnover": 0.20,
        "tail_loss": 0.10,
        "direction": 1,
        "coverage": 1.0,
        "symbols": 100,
        "dates": 500,
        "failure_reasons": [],
    }
    metric.update(overrides)
    return metric


def _selection_fixture():
    specs = [("mom_5", "momentum"), ("mom_10", "momentum"), ("mom_20", "momentum")]
    for family in ("reversal", "volatility", "range", "volume", "turnover",
                   "liquidity", "price_volume", "candle", "distribution"):
        specs += [(f"{family}_{i}", family) for i in range(1, 4)]
    metrics = {
        name: _selection_metric(name, family, order)
        for order, (name, family) in enumerate(specs)
    }
    names = list(metrics)
    return_corr = pd.DataFrame(0.0, index=names, columns=names)
    cross_section_corr = pd.DataFrame(0.0, index=names, columns=names)
    for frame in (return_corr, cross_section_corr):
        np.fill_diagonal(frame.values, 1.0)
        frame.loc["mom_5", "mom_10"] = frame.loc["mom_10", "mom_5"] = 0.95
    return metrics, return_corr, cross_section_corr


def _relaxation_fixture():
    metrics, return_corr, cross_section_corr = _selection_fixture()
    names = list(metrics)
    # Sixteen strict candidates (one is a correlated duplicate); later
    # candidates enter at levels 1, 2, 3, 4.
    for name in names[16:]:
        metrics[name].update(mean_ic=0.012, icir=0.08, sign_consistency=0.56,
                             net_factor_sharpe=-0.05)
    metrics[names[16:19][0]].update(mean_ic=0.012, icir=0.08, sign_consistency=0.56,
                                    net_factor_sharpe=-0.05)
    metrics[names[16:19][1]].update(mean_ic=0.012, icir=0.08, sign_consistency=0.56,
                                    net_factor_sharpe=-0.05)
    metrics[names[16:19][2]].update(mean_ic=0.012, icir=0.08, sign_consistency=0.56,
                                    net_factor_sharpe=-0.05)
    metrics[names[19]].update(mean_ic=0.012, icir=0.08, sign_consistency=0.56,
                              net_factor_sharpe=-0.05, family="reversal")
    metrics[names[20]].update(mean_ic=0.004, icir=0.08, sign_consistency=0.56,
                              net_factor_sharpe=-0.05)
    for name in names[21:]:
        metrics[name].update(mean_ic=0.001, icir=0.01, sign_consistency=0.10,
                             positive_ic_rate=0.10, net_factor_sharpe=-1.0)
    return metrics, return_corr, cross_section_corr


def _level4_no_cap_fixture():
    names = [f"synthetic_{index:02d}" for index in range(20)]
    metrics = {
        name: _selection_metric(name, "one_family", index)
        for index, name in enumerate(names)
    }
    return_corr = pd.DataFrame(1.0, index=names, columns=names)
    cross_section_corr = pd.DataFrame(1.0, index=names, columns=names)
    return metrics, return_corr, cross_section_corr


def _hard_valid_21_fixture():
    names = [f"eligible_{index:02d}" for index in range(21)]
    metrics = {
        name: _selection_metric(name, "family", index)
        for index, name in enumerate(names)
    }
    return_corr = pd.DataFrame(0.0, index=names, columns=names)
    cross_section_corr = pd.DataFrame(0.0, index=names, columns=names)
    np.fill_diagonal(return_corr.values, 1.0)
    np.fill_diagonal(cross_section_corr.values, 1.0)
    return metrics, return_corr, cross_section_corr


def test_selector_caps_family_and_rejects_correlated_duplicate():
    metrics, return_corr, cross_section_corr = _selection_fixture()

    result = select_factors(metrics, return_corr, cross_section_corr, target_count=20)

    families = Counter(item["family"] for item in result.selected)
    assert len(result.selected) == 20
    assert max(families.values()) <= result.final_family_cap
    assert not ({"mom_5", "mom_10"} <= {item["name"] for item in result.selected})


def test_selector_uses_catalog_order_for_equal_scores():
    metrics, return_corr, cross_section_corr = _selection_fixture()
    for index, item in enumerate(metrics.values()):
        item["catalog_order"] = 1000 - index
    metrics["mom_5"]["family"] = "tampered_family"
    reordered = dict(reversed(list(metrics.items())))

    first = select_factors(metrics, return_corr, cross_section_corr, target_count=20)
    second = select_factors(reordered, return_corr, cross_section_corr, target_count=20)

    assert [item["name"] for item in first.selected] == [item["name"] for item in second.selected]
    assert first.selected[0]["name"] == "mom_5"
    assert first.selected[0]["family"] == "momentum"


def test_level4_fills_hard_valid_candidates_without_family_or_correlation_caps(tmp_path: Path):
    metrics, return_corr, cross_section_corr = _level4_no_cap_fixture()

    result = select_factors(metrics, return_corr, cross_section_corr, target_count=20)

    assert len(result.selected) == 20
    assert result.final_family_cap == 20
    assert result.final_correlation_ceiling == 1.0
    level4 = result.relaxation_log[-1]
    assert level4["level"] == 4
    assert level4["thresholds"]["family_cap"] is None
    assert level4["thresholds"]["correlation_ceiling"] is None
    assert level4["no_family_or_correlation_cap"] is True

    write_selection_artifacts(
        result,
        metrics,
        {"return_corr": return_corr, "cross_section_corr": cross_section_corr},
        tmp_path,
        fold=3,
        train_range=("2018-01-01", "2020-12-31"),
    )
    selected_payload = json.loads((tmp_path / "selected_factors.json").read_text())
    report = json.loads((tmp_path / "selection_report.json").read_text())
    relaxation = json.loads((tmp_path / "relaxation_log.json").read_text())
    assert selected_payload["level4_no_family_or_correlation_cap"] is True
    assert report["level4_no_family_or_correlation_cap"] is True
    assert relaxation["events"][-1]["no_family_or_correlation_cap"] is True


def test_selector_rejects_missing_correlation_input_or_pair(tmp_path: Path):
    metrics, return_corr, cross_section_corr = _selection_fixture()

    with pytest.raises(ValueError, match="correlation matrix missing pair"):
        select_factors(metrics, None, cross_section_corr, target_count=20)

    missing_pair = return_corr.drop(columns=["reversal_1"])
    with pytest.raises(ValueError, match="correlation matrix missing pair"):
        select_factors(metrics, missing_pair, cross_section_corr, target_count=20)

    non_finite_pair = return_corr.copy()
    non_finite_pair.loc["mom_5", "reversal_1"] = None
    with pytest.raises(ValueError, match="correlation out of range"):
        select_factors(metrics, non_finite_pair, cross_section_corr, target_count=20)

    result = select_factors(metrics, return_corr, cross_section_corr, target_count=20)
    with pytest.raises(ValueError, match="correlation matrix missing pair"):
        write_selection_artifacts(
            result,
            metrics,
            {"return_corr": missing_pair, "cross_section_corr": cross_section_corr},
            tmp_path / "missing-correlation-artifacts",
            fold=3,
            train_range=("2018-01-01", "2020-12-31"),
        )

    level4_metrics, level4_return_corr, level4_cross_section_corr = _level4_no_cap_fixture()
    level4_missing_pair = level4_return_corr.drop(columns=["synthetic_19"])
    with pytest.raises(ValueError, match="correlation matrix missing pair"):
        select_factors(
            level4_metrics,
            level4_missing_pair,
            level4_cross_section_corr,
            target_count=20,
        )


def test_selector_relaxes_in_documented_order():
    metrics, return_corr, cross_section_corr = _relaxation_fixture()

    result = select_factors(metrics, return_corr, cross_section_corr, target_count=20)

    assert len(result.selected) == 20
    assert [event["level"] for event in result.relaxation_log] == [0, 1, 2, 3, 4]


def test_strict_level0_rejects_zero_net_factor_sharpe():
    metrics, return_corr, cross_section_corr = _selection_fixture()
    for item in metrics.values():
        item["net_factor_sharpe"] = 0.0

    result = select_factors(metrics, return_corr, cross_section_corr, target_count=20)

    assert result.relaxation_log[0]["admitted"] == []
    assert all(item["selection_level"] >= 1 for item in result.selected)


def test_selector_fails_when_non_relaxable_candidates_are_insufficient():
    metrics, return_corr, cross_section_corr = _selection_fixture()
    for item in metrics.values():
        item["coverage"] = 0.5
        item["passed"] = False
        item["failure_reasons"] = ["coverage"]

    with pytest.raises(ValueError, match="fewer than 20 hard-valid factors"):
        select_factors(metrics, return_corr, cross_section_corr, target_count=20)


@pytest.mark.parametrize(
    "missing_field",
    [
        "passed",
        "failure_reasons",
        "mean_ic",
        "icir",
        "sign_consistency",
        "positive_ic_rate",
        "net_factor_sharpe",
    ],
)
def test_selector_requires_non_relaxable_input_and_gate_fields(missing_field):
    metrics, return_corr, cross_section_corr = _selection_fixture()
    for item in metrics.values():
        item.pop(missing_field)
        item["score"] = 1.0

    with pytest.raises(ValueError, match="fewer than 20 hard-valid factors"):
        select_factors(metrics, return_corr, cross_section_corr, target_count=20)


@pytest.mark.parametrize("missing_field", ["passed", "failure_reasons", "coverage", "symbols", "dates"])
def test_percentile_scores_fail_closed_for_missing_validity_fields(missing_field):
    good = _selection_metric("good", "family", 0)
    bad = _selection_metric("bad", "family", 1, mean_ic=999.0)
    bad.pop(missing_field)

    scores = percentile_scores({"good": good, "bad": bad})

    assert set(scores) == {"good"}


def test_invalid_percentile_candidate_cannot_pollute_valid_ranking():
    good = _selection_metric("good", "family", 0, mean_ic=0.02)
    other = _selection_metric("other", "family", 1, mean_ic=0.01)
    invalid = _selection_metric("invalid", "family", 2, mean_ic=999.0)
    invalid.pop("passed")
    invalid.pop("failure_reasons")
    invalid.pop("coverage")
    invalid.pop("symbols")
    invalid.pop("dates")

    valid_scores = percentile_scores({"good": good, "other": other})
    polluted_scores = percentile_scores({"good": good, "other": other, "invalid": invalid})

    assert polluted_scores == valid_scores


@pytest.mark.parametrize("missing_field", ["coverage", "symbols", "dates"])
def test_selector_rejects_missing_non_relaxable_hard_gate(missing_field):
    metrics, return_corr, cross_section_corr = _selection_fixture()
    for item in metrics.values():
        item.pop(missing_field)

    with pytest.raises(ValueError, match="fewer than 20 hard-valid factors"):
        select_factors(metrics, return_corr, cross_section_corr, target_count=20)


def test_selector_prevalidates_correlation_matrix_for_all_eligible_candidates():
    metrics, return_corr, cross_section_corr = _hard_valid_21_fixture()
    missing_name = "eligible_20"
    incomplete_return_corr = return_corr.drop(index=missing_name, columns=missing_name)

    with pytest.raises(ValueError, match="correlation matrix missing pair"):
        select_factors(
            metrics,
            incomplete_return_corr,
            cross_section_corr,
            target_count=20,
        )


def test_selector_rejects_correlation_out_of_range():
    metrics, return_corr, cross_section_corr = _selection_fixture()
    return_corr.loc["mom_5", "reversal_1"] = 1.01

    with pytest.raises(ValueError, match="correlation out of range"):
        select_factors(metrics, return_corr, cross_section_corr, target_count=20)


def test_selector_requires_diagonal_entries_for_single_candidate():
    metrics = {"synthetic_only": _selection_metric("synthetic_only", "family", 0)}
    empty_return_corr = pd.DataFrame()
    empty_cross_section_corr = pd.DataFrame()

    with pytest.raises(ValueError, match="correlation matrix missing pair"):
        select_factors(metrics, empty_return_corr, empty_cross_section_corr, target_count=1)


def test_selection_artifacts_preserve_order_hash_and_are_atomic(tmp_path: Path):
    metrics, return_corr, cross_section_corr = _selection_fixture()
    result = select_factors(metrics, return_corr, cross_section_corr, target_count=20)

    write_selection_artifacts(
        result,
        metrics,
        {"return_corr": return_corr, "cross_section_corr": cross_section_corr},
        tmp_path,
        fold=3,
        train_range=("2018-01-01", "2020-12-31"),
    )

    expected = {
        "candidates.parquet",
        "selected_factors.json",
        "factor_metrics.json",
        "correlation_matrix.parquet",
        "relaxation_log.json",
        "selection_report.json",
    }
    assert {path.name for path in tmp_path.iterdir()} == expected
    selected_payload = json.loads((tmp_path / "selected_factors.json").read_text())
    assert [item["name"] for item in selected_payload["selected_factors"]] == [
        item["name"] for item in result.selected
    ]
    assert selected_payload["catalog_hash"] == result.catalog_hash
    json.dumps(json.loads((tmp_path / "selection_report.json").read_text()), allow_nan=False)

    bad_metrics = dict(metrics)
    bad_metrics["mom_5"] = dict(metrics["mom_5"], mean_ic=np.inf)
    failed_dir = tmp_path / "failed"
    with pytest.raises(ValueError, match="finite"):
        write_selection_artifacts(
            result,
            bad_metrics,
            {"return_corr": return_corr, "cross_section_corr": cross_section_corr},
            failed_dir,
            fold=3,
            train_range=("2018-01-01", "2020-12-31"),
        )
    assert not failed_dir.exists()


def test_factor_metrics_artifact_is_canonically_sorted(tmp_path: Path):
    metrics, return_corr, cross_section_corr = _selection_fixture()
    for item in metrics.values():
        item["nested"] = {"z": 1, "a": {"d": 4, "b": 2}}
    result = select_factors(metrics, return_corr, cross_section_corr, target_count=20)
    reversed_metrics = {}
    for name, item in reversed(list(metrics.items())):
        reordered = dict(item)
        reordered["nested"] = {"a": {"b": 2, "d": 4}, "z": 1}
        reversed_metrics[name] = reordered
    first_dir = tmp_path / "first"
    second_dir = tmp_path / "second"
    correlations = {"return_corr": return_corr, "cross_section_corr": cross_section_corr}

    write_selection_artifacts(result, metrics, correlations, first_dir, 3, ("2018-01-01", "2020-12-31"))
    write_selection_artifacts(result, reversed_metrics, correlations, second_dir, 3, ("2018-01-01", "2020-12-31"))

    assert (first_dir / "factor_metrics.json").read_bytes() == (
        second_dir / "factor_metrics.json"
    ).read_bytes()
    payload = json.loads((first_dir / "factor_metrics.json").read_text())
    assert list(payload["metrics"]) == sorted(metrics)


def test_selection_artifacts_reject_forged_catalog_hash(tmp_path: Path):
    metrics, return_corr, cross_section_corr = _selection_fixture()
    result = select_factors(metrics, return_corr, cross_section_corr, target_count=20)
    forged = dataclasses.replace(result, catalog_hash="sha256:forged")
    output_dir = tmp_path / "forged"

    with pytest.raises(ValueError, match="catalog hash mismatch"):
        write_selection_artifacts(
            forged,
            metrics,
            {"return_corr": return_corr, "cross_section_corr": cross_section_corr},
            output_dir,
            fold=3,
            train_range=("2018-01-01", "2020-12-31"),
        )

    assert not output_dir.exists()


def test_selection_artifacts_ignore_old_backup_cleanup_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    metrics, return_corr, cross_section_corr = _selection_fixture()
    output_dir = tmp_path / "selection"
    correlations = {"return_corr": return_corr, "cross_section_corr": cross_section_corr}
    result = select_factors(metrics, return_corr, cross_section_corr, target_count=20)
    write_selection_artifacts(result, metrics, correlations, output_dir, 3, ("2018-01-01", "2020-12-31"))

    previous = tmp_path / f".{output_dir.name}.previous"
    original_rmtree = factor_selection_module.shutil.rmtree

    def fail_previous_cleanup(path, *args, **kwargs):
        if Path(path) == previous:
            raise OSError("injected old backup cleanup failure")
        return original_rmtree(path, *args, **kwargs)

    monkeypatch.setattr(factor_selection_module.shutil, "rmtree", fail_previous_cleanup)
    write_selection_artifacts(result, metrics, correlations, output_dir, 3, ("2018-01-01", "2020-12-31"))

    assert output_dir.is_dir()
    assert (output_dir / "selected_factors.json").is_file()
    stage_dirs = [
        path for path in tmp_path.iterdir()
        if path.name.startswith(f".{output_dir.name}.")
        and path.name != previous.name
    ]
    assert stage_dirs == []


def test_selection_artifacts_ignore_existing_backup_cleanup_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    metrics, return_corr, cross_section_corr = _selection_fixture()
    output_dir = tmp_path / "selection"
    correlations = {"return_corr": return_corr, "cross_section_corr": cross_section_corr}
    result = select_factors(metrics, return_corr, cross_section_corr, target_count=20)
    write_selection_artifacts(result, metrics, correlations, output_dir, 3, ("2018-01-01", "2020-12-31"))

    previous = tmp_path / f".{output_dir.name}.previous"
    previous.mkdir()
    (previous / "old.txt").write_text("old")
    original_rmtree = factor_selection_module.shutil.rmtree

    def fail_previous_cleanup(path, *args, **kwargs):
        if Path(path) == previous:
            raise OSError("injected existing backup cleanup failure")
        return original_rmtree(path, *args, **kwargs)

    monkeypatch.setattr(factor_selection_module.shutil, "rmtree", fail_previous_cleanup)
    write_selection_artifacts(result, metrics, correlations, output_dir, 3, ("2018-01-01", "2020-12-31"))

    assert output_dir.is_dir()
    assert (output_dir / "selected_factors.json").is_file()
    stage_dirs = [
        path for path in tmp_path.iterdir()
        if path.name.startswith(f".{output_dir.name}.")
        and path.name not in {previous.name}
    ]
    assert stage_dirs == []


def _cfg(**overrides):
    cfg = {
        "commission_bps": 3.0,
        "stamp_tax_bps": 10.0,
        "impact_bps": 5.0,
        "borrow_rate_annual": 0.08,
        "trading_days_per_year": 252,
        "selection_thresholds": SelectionThresholds(
            min_coverage=0.98, min_symbols=100, min_dates=500
        ),
    }
    cfg.update(overrides)
    return cfg


@pytest.fixture
def candidate_panel():
    dates = pd.date_range("2018-01-01", periods=520, freq="B")
    rows = []
    symbols = [f"S{i:03d}" for i in range(101)]
    for date_index, date in enumerate(dates):
        for symbol_index, symbol in enumerate(symbols):
            signal = np.sin(date_index / 13.0) + (symbol_index - 50) / 100.0
            previous_signal = np.sin((date_index - 1) / 13.0) + (symbol_index - 50) / 100.0
            rows.append(
                {
                    "trade_date": date,
                    "symbol": symbol,
                    "ret_1d": 0.002 * previous_signal,
                    "is_suspended": False,
                    "f1": signal,
                    "negative_predictor": -signal,
                }
            )
    return pd.DataFrame(rows)


def test_selection_thresholds_are_frozen_with_required_defaults():
    thresholds = SelectionThresholds()

    assert thresholds == SelectionThresholds(0.98, 100, 500)
    with pytest.raises(dataclasses.FrozenInstanceError):
        thresholds.min_dates = 1


def test_metrics_use_only_inclusive_training_dates(candidate_panel):
    metrics = compute_factor_metrics(
        candidate_panel, ["f1"], "2018-01-01", "2019-12-31", _cfg()
    )
    mutated = candidate_panel.copy()
    future = mutated["trade_date"] > pd.Timestamp("2019-12-31")
    mutated.loc[future, ["f1", "ret_1d"]] = -1000000.0
    changed = compute_factor_metrics(
        mutated, ["f1"], "2018-01-01", "2019-12-31", _cfg()
    )

    assert metrics == changed


def test_negative_training_ic_orients_factor_positive(candidate_panel):
    metrics = compute_factor_metrics(
        candidate_panel, ["negative_predictor"], "2018-01-01", "2019-12-31", _cfg()
    )
    result = metrics["negative_predictor"]

    assert result["direction"] == -1
    assert result["oriented_mean_ic"] > 0


def test_negative_direction_ranks_effective_signal_and_keeps_net_below_gross(candidate_panel):
    cfg = _cfg()
    positive = compute_factor_metrics(
        candidate_panel, ["f1"], "2018-01-01", "2019-12-31", cfg
    )["f1"]
    negative = compute_factor_metrics(
        candidate_panel, ["negative_predictor"], "2018-01-01", "2019-12-31", cfg
    )["negative_predictor"]

    assert negative["direction"] == -1
    assert negative["weekly_target_weights"] == positive["weekly_target_weights"]
    assert all(
        net <= gross + 1e-12
        for gross, net in zip(negative["weekly_gross_returns"], negative["weekly_net_returns"])
    )


def test_normal_training_candidate_passes_and_produces_percentile_scores(candidate_panel):
    metrics = compute_factor_metrics(
        candidate_panel,
        ["f1", "negative_predictor"],
        "2018-01-01",
        "2019-12-31",
        _cfg(),
    )

    assert metrics["f1"]["passed"] is True
    assert metrics["negative_predictor"]["passed"] is True
    assert metrics["f1"]["failure_reasons"] == []
    assert metrics["negative_predictor"]["failure_reasons"] == []
    assert set(percentile_scores(metrics)) == {"f1", "negative_predictor"}


def test_normal_short_trading_week_does_not_fail_short_week_gate(candidate_panel):
    shortened = candidate_panel.loc[
        candidate_panel["trade_date"] != pd.Timestamp("2018-01-10")
    ].copy()
    result = compute_factor_metrics(
        shortened, ["f1"], "2018-01-01", "2019-12-31", _cfg()
    )["f1"]

    assert result["passed"] is True
    assert "short_week" not in result["failure_reasons"]


def test_factor_metrics_are_directly_json_serializable(candidate_panel):
    metrics = compute_factor_metrics(
        candidate_panel, ["f1", "negative_predictor"], "2018-01-01", "2019-12-31", _cfg()
    )

    json.dumps(metrics)


def test_ic_uses_one_previous_trading_date_per_symbol(candidate_panel):
    metrics = compute_factor_metrics(
        candidate_panel, ["f1"], "2018-01-01", "2019-12-31", _cfg()
    )

    assert metrics["f1"]["mean_ic"] > 0.95


def test_coverage_symbol_and_date_gates_are_recorded(candidate_panel):
    sparse = candidate_panel.copy()
    sparse.loc[sparse["symbol"] == "S000", "f1"] = np.nan
    sparse.loc[sparse["trade_date"] < pd.Timestamp("2019-01-01"), "f1"] = np.nan
    thresholds = SelectionThresholds(min_coverage=0.98, min_symbols=101, min_dates=500)
    metrics = compute_factor_metrics(
        sparse, ["f1"], "2018-01-01", "2019-12-31", _cfg(selection_thresholds=thresholds)
    )
    result = metrics["f1"]

    assert result["coverage"] < 0.98
    assert result["symbols"] < 101
    assert result["dates"] < 500
    assert {"coverage", "symbols", "dates"} <= set(result["failure_reasons"])
    assert result["passed"] is False


def test_metrics_are_deterministic_under_row_order_change(candidate_panel):
    expected = compute_factor_metrics(
        candidate_panel, ["f1"], "2018-01-01", "2019-12-31", _cfg()
    )
    shuffled = candidate_panel.sample(frac=1.0, random_state=17).reset_index(drop=True)
    actual = compute_factor_metrics(
        shuffled, ["f1"], "2018-01-01", "2019-12-31", _cfg()
    )

    assert expected == actual


def test_weekly_returns_expose_gross_net_turnover_and_doubled_costs(candidate_panel):
    result = compute_factor_metrics(
        candidate_panel, ["f1"], "2018-01-01", "2019-12-31", _cfg()
    )["f1"]

    assert result["weekly_gross_returns"]
    assert len(result["weekly_gross_returns"]) == len(result["weekly_net_returns"])
    assert len(result["weekly_net_returns"]) == len(result["weekly_doubled_cost_returns"])
    assert result["weekly_turnover"]
    assert result["costs"]["total"] >= 0.0
    assert result["doubled_costs"]["total"] == pytest.approx(2.0 * result["costs"]["total"])
    assert np.isfinite(result["net_factor_sharpe"])
    assert np.isfinite(result["doubled_cost_sharpe"])


def test_weekly_portfolio_has_one_decision_per_week_and_full_initial_turnover(candidate_panel):
    small = candidate_panel.loc[candidate_panel["symbol"] < "S010"].copy()
    cfg = _cfg(selection_thresholds=SelectionThresholds(min_coverage=0.0, min_symbols=1, min_dates=1))
    result = compute_factor_metrics(small, ["f1"], "2018-01-01", "2018-02-16", cfg)["f1"]

    assert len(result["weekly_decision_dates"]) == len(set(result["weekly_decision_dates"]))
    assert len(result["weekly_decision_dates"]) == len(result["weekly_turnover"])
    assert result["weekly_turnover"][0] == pytest.approx(1.0)
    assert all(count == 10 for count in result["weekly_signal_symbol_counts"])
    assert all(value == pytest.approx(0.5) for value in result["weekly_long_notionals"])
    assert all(value == pytest.approx(0.5) for value in result["weekly_short_notionals"])


def test_weekly_settlement_starts_on_next_trading_date_and_records_holding_costs(candidate_panel):
    small = candidate_panel.loc[candidate_panel["symbol"] < "S010"].copy()
    cfg = _cfg(selection_thresholds=SelectionThresholds(min_coverage=0.0, min_symbols=1, min_dates=1))
    result = compute_factor_metrics(small, ["f1"], "2018-01-01", "2018-02-16", cfg)["f1"]

    next_trading_date = (
        pd.Timestamp(result["weekly_decision_dates"][0]) + pd.offsets.BDay(1)
    ).date().isoformat()
    assert result["weekly_settlement_dates"][0][0] == next_trading_date
    for index, settlement_dates in enumerate(result["weekly_settlement_dates"][:-1]):
        assert result["weekly_decision_dates"][index + 1] in settlement_dates
    assert len(result["weekly_settlement_dates"][0]) >= 1
    assert result["weekly_costs"][0]["borrow"] > 0.0
    assert len(result["weekly_doubled_cost_returns"]) == len(result["weekly_net_returns"])


def test_every_scored_date_keeps_eligible_count_and_weak_date_fails_gate(candidate_panel):
    sparse = candidate_panel.copy()
    date = sparse["trade_date"].iloc[200]
    sparse.loc[(sparse["trade_date"] == date) & (sparse["symbol"] < "S020"), "f1"] = np.nan
    thresholds = SelectionThresholds(min_coverage=0.0, min_symbols=100, min_dates=1)
    result = compute_factor_metrics(
        sparse, ["f1"], "2018-01-01", "2019-12-31", _cfg(selection_thresholds=thresholds)
    )["f1"]

    counts = dict(zip(result["scored_dates"], result["eligible_symbols_by_date"]))
    scored_date = (pd.Timestamp(date) + pd.offsets.BDay(1)).date().isoformat()
    assert counts[scored_date] == 81
    assert result["symbols"] == 81
    assert "insufficient_symbols_on_scored_date" in result["failure_reasons"]


def test_regime_stability_reports_training_only_regime_scores(candidate_panel):
    result = compute_factor_metrics(
        candidate_panel, ["f1"], "2018-01-01", "2019-12-31", _cfg()
    )["f1"]

    assert set(result["regime_scores"]) == {"bull", "bear", "high_vol", "low_vol"}
    assert set(result["regime_observations"]) == set(result["regime_scores"])
    assert result["regime_stability"] == pytest.approx(
        np.mean([value for value in result["regime_scores"].values() if value is not None])
    )
    assert all(value is None or 0.0 <= value <= 1.0 for value in result["regime_scores"].values())


def test_missing_or_invalid_weekly_data_is_a_failure_not_silent_zero(candidate_panel):
    short = candidate_panel.loc[candidate_panel["trade_date"] == pd.Timestamp("2018-01-01")].copy()
    cfg = _cfg(selection_thresholds=SelectionThresholds(min_coverage=0.0, min_symbols=1, min_dates=1))
    result = compute_factor_metrics(short, ["f1"], "2018-01-01", "2018-01-01", cfg)["f1"]

    assert result["weekly_net_returns"] == []
    assert "weekly_data" in result["failure_reasons"]
    assert "insufficient_weekly_observations" in result["failure_reasons"]
    assert result["weekly_failure_reasons"]


def test_invalid_week_is_recorded_and_not_replaced_by_zero(candidate_panel):
    small = candidate_panel.loc[candidate_panel["symbol"] < "S010"].copy()
    invalid_date = pd.Timestamp("2018-01-09")
    small.loc[(small["trade_date"] == invalid_date) & (small["symbol"] == "S000"), "ret_1d"] = np.inf
    cfg = _cfg(selection_thresholds=SelectionThresholds(min_coverage=0.0, min_symbols=1, min_dates=1))
    result = compute_factor_metrics(small, ["f1"], "2018-01-01", "2018-02-16", cfg)["f1"]

    assert "invalid_week" in result["failure_reasons"]
    assert any(item["reason"] == "invalid_week" for item in result["weekly_failure_reasons"])


def test_invalid_future_rows_do_not_change_training_metrics(candidate_panel):
    expected = compute_factor_metrics(
        candidate_panel, ["f1"], "2018-01-01", "2019-12-31", _cfg()
    )
    future_bad = pd.DataFrame(
        [
            {
                "trade_date": "not-a-date",
                "symbol": "",
                "ret_1d": "bad-return",
                "is_suspended": False,
                "f1": "bad-factor",
                "negative_predictor": np.inf,
            },
            {
                "trade_date": "2025-01-02",
                "symbol": "S000",
                "ret_1d": np.inf,
                "is_suspended": False,
                "f1": "bad-factor",
                "negative_predictor": np.nan,
            },
            {
                "trade_date": "2025-01-02",
                "symbol": "S000",
                "ret_1d": -np.inf,
                "is_suspended": False,
                "f1": "another-bad-factor",
                "negative_predictor": np.nan,
            },
        ]
    )
    changed = compute_factor_metrics(
        pd.concat([candidate_panel, future_bad], ignore_index=True),
        ["f1"],
        "2018-01-01",
        "2019-12-31",
        _cfg(),
    )

    assert changed == expected


def test_inf_and_nan_are_filtered_without_runtime_warnings(candidate_panel):
    noisy = candidate_panel.copy()
    noisy.loc[noisy.index[0], "f1"] = np.inf
    noisy.loc[noisy.index[1], "f1"] = -np.inf
    noisy.loc[noisy.index[2], "ret_1d"] = np.nan

    result = compute_factor_metrics(
        noisy, ["f1"], "2018-01-01", "2019-12-31", _cfg()
    )["f1"]

    assert result["coverage"] < 1.0
    assert all(np.isfinite(value) for value in result["weekly_net_returns"])
    assert "non_finite_values" in result["failure_reasons"] or result["passed"]


def test_schema_factor_and_key_validation(candidate_panel):
    with pytest.raises(ValueError, match="unknown factor"):
        compute_factor_metrics(candidate_panel, ["missing"], "2018-01-01", "2019-12-31", _cfg())

    duplicate = pd.concat([candidate_panel, candidate_panel.iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match="duplicate"):
        compute_factor_metrics(duplicate, ["f1"], "2018-01-01", "2019-12-31", _cfg())

    invalid_symbol = candidate_panel.copy()
    invalid_symbol.loc[0, "symbol"] = ""
    with pytest.raises(ValueError, match="symbol"):
        compute_factor_metrics(invalid_symbol, ["f1"], "2018-01-01", "2019-12-31", _cfg())


def test_percentile_scores_are_deterministic_and_exclude_failed_or_incomplete_candidates():
    metrics = {
        "a": {
            "passed": True,
            "mean_ic": 0.02,
            "icir": 0.20,
            "sign_consistency": 0.8,
            "net_factor_sharpe": 1.0,
            "doubled_cost_sharpe": 0.5,
            "regime_stability": 0.7,
            "factor_turnover": 0.2,
            "tail_loss": -0.1,
            "positive_ic_rate": 0.6,
            "coverage": 1.0,
            "symbols": 100,
            "dates": 500,
            "failure_reasons": [],
        },
        "b": {
            "passed": True,
            "mean_ic": 0.01,
            "icir": 0.10,
            "sign_consistency": 0.6,
            "net_factor_sharpe": 0.5,
            "doubled_cost_sharpe": 0.2,
            "regime_stability": 0.4,
            "factor_turnover": 0.4,
            "tail_loss": -0.2,
            "positive_ic_rate": 0.6,
            "coverage": 1.0,
            "symbols": 100,
            "dates": 500,
            "failure_reasons": [],
        },
        "failed": {"passed": False, "mean_ic": 999.0},
        "incomplete": {"passed": True, "mean_ic": 0.5},
    }

    scores = percentile_scores(metrics)
    reordered = percentile_scores(dict(reversed(list(metrics.items()))))

    assert set(scores) == {"a", "b"}
    assert scores == reordered
    assert scores["a"] > scores["b"]


def test_warmup_nan_does_not_fail_non_finite_but_inf_does(candidate_panel):
    panel = candidate_panel.copy()
    # Rolling warmup: first rows per symbol are NaN, like real data.
    panel.loc[panel.groupby("symbol").head(3).index, "f1"] = np.nan
    metrics = compute_factor_metrics(panel, ["f1"], "2018-01-01", "2019-12-31", _cfg())
    assert "non_finite_values" not in metrics["f1"]["failure_reasons"]

    with_inf = candidate_panel.copy()
    with_inf.loc[with_inf.index[100], "f1"] = np.inf
    metrics_inf = compute_factor_metrics(with_inf, ["f1"], "2018-01-01", "2019-12-31", _cfg())
    assert "non_finite_values" in metrics_inf["f1"]["failure_reasons"]


def test_missing_return_for_never_held_symbol_does_not_invalidate_week(candidate_panel):
    # S050 has the exact median signal every date, so the 50/50 long-short
    # never holds it; its missing ret_1d must not invalidate the week.
    panel = candidate_panel.copy()
    mid = panel["symbol"] == "S050"
    drop_dates = panel.loc[mid, "trade_date"].iloc[40:80]
    mask = mid & panel["trade_date"].isin(drop_dates)
    panel.loc[mask, "ret_1d"] = np.nan
    metrics = compute_factor_metrics(panel, ["f1"], "2018-01-01", "2019-12-31", _cfg())
    reasons = metrics["f1"]["failure_reasons"]
    assert "invalid_week" not in reasons
    assert "no_valid_weekly_data" not in reasons
    assert "weekly_data" not in reasons
    assert metrics["f1"]["weekly_net_returns"]
