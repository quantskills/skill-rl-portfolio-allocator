from __future__ import annotations

import dataclasses
import json

import numpy as np
import pandas as pd
import pytest

from scripts.factor_selection import (
    SelectionThresholds,
    compute_factor_metrics,
    percentile_scores,
)


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
        },
        "failed": {"passed": False, "mean_ic": 999.0},
        "incomplete": {"passed": True, "mean_ic": 0.5},
    }

    scores = percentile_scores(metrics)
    reordered = percentile_scores(dict(reversed(list(metrics.items()))))

    assert set(scores) == {"a", "b"}
    assert scores == reordered
    assert scores["a"] > scores["b"]
