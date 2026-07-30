from __future__ import annotations

import math
from typing import Callable, Any


def _gate(name: str, actual: Any, threshold: Any, predicate: Callable[[Any, Any], bool]) -> dict:
    passed = actual is not None and threshold is not None and bool(predicate(actual, threshold))
    return {
        "name": name,
        "actual": actual,
        "threshold": threshold,
        "passed": passed,
    }


def evaluate_research_gates(summary: dict) -> dict:
    def get(name: str):
        return summary.get(name)

    total_folds = get("total_folds")
    positive_folds = get("positive_excess_return_folds")
    fold_threshold = (
        math.ceil(2 * total_folds / 3)
        if total_folds is not None and total_folds >= 3
        else None
    )

    baseline_sharpe = get("strongest_baseline_sharpe")
    sharpe_threshold = baseline_sharpe + 0.10 if baseline_sharpe is not None else None

    baseline_mdd = get("strongest_baseline_mdd")
    mdd_threshold = abs(baseline_mdd) + 0.05 if baseline_mdd is not None else None

    gates = [
        _gate("combined_oos_arr", get("combined_oos_arr"), 0, lambda actual, threshold: actual > threshold),
        _gate(
            "median_seed_oos_sharpe",
            get("median_seed_oos_sharpe") if sharpe_threshold is not None else None,
            sharpe_threshold,
            lambda actual, threshold: actual > threshold,
        ),
        _gate(
            "positive_excess_return_folds",
            positive_folds if fold_threshold is not None else None,
            fold_threshold,
            lambda actual, threshold: actual >= threshold,
        ),
        _gate(
            "median_seed_excess_return",
            get("median_seed_excess_return"),
            0,
            lambda actual, threshold: actual > threshold,
        ),
        _gate(
            "oos_mdd",
            abs(get("oos_mdd")) if get("oos_mdd") is not None and mdd_threshold is not None else None,
            mdd_threshold,
            lambda actual, threshold: actual <= threshold,
        ),
        _gate(
            "annualized_turnover",
            get("annualized_turnover"),
            12,
            lambda actual, threshold: actual <= threshold,
        ),
        _gate(
            "cost_2x_oos_sharpe",
            get("cost_2x_oos_sharpe"),
            0,
            lambda actual, threshold: actual > threshold,
        ),
        _gate(
            "no_leakage_tests_passed",
            get("no_leakage_tests_passed"),
            True,
            lambda actual, threshold: actual is True,
        ),
        _gate(
            "state_quality_tests_passed",
            get("state_quality_tests_passed"),
            True,
            lambda actual, threshold: actual is True,
        ),
    ]
    return {"research_ok": all(gate["passed"] for gate in gates), "gates": gates}
