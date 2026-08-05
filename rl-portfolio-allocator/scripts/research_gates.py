from __future__ import annotations

import math
from typing import Callable, Any

from scripts.config import TRAIN_SEEDS


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


def evaluate_candidate_gates(summary: dict) -> dict:
    """Fail closed unless the selected-factor branch beats its paired control."""
    evidence_failures = _paired_evidence_failures(summary.get("paired_evidence"))

    if evidence_failures:
        gates = [
            {
                "name": name,
                "actual": None,
                "threshold": threshold,
                "passed": False,
                "failure_reason": "; ".join(evidence_failures),
            }
            for name, threshold in (
                ("complete_paired_evidence", True),
                ("median_oos_sharpe_gain", 0.10),
                ("positive_excess_folds", 2),
                ("candidate_cost_2x_oos_sharpe", 0),
                ("candidate_annualized_turnover", 12),
                ("candidate_stress_mdd_excess", 0.05),
                ("candidate_stress_calmar_excess", 0),
                ("candidate_stress_long_exposure_util", 0.5),
            )
        ]
        return {"research_ok": False, "gates": gates, "failure_reasons": evidence_failures}

    def get(name: str):
        return summary.get(name)

    candidate_sharpe = get("candidate_median_oos_sharpe")
    control_sharpe = get("control_median_oos_sharpe")
    sharpe_gain = (
        candidate_sharpe - control_sharpe
        if candidate_sharpe is not None and control_sharpe is not None
        else None
    )
    candidate_mdd = get("candidate_stress_mdd")
    control_mdd = get("control_stress_mdd")
    stress_mdd_excess = (
        abs(candidate_mdd) - abs(control_mdd)
        if candidate_mdd is not None and control_mdd is not None
        else None
    )
    gates = [
        _gate("median_oos_sharpe_gain", sharpe_gain, 0.10, lambda actual, threshold: actual >= threshold),
        _gate("positive_excess_folds", get("positive_excess_folds"), 2, lambda actual, threshold: actual >= threshold),
        _gate("candidate_cost_2x_oos_sharpe", get("candidate_cost_2x_oos_sharpe"), 0, lambda actual, threshold: actual > threshold),
        _gate("candidate_annualized_turnover", get("candidate_annualized_turnover"), 12, lambda actual, threshold: actual <= threshold),
        _gate("candidate_stress_mdd_excess", stress_mdd_excess, 0.05, lambda actual, threshold: actual <= threshold),
        _gate("candidate_stress_calmar_excess", get("candidate_stress_calmar_excess"), 0, lambda actual, threshold: actual >= threshold),
        _gate("candidate_stress_long_exposure_util", get("candidate_stress_long_exposure_util"), 0.5, lambda actual, threshold: actual >= threshold),
        _gate("complete_paired_evidence", True, True, lambda actual, threshold: actual is True),
    ]
    return {"research_ok": all(gate["passed"] for gate in gates), "gates": gates, "failure_reasons": []}


def _paired_evidence_failures(evidence: Any) -> list[str]:
    """Validate that exactly the full 3-fold x 5-seed paired run was supplied."""
    if not isinstance(evidence, dict):
        return ["paired evidence is missing"]

    required_metrics = (
        "oos_sharpe",
        "cost_2x_oos_sharpe",
        "annualized_turnover",
        "stress_mdd",
        "stress_calmar_excess",
        "stress_long_exposure_util",
    )
    expected_branches = ("candidate_20f", "control_6f")
    branch_keys = {}
    failures = []
    for branch in expected_branches:
        branch_evidence = evidence.get(branch)
        rows = branch_evidence.get("rows") if isinstance(branch_evidence, dict) else None
        if not isinstance(rows, list):
            failures.append(f"{branch} rows are missing")
            continue
        expected_rows = 3 * len(TRAIN_SEEDS)
        if len(rows) != expected_rows:
            failures.append(f"{branch} has {len(rows)} rows; expected {expected_rows}")

        keys = []
        stress_artifacts = []
        for index, row in enumerate(rows):
            prefix = f"{branch} row {index}"
            if not isinstance(row, dict):
                failures.append(f"{prefix} is not a metric row")
                continue
            if row.get("branch") != branch:
                failures.append(f"{prefix} has branch {row.get('branch')!r}; expected {branch!r}")
            if "fold" not in row or "seed" not in row:
                failures.append(f"{prefix} is missing fold or seed")
            elif (
                not isinstance(row["fold"], int) or isinstance(row["fold"], bool)
                or not isinstance(row["seed"], int) or isinstance(row["seed"], bool)
            ):
                failures.append(f"{prefix} has invalid fold or seed")
            else:
                keys.append((row["fold"], row["seed"]))
            for metric in required_metrics:
                value = row.get(metric)
                if value is None:
                    failures.append(f"{prefix} is missing {metric}")
                    continue
                try:
                    finite = math.isfinite(float(value))
                except (TypeError, ValueError):
                    finite = False
                if not finite:
                    failures.append(f"{prefix} has non-finite {metric}")
            artifact_path = row.get("stress_artifact_path")
            if not isinstance(artifact_path, str) or not artifact_path.strip():
                failures.append(f"{prefix} is missing persisted stress artifact")
            else:
                stress_artifacts.append(artifact_path)
            artifact_hash = row.get("stress_artifact_sha256")
            if not isinstance(artifact_hash, str) or not artifact_hash.startswith("sha256:"):
                failures.append(f"{prefix} is missing stress artifact hash")
        if len(keys) != len(set(keys)):
            failures.append(f"{branch} has duplicate fold/seed pairs")
        if len(stress_artifacts) != len(set(stress_artifacts)):
            failures.append(f"{branch} has duplicate stress artifacts")
        branch_keys[branch] = set(keys)

    if len(branch_keys) != 2:
        return failures
    candidate_keys = branch_keys["candidate_20f"]
    control_keys = branch_keys["control_6f"]
    if candidate_keys != control_keys:
        failures.append("candidate_20f and control_6f fold/seed pairs do not match")
    if len({fold for fold, _ in candidate_keys}) != 3:
        failures.append("paired evidence does not contain exactly 3 folds")
    if len({seed for _, seed in candidate_keys}) != len(TRAIN_SEEDS):
        failures.append(f"paired evidence does not contain exactly {len(TRAIN_SEEDS)} seed(s)")
    if len(candidate_keys) != 3 * len(TRAIN_SEEDS):
        failures.append(f"paired evidence does not contain all 3 folds x {len(TRAIN_SEEDS)} seed(s)")
    return failures
