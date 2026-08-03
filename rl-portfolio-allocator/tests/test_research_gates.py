import pytest

from scripts.research_gates import evaluate_candidate_gates, evaluate_research_gates
from scripts.walk_forward import aggregate_candidate_comparison


def passing_summary():
    return {
        "combined_oos_arr": 0.08,
        "median_seed_oos_sharpe": 0.60,
        "strongest_baseline_sharpe": 0.40,
        "positive_excess_return_folds": 3,
        "total_folds": 3,
        "median_seed_excess_return": 0.03,
        "oos_mdd": -0.25,
        "strongest_baseline_mdd": -0.24,
        "annualized_turnover": 8.0,
        "cost_2x_oos_sharpe": 0.12,
        "no_leakage_tests_passed": True,
        "state_quality_tests_passed": True,
    }


def test_all_hard_gates_can_pass():
    result = evaluate_research_gates(passing_summary())

    assert result["research_ok"] is True
    assert all(gate["passed"] for gate in result["gates"])


def test_missing_metric_fails_closed():
    summary = passing_summary()
    del summary["combined_oos_arr"]

    result = evaluate_research_gates(summary)

    arr_gate = next(gate for gate in result["gates"] if gate["name"] == "combined_oos_arr")
    assert result["research_ok"] is False
    assert arr_gate["passed"] is False
    assert arr_gate["actual"] is None


def test_negative_mdd_uses_absolute_magnitude():
    summary = passing_summary()
    summary["oos_mdd"] = -0.30

    result = evaluate_research_gates(summary)

    mdd_gate = next(gate for gate in result["gates"] if gate["name"] == "oos_mdd")
    assert result["research_ok"] is False
    assert mdd_gate["passed"] is False


def complete_paired_rows():
    candidate_rows = []
    control_rows = []
    for fold in range(1, 4):
        for seed in range(5):
            candidate_rows.append({
                "branch": "candidate_20f", "fold": fold, "seed": seed,
                "oos_sharpe": 0.60, "cost_2x_oos_sharpe": 0.12,
                "annualized_turnover": 8.0, "stress_mdd": -0.25,
                "stress_artifact_path": f"candidate_20f/stress/fold{fold}/seed{seed}.json",
                "stress_artifact_sha256": "sha256:" + "0" * 64,
            })
            control_rows.append({
                "branch": "control_6f", "fold": fold, "seed": seed,
                "oos_sharpe": 0.40, "cost_2x_oos_sharpe": 0.08,
                "annualized_turnover": 7.0, "stress_mdd": -0.24,
                "stress_artifact_path": f"control_6f/stress/fold{fold}/seed{seed}.json",
                "stress_artifact_sha256": "sha256:" + "1" * 64,
            })
    return candidate_rows, control_rows


def candidate_passing_summary():
    candidate_rows, control_rows = complete_paired_rows()
    return aggregate_candidate_comparison(
        candidate_rows, control_rows, fold_count=3, seed_count=5,
    )


def test_candidate_gates_pass_for_complete_paired_full_run():
    result = evaluate_candidate_gates(candidate_passing_summary())

    assert result["research_ok"] is True
    assert all(gate["passed"] for gate in result["gates"])


def test_candidate_gates_reject_insufficient_sharpe_gain():
    summary = candidate_passing_summary()
    summary["candidate_median_oos_sharpe"] = 0.49

    result = evaluate_candidate_gates(summary)

    sharpe_gate = next(gate for gate in result["gates"] if gate["name"] == "median_oos_sharpe_gain")
    assert result["research_ok"] is False
    assert sharpe_gate["actual"] == pytest.approx(0.09)
    assert sharpe_gate["passed"] is False


def test_paired_comparison_does_not_substitute_oos_mdd_for_missing_stress_mdd():
    comparison = aggregate_candidate_comparison(
        [{"fold": 1, "seed": 0, "oos_sharpe": 0.6, "oos_mdd": -0.20}],
        [{"fold": 1, "seed": 0, "oos_sharpe": 0.4, "oos_mdd": -0.10}],
        fold_count=3,
        seed_count=5,
    )

    result = evaluate_candidate_gates(comparison)
    stress_gate = next(gate for gate in result["gates"] if gate["name"] == "candidate_stress_mdd_excess")
    assert comparison["candidate_stress_mdd"] is None
    assert comparison["control_stress_mdd"] is None
    assert stress_gate["actual"] is None
    assert stress_gate["passed"] is False
    assert result["research_ok"] is False


def test_candidate_gates_reject_fourteen_of_fifteen_paired_rows():
    candidate_rows, control_rows = complete_paired_rows()
    candidate_rows.pop()

    summary = aggregate_candidate_comparison(
        candidate_rows, control_rows, fold_count=3, seed_count=5,
    )
    result = evaluate_candidate_gates(summary)

    assert result["research_ok"] is False
    assert any("candidate_20f has 14 rows; expected 15" in reason for reason in result["failure_reasons"])


def test_candidate_gates_reject_missing_required_pair_metric():
    candidate_rows, control_rows = complete_paired_rows()
    candidate_rows[0].pop("cost_2x_oos_sharpe")
    summary = aggregate_candidate_comparison(
        candidate_rows, control_rows, fold_count=3, seed_count=5,
    )

    result = evaluate_candidate_gates(summary)

    assert result["research_ok"] is False
    assert any("cost_2x_oos_sharpe" in reason for reason in result["failure_reasons"])


def test_candidate_gates_reject_rows_without_distinct_persisted_stress_evidence():
    candidate_rows, control_rows = complete_paired_rows()
    for row in candidate_rows + control_rows:
        row["stress_artifact_path"] = "stress/shared.json"

    summary = aggregate_candidate_comparison(
        candidate_rows, control_rows, fold_count=3, seed_count=5,
    )
    result = evaluate_candidate_gates(summary)

    assert result["research_ok"] is False
    assert any("stress artifact" in reason for reason in result["failure_reasons"])
