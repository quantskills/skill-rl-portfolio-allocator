from scripts.research_gates import evaluate_research_gates


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
