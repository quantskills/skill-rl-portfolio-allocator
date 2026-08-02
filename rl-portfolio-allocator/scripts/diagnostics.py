"""训练诊断:年化换手/平均持仓/敞口利用率/成本占比。设计 §3.3 强制要求。"""
from __future__ import annotations
import numpy as np


def reward_quality_report(rewards) -> dict:
    values = np.asarray(rewards, dtype=float)
    if values.size == 0:
        return {"std": 0.0, "std_in_range": False, "abs_q999": 0.0,
                "max_abs_share": 0.0, "passed": False}
    absolute = np.abs(values)
    abs_q999 = float(np.quantile(absolute, 0.999))
    total_absolute = float(absolute.sum())
    max_abs_share = float(absolute.max() / total_absolute) if total_absolute else 0.0
    std = float(values.std())
    return {
        "std": std,
        "std_in_range": 0.5 <= std <= 2.0,
        "abs_q999": abs_q999,
        "max_abs_share": max_abs_share,
        "passed": 0.5 <= std <= 2.0 and abs_q999 <= 5.0 and max_abs_share <= 0.01,
    }


def summarize_rollout(infos: list, trading_days: int = 252) -> dict:
    if not infos:
        return {}
    turnovers = np.array([i["turnover"] for i in infos])
    n_holds = np.array([i["n_long"] + i["n_short"] for i in infos])
    long_util = np.array([i["long_notional"] for i in infos])
    short_util = np.array([i["short_notional"] for i in infos])
    daily_turnover = float(turnovers.mean())
    return {
        "annualized_turnover": daily_turnover * trading_days,
        "avg_n_holdings": float(n_holds.mean()),
        "long_exposure_util": float(long_util.mean()),
        "short_exposure_util": float(short_util.mean()),
        "cost_breakdown": {
            "commission_bps_per_day": float(np.mean([i["commission"] for i in infos])) * 1e4,
            "stamp_tax_bps_per_day": float(np.mean([i["stamp_tax"] for i in infos])) * 1e4,
            "impact_bps_per_day": float(np.mean([i["impact"] for i in infos])) * 1e4,
            "borrow_bps_per_day": float(np.mean([i["borrow"] for i in infos])) * 1e4,
        },
        "dsr_metric_mean": float(np.mean([i.get("dsr", 0.0) for i in infos])),
        "reward_breakdown": {
            "scaled_net_return_mean": float(np.mean([i["reward_parts"].get("scaled_net_return", 0.0) for i in infos])),
            "incremental_drawdown_penalty_mean": float(np.mean([i["reward_parts"].get("incremental_drawdown_penalty", i["reward_parts"].get("drawdown_penalty", 0.0)) for i in infos])),
            "turnover_penalty_mean": float(np.mean([i["reward_parts"]["turnover_penalty"] for i in infos])),
            "concentration_penalty_mean": float(np.mean([i["reward_parts"]["concentration_penalty"] for i in infos])),
        },
    }


def check_degeneracy(summary: dict, cfg: dict) -> list:
    warnings = []
    if summary.get("annualized_turnover", 0) < 0.5:
        warnings.append("DEGENERATE: annualized_turnover < 0.5 → agent 可能躺平,建议调小 λ_turnover")
    if summary.get("avg_n_holdings", 0) < 5:
        warnings.append("DEGENERATE: avg_n_holdings < 5 → 集中度过高或空仓,建议调小 λ_concentration")
    if summary.get("long_exposure_util", 0) < 0.5:
        warnings.append("DEGENERATE: long_exposure_util < 0.5 → 多头敞口未充分使用,建议调小 λ_drawdown")
    return warnings


def print_report(summary: dict, warnings: list) -> None:
    import json
    print("=== Rollout Diagnostics ===")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    if warnings:
        print("\n[WARNINGS]")
        for w in warnings:
            print(f"  - {w}")
    else:
        print("\n[OK] no degeneracy detected")
