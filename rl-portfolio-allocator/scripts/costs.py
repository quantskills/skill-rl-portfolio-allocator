"""交易/印花税/冲击/融券成本。所有函数返回该日成本占组合总名义(=1.0)的比例。"""
from __future__ import annotations
import numpy as np


def scaled_cost_config(cfg: dict, multiplier: float) -> dict:
    if not np.isfinite(multiplier) or multiplier <= 0:
        raise ValueError("cost multiplier must be positive")
    out = dict(cfg)
    for key in ("commission_bps", "stamp_tax_bps", "impact_bps", "borrow_rate_annual"):
        out[key] = cfg[key] * multiplier
    return out


def commission_cost(prev_w: np.ndarray, target_w: np.ndarray, commission_bps: float) -> float:
    turnover = float(np.abs(target_w - prev_w).sum())
    return commission_bps / 1e4 * turnover


def stamp_tax_cost(prev_w: np.ndarray, target_w: np.ndarray, stamp_bps: float) -> float:
    delta = target_w - prev_w
    sell = float(np.clip(-delta, 0.0, None).sum())
    return stamp_bps / 1e4 * sell


def impact_cost(
    prev_w: np.ndarray, target_w: np.ndarray, impact_bps: float, nonlinear: bool = False
) -> float:
    d = np.abs(target_w - prev_w)
    magnitude = float((d ** 1.5).sum()) if nonlinear else float(d.sum())
    return impact_bps / 1e4 * magnitude


def borrow_cost(
    target_w: np.ndarray, borrow_rate_annual: float, trading_days_per_year: int
) -> float:
    short_notional = float(np.clip(-target_w, 0.0, None).sum())
    return borrow_rate_annual / trading_days_per_year * short_notional


def total_costs(prev_w: np.ndarray, target_w: np.ndarray, cfg: dict) -> dict:
    c = commission_cost(prev_w, target_w, cfg["commission_bps"])
    s = stamp_tax_cost(prev_w, target_w, cfg["stamp_tax_bps"])
    i = impact_cost(prev_w, target_w, cfg["impact_bps"], nonlinear=False)
    b = borrow_cost(target_w, cfg["borrow_rate_annual"], cfg["trading_days_per_year"])
    return {"commission": c, "stamp_tax": s, "impact": i, "borrow": b, "total": c + s + i + b}
