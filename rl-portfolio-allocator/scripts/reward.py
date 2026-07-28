"""差分 Sharpe(Moody & Saffell 1998)+ 惩罚项。R_t 必须是扣完全成本的净收益。"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np


@dataclass
class DSRState:
    A: float = 0.0
    B: float = 1e-6
    nav: float = 1.0
    peak: float = 1.0

    def update(self, r_net: float, eta: float, sortino: bool = False) -> float:
        """更新内部状态,返回本步 DSR 增量。"""
        delta_A = r_net - self.A
        r2 = min(r_net, 0.0) ** 2 if sortino else r_net ** 2
        delta_B = r2 - self.B
        denom = (self.B - self.A ** 2) ** 1.5
        if denom <= 0 or not np.isfinite(denom):
            dsr = 0.0
        else:
            dsr = float((self.B * delta_A - 0.5 * self.A * delta_B) / denom)
        self.A = self.A + eta * delta_A
        self.B = self.B + eta * delta_B
        self.nav = self.nav * (1.0 + r_net)
        if self.nav > self.peak:
            self.peak = self.nav
        return dsr


def hhi(weights: np.ndarray) -> float:
    """赫芬达尔指数,基于**绝对**权重(空头也计入集中度)。"""
    aw = np.abs(weights)
    denom = aw.sum() + 1e-12
    p = aw / denom
    return float((p ** 2).sum())


def compose_reward(
    dsr_delta: float, drawdown: float, turnover: float, hhi_val: float, cfg: dict,
    net_ret: float,
    long_notional: float = 0.0, short_notional: float = 0.0, long_cap: float = 1.0, short_cap: float = 0.3
) -> tuple[float, dict]:
    ret_term = cfg["reward_ret_weight"] * net_ret
    dd_pen = -cfg["lambda_drawdown"] * max(0.0, drawdown)
    to_pen = -cfg["lambda_turnover"] * turnover
    conc_pen = -cfg["lambda_concentration"] * hhi_val

    # Hard constraint penalty: if notional exceeds cap, apply large penalty
    constraint_pen = 0.0
    if long_notional > long_cap * 1.01:
        constraint_pen -= 1.0 * (long_notional - long_cap)  # 1.0 = very high cost per unit excess
    if short_notional > short_cap * 1.01:
        constraint_pen -= 1.0 * (short_notional - short_cap)

    total = ret_term + dsr_delta + dd_pen + to_pen + conc_pen + constraint_pen
    return total, {
        "ret_term": ret_term,
        "dsr": dsr_delta,
        "drawdown_penalty": dd_pen,
        "turnover_penalty": to_pen,
        "concentration_penalty": conc_pen,
        "constraint_penalty": constraint_pen,
        "total": total,
    }
