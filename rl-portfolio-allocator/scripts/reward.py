"""Reward composition and the explicit legacy DSR reward path."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np


_REWARD_COEFFICIENTS = {
    "none": (0.0, 0.0, 0.0),
    "gentle": (0.10, 0.20, 0.05),
    "low": (0.5, 0.5, 0.05),
    "medium": (1.0, 1.0, 0.10),
}


def reward_coefficients(variant: str) -> tuple[float, float, float]:
    try:
        return _REWARD_COEFFICIENTS[variant]
    except KeyError as exc:
        raise ValueError(f"unknown reward variant: {variant}") from exc


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


@dataclass
class DualState:
    """约束式 reward 的对偶变量:λ 随 episode 最大回撤相对目标自适应。

    回撤远低于 target 时 λ 衰减到 0(不压多头,修复退化解);
    回撤逼近/超过 target 时 λ 单调放大(防守力度自适应)。
    """
    lam: float = 0.0
    episode_mdd: float = 0.0

    def update(self, drawdown: float, target_mdd: float, lr_dual: float) -> float:
        self.episode_mdd = max(self.episode_mdd, drawdown)
        self.lam = max(0.0, self.lam + lr_dual * (self.episode_mdd - target_mdd))
        return self.lam


def compose_reward(
    net_ret: float, prev_drawdown: float, drawdown: float, turnover: float,
    hhi_val: float, cfg: dict,
) -> tuple[float, dict]:
    dd_coeff, concentration_coeff, turnover_coeff = reward_coefficients(cfg["reward_variant"])
    scaled_net_return = cfg["reward_scale"] * net_ret
    incremental_drawdown = max(0.0, drawdown - prev_drawdown)
    dd_penalty = -dd_coeff * incremental_drawdown
    turnover_penalty = -turnover_coeff * max(0.0, turnover - cfg["turnover_budget"])
    concentration_penalty = -concentration_coeff * max(0.0, hhi_val - cfg["hhi_target"])
    raw_total = scaled_net_return + dd_penalty + turnover_penalty + concentration_penalty
    total = float(np.clip(raw_total, -cfg["reward_clip"], cfg["reward_clip"]))
    return total, {
        "scaled_net_return": scaled_net_return,
        "incremental_drawdown_penalty": dd_penalty,
        "turnover_penalty": turnover_penalty,
        "concentration_penalty": concentration_penalty,
        "total": total,
    }


def compose_constrained_reward(
    net_ret: float, prev_drawdown: float, drawdown: float, turnover: float,
    hhi_val: float, lam: float, cfg: dict,
) -> tuple[float, dict]:
    """约束式 reward:固定 dd_coeff 换成对偶变量 λ_t,加恢复信用与下行半方差。

    total = 100·net_ret − λ_t·max(0, dd−prev_dd) + κ_rec·max(0, prev_dd−dd)
            − σ_down·min(0, net_ret)² − 0.05·max(0, turnover−0.2) − 0.5·max(0, hhi−0.03)
    """
    _, concentration_coeff, turnover_coeff = reward_coefficients("low")
    scaled_net_return = cfg["reward_scale"] * net_ret
    dd_penalty = -lam * max(0.0, drawdown - prev_drawdown)
    recovery_credit = cfg["recovery_credit"] * max(0.0, prev_drawdown - drawdown)
    downside_penalty = -cfg["downside_vol_coeff"] * min(0.0, net_ret) ** 2
    turnover_penalty = -turnover_coeff * max(0.0, turnover - cfg["turnover_budget"])
    concentration_penalty = -concentration_coeff * max(0.0, hhi_val - cfg["hhi_target"])
    raw_total = (
        scaled_net_return + dd_penalty + recovery_credit
        + downside_penalty + turnover_penalty + concentration_penalty
    )
    total = float(np.clip(raw_total, -cfg["reward_clip"], cfg["reward_clip"]))
    return total, {
        "scaled_net_return": scaled_net_return,
        "incremental_drawdown_penalty": dd_penalty,
        "recovery_credit": recovery_credit,
        "downside_vol_penalty": downside_penalty,
        "turnover_penalty": turnover_penalty,
        "concentration_penalty": concentration_penalty,
        "dual_lambda": float(lam),
        "total": total,
    }


def compose_legacy_dsr_reward(
    dsr_delta: float, drawdown: float, turnover: float, hhi_val: float, cfg: dict,
    net_ret: float, long_notional: float = 0.0, short_notional: float = 0.0,
    long_cap: float = 1.0, short_cap: float = 0.3,
) -> tuple[float, dict]:
    ret_term = cfg["reward_ret_weight"] * net_ret
    dd_pen = -cfg["lambda_drawdown"] * max(0.0, drawdown)
    to_pen = -cfg["lambda_turnover"] * turnover
    conc_pen = -cfg["lambda_concentration"] * hhi_val
    constraint_pen = 0.0
    if long_notional > long_cap * 1.01:
        constraint_pen -= long_notional - long_cap
    if short_notional > short_cap * 1.01:
        constraint_pen -= short_notional - short_cap
    total = ret_term + dsr_delta + dd_pen + to_pen + conc_pen + constraint_pen
    return total, {
        "ret_term": ret_term, "dsr": dsr_delta,
        "drawdown_penalty": dd_pen, "turnover_penalty": to_pen,
        "concentration_penalty": conc_pen, "constraint_penalty": constraint_pen,
        "total": total,
    }
