"""因子权重 → Top-N 多头 + Bottom-M 空头 → 目标持仓。"""
from __future__ import annotations
import numpy as np


def composite_score(F: np.ndarray, factor_w: np.ndarray) -> np.ndarray:
    # NaN in factor matrix means "no signal" for that stock/factor — treat as 0
    F_clean = np.nan_to_num(F, nan=0.0)
    return F_clean @ factor_w


def select_long_short(
    scores: np.ndarray, is_suspended: np.ndarray, top_n: int, bottom_m: int
) -> tuple[np.ndarray, np.ndarray]:
    n = len(scores)
    mask = ~is_suspended
    idx = np.arange(n)[mask]
    if len(idx) == 0:
        return np.array([], dtype=int), np.array([], dtype=int)
    s = scores[mask]
    order = np.argsort(-s)
    long_local = order[: min(top_n, len(order))]
    short_local = order[-min(bottom_m, len(order)) :] if bottom_m > 0 else np.array([], dtype=int)
    return idx[long_local], idx[short_local]


def _score_weighted(scores_pos: np.ndarray) -> np.ndarray:
    tot = scores_pos.sum()
    if tot <= 0:
        return np.ones_like(scores_pos) / max(len(scores_pos), 1)
    return scores_pos / tot


def target_weights(
    scores: np.ndarray,
    long_idx: np.ndarray,
    short_idx: np.ndarray,
    long_notional: float,
    short_cap: float,
) -> np.ndarray:
    n = len(scores)
    w = np.zeros(n)
    if len(long_idx) > 0:
        s_long = np.clip(scores[long_idx], a_min=1e-8, a_max=None)
        w[long_idx] = long_notional * _score_weighted(s_long)
    if len(short_idx) > 0:
        s_short = np.clip(-scores[short_idx], a_min=1e-8, a_max=None)
        w[short_idx] = -short_cap * _score_weighted(s_short)

    # Hard constraint: ensure total notional respects limits
    long_sum = np.clip(w, 0, None).sum()
    short_sum = np.clip(-w, 0, None).sum()
    if long_sum > long_notional * 1.01:  # 1% tolerance
        w[w > 0] *= long_notional / long_sum
    if short_sum > short_cap * 1.01:  # 1% tolerance
        w[w < 0] *= short_cap / short_sum

    return w


def freeze_suspended(
    target_w: np.ndarray, prev_w: np.ndarray, is_suspended: np.ndarray
) -> np.ndarray:
    out = target_w.copy()
    out[is_suspended] = prev_w[is_suspended]

    # Re-apply hard constraint after freezing to avoid exceeding limits
    long_sum = np.clip(out, 0, None).sum()
    short_sum = np.clip(-out, 0, None).sum()
    # Enforce hard limits (0.1% tolerance)
    if long_sum > 1.001:
        out[out > 0] *= 1.0 / long_sum
    if short_sum > 0.3001:
        out[out < 0] *= 0.3 / short_sum
    return out
