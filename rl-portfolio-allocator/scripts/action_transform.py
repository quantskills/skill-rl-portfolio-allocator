"""RL 原始动作 → K 维因子权重的变换:tanh → L1 归一化 → EMA 平滑 → 再 L1。"""
from __future__ import annotations
import numpy as np

_EPS = 1e-8


def tanh_l1_normalize(raw: np.ndarray, eps: float = _EPS) -> np.ndarray:
    t = np.tanh(raw)
    denom = np.abs(t).sum() + eps
    return t / denom


def ema_smooth(w_new: np.ndarray, w_prev: np.ndarray, alpha: float) -> np.ndarray:
    return alpha * w_new + (1.0 - alpha) * w_prev


def transform_action(raw: np.ndarray, w_prev: np.ndarray, alpha: float) -> np.ndarray:
    w_new = tanh_l1_normalize(raw)
    w_ema = ema_smooth(w_new, w_prev, alpha)
    denom = np.abs(w_ema).sum() + _EPS
    return w_ema / denom


def transform_delta_action(
    raw_delta: np.ndarray,
    previous: np.ndarray,
    max_delta: float,
    ema_alpha: float,
) -> np.ndarray:
    """Convert an RL action into a smoothed, unit-L1 portfolio delta target."""
    raw = np.asarray(raw_delta, dtype=float)
    prev = np.asarray(previous, dtype=float)
    if raw.shape != prev.shape:
        raise ValueError("raw_delta and previous must have the same shape")
    if not (np.isfinite(raw).all() and np.isfinite(prev).all()):
        raise ValueError("action and previous weights must be finite")
    if not np.isfinite(max_delta) or max_delta < 0:
        raise ValueError("max_delta must be finite and non-negative")
    if not np.isfinite(ema_alpha) or not 0.0 <= ema_alpha <= 1.0:
        raise ValueError("ema_alpha must be between zero and one")
    if np.all(raw == 0):
        return prev.copy()

    delta = np.clip(np.tanh(raw), -max_delta, max_delta)
    target = prev + delta
    target /= max(np.abs(target).sum(), _EPS)
    smoothed = ema_alpha * target + (1.0 - ema_alpha) * prev
    smoothed /= max(np.abs(smoothed).sum(), _EPS)
    if not np.isfinite(smoothed).all():
        raise ValueError("action transform produced non-finite weights")
    return smoothed
