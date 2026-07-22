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
