"""与 DLTX 对齐口径的组合层指标。ARR/MDD/Sharpe/Calmar/Sortino/win_rate。"""
from __future__ import annotations
import numpy as np

_TDAYS = 252


def annualized_return(rets: np.ndarray) -> float:
    rets = np.asarray(rets)
    if len(rets) == 0:
        return 0.0
    total = float(np.prod(1.0 + rets))
    return total ** (_TDAYS / len(rets)) - 1.0


def annualized_vol(rets: np.ndarray) -> float:
    if len(rets) < 2:
        return 0.0
    return float(np.std(rets, ddof=1)) * np.sqrt(_TDAYS)


def sharpe(rets: np.ndarray) -> float:
    v = annualized_vol(rets)
    if v <= 0:
        return 0.0
    return annualized_return(rets) / v


def sortino(rets: np.ndarray) -> float:
    downside = np.minimum(rets, 0.0)
    dvol = float(np.std(downside, ddof=1)) * np.sqrt(_TDAYS) if len(downside) > 1 else 0.0
    if dvol <= 0:
        return 0.0
    return annualized_return(rets) / dvol


def max_drawdown(rets: np.ndarray) -> float:
    if len(rets) == 0:
        return 0.0
    nav = np.cumprod(1.0 + rets)
    peak = np.maximum.accumulate(nav)
    dd = (nav - peak) / peak
    return float(dd.min())


def calmar(rets: np.ndarray) -> float:
    mdd = max_drawdown(rets)
    if mdd == 0:
        return 0.0
    return annualized_return(rets) / abs(mdd)


def win_rate(rets: np.ndarray) -> float:
    if len(rets) == 0:
        return 0.0
    return float((rets > 0).mean())


def _daily_returns(rets: np.ndarray) -> np.ndarray:
    values = np.asarray(rets, dtype=object).reshape(-1)
    flattened = []
    for value in values:
        if np.isscalar(value):
            flattened.append(value)
        else:
            flattened.extend(np.asarray(value, dtype=float).reshape(-1))
    return np.asarray(flattened, dtype=float)


def metrics_pack(rets: np.ndarray, name: str) -> dict:
    daily_rets = _daily_returns(rets)
    return {
        "name": name,
        "arr": annualized_return(daily_rets),
        "vol": annualized_vol(daily_rets),
        "sharpe": sharpe(daily_rets),
        "sortino": sortino(daily_rets),
        "mdd": max_drawdown(daily_rets),
        "calmar": calmar(daily_rets),
        "win_rate": win_rate(daily_rets),
        "cumret": float(np.prod(1.0 + daily_rets) - 1.0),
    }
