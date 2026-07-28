"""RL State 构造:波动/相关性/因子暴露/持仓。所有量只用 ≤ t 的信息。"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional
import numpy as np
import pandas as pd


def state_dim(k: int) -> int:
    # [vol_20, vol_60, market_vol_20, vol_regime_q] (4)
    # + [avg_corr_20, avg_corr_60] (2)
    # + exposure (k) + factor_ic (k) + prev_factor_w (k) (3*k)
    # + [cash, recent_to] (2)
    # + [volatility_regime, drawdown_flag] (2) ← NEW market regime indicators
    return 8 + 3 * k + 2


def _safe_std(x: np.ndarray) -> float:
    return float(np.std(x)) if len(x) >= 2 else 0.0


def _quantile_rank(x: float, arr: np.ndarray) -> float:
    if len(arr) == 0:
        return 0.5
    return float((arr <= x).mean())


@dataclass
class StateBuilder:
    factor_panel_by_date: dict
    index_returns: pd.Series
    portfolio_returns_history: list = field(default_factory=list)
    recent_turnover_history: list = field(default_factory=list)
    _vol_history: list = field(default_factory=list)

    def build(
        self,
        date,
        holdings_w: np.ndarray,
        factor_names: list,
        prev_factor_w: np.ndarray,
        cash: float,
    ) -> np.ndarray:
        rets = np.asarray(self.portfolio_returns_history, dtype=float)
        vol_20 = _safe_std(rets[-20:])
        vol_60 = _safe_std(rets[-60:])
        idx_slice = self.index_returns.loc[:date].values
        market_vol_20 = _safe_std(idx_slice[-20:])
        self._vol_history.append(vol_20)
        vol_regime_q = _quantile_rank(vol_20, np.asarray(self._vol_history[:-1]))

        F = self.factor_panel_by_date.get(date)
        if F is not None and len(F) >= 5:
            corr = F.corr().values
            iu = np.triu_indices_from(corr, k=1)
            if iu[0].size > 0 and np.isfinite(corr[iu]).any():
                avg_corr_20 = float(np.nanmean(corr[iu]))
            else:
                avg_corr_20 = 0.0
        else:
            avg_corr_20 = 0.0
        avg_corr_60 = avg_corr_20

        if F is not None:
            exposure = F.values.T @ holdings_w[: len(F)] if len(holdings_w) >= len(F) else np.zeros(len(factor_names))
        else:
            exposure = np.zeros(len(factor_names))

        factor_ic = np.zeros(len(factor_names))

        recent_to = float(np.mean(self.recent_turnover_history[-20:])) if self.recent_turnover_history else 0.0

        # Market regime indicators: detect structural breaks
        vol_regime = 1.0 if vol_20 > 2 * max(np.percentile(self._vol_history[-252:], 75), 0.01) else 0.0
        drawdown_flag = 1.0 if (self.portfolio_returns_history and np.sum(np.array(self.portfolio_returns_history[-60:]) < 0) > 30) else 0.0

        vec = np.concatenate([
            np.array([vol_20, vol_60, market_vol_20, vol_regime_q], dtype=float),
            np.array([avg_corr_20, avg_corr_60], dtype=float),
            exposure.astype(float),
            factor_ic.astype(float),
            prev_factor_w.astype(float),
            np.array([cash, recent_to, vol_regime, drawdown_flag], dtype=float),
        ])
        vec = np.nan_to_num(vec, nan=0.0, posinf=0.0, neginf=0.0)
        return vec.astype(np.float32)
