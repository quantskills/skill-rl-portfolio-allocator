"""Strictly causal state construction for the portfolio allocator."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


STATE_SCHEMA_VERSION = "state-v2-dynamic-factors"
BASE_MARKET_FIELDS = (
    "market_ret_20",
    "market_ret_60",
    "market_vol_20",
    "market_vol_60",
    "market_drawdown_60",
    "market_vol_regime",
)


def exogenous_fields(factor_names: list[str] | tuple[str, ...]) -> list[str]:
    fields = list(BASE_MARKET_FIELDS)
    for name in factor_names:
        fields.extend(
            [
                f"{name}_ic_mean_20",
                f"{name}_ic_mean_60",
                f"{name}_icir_20",
                f"{name}_ic_positive_20",
                f"{name}_factor_ret_20",
                f"{name}_factor_ret_60",
                f"{name}_factor_vol_20",
                f"{name}_factor_vol_60",
            ]
        )
    fields.extend(["factor_corr_20", "factor_corr_60"])
    return fields


def state_fields(factor_names: list[str] | tuple[str, ...]) -> list[str]:
    names = list(factor_names)
    fields = exogenous_fields(names)
    fields.extend(f"exposure_{name}" for name in names)
    fields.extend(f"prev_factor_w_{name}" for name in names)
    fields.extend(
        [
            "cash",
            "recent_turnover",
            "portfolio_vol_20",
            "portfolio_vol_60",
            "portfolio_drawdown",
        ]
    )
    return fields


def state_dim(factor_names: list[str] | tuple[str, ...]) -> int:
    return len(state_fields(factor_names))


def _safe_std(values: np.ndarray) -> float:
    return float(np.std(values)) if len(values) >= 2 else 0.0


def _finite_panel_exposure(panel: pd.DataFrame | None, holdings_w: np.ndarray, names: list[str]) -> np.ndarray:
    if panel is None:
        return np.zeros(len(names), dtype=float)
    exposure = np.zeros(len(names), dtype=float)
    weights = np.asarray(holdings_w, dtype=float)[: len(panel)]
    for i, name in enumerate(names):
        if name not in panel:
            continue
        values = pd.to_numeric(panel[name], errors="coerce").to_numpy(dtype=float)
        finite = np.isfinite(values[: len(weights)])
        exposure[i] = float(np.dot(values[: len(weights)][finite], weights[finite]))
    return exposure


@dataclass
class StateBuilder:
    factor_panel_by_date: dict
    market_state: pd.DataFrame | pd.Series | None = None
    portfolio_returns_history: list = field(default_factory=list)
    recent_turnover_history: list = field(default_factory=list)
    index_returns: pd.Series | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.market_state, pd.DataFrame):
            raise TypeError("market_state must be an explicit pandas DataFrame")
        if self.index_returns is not None:
            raise TypeError("index_returns is unsupported; pass market_state explicitly")
        self.market_state = self.market_state.copy()
        self.market_state.index = pd.to_datetime(self.market_state.index)
        self._field_index: dict[str, int] = {}

    def field_index(self, field_name: str, factor_names: list[str] | tuple[str, ...] | None = None) -> int:
        if not self._field_index and factor_names is None:
            raise TypeError("factor_names are required for state field construction")
        if not self._field_index or (factor_names is not None and field_name not in self._field_index):
            names = list(factor_names)
            self._field_index = {name: i for i, name in enumerate(state_fields(names))}
        return self._field_index[field_name]

    def build(
        self,
        date,
        holdings_w: np.ndarray,
        factor_names: list[str],
        prev_factor_w: np.ndarray,
        cash: float,
    ) -> np.ndarray:
        names = list(factor_names)
        fields = state_fields(names)
        try:
            row = self.market_state.loc[pd.Timestamp(date)]
        except KeyError as exc:
            raise KeyError(f"missing market state for {date}") from exc
        exogenous = pd.to_numeric(row.reindex(exogenous_fields(names)), errors="coerce").to_numpy(dtype=float)
        if not np.isfinite(exogenous).all():
            raise ValueError(f"non-finite exogenous market state for {date}")

        panel = self.factor_panel_by_date.get(date)
        exposure = _finite_panel_exposure(panel, holdings_w, names)
        returns = np.asarray(self.portfolio_returns_history, dtype=float)
        recent = np.asarray(self.recent_turnover_history[-20:], dtype=float)
        wealth = np.cumprod(1.0 + returns[-60:]) if len(returns) else np.array([])
        drawdown = float(wealth[-1] / np.max(wealth) - 1.0) if len(wealth) else 0.0
        portfolio = np.array(
            [
                float(cash),
                float(np.mean(recent)) if len(recent) else 0.0,
                _safe_std(returns[-20:]),
                _safe_std(returns[-60:]),
                drawdown,
            ],
            dtype=float,
        )
        values = np.concatenate([exogenous, exposure, np.asarray(prev_factor_w, dtype=float), portfolio])
        self._field_index = {name: i for i, name in enumerate(fields)}
        return values.astype(np.float32)
