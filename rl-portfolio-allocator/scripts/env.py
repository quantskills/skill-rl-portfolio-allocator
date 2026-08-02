"""Gymnasium 环境:state=波动/相关/暴露/持仓, action=K维 RL 输出,
step 内嵌成本、用次日收益结算、reward=DSR+惩罚。"""
from __future__ import annotations
from typing import Optional
import numpy as np
import pandas as pd

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError:
    import gym
    from gym import spaces

from scripts.config import FACTOR_NAMES, K
from scripts.action_transform import transform_delta_action
from scripts.costs import total_costs
from scripts.portfolio import (
    composite_score, select_long_short, target_weights, freeze_suspended
)
from scripts.reward import DSRState, hhi, compose_reward, compose_legacy_dsr_reward
from scripts.state import StateBuilder, exogenous_fields, state_dim
from scripts.rebalance import buffered_long_short, project_turnover, weekly_decision_indices


def extract_settle_holding_period(prev_w, target_w, settlement_dates, returns_by_date, cfg):
    """Settle a target portfolio over a period, compounding daily net returns."""
    dates = list(pd.to_datetime(settlement_dates))
    if not dates:
        raise ValueError("settlement_dates must not be empty")
    prev = np.asarray(prev_w, dtype=float)
    target = np.asarray(target_w, dtype=float)
    trade_cfg = dict(cfg)
    trade_cfg["borrow_rate_annual"] = 0.0
    trade_costs = total_costs(prev, target, trade_cfg)
    daily_gross = []
    daily_net = []
    borrow_total = 0.0
    for i, date in enumerate(dates):
        gross = float(target @ np.asarray(returns_by_date[date], dtype=float))
        borrow = float(
            total_costs(target, target, cfg)["borrow"]
        )
        borrow_total += borrow
        day_cost = borrow + (trade_costs["total"] if i == 0 else 0.0)
        daily_gross.append(gross)
        daily_net.append(gross - day_cost)
    costs = {
        "commission": trade_costs["commission"],
        "stamp_tax": trade_costs["stamp_tax"],
        "impact": trade_costs["impact"],
        "borrow": borrow_total,
        "total": trade_costs["total"] + borrow_total,
    }
    return {
        "gross_ret": float(np.prod(1.0 + np.asarray(daily_gross)) - 1.0),
        "net_ret": float(np.prod(1.0 + np.asarray(daily_net)) - 1.0),
        "daily_gross_rets": daily_gross,
        "daily_net_rets": daily_net,
        "settlement_dates": dates,
        "costs": costs,
    }


class PortfolioEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(
        self,
        features_df: pd.DataFrame,
        market_state_df: pd.DataFrame,
        cfg: dict,
        start_date,
        end_date,
        nonlinear_impact: bool = False,
        observation_scaler=None,
    ):
        super().__init__()
        self.cfg = cfg
        self.nonlinear_impact = nonlinear_impact
        self.observation_scaler = observation_scaler
        if not isinstance(market_state_df, pd.DataFrame):
            raise TypeError("market_state_df must be an explicit pandas DataFrame")
        df = features_df.copy()
        df["trade_date"] = pd.to_datetime(df["trade_date"])
        requested_start = pd.Timestamp(start_date)
        requested_end = pd.Timestamp(end_date)
        df = df[(df["trade_date"] >= requested_start) & (df["trade_date"] <= requested_end)]
        df = df.sort_values(["trade_date", "symbol"]).reset_index(drop=True)
        market_state = market_state_df.copy()
        if "trade_date" in market_state.columns:
            market_state["trade_date"] = pd.to_datetime(market_state["trade_date"])
            market_state = market_state.set_index("trade_date")
        else:
            market_state.index = pd.to_datetime(market_state.index)
        if market_state.index.has_duplicates:
            raise ValueError("duplicate market state dates")
        market_state = market_state.sort_index()
        feature_dates = pd.DatetimeIndex(df["trade_date"].unique()).sort_values()
        state_dates = pd.DatetimeIndex(market_state.index.unique()).sort_values()
        missing = feature_dates.difference(state_dates)
        if len(missing):
            raise ValueError(
                "missing causal market state dates in requested range: "
                f"{missing[0].date()}..{missing[-1].date()} ({len(missing)} dates)"
            )
        dates = feature_dates.intersection(state_dates)
        if len(dates) == 0:
            raise ValueError("features and market_state have no dates in requested range")
        missing_fields = set(exogenous_fields(FACTOR_NAMES)) - set(market_state.columns)
        if missing_fields:
            raise ValueError(f"market_state missing columns: {sorted(missing_fields)}")
        self.features = df
        self.market_state = market_state.loc[dates]

        self.all_dates = list(dates)
        self.dates = self.all_dates
        self.decision_indices = weekly_decision_indices(self.all_dates)
        self.decision_dates = [self.all_dates[i] for i in self.decision_indices]
        self.symbols = sorted(df["symbol"].unique())
        self._sym_to_idx = {s: i for i, s in enumerate(self.symbols)}
        self.n = len(self.symbols)

        self._F_by_date: dict = {}
        self._ret_by_date: dict = {}
        self._susp_by_date: dict = {}
        for d, g in df.groupby("trade_date"):
            F = np.zeros((self.n, K), dtype=float)
            r = np.zeros(self.n, dtype=float)
            s = np.ones(self.n, dtype=bool)
            for _, row in g.iterrows():
                i = self._sym_to_idx[row["symbol"]]
                F[i] = [row[fn] for fn in FACTOR_NAMES]
                r[i] = 0.0 if pd.isna(row["ret_1d"]) else float(row["ret_1d"])
                s[i] = bool(row["is_suspended"])
            self._F_by_date[d] = F
            self._ret_by_date[d] = r
            self._susp_by_date[d] = s

        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(K,), dtype=np.float32)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(state_dim(FACTOR_NAMES),), dtype=np.float32,
        )
        self._reset_internal()

    def _scale(self, obs: np.ndarray) -> np.ndarray:
        if self.observation_scaler is None:
            return np.asarray(obs, dtype=np.float32)
        return self.observation_scaler.transform(obs)

    def _reset_internal(self):
        self.t = 0
        self.prev_factor_w = np.zeros(K)
        self.prev_stock_w = np.zeros(self.n)
        self.dsr = DSRState()
        self.prev_drawdown = 0.0
        panels = {}
        for d, F in self._F_by_date.items():
            panels[d] = pd.DataFrame(F, index=self.symbols, columns=FACTOR_NAMES)
        self.state_builder = StateBuilder(
            factor_panel_by_date=panels,
            market_state=self.market_state,
            portfolio_returns_history=[],
            recent_turnover_history=[],
        )

    def reset(self, seed: Optional[int] = None, options=None):
        super().reset(seed=seed)
        self._reset_internal()
        obs = self.state_builder.build(
            self.decision_dates[self.t], self.prev_stock_w, FACTOR_NAMES,
            self.prev_factor_w, cash=1.0,
        )
        return self._scale(obs), {}

    def step(self, action: np.ndarray):
        d = self.decision_dates[self.t]
        F = self._F_by_date[d]
        susp = self._susp_by_date[d]

        factor_w = transform_delta_action(
            np.asarray(action, dtype=float), self.prev_factor_w,
            self.cfg.get("max_delta", 0.2), self.cfg["ema_alpha"],
        )
        scores = composite_score(F, factor_w)
        long_idx, short_idx = buffered_long_short(
            scores, susp, self.prev_stock_w,
            self.cfg["top_n"], self.cfg.get("long_exit", self.cfg["top_n"]),
            self.cfg["bottom_m"], self.cfg.get("short_exit", self.cfg["bottom_m"]),
        )
        target_w = target_weights(scores, long_idx, short_idx, self.cfg["long_notional"], self.cfg["short_notional_cap"])
        target_w = freeze_suspended(target_w, self.prev_stock_w, susp)
        target_w = project_turnover(
            self.prev_stock_w, target_w, susp,
            self.cfg["turnover_budget"], self.cfg["long_notional"], self.cfg["short_notional_cap"],
        )
        next_t = self.t + 1
        end_idx = self.decision_indices[next_t] if next_t < len(self.decision_indices) else len(self.all_dates) - 1
        settlement_dates = self.all_dates[self.all_dates.index(d) + 1:end_idx + 1]
        settled = extract_settle_holding_period(
            self.prev_stock_w, target_w, settlement_dates, self._ret_by_date, self.cfg,
        )
        costs = settled["costs"]
        gross = settled["gross_ret"]
        net = settled["net_ret"]
        terminated = next_t >= len(self.decision_dates) - 1
        truncated = False
        turnover = float(np.abs(target_w - self.prev_stock_w).sum())
        long_notional = float(np.clip(target_w, 0, None).sum())
        short_notional = float(np.clip(-target_w, 0, None).sum())

        # Hard clip to enforce constraints (safety guard after all processing)
        if long_notional > self.cfg["long_notional"] * 1.001:
            target_w[target_w > 0] *= self.cfg["long_notional"] / long_notional
            long_notional = self.cfg["long_notional"]
        if short_notional > self.cfg["short_notional_cap"] * 1.001:
            target_w[target_w < 0] *= self.cfg["short_notional_cap"] / short_notional
            short_notional = self.cfg["short_notional_cap"]
        hhi_v = hhi(target_w)
        dsr_delta = self.dsr.update(net, self.cfg["dsr_eta"], sortino=(self.cfg["reward_type"] == "sortino"))
        drawdown = 0.0 if self.dsr.peak <= 0 else (self.dsr.peak - self.dsr.nav) / self.dsr.peak
        if self.cfg["reward_variant"] == "legacy_dsr":
            reward, parts = compose_legacy_dsr_reward(
                dsr_delta, drawdown, turnover, hhi_v, self.cfg, net,
                long_notional, short_notional, self.cfg["long_notional"],
                self.cfg["short_notional_cap"],
            )
        else:
            reward, parts = compose_reward(net, self.prev_drawdown, drawdown, turnover, hhi_v, self.cfg)
        self.prev_drawdown = drawdown

        self.state_builder.portfolio_returns_history.append(net)
        self.state_builder.recent_turnover_history.append(turnover)

        self.prev_factor_w = factor_w
        self.prev_stock_w = target_w
        self.t = next_t

        if not terminated:
            obs = self.state_builder.build(
                self.decision_dates[self.t], self.prev_stock_w, FACTOR_NAMES,
                self.prev_factor_w, cash=max(0.0, 1.0 - long_notional - short_notional),
            )
        else:
            obs = np.zeros(state_dim(FACTOR_NAMES), dtype=np.float32)

        info = {
            **costs,
            "gross_ret": gross, "net_ret": net,
            "turnover": turnover,
            "long_notional": long_notional, "short_notional": short_notional,
            "n_long": int(len(long_idx)), "n_short": int(len(short_idx)),
            "factor_w": factor_w.tolist(),
            "hhi": hhi_v, "drawdown": drawdown,
            "diagnostics": {"dsr_metric": dsr_delta},
            "reward_parts": parts,
            "daily_net_rets": settled["daily_net_rets"],
            "settlement_dates": settled["settlement_dates"],
            "daily_gross_rets": settled["daily_gross_rets"],
            "cost_breakdown": costs,
            "ret_source": "weekly_settlement",
        }
        if self.cfg["reward_variant"] == "legacy_dsr":
            info["dsr"] = dsr_delta
        return self._scale(obs), float(reward), terminated, truncated, info


def effective_range(features_df: pd.DataFrame, market_state_df: pd.DataFrame,
                   start, end) -> tuple[pd.Timestamp, pd.Timestamp]:
    if not isinstance(market_state_df, pd.DataFrame):
        raise TypeError("market_state_df must be an explicit pandas DataFrame")
    feature_dates = pd.to_datetime(features_df["trade_date"]).unique()
    if "trade_date" in market_state_df.columns:
        state_dates = pd.to_datetime(market_state_df["trade_date"]).unique()
    else:
        state_dates = pd.to_datetime(market_state_df.index).unique()
    common = pd.DatetimeIndex(feature_dates).intersection(state_dates).sort_values()
    requested = common[(common >= pd.Timestamp(start)) & (common <= pd.Timestamp(end))]
    if len(requested) == 0:
        raise ValueError("features and market_state have no dates in requested range")
    return requested[0], requested[-1]


def make_env(features_path, market_state_path, cfg, start, end) -> PortfolioEnv:
    feats = pd.read_parquet(features_path)
    market_state = pd.read_parquet(market_state_path)
    return PortfolioEnv(feats, market_state, cfg, start, end)
