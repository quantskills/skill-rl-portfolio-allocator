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
from scripts.action_transform import transform_action
from scripts.costs import total_costs
from scripts.portfolio import (
    composite_score, select_long_short, target_weights, freeze_suspended
)
from scripts.reward import DSRState, hhi, compose_reward
from scripts.state import StateBuilder, exogenous_fields, state_dim


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

        self.dates = list(dates)
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
            self.dates[self.t], self.prev_stock_w, FACTOR_NAMES,
            self.prev_factor_w, cash=1.0,
        )
        return self._scale(obs), {}

    def step(self, action: np.ndarray):
        d = self.dates[self.t]
        F = self._F_by_date[d]
        susp = self._susp_by_date[d]

        factor_w = transform_action(np.asarray(action, dtype=float), self.prev_factor_w, self.cfg["ema_alpha"])
        scores = composite_score(F, factor_w)
        long_idx, short_idx = select_long_short(scores, susp, self.cfg["top_n"], self.cfg["bottom_m"])
        target_w = target_weights(scores, long_idx, short_idx, self.cfg["long_notional"], self.cfg["short_notional_cap"])
        target_w = freeze_suspended(target_w, self.prev_stock_w, susp)

        costs = total_costs(self.prev_stock_w, target_w, self.cfg)

        next_t = self.t + 1
        terminated = False
        truncated = False
        if next_t >= len(self.dates):
            gross = 0.0
            terminated = True
        else:
            r_next = self._ret_by_date[self.dates[next_t]]
            gross = float(target_w @ r_next)

        net = gross - costs["total"]
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
        reward, parts = compose_reward(
            dsr_delta, drawdown, turnover, hhi_v, self.cfg,
            net_ret=net,
            long_notional=long_notional, short_notional=short_notional,
            long_cap=self.cfg["long_notional"], short_cap=self.cfg["short_notional_cap"]
        )

        self.state_builder.portfolio_returns_history.append(net)
        self.state_builder.recent_turnover_history.append(turnover)

        self.prev_factor_w = factor_w
        self.prev_stock_w = target_w
        self.t = next_t

        if not terminated:
            obs = self.state_builder.build(
                self.dates[self.t], self.prev_stock_w, FACTOR_NAMES,
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
            "hhi": hhi_v,
            "dsr": dsr_delta, "drawdown": drawdown,
            "reward_parts": parts,
            "ret_source": "t_plus_1",
        }
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
