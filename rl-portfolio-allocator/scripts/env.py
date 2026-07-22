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
from scripts.state import StateBuilder, state_dim


class PortfolioEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(
        self,
        features_df: pd.DataFrame,
        index_returns: pd.Series,
        cfg: dict,
        start_date,
        end_date,
        nonlinear_impact: bool = False,
    ):
        super().__init__()
        self.cfg = cfg
        self.nonlinear_impact = nonlinear_impact
        df = features_df.copy()
        df["trade_date"] = pd.to_datetime(df["trade_date"])
        df = df[(df["trade_date"] >= pd.Timestamp(start_date)) & (df["trade_date"] <= pd.Timestamp(end_date))]
        df = df.sort_values(["trade_date", "symbol"]).reset_index(drop=True)
        self.features = df

        self.dates = sorted(df["trade_date"].unique())
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

        self.index_returns = index_returns.sort_index()

        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(K,), dtype=np.float32)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(state_dim(K),), dtype=np.float32,
        )
        self._reset_internal()

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
            index_returns=self.index_returns,
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
        return obs, {}

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
        hhi_v = hhi(target_w)
        dsr_delta = self.dsr.update(net, self.cfg["dsr_eta"], sortino=(self.cfg["reward_type"] == "sortino"))
        drawdown = 0.0 if self.dsr.peak <= 0 else (self.dsr.peak - self.dsr.nav) / self.dsr.peak
        reward, parts = compose_reward(dsr_delta, drawdown, turnover, hhi_v, self.cfg)

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
            obs = np.zeros(state_dim(K), dtype=np.float32)

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
        return obs, float(reward), terminated, truncated, info


def make_env(features_path, index_returns_path, cfg, start, end) -> PortfolioEnv:
    feats = pd.read_parquet(features_path)
    idx = pd.read_parquet(index_returns_path)
    idx = pd.Series(idx["ret"].values, index=pd.to_datetime(idx["trade_date"]))
    return PortfolioEnv(feats, idx, cfg, start, end)
