"""集中式常量与环境变量读取。所有其他脚本从这里取配置,不再直接读 os.environ。"""
from __future__ import annotations
import os

FACTOR_NAMES: list[str] = [
    "mom_20",
    "reversal_5",
    "vol_20",
    "turnover_20",
    "amihud_20",
    "ret_skew_60",
]
K: int = len(FACTOR_NAMES)

TOP_N: int = 30
BOTTOM_M: int = 15
LONG_NOTIONAL: float = 1.0
SHORT_NOTIONAL_CAP: float = 0.30

COMMISSION_BPS: float = 3.0
STAMP_TAX_BPS: float = 10.0
IMPACT_BPS: float = 5.0
BORROW_RATE_ANNUAL: float = 0.08
TRADING_DAYS_PER_YEAR: int = 252

EMA_ALPHA: float = 0.5

DSR_ETA: float = 0.01
LAMBDA_DRAWDOWN: float = 0.005
LAMBDA_TURNOVER: float = 0.002
LAMBDA_CONCENTRATION: float = 0.02

STRATEGY_ID: str = "RLPA"
DATA_VERSION: str = "real-v1"


def get_config() -> dict:
    return {
        "factor_names": FACTOR_NAMES,
        "k": K,
        "top_n": TOP_N,
        "bottom_m": BOTTOM_M,
        "long_notional": LONG_NOTIONAL,
        "short_notional_cap": SHORT_NOTIONAL_CAP,
        "commission_bps": COMMISSION_BPS,
        "stamp_tax_bps": STAMP_TAX_BPS,
        "impact_bps": IMPACT_BPS,
        "borrow_rate_annual": BORROW_RATE_ANNUAL,
        "trading_days_per_year": TRADING_DAYS_PER_YEAR,
        "ema_alpha": EMA_ALPHA,
        "dsr_eta": DSR_ETA,
        "lambda_drawdown": LAMBDA_DRAWDOWN,
        "lambda_turnover": LAMBDA_TURNOVER,
        "lambda_concentration": LAMBDA_CONCENTRATION,
        "reward_ret_weight": float(os.environ.get("REWARD_RET_WEIGHT", "1.0")),
        "strategy_id": STRATEGY_ID,
        "data_version": DATA_VERSION,
        "panda_username": os.environ.get("PANDA_DATA_USERNAME"),
        "panda_password": os.environ.get("PANDA_DATA_PASSWORD"),
        "start_date": os.environ.get("PANDA_DATA_START_DATE", "2004-01-01"),
        "end_date": os.environ.get("PANDA_DATA_END_DATE", "2024-12-31"),
        "rl_algo": os.environ.get("RL_ALGO", "ppo"),
        "reward_type": os.environ.get("REWARD_TYPE", "sharpe"),
        "train_device": os.environ.get("TRAIN_DEVICE", "auto"),
        "retrain_cadence": os.environ.get("RETRAIN_CADENCE", "monthly"),
    }
