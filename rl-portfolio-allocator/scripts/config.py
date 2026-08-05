"""集中式常量与环境变量读取。所有其他脚本从这里取配置,不再直接读 os.environ。"""
from __future__ import annotations
import os

CONTROL_FACTOR_NAMES: list[str] = [
    "mom_20",
    "reversal_5",
    "vol_20",
    "turnover_20",
    "amihud_20",
    "ret_skew_60",
]
FACTOR_NAMES = CONTROL_FACTOR_NAMES  # compatibility during migration
K: int = len(CONTROL_FACTOR_NAMES)

TRAIN_SEEDS: tuple[int, ...] = (0,)

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
LAMBDA_DRAWDOWN: float = 0.5
LAMBDA_TURNOVER: float = 0.05
LAMBDA_CONCENTRATION: float = 0.5
REWARD_SCALE: float = 100.0
REWARD_CLIP: float = 5.0
HHI_TARGET: float = 0.03
TURNOVER_BUDGET: float = 0.20
TARGET_MDD: float = 0.10
DUAL_LR: float = 0.05
RECOVERY_CREDIT: float = 0.1
DOWNSIDE_VOL_COEFF: float = 0.2

EPISODE_MIN_WEEKS: int = 52
EPISODE_MAX_WEEKS: int = 156
CRISIS_OVERSAMPLE_WEIGHT: float = 3.0
ENT_COEF: float = 0.02

STRATEGY_ID: str = "RLPA"
DATA_VERSION: str = "real-v1"


def get_config() -> dict:
    reward_variant = os.environ.get("REWARD_VARIANT", "low")
    if reward_variant not in {"none", "gentle", "low", "medium", "legacy_dsr", "constrained"}:
        raise ValueError(f"unknown reward variant: {reward_variant}")
    return {
        "factor_names": list(CONTROL_FACTOR_NAMES),
        "k": len(CONTROL_FACTOR_NAMES),
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
        "reward_scale": REWARD_SCALE,
        "reward_clip": REWARD_CLIP,
        "hhi_target": HHI_TARGET,
        "turnover_budget": TURNOVER_BUDGET,
        "lambda_drawdown": float(os.environ.get("RLPA_LAMBDA_DRAWDOWN", LAMBDA_DRAWDOWN)),
        "lambda_turnover": float(os.environ.get("RLPA_LAMBDA_TURNOVER", LAMBDA_TURNOVER)),
        "lambda_concentration": float(os.environ.get("RLPA_LAMBDA_CONCENTRATION", LAMBDA_CONCENTRATION)),
        "target_mdd": float(os.environ.get("RLPA_TARGET_MDD", TARGET_MDD)),
        "dual_lr": float(os.environ.get("RLPA_DUAL_LR", DUAL_LR)),
        "recovery_credit": float(os.environ.get("RLPA_RECOVERY_CREDIT", RECOVERY_CREDIT)),
        "downside_vol_coeff": float(os.environ.get("RLPA_DOWNSIDE_VOL_COEFF", DOWNSIDE_VOL_COEFF)),
        "episode_min_weeks": int(os.environ.get("RLPA_EPISODE_MIN_WEEKS", EPISODE_MIN_WEEKS)),
        "episode_max_weeks": int(os.environ.get("RLPA_EPISODE_MAX_WEEKS", EPISODE_MAX_WEEKS)),
        "crisis_oversample_weight": float(os.environ.get("RLPA_CRISIS_OVERSAMPLE_WEIGHT", CRISIS_OVERSAMPLE_WEIGHT)),
        "ent_coef": float(os.environ.get("RLPA_ENT_COEF", ENT_COEF)),
        "selection_target_count": int(os.environ.get("RLPA_SELECTION_TARGET_COUNT", "20")),
        "reward_variant": reward_variant,
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
