"""生产模型:--retrain 用数据起点~最新日全部数据训 PPO;--infer-only 复用现有模型每日推理。
落盘持仓表到 rl-portfolio-allocator-production/data/allocations.parquet。"""
from __future__ import annotations
import argparse
import hashlib
import json
import shutil
import pathlib
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from scripts.config import (
    get_config, FACTOR_NAMES, K, STRATEGY_ID, DATA_VERSION
)
from scripts.env import PortfolioEnv, effective_range
from scripts.train import train_ppo, load_ppo, select_device
from scripts.walk_forward import frozen_method_id
from scripts.validate import run_all
from scripts.observation import ObservationScaler, collect_training_observations
from scripts.state import STATE_SCHEMA_VERSION, state_fields


def load_research_approval(path) -> dict:
    approval_path = pathlib.Path(path)
    if not approval_path.exists():
        raise FileNotFoundError(f"research approval missing: {approval_path}")
    data = json.loads(approval_path.read_text(encoding="utf-8"))
    if data.get("research_ok") is not True:
        raise RuntimeError("research_ok gate did not pass")
    if data.get("run_mode") == "smoke":
        raise RuntimeError("approval must come from a full walk-forward run")
    required = {
        "schema_version", "method_id", "method_path", "gates_path",
        "run_mode", "fold_count", "seed_count",
    }
    missing = required - set(data)
    if missing:
        raise ValueError(f"approval missing fields: {sorted(missing)}")
    if (data["run_mode"] != "full" or data["fold_count"] < 3
            or data["seed_count"] < 5):
        raise ValueError("approval missing complete full-run metadata")
    method_path = approval_path.parent / data["method_path"]
    gates_path = approval_path.parent / data["gates_path"]
    if not method_path.exists() or not gates_path.exists():
        raise FileNotFoundError("approval references missing method or gates")
    method = json.loads(method_path.read_text(encoding="utf-8"))
    if data["schema_version"] != method.get("schema_version"):
        raise RuntimeError("approval and method schema mismatch")
    if frozen_method_id(method) != data["method_id"]:
        raise RuntimeError("approved method hash mismatch")
    gates = json.loads(gates_path.read_text(encoding="utf-8"))
    if gates.get("research_ok") is not True:
        raise RuntimeError("approved gates did not pass")
    return data


def load_approved_method(path) -> tuple[dict, dict]:
    """Return the validated approval and its frozen method configuration."""
    approval = load_research_approval(path)
    approval_path = pathlib.Path(path)
    method_path = approval_path.parent / approval["method_path"]
    return approval, json.loads(method_path.read_text(encoding="utf-8"))


def copy_approval_bundle(approval_path, candidate_dir) -> None:
    """Copy approval and its relative method/gate references into a candidate."""
    source_approval = pathlib.Path(approval_path)
    candidate = pathlib.Path(candidate_dir)
    data = json.loads(source_approval.read_text(encoding="utf-8"))
    candidate.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_approval, candidate / "approval.json")
    for field in ("method_path", "gates_path"):
        reference = pathlib.Path(data[field])
        if reference.is_absolute() or ".." in reference.parts:
            raise ValueError(f"approval {field} must be a relative candidate path")
        source = source_approval.parent / reference
        if not source.exists():
            raise FileNotFoundError(f"approval references missing {field}: {source}")
        target = candidate / reference
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)


def atomic_publish(candidate_dir, production_dir) -> None:
    candidate = pathlib.Path(candidate_dir)
    production = pathlib.Path(production_dir)
    approval = candidate / "approval.json"
    if not approval.exists():
        raise FileNotFoundError("candidate approval missing")
    scaler = candidate / "scaler.json"
    checkpoint = candidate / "checkpoint.zip"
    allocations = candidate / "allocations.parquet"
    if not scaler.exists() or not checkpoint.exists() or not allocations.exists():
        raise FileNotFoundError("candidate allocations, scaler, or checkpoint missing")
    load_research_approval(approval)
    if json.loads(scaler.read_text(encoding="utf-8")).get("schema_version") != "state-v1":
        raise ValueError("candidate scaler schema mismatch")
    if checkpoint.stat().st_size == 0:
        raise ValueError("candidate checkpoint is empty")
    metadata = candidate / "checkpoint_metadata.json"
    if not metadata.exists():
        raise FileNotFoundError("candidate checkpoint metadata missing")
    checkpoint_meta = json.loads(metadata.read_text(encoding="utf-8"))
    approval_data = json.loads(approval.read_text(encoding="utf-8"))
    scaler_data = json.loads(scaler.read_text(encoding="utf-8"))
    expected_fields = tuple(state_fields(FACTOR_NAMES))
    if (scaler_data.get("schema_version") != STATE_SCHEMA_VERSION
            or tuple(scaler_data.get("fields", ())) != expected_fields
            or len(scaler_data.get("mean", ())) != len(expected_fields)
            or len(scaler_data.get("scale", ())) != len(expected_fields)):
        raise ValueError("candidate scaler schema or fields mismatch")
    if checkpoint_meta.get("schema_version") != approval_data["schema_version"]:
        raise ValueError("candidate checkpoint metadata schema mismatch")
    if checkpoint_meta.get("method_id") != approval_data["method_id"]:
        raise ValueError("candidate checkpoint method mismatch")
    if checkpoint_meta.get("checkpoint_id") != _file_id(checkpoint):
        raise ValueError("candidate checkpoint hash mismatch")
    if checkpoint_meta.get("scaler_id") != _file_id(scaler):
        raise ValueError("candidate scaler hash mismatch")
    ok, errors = run_all(str(allocations), get_config())
    if not ok:
        raise ValueError("candidate allocations failed validation: " + "; ".join(errors))
    production.parent.mkdir(parents=True, exist_ok=True)
    staging = production.parent / (production.name + ".staging")
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir()
    publish_files = ("allocations.parquet", "checkpoint.zip", "scaler.json", "approval.json")
    for name in publish_files:
        shutil.copy2(candidate / name, staging / name)
    if metadata.exists():
        shutil.copy2(metadata, staging / metadata.name)
    # Publish a complete staged bundle. The pointer is replaced only after every
    # artifact has been copied and validated, so readers never observe a partial
    # candidate bundle through CURRENT.
    pointer = production.parent / (production.name + ".CURRENT")
    pointer_tmp = production.parent / (production.name + ".CURRENT.tmp")
    pointer_tmp.write_text(production.name, encoding="utf-8")
    pointer_tmp.replace(pointer)
    backup = production.parent / (production.name + ".previous")
    if backup.exists():
        shutil.rmtree(backup)
    try:
        if production.exists():
            production.replace(backup)
        staging.replace(production)
    except Exception:
        if production.exists():
            shutil.rmtree(production)
        if backup.exists():
            backup.replace(production)
        raise
    if backup.exists():
        shutil.rmtree(backup)


def fit_production_scaler(env, seed: int) -> ObservationScaler:
    observations = collect_training_observations(env, seed=seed)
    return ObservationScaler.fit(
        observations,
        STATE_SCHEMA_VERSION,
        tuple(state_fields(FACTOR_NAMES)),
    )


def _file_id(path) -> str:
    digest = hashlib.sha256()
    with pathlib.Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def load_production_scaler(path) -> ObservationScaler:
    return ObservationScaler.load(
        path,
        expected_schema=STATE_SCHEMA_VERSION,
        expected_fields=tuple(state_fields(FACTOR_NAMES)),
    )


def retrain_production(features_df: pd.DataFrame, cfg: dict, timesteps: int,
                        seed: int, checkpoint_path: str,
                        market_state_df: pd.DataFrame, scaler_path: str | None = None,
                        metadata_path: str | None = None,
                        method_id: str | None = None) -> str:
    dates = pd.to_datetime(features_df["trade_date"])
    start, end = effective_range(features_df, market_state_df, dates.min(), dates.max())
    print(f"effective range: {start.date()} ~ {end.date()}")
    env = PortfolioEnv(features_df, market_state_df, cfg, start, end)
    scaler = fit_production_scaler(env, seed=seed)
    env.observation_scaler = scaler
    scaler_target = pathlib.Path(scaler_path) if scaler_path else pathlib.Path(checkpoint_path).with_name("scaler.json")
    scaler.save(scaler_target)
    device = select_device(cfg["train_device"])
    train_ppo(env, total_timesteps=timesteps, seed=seed, device=device, save_path=checkpoint_path)
    metadata_target = pathlib.Path(metadata_path) if metadata_path else pathlib.Path(checkpoint_path).with_name("checkpoint_metadata.json")
    metadata_target.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "schema_version": STATE_SCHEMA_VERSION,
        "scaler_path": str(scaler_target),
        "train_range": {"start": str(start), "end": str(end)},
        "seed": seed,
        "timesteps": timesteps,
        "checkpoint_id": _file_id(checkpoint_path),
        "scaler_id": _file_id(scaler_target),
    }
    if method_id is not None:
        metadata["method_id"] = method_id
    metadata_target.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")
    return checkpoint_path


def infer_latest(features_df: pd.DataFrame, cfg: dict, model_path: str,
                 market_state_df: pd.DataFrame, observation_scaler=None) -> pd.DataFrame:
    dates = pd.to_datetime(features_df["trade_date"]).unique()
    dates = sorted(dates)
    ctx_start = dates[max(0, len(dates) - 60)]
    end = dates[-1]
    ctx_start, end = effective_range(features_df, market_state_df, ctx_start, end)
    env = PortfolioEnv(
        features_df, market_state_df, cfg, ctx_start, end,
        observation_scaler=observation_scaler,
    )
    model = load_ppo(model_path, env)

    obs, _ = env.reset(seed=0)
    last_info = None
    last_target_w = None
    last_symbols = env.symbols
    done = False
    while not done:
        act, _ = model.predict(obs, deterministic=True)
        obs, _, term, trunc, info = env.step(act)
        last_info = info
        last_target_w = env.prev_stock_w.copy()
        done = term or trunc

    if last_info is None:
        raise RuntimeError("env produced no step; features_df too short")

    factor_w = np.asarray(last_info["factor_w"], dtype=float)
    scores = env._F_by_date[env.dates[env.t - 1]] @ factor_w

    # Normalize weights to respect caps (HARD CONSTRAINT - no exceptions)
    weights = last_target_w.copy()
    long_sum = float(np.clip(weights, 0, None).sum())
    short_sum = float(np.clip(-weights, 0, None).sum())
    long_cap = cfg["long_notional"]
    short_cap = cfg["short_notional_cap"]

    # Scale down if exceeds (always enforce, never allow violation)
    if long_sum > long_cap * 1.0001:  # 0.01% tolerance only
        scale_factor = long_cap / long_sum
        weights = weights * (weights > 0) * scale_factor + weights * (weights <= 0)
    if short_sum > short_cap * 1.0001:  # 0.01% tolerance only
        scale_factor = short_cap / short_sum
        weights = weights * (weights < 0) * scale_factor + weights * (weights >= 0)

    # Verify constraint
    final_long = float(np.clip(weights, 0, None).sum())
    final_short = float(np.clip(-weights, 0, None).sum())
    if final_long > long_cap * 1.001 or final_short > short_cap * 1.001:
        raise RuntimeError(f"Weight constraint FAILED: long={final_long}, short={final_short}")

    now = datetime.now(timezone.utc).isoformat()
    rows = []
    fw_json = json.dumps(dict(zip(FACTOR_NAMES, factor_w.tolist())))
    trade_date = pd.Timestamp(end).normalize()
    for i, s in enumerate(last_symbols):
        w = float(weights[i])
        if abs(w) < 1e-9:
            continue
        rows.append({
            "trade_date": trade_date, "symbol": s, "weight": w,
            "side": "long" if w > 0 else "short",
            "factor_weights": fw_json,
            "composite_score": float(scores[i]),
            "strategy_id": STRATEGY_ID, "data_version": DATA_VERSION,
            "update_time": now,
        })
    cash = 1.0 - float(np.clip(weights, 0, None).sum()) - float(np.clip(-weights, 0, None).sum())
    if abs(cash) > 1e-9:
        rows.append({
            "trade_date": trade_date, "symbol": "CASH", "weight": cash, "side": "cash",
            "factor_weights": fw_json, "composite_score": 0.0,
            "strategy_id": STRATEGY_ID, "data_version": DATA_VERSION, "update_time": now,
        })
    return pd.DataFrame(rows)


def save_allocations(df: pd.DataFrame, path: str) -> None:
    p = pathlib.Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    if p.exists():
        old = pd.read_parquet(p)
        # 一个 trade_date 的持仓是一个整体组合(多空各有 cap),必须整体覆盖:
        # 丢弃旧文件里与新数据同 trade_date 的所有行,避免上一批被淘汰的
        # symbol 残留、破坏组合加总约束。
        new_dates = set(df["trade_date"].unique())
        old = old[~old["trade_date"].isin(new_dates)]
        combined = pd.concat([old, df], ignore_index=True)
    else:
        combined = df
    combined.to_parquet(p, index=False)


def main() -> None:
    cfg = get_config()
    root = pathlib.Path(__file__).resolve().parent.parent
    feats_path = root / "data" / "features.parquet"
    market_state_path = root / "data" / "market_state.parquet"
    formal_ckpt = root / "checkpoints" / "production.zip"
    formal_out = root.parent / "rl-portfolio-allocator-production" / "data" / "allocations.parquet"

    p = argparse.ArgumentParser()
    grp = p.add_mutually_exclusive_group(required=True)
    grp.add_argument("--retrain", action="store_true", help="用数据起点~最新日全部数据重训生产模型")
    grp.add_argument("--infer-only", action="store_true", help="复用现有生产模型仅推理当日持仓")
    p.add_argument("--timesteps", type=int, default=200_000)
    p.add_argument("--approval", required=True,
                   help="research approval.json; production execution fails closed without it")
    p.add_argument(
        "--candidate-dir",
        help="write retrain/inference artifacts to this candidate bundle",
    )
    args = p.parse_args()

    approved, approved_method = load_approved_method(args.approval)

    candidate = pathlib.Path(args.candidate_dir) if args.candidate_dir else None
    if candidate is None:
        raise SystemExit("production execution requires --candidate-dir; formal paths are never written")
    if candidate is not None:
        candidate.mkdir(parents=True, exist_ok=True)
        ckpt = candidate / "checkpoint.zip"
        scaler_path = candidate / "scaler.json"
        metadata_path = candidate / "checkpoint_metadata.json"
        out_path = candidate / "allocations.parquet"
        copy_approval_bundle(args.approval, candidate)
    else:
        ckpt = formal_ckpt
        scaler_path = None
        metadata_path = None
        out_path = formal_out

    feats = pd.read_parquet(feats_path)
    market_state = pd.read_parquet(market_state_path)
    cfg.update({
        key: approved_method[key]
        for key in ("reward_variant", "buffer_variant")
        if key in approved_method
    })
    if approved_method.get("buffer_config"):
        cfg.update(approved_method["buffer_config"])
    if args.retrain:
        retrain_production(
            feats, cfg, args.timesteps, seed=0,
            checkpoint_path=str(ckpt), market_state_df=market_state,
            scaler_path=str(scaler_path) if scaler_path else None,
            metadata_path=str(metadata_path) if metadata_path else None,
            method_id=approved["method_id"],
        )
        print(f"candidate checkpoint saved: {ckpt}" if candidate else f"production checkpoint saved: {ckpt}")
    if not ckpt.exists():
        raise SystemExit(f"no production checkpoint at {ckpt}; run --retrain first")
    if candidate is not None and not scaler_path.exists():
        raise SystemExit(f"no candidate scaler at {scaler_path}; candidate is incomplete")
    scaler = load_production_scaler(scaler_path) if scaler_path and scaler_path.exists() else None
    allocations = infer_latest(
        feats, cfg, model_path=str(ckpt), market_state_df=market_state,
        observation_scaler=scaler,
    )
    save_allocations(allocations, str(out_path))
    print(f"allocations saved: {out_path}  rows={len(allocations)}")


if __name__ == "__main__":
    main()
