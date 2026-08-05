"""生产模型:--retrain 用数据起点~最新日全部数据训 PPO;--infer-only 复用现有模型每日推理。
落盘持仓表到 rl-portfolio-allocator-production/data/allocations.parquet。"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import shutil
import pathlib
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from scripts.config import (
    get_config, FACTOR_NAMES, K, STRATEGY_ID, DATA_VERSION, TRAIN_SEEDS
)
from scripts.env import PortfolioEnv, effective_range
from scripts.train import (
    FACTOR_CONTRACT_FIELDS,
    _canonical_contract_payload,
    _normalized_contract_values,
    load_ppo,
    require_factor_contract,
    select_device,
    train_ppo,
    validate_factor_contract,
)
from scripts.walk_forward import frozen_method_id
from scripts.validate import run_all
from scripts.observation import ObservationScaler, collect_training_observations
from scripts.state import STATE_SCHEMA_VERSION, state_fields


def _factor_contract_from_payload(payload: dict | None, *, default_factors=None) -> dict | None:
    if payload is None:
        return None
    source = _canonical_contract_payload(payload)
    normalized = _normalized_contract_values(source)
    explicit_fields = {
        key: normalized[key]
        for key in FACTOR_CONTRACT_FIELDS
        if key in normalized and normalized[key] is not None
    }
    if not explicit_fields:
        return None
    if "selected_factors" in explicit_fields:
        explicit_fields["selected_factors"] = list(explicit_fields["selected_factors"])
    elif default_factors is not None:
        explicit_fields["selected_factors"] = list(default_factors)
    return explicit_fields


def _factor_names_from_payload(payload: dict | None, *, fallback=None) -> list[str]:
    source = _canonical_contract_payload(payload)
    if "selected_factors" in source and source["selected_factors"] is not None:
        return list(source["selected_factors"])
    if "factor_names" in source and source["factor_names"] is not None:
        return list(source["factor_names"])
    if fallback is not None:
        return list(fallback)
    return list(FACTOR_NAMES)


def _runtime_factor_names(cfg: dict, factor_contract: dict | None = None) -> tuple[str, ...]:
    """Resolve runtime names once and reject config/contract disagreement."""
    contract_source = _canonical_contract_payload(factor_contract)
    contract_names = contract_source.get("selected_factors")
    config_names = cfg.get("factor_names")
    if contract_names is not None and config_names is not None:
        if tuple(contract_names) != tuple(config_names):
            raise ValueError("factor contract selected_factors disagree with config")
    names = contract_names if contract_names is not None else config_names
    return tuple(names or FACTOR_NAMES)


def _load_checkpoint_metadata(path, expected_factor_contract: dict | None = None) -> dict:
    metadata = json.loads(pathlib.Path(path).read_text(encoding="utf-8"))
    if expected_factor_contract is not None:
        require_factor_contract(metadata, context="checkpoint metadata")
    if expected_factor_contract is not None:
        validate_factor_contract(metadata, expected_factor_contract)
    return metadata


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
        "comparison_path", "comparison_id", "run_mode", "fold_count", "seed_count",
        "factor_selection_path", "factor_selection_id",
    }
    missing = required - set(data)
    if missing:
        raise ValueError(f"approval missing fields: {sorted(missing)}")
    if (data["run_mode"] != "full" or data["fold_count"] < 3
            or data["seed_count"] < len(TRAIN_SEEDS)):
        raise ValueError("approval missing complete full-run metadata")
    method_path = _approval_reference(approval_path, data["method_path"], "method_path")
    gates_path = _approval_reference(approval_path, data["gates_path"], "gates_path")
    comparison_path = _approval_reference(
        approval_path, data["comparison_path"], "comparison_path"
    )
    if not method_path.exists() or not gates_path.exists() or not comparison_path.exists():
        raise FileNotFoundError("approval references missing method, gates, or comparison")
    method = json.loads(method_path.read_text(encoding="utf-8"))
    method_contract = require_factor_contract(method, context="approved method")
    method_schema = method.get("schema_version", method_contract["state_schema_version"])
    if data["schema_version"] != method_schema:
        raise RuntimeError("approval and method schema mismatch")
    if frozen_method_id(method) != data["method_id"]:
        raise RuntimeError("approved method hash mismatch")
    selection_path = _approval_reference(
        approval_path, data["factor_selection_path"], "factor_selection_path"
    )
    if not selection_path.exists():
        raise FileNotFoundError("approval references missing selected-factor bundle")
    if _file_id(selection_path) != data["factor_selection_id"]:
        raise RuntimeError("approval selected-factor hash mismatch")
    _validate_selected_factor_bundle(selection_path, method_contract)
    gates = json.loads(gates_path.read_text(encoding="utf-8"))
    if gates.get("research_ok") is not True:
        raise RuntimeError("approved gates did not pass")
    comparison_id = _file_id(comparison_path)
    if data["comparison_id"] != comparison_id:
        raise RuntimeError("approval comparison hash mismatch")
    if (gates.get("comparison_path") != data["comparison_path"]
            or gates.get("comparison_id") != comparison_id):
        raise RuntimeError("approved gates are not bound to the comparison evidence")
    comparison = json.loads(comparison_path.read_text(encoding="utf-8"))
    from scripts.research_gates import evaluate_candidate_gates

    if evaluate_candidate_gates(comparison).get("research_ok") is not True:
        raise RuntimeError("comparison evidence did not pass research gates")
    _validate_persisted_paired_evidence(approval_path.parent, comparison)
    return data


def _validate_selected_factor_bundle(path: pathlib.Path, method_contract: dict) -> None:
    """Bind the persisted selection artifact to the approved method contract."""
    payload = json.loads(pathlib.Path(path).read_text(encoding="utf-8"))
    records = payload.get("selected_factors") if isinstance(payload, dict) else None
    if not isinstance(records, list) or not records:
        raise ValueError("selected-factor bundle must contain ordered selected_factors")
    names = []
    directions = []
    for record in records:
        if not isinstance(record, dict) or "name" not in record or "direction" not in record:
            raise ValueError("selected-factor bundle records must contain name and direction")
        names.append(record["name"])
        directions.append(int(record["direction"]))
    if names != list(method_contract["selected_factors"]):
        raise RuntimeError("selected-factor bundle disagrees with approved method")
    if directions != [int(direction) for direction in method_contract["factor_directions"]]:
        raise RuntimeError("selected-factor bundle directions disagree with approved method")
    fold = payload.get("fold")
    if fold is not None and int(fold) != int(method_contract["fold"]):
        raise RuntimeError("selected-factor bundle fold disagrees with approved method")


def _approval_reference(approval_path: pathlib.Path, value, field: str) -> pathlib.Path:
    reference = pathlib.Path(value) if isinstance(value, str) else None
    if reference is None or reference.is_absolute() or ".." in reference.parts:
        raise ValueError(f"approval {field} must be a relative bundle path")
    return approval_path.parent / reference


def _validate_persisted_paired_evidence(run_root: pathlib.Path, comparison: dict) -> None:
    evidence = comparison.get("paired_evidence") if isinstance(comparison, dict) else None
    if not isinstance(evidence, dict):
        raise ValueError("comparison paired evidence is missing")
    expected_branches = ("candidate_20f", "control_6f")
    artifact_paths = set()
    branch_keys = {}
    for branch in expected_branches:
        branch_evidence = evidence.get(branch)
        rows = branch_evidence.get("rows") if isinstance(branch_evidence, dict) else None
        expected_rows = 3 * len(TRAIN_SEEDS)
        if not isinstance(rows, list) or len(rows) != expected_rows:
            raise ValueError(f"comparison {branch} must contain exactly {expected_rows} evidence rows")
        keys = set()
        for row in rows:
            if not isinstance(row, dict):
                raise ValueError("comparison evidence row is invalid")
            fold, seed = row.get("fold"), row.get("seed")
            if (row.get("branch") != branch or not isinstance(fold, int)
                    or isinstance(fold, bool) or not isinstance(seed, int)
                    or isinstance(seed, bool)):
                raise ValueError("comparison evidence branch, fold, or seed is invalid")
            key = (fold, seed)
            if key in keys:
                raise ValueError("comparison evidence has duplicate branch/fold/seed rows")
            keys.add(key)
            artifact = _approval_reference(
                run_root / "approval.json", row.get("stress_artifact_path"),
                "stress_artifact_path",
            )
            relative_artifact = str(artifact.relative_to(run_root))
            if relative_artifact in artifact_paths:
                raise ValueError("comparison evidence stress artifact paths must be distinct")
            if not artifact.is_file():
                raise FileNotFoundError(f"persisted stress artifact missing: {artifact}")
            artifact_paths.add(relative_artifact)
            persisted = json.loads(artifact.read_text(encoding="utf-8"))
            if (persisted.get("branch") != branch or persisted.get("fold") != fold
                    or persisted.get("seed") != seed):
                raise RuntimeError("persisted stress artifact branch/fold/seed mismatch")
            expected_mdd = row.get("stress_mdd")
            actual_mdd = persisted.get("stress_mdd")
            try:
                matches_mdd = math.isfinite(float(expected_mdd)) and math.isfinite(float(actual_mdd)) and math.isclose(
                    float(expected_mdd), float(actual_mdd), rel_tol=0.0, abs_tol=1e-12
                )
            except (TypeError, ValueError):
                matches_mdd = False
            if not matches_mdd:
                raise RuntimeError("persisted stress artifact MDD mismatch")
            expected_hash = row.get("stress_artifact_sha256")
            if not isinstance(expected_hash, str) or not expected_hash:
                raise ValueError("comparison evidence stress artifact hash is missing")
            if _file_id(artifact) != expected_hash:
                raise RuntimeError("stress artifact hash mismatch")
        branch_keys[branch] = keys
    candidate_keys = branch_keys["candidate_20f"]
    if candidate_keys != branch_keys["control_6f"]:
        raise ValueError("comparison evidence branch fold/seed pairs do not match")
    expected_pairs = 3 * len(TRAIN_SEEDS)
    if (len(candidate_keys) != expected_pairs
            or len({fold for fold, _ in candidate_keys}) != 3
            or len({seed for _, seed in candidate_keys}) != len(TRAIN_SEEDS)):
        raise ValueError(f"comparison evidence must contain exactly 3 folds x {len(TRAIN_SEEDS)} seed(s)")
    if len(artifact_paths) != expected_pairs * 2:
        raise ValueError(f"comparison evidence must contain {expected_pairs * 2} distinct persisted stress artifacts")


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
    load_research_approval(source_approval)
    candidate.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_approval, candidate / "approval.json")
    for field in ("method_path", "gates_path", "comparison_path", "factor_selection_path"):
        reference = pathlib.Path(data[field])
        source = _approval_reference(source_approval, data[field], field)
        if not source.exists():
            raise FileNotFoundError(f"approval references missing {field}: {source}")
        target = candidate / reference
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
    comparison = json.loads((source_approval.parent / data["comparison_path"]).read_text(encoding="utf-8"))
    for branch in ("candidate_20f", "control_6f"):
        for row in comparison["paired_evidence"][branch]["rows"]:
            reference = pathlib.Path(row["stress_artifact_path"])
            source = _approval_reference(source_approval, row["stress_artifact_path"], "stress_artifact_path")
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
    approval_data, approved_method = load_approved_method(approval)
    if json.loads(scaler.read_text(encoding="utf-8")).get("schema_version") != STATE_SCHEMA_VERSION:
        raise ValueError("candidate scaler schema mismatch")
    if checkpoint.stat().st_size == 0:
        raise ValueError("candidate checkpoint is empty")
    metadata = candidate / "checkpoint_metadata.json"
    if not metadata.exists():
        raise FileNotFoundError("candidate checkpoint metadata missing")
    expected_contract = require_factor_contract(
        approved_method, context="approved method"
    )
    expected_factor_names = expected_contract["selected_factors"]
    checkpoint_meta = _load_checkpoint_metadata(metadata, expected_contract)
    scaler_data = json.loads(scaler.read_text(encoding="utf-8"))
    require_factor_contract(scaler_data, context="candidate scaler")
    expected_fields = tuple(state_fields(expected_factor_names))
    if (scaler_data.get("schema_version") != STATE_SCHEMA_VERSION
            or tuple(scaler_data.get("fields", ())) != expected_fields
            or len(scaler_data.get("mean", ())) != len(expected_fields)
            or len(scaler_data.get("scale", ())) != len(expected_fields)):
        raise ValueError("candidate scaler schema or fields mismatch")
    validate_factor_contract(scaler_data, expected_contract)
    validate_factor_contract(checkpoint_meta, expected_contract)
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
    copy_approval_bundle(approval, staging)
    publish_files = ("allocations.parquet", "checkpoint.zip", "scaler.json")
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


def _load_production_panels(root, factor_contract: dict, cfg: dict):
    """Materialize exactly the approved ordered factors for production runs.

    The legacy six control factors keep reading the maintained features and
    market_state files; any other approved selection is materialized from the
    family-partitioned factor cache so only those columns are loaded, and the
    market state is rebuilt for the same ordered names.
    """
    contract = require_factor_contract(
        factor_contract, context="production panel factor contract"
    )
    names = list(contract["selected_factors"])
    root = pathlib.Path(root)
    if tuple(names) == tuple(FACTOR_NAMES):
        return (
            pd.read_parquet(root / "data" / "features.parquet"),
            pd.read_parquet(root / "data" / "market_state.parquet"),
        )
    from scripts.factor_cache import materialize_selected_panel
    from scripts.market_state import build_market_state

    records = [
        {"name": name, "direction": int(direction)}
        for name, direction in zip(names, contract["factor_directions"])
    ]
    features = materialize_selected_panel(root / "data" / "factors", records)
    index_returns = pd.read_parquet(root / "data" / "index_returns.parquet")
    market_state = build_market_state(features, index_returns, cfg, factor_names=names)
    return features, market_state


def fit_production_scaler(env, seed: int, factor_names=None) -> ObservationScaler:
    names = tuple(factor_names or FACTOR_NAMES)
    observations = collect_training_observations(env, seed=seed)
    return ObservationScaler.fit(
        observations,
        STATE_SCHEMA_VERSION,
        tuple(state_fields(names)),
    )


def _file_id(path) -> str:
    digest = hashlib.sha256()
    with pathlib.Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def load_production_scaler(path, factor_names=None,
                           factor_contract: dict | None = None) -> ObservationScaler:
    complete_contract = require_factor_contract(
        factor_contract, context="production scaler factor contract"
    )
    contract_source = _canonical_contract_payload(complete_contract)
    contract_names = contract_source.get("selected_factors")
    if (contract_names is not None and factor_names is not None
            and tuple(contract_names) != tuple(factor_names)):
        raise ValueError("factor contract selected_factors disagree with scaler fields")
    names = tuple(contract_names if contract_names is not None
                  else (factor_names or FACTOR_NAMES))
    return ObservationScaler.load(
        path,
        expected_schema=STATE_SCHEMA_VERSION,
        expected_fields=tuple(state_fields(names)),
        expected_factor_contract=complete_contract,
    )


def retrain_production(features_df: pd.DataFrame, cfg: dict, timesteps: int,
                        seed: int, checkpoint_path: str,
                        market_state_df: pd.DataFrame, scaler_path: str | None = None,
                        metadata_path: str | None = None,
                        factor_contract: dict | None = None,
                        method_id: str | None = None) -> str:
    contract = require_factor_contract(
        factor_contract, context="production training factor contract"
    )
    factor_names = _runtime_factor_names(cfg, contract)
    cfg["factor_names"] = list(factor_names)
    cfg["k"] = len(factor_names)
    dates = pd.to_datetime(features_df["trade_date"])
    start, end = effective_range(
        features_df, market_state_df, dates.min(), dates.max(), cfg=cfg
    )
    print(f"effective range: {start.date()} ~ {end.date()}")
    env = PortfolioEnv(features_df, market_state_df, cfg, start, end)
    scaler = fit_production_scaler(env, seed=seed, factor_names=factor_names)
    env.observation_scaler = scaler
    scaler_target = pathlib.Path(scaler_path) if scaler_path else pathlib.Path(checkpoint_path).with_name("scaler.json")
    scaler.save(scaler_target, factor_contract=contract)
    device = select_device(cfg["train_device"])
    metadata_target = pathlib.Path(metadata_path) if metadata_path else pathlib.Path(checkpoint_path).with_name("checkpoint_metadata.json")
    train_ppo(
        env, total_timesteps=timesteps, seed=seed, device=device,
        save_path=checkpoint_path, factor_contract=contract,
        metadata_path=metadata_target,
    )
    metadata_target.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "schema_version": STATE_SCHEMA_VERSION,
        "scaler_path": str(scaler_target),
        "train_range": {"start": str(start), "end": str(end)},
        "seed": seed,
        "timesteps": timesteps,
        "checkpoint_id": _file_id(checkpoint_path),
        "scaler_id": _file_id(scaler_target),
        **contract,
    }
    if method_id is not None:
        metadata["method_id"] = method_id
    metadata_target.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")
    return checkpoint_path


def infer_latest(features_df: pd.DataFrame, cfg: dict, model_path: str,
                 market_state_df: pd.DataFrame, observation_scaler=None,
                 factor_contract: dict | None = None,
                 metadata_path=None) -> pd.DataFrame:
    runtime_contract = require_factor_contract(
        factor_contract, context="inference factor contract"
    )
    factor_names = _runtime_factor_names(cfg, runtime_contract)
    from scripts.observation import ObservationScaler
    expected_fields = tuple(state_fields(factor_names))
    if not isinstance(observation_scaler, ObservationScaler):
        raise ValueError("inference requires a validated observation scaler")
    if (observation_scaler.schema_version != STATE_SCHEMA_VERSION
            or observation_scaler.fields != expected_fields):
        raise ValueError("inference observation scaler does not match factor contract")
    runtime_cfg = dict(cfg)
    runtime_cfg["factor_names"] = list(factor_names)
    runtime_cfg["k"] = len(factor_names)
    dates = pd.to_datetime(features_df["trade_date"]).unique()
    dates = sorted(dates)
    ctx_start = dates[max(0, len(dates) - 60)]
    end = dates[-1]
    ctx_start, end = effective_range(
        features_df, market_state_df, ctx_start, end, cfg=runtime_cfg
    )
    env = PortfolioEnv(
        features_df, market_state_df, runtime_cfg, ctx_start, end,
        observation_scaler=observation_scaler,
    )
    model = load_ppo(
        model_path,
        env,
        expected_factor_contract=runtime_contract,
        metadata_path=metadata_path,
    )

    env.ood_mix_enabled = True
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
    fw_json = json.dumps(dict(zip(factor_names, factor_w.tolist())))
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
    factor_contract = require_factor_contract(
        approved_method, context="approved method"
    )
    approved_factor_names = list(factor_contract["selected_factors"])

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

    cfg.update({
        key: approved_method[key]
        for key in ("reward_variant", "buffer_variant")
        if key in approved_method
    })
    if "factor_names" in approved_method or "selected_factors" in approved_method:
        cfg["factor_names"] = list(approved_factor_names)
        cfg["k"] = len(approved_factor_names)
    if approved_method.get("buffer_config"):
        cfg.update(approved_method["buffer_config"])
    feats, market_state = _load_production_panels(root, factor_contract, cfg)
    if args.retrain:
        retrain_production(
            feats, cfg, args.timesteps, seed=0,
            checkpoint_path=str(ckpt), market_state_df=market_state,
            scaler_path=str(scaler_path) if scaler_path else None,
            metadata_path=str(metadata_path) if metadata_path else None,
            factor_contract=factor_contract,
            method_id=approved["method_id"],
        )
        print(f"candidate checkpoint saved: {ckpt}" if candidate else f"production checkpoint saved: {ckpt}")
    if not ckpt.exists():
        raise SystemExit(f"no production checkpoint at {ckpt}; run --retrain first")
    if candidate is not None and not scaler_path.exists():
        raise SystemExit(f"no candidate scaler at {scaler_path}; candidate is incomplete")
    if candidate is not None and not metadata_path.exists():
        raise SystemExit(
            f"no candidate checkpoint metadata at {metadata_path}; candidate is incomplete"
        )
    metadata_contract = None
    if metadata_path and metadata_path.exists():
        checkpoint_meta = _load_checkpoint_metadata(metadata_path, factor_contract)
        metadata_contract = require_factor_contract(
            checkpoint_meta, context="candidate checkpoint metadata"
        )
    scaler = (
        load_production_scaler(
            scaler_path,
            factor_names=approved_factor_names,
            factor_contract=metadata_contract or factor_contract,
        )
        if scaler_path and scaler_path.exists()
        else None
    )
    allocations = infer_latest(
        feats, cfg, model_path=str(ckpt), market_state_df=market_state,
        observation_scaler=scaler,
        factor_contract=metadata_contract or factor_contract,
        metadata_path=metadata_path,
    )
    save_allocations(allocations, str(out_path))
    print(f"allocations saved: {out_path}  rows={len(allocations)}")


if __name__ == "__main__":
    main()
