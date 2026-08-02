"""Leakage-safe walk-forward orchestration for the research-only OOS report."""
from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import sys
import uuid
from collections import defaultdict
from statistics import median
from typing import Callable

if __package__ in (None, ""):
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from scripts.check_data_coverage import Fold, check_folds, default_folds
from scripts.research_gates import evaluate_research_gates


def frozen_method_id(method: dict) -> str:
    payload = json.dumps(method, sort_keys=True, separators=(",", ":")).encode()
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def write_approval(run_root: pathlib.Path, method: dict, gate_report: dict,
                   run_id: str, created_at: str, *, run_mode: str = "full",
                   fold_count: int | None = None, seed_count: int | None = None):
    if gate_report.get("research_ok") is not True:
        return None
    run_root.mkdir(parents=True, exist_ok=True)
    method_path = run_root / "method.json"
    gates_path = run_root / "gates.json"
    method_path.write_text(json.dumps(_jsonable(method), indent=2, sort_keys=True), encoding="utf-8")
    gates_path.write_text(json.dumps(_jsonable(gate_report), indent=2, sort_keys=True), encoding="utf-8")
    approval = {
        "research_ok": True, "schema_version": method.get("schema_version", "state-v1"),
        "method_id": frozen_method_id(method), "method_path": "method.json",
        "gates_path": "gates.json", "run_id": run_id, "created_at": created_at,
        "run_mode": run_mode, "fold_count": fold_count, "seed_count": seed_count,
    }
    path = run_root / "approval.json"
    path.write_text(json.dumps(approval, indent=2, sort_keys=True), encoding="utf-8")
    return path


SEEDS = (0, 1, 2, 3, 4)
REWARD_CANDIDATES = ("none", "low", "medium", "legacy_dsr")
BUFFER_CANDIDATES = ("tight", "default", "wide")
BUFFER_CONFIGS = {
    "tight": {"long_entry": 30, "long_exit": 40, "short_entry": 15, "short_exit": 25},
    "default": {"long_entry": 30, "long_exit": 45, "short_entry": 15, "short_exit": 30},
    "wide": {"long_entry": 30, "long_exit": 60, "short_entry": 15, "short_exit": 45},
}


def select_candidate_on_validation(rows):
    """Select the highest-median candidate without inspecting test rows."""
    scores = defaultdict(list)
    for row in rows:
        scores[row["candidate"]].append(float(row["val_sharpe"]))
    if not scores:
        raise ValueError("validation rows must not be empty")
    return max(scores, key=lambda candidate: (median(scores[candidate]), -list(scores).index(candidate)))


def _jsonable(value):
    if hasattr(value, "tolist"):
        return _jsonable(value.tolist())
    if isinstance(value, pathlib.Path):
        return str(value)
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    return value


def _write(path: pathlib.Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(payload), ensure_ascii=False, indent=2), encoding="utf-8")


def _invoke_trainer(trainer: Callable, **kwargs) -> dict:
    result = trainer(**kwargs)
    if "val_sharpe" not in result:
        raise ValueError("trainer result must contain val_sharpe")
    return result


def aggregate_research_summary(test_rows, total_folds: int) -> dict:
    """Aggregate supplied per-test metrics without inventing missing evidence."""
    def median_field(name):
        values = [float(row[name]) for row in test_rows if row.get(name) is not None]
        return median(values) if values else None
    by_fold = defaultdict(list)
    for row in test_rows:
        by_fold[row.get("fold")].append(row)
    positive_folds = 0
    for rows in by_fold.values():
        excess = [float(row["excess_return"]) for row in rows if row.get("excess_return") is not None]
        if excess and median(excess) > 0:
            positive_folds += 1
    return {
        "combined_oos_arr": median_field("oos_arr"),
        "median_seed_oos_sharpe": median_field("oos_sharpe"),
        "strongest_baseline_sharpe": median_field("strongest_baseline_sharpe"),
        "positive_excess_return_folds": positive_folds,
        "total_folds": total_folds,
        "median_seed_excess_return": median_field("excess_return"),
        "oos_mdd": median_field("oos_mdd"),
        "strongest_baseline_mdd": median_field("strongest_baseline_mdd"),
        "annualized_turnover": median_field("annualized_turnover"),
        "cost_2x_oos_sharpe": median_field("cost_2x_oos_sharpe"),
        "no_leakage_tests_passed": all(row.get("no_leakage_tests_passed") is True for row in test_rows) if test_rows else None,
        "state_quality_tests_passed": all(row.get("state_quality_tests_passed") is True for row in test_rows) if test_rows else None,
    }


def run_walk_forward(*, folds=None, output_root, smoke=False, trainer=None, tester=None,
                     coverage_checker=None, features_df=None, index_df=None,
                     cfg=None, timesteps=None, run_id=None) -> dict:
    """Run validation-only candidate selection followed by one frozen test per seed."""
    folds = list(folds or default_folds())
    if coverage_checker is not None:
        coverage_checker()
    elif features_df is not None and index_df is not None:
        report = check_folds(features_df, index_df, folds)
        if len(report["usable_folds"]) != len(folds):
            raise ValueError("data coverage check rejected one or more walk-forward folds")
    if trainer is None or tester is None:
        raise ValueError("trainer and tester dependency injections are required")

    selected_folds = [folds[-1]] if smoke else folds
    selected_seeds = (0,) if smoke else SEEDS
    root = pathlib.Path(output_root)
    run_id = "smoke" if smoke else (run_id or uuid.uuid4().hex)
    run_root = root / "smoke" if smoke else root / run_id
    validation_root = run_root / "validation"
    test_root = run_root / "test"
    validation_rows = []
    reward_results = {}
    selected_rewards = {}

    # Phase 1: default rank buffer, reward ablation. Tests are impossible here.
    for fold in selected_folds:
        for reward in REWARD_CANDIDATES:
            for seed in selected_seeds:
                result = _invoke_trainer(
                    trainer, stage="reward_ablation", fold=fold.fold, seed=seed,
                    candidate=reward, reward_variant=reward, buffer_variant="default",
                    buffer_config=BUFFER_CONFIGS["default"], train_range=fold.train,
                    val_range=fold.val, test_range=fold.test, timesteps=timesteps or (128 if smoke else 100_000),
                    cfg=cfg, artifact_dir=validation_root / f"fold{fold.fold}" / reward / f"seed{seed}",
                    features_df=features_df, market_state_df=index_df,
                )
                row = {"fold": fold.fold, "seed": seed, "candidate": reward,
                       "val_sharpe": float(result["val_sharpe"]), "stage": "reward_ablation",
                       "trainer_result": _jsonable(result)}
                reward_results[(fold.fold, seed, reward)] = result
                validation_rows.append(row)
                _write(validation_root / f"fold{fold.fold}" / "reward" / f"{reward}_seed{seed}.json", row)
        selected_rewards[fold.fold] = select_candidate_on_validation(
            [row for row in validation_rows if row["fold"] == fold.fold]
        )

    # Phase 2: buffer ablation with the reward choice frozen by validation only.
    buffer_rows = []
    buffer_results = {}
    selected_buffers = {}
    for fold in selected_folds:
        selected_reward = selected_rewards[fold.fold]
        for buffer in BUFFER_CANDIDATES:
            for seed in selected_seeds:
                result = _invoke_trainer(
                    trainer, stage="buffer_ablation", fold=fold.fold, seed=seed,
                    candidate=buffer, reward_variant=selected_reward, buffer_variant=buffer,
                    buffer_config=BUFFER_CONFIGS[buffer], train_range=fold.train,
                    val_range=fold.val, test_range=fold.test, timesteps=timesteps or (128 if smoke else 100_000),
                    cfg=cfg, artifact_dir=validation_root / f"fold{fold.fold}" / buffer / f"seed{seed}",
                    features_df=features_df, market_state_df=index_df,
                )
                row = {"fold": fold.fold, "seed": seed, "candidate": buffer,
                       "val_sharpe": float(result["val_sharpe"]), "stage": "buffer_ablation",
                       "reward_variant": selected_reward, "trainer_result": _jsonable(result)}
                buffer_results[(fold.fold, seed, buffer)] = result
                buffer_rows.append(row)
                _write(validation_root / f"fold{fold.fold}" / "buffer" / f"{buffer}_seed{seed}.json", row)
        selected_buffers[fold.fold] = select_candidate_on_validation(
            [row for row in buffer_rows if row["fold"] == fold.fold]
        )

    # Phase 3: freeze the method, then evaluate each seed exactly once on test.
    test_rows = []
    for fold in selected_folds:
        selected_reward = selected_rewards[fold.fold]
        selected_buffer = selected_buffers[fold.fold]
        frozen_candidate = f"{selected_reward}__{selected_buffer}"
        for seed in selected_seeds:
            validation_result = buffer_results[(fold.fold, seed, selected_buffer)]
            checkpoint_path = validation_result.get("checkpoint_path")
            result = tester(
                fold=fold.fold, seed=seed, candidate=frozen_candidate,
                reward_variant=selected_reward, buffer_variant=selected_buffer,
                buffer_config=BUFFER_CONFIGS[selected_buffer], train_range=fold.train,
                val_range=fold.val, test_range=fold.test, cfg=cfg,
                artifact_dir=test_root / f"fold{fold.fold}" / frozen_candidate / f"seed{seed}",
                validation_result=validation_result, checkpoint_path=checkpoint_path,
                features_df=features_df, market_state_df=index_df,
            )
            row = {"fold": fold.fold, "seed": seed, "candidate": frozen_candidate,
                   "reward_variant": selected_reward, "buffer_variant": selected_buffer, **result}
            test_rows.append(row)
            _write(test_root / f"fold{fold.fold}" / frozen_candidate / f"seed{seed}.json", row)

    summary = {
        "run_id": run_id, "publishable": False if smoke else False,
        "selected_reward": selected_rewards if not smoke else selected_rewards[selected_folds[0].fold],
        "selected_buffer": selected_buffers if not smoke else selected_buffers[selected_folds[0].fold],
        "frozen_candidate": {
            str(fold.fold): f"{selected_rewards[fold.fold]}__{selected_buffers[fold.fold]}"
            for fold in selected_folds
        }, "validation": validation_rows + buffer_rows,
        "test": test_rows,
    }
    method_by_fold = {
        str(fold.fold): {
            "frozen_candidate": f"{selected_rewards[fold.fold]}__{selected_buffers[fold.fold]}",
            "schema_version": (cfg or {}).get("schema_version", "state-v1"),
            "reward_variant": selected_rewards[fold.fold],
            "buffer_variant": selected_buffers[fold.fold],
            "buffer_config": BUFFER_CONFIGS[selected_buffers[fold.fold]],
            "training_budget": timesteps or (128 if smoke else 100_000),
        }
        for fold in selected_folds
    }
    summary["method_by_fold"] = method_by_fold
    if smoke:
        summary["frozen_method"] = method_by_fold[str(selected_folds[0].fold)]
    research_summary = aggregate_research_summary(test_rows, len(selected_folds))
    if smoke:
        research_summary["no_leakage_tests_passed"] = False
        research_summary["state_quality_tests_passed"] = False
    gate_report = evaluate_research_gates(research_summary)
    _write(run_root / "research_summary.json", research_summary)
    _write(run_root / "gates.json", gate_report)
    if not smoke:
        method = method_by_fold.get(str(selected_folds[0].fold), {})
        write_approval(
            run_root, method, gate_report, run_id, "", run_mode="full",
            fold_count=len(selected_folds), seed_count=len(selected_seeds),
        )
    summary["research_summary"] = research_summary
    summary["gates"] = gate_report
    summary["publishable"] = bool(not smoke and gate_report["research_ok"])
    _write(run_root / "summary.json", summary)
    return summary


def _default_trainer(**kwargs) -> dict:
    """Small production adapter; tests can replace it through injection."""
    if kwargs.get("features_df") is None or kwargs.get("market_state_df") is None:
        raise ValueError("default trainer requires features_df and market_state_df")
    from scripts.env import PortfolioEnv
    from scripts.metrics import sharpe
    from scripts.train import train_ppo, select_device

    cfg = dict(kwargs.get("cfg") or {})
    cfg["reward_variant"] = kwargs["reward_variant"]
    cfg.update(kwargs["buffer_config"])
    env = PortfolioEnv(kwargs["features_df"], kwargs["market_state_df"], cfg,
                       *kwargs["train_range"])
    artifact_dir = pathlib.Path(kwargs["artifact_dir"])
    checkpoint_path = artifact_dir / "best.zip"
    model = train_ppo(env, total_timesteps=kwargs["timesteps"], seed=kwargs["seed"],
                      save_path=str(checkpoint_path),
                      device=select_device(cfg.get("train_device", "auto")))
    val_env = PortfolioEnv(kwargs["features_df"], kwargs["market_state_df"], cfg,
                           *kwargs["val_range"])
    obs, _ = val_env.reset(seed=kwargs["seed"])
    returns = []
    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, _, terminated, truncated, info = val_env.step(action)
        returns.extend(info["daily_net_rets"])
        done = terminated or truncated
    if not checkpoint_path.exists():
        raise ValueError(f"default trainer did not produce checkpoint: {checkpoint_path}")
    return {"val_sharpe": float(sharpe(returns)), "checkpoint_path": str(checkpoint_path),
            "schema_version": cfg.get("schema_version"), "training_budget": kwargs["timesteps"]}


def _default_tester(**kwargs) -> dict:
    checkpoint_path = kwargs.get("checkpoint_path")
    if not checkpoint_path or not pathlib.Path(checkpoint_path).exists():
        raise ValueError("frozen validation checkpoint is required; tester will not retrain")
    from scripts.env import PortfolioEnv
    from scripts.metrics import metrics_pack
    from scripts.train import load_ppo
    cfg = dict(kwargs.get("cfg") or {})
    cfg["reward_variant"] = kwargs["reward_variant"]
    cfg.update(kwargs["buffer_config"])
    env = PortfolioEnv(kwargs["features_df"], kwargs["market_state_df"], cfg,
                       *kwargs["test_range"])
    model = load_ppo(str(checkpoint_path), env)
    obs, _ = env.reset(seed=kwargs["seed"])
    returns = []
    gross_returns = []
    turnovers = []
    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, _, terminated, truncated, info = env.step(action)
        returns.extend(info["daily_net_rets"])
        gross_returns.extend(info.get("daily_gross_rets", ()))
        turnovers.append(float(info.get("turnover", 0.0)))
        done = terminated or truncated
    metrics = metrics_pack(returns, "test")
    gross = list(gross_returns)
    net = list(returns)
    doubled_cost_returns = [2.0 * n - g for n, g in zip(net, gross)]
    cost_2x = metrics_pack(doubled_cost_returns, "test_cost_2x")
    return {
        "test_sharpe": metrics["sharpe"],
        "oos_sharpe": metrics["sharpe"],
        "oos_arr": metrics["arr"],
        "oos_mdd": metrics["mdd"],
        "daily_returns": returns,
        "annualized_turnover": (sum(turnovers) / max(len(turnovers), 1)) * 252.0,
        "cost_2x_oos_sharpe": cost_2x["sharpe"],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--full", action="store_true")
    parser.add_argument("--output-root", default="artifacts/walk_forward")
    parser.add_argument("--run-id", default=None,
                        help="explicit full-run directory name for pipeline integration")
    parser.add_argument("--timesteps", type=int, default=None)
    args = parser.parse_args()
    if args.smoke and args.full:
        parser.error("choose only one of --smoke or --full")
    if not args.smoke and not args.full:
        parser.error("one of --smoke or --full is required")
    root = pathlib.Path(__file__).resolve().parent.parent
    features_path = root / "data" / "features.parquet"
    market_state_path = root / "data" / "market_state.parquet"
    if not features_path.exists() or not market_state_path.exists():
        parser.error(f"required input missing: {features_path} and {market_state_path}")
    import pandas as pd
    from scripts.config import get_config
    features = pd.read_parquet(features_path)
    market_state = pd.read_parquet(market_state_path)
    run_walk_forward(
        folds=default_folds(), output_root=root / args.output_root,
        smoke=args.smoke, trainer=_default_trainer, tester=_default_tester,
        features_df=features, index_df=market_state, cfg=get_config(),
        timesteps=args.timesteps or (128 if args.smoke else 100_000),
        run_id=args.run_id,
    )
    return 0


if __name__ == "__main__":
    main()
