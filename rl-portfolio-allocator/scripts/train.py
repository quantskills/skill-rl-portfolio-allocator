"""SB3 PPO 训练核心。被 backtest/stress_test/allocate 复用。"""
from __future__ import annotations
import os
import json
import pathlib
from typing import Optional

import numpy as np
import pandas as pd

FACTOR_CONTRACT_FIELDS = (
    "factor_catalog_version",
    "factor_catalog_hash",
    "selected_factors",
    "factor_directions",
    "selection_run_id",
    "fold",
    "state_schema_version",
)


def _canonical_contract_payload(contract: dict | None) -> dict:
    """Return contract keys in their canonical names without filling defaults."""
    source = dict(contract or {})
    if "selected_factors" in source and "factor_names" in source:
        try:
            selected_factors = list(source["selected_factors"])
            factor_names = list(source["factor_names"])
        except TypeError as exc:
            raise ValueError(
                "selected_factors and factor_names must agree"
            ) from exc
        if selected_factors != factor_names:
            raise ValueError("selected_factors and factor_names must agree")
    if ("state_schema_version" in source and "schema_version" in source
            and source["state_schema_version"] != source["schema_version"]):
        raise ValueError("state_schema_version and schema_version must agree")
    if "selected_factors" not in source and "factor_names" in source:
        source["selected_factors"] = source["factor_names"]
    if "state_schema_version" not in source and "schema_version" in source:
        source["state_schema_version"] = source["schema_version"]
    return source


def _normalized_contract_values(contract: dict | None) -> dict:
    """Normalize aliases and ordered values while preserving omitted fields."""
    source = _canonical_contract_payload(contract)
    normalized = {}
    for key in FACTOR_CONTRACT_FIELDS:
        if key not in source:
            continue
        value = source[key]
        if value is None:
            normalized[key] = None
        elif key == "selected_factors":
            normalized[key] = list(value)
        elif key == "factor_directions":
            if isinstance(value, dict) and source.get("selected_factors") is not None:
                try:
                    normalized[key] = [
                        int(value[name]) for name in source["selected_factors"]
                    ]
                except KeyError:
                    # Keep malformed direction maps comparable so validation
                    # fails closed with the contract error below.
                    normalized[key] = {
                        str(name): int(direction)
                        for name, direction in value.items()
                    }
            elif isinstance(value, dict):
                normalized[key] = {
                    str(name): int(direction)
                    for name, direction in value.items()
                }
            else:
                normalized[key] = [int(direction) for direction in value]
        else:
            normalized[key] = value
    return normalized


def resolve_factor_contract(
    contract: dict | None,
    *,
    selected_factors=None,
    factor_directions=None,
    fold=None,
    state_schema_version=None,
) -> dict:
    source = _canonical_contract_payload(contract)
    selected_source = source.get("selected_factors")
    if selected_source is None:
        selected_source = selected_factors or ()
    ordered_factors = list(selected_source)
    resolved_directions = source.get("factor_directions")
    if resolved_directions is None:
        resolved_directions = factor_directions
    if isinstance(resolved_directions, dict):
        resolved_directions = [int(resolved_directions[name]) for name in ordered_factors]
    elif resolved_directions is None:
        resolved_directions = [1] * len(ordered_factors)
    else:
        resolved_directions = [int(value) for value in resolved_directions]
    if len(resolved_directions) != len(ordered_factors):
        raise ValueError("factor contract directions must align with selected_factors")
    return {
        "factor_catalog_version": source.get("factor_catalog_version"),
        "factor_catalog_hash": source.get("factor_catalog_hash"),
        "selected_factors": ordered_factors,
        "factor_directions": resolved_directions,
        "selection_run_id": source.get("selection_run_id"),
        "fold": source.get("fold", fold),
        "state_schema_version": source.get("state_schema_version", state_schema_version),
    }


def validate_factor_contract(actual: dict, expected: dict) -> None:
    actual_contract = _normalized_contract_values(actual)
    expected_contract = _normalized_contract_values(expected)
    fields_to_check = [
        key for key in FACTOR_CONTRACT_FIELDS
        if key in expected_contract and expected_contract[key] is not None
    ]
    if not fields_to_check:
        return
    mismatches = [
        key for key in fields_to_check
        if key not in actual_contract
        or actual_contract.get(key) != expected_contract.get(key)
    ]
    if mismatches:
        raise ValueError("factor checkpoint contract mismatch: " + ", ".join(mismatches))


def require_factor_contract(contract: dict, *, context: str = "factor contract") -> dict:
    """Canonicalize and require a complete factor contract at a hard boundary.

    ``validate_factor_contract`` intentionally supports partial expected contracts
    for low-level compatibility tests.  Published approvals and artifacts use
    this stricter entry point so omitted directions or identity fields cannot be
    replaced by defaults.
    """
    if not isinstance(contract, dict):
        raise ValueError(f"{context} must be an object")
    normalized = _normalized_contract_values(contract)
    missing = [
        key for key in FACTOR_CONTRACT_FIELDS
        if key not in normalized or normalized[key] is None
    ]
    if missing:
        raise ValueError(
            f"{context} missing complete factor contract fields: "
            + ", ".join(missing)
        )

    selected = normalized.get("selected_factors")
    directions = normalized.get("factor_directions")
    if (not isinstance(selected, list) or not selected
            or any(not isinstance(name, str) or not name for name in selected)
            or len(set(selected)) != len(selected)):
        raise ValueError(f"{context} selected_factors must be unique non-empty names")
    if (not isinstance(directions, list)
            or len(directions) != len(selected)
            or any(direction not in (-1, 1) for direction in directions)):
        raise ValueError(
            f"{context} factor_directions must explicitly align with selected_factors"
        )
    for key in (
        "factor_catalog_version", "factor_catalog_hash",
        "selection_run_id", "state_schema_version",
    ):
        if not isinstance(normalized.get(key), str) or not normalized[key]:
            raise ValueError(f"{context} {key} must be a non-empty string")
    if (not isinstance(normalized["fold"], int)
            or isinstance(normalized["fold"], bool)):
        raise ValueError(f"{context} fold must be an integer")
    return {key: normalized[key] for key in FACTOR_CONTRACT_FIELDS}


def require_complete_factor_contract(
    contract: dict,
    *,
    context: str = "factor contract",
    require_canonical: bool = True,
) -> dict:
    """Backward-compatible alias for the strict contract boundary."""
    del require_canonical
    return require_factor_contract(contract, context=context)


def artifact_paths(root, fold: int, seed: int, candidate: str,
                   schema_version: str) -> dict[str, pathlib.Path]:
    """Return the four stable paths for one fold/seed/candidate run."""
    root = pathlib.Path(root)
    stem = f"fold{fold}_seed{seed}"
    return {
        "model": root / f"{stem}_{candidate}.zip",
        "scaler": root / f"{stem}_scaler.json",
        "log": root / f"{stem}_{candidate}_training.jsonl",
        "metadata": root / f"{stem}_{candidate}_metadata.json",
    }


def _write_checkpoint_metadata(metadata_path: pathlib.Path, *, checkpoint_path,
                               contract: dict) -> None:
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(json.dumps({
        "schema_version": contract["state_schema_version"],
        "checkpoint_path": str(checkpoint_path),
        **contract,
    }, indent=2, sort_keys=True), encoding="utf-8")


def _json_number(value, default=0.0):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return default
    return value if np.isfinite(value) else default


def _sharpe(rets) -> float:
    from scripts.metrics import sharpe
    return float(sharpe(np.asarray(rets, dtype=float)))


def _eval_daily_net_rets(env, model) -> list[float]:
    obs, _ = env.reset()
    rets = []
    while True:
        action, _ = model.predict(obs, deterministic=True)
        obs, _, terminated, truncated, info = env.step(action)
        rets.extend(float(x) for x in info.get("daily_net_rets", ()))
        if terminated or truncated:
            return rets


from stable_baselines3.common.callbacks import BaseCallback


class TrainingMetricsCallback(BaseCallback):
    def __init__(self, log_path, verbose=0):
        super().__init__(verbose)
        self.log_path = pathlib.Path(log_path)
        self._actions = []
        self._dual_lambdas: list[float] = []

    def _on_training_start(self):
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.log_path.write_text("", encoding="utf-8")

    def _on_step(self) -> bool:
        actions = self.locals.get("actions")
        if actions is not None:
            self._actions.append(np.asarray(actions, dtype=float).reshape(-1))
        infos = self.locals.get("infos")
        if infos:
            lam = infos[-1].get("reward_parts", {}).get("dual_lambda")
            if lam is not None:
                self._dual_lambdas.append(float(lam))
        return True

    def _on_rollout_end(self):
        values = self.logger.name_to_value
        actions = np.concatenate(self._actions) if self._actions else np.array([])
        ep_rewards = [x["r"] for x in self.model.ep_info_buffer]
        record = {
            "timesteps": int(self.num_timesteps),
            "rollout_ep_rew_mean": _json_number(np.mean(ep_rewards)) if ep_rewards else 0.0,
            "policy_gradient_loss": _json_number(values.get("train/policy_gradient_loss")),
            "value_loss": _json_number(values.get("train/value_loss")),
            "entropy_loss": _json_number(values.get("train/entropy_loss")),
            "approx_kl": _json_number(values.get("train/approx_kl")),
            "clip_fraction": _json_number(values.get("train/clip_fraction")),
            "explained_variance": _json_number(values.get("train/explained_variance")),
            "action_mean": _json_number(np.mean(actions)) if actions.size else 0.0,
            "action_std": _json_number(np.std(actions)) if actions.size else 0.0,
            "action_saturation": _json_number(np.mean(np.abs(actions) >= 0.999)) if actions.size else 0.0,
            "dual_lambda_last": _json_number(self._dual_lambdas[-1]) if self._dual_lambdas else None,
            "dual_lambda_max": _json_number(max(self._dual_lambdas)) if self._dual_lambdas else None,
        }
        with self.log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
        self._actions.clear()
        self._dual_lambdas.clear()

    def _on_training_end(self):
        if not self.log_path.exists() or not self.log_path.read_text(encoding="utf-8").strip():
            self._on_rollout_end()


class ValidationSharpeCallback(BaseCallback):
    def __init__(self, eval_env, eval_freq=10000, patience=5,
                 best_model_path=None, log_path=None, verbose=0,
                 factor_contract: dict | None = None, metadata_path=None):
        super().__init__(verbose)
        self.eval_env = eval_env
        self.eval_freq = int(eval_freq)
        self.patience = int(patience)
        self.best_model_path = pathlib.Path(best_model_path) if best_model_path else None
        self.factor_contract = (
            require_factor_contract(
                factor_contract, context="best checkpoint factor contract"
            ) if self.best_model_path is not None else None
        )
        self.metadata_path = pathlib.Path(metadata_path) if metadata_path else None
        self.log_path = pathlib.Path(log_path) if log_path else None
        self.best_eval_metric = float("-inf")
        self.no_improvement_evals = 0
        self.deterministic = True
        self.eval_history = []

    def _evaluate(self):
        rets = _eval_daily_net_rets(self.eval_env, self.model)
        metric = _sharpe(rets)
        self.eval_history.append(metric)
        if metric > self.best_eval_metric:
            self.best_eval_metric = metric
            self.no_improvement_evals = 0
            if self.best_model_path is not None:
                self.best_model_path.parent.mkdir(parents=True, exist_ok=True)
                self.model.save(str(self.best_model_path.with_suffix("")))
                if self.factor_contract is not None:
                    metadata_path = self.metadata_path or self.best_model_path.with_name(
                        self.best_model_path.stem + "_metadata.json"
                    )
                    _write_checkpoint_metadata(
                        metadata_path, checkpoint_path=self.best_model_path,
                        contract=self.factor_contract,
                    )
        else:
            self.no_improvement_evals += 1
            if self.no_improvement_evals >= self.patience:
                self.model.stop_training = True
        if self.log_path is not None:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            record = {
                "timesteps": int(self.num_timesteps),
                "validation_net_sharpe": metric,
                "best_score": self.best_eval_metric,
            }
            with self.log_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, sort_keys=True) + "\n")

    def _on_step(self) -> bool:
        if self.num_timesteps and self.num_timesteps % self.eval_freq == 0:
            self._evaluate()
        return True

    def _on_training_end(self):
        if not self.eval_history:
            self._evaluate()


def select_device(pref: str) -> str:
    """Select compute device. MPS is excluded for PPO MlpPolicy due to known
    float32 numerical instability that causes NaN in policy parameters."""
    if pref == "cpu":
        return "cpu"
    if pref == "cuda":
        return "cuda"
    # MPS (Apple Silicon) causes NaN in SB3 PPO MlpPolicy — always use CPU.
    # See: https://github.com/DLR-RM/stable-baselines3/issues/1245
    if pref == "mps":
        return "cpu"
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
    except Exception:
        pass
    return "cpu"


def train_ppo(env, total_timesteps: int, seed: int = 0, device: str = "auto",
              save_path: Optional[str] = None, eval_env=None,
              eval_freq: int = 10_000, n_eval_episodes: int = 1,
              patience: Optional[int] = None, callback=None,
              training_log_path=None, factor_contract: dict | None = None,
              metadata_path=None):
    contract = None
    if save_path:
        contract = require_factor_contract(
            factor_contract, context="checkpoint factor contract"
        )
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv
    from stable_baselines3.common.monitor import Monitor

    def _mk():
        return Monitor(env)

    vec = DummyVecEnv([_mk])
    dev = select_device(device)
    model = PPO(
        "MlpPolicy", vec, verbose=0, seed=seed, device=dev,
        n_steps=1024, batch_size=256, learning_rate=3e-4,
        gamma=0.99, gae_lambda=0.95, clip_range=0.2, ent_coef=0.01,
    )

    callbacks = []
    if training_log_path is not None:
        callbacks.append(TrainingMetricsCallback(training_log_path))
    if callback is not None:
        if (isinstance(callback, ValidationSharpeCallback)
                and callback.best_model_path is not None):
            if contract is None:
                contract = callback.factor_contract
            else:
                validate_factor_contract(callback.factor_contract, contract)
        callbacks.append(callback)
    if eval_env is not None and callback is None:
        callbacks.append(ValidationSharpeCallback(
            eval_env, eval_freq=eval_freq, patience=patience or 5,
            best_model_path=(pathlib.Path(save_path).with_name(
                pathlib.Path(save_path).stem + "_best.zip") if save_path else None),
            factor_contract=contract))

    callback_arg = callbacks if len(callbacks) > 1 else (callbacks[0] if callbacks else None)
    model.learn(total_timesteps=total_timesteps, callback=callback_arg)
    validation_callback = next((x for x in callbacks if isinstance(x, ValidationSharpeCallback)), None)
    if validation_callback is not None and validation_callback.best_model_path is not None:
        best_path = validation_callback.best_model_path
        if best_path.exists():
            best_metadata = validation_callback.metadata_path or best_path.with_name(
                best_path.stem + "_metadata.json"
            )
            model = load_ppo(
                str(best_path), env,
                expected_factor_contract=validation_callback.factor_contract,
                metadata_path=best_metadata,
            )
    if save_path:
        pathlib.Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        model.save(save_path)
        metadata_target = (pathlib.Path(metadata_path) if metadata_path else
                           pathlib.Path(save_path).with_name(
                               pathlib.Path(save_path).stem + "_metadata.json"))
        metadata_target.parent.mkdir(parents=True, exist_ok=True)
        _write_checkpoint_metadata(
            metadata_target, checkpoint_path=save_path, contract=contract
        )
    return model


def train_candidates(*, root, fold, seed, candidates, schema_version,
                      raw_train_env, eval_env, total_timesteps, device="auto",
                      train_range=None, val_range=None, reward_variant=None,
                      buffer_variant=None, factor_contract: dict | None = None):
    """Fit one scaler, then train each candidate against that frozen scaler."""
    from scripts.observation import collect_training_observations, ObservationScaler
    from scripts.state import STATE_SCHEMA_VERSION, state_fields

    if schema_version != STATE_SCHEMA_VERSION:
        raise ValueError("schema_version must match the state schema")
    contract = require_factor_contract(
        factor_contract, context="training factor contract"
    )
    if contract["fold"] != fold:
        raise ValueError("training factor contract fold disagrees with training fold")
    if contract["state_schema_version"] != schema_version:
        raise ValueError("training factor contract schema disagrees with schema_version")
    contract_names = contract["selected_factors"]
    env_names = [
        tuple(names)
        for names in (
            getattr(raw_train_env, "factor_names", None),
            getattr(eval_env, "factor_names", None),
        )
        if names is not None
    ]
    factor_names = tuple(contract_names)
    if any(names != factor_names for names in env_names):
        raise ValueError("factor contract selected_factors disagree with environment")
    for env in (raw_train_env, eval_env):
        config_names = getattr(env, "cfg", {}).get("factor_names") if isinstance(
            getattr(env, "cfg", None), dict
        ) else None
        if config_names is not None and tuple(config_names) != factor_names:
            raise ValueError("factor contract selected_factors disagree with environment config")
    observations = collect_training_observations(raw_train_env, seed=seed)
    fields = tuple(state_fields(factor_names))
    if observations.shape[1] != len(fields):
        raise ValueError("raw training observations do not match the state schema")
    scaler = ObservationScaler.fit(observations, STATE_SCHEMA_VERSION, fields)
    results = {}
    for candidate in candidates:
        paths = artifact_paths(root, fold, seed, candidate, schema_version)
        paths["scaler"].parent.mkdir(parents=True, exist_ok=True)
        scaler.save(paths["scaler"], factor_contract=contract)
        raw_train_env.observation_scaler = scaler
        eval_env.observation_scaler = scaler
        validation = ValidationSharpeCallback(
            eval_env, eval_freq=10000, patience=5,
            best_model_path=paths["model"], log_path=paths["log"],
            factor_contract=contract, metadata_path=paths["metadata"])
        train_ppo(raw_train_env, total_timesteps=total_timesteps, seed=seed,
                  device=device, save_path=str(paths["model"]), callback=validation,
                  training_log_path=paths["log"], factor_contract=contract,
                  metadata_path=paths["metadata"])
        metadata = {
            "fold": fold, "seed": seed, "schema_version": schema_version,
            "scaler_path": str(paths["scaler"]), "train_range": train_range,
            "val_range": val_range, "reward_variant": reward_variant,
            "buffer_variant": buffer_variant, "total_timesteps": total_timesteps,
            "best_eval_metric": validation.best_eval_metric,
            **contract,
        }
        paths["metadata"].write_text(json.dumps(metadata, indent=2, default=str), encoding="utf-8")
        results[candidate] = paths
    return results


def load_ppo(path: str, env, *, expected_factor_contract: dict,
             metadata_path):
    expected_contract = require_factor_contract(
        expected_factor_contract, context="expected checkpoint factor contract"
    )
    if metadata_path is None:
        raise ValueError("checkpoint metadata path is required")
    metadata = pathlib.Path(metadata_path)
    if not metadata.exists():
        raise FileNotFoundError(
            f"checkpoint metadata required for contract-bound load: {metadata}"
        )
    actual_contract = require_factor_contract(
        json.loads(metadata.read_text(encoding="utf-8")),
        context="checkpoint metadata",
    )
    validate_factor_contract(actual_contract, expected_contract)
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv
    from stable_baselines3.common.monitor import Monitor
    vec = DummyVecEnv([lambda: Monitor(env)])
    return PPO.load(path, env=vec)


def main() -> None:
    import argparse
    from scripts.config import get_config
    from scripts.env import effective_range, make_env
    cfg = get_config()
    root = pathlib.Path(__file__).resolve().parent.parent
    features_path = root / "data" / "features.parquet"
    market_state_path = root / "data" / "market_state.parquet"
    ckpt = root / "checkpoints" / "smoke.zip"

    ap = argparse.ArgumentParser()
    ap.add_argument("--timesteps", type=int, default=5000,
                    help="训练步数;默认 5000 为快速自检(smoke)")
    ap.add_argument("--factor-contract", required=True,
                    help="complete selected-factor contract JSON for the checkpoint")
    args = ap.parse_args()

    feats = pd.read_parquet(features_path)
    market_state = pd.read_parquet(market_state_path)
    start, end = effective_range(
        feats, market_state, cfg["start_date"], cfg["end_date"] or "2099-12-31",
        cfg=cfg,
    )
    print(f"effective range: {start.date()} ~ {end.date()}")
    env = make_env(str(features_path), str(market_state_path), cfg, start, end)
    device = select_device(cfg["train_device"])
    print(f"train device: {device}")
    contract = json.loads(pathlib.Path(args.factor_contract).read_text(encoding="utf-8"))
    model = train_ppo(
        env, total_timesteps=args.timesteps, seed=0, device=device,
        save_path=str(ckpt), factor_contract=contract,
    )
    print(f"checkpoint saved: {ckpt}  (timesteps={args.timesteps})")


if __name__ == "__main__":
    main()
