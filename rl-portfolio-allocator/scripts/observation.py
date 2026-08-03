from __future__ import annotations

from dataclasses import dataclass
import json
import pathlib

import numpy as np


@dataclass(frozen=True)
class ObservationScaler:
    schema_version: str
    fields: tuple[str, ...]
    mean: tuple[float, ...]
    scale: tuple[float, ...]

    @classmethod
    def fit(cls, x: np.ndarray, schema_version: str,
            fields: tuple[str, ...]) -> "ObservationScaler":
        arr = np.asarray(x, dtype=float)
        if arr.ndim != 2 or arr.shape[1] != len(fields):
            raise ValueError("observation matrix shape does not match fields")
        if arr.shape[0] == 0:
            raise ValueError("observation matrix must not be empty")
        if not np.isfinite(arr).all():
            raise ValueError("non-finite observation in scaler fit")
        mean = arr.mean(axis=0)
        std = arr.std(axis=0)
        scale = np.where(std < 1e-8, 1.0, std)
        return cls(schema_version, tuple(fields), tuple(mean), tuple(scale))

    def transform(self, x: np.ndarray) -> np.ndarray:
        arr = np.asarray(x, dtype=float)
        out = (arr - np.asarray(self.mean)) / np.asarray(self.scale)
        return np.clip(out, -10.0, 10.0).astype(np.float32)

    def to_dict(self, factor_contract: dict) -> dict:
        payload = {
            "schema_version": self.schema_version,
            "fields": list(self.fields),
            "mean": list(self.mean),
            "scale": list(self.scale),
        }
        from scripts.train import require_factor_contract

        payload.update(require_factor_contract(
            factor_contract, context="scaler factor contract"
        ))
        return payload

    def save(self, path, factor_contract: dict) -> None:
        target = pathlib.Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            json.dumps(self.to_dict(factor_contract=factor_contract), indent=2),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path, expected_schema: str, expected_factor_contract: dict,
             fields: tuple[str, ...] | None = None,
             expected_fields: tuple[str, ...] | None = None) -> "ObservationScaler":
        expected = expected_fields if expected_fields is not None else fields
        data = json.loads(pathlib.Path(path).read_text(encoding="utf-8"))
        if data.get("schema_version") != expected_schema:
            raise ValueError("scaler schema version mismatch")
        if expected is None or tuple(data.get("fields", ())) != tuple(expected):
            raise ValueError("scaler fields mismatch")
        from scripts.train import require_factor_contract, validate_factor_contract

        expected_contract = require_factor_contract(
            expected_factor_contract, context="expected scaler factor contract"
        )
        actual_contract = require_factor_contract(
            data, context="scaler metadata"
        )
        validate_factor_contract(actual_contract, expected_contract)
        return cls(data["schema_version"], tuple(data["fields"]),
                   tuple(data["mean"]), tuple(data["scale"]))


def collect_training_observations(env, seed: int, max_steps: int = 4096) -> np.ndarray:
    if getattr(env, "observation_scaler", None) is not None:
        raise ValueError("scaler prefit requires a raw training environment")
    rng = np.random.default_rng(seed)
    observations = []
    obs, _ = env.reset(seed=seed)
    while len(observations) < max_steps:
        observations.append(np.asarray(obs, dtype=float))
        action = rng.uniform(env.action_space.low, env.action_space.high).astype(
            env.action_space.dtype)
        obs, _, terminated, truncated, _ = env.step(action)
        if terminated or truncated:
            obs, _ = env.reset(seed=seed)
    return np.vstack(observations)
