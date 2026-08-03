import json

import numpy as np
import pandas as pd
import pytest

from scripts.config import FACTOR_NAMES, get_config
from scripts.env import PortfolioEnv
from scripts.observation import ObservationScaler, collect_training_observations
from scripts.state import exogenous_fields, state_dim


def _env(observation_scaler=None):
    cfg = get_config()
    dates = pd.date_range("2020-01-01", periods=4, freq="D")
    rows = []
    for d in dates:
        for i in range(40):
            row = {
                "trade_date": d,
                "symbol": f"S{i:03d}",
                "ret_1d": 0.01 if i % 2 else -0.01,
                "is_suspended": False,
            }
            row.update({fn: float((i % 5) - 2) for fn in FACTOR_NAMES})
            rows.append(row)
    features = pd.DataFrame(rows)
    market_state = pd.DataFrame(
        0.1, index=dates, columns=exogenous_fields(FACTOR_NAMES)
    )
    return PortfolioEnv(
        features,
        market_state,
        cfg,
        dates.min(),
        dates.max(),
        observation_scaler=observation_scaler,
    )


def test_scaler_fit_transform_and_constant_columns():
    x = np.array([[1.0, 10.0], [3.0, 10.0], [5.0, 10.0]])
    scaler = ObservationScaler.fit(x, "obs-v1", ("a", "b"))

    assert scaler.mean == (3.0, 10.0)
    assert scaler.scale == (np.std(x[:, 0]), 1.0)
    np.testing.assert_allclose(scaler.transform(x), [[-1.2247449, 0], [0, 0], [1.2247449, 0]])
    assert scaler.transform(x).dtype == np.float32


def test_scaler_fit_rejects_invalid_shape_or_values():
    with pytest.raises(ValueError):
        ObservationScaler.fit(np.ones(3), "v1", ("a",))
    with pytest.raises(ValueError, match="empty"):
        ObservationScaler.fit(np.empty((0, 1)), "v1", ("a",))
    with pytest.raises(ValueError):
        ObservationScaler.fit(np.ones((2, 2)), "v1", ("a",))
    with pytest.raises(ValueError):
        ObservationScaler.fit(np.array([[1.0, np.nan]]), "v1", ("a", "b"))


def test_scaler_json_round_trip_and_metadata_validation(tmp_path):
    scaler = ObservationScaler.fit(np.array([[1.0, 2.0], [2.0, 4.0]]), "obs-v1", ("a", "b"))
    path = tmp_path / "scaler.json"
    factor_contract = {
        "factor_catalog_version": "catalog-v1",
        "factor_catalog_hash": "sha256:catalog",
        "selected_factors": ["mom_20", "vol_20"],
        "factor_directions": {"mom_20": 1, "vol_20": -1},
        "selection_run_id": "selection-42",
        "fold": 3,
        "state_schema_version": "obs-v1",
    }
    scaler.save(path, factor_contract=factor_contract)

    assert ObservationScaler.load(
        path,
        expected_schema="obs-v1",
        fields=("a", "b"),
        expected_factor_contract={
            **factor_contract,
            "factor_directions": [1, -1],
        },
    ) == scaler
    with pytest.raises(ValueError, match="schema"):
        ObservationScaler.load(
            path, expected_schema="obs-v2", fields=("a", "b"),
            expected_factor_contract=factor_contract,
        )
    with pytest.raises(ValueError, match="fields"):
        ObservationScaler.load(
            path, expected_schema="obs-v1", fields=("b", "a"),
            expected_factor_contract=factor_contract,
        )
    with pytest.raises(ValueError, match="factor checkpoint contract mismatch: factor_catalog_hash"):
        ObservationScaler.load(
            path,
            expected_schema="obs-v1",
            fields=("a", "b"),
            expected_factor_contract={
                **factor_contract,
                "factor_directions": [1, -1],
                "factor_catalog_hash": "sha256:wrong",
            },
        )
    assert json.loads(path.read_text()) == scaler.to_dict(factor_contract={
        **factor_contract,
        "factor_directions": [1, -1],
    })


def test_scaler_load_requires_complete_factor_contract(tmp_path):
    scaler = ObservationScaler.fit(np.array([[1.0]]), "obs-v1", ("a",))
    path = tmp_path / "scaler.json"
    path.write_text(json.dumps({
        "schema_version": "obs-v1", "fields": ["a"],
        "mean": [1.0], "scale": [1.0],
    }), encoding="utf-8")

    with pytest.raises(ValueError, match="expected scaler factor contract"):
        ObservationScaler.load(
            path, expected_schema="obs-v1", expected_factor_contract=None,
            fields=("a",),
        )


def test_transform_does_not_update_statistics_and_clips():
    scaler = ObservationScaler.fit(np.array([[0.0], [1.0]]), "v1", ("a",))
    before = (scaler.mean, scaler.scale)
    transformed = scaler.transform(np.array([[-100.0], [100.0]]))

    assert transformed.tolist() == [[-10.0], [10.0]]
    assert (scaler.mean, scaler.scale) == before


def test_collect_training_observations_is_seeded_and_requires_raw_env():
    env = _env()
    first = collect_training_observations(env, seed=7, max_steps=5)
    second = collect_training_observations(env, seed=7, max_steps=5)
    np.testing.assert_array_equal(first, second)
    assert first.shape == (5, state_dim(FACTOR_NAMES))

    with pytest.raises(ValueError, match="raw"):
        collect_training_observations(_env(ObservationScaler.fit(
            np.zeros((2, state_dim(FACTOR_NAMES))), "v1",
            tuple(f"f{i}" for i in range(state_dim(FACTOR_NAMES))),
        )), seed=0, max_steps=1)


def test_portfolio_env_scales_only_returned_observations():
    dim = state_dim(FACTOR_NAMES)
    scaler = ObservationScaler(
        schema_version="v1", fields=tuple(f"f{i}" for i in range(dim)),
        mean=tuple(0.0 for _ in range(dim)), scale=tuple(1.0 for _ in range(dim)),
    )
    env = _env(scaler)
    obs, _ = env.reset(seed=0)
    assert obs.dtype == np.float32
    raw = env.state_builder.build(env.dates[env.t], env.prev_stock_w, FACTOR_NAMES,
                                  env.prev_factor_w, cash=1.0)
    np.testing.assert_array_equal(obs, scaler.transform(raw))
    next_obs, *_ = env.step(np.zeros(env.action_space.shape, dtype=np.float32))
    assert next_obs.dtype == np.float32


def test_portfolio_env_scales_terminal_observation_once():
    dim = state_dim(FACTOR_NAMES)
    scaler = ObservationScaler(
        schema_version="v1", fields=tuple(f"f{i}" for i in range(dim)),
        mean=tuple(2.0 for _ in range(dim)), scale=tuple(2.0 for _ in range(dim)),
    )
    env = _env(scaler)
    action = np.zeros(env.action_space.shape, dtype=np.float32)

    env.reset(seed=0)
    for _ in range(max(0, len(env.decision_dates) - 1)):
        obs, _, terminated, _, _ = env.step(action)
        assert not terminated
    terminal_obs, _, terminated, _, _ = env.step(action)

    assert terminated
    np.testing.assert_array_equal(terminal_obs, np.full(dim, -1.0, dtype=np.float32))
