import json
import pytest

from scripts.config import FACTOR_NAMES
from scripts.observation import ObservationScaler
from scripts.state import STATE_SCHEMA_VERSION, state_fields
from scripts.train import FACTOR_CONTRACT_FIELDS, artifact_paths, train_candidates
from tests.test_train import _toy_env


def test_artifact_paths_are_candidate_specific(tmp_path):
    paths = artifact_paths(tmp_path, fold=2, seed=7, candidate="ppo", schema_version="obs-v1")
    assert paths == {
        "model": tmp_path / "fold2_seed7_ppo.zip",
        "scaler": tmp_path / "fold2_seed7_scaler.json",
        "log": tmp_path / "fold2_seed7_ppo_training.jsonl",
        "metadata": tmp_path / "fold2_seed7_ppo_metadata.json",
    }


def test_toy_training_writes_four_complete_artifacts(tmp_path):
    factor_contract = {
        "factor_catalog_version": "catalog-v1",
        "factor_catalog_hash": "sha256:abc123",
        "selected_factors": list(FACTOR_NAMES),
        "factor_directions": {name: (-1 if index % 2 else 1) for index, name in enumerate(FACTOR_NAMES)},
        "selection_run_id": "selection-42",
        "fold": 1,
        "state_schema_version": STATE_SCHEMA_VERSION,
    }
    result = train_candidates(
        root=tmp_path,
        fold=1,
        seed=3,
        candidates=("ppo",),
        schema_version=STATE_SCHEMA_VERSION,
        raw_train_env=_toy_env(),
        eval_env=_toy_env(),
        total_timesteps=128,
        device="cpu",
        train_range=("2020-01-01", "2020-01-05"),
        val_range=("2020-01-06", "2020-01-08"),
        reward_variant="low",
        buffer_variant="none",
        factor_contract=factor_contract,
    )
    paths = result["ppo"]
    assert all(path.exists() for path in paths.values())
    assert paths["model"].suffix == ".zip"
    scaler = ObservationScaler.load(
        paths["scaler"], expected_schema=STATE_SCHEMA_VERSION,
        fields=tuple(state_fields(FACTOR_NAMES)),
        expected_factor_contract=factor_contract,
    )
    assert scaler.fields == tuple(state_fields(FACTOR_NAMES))
    metadata = json.loads(paths["metadata"].read_text())
    for key in ("fold", "seed", "schema_version", "scaler_path", "train_range",
                "val_range", "reward_variant", "buffer_variant", "total_timesteps",
                "best_eval_metric"):
        assert key in metadata
    scaler_payload = json.loads(paths["scaler"].read_text())
    for field in FACTOR_CONTRACT_FIELDS:
        assert metadata[field] == scaler_payload[field]
    assert metadata["factor_directions"] == [
        factor_contract["factor_directions"][name]
        for name in metadata["selected_factors"]
    ]
    records = [json.loads(line) for line in paths["log"].read_text().splitlines()]
    assert records
    training_record = next(record for record in records if "rollout_ep_rew_mean" in record)
    assert {"timesteps", "rollout_ep_rew_mean", "policy_gradient_loss",
            "value_loss", "entropy_loss", "approx_kl", "clip_fraction",
            "explained_variance", "action_mean", "action_std",
            "action_saturation"} <= training_record.keys()
    validation_record = next(record for record in records if "validation_net_sharpe" in record)
    assert {"validation_net_sharpe", "best_score", "timesteps"} <= validation_record.keys()


def test_training_rejects_non_state_schema_version(tmp_path):
    with pytest.raises(ValueError, match="state schema"):
        train_candidates(
            root=tmp_path, fold=1, seed=3, candidates=("ppo",),
            schema_version="obs-v1", raw_train_env=_toy_env(), eval_env=_toy_env(),
            total_timesteps=1, device="cpu",
        )


def test_training_artifacts_follow_contract_selected_factors(tmp_path):
    selected_factors = [FACTOR_NAMES[4], FACTOR_NAMES[1], FACTOR_NAMES[5]]
    fields = tuple(state_fields(selected_factors))

    class DummyEnv:
        def __init__(self):
            self.observation_scaler = None

    raw_train_env = DummyEnv()
    eval_env = DummyEnv()
    factor_contract = {
        "factor_catalog_version": "catalog-v1",
        "factor_catalog_hash": "sha256:subset",
        "selected_factors": selected_factors,
        "factor_directions": {name: (-1 if index % 2 else 1) for index, name in enumerate(selected_factors)},
        "selection_run_id": "selection-subset",
        "fold": 2,
        "state_schema_version": STATE_SCHEMA_VERSION,
    }

    observations = __import__("numpy").zeros((4, len(fields)))

    from scripts import train as train_module
    from scripts.observation import ObservationScaler

    original_collect = train_module.__dict__.get("collect_training_observations")
    original_fit = ObservationScaler.fit
    original_train_ppo = train_module.train_ppo
    original_validation = train_module.ValidationSharpeCallback
    try:
        # function-local imports mean monkeypatching module globals isn't enough;
        # patch through imported modules used in the function body.
        import scripts.observation as observation_module
        observation_module_collect = observation_module.collect_training_observations
        observation_module.collect_training_observations = lambda env, seed: observations

        captured = {}

        def fake_fit(x, schema_version, fit_fields):
            captured["fit_fields"] = tuple(fit_fields)
            return ObservationScaler(
                schema_version=schema_version,
                fields=tuple(fit_fields),
                mean=tuple(0.0 for _ in fit_fields),
                scale=tuple(1.0 for _ in fit_fields),
            )

        class FakeValidation:
            def __init__(self, *args, **kwargs):
                self.best_eval_metric = 1.23

        def fake_train(env, **kwargs):
            __import__("pathlib").Path(kwargs["save_path"]).write_bytes(b"checkpoint")

        ObservationScaler.fit = staticmethod(fake_fit)
        train_module.ValidationSharpeCallback = FakeValidation
        train_module.train_ppo = fake_train

        result = train_candidates(
            root=tmp_path,
            fold=2,
            seed=7,
            candidates=("ppo",),
            schema_version=STATE_SCHEMA_VERSION,
            raw_train_env=raw_train_env,
            eval_env=eval_env,
            total_timesteps=4,
            device="cpu",
            factor_contract=factor_contract,
        )
    finally:
        observation_module.collect_training_observations = observation_module_collect
        ObservationScaler.fit = original_fit
        train_module.train_ppo = original_train_ppo
        train_module.ValidationSharpeCallback = original_validation

    paths = result["ppo"]
    scaler = ObservationScaler.load(
        paths["scaler"], expected_schema=STATE_SCHEMA_VERSION, fields=fields,
        expected_factor_contract=factor_contract,
    )
    metadata = json.loads(paths["metadata"].read_text())

    assert captured["fit_fields"] == fields
    assert scaler.fields == fields
    assert metadata["selected_factors"] == selected_factors
    assert metadata["factor_directions"] == [1, -1, 1]


def test_training_rejects_environment_factor_order_that_disagrees_with_contract(tmp_path):
    class DummyEnv:
        factor_names = (FACTOR_NAMES[0], FACTOR_NAMES[1])
        observation_scaler = None

    with pytest.raises(ValueError, match="disagree with environment"):
        train_candidates(
            root=tmp_path,
            fold=1,
            seed=0,
            candidates=("ppo",),
            schema_version=STATE_SCHEMA_VERSION,
            raw_train_env=DummyEnv(),
            eval_env=DummyEnv(),
            total_timesteps=1,
            factor_contract={
                "factor_catalog_version": "catalog-v1",
                "factor_catalog_hash": "sha256:catalog",
                "selected_factors": [FACTOR_NAMES[1], FACTOR_NAMES[0]],
                "factor_directions": [1, -1],
                "selection_run_id": "selection-1",
                "fold": 1,
                "state_schema_version": STATE_SCHEMA_VERSION,
            },
        )


def test_training_rejects_incomplete_factor_contract_before_observation_collection(tmp_path):
    with pytest.raises(ValueError, match="factor_directions"):
        train_candidates(
            root=tmp_path, fold=1, seed=0, candidates=("ppo",),
            schema_version=STATE_SCHEMA_VERSION,
            raw_train_env=_toy_env(), eval_env=_toy_env(), total_timesteps=1,
            factor_contract={
                "factor_catalog_version": "catalog-v1",
                "factor_catalog_hash": "sha256:catalog",
                "selected_factors": list(FACTOR_NAMES),
                "selection_run_id": "selection-1",
                "fold": 1,
                "state_schema_version": STATE_SCHEMA_VERSION,
            },
        )
