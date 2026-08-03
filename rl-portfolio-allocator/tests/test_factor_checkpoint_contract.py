import pytest

from scripts.train import (
    FACTOR_CONTRACT_FIELDS,
    load_ppo,
    require_complete_factor_contract,
    train_ppo,
    validate_factor_contract,
)


@pytest.fixture
def valid_contract():
    return {
        "factor_catalog_version": "catalog-v1",
        "factor_catalog_hash": "sha256:abc123",
        "selected_factors": ["mom_20", "vol_20"],
        "factor_directions": [1, -1],
        "selection_run_id": "selection-42",
        "fold": 3,
        "state_schema_version": "state-v1",
    }


def test_factor_contract_fields_are_stable():
    assert FACTOR_CONTRACT_FIELDS == (
        "factor_catalog_version",
        "factor_catalog_hash",
        "selected_factors",
        "factor_directions",
        "selection_run_id",
        "fold",
        "state_schema_version",
    )


def test_checkpoint_contract_accepts_exact_match(valid_contract):
    validate_factor_contract(valid_contract, dict(valid_contract))


def test_checkpoint_contract_allows_partial_expected_contract(valid_contract):
    expected = {
        "selected_factors": list(valid_contract["selected_factors"]),
    }

    validate_factor_contract(valid_contract, expected)


def test_checkpoint_contract_treats_explicit_null_like_omitted_partial_field(valid_contract):
    expected = {
        "selected_factors": None,
        "factor_directions": None,
    }

    validate_factor_contract(valid_contract, expected)


def test_checkpoint_contract_accepts_legacy_actual_aliases(valid_contract):
    actual = {
        "factor_catalog_version": valid_contract["factor_catalog_version"],
        "factor_catalog_hash": valid_contract["factor_catalog_hash"],
        "factor_names": list(valid_contract["selected_factors"]),
        "factor_directions": list(valid_contract["factor_directions"]),
        "selection_run_id": valid_contract["selection_run_id"],
        "fold": valid_contract["fold"],
        "schema_version": valid_contract["state_schema_version"],
    }

    validate_factor_contract(actual, valid_contract)


def test_complete_artifact_contract_accepts_legacy_actual_aliases(valid_contract):
    actual = {
        **valid_contract,
        "factor_names": list(valid_contract["selected_factors"]),
        "schema_version": valid_contract["state_schema_version"],
    }
    actual.pop("selected_factors")
    actual.pop("state_schema_version")

    normalized = require_complete_factor_contract(
        actual, require_canonical=False, context="artifact"
    )

    assert normalized["selected_factors"] == valid_contract["selected_factors"]
    assert normalized["state_schema_version"] == valid_contract["state_schema_version"]


def test_complete_artifact_contract_rejects_conflicting_factor_name_alias(valid_contract):
    actual = {
        **valid_contract,
        "factor_names": list(reversed(valid_contract["selected_factors"])),
    }

    with pytest.raises(ValueError, match="selected_factors and factor_names"):
        require_complete_factor_contract(actual, context="artifact")


def test_complete_artifact_contract_rejects_conflicting_schema_alias(valid_contract):
    actual = {
        **valid_contract,
        "schema_version": "state-v2",
    }

    with pytest.raises(ValueError, match="state_schema_version and schema_version"):
        require_complete_factor_contract(actual, context="artifact")


def test_checkpoint_contract_validates_legacy_expected_aliases(valid_contract):
    expected = {
        "factor_names": list(valid_contract["selected_factors"]),
        "schema_version": valid_contract["state_schema_version"],
    }
    actual = dict(valid_contract)
    actual["selected_factors"] = ["vol_20", "mom_20"]

    with pytest.raises(ValueError, match="selected_factors"):
        validate_factor_contract(actual, expected)


def test_checkpoint_contract_rejects_missing_expected_direction_metadata(valid_contract):
    actual = dict(valid_contract)
    actual.pop("factor_directions")

    with pytest.raises(ValueError, match="factor_directions"):
        validate_factor_contract(actual, valid_contract)


def test_contract_bound_checkpoint_load_requires_metadata(tmp_path, valid_contract):
    checkpoint = tmp_path / "checkpoint.zip"
    checkpoint.write_bytes(b"checkpoint")

    with pytest.raises(FileNotFoundError, match="metadata required"):
        load_ppo(
            str(checkpoint), object(), expected_factor_contract=valid_contract,
            metadata_path=tmp_path / "checkpoint_metadata.json",
        )


def test_checkpoint_save_requires_complete_factor_contract_before_training(tmp_path):
    with pytest.raises(ValueError, match="checkpoint factor contract"):
        train_ppo(
            object(), total_timesteps=1, device="cpu",
            save_path=str(tmp_path / "checkpoint.zip"),
        )


def test_checkpoint_load_requires_an_explicit_complete_contract(tmp_path):
    checkpoint = tmp_path / "checkpoint.zip"
    checkpoint.write_bytes(b"checkpoint")

    with pytest.raises(TypeError, match="expected_factor_contract"):
        load_ppo(str(checkpoint), object(), metadata_path=tmp_path / "metadata.json")


def test_contract_bound_checkpoint_load_rejects_partial_expected_contract(
    tmp_path, valid_contract
):
    checkpoint = tmp_path / "checkpoint.zip"
    checkpoint.write_bytes(b"checkpoint")
    metadata = tmp_path / "checkpoint_metadata.json"
    metadata.write_text(__import__("json").dumps(valid_contract), encoding="utf-8")

    with pytest.raises(ValueError, match="complete factor contract"):
        load_ppo(
            str(checkpoint),
            object(),
            expected_factor_contract={"selected_factors": ["mom_20", "vol_20"]},
            metadata_path=metadata,
        )


@pytest.mark.parametrize("field,value", [
    ("selected_factors", ["vol_20", "mom_20"]),
    ("factor_directions", {"mom_20": -1, "vol_20": 1}),
    ("factor_catalog_hash", "sha256:wrong"),
    ("state_schema_version", "wrong"),
])
def test_checkpoint_contract_rejects_mismatch(valid_contract, field, value):
    actual = dict(valid_contract)
    actual[field] = value
    with pytest.raises(ValueError, match="factor checkpoint contract mismatch"):
        validate_factor_contract(actual, valid_contract)
