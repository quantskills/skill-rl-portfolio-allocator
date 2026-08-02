from __future__ import annotations

import json
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import pytest

import scripts.factor_cache as factor_cache_module
import scripts.features as features_module
from scripts.factor_catalog import CATALOG_VERSION, FACTOR_CATALOG, catalog_hash
from scripts.factor_cache import (
    BASE_COLUMNS,
    load_family,
    materialize_selected_panel,
    write_factor_cache,
)


@pytest.fixture
def factor_panel() -> pd.DataFrame:
    dates = pd.to_datetime(["2024-01-02", "2024-01-03"])
    rows = []
    for date_idx, date in enumerate(dates):
        for symbol_idx, symbol in enumerate(("AAA", "BBB")):
            row = {
                "trade_date": date,
                "symbol": symbol,
                "ret_1d": np.float32(0.01 * (date_idx + symbol_idx + 1)),
                "is_suspended": False,
            }
            for factor_idx, spec in enumerate(FACTOR_CATALOG):
                row[spec.name] = np.float32(
                    factor_idx + 0.1 * date_idx + 0.01 * symbol_idx
                )
            rows.append(row)
    return pd.DataFrame(rows, columns=[*BASE_COLUMNS, *[s.name for s in FACTOR_CATALOG]])


def test_write_cache_partitions_catalog_by_family(tmp_path: Path, factor_panel: pd.DataFrame):
    manifest = write_factor_cache(factor_panel, tmp_path)

    assert (tmp_path / "catalog.json").exists()
    assert (tmp_path / "base.parquet").exists()
    assert len(list(tmp_path.glob("*.parquet"))) == 11
    assert manifest["catalog_version"] == CATALOG_VERSION
    assert manifest["catalog_hash"] == catalog_hash(FACTOR_CATALOG)
    assert set(manifest["families"]) == {spec.family for spec in FACTOR_CATALOG}
    assert manifest["row_count"] == len(factor_panel)
    assert manifest["schema"]["base"] == list(BASE_COLUMNS)

    for spec in FACTOR_CATALOG:
        loaded = load_family(tmp_path, spec.family)
        assert loaded[spec.name].dtype == np.dtype("float32")


def test_materialize_selected_panel_preserves_order_and_direction(
    tmp_path: Path, factor_panel: pd.DataFrame
):
    write_factor_cache(factor_panel, tmp_path)
    selected = [
        {"name": "vol_20", "direction": -1},
        {"name": "mom_20", "direction": 1},
    ]

    out = materialize_selected_panel(tmp_path, selected)

    assert list(out.columns) == [*BASE_COLUMNS, "vol_20", "mom_20"]
    expected = -factor_panel["vol_20"].to_numpy()
    np.testing.assert_allclose(out["vol_20"], expected, equal_nan=True)
    np.testing.assert_allclose(out["mom_20"], factor_panel["mom_20"], equal_nan=True)


def test_writer_rejects_duplicate_row_keys(tmp_path: Path, factor_panel: pd.DataFrame):
    duplicate = pd.concat([factor_panel, factor_panel.iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match="duplicate.*trade_date.*symbol"):
        write_factor_cache(duplicate, tmp_path)


@pytest.mark.parametrize(
    "mutator, message",
    [
        (lambda frame: frame.drop(columns=["mom_20"]), "missing factor"),
        (lambda frame: frame.assign(unknown_factor=np.float32(1.0)), "unknown"),
        (lambda frame: frame.assign(mom_20="bad"), "dtype"),
        (lambda frame: frame.assign(symbol=None), "key"),
    ],
)
def test_writer_rejects_bad_schema_and_keys(
    tmp_path: Path, factor_panel: pd.DataFrame, mutator, message: str
):
    with pytest.raises((TypeError, ValueError), match=message):
        write_factor_cache(mutator(factor_panel.copy()), tmp_path)


def test_writer_failure_does_not_replace_existing_cache(
    tmp_path: Path, factor_panel: pd.DataFrame, monkeypatch
):
    root = tmp_path / "factors"
    write_factor_cache(factor_panel, root)
    before = (root / "catalog.json").read_bytes()

    original_to_parquet = pd.DataFrame.to_parquet
    calls = 0

    def fail_on_second_write(self, *args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("simulated parquet failure")
        return original_to_parquet(self, *args, **kwargs)

    monkeypatch.setattr(pd.DataFrame, "to_parquet", fail_on_second_write)
    with pytest.raises(OSError, match="simulated parquet failure"):
        write_factor_cache(factor_panel.assign(mom_20=np.float32(99.0)), root)

    assert (root / "catalog.json").read_bytes() == before
    assert not list(tmp_path.glob(".factors.tmp-*"))


def test_materialize_rejects_unknown_duplicate_and_invalid_selection(
    tmp_path: Path, factor_panel: pd.DataFrame
):
    write_factor_cache(factor_panel, tmp_path)
    cases = [
        ([{"name": "not_a_factor", "direction": 1}], "unknown"),
        ([{"name": "mom_20", "direction": 1}, {"name": "mom_20", "direction": 1}], "duplicate"),
        ([{"name": "mom_20", "direction": 0}], "direction"),
        ([{"name": "mom_20", "direction": -1.0}], "direction"),
    ]
    for selected, message in cases:
        with pytest.raises((TypeError, ValueError), match=message):
            materialize_selected_panel(tmp_path, selected)


def test_reader_rejects_missing_family_and_row_mismatch(
    tmp_path: Path, factor_panel: pd.DataFrame
):
    write_factor_cache(factor_panel, tmp_path)
    (tmp_path / "momentum.parquet").unlink()
    with pytest.raises((FileNotFoundError, ValueError), match="momentum"):
        load_family(tmp_path, "momentum")

    write_factor_cache(factor_panel, tmp_path)
    family_path = tmp_path / "momentum.parquet"
    family = pd.read_parquet(family_path).iloc[:-1]
    family.to_parquet(family_path, index=False)
    with pytest.raises(ValueError, match="row|key|hash"):
        materialize_selected_panel(tmp_path, [{"name": "mom_20", "direction": 1}])


def test_reader_rejects_catalog_hash_mismatch(
    tmp_path: Path, factor_panel: pd.DataFrame
):
    write_factor_cache(factor_panel, tmp_path)
    catalog_path = tmp_path / "catalog.json"
    catalog = json.loads(catalog_path.read_text())
    catalog["catalog_hash"] = "sha256:" + "0" * 64
    catalog_path.write_text(json.dumps(catalog))

    with pytest.raises(ValueError, match="hash"):
        load_family(tmp_path, "momentum")


def test_reader_rejects_null_file_hash(tmp_path: Path, factor_panel: pd.DataFrame):
    write_factor_cache(factor_panel, tmp_path)
    catalog_path = tmp_path / "catalog.json"
    catalog = json.loads(catalog_path.read_text())
    catalog["file_hashes"]["base.parquet"] = None
    catalog_path.write_text(json.dumps(catalog))

    with pytest.raises(ValueError, match="file hash"):
        load_family(tmp_path, "momentum")


@pytest.mark.parametrize("bad_hash", ["sha512:" + "0" * 64, "not-a-hash"])
def test_reader_rejects_non_sha256_file_hash(
    tmp_path: Path, factor_panel: pd.DataFrame, bad_hash: str
):
    write_factor_cache(factor_panel, tmp_path)
    catalog_path = tmp_path / "catalog.json"
    catalog = json.loads(catalog_path.read_text())
    catalog["file_hashes"]["base.parquet"] = bad_hash
    catalog_path.write_text(json.dumps(catalog))

    with pytest.raises(ValueError, match="file hash"):
        load_family(tmp_path, "momentum")


def test_reader_rejects_missing_file_hash(
    tmp_path: Path, factor_panel: pd.DataFrame
):
    write_factor_cache(factor_panel, tmp_path)
    catalog_path = tmp_path / "catalog.json"
    catalog = json.loads(catalog_path.read_text())
    del catalog["file_hashes"]["base.parquet"]
    catalog_path.write_text(json.dumps(catalog))

    with pytest.raises(ValueError, match="file hash"):
        load_family(tmp_path, "momentum")


def test_malformed_catalog_json_has_explicit_cache_error(
    tmp_path: Path, factor_panel: pd.DataFrame
):
    write_factor_cache(factor_panel, tmp_path)
    (tmp_path / "catalog.json").write_text("[]")

    with pytest.raises(ValueError, match="catalog"):
        load_family(tmp_path, "momentum")


def test_reader_rejects_manifest_without_schema(
    tmp_path: Path, factor_panel: pd.DataFrame
):
    write_factor_cache(factor_panel, tmp_path)
    catalog_path = tmp_path / "catalog.json"
    catalog = json.loads(catalog_path.read_text())
    del catalog["schema"]
    catalog_path.write_text(json.dumps(catalog))

    with pytest.raises(ValueError, match="schema"):
        load_family(tmp_path, "momentum")


def test_reader_rejects_manifest_without_row_keys(
    tmp_path: Path, factor_panel: pd.DataFrame
):
    write_factor_cache(factor_panel, tmp_path)
    catalog_path = tmp_path / "catalog.json"
    catalog = json.loads(catalog_path.read_text())
    catalog.pop("row_keys", None)
    catalog_path.write_text(json.dumps(catalog))

    with pytest.raises(ValueError, match="row_keys"):
        load_family(tmp_path, "momentum")


@pytest.mark.parametrize("mutation", ["row_counts", "family_row_count"])
def test_reader_rejects_manifest_row_count_mismatch(
    tmp_path: Path, factor_panel: pd.DataFrame, mutation: str
):
    write_factor_cache(factor_panel, tmp_path)
    catalog_path = tmp_path / "catalog.json"
    catalog = json.loads(catalog_path.read_text())
    if mutation == "row_counts":
        catalog["row_counts"]["base"] += 1
    else:
        catalog["families"]["momentum"]["row_count"] += 1
    catalog_path.write_text(json.dumps(catalog))

    with pytest.raises(ValueError, match="row count"):
        load_family(tmp_path, "momentum")


def test_install_succeeds_when_old_backup_cleanup_fails(
    tmp_path: Path, factor_panel: pd.DataFrame, monkeypatch
):
    root = tmp_path / "factors"
    write_factor_cache(factor_panel, root)
    original_rmtree = factor_cache_module.shutil.rmtree

    def fail_old_backup(path, *args, **kwargs):
        if ".old-" in str(path):
            raise OSError("simulated old backup cleanup failure")
        return original_rmtree(path, *args, **kwargs)

    monkeypatch.setattr(factor_cache_module.shutil, "rmtree", fail_old_backup)
    manifest = write_factor_cache(factor_panel.assign(mom_20=np.float32(9.0)), root)

    assert manifest["catalog_hash"] == catalog_hash(FACTOR_CATALOG)
    assert load_family(root, "momentum")["mom_20"].tolist() == [9.0, 9.0, 9.0, 9.0]
    assert not list(tmp_path.glob(".factors.tmp-*"))


def test_materialize_projects_selected_family_columns(
    tmp_path: Path, factor_panel: pd.DataFrame, monkeypatch
):
    write_factor_cache(factor_panel, tmp_path)
    original_read_parquet = factor_cache_module.pd.read_parquet
    calls = []

    def record_read(path, *args, **kwargs):
        calls.append((Path(path).name, kwargs.get("columns")))
        return original_read_parquet(path, *args, **kwargs)

    monkeypatch.setattr(factor_cache_module.pd, "read_parquet", record_read)
    out = materialize_selected_panel(
        tmp_path,
        [{"name": "mom_20", "direction": 1}, {"name": "mom_60", "direction": -1}],
    )

    assert list(out.columns[-2:]) == ["mom_20", "mom_60"]
    assert ("momentum.parquet", ["trade_date", "symbol", "mom_20", "mom_60"]) in calls
    assert all(
        not (name == "momentum.parquet" and columns and "mom_5" in columns)
        for name, columns in calls
    )


def test_compute_factors_has_no_groupby_apply_deprecation_warning():
    dates = pd.bdate_range("2020-01-01", periods=30)
    rows = []
    for symbol_idx, symbol in enumerate(("AAA", "BBB")):
        for date_idx, date in enumerate(dates):
            close = 100.0 + date_idx + symbol_idx
            rows.append(
                {
                    "trade_date": date,
                    "symbol": symbol,
                    "open": close - 0.5,
                    "high": close + 1.0,
                    "low": close - 1.0,
                    "close": close,
                    "volume": 1000.0 + date_idx,
                    "amount": 100000.0 + date_idx,
                    "is_suspended": False,
                }
            )
    prices = pd.DataFrame(rows)
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        actual = features_module.compute_factors(prices)

    expected_parts = [
        features_module._compute_single_symbol(group)
        for _, group in prices.groupby("symbol", sort=True)
    ]
    expected = features_module._cross_sectional_zscore(
        pd.concat(expected_parts, axis=0), features_module.FACTOR_NAMES
    )
    expected = expected[
        ["trade_date", "symbol", "ret_1d", "is_suspended", *features_module.FACTOR_NAMES]
    ].reset_index(drop=True)
    pd.testing.assert_frame_equal(actual, expected)


def test_materialize_retains_cache_row_order(tmp_path: Path, factor_panel: pd.DataFrame):
    shuffled = factor_panel.sample(frac=1.0, random_state=7).reset_index(drop=True)
    write_factor_cache(shuffled, tmp_path)
    out = materialize_selected_panel(tmp_path, [{"name": "mom_20", "direction": 1}])

    pd.testing.assert_frame_equal(
        out[[*BASE_COLUMNS, "mom_20"]],
        shuffled[[*BASE_COLUMNS, "mom_20"]],
        check_dtype=True,
    )
