"""Atomic, family-partitioned storage for the candidate factor panel."""
from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
import logging
from pathlib import Path
import re
import shutil
import uuid
from numbers import Integral
from typing import Any

import numpy as np
import pandas as pd

from scripts.factor_catalog import CATALOG_VERSION, FACTOR_CATALOG, catalog_hash


BASE_COLUMNS = ("trade_date", "symbol", "ret_1d", "is_suspended")
KEY_COLUMNS = ("trade_date", "symbol")
FACTOR_NAMES = tuple(spec.name for spec in FACTOR_CATALOG)
FAMILY_NAMES = tuple(dict.fromkeys(spec.family for spec in FACTOR_CATALOG))
FAMILY_FACTORS = {
    family: tuple(spec.name for spec in FACTOR_CATALOG if spec.family == family)
    for family in FAMILY_NAMES
}
EXPECTED_CATALOG_HASH = catalog_hash(FACTOR_CATALOG)
_SHA256_RE = re.compile(r"sha256:[0-9a-f]{64}\Z")
_LOGGER = logging.getLogger(__name__)


def _error(message: str) -> ValueError:
    return ValueError(f"factor cache: {message}")


def _validate_panel(panel: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(panel, pd.DataFrame):
        raise TypeError("factor cache: panel must be a pandas DataFrame")

    expected = set(BASE_COLUMNS) | set(FACTOR_NAMES)
    unknown = sorted(set(panel.columns).difference(expected))
    if unknown:
        raise _error(f"unknown columns: {unknown}")
    missing_base = [column for column in BASE_COLUMNS if column not in panel]
    if missing_base:
        raise _error(f"missing base columns: {missing_base}")
    missing_factors = [name for name in FACTOR_NAMES if name not in panel]
    if missing_factors:
        raise _error(f"missing factor columns: {missing_factors}")

    if not pd.api.types.is_datetime64_any_dtype(panel["trade_date"]):
        raise TypeError("factor cache: trade_date must have a datetime dtype")
    if panel["trade_date"].isna().any() or panel["symbol"].isna().any():
        raise _error("key columns may not contain nulls")
    if not panel["symbol"].map(lambda value: isinstance(value, str)).all():
        raise TypeError("factor cache: symbol key must contain strings")
    if panel[list(KEY_COLUMNS)].duplicated().any():
        raise _error("duplicate (trade_date, symbol) row keys")

    if not pd.api.types.is_numeric_dtype(panel["ret_1d"]):
        raise TypeError("factor cache: ret_1d must have a numeric dtype")
    if pd.api.types.is_bool_dtype(panel["ret_1d"]):
        raise TypeError("factor cache: ret_1d must not have a boolean dtype")
    if np.isinf(panel["ret_1d"].to_numpy(dtype=np.float64, copy=False)).any():
        raise _error("ret_1d contains infinity")
    if not pd.api.types.is_bool_dtype(panel["is_suspended"]):
        raise TypeError("factor cache: is_suspended must have a boolean dtype")
    if panel["is_suspended"].isna().any():
        raise _error("is_suspended may not contain nulls")

    for name in FACTOR_NAMES:
        series = panel[name]
        if not pd.api.types.is_float_dtype(series):
            raise TypeError(f"factor cache: factor {name} must have a floating dtype")
        values = series.to_numpy(dtype=np.float64, copy=False)
        if np.isinf(values).any():
            raise _error(f"factor {name} contains infinity")

    columns = [*BASE_COLUMNS, *FACTOR_NAMES]
    prepared = panel.loc[:, columns].copy()
    for name in FACTOR_NAMES:
        prepared[name] = prepared[name].astype(np.float32, copy=False)
    return prepared


def _dtype_name(dtype: Any) -> str:
    return str(pd.api.types.pandas_dtype(dtype))


def _schema(frame: pd.DataFrame) -> dict[str, str]:
    return {column: _dtype_name(frame[column].dtype) for column in frame.columns}


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _keys(frame: pd.DataFrame) -> pd.MultiIndex:
    return pd.MultiIndex.from_frame(frame.loc[:, list(KEY_COLUMNS)], names=list(KEY_COLUMNS))


def _manifest_for(panel: pd.DataFrame) -> dict[str, Any]:
    family_files = {family: f"{family}.parquet" for family in FAMILY_NAMES}
    family_columns = {
        family: [*KEY_COLUMNS, *FAMILY_FACTORS[family]] for family in FAMILY_NAMES
    }
    return {
        "catalog_version": CATALOG_VERSION,
        "catalog_hash": EXPECTED_CATALOG_HASH,
        "row_keys": list(KEY_COLUMNS),
        "row_count": len(panel),
        "row_counts": {"base": len(panel), **{family: len(panel) for family in FAMILY_NAMES}},
        "base_file": "base.parquet",
        "family_files": family_files,
        "files": ["base.parquet", *family_files.values()],
        "factor_count": len(FACTOR_NAMES),
        "families": {
            family: {
                "file": family_files[family],
                "columns": family_columns[family],
                "factor_columns": list(FAMILY_FACTORS[family]),
                "row_count": len(panel),
            }
            for family in FAMILY_NAMES
        },
        "schema": {
            "base": list(BASE_COLUMNS),
            "families": family_columns,
        },
        "factor_catalog": [asdict(spec) for spec in FACTOR_CATALOG],
    }


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _install_atomically(temp_root: Path, root: Path) -> None:
    """Install a complete sibling directory while retaining rollback ability."""
    backup: Path | None = None
    if root.exists():
        if not root.is_dir():
            raise _error(f"cache root is not a directory: {root}")
        backup = root.with_name(f".{root.name}.old-{uuid.uuid4().hex}")
        root.replace(backup)
    try:
        temp_root.replace(root)
    except Exception:
        if backup is not None and not root.exists():
            backup.replace(root)
        raise
    if backup is not None:
        try:
            shutil.rmtree(backup)
        except OSError as exc:
            _LOGGER.warning(
                "factor cache installed at %s; old backup cleanup failed for %s: %s",
                root,
                backup,
                exc,
            )


def write_factor_cache(panel: pd.DataFrame, root: Path) -> dict[str, Any]:
    """Validate and atomically write the complete family-partitioned cache."""
    root = Path(root)
    prepared = _validate_panel(panel)
    root.parent.mkdir(parents=True, exist_ok=True)
    temp_root = root.with_name(f".{root.name}.tmp-{uuid.uuid4().hex}")
    manifest = _manifest_for(prepared)

    try:
        temp_root.mkdir()
        base = prepared.loc[:, list(BASE_COLUMNS)]
        base.to_parquet(temp_root / manifest["base_file"], index=False)
        for family in FAMILY_NAMES:
            columns = [*KEY_COLUMNS, *FAMILY_FACTORS[family]]
            family_frame = prepared.loc[:, columns]
            family_frame.to_parquet(temp_root / manifest["family_files"][family], index=False)

        file_hashes = {
            manifest["base_file"]: _file_hash(temp_root / manifest["base_file"]),
            **{
                manifest["family_files"][family]: _file_hash(
                    temp_root / manifest["family_files"][family]
                )
                for family in FAMILY_NAMES
            },
        }
        manifest["file_hashes"] = file_hashes
        manifest["schema"]["base_dtypes"] = _schema(base)
        manifest["schema"]["family_dtypes"] = {
            family: _schema(prepared.loc[:, [*KEY_COLUMNS, *FAMILY_FACTORS[family]]])
            for family in FAMILY_NAMES
        }
        for family in FAMILY_NAMES:
            manifest["families"][family]["schema"] = manifest["schema"]["family_dtypes"][family]
            manifest["families"][family]["row_count"] = len(prepared)
        _write_json(temp_root / "catalog.json", manifest)
        _install_atomically(temp_root, root)
        return manifest
    except Exception:
        if temp_root.exists():
            shutil.rmtree(temp_root)
        raise


def _read_manifest(root: Path) -> dict[str, Any]:
    root = Path(root)
    catalog_path = root / "catalog.json"
    try:
        manifest = json.loads(catalog_path.read_text())
    except FileNotFoundError:
        raise FileNotFoundError(f"factor cache catalog missing: {catalog_path}") from None
    except (OSError, json.JSONDecodeError) as exc:
        raise _error(f"invalid catalog.json: {exc}") from exc

    if not isinstance(manifest, dict):
        raise _error("catalog.json must contain a JSON object")
    if manifest.get("catalog_version") != CATALOG_VERSION:
        raise _error("catalog version mismatch")
    if manifest.get("catalog_hash") != EXPECTED_CATALOG_HASH:
        raise _error("catalog hash mismatch")
    if manifest.get("row_keys") != list(KEY_COLUMNS):
        raise _error("row_keys manifest mismatch")
    row_count = manifest.get("row_count")
    if not isinstance(row_count, int) or isinstance(row_count, bool) or row_count < 0:
        raise _error("row count manifest missing or invalid")
    expected_row_count_keys = {"base", *FAMILY_NAMES}
    row_counts = manifest.get("row_counts")
    if not isinstance(row_counts, dict) or set(row_counts) != expected_row_count_keys:
        raise _error("row count manifest mismatch")
    for name in expected_row_count_keys:
        value = row_counts.get(name)
        if not isinstance(value, int) or isinstance(value, bool) or value != row_count:
            raise _error("row count manifest mismatch")
    schema = manifest.get("schema")
    expected_schema_keys = {"base", "families", "base_dtypes", "family_dtypes"}
    if not isinstance(schema, dict) or set(schema) != expected_schema_keys:
        raise _error("schema manifest missing or invalid")
    expected_family_columns = {
        family: [*KEY_COLUMNS, *FAMILY_FACTORS[family]]
        for family in FAMILY_NAMES
    }
    if schema["base"] != list(BASE_COLUMNS) or schema["families"] != expected_family_columns:
        raise _error("schema manifest mismatch")
    base_dtypes = schema["base_dtypes"]
    if (
        not isinstance(base_dtypes, dict)
        or set(base_dtypes) != set(BASE_COLUMNS)
        or not all(isinstance(value, str) for value in base_dtypes.values())
    ):
        raise _error("schema base_dtypes manifest mismatch")
    family_dtypes = schema["family_dtypes"]
    if not isinstance(family_dtypes, dict) or set(family_dtypes) != set(FAMILY_NAMES):
        raise _error("schema family_dtypes manifest mismatch")
    for family in FAMILY_NAMES:
        dtypes = family_dtypes[family]
        if (
            not isinstance(dtypes, dict)
            or set(dtypes) != set(expected_family_columns[family])
            or not all(isinstance(value, str) for value in dtypes.values())
        ):
            raise _error(f"schema family_dtypes manifest mismatch: {family}")
    if manifest.get("family_files") != {family: f"{family}.parquet" for family in FAMILY_NAMES}:
        raise _error("family file manifest mismatch")
    if manifest.get("base_file") != "base.parquet":
        raise _error("base file manifest mismatch")
    if manifest.get("factor_count") != len(FACTOR_NAMES):
        raise _error("factor count mismatch")
    expected_factor_catalog = json.loads(
        json.dumps([asdict(spec) for spec in FACTOR_CATALOG])
    )
    if manifest.get("factor_catalog") != expected_factor_catalog:
        raise _error("factor catalog metadata mismatch")
    expected_family_files = {family: f"{family}.parquet" for family in FAMILY_NAMES}
    families = manifest.get("families")
    if not isinstance(families, dict) or set(families) != set(FAMILY_NAMES):
        raise _error("family manifest mismatch")
    for family in FAMILY_NAMES:
        entry = families.get(family)
        if not isinstance(entry, dict):
            raise _error(f"{family} manifest entry missing")
        if entry.get("file") != expected_family_files[family]:
            raise _error(f"{family} file manifest mismatch")
        if entry.get("columns") != [*KEY_COLUMNS, *FAMILY_FACTORS[family]]:
            raise _error(f"{family} column manifest mismatch")
        if entry.get("factor_columns") != list(FAMILY_FACTORS[family]):
            raise _error(f"{family} factor manifest mismatch")
        if "row_count" not in entry:
            raise _error(f"{family} row count manifest missing")
        if entry["row_count"] != row_count or row_counts[family] != entry["row_count"]:
            raise _error(f"{family} row count manifest mismatch")
    file_hashes = manifest.get("file_hashes")
    expected_hash_files = {"base.parquet", *expected_family_files.values()}
    if not isinstance(file_hashes, dict) or set(file_hashes) != expected_hash_files:
        raise _error("file hash manifest mismatch")
    for filename in expected_hash_files:
        _validate_hash_value(filename, file_hashes[filename])
    return manifest


def _validate_base_frame(base: pd.DataFrame, manifest: dict[str, Any]) -> None:
    if list(base.columns) != list(BASE_COLUMNS):
        raise _error("base schema mismatch")
    if len(base) != manifest.get("row_count"):
        raise _error("base row count mismatch")
    _validate_keys_only(base)
    if not pd.api.types.is_numeric_dtype(base["ret_1d"]):
        raise _error("base ret_1d dtype mismatch")
    if pd.api.types.is_bool_dtype(base["ret_1d"]):
        raise _error("base ret_1d dtype mismatch")
    if not pd.api.types.is_bool_dtype(base["is_suspended"]):
        raise _error("base is_suspended dtype mismatch")
    expected_dtypes = manifest.get("schema", {}).get("base_dtypes")
    if expected_dtypes and _schema(base) != expected_dtypes:
        raise _error("base schema dtype mismatch")


def _validate_keys_only(frame: pd.DataFrame) -> None:
    if not pd.api.types.is_datetime64_any_dtype(frame["trade_date"]):
        raise _error("trade_date key dtype mismatch")
    if frame["trade_date"].isna().any() or frame["symbol"].isna().any():
        raise _error("null row key")
    if not frame["symbol"].map(lambda value: isinstance(value, str)).all():
        raise _error("symbol key dtype mismatch")
    if frame[list(KEY_COLUMNS)].duplicated().any():
        raise _error("duplicate row key")


def _validate_file_hash(path: Path, manifest: dict[str, Any]) -> None:
    file_hashes = manifest.get("file_hashes")
    if not isinstance(file_hashes, dict) or path.name not in file_hashes:
        raise _error(f"file hash manifest missing: {path.name}")
    expected = file_hashes[path.name]
    _validate_hash_value(path.name, expected)
    if _file_hash(path) != expected:
        raise _error(f"file hash mismatch: {path.name}")


def _validate_hash_value(filename: str, value: Any) -> None:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise _error(f"invalid file hash for {filename}")


def _read_base(root: Path, manifest: dict[str, Any]) -> pd.DataFrame:
    path = root / manifest["base_file"]
    if not path.exists():
        raise FileNotFoundError(f"factor cache base file missing: {path}")
    _validate_file_hash(path, manifest)
    try:
        base = pd.read_parquet(path)
    except Exception as exc:
        raise _error(f"cannot read base.parquet: {exc}") from exc
    _validate_base_frame(base, manifest)
    return base


def _load_family(
    root: Path,
    manifest: dict[str, Any],
    family: str,
    selected_factors: list[str] | None = None,
) -> pd.DataFrame:
    expected_factors = list(FAMILY_FACTORS[family])
    if selected_factors is None:
        factors_to_read = expected_factors
    else:
        factors_to_read = list(selected_factors)
        if not factors_to_read or not set(factors_to_read) <= set(expected_factors):
            raise _error(f"{family} selected schema mismatch")
    if family not in FAMILY_FACTORS:
        raise _error(f"unknown family: {family}")
    root = Path(root)
    path = root / manifest["family_files"][family]
    if not path.exists():
        raise FileNotFoundError(f"factor cache family missing: {family}")
    _validate_file_hash(path, manifest)
    requested_columns = [*KEY_COLUMNS, *factors_to_read]
    try:
        frame = pd.read_parquet(path, columns=requested_columns)
    except Exception as exc:
        raise _error(f"cannot read family {family}: {exc}") from exc
    if list(frame.columns) != requested_columns:
        raise _error(f"{family} schema mismatch")
    expected_rows = manifest.get("families", {}).get(family, {}).get("row_count")
    if len(frame) != expected_rows:
        raise _error(f"{family} row count mismatch")
    _validate_keys_only(frame)
    for name in factors_to_read:
        if not pd.api.types.is_float_dtype(frame[name]) or frame[name].dtype != np.dtype("float32"):
            raise _error(f"{family} factor dtype mismatch: {name}")
        if np.isinf(frame[name].to_numpy(dtype=np.float64, copy=False)).any():
            raise _error(f"{family} contains infinity")
    expected_dtypes = (
        manifest.get("schema", {}).get("family_dtypes", {}).get(family)
    )
    expected_projected_dtypes = {
        column: expected_dtypes[column]
        for column in requested_columns
    } if expected_dtypes else None
    if expected_projected_dtypes and _schema(frame) != expected_projected_dtypes:
        raise _error(f"{family} schema dtype mismatch")
    return frame


def load_family(root: Path, family: str) -> pd.DataFrame:
    """Load one family as ``trade_date, symbol`` plus all factor columns."""
    if family not in FAMILY_FACTORS:
        raise _error(f"unknown family: {family}")
    root = Path(root)
    manifest = _read_manifest(root)
    return _load_family(root, manifest, family)


def _validate_selection(selected: list[dict]) -> tuple[tuple[str, int], ...]:
    if not isinstance(selected, list):
        raise TypeError("factor cache: selected must be a list")
    result: list[tuple[str, int]] = []
    seen: set[str] = set()
    for item in selected:
        if not isinstance(item, dict) or "name" not in item or "direction" not in item:
            raise TypeError("factor cache: selected entries require name and direction")
        name = item["name"]
        direction = item["direction"]
        if not isinstance(name, str) or name not in FACTOR_NAMES:
            raise _error(f"unknown selected factor: {name}")
        if name in seen:
            raise _error(f"duplicate selected factor: {name}")
        if (
            isinstance(direction, bool)
            or not isinstance(direction, Integral)
            or direction not in (-1, 1)
        ):
            raise _error(f"invalid direction for {name}: {direction}")
        seen.add(name)
        result.append((name, int(direction)))
    return tuple(result)


def materialize_selected_panel(root: Path, selected: list[dict]) -> pd.DataFrame:
    """Read only selected families and return base columns plus ordered factors."""
    choices = _validate_selection(selected)
    root = Path(root)
    manifest = _read_manifest(root)
    base = _read_base(root, manifest)
    base_keys = _keys(base)
    selected_by_family: dict[str, list[str]] = {}
    for name, _ in choices:
        family = next(spec.family for spec in FACTOR_CATALOG if spec.name == name)
        selected_by_family.setdefault(family, []).append(name)
    family_frames: dict[str, pd.DataFrame] = {}
    for family, names in selected_by_family.items():
        family_frame = _load_family(root, manifest, family, names)
        if not _keys(family_frame).equals(base_keys):
            raise _error(f"{family} row keys mismatch with base")
        family_frames[family] = family_frame.set_index(list(KEY_COLUMNS))

    result = base.loc[:, list(BASE_COLUMNS)].copy()
    for name, direction in choices:
        family = next(spec.family for spec in FACTOR_CATALOG if spec.name == name)
        values = family_frames[family][name].reindex(base_keys)
        if len(values) != len(result):
            raise _error(f"{family} factor alignment failed: {name}")
        result[name] = (values.to_numpy(dtype=np.float32, copy=False) * direction).astype(
            np.float32, copy=False
        )
    return result


__all__ = [
    "BASE_COLUMNS",
    "load_family",
    "materialize_selected_panel",
    "write_factor_cache",
]
