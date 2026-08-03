import json

import pandas as pd
import pytest

from scripts.check_data_coverage import Fold
from scripts.walk_forward import prepare_fold_factors


def _panel(values):
    return pd.DataFrame({
        "trade_date": pd.to_datetime(["2020-01-01", "2020-01-02", "2020-01-03"]),
        "symbol": ["A", "A", "A"],
        "ret_1d": [0.0, 0.0, 0.0],
        "is_suspended": [False, False, False],
        "mom_20": values,
        "vol_20": [0.1, 0.1, 0.1],
    })


def test_prepare_fold_factors_selects_only_train_data_and_freezes_outputs(tmp_path, monkeypatch):
    import scripts.walk_forward as walk_forward

    panel = _panel([1.0, 2.0, 999.0])
    fold = Fold(7, ("2020-01-01", "2020-01-02"), ("2020-01-03", "2020-01-03"),
                ("2020-01-03", "2020-01-03"))
    seen = []

    class CandidateCache:
        def materialize_selected_panel(self, root, selected):
            return panel.copy()

    def selector(**kwargs):
        seen.append(kwargs["panel"].copy())
        if kwargs["panel"]["mom_20"].iloc[-1] > 0:
            return [{"name": "mom_20", "direction": 1}, {"name": "vol_20", "direction": -1}]
        return [{"name": "vol_20", "direction": -1}, {"name": "mom_20", "direction": 1}]

    monkeypatch.setattr(walk_forward, "build_market_state", lambda features, index_returns, cfg, factor_names: pd.DataFrame({
        "trade_date": pd.to_datetime(["2020-01-01", "2020-01-02", "2020-01-03"]), "state": [1.0, 2.0, 3.0],
    }))
    candidate_cache = CandidateCache()
    cfg = {"state_schema_version": "state-v1"}
    frozen = prepare_fold_factors(
        fold=fold, cache_root=tmp_path / "cache", index_returns=pd.DataFrame(),
        selection_root=tmp_path / "selection", feature_root=tmp_path / "features",
        state_root=tmp_path / "state", cfg=cfg, selector=selector,
        candidate_cache=candidate_cache,
    )

    panel.loc[panel["trade_date"] > pd.Timestamp(fold.train[1]), "mom_20"] = -999.0
    frozen_after_cache_mutation = prepare_fold_factors(
        fold=fold, cache_root=tmp_path / "cache", index_returns=pd.DataFrame(),
        selection_root=tmp_path / "selection", feature_root=tmp_path / "features",
        state_root=tmp_path / "state", cfg=cfg, selector=selector,
        candidate_cache=candidate_cache,
    )

    assert seen[0]["trade_date"].max() == pd.Timestamp(fold.train[1])
    assert frozen.factor_bundle["names"] == ("mom_20", "vol_20")
    assert frozen.factor_bundle["directions"] == (1, -1)
    assert frozen_after_cache_mutation.factor_bundle["names"] == frozen.factor_bundle["names"]
    assert frozen.factor_bundle["factor_contract"]["selected_factors"] == ["mom_20", "vol_20"]
    assert json.loads(frozen.factor_bundle["selection_artifact_path"].read_text())["selected_factors"] == [
        {"name": "mom_20", "direction": 1}, {"name": "vol_20", "direction": -1}
    ]


def test_prepare_fold_factors_materializes_candidates_then_selects_train_panel_then_materializes_selection(tmp_path, monkeypatch):
    import scripts.walk_forward as walk_forward

    panel = pd.concat([_panel([1.0, 2.0, 3.0]), pd.DataFrame({
        "trade_date": [pd.Timestamp("2020-01-04")], "symbol": ["A"],
        "ret_1d": [0.0], "is_suspended": [False], "mom_20": [4.0], "vol_20": [0.1],
    })], ignore_index=True)
    fold = Fold(7, ("2020-01-01", "2020-01-02"), ("2020-01-03", "2020-01-03"),
                ("2020-01-03", "2020-01-03"))
    events = []

    class CandidateCache:
        def materialize_selected_panel(self, root, selected):
            events.append(("materialize", tuple(item["name"] for item in selected)))
            return panel.copy()

    def selector(**kwargs):
        selected_panel = kwargs["panel"]
        events.append((
            "select", kwargs["fold"].fold,
            selected_panel["trade_date"].min(), selected_panel["trade_date"].max(),
            len(selected_panel),
        ))
        return [{"name": "mom_20", "direction": 1}, {"name": "vol_20", "direction": -1}]

    monkeypatch.setattr(walk_forward, "build_market_state", lambda features, index_returns, cfg, factor_names: pd.DataFrame({
        "trade_date": panel["trade_date"], "state": [1.0, 2.0, 3.0, 4.0],
    }))
    prepared = prepare_fold_factors(
        fold=fold, cache_root=tmp_path / "cache", index_returns=pd.DataFrame(),
        selection_root=tmp_path / "selection", feature_root=tmp_path / "features",
        state_root=tmp_path / "state",
        cfg={"state_schema_version": "state-v1"}, selector=selector,
        candidate_cache=CandidateCache(),
    )

    assert events[0][0] == "materialize"
    assert len(events[0][1]) > 2
    assert events[1] == (
        "select", 7, pd.Timestamp(fold.train[0]), pd.Timestamp(fold.train[1]), 2,
    )
    assert events[2] == ("materialize", ("mom_20", "vol_20"))
    assert pd.read_parquet(prepared.factor_bundle["feature_path"])["trade_date"].max() == pd.Timestamp(fold.test[1])
    assert "features" not in prepared.factor_bundle
    assert "market_state" not in prepared.factor_bundle


def test_prepare_fold_factors_rejects_non_twenty_candidate_selection_before_artifacts(tmp_path, monkeypatch):
    import scripts.walk_forward as walk_forward

    fold = Fold(7, ("2020-01-01", "2020-01-02"), ("2020-01-03", "2020-01-03"),
                ("2020-01-03", "2020-01-03"))

    class CandidateCache:
        def materialize_selected_panel(self, root, selected):
            return _panel([1.0, 2.0, 3.0])

    monkeypatch.setattr(walk_forward, "build_market_state", lambda *args, **kwargs: pytest.fail(
        "candidate artifacts must not be materialized for a non-20-factor selection"
    ))

    with pytest.raises(ValueError, match="candidate factor selection must contain exactly 20 ordered factors; got 2"):
        prepare_fold_factors(
            fold=fold, cache_root=tmp_path / "cache", index_returns=pd.DataFrame(),
            selection_root=tmp_path / "selection", feature_root=tmp_path / "features",
            state_root=tmp_path / "state", cfg={"state_schema_version": "state-v1"},
            selector=lambda **_: [
                {"name": "mom_20", "direction": 1},
                {"name": "vol_20", "direction": -1},
            ],
            candidate_cache=CandidateCache(), expected_factor_count=20,
            factor_bundle_name="candidate",
        )

    assert not (tmp_path / "selection").exists()
    assert not (tmp_path / "features").exists()
    assert not (tmp_path / "state").exists()


@pytest.mark.parametrize("dependency_key, dependency", [
    ("selector", lambda **_: []),
    ("candidate_cache", object()),
])
def test_prepare_fold_factors_rejects_config_carried_dependencies(
        tmp_path, dependency_key, dependency):
    fold = Fold(7, ("2020-01-01", "2020-01-02"), ("2020-01-03", "2020-01-03"),
                ("2020-01-03", "2020-01-03"))

    with pytest.raises(ValueError, match="must be passed explicitly"):
        prepare_fold_factors(
            fold=fold, cache_root=tmp_path / "cache", index_returns=pd.DataFrame(),
            selection_root=tmp_path / "selection", feature_root=tmp_path / "features",
            state_root=tmp_path / "state",
            cfg={"state_schema_version": "state-v1", dependency_key: dependency},
        )

    assert not (tmp_path / "selection").exists()
    assert not (tmp_path / "features").exists()
    assert not (tmp_path / "state").exists()
