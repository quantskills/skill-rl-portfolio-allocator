import pandas as pd
import pytest

import scripts.train as train


def test_train_main_threads_dynamic_config_to_effective_range(monkeypatch):
    cfg = {
        "factor_names": ["factor_a", "factor_b", "factor_c"],
        "k": 3,
        "start_date": "2024-01-01",
        "end_date": "2024-01-02",
        "train_device": "cpu",
    }
    seen = []

    def fake_range(features_df, market_state_df, start, end, cfg=None):
        seen.append(cfg)
        raise RuntimeError("range-called")

    monkeypatch.setattr("scripts.config.get_config", lambda: cfg)
    monkeypatch.setattr(train.pd, "read_parquet", lambda path: pd.DataFrame())
    monkeypatch.setattr("scripts.env.effective_range", fake_range)
    monkeypatch.setattr("sys.argv", ["train.py"])

    with pytest.raises(RuntimeError, match="range-called"):
        train.main()

    assert seen == [cfg]
