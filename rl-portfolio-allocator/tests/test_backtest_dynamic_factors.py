import pandas as pd

import scripts.backtest as backtest
from scripts.config import FACTOR_NAMES, get_config
from scripts.state import exogenous_fields


def test_backtest_main_threads_dynamic_config_to_effective_range(monkeypatch):
    names = list(FACTOR_NAMES[:3])
    cfg = get_config()
    cfg.update({
        "factor_names": names,
        "k": len(names),
        "start_date": "2024-01-01",
        "end_date": "2024-01-04",
    })
    dates = pd.date_range("2024-01-01", periods=4, freq="D")
    features = pd.DataFrame({"trade_date": dates})
    market_state = pd.DataFrame(
        0.1, index=dates, columns=exogenous_fields(names)
    ).rename_axis("trade_date").reset_index()
    seen = []

    def fake_read_parquet(path):
        return market_state if path.name == "market_state.parquet" else features

    def fake_run_backtest(*args, **kwargs):
        seen.append((args, kwargs))
        return {"metrics": {}, "warnings": [], "research_ok": False}

    monkeypatch.setattr(backtest, "get_config", lambda: cfg)
    monkeypatch.setattr(backtest.pd, "read_parquet", fake_read_parquet)
    monkeypatch.setattr(backtest, "run_backtest", fake_run_backtest)
    monkeypatch.setattr("sys.argv", ["backtest.py", "--train-ratio", "0.5"])

    backtest.main()

    assert len(seen) == 1
    assert seen[0][0][2] is cfg
