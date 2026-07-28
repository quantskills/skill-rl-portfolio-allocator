"""save_allocations 必须按 trade_date 整体覆盖,而非按 symbol 残留旧批次。"""
import pandas as pd
from scripts.allocate import save_allocations


def _mk(update_time, symbols, weight, side="long", td="2024-12-31"):
    return pd.DataFrame([
        {
            "trade_date": pd.Timestamp(td), "symbol": s, "weight": weight,
            "side": side, "factor_weights": "{}", "composite_score": 0.0,
            "strategy_id": "x", "data_version": "v", "update_time": update_time,
        }
        for s in symbols
    ])


def test_rerun_same_date_replaces_not_accumulates(tmp_path):
    p = tmp_path / "alloc.parquet"

    # 第一次:选股票池 A,long 加总 = 1.0
    first = _mk("2026-01-01T00:00:00+00:00", [f"A{i}" for i in range(10)], 0.1)
    save_allocations(first, str(p))

    # 第二次(同一 trade_date):选了不同的股票池 B,long 加总 = 1.0
    second = _mk("2026-01-02T00:00:00+00:00", [f"B{i}" for i in range(10)], 0.1)
    save_allocations(second, str(p))

    out = pd.read_parquet(p)
    long_sum = out[out.side == "long"]["weight"].sum()
    # 若按 symbol 残留旧批次,A0..A9 会和 B0..B9 一起留下 -> long_sum = 2.0 (BUG)
    assert long_sum == 1.0, f"expected 1.0, got {long_sum} (old batch leaked)"
    assert len(out) == 10
