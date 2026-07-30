"""check_data_coverage (§7.0) 单元测试:用合成数据验证覆盖判定,不依赖真实 parquet。"""
import pandas as pd
import pytest

from scripts.check_data_coverage import (
    Fold, check_folds, coverage_of, effective_start,
)


def _daily(start: str, end: str) -> pd.DataFrame:
    # 用工作日近似交易日,足以测试覆盖逻辑。
    return pd.DataFrame({"trade_date": pd.bdate_range(start, end)})


def test_coverage_of_reports_bounds():
    df = _daily("2010-01-01", "2010-12-31")
    cov = coverage_of(df, "x")
    assert cov.start == "2010-01-01"
    assert cov.end == "2010-12-31"
    assert cov.n_days > 250


def test_effective_start_skips_warmup():
    idx = _daily("2010-01-01", "2011-12-31")["trade_date"]
    eff = effective_start(idx, warmup=60)
    # 有效起点应晚于首日约 60 个交易日(~3 个月)。
    assert eff > pd.Timestamp("2010-03-01")
    assert eff < pd.Timestamp("2010-05-01")


def test_index_gap_fails_early_train_segment():
    # features 覆盖 2005+,但 index 仅 2010+;2005 起点的 train 段必须判失败。
    feats = _daily("2005-04-08", "2024-12-31")
    index = _daily("2010-01-05", "2024-12-31")
    folds = [Fold(1, ("2005-01-01", "2012-12-31"),
                  ("2013-01-01", "2014-12-31"), ("2015-01-01", "2016-12-31"))]
    rep = check_folds(feats, index, folds, warmup=60)
    assert rep["fold_ok"][1] is False
    train_chk = next(c for c in rep["segment_checks"] if c["segment"] == "train")
    assert train_chk["covered"] is False
    assert "早于" in train_chk["reason"]


def test_valid_folds_pass():
    feats = _daily("2005-04-08", "2024-12-31")
    index = _daily("2010-01-05", "2024-12-31")
    folds = [
        Fold(1, ("2010-05-01", "2016-12-31"), ("2017-01-01", "2018-12-31"), ("2019-01-01", "2020-12-31")),
        Fold(2, ("2010-05-01", "2020-12-31"), ("2021-01-01", "2021-12-31"), ("2022-01-01", "2022-12-31")),
    ]
    rep = check_folds(feats, index, folds, warmup=60)
    assert rep["usable_folds"] == [1, 2]
    assert all(c["covered"] for c in rep["segment_checks"])


def test_segment_past_data_end_fails():
    feats = _daily("2010-01-01", "2022-12-31")
    index = _daily("2010-01-05", "2022-12-31")
    # test 段越过数据终点。
    folds = [Fold(1, ("2010-05-01", "2018-12-31"),
                  ("2019-01-01", "2020-12-31"), ("2023-01-01", "2024-12-31"))]
    rep = check_folds(feats, index, folds, warmup=60)
    assert rep["fold_ok"][1] is False
    test_chk = next(c for c in rep["segment_checks"] if c["segment"] == "test")
    assert test_chk["covered"] is False
    assert "晚于" in test_chk["reason"]


def test_empty_source_is_rejected():
    empty = pd.DataFrame({"trade_date": pd.Series(dtype="datetime64[ns]")})
    with pytest.raises(ValueError, match="empty"):
        coverage_of(empty, "features")


def test_report_contains_serializable_fold_table():
    feats = _daily("2005-04-08", "2024-12-31")
    index = _daily("2010-01-05", "2024-12-31")
    folds = [
        Fold(1, ("2010-05-01", "2016-12-31"),
             ("2017-01-01", "2018-12-31"),
             ("2019-01-01", "2020-12-31")),
    ]
    rep = check_folds(feats, index, folds, warmup=60)
    assert rep["folds"] == [{
        "fold": 1,
        "train": ["2010-05-01", "2016-12-31"],
        "val": ["2017-01-01", "2018-12-31"],
        "test": ["2019-01-01", "2020-12-31"],
    }]
