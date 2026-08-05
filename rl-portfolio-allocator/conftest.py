import sys
import pathlib

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).parent))


@pytest.fixture(autouse=True)
def _clear_rlpa_tuning_env(monkeypatch):
    """run_pipeline.sh 导出的调参变量不应影响测试结果。"""
    for name in (
        "RLPA_REWARD_CANDIDATES",
        "RLPA_SELECTION_TARGET_COUNT",
        "RLPA_LAMBDA_DRAWDOWN",
        "RLPA_LAMBDA_TURNOVER",
        "RLPA_LAMBDA_CONCENTRATION",
    ):
        monkeypatch.delenv(name, raising=False)
