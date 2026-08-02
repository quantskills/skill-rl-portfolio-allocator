import json

from scripts.train import artifact_paths, train_candidates
from tests.test_train import _toy_env


def test_artifact_paths_are_candidate_specific(tmp_path):
    paths = artifact_paths(tmp_path, fold=2, seed=7, candidate="ppo", schema_version="obs-v1")
    assert paths == {
        "model": tmp_path / "fold2_seed7_ppo.zip",
        "scaler": tmp_path / "fold2_seed7_scaler.json",
        "log": tmp_path / "fold2_seed7_ppo_training.jsonl",
        "metadata": tmp_path / "fold2_seed7_ppo.json",
    }


def test_toy_training_writes_four_complete_artifacts(tmp_path):
    result = train_candidates(
        root=tmp_path,
        fold=1,
        seed=3,
        candidates=("ppo",),
        schema_version="obs-v1",
        raw_train_env=_toy_env(),
        eval_env=_toy_env(),
        total_timesteps=128,
        device="cpu",
        train_range=("2020-01-01", "2020-01-05"),
        val_range=("2020-01-06", "2020-01-08"),
        reward_variant="low",
        buffer_variant="none",
    )
    paths = result["ppo"]
    assert all(path.exists() for path in paths.values())
    assert paths["model"].suffix == ".zip"
    metadata = json.loads(paths["metadata"].read_text())
    for key in ("fold", "seed", "schema_version", "scaler_path", "train_range",
                "val_range", "reward_variant", "buffer_variant", "total_timesteps",
                "best_eval_metric"):
        assert key in metadata
    records = [json.loads(line) for line in paths["log"].read_text().splitlines()]
    assert records
    assert {"timesteps", "rollout_ep_rew_mean", "policy_gradient_loss",
            "value_loss", "entropy_loss", "approx_kl", "clip_fraction",
            "explained_variance", "action_mean", "action_std",
            "action_saturation"} <= records[-1].keys()
