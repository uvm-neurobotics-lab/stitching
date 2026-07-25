"""
End-to-end tests: run the entry point and check what it leaves behind.

This is the regression test for the results contract -- the file names, the columns, and the checkpoint layout that
`utils/postprocess.py`, the notebooks, and a supervised config's `ckp_path` all depend on.
"""
import os

import pandas as pd
import pytest
import torch

from rl_train import main

CONFIG = "tests/rl-ppo-conv4-minigrid.yml"


def run(tmp_path, *extra_args):
    """ Run a smoke-test-sized training run into a temporary directory and return that directory. """
    os.environ["WANDB_MODE"] = "disabled"
    assert main(["-c", CONFIG, "--st", "--env", "MiniGrid-Empty-5x5-v0", "-o", str(tmp_path),
                 *extra_args]) == 0
    return tmp_path


def test_run_produces_the_expected_artifacts(tmp_path):
    run(tmp_path)
    assert (tmp_path / "result.pkl").is_file()
    assert (tmp_path / "checkpoint.pth").is_file(), "The rolling checkpoint is what configs reference by ckp_path."
    assert list(tmp_path.glob("model-*.pth")), "Step-stamped checkpoints should be written alongside it."


def test_results_are_step_indexed_with_rl_metric_names(tmp_path):
    df = pd.read_pickle(run(tmp_path) / "result.pkl")

    assert "Step" in df.columns
    assert "Epoch" not in df.columns, "RL runs are indexed by environment step; there are no epochs."
    assert "Accuracy" not in " ".join(df.columns), "No classification metrics should be invented for RL."
    assert not [c for c in df.columns if c.startswith("Overall/")], "The Overall/ prefix is supervised-only."

    for column in ["Eval/Reward", "Eval/Success Rate", "Eval/Episode Length",
                   "Loss", "Policy Loss", "Value Loss", "Entropy", "LR"]:
        assert column in df.columns, f"Missing expected metric: {column}"

    assert df["Step"].is_monotonic_increasing
    assert df["Step"].iloc[-1] == 48, "Three rollouts of two environments by eight steps."


def test_update_statistics_land_on_the_rollout_that_produced_them(tmp_path):
    # PPO writes its loss statistics after the rollout ends, so they have to be collected on the next hook and
    # recorded against the earlier step. If that is wrong, they end up on their own rows or shifted by one.
    df = pd.read_pickle(run(tmp_path) / "result.pkl").set_index("Step")
    for step in df.index:
        assert pd.notna(df.loc[step, "Loss"]), f"Step {step} has no loss recorded."
        assert pd.notna(df.loc[step, "Eval/Reward"]), f"Step {step} has no evaluation recorded."


def test_results_load_through_the_postprocessing_helpers(tmp_path):
    # The whole point of matching the result format is that the existing analysis code keeps working.
    from utils.postprocess import combine_result_dataframes, last_step_only

    df = pd.read_pickle(run(tmp_path) / "result.pkl")
    combined = combine_result_dataframes([df], [{"model": "conv4"}])
    assert list(combined.index.names) == ["model", "Step"]

    final = last_step_only(combined, require="Eval/Reward")
    assert len(final) == 1
    assert final.index.get_level_values("Step").tolist() == [48]


def test_checkpoint_trunk_is_loadable_by_a_supervised_config(tmp_path):
    # `model` must hold the trunk on its own, in the layout utils/models.load_model() expects from a ckp_path.
    checkpoint = torch.load(run(tmp_path) / "checkpoint.pth", map_location="cpu", weights_only=True)
    assert set(checkpoint) >= {"model", "policy", "optimizer", "step", "config"}

    trunk_keys = list(checkpoint["model"])
    assert all(not k.startswith("features_extractor.") for k in trunk_keys), \
        "The trunk state dict must not carry SB3's policy prefixes."
    assert any(k.startswith("parts.0.net.") for k in trunk_keys), \
        "Expected the Assembly layout that load_model() knows how to strip."
    assert any(k.startswith("features_extractor.") for k in checkpoint["policy"]), \
        "The policy state dict should be the whole SB3 policy, heads included."


def test_load_from_a_previous_run(tmp_path):
    # Training a second run from the first one's weights is the mechanism a curriculum would be built on.
    first = run(tmp_path / "first")
    second = run(tmp_path / "second", "--load-from", str(first / "checkpoint.pth"))
    assert (second / "result.pkl").is_file()


def test_torchrun_is_refused(tmp_path):
    # Under torchrun this would silently become N identical runs racing over one result file.
    os.environ["WORLD_SIZE"] = "2"
    try:
        with pytest.raises(SystemExit):
            main(["-c", CONFIG, "--st", "-o", str(tmp_path)])
    finally:
        del os.environ["WORLD_SIZE"]
