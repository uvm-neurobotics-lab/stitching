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


def test_checkpoint_holds_the_whole_policy(tmp_path):
    # One entry, not two: "model" is the entire SB3 policy, so there is no second copy of the weights to keep in
    # sync with it.
    checkpoint = torch.load(run(tmp_path) / "checkpoint.pth", map_location="cpu", weights_only=True)
    assert set(checkpoint) == {"model", "optimizer", "step", "config"}

    keys = list(checkpoint["model"])
    assert any(k.startswith("features_extractor.") for k in keys), "The trunk should be in there..."
    assert any(k.startswith("action_net.") for k in keys), "...along with the actor..."
    assert any(k.startswith("value_net.") for k in keys), "...and the critic."


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


def test_recurrent_ppo_trains_and_saves_its_lstm(tmp_path):
    import yaml
    with open(CONFIG) as f:
        config = yaml.safe_load(f)
    config["train_config"]["algo"] = "RecurrentPPO"
    config.setdefault("policy", {})["lstm_hidden_size"] = 32
    config_path = tmp_path / "recurrent.yml"
    config_path.write_text(yaml.dump(config))

    os.environ["WANDB_MODE"] = "disabled"
    out = tmp_path / "out"
    assert main(["-c", str(config_path), "--st", "--env", "MiniGrid-Empty-5x5-v0", "-o", str(out)]) == 0
    keys = list(torch.load(out / "checkpoint.pth", map_location="cpu", weights_only=True)["model"])
    assert any(k.startswith("lstm_actor.") for k in keys), "The checkpoint should include the actor's LSTM."
    assert list((out / "video").glob("*.mp4")), "Recording must thread the LSTM state through predict()."


def test_lstm_options_are_rejected_without_a_recurrent_algo():
    from rl_train import validate_config
    from tests.rl.configs import smoke_config
    with pytest.raises(RuntimeError, match="only apply to a recurrent algorithm"):
        validate_config(smoke_config(policy={"features_dim": 64, "lstm_hidden_size": 32}), print_config=False)


def test_stochastic_eval_is_recorded_alongside_the_deterministic_one(tmp_path):
    import yaml
    with open(CONFIG) as f:
        config = yaml.safe_load(f)
    config["train_config"]["stochastic_eval"] = True
    config_path = tmp_path / "stochastic.yml"
    config_path.write_text(yaml.dump(config))

    os.environ["WANDB_MODE"] = "disabled"
    out = tmp_path / "out"
    assert main(["-c", str(config_path), "--st", "--env", "MiniGrid-Empty-5x5-v0", "-o", str(out)]) == 0
    df = pd.read_pickle(out / "result.pkl")
    for name in ("Reward", "Episode Length", "Success Rate"):
        assert f"Eval/{name}" in df.columns
        assert f"Eval/Stochastic {name}" in df.columns
