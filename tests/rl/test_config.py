from copy import deepcopy

import pytest

from rl_train import create_arg_parser, prep_config, validate_config
from tests.rl.configs import CONV_TRUNK, smoke_config, validated

EXAMPLE_CONFIGS = ["tests/rl-ppo-conv4-minigrid.yml", "tests/rl-ppo-mlp-minigrid-flat.yml"]


def test_defaults_are_filled_in():
    # A config should only have to name what it wants to change.
    minimal = {"trunk": deepcopy(CONV_TRUNK),
               "train_config": {"env": "MiniGrid-Empty-5x5-v0", "seed": 1},
               "device": "cpu"}
    config = validate_config(minimal, print_config=False)

    train_cfg = config["train_config"]
    assert train_cfg["benchmark"] == "minigrid"
    assert train_cfg["obs_mode"] == "symbolic"
    assert train_cfg["algo"] == "PPO"
    # Straight from RL Zoo's tuned MiniGrid values.
    assert train_cfg["algo_args"]["n_steps"] == 128
    assert train_cfg["algo_args"]["gae_lambda"] == 0.95
    assert train_cfg["optimizer_args"]["lr"] == 2.5e-4
    # MiniGrid is small enough that CPU is the sensible default.
    assert config["device"] == "cpu"
    assert config["policy"]["net_arch"] == {"pi": [64, 64], "vf": [64, 64]}


def test_symbolic_observations_are_not_scaled_by_255():
    # SB3 cannot tell MiniGrid's (7, 7, 3) uint8 index codes apart from pixels, but they max out around 10.
    assert validated(train_config={"obs_mode": "symbolic"})["policy"]["normalize_images"] is False


def test_rgb_observations_are_scaled_by_255():
    assert validated(train_config={"obs_mode": "rgb"})["policy"]["normalize_images"] is True


def test_explicit_normalize_images_wins_over_the_default():
    assert validated(policy={"normalize_images": True},
                     train_config={"obs_mode": "symbolic"})["policy"]["normalize_images"] is True


def test_batch_size_must_divide_the_rollout():
    # Otherwise SB3 silently drops a truncated minibatch from every update.
    with pytest.raises(RuntimeError, match="must divide"):
        validate_config(smoke_config(train_config={"n_envs": 3, "algo_args": {"n_steps": 8, "batch_size": 16}}),
                        print_config=False)


def test_reporting_frequencies_are_aligned_to_rollouts():
    # num_timesteps only ever lands on multiples of the rollout size, so a frequency which is not a multiple of it
    # would never fire.
    config = validated(train_config={"n_envs": 2, "algo_args": {"n_steps": 8, "batch_size": 16},
                                     "eval_freq": 17, "save_freq": 100, "total_timesteps": 1000})
    rollout = 2 * 8
    assert config["train_config"]["eval_freq"] == 32, "17 should round up to the next multiple of 16."
    assert config["train_config"]["save_freq"] % rollout == 0
    assert config["train_config"]["record_freq"] % rollout == 0


def test_training_must_run_at_least_one_rollout():
    with pytest.raises(RuntimeError, match="less than one rollout"):
        validate_config(smoke_config(train_config={"total_timesteps": 4}), print_config=False)


def test_trunk_accepts_the_supervised_spellings():
    # A config written for stitch_train.py should be usable here unchanged.
    for alias in ("trunk", "model"):
        config = smoke_config()
        config[alias] = config.pop("trunk")
        assert validate_config(config, print_config=False)[alias]


def test_ambiguous_trunk_is_rejected():
    config = smoke_config()
    config["model"] = deepcopy(config["trunk"])
    with pytest.raises(RuntimeError, match="more than one model"):
        validate_config(config, print_config=False)


def test_missing_trunk_is_rejected():
    config = smoke_config()
    del config["trunk"]
    with pytest.raises(RuntimeError, match="No model found"):
        validate_config(config, print_config=False)


def test_top_level_part_must_accept_the_runtime_arguments():
    # A bare Net would receive num_classes and forward it into the architecture, failing confusingly deep down.
    config = smoke_config()
    config["trunk"] = {"Net": {"model_name": "convnet", "pretrained": False}}
    with pytest.raises(RuntimeError, match="must accept"):
        validate_config(config, print_config=False)


def test_resume_is_reported_as_unsupported():
    config = smoke_config(resume_from="/tmp/checkpoint.pth")
    with pytest.raises(RuntimeError, match="not implemented yet"):
        validate_config(config, print_config=False)


def test_unknown_optimizer_is_rejected():
    with pytest.raises(RuntimeError, match="Unrecognized optimizer"):
        validate_config(smoke_config(train_config={"optimizer": "NotAnOptimizer"}), print_config=False)


@pytest.mark.parametrize("config_path", EXAMPLE_CONFIGS)
def test_example_configs_are_valid(config_path):
    parser = create_arg_parser("test")
    args = parser.parse_args(["-c", config_path])
    config = prep_config(parser, args)
    assert config["train_config"]["env"]
    assert config["policy"]["features_dim"] > 0


def test_command_line_overrides_reach_every_level():
    # The three nesting levels each have their own override list; make sure all of them are wired up.
    parser = create_arg_parser("test")
    args = parser.parse_args(["-c", EXAMPLE_CONFIGS[0], "--env", "MiniGrid-Empty-5x5-v0", "--lr", "1e-3",
                              "-b", "16", "--ppo-epochs", "3", "--features-dim", "32", "-t", "2048"])
    config = prep_config(parser, args)
    assert config["train_config"]["env"] == "MiniGrid-Empty-5x5-v0"        # train_config
    assert config["train_config"]["total_timesteps"] == 2048               # train_config
    assert config["train_config"]["algo_args"]["batch_size"] == 16         # algo_args
    assert config["train_config"]["algo_args"]["n_epochs"] == 3            # algo_args
    assert config["train_config"]["optimizer_args"]["lr"] == 1e-3          # optimizer_args
    assert config["policy"]["features_dim"] == 32                          # policy


def test_smoke_test_flag_shrinks_the_run():
    parser = create_arg_parser("test")
    config = prep_config(parser, parser.parse_args(["-c", EXAMPLE_CONFIGS[0], "--st"]))
    train_cfg = config["train_config"]
    assert train_cfg["total_timesteps"] == 48, "A smoke test should be three tiny rollouts, not three real ones."
    assert train_cfg["n_envs"] == 2
    assert config["save_checkpoints"] and config["eval_checkpoints"]
