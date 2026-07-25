"""
Shared config fixtures for the RL tests.

These are plain functions rather than pytest fixtures, matching the style of the rest of the test suite, and they
return fresh copies so a test can edit one without disturbing another.
"""
from copy import deepcopy

# The smallest thing that exercises the whole pipeline: two environments, eight steps each.
SMOKE_TRAIN_CONFIG = {
    "benchmark": "minigrid",
    "env": "MiniGrid-Empty-5x5-v0",
    "obs_mode": "symbolic",
    "n_envs": 2,
    "vec_env": "dummy",
    "seed": 12345,
    "total_timesteps": 48,
    "eval_freq": 16,
    "save_freq": 16,
    "record_freq": 16,
    "eval_episodes": 2,
    "eval_n_envs": 1,
    "algo": "PPO",
    "algo_args": {"n_steps": 8, "batch_size": 16, "n_epochs": 1},
    "optimizer": "Adam",
    "optimizer_args": {"lr": 2.5e-4},
}

CONV_TRUNK = {
    "Assembly": {
        "parts": [{"Net": {"model_name": "convnet", "pretrained": False, "x_dim": 3, "num_blocks": 3,
                           "num_filters": [16, 32, 64], "kernel_size": 2, "stride": 1, "padding": 0,
                           "pool_size": None, "norm_type": None, "in_format": "img", "out_format": "img"}}],
        "head": {"FeatureHead": {"pooled_size": [4, 4], "activation": "relu"}},
    }
}


def smoke_config(**overrides):
    """ A complete, minimal RL config. Keyword arguments are merged into the top level. """
    config = {
        "trunk": deepcopy(CONV_TRUNK),
        # normalize_images is left unset on purpose: it should be derived from obs_mode.
        "policy": {"features_dim": 64},
        "train_config": deepcopy(SMOKE_TRAIN_CONFIG),
        "device": "cpu",
        "eval_checkpoints": True,
        "save_checkpoints": False,
    }
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(config.get(key), dict):
            config[key].update(value)
        else:
            config[key] = value
    return config


def validated(**overrides):
    """ A smoke config which has been through validation, so all defaults are filled in. """
    from rl_train import validate_config
    return validate_config(smoke_config(**overrides), print_config=False)
