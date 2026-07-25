"""
Default algorithm hyperparameters per benchmark.

These are adapted from RL Baselines3 Zoo's tuned values:
https://github.com/DLR-RM/rl-baselines3-zoo/blob/master/hyperparams/ppo.yml

IMPORTANT: the Zoo tuned these for MiniGrid with `FlatObsWrapper` observations and an MlpPolicy. We use them as the
starting point for the other observation modes as well, but for those they are *derived*, not tuned. Treat them as a
reasonable default rather than as a reference configuration; the only setup here that reproduces a tuned Zoo run is
`obs_mode: flat`.
"""

# RL Zoo's `minigrid-defaults` anchor, minus the parts that are not algorithm arguments (n_envs, n_timesteps,
# policy, env_wrapper, normalize). Its learning rate, 2.5e-4, is applied in `algo.check_algo_config()` instead,
# since the optimizer is configured separately.
MINIGRID_PPO = {
    "n_steps": 128,
    "batch_size": 64,
    "n_epochs": 10,
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "clip_range": 0.2,
    "ent_coef": 0.0,
    "vf_coef": 0.5,
    "max_grad_norm": 0.5,
}

MINIGRID_LR = 2.5e-4

ALGO_DEFAULTS = {
    ("minigrid", "PPO"): MINIGRID_PPO,
}

LR_DEFAULTS = {
    ("minigrid", "PPO"): MINIGRID_LR,
}


def algo_defaults(benchmark, algo):
    """ Default `algo_args` for the given benchmark and algorithm; empty if we have no tuned values for it. """
    return dict(ALGO_DEFAULTS.get((benchmark, algo), {}))


def default_lr(benchmark, algo):
    """ Default learning rate for the given benchmark and algorithm. """
    return LR_DEFAULTS.get((benchmark, algo), 3e-4)  # 3e-4 is SB3's own default for PPO.
