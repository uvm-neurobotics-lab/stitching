"""
Constructing and running the reinforcement learning algorithm.

Stable-Baselines3 owns the algorithm and its training loop. This module configures it from the repo's config format
and wires its callbacks up to the repo's logging, so that a run produces the same kind of artifacts as a supervised
one: a `result.pkl`, a `checkpoint.pth`, and W&B metrics.
"""
import logging
import math
from pathlib import Path

import stable_baselines3
import torch

from rl import hyperparams
from rl.logging import RLCallback, RLLog
from rl.envs import get_benchmark
from rl.models import describe_parameters, restore_pretrained_weights
from rl.policies import policy_kwargs_from_config, policy_name_for
from utils import ensure_config_param, gt_zero, gte_zero, of_type, one_of

ALGOS = {"PPO": stable_baselines3.PPO}


def check_algo_config(config):
    """ Validate and fill in the algorithm and optimization portions of the config. Modifies `config` in place. """
    # These top-level keys mean the same thing as they do for supervised training.
    ensure_config_param(config, "verbose", gte_zero, required=False)
    ensure_config_param(config, "save_checkpoints", of_type(bool), required=False)
    ensure_config_param(config, "eval_checkpoints", of_type(bool), required=False)
    ensure_config_param(config, "checkpoint_initial_model", of_type(bool), required=False)
    ensure_config_param(config, "resume_from", of_type((str, Path)), required=False)
    ensure_config_param(config, "load_from", of_type((str, Path)), required=False)
    ensure_config_param(config, "strict_load", of_type(bool), required=False)
    ensure_config_param(config, "test_only", of_type(bool), required=False)
    if "resume_from" in config and "load_from" in config:
        raise RuntimeError("'load_from' and 'resume_from' are mutually exclusive. Supply only one.")
    if config.get("resume_from"):
        raise RuntimeError("Resuming an RL run is not implemented yet: an exact resume also needs the environment "
                           "and rollout buffer state, not just the weights. Use --load-from to start a new run from "
                           "an existing model's weights.")
    ensure_config_param(config, "record_video", of_type(bool), dflt=True)
    ensure_config_param(config, ["train_config", "video_episodes"], gt_zero, dflt=5)
    ensure_config_param(config, ["train_config", "video_fps"], gt_zero, dflt=8)
    ensure_config_param(config, "save_dir", of_type((str, Path)), required=config.get("save_checkpoints"))
    if "save_dir" in config:
        config["save_dir"] = Path(config["save_dir"]).expanduser().resolve()
    # MiniGrid is small enough that PPO is usually faster on CPU than on GPU, so default to it. Asking for CUDA
    # explicitly on a machine without it is still an error, raised by `argutils.get_device()`.
    ensure_config_param(config, "device", of_type(str), dflt="cpu")

    ensure_config_param(config, "train_config", of_type(dict))
    train_cfg = config["train_config"]
    ensure_config_param(config, ["train_config", "seed"], of_type(int))
    ensure_config_param(config, ["train_config", "total_timesteps"], gt_zero, dflt=100_000)
    ensure_config_param(config, ["train_config", "algo"], one_of(sorted(ALGOS)), dflt="PPO")

    # Seed the algorithm arguments from the benchmark's tuned defaults before checking types, so that a config only
    # has to name the values it wants to change.
    train_cfg.setdefault("algo_args", {})
    for key, value in hyperparams.algo_defaults(train_cfg["benchmark"], train_cfg["algo"]).items():
        train_cfg["algo_args"].setdefault(key, value)

    ensure_config_param(config, ["train_config", "algo_args"], of_type(dict))
    ensure_config_param(config, ["train_config", "algo_args", "n_steps"], gt_zero)
    ensure_config_param(config, ["train_config", "algo_args", "batch_size"], gt_zero)
    ensure_config_param(config, ["train_config", "algo_args", "n_epochs"], gt_zero)
    ensure_config_param(config, ["train_config", "algo_args", "gamma"], of_type((int, float)))
    ensure_config_param(config, ["train_config", "algo_args", "gae_lambda"], of_type((int, float)))
    ensure_config_param(config, ["train_config", "algo_args", "clip_range"], of_type((int, float)))
    ensure_config_param(config, ["train_config", "algo_args", "clip_range_vf"], gt_zero, required=False)
    ensure_config_param(config, ["train_config", "algo_args", "normalize_advantage"], of_type(bool), dflt=True)
    ensure_config_param(config, ["train_config", "algo_args", "ent_coef"], gte_zero)
    ensure_config_param(config, ["train_config", "algo_args", "vf_coef"], gte_zero)
    ensure_config_param(config, ["train_config", "algo_args", "max_grad_norm"], gte_zero)
    ensure_config_param(config, ["train_config", "algo_args", "target_kl"], gt_zero, required=False)
    ensure_config_param(config, ["train_config", "algo_args", "stats_window_size"], gt_zero, dflt=100)

    ensure_config_param(config, ["train_config", "optimizer"], of_type(str), dflt="Adam")
    if not hasattr(torch.optim, train_cfg["optimizer"]):
        raise RuntimeError(f"Unrecognized optimizer: '{train_cfg['optimizer']}' is not in torch.optim.")
    ensure_config_param(config, ["train_config", "optimizer_args"], of_type(dict), dflt={})
    ensure_config_param(config, ["train_config", "optimizer_args", "lr"], gt_zero,
                        dflt=hyperparams.default_lr(train_cfg["benchmark"], train_cfg["algo"]))
    ensure_config_param(config, ["train_config", "lr_schedule"], one_of(["constant", "linear"]), dflt="constant")

    check_frequencies(config)


def rollout_size(train_cfg):
    """ How many environment steps one rollout collects, across all parallel environments. """
    return train_cfg["n_envs"] * train_cfg["algo_args"]["n_steps"]


def check_frequencies(config):
    """
    Validate the rollout/minibatch arithmetic and align the reporting frequencies to rollout boundaries.

    Everything happens at rollout boundaries: the policy is only in a consistent state there, and `num_timesteps`
    only ever takes multiples of the rollout size, which is what `BaseLog`'s modulo test needs in order to fire.
    """
    train_cfg = config["train_config"]
    rollout = rollout_size(train_cfg)
    batch_size = train_cfg["algo_args"]["batch_size"]

    if rollout <= 1:
        raise RuntimeError(f"n_envs * n_steps must be greater than 1, but n_envs={train_cfg['n_envs']} and "
                           f"n_steps={train_cfg['algo_args']['n_steps']} give {rollout}.")
    if rollout % batch_size != 0:
        raise RuntimeError(f"batch_size ({batch_size}) must divide n_envs * n_steps ({rollout}), otherwise the last "
                           "minibatch of each update is silently truncated.")

    ensure_config_param(config, ["train_config", "eval_freq"], gt_zero, dflt=max(rollout, 10_000))
    ensure_config_param(config, ["train_config", "save_freq"], gt_zero, dflt=max(rollout, 25_000))
    ensure_config_param(config, ["train_config", "record_freq"], gt_zero, dflt=rollout)
    for name in ("eval_freq", "save_freq", "record_freq"):
        aligned = int(math.ceil(train_cfg[name] / rollout) * rollout)
        if aligned != train_cfg[name]:
            logging.info(f"Rounding {name} up from {train_cfg[name]} to {aligned}, the next multiple of the rollout "
                         f"size ({rollout} steps), since reporting happens at rollout boundaries.")
            train_cfg[name] = aligned

    if train_cfg["total_timesteps"] < rollout:
        raise RuntimeError(f"total_timesteps ({train_cfg['total_timesteps']}) is less than one rollout ({rollout} "
                           "steps), so no update would ever run.")


def lr_schedule_from_config(train_cfg):
    """
    The learning rate, in the form SB3 expects: either a constant or a callable of the remaining progress.

    This deliberately does not go through `utils.optimization`, whose schedulers step per epoch or per batch against
    a torch optimizer we do not own; SB3 recomputes the rate from this callable on every update instead.
    """
    lr = train_cfg["optimizer_args"]["lr"]
    if train_cfg["lr_schedule"] == "linear":
        return lambda progress_remaining: progress_remaining * lr
    return lr


def model_from_config(config, train_env, device):
    """ Construct the algorithm, with the trunk from `trunk`/`model` and the heads from `policy`. """
    train_cfg = config["train_config"]
    algo_cls = ALGOS[train_cfg["algo"]]
    return algo_cls(
        policy_name_for(config),
        train_env,
        learning_rate=lr_schedule_from_config(train_cfg),
        policy_kwargs=policy_kwargs_from_config(config),
        seed=train_cfg["seed"],
        device=device,
        verbose=0,  # We do our own reporting; SB3's would duplicate it in a different format.
        **train_cfg["algo_args"],
    )


def build_model(config, train_env, device):
    """ Construct the algorithm and put its weights into the state training should start from. """
    from rl.models import apply_freezing, load_trunk_weights

    sb3_model = model_from_config(config, train_env, device)

    # Must happen after construction: building the policy re-initializes the features extractor.
    restored = restore_pretrained_weights(sb3_model)
    if restored:
        logging.debug(f"Restored the constructed weights of {restored} features extractor(s) after SB3's "
                      "initialization.")
    if config.get("load_from"):
        load_trunk_weights(sb3_model, config["load_from"], strict=config.get("strict_load", True))
    apply_freezing(sb3_model, config)

    logging.info(describe_parameters(sb3_model))
    return sb3_model


def train(config, sb3_model, eval_env, device):
    """
    Run the algorithm.

    Returns:
        dict: Recorded metrics, keyed by environment timestep, ready for `training.metrics_to_dataframe()`.
    """
    try:
        import wandb
    except ImportError:
        wandb = None

    train_cfg = config["train_config"]
    benchmark = get_benchmark(train_cfg["benchmark"])
    total_timesteps = train_cfg["total_timesteps"]

    log = RLLog(
        benchmark=benchmark,
        expected_steps=total_timesteps,
        eval_episodes=train_cfg["eval_episodes"],
        print_freq=config.get("print_freq", 10) if config.get("verbose", 0) <= 1 else 1,
        save_freq=train_cfg["save_freq"] if config.get("save_checkpoints") else 0,
        eval_freq=train_cfg["eval_freq"] if config.get("eval_checkpoints", True) else 0,
        save_dir=config.get("save_dir"),
        use_wandb=wandb is not None,
        checkpoint_initial_model=config.get("checkpoint_initial_model", False),
    )

    if config.get("test_only"):
        return log.close(0, sb3_model, eval_env, config, should_eval=True, should_save=False)

    callback = RLCallback(log, config, eval_env, train_cfg["record_freq"])
    sb3_model.learn(total_timesteps=total_timesteps, callback=callback, log_interval=None,
                    reset_num_timesteps=True, progress_bar=False)

    return log.close(sb3_model.num_timesteps, sb3_model, eval_env, config,
                     should_eval=bool(config.get("eval_checkpoints", True)),
                     should_save=bool(config.get("save_checkpoints")))
