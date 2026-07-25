"""
Environments for reinforcement learning, organized by "benchmark".

A benchmark is the analogue of a dataset family in the supervised code: it says which package registers the
environment ids, which observation modes make sense, how to tell a successful episode from a failed one, and which
hyperparameters to start from. The environment id itself is a separate config value, so adding a second benchmark
does not disturb the first.

    benchmark: minigrid
    env: BabyAI-GoToRedBallNoDists-v0
"""
import logging
from typing import Callable

from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.preprocessing import is_image_space, is_image_space_channels_first
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize, VecTransposeImage

from utils import ensure_config_param, gt_zero, of_type, one_of

VEC_ENV_CLASSES = {"dummy": DummyVecEnv, "subproc": SubprocVecEnv}


class Benchmark:
    """ A family of environments which share an observation format, a success criterion, and hyperparameters. """

    name = None
    supported_obs_modes = ()
    default_obs_mode = None

    def register(self):
        """ Import whatever package registers this benchmark's environment ids with Gymnasium. """
        raise NotImplementedError

    def wrapper_class(self, obs_mode, obs_kwargs=None) -> Callable:
        """ Return a callable which applies this benchmark's wrappers for the given observation mode to one env. """
        raise NotImplementedError

    def is_success(self, ep_info) -> bool:
        """ Whether a finished episode, as summarized by a Monitor `episode` info dict, counts as a success. """
        raise NotImplementedError

    def __repr__(self):
        return f"<{type(self).__name__} '{self.name}'>"


class MiniGridBenchmark(Benchmark):
    """
    MiniGrid and BabyAI (https://minigrid.farama.org). Observations are natively a dict of an image, a direction, and
    a mission string; every mode here reduces that to a single array, discarding the mission. That is deliberate: the
    environments we target state their mission as a constant string ("go to the red ball"), so nothing is lost, and
    it keeps the model a plain vision model. Language conditioning would need a dict observation and a policy which
    can consume it.

    Observation modes:
      symbolic  (7, 7, 3) uint8 of (OBJECT_IDX, COLOR_IDX, STATE) codes. The native BabyAI input, and the fastest.
      rgb       The agent's partial view rendered to pixels, (7*tile_size, 7*tile_size, 3) uint8. Slower, but the
                only mode where a pretrained image backbone is meaningful.
      rgb_full  The whole grid rendered to pixels, rather than just the agent's view. Removes partial observability.
      flat      The symbolic view flattened and concatenated with a one-hot encoding of the mission, for MLPs. This
                is the observation RL Zoo tuned its MiniGrid hyperparameters against.
    """

    name = "minigrid"
    supported_obs_modes = ("symbolic", "rgb", "rgb_full", "flat")
    default_obs_mode = "symbolic"

    def register(self):
        import minigrid  # noqa: F401  (importing is what registers the env ids)

    def wrapper_class(self, obs_mode, obs_kwargs=None):
        from minigrid.wrappers import (FlatObsWrapper, ImgObsWrapper, RGBImgObsWrapper, RGBImgPartialObsWrapper)
        obs_kwargs = dict(obs_kwargs or {})

        if obs_mode == "symbolic":
            return lambda env: ImgObsWrapper(env)
        if obs_mode == "rgb":
            tile_size = obs_kwargs.pop("tile_size", 8)
            return lambda env: ImgObsWrapper(RGBImgPartialObsWrapper(env, tile_size=tile_size))
        if obs_mode == "rgb_full":
            tile_size = obs_kwargs.pop("tile_size", 8)
            return lambda env: ImgObsWrapper(RGBImgObsWrapper(env, tile_size=tile_size))
        if obs_mode == "flat":
            max_str_len = obs_kwargs.pop("max_str_len", 96)
            return lambda env: FlatObsWrapper(env, maxStrLen=max_str_len)
        raise ValueError(f"Unrecognized observation mode: '{obs_mode}'.")

    def is_success(self, ep_info):
        # MiniGrid pays out `1 - 0.9 * (step_count / max_steps)` on success and exactly 0 otherwise, so any positive
        # return means the agent solved the task.
        return ep_info["r"] > 0


BENCHMARKS = {b.name: b for b in [MiniGridBenchmark()]}


def get_benchmark(name):
    if name not in BENCHMARKS:
        raise ValueError(f"Unrecognized benchmark: '{name}'. Choose one of: {sorted(BENCHMARKS)}")
    return BENCHMARKS[name]


def add_env_args(parser):
    """ Add the environment arguments, the RL counterpart of `datasets.add_dataset_arg()`. """
    parser.add_argument("--benchmark", type=str.lower, choices=sorted(BENCHMARKS),
                        help="The benchmark (environment family) to train on.")
    parser.add_argument("--env", "--env-id", dest="env", metavar="ID",
                        help="The environment to train on, e.g. BabyAI-GoToRedBallNoDists-v0.")
    parser.add_argument("--obs-mode", type=str.lower, help="How to present observations to the model.")
    parser.add_argument("-n", "--n-envs", type=int, metavar="N",
                        help="Number of environments to step in parallel. This is how RL parallelizes; there is no"
                             " distributed (DDP) mode.")
    parser.add_argument("--vec-env", type=str.lower, choices=sorted(VEC_ENV_CLASSES),
                        help="Whether to run the parallel environments in this process ('dummy') or in worker"
                             " subprocesses ('subproc'). Subprocesses only pay off when stepping is expensive.")
    return parser


def check_env_config(config):
    """ Validate and fill in the environment portion of the config. Modifies `config` in place. """
    ensure_config_param(config, ["train_config", "benchmark"], one_of(sorted(BENCHMARKS)), dflt="minigrid")
    train_cfg = config["train_config"]
    benchmark = get_benchmark(train_cfg["benchmark"])

    ensure_config_param(config, ["train_config", "env"], of_type(str))
    ensure_config_param(config, ["train_config", "env_kwargs"], of_type(dict), required=False)
    ensure_config_param(config, ["train_config", "obs_mode"], one_of(list(benchmark.supported_obs_modes)),
                        dflt=benchmark.default_obs_mode)
    ensure_config_param(config, ["train_config", "obs_kwargs"], of_type(dict), required=False)
    ensure_config_param(config, ["train_config", "max_episode_steps"], gt_zero, required=False)
    ensure_config_param(config, ["train_config", "n_envs"], gt_zero, dflt=8)
    ensure_config_param(config, ["train_config", "vec_env"], one_of(sorted(VEC_ENV_CLASSES)), dflt="dummy")
    ensure_config_param(config, ["train_config", "normalize_obs"], of_type(bool), dflt=False)
    ensure_config_param(config, ["train_config", "normalize_reward"], of_type(bool), dflt=False)
    ensure_config_param(config, ["train_config", "eval_env"], of_type(str), required=False)
    ensure_config_param(config, ["train_config", "eval_episodes"], gt_zero, dflt=20)
    ensure_config_param(config, ["train_config", "eval_n_envs"], gt_zero, dflt=1)
    ensure_config_param(config, ["train_config", "eval_seed_offset"], of_type(int), dflt=10000)

    # Running statistics only make sense for real-valued observations. On uint8 images VecNormalize would both
    # destroy the image dtype and fight with the policy's own /255 scaling.
    if train_cfg["normalize_obs"] and train_cfg["obs_mode"] != "flat":
        raise RuntimeError(f"normalize_obs is only supported for obs_mode 'flat', but obs_mode is "
                           f"'{train_cfg['obs_mode']}'. Normalizing uint8 image observations is not meaningful.")

    # Fail here, with the list of ids to look at, rather than deep inside env construction.
    benchmark.register()
    import gymnasium as gym
    try:
        gym.spec(train_cfg["env"])
    except Exception as e:
        raise RuntimeError(f"Environment '{train_cfg['env']}' is not registered for benchmark "
                           f"'{benchmark.name}' (e.g. 'BabyAI-GoToRedBallNoDists-v0'). Original error: {e}")


def make_vec_envs(config, n_envs=None, seed=None, is_eval=False, render_mode=None):
    """
    Build one vectorized environment according to the config.

    Args:
        config: The full training config.
        n_envs: (Optional) Override the number of parallel environments.
        seed: (Optional) Override the seed.
        is_eval: Whether this is the evaluation environment, which uses `eval_env` if one is configured.
        render_mode: (Optional) Gymnasium render mode, e.g. "rgb_array" to capture frames for a video.
    Returns:
        VecEnv: The vectorized environment.
    """
    train_cfg = config["train_config"]
    benchmark = get_benchmark(train_cfg["benchmark"])
    benchmark.register()

    env_id = (train_cfg.get("eval_env") or train_cfg["env"]) if is_eval else train_cfg["env"]
    n_envs = n_envs if n_envs is not None else train_cfg["n_envs"]
    seed = seed if seed is not None else train_cfg["seed"]

    env_kwargs = dict(train_cfg.get("env_kwargs") or {})
    if train_cfg.get("max_episode_steps"):
        env_kwargs["max_episode_steps"] = train_cfg["max_episode_steps"]
    if render_mode:
        env_kwargs["render_mode"] = render_mode

    # `make_vec_env` applies a Monitor to each env, which is what populates `ep_info_buffer` with the episode
    # returns and lengths that our training metrics are derived from.
    venv = make_vec_env(
        env_id,
        n_envs=n_envs,
        seed=seed,
        env_kwargs=env_kwargs,
        wrapper_class=benchmark.wrapper_class(train_cfg["obs_mode"], train_cfg.get("obs_kwargs")),
        vec_env_cls=VEC_ENV_CLASSES[train_cfg["vec_env"]] if n_envs > 1 else DummyVecEnv,
    )
    return maybe_transpose_images(venv)


def maybe_transpose_images(venv):
    """
    Convert channels-last image observations to channels-first, which is what Torch models expect.

    SB3 does this automatically for the training environment, but not for one we construct ourselves, so apply it
    explicitly using SB3's own criteria to guarantee the two environments agree.
    """
    space = venv.observation_space
    if is_image_space(space) and not is_image_space_channels_first(space):
        return VecTransposeImage(venv)
    return venv


def make_train_and_eval_envs(config):
    """
    Build the training and evaluation environments.

    The evaluation environment is seeded differently from the training one so that evaluation episodes are not the
    same layouts the agent just trained on.

    Returns:
        tuple: (train_env, eval_env). `eval_env` is None if evaluation is turned off.
    """
    train_cfg = config["train_config"]
    train_env = make_vec_envs(config)

    eval_env = None
    if config.get("eval_checkpoints", True):
        eval_env = make_vec_envs(config, n_envs=train_cfg["eval_n_envs"],
                                 seed=train_cfg["seed"] + train_cfg["eval_seed_offset"], is_eval=True)

    if train_cfg["normalize_obs"] or train_cfg["normalize_reward"]:
        train_env = VecNormalize(train_env, norm_obs=train_cfg["normalize_obs"],
                                 norm_reward=train_cfg["normalize_reward"])
        if eval_env is not None:
            # Share the running statistics with the training env, but never update them from evaluation episodes,
            # and never normalize the rewards we are reporting.
            eval_env = VecNormalize(eval_env, training=False, norm_obs=train_cfg["normalize_obs"],
                                    norm_reward=False)
            eval_env.obs_rms = train_env.obs_rms

    logging.info(f"Training on {train_cfg['env']} ({train_cfg['benchmark']}, obs_mode={train_cfg['obs_mode']}) with "
                 f"{train_cfg['n_envs']} parallel envs. Observation space: {train_env.observation_space}, "
                 f"action space: {train_env.action_space}.")
    return train_env, eval_env
