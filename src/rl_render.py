"""
A script to record a video of a trained policy.

Point it at a run directory produced by `rl_train.py` and it will replay the saved policy, writing an mp4 and
reporting how each episode went:
    python src/rl_render.py experiments/poc/go-to-red-ball-nodists

Pass a specific checkpoint to watch an earlier stage of training, which is a good way to see a policy improve:
    python src/rl_render.py experiments/poc/go-to-red-ball-nodists --checkpoint model-20480.pth
"""

import logging
import sys
from pathlib import Path

import numpy as np
import torch
from stable_baselines3.common.vec_env import VecVideoRecorder

import rl.algo as algo
import rl.envs as envs
import rl_train
import utils.argparsing as argutils
from rl.envs import get_benchmark
from utils import load_yaml

DEFAULT_CHECKPOINT = "checkpoint.pth"


def create_arg_parser(desc, allow_abbrev=True):
    parser = argutils.create_parser(desc, allow_abbrev=allow_abbrev)
    parser.add_argument("run", metavar="RUN", type=argutils.existing_path,
                        help="A run directory produced by rl_train.py, or a checkpoint file directly.")
    parser.add_argument("--checkpoint", metavar="FILE",
                        help=f"Which checkpoint within the run directory to load. (default: {DEFAULT_CHECKPOINT})")
    parser.add_argument("-c", "--config", metavar="FILE", type=argutils.existing_path,
                        help="Config to rebuild the model from. (default: config.yml in the run directory, else the"
                             " copy stored inside the checkpoint)")
    parser.add_argument("-o", "--output", metavar="FOLDER", type=Path,
                        help="Where to write the video. (default: a 'video' folder inside the run directory)")
    parser.add_argument("-n", "--episodes", default=5, type=int, metavar="N", help="Number of episodes to record.")
    parser.add_argument("--fps", default=8, type=int, metavar="N",
                        help="Frames per second. MiniGrid episodes are short, so a low rate is easier to follow.")
    parser.add_argument("--stochastic", action="store_true",
                        help="Sample from the policy instead of taking its most likely action. This is how the"
                             " policy behaves while training, rather than how it is evaluated.")
    parser.add_argument("--seed", type=int, metavar="N",
                        help="Seed for the environment. (default: the run's evaluation seed, so the episodes are"
                             " ones the agent was scored on rather than ones it trained on)")
    argutils.add_device_arg(parser)
    argutils.add_verbose_arg(parser)
    return parser


def resolve_run(args, parser):
    """
    Work out which checkpoint and which config to use.

    Returns:
        tuple: (run_dir, checkpoint_path, config)
    """
    run = Path(args.run)
    if run.is_dir():
        run_dir = run
        ckpt_path = run_dir / (args.checkpoint or DEFAULT_CHECKPOINT)
    else:
        run_dir = run.parent
        ckpt_path = run
    if not ckpt_path.is_file():
        parser.error(f"Checkpoint not found: {ckpt_path}")

    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    if "policy" not in checkpoint:
        parser.error(f"{ckpt_path} has no 'policy' in it, so it does not hold a trained policy.")

    backup_path = Path(args.config) if args.config else run_dir / "config.yml"
    if args.config:
        cfg_path = Path(args.config)
        if cfg_path.is_file():
            logging.info(f"Loading config from: {cfg_path}")
            config = load_yaml(cfg_path)
        else:
            parser.error(f"Config not found: {cfg_path}.")
    elif "config" in checkpoint:
        # Every checkpoint carries a copy of the config that produced it, so a run directory is not strictly needed.
        logging.info(f"Loading config from checkpoint: {ckpt_path.name}")
        config = checkpoint["config"]
    elif backup_path.is_file():
        logging.info(f"Loading config from: {backup_path}")
        config = load_yaml(backup_path)
    else:
        parser.error(f"No config found. Looked for {backup_path} and inside {ckpt_path.name}.")

    return run_dir, ckpt_path, config


def prep_config(config, args):
    """ Strip out anything that would train, checkpoint, or report; we are only replaying. """
    config = dict(config)
    config["save_checkpoints"] = False
    config["eval_checkpoints"] = False
    config.pop("load_from", None)
    config.pop("resume_from", None)
    config.pop("save_dir", None)
    if args.device:
        config["device"] = args.device

    train_cfg = config["train_config"] = dict(config["train_config"])
    train_cfg["n_envs"] = 1  # One environment, so the video follows a single agent.
    train_cfg["vec_env"] = "dummy"
    # Narrowing to a single environment shrinks the rollout, which can leave the configured batch size no longer
    # dividing it. That constraint protects the update step, and we never take one here, so relax it to match.
    train_cfg["algo_args"] = dict(train_cfg.get("algo_args") or {})
    train_cfg["algo_args"]["batch_size"] = train_cfg["algo_args"].get("n_steps", 1)
    # Validate exactly as training does, so the model is rebuilt the same way it was built.
    return rl_train.validate_config(config, print_config=False)


def rollout(sb3_model, venv, benchmark, episodes, deterministic=True):
    """
    Step the environment until `episodes` episodes have finished.

    Returns:
        list: One dict per episode, with its return, length, and whether it succeeded.
    """
    obs = venv.reset()
    results = []
    ep_reward, ep_length = 0.0, 0
    while len(results) < episodes:
        action, _ = sb3_model.predict(obs, deterministic=deterministic)
        obs, rewards, dones, _ = venv.step(action)
        ep_reward += float(rewards[0])
        ep_length += 1
        if dones[0]:
            results.append({"reward": ep_reward, "length": ep_length,
                            "success": bool(benchmark.is_success({"r": ep_reward}))})
            ep_reward, ep_length = 0.0, 0
    return results


def report(results):
    for i, ep in enumerate(results, start=1):
        outcome = "solved" if ep["success"] else "FAILED"
        logging.info(f"  Episode {i}: {outcome} in {ep['length']} steps, return {ep['reward']:.3f}")
    rewards = [ep["reward"] for ep in results]
    successes = [ep["success"] for ep in results]
    lengths = [ep["length"] for ep in results]
    logging.info(f"Over {len(results)} episodes: success rate {np.mean(successes):.2f}, "
                 f"mean return {np.mean(rewards):.3f}, mean length {np.mean(lengths):.1f}")


def setup_and_render(parser, args):
    run_dir, ckpt_path, raw_config = resolve_run(args, parser)
    config = prep_config(raw_config, args)
    train_cfg = config["train_config"]

    device = argutils.get_device(parser, config)
    # Default to the seed the run was evaluated on, so these are unseen layouts rather than training ones.
    seed = args.seed if args.seed is not None else train_cfg["seed"] + train_cfg["eval_seed_offset"]
    argutils.set_seed(seed)

    video_dir = Path(args.output) if args.output else run_dir / "video"
    name_prefix = ckpt_path.stem if ckpt_path.stem != "checkpoint" else train_cfg["env"]
    final_path = video_dir / f"{name_prefix}.mp4"
    idx = 0
    while final_path.is_file():
        idx += 1
        final_path = video_dir / f"{name_prefix}-{idx}.mp4"

    venv = envs.make_vec_envs(config, n_envs=1, seed=seed, render_mode="rgb_array")
    try:
        sb3_model = algo.model_from_config(config, venv, device)
        checkpoint = torch.load(ckpt_path, map_location=device, weights_only=True)
        sb3_model.policy.load_state_dict(checkpoint["policy"])
        sb3_model.policy.set_training_mode(False)
        logging.info(f"Loaded policy from {ckpt_path}"
                     + (f" (step {checkpoint['step']})" if "step" in checkpoint else ""))

        # `video_length` is a step budget, and we cannot know one in advance: an episode runs until the agent
        # solves it or times out, and a BabyAI level does not even fix its own step limit until reset(), when the
        # limit is derived from the mission it just generated. So give the recorder an effectively unlimited
        # budget and let the rollout loop decide when to stop; closing the recorder is what writes the file.
        venv = VecVideoRecorder(venv, str(video_dir), record_video_trigger=lambda step: step == 0,
                                video_length=10 ** 9, name_prefix=name_prefix)
        # Read from the environment's metadata at construction, so it has to be overridden afterwards.
        venv.frames_per_sec = args.fps

        benchmark = get_benchmark(train_cfg["benchmark"])
        logging.info(f"Recording {args.episodes} episodes of {train_cfg['env']} "
                     f"({'deterministic' if not args.stochastic else 'stochastic'} policy, seed {seed}).")
        results = rollout(sb3_model, venv, benchmark, args.episodes, deterministic=not args.stochastic)
        report(results)
        recorded_path = Path(venv.video_path)
    finally:
        venv.close()  # Closing is what flushes the video to disk.

    if not recorded_path.is_file():
        logging.warning(f"No video was written to {video_dir}.")
        return 1
    # The recorder names the file after the step budget it was given, which is a sentinel here. Give it a stable
    # name instead, so re-rendering replaces the old video rather than piling up next to it.
    recorded_path.replace(final_path)
    logging.info(f"Wrote {final_path}")
    return 0


def main(argv=None):
    parser = create_arg_parser(__doc__)
    args = parser.parse_args(argv)
    argutils.configure_logging(args, level=logging.INFO)
    return setup_and_render(parser, args)


if __name__ == "__main__":
    sys.exit(main())
