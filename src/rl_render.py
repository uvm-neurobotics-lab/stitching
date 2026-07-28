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

import torch

import rl.training as algo
import rl.envs as envs
import rl.video as video
import rl_train
import utils.argparsing as argutils
from utils import load_yaml

DEFAULT_CHECKPOINT = "checkpoint.pth"


def create_arg_parser(desc, allow_abbrev=True):
    parser = argutils.create_parser(desc, allow_abbrev=allow_abbrev)
    parser.add_argument("run", metavar="RUN", type=argutils.existing_path,
                        help="A run directory produced by rl_train.py, or a checkpoint file directly.")
    parser.add_argument("--checkpoint", "--ckp", metavar="FILENAME", default=DEFAULT_CHECKPOINT,
                        help=f"Which checkpoint within the run directory to load.")
    parser.add_argument("-c", "--config", metavar="FILE", type=argutils.existing_path,
                        help="Config to rebuild the model from. (default: copy stored inside the checkpoint, if"
                             " available; else the config.yml in the run directory)")
    parser.add_argument("-o", "--output", metavar="FOLDER", type=Path,
                        help="Where to write the video. (default: a 'video/' folder inside the run directory)")
    parser.add_argument("-n", "--episodes", default=10, type=int, metavar="N", help="Number of episodes to record.")
    parser.add_argument("--fps", default=6, type=int, metavar="N",
                        help="Frames per second. Lower rates will be easier to follow.")
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
        ckpt_path = run_dir / args.checkpoint
    else:
        run_dir = run.parent
        ckpt_path = run
    if not ckpt_path.is_file():
        parser.error(f"Checkpoint not found: {ckpt_path}")

    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    if "model" not in checkpoint:
        parser.error(f"{ckpt_path} has no 'model' in it, so it does not hold a trained policy.")

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


def setup_and_render(parser, args):
    run_dir, ckpt_path, raw_config = resolve_run(args, parser)
    config = prep_config(raw_config, args)
    train_cfg = config["train_config"]

    device = argutils.get_device(parser, config)
    # Default to the seed the run was evaluated on, so these are unseen layouts rather than training ones.
    seed = args.seed if args.seed is not None else train_cfg["seed"] + train_cfg["eval_seed_offset"]
    argutils.set_seed(seed)

    # Build the model against a throwaway environment, then let `record_policy` make the one it will record.
    setup_env = envs.make_vec_envs(config, n_envs=1, seed=seed)
    try:
        sb3_model = algo.model_from_config(config, setup_env, device)
    finally:
        setup_env.close()
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=True)
    sb3_model.policy.load_state_dict(checkpoint["model"])
    sb3_model.policy.set_training_mode(False)
    logging.info(f"Loaded policy from {ckpt_path}"
                 + (f" (step {checkpoint['step']})" if "step" in checkpoint else ""))

    written = video.record_policy(
        sb3_model, config,
        video_dir=Path(args.output) if args.output else run_dir / "video",
        episodes=args.episodes, fps=args.fps, deterministic=not args.stochastic, seed=seed,
        # Name the video after the checkpoint, unless it is the generic rolling one.
        name_prefix=ckpt_path.stem if ckpt_path.stem != "checkpoint" else train_cfg["env"])
    return 0 if written else 1


def main(argv=None):
    parser = create_arg_parser(__doc__)
    args = parser.parse_args(argv)
    argutils.configure_logging(args, level=logging.INFO)
    return setup_and_render(parser, args)


if __name__ == "__main__":
    sys.exit(main())
