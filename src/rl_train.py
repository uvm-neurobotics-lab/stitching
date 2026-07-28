"""
A script to train an assembled architecture with reinforcement learning.

This is the sibling of `stitch_train.py`: same config format, same model construction, same result files. What
changes is that instead of a dataset it takes a benchmark and an environment, instead of supervised optimization
parameters it takes RL ones, and progress is measured in environment timesteps rather than epochs.

To test this script, try:
    WANDB_MODE=disabled python src/rl_train.py -c tests/rl-ppo-conv4-minigrid.yml --st
Parallelism comes from stepping many environments at once (`--n-envs`), not from torchrun; running this under
torchrun with more than one process would just start several identical runs fighting over the same output files.
"""

import logging
import os
import sys
from pathlib import Path

import yaml

import rl.training as algo
import rl.envs as envs
import rl.policies as policies
import rl.video as video
import utils.argparsing as argutils
import utils.distributed as dist
import utils.training as training
from stitch_train import get_result_file, save_results
from utils import make_pretty

SCRIPT_DIR = Path(__file__).parent.resolve()

NUM_CORES = os.cpu_count()
if hasattr(os, "sched_getaffinity"):
    # This function is only available on certain platforms. When running with Slurm, it can tell us the true
    # number of cores we have access to.
    NUM_CORES = len(os.sched_getaffinity(0))


def create_arg_parser(desc, allow_abbrev=True, allow_id=True):
    """
    Creates the argument parser for this program.

    Args:
        desc (str): The human-readable description for the arg parser.
        allow_abbrev (bool): The `allow_abbrev` argument to `argparse.ArgumentParser()`.
        allow_id (bool): The `allow_id` argument to the `argutils.add_wandb_args()` function.

    Returns:
        argutils.ArgParser: The parser.
    """
    parser = argutils.create_parser(desc, allow_abbrev=allow_abbrev)
    parser.add_argument("-c", "--config", metavar="FILE", type=argutils.existing_path, required=True,
                        help="Training config file.")
    envs.add_env_args(parser)

    # Output/checkpoint args.
    parser.add_argument("--print-freq", default=10, type=int, metavar="N",
                        help="Print frequency, counted in recorded rollouts.")
    parser.add_argument("--save-checkpoints", action="store_true", help="Save the model weights periodically.")
    parser.add_argument("--checkpoint-initial-model", action="store_true",
                        help="Evaluate the randomly-initialized model before any updates.")
    parser.add_argument("--no-eval-checkpoints", dest="eval_checkpoints", action="store_false",
                        help="Do not periodically evaluate the policy on a held-out environment. This speeds up"
                             " training, but the only remaining measure of progress is the return the behavior"
                             " policy happens to earn while exploring.")
    parser.add_argument("-o", "--output", "--dest", metavar="FOLDER", dest="save_dir", type=Path,
                        default=Path(".").resolve(),
                        help="Location to save the model checkpoints. By default, they will be saved in the current "
                             "directory.")
    parser.add_argument("--metrics-output", "--metrics-dest", metavar="PATH", type=Path,
                        help="Location to save a dataframe of recorded metrics. (default: result.pkl in --output dir)")
    parser.add_argument("--no-video", dest="record_video", action="store_false",
                        help="Do not record a video of the final policy. By default one is written to a 'video'"
                             " folder next to the results.")
    parser.add_argument("--video-episodes", type=int, metavar="N", default=10,
                        help="Number of episodes to record at the end of training.")
    parser.add_argument("--video-fps", type=int, metavar="N", default=6, help="Frames per second of recorded video.")
    parser.add_argument("--resume-from", "--resume", metavar="FILE", type=argutils.existing_path,
                        help="Not yet supported for RL; use --load-from.")
    parser.add_argument("--load-from", "--weights", metavar="FILE", type=argutils.existing_path,
                        help="Path of a checkpoint to load the whole policy from -- trunk, actor, and critic. Must "
                             "be a checkpoint written by rl_train.py; training resumes from those weights but at "
                             "step zero.")
    parser.add_argument("--non-strict", dest="strict_load", action="store_false",
                        help="Use non-strict matching of weights when loading the checkpoint.")
    parser.add_argument("--unfrozen", action="store_true", help="Train all parameters. Overrides config.")
    parser.add_argument("--test-only", action="store_true", help="Only evaluate the model.")

    # Hardware args.
    parser.add_argument("-j", "--workers", default=NUM_CORES, type=int, metavar="N",
                        help="Maximum number of worker processes, which caps --n-envs when --vec-env is 'subproc'.")
    argutils.add_device_arg(parser)
    parser.add_argument("--deterministic", action="store_true", help="Use only deterministic algorithms.")

    # Optimization args.
    parser.add_argument("-t", "--timesteps", "--total-timesteps", dest="total_timesteps", type=int, metavar="N",
                        help="Total number of environment steps to train for.")
    parser.add_argument("--eval-freq", type=int, metavar="N", help="Evaluate every N environment steps.")
    parser.add_argument("--save-freq", type=int, metavar="N", help="Checkpoint every N environment steps.")
    parser.add_argument("--eval-episodes", type=int, metavar="N", help="Number of episodes per evaluation.")
    parser.add_argument("--n-steps", type=int, metavar="N",
                        help="Steps to collect from each environment per rollout. One rollout is n_envs * n_steps.")
    parser.add_argument("--ppo-epochs", dest="n_epochs", type=int, metavar="N",
                        help="Number of passes the algorithm makes over each rollout. Distinct from the number of"
                             " environment steps, which is --timesteps.")
    parser.add_argument("-b", "--batch-size", type=int, metavar="N",
                        help="Minibatch size for each update. Must divide n_envs * n_steps.")
    parser.add_argument("--gamma", type=float, metavar="VAL", help="Discount factor.")
    parser.add_argument("--ent-coef", type=float, metavar="VAL",
                        help="Entropy bonus weight. Raise this if the policy stops exploring too early.")
    parser.add_argument("--clip-range", type=float, metavar="VAL", help="PPO clipping parameter.")
    parser.add_argument("--lr", "--learning-rate", type=float, metavar="RATE", help="Learning rate for the optimizer.")
    parser.add_argument("--wd", "--weight-decay", type=float, metavar="VAL", dest="weight_decay",
                        help="Weight decay for the optimizer, if applicable.")
    parser.add_argument("--momentum", type=float, metavar="VAL", help="Momentum for the optimizer, if applicable.")
    parser.add_argument("--features-dim", type=int, metavar="N",
                        help="Width of the feature vector the trunk produces for the policy heads.")

    # Other args.
    argutils.add_seed_arg(parser, default_seed=1)
    argutils.add_wandb_args(parser, allow_id=allow_id)
    argutils.add_verbose_arg(parser)
    parser.add_argument("--st", "--smoke-test", dest="smoke_test", action="store_true",
                        help="Conduct a quick, full test of the training pipeline. If enabled, then a number of"
                             " arguments will be overridden to make the training run as short as possible and print in"
                             " verbose/debug mode.")
    return parser


def validate_config(config, print_config=True):
    """
    Prints and validates the given training config. Throws an exception in the case of invalid or missing required
    values. Non-required missing values are filled in with their defaults; note that this modifies the config in-place.

    Args:
        config: A config dict which describes the hyperparams, environment, etc. for training an architecture.
        print_config: Whether to print the config for reference.
    Returns:
        The config after validation (the same instance as was passed in).
    """
    if print_config:
        # Output config for reference. Do it before checking config to assist debugging.
        logging.info("\n------- Config -------\n" + yaml.dump(make_pretty(config)) + "----------------------")

    # The model is described exactly as it is for supervised training; `policies.trunk_key()` accepts the "model" and
    # "assembly" spellings as aliases of "trunk", so an existing config can be used unchanged.
    policies.trunk_key(config)
    ensure_workers(config)
    envs.check_env_config(config)
    policies.check_policy_config(config)
    algo.check_algo_config(config)

    return config


def ensure_workers(config):
    """ Cap the number of parallel environments by the number of worker processes we are allowed to use. """
    from utils import ensure_config_param, of_type

    ensure_config_param(config, "workers", of_type(int), dflt=NUM_CORES)
    train_cfg = config["train_config"]
    if train_cfg.get("vec_env") == "subproc" and train_cfg.get("n_envs", 0) > config["workers"]:
        logging.warning(f"Reducing n_envs from {train_cfg['n_envs']} to {config['workers']}, the number of workers "
                        "available. Pass -j to raise the limit.")
        train_cfg["n_envs"] = config["workers"]


def prep_config(parser, args):
    """ Process command line arguments to produce a full training config. May also edit the arguments. """
    # If we're doing a smoke test, then we need to modify the verbosity before configuring the logger.
    if args.smoke_test and args.verbose < 2:
        args.verbose = 2

    argutils.configure_logging(args, level=logging.INFO)

    # This list governs which _top-level_ args can be overridden from the command line.
    config = argutils.load_config_from_args(parser, args, ["print_freq", "unfrozen", "save_checkpoints",
                                                           "eval_checkpoints", "checkpoint_initial_model", "load_from",
                                                           "resume_from", "strict_load", "test_only", "save_dir",
                                                           "metrics_output", "id", "project", "entity", "group",
                                                           "device", "workers", "deterministic", "verbose",
                                                           "record_video", "video_episodes", "video_fps"])
    if not config.get("train_config"):
        # Exits the program with a usage error.
        parser.error(f'The given config does not have a "train_config" sub-config: {args.config}')
    # This list governs which _training_ args can be overridden from the command line.
    config["train_config"] = argutils.override_from_command_line(config["train_config"], parser, args,
                                                                 ["benchmark", "env", "obs_mode", "n_envs", "vec_env",
                                                                  "seed", "total_timesteps", "eval_freq", "save_freq",
                                                                  "eval_episodes"])
    # Special option to override some algorithm parameters.
    algoconf = config["train_config"].setdefault("algo_args", {})
    config["train_config"]["algo_args"] = argutils.override_from_command_line(
        algoconf, parser, args, ["batch_size", "n_steps", "n_epochs", "gamma", "ent_coef", "clip_range"])
    # Special option to override some optimization parameters.
    optconf = config["train_config"].setdefault("optimizer_args", {})
    config["train_config"]["optimizer_args"] = argutils.override_from_command_line(
        optconf, parser, args, ["lr", "weight_decay", "momentum"])
    # And the width of the interface between the trunk and the policy heads.
    config["policy"] = argutils.override_from_command_line(config.setdefault("policy", {}), parser, args,
                                                           ["features_dim"])

    # Conduct a quick test. Every value here is forced rather than scaled, because the point is to exercise the
    # whole pipeline in seconds: at the configured settings a single rollout can be thousands of environment steps.
    if args.smoke_test:
        config["save_checkpoints"] = True
        config["eval_checkpoints"] = True
        config["checkpoint_initial_model"] = False
        train_cfg = config["train_config"]
        train_cfg["n_envs"] = 2
        train_cfg["eval_episodes"] = 2
        train_cfg["eval_n_envs"] = 1
        train_cfg["video_episodes"] = 1  # Still record, so the smoke test covers that path too.
        train_cfg["algo_args"].update({"n_steps": 8, "batch_size": 16, "n_epochs": 1})
        train_cfg["total_timesteps"] = 3 * 2 * 8  # Three rollouts, mirroring the supervised smoke test's 3 batches.
        train_cfg["eval_freq"] = train_cfg["save_freq"] = train_cfg["record_freq"] = 2 * 8

    return validate_config(config)


def setup_and_train(parser, config):
    """ Setup W&B, build the environments, build the model, and commence training. """
    resfile = get_result_file(config)
    if resfile.exists():
        logging.warning(f"WARNING: Will overwrite existing result file: {resfile}")

    # RL parallelizes over environments, not processes; N copies of this script would be N independent runs racing
    # to write the same result file, not one distributed run.
    if int(os.environ.get("WORLD_SIZE", 1)) > 1:
        parser.error("Distributed (torchrun) execution is not supported for RL training. Parallelism comes from"
                     " stepping many environments at once; use --n-envs instead.")
    config["distributed"] = False

    device = argutils.get_device(parser, config)
    argutils.set_seed(config["train_config"]["seed"])
    argutils.prepare_wandb(config)

    logging.info("Creating environments.")
    train_env, eval_env = envs.make_train_and_eval_envs(config)

    try:
        logging.info("Constructing model.")
        sb3_model = algo.build_model(config, train_env, device)

        raw_metrics = algo.train(config, sb3_model, eval_env, device)
    finally:
        train_env.close()
        if eval_env is not None:
            eval_env.close()

    # Save before recording, so that a problem while rendering cannot cost us the run's results.
    save_results(training.metrics_to_dataframe(raw_metrics), resfile)
    video.record_after_training(config, sb3_model, resfile)
    return 0


def main(argv=None):
    parser = create_arg_parser(__doc__)
    args = parser.parse_args(argv)

    config = prep_config(parser, args)

    try:
        return setup_and_train(parser, config)
    finally:
        dist.tear_down_distributed_mode()


if __name__ == "__main__":
    sys.exit(main())
