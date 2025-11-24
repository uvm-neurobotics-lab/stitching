"""
A script to train an assembled architecture.

To test this script, try:
    WANDB_MODE=disabled python src/stitch_train.py -c tests/stitch-mobilenetv3.yml --st
To run distributed, use torchrun and specify number of workers. For example, on a single node with 8 GPUs and 48 CPUs:
    WANDB_MODE=disabled torchrun --nproc-per-node=8 src/stitch_train.py -c tests/stitch-mobilenetv3.yml -j 6
"""

import logging
import os
import sys
from pathlib import Path

import yaml
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler

import utils.argparsing as argutils
import utils.datasets as datasets
import utils.distributed as dist
import utils.training as training
from assembly import model_from_config, validate_part, validate_part_list
from utils import as_strings, ensure_config_param, make_pretty, _and, num_params, num_trainable_params, of_type

# Get the resolved path of this script, before we switch directories.
SCRIPT_DIR = Path(__file__).parent.resolve()

NUM_CORES = os.cpu_count()
if hasattr(os, "sched_getaffinity"):
    # This function is only available on certain platforms. When running with Slurm, it can tell us the true
    # number of cores we have access to.
    NUM_CORES = len(os.sched_getaffinity(0))


def build_command(hardware, conda_env, config_path, seed, result_file, verbosity, launcher_args):
    """
    Builds an `sbatch` call suitable for launching this script on a Slurm cluster. Once built, the command can be
    passed to `utils.slurm.call_sbatch()`.
    Args:
        hardware: The type of hardware to launch on (actually this just maps to the pre-baked sbatch scripts in the
                 same directory as this script, and is specifically based on UVM's Slurm cluster).
        conda_env: The name of the conda environment to activate before running the script.
        config_path: The path of the config to pass to --config.
        seed: The seed to use for --seed.
        result_file: The path or filename to use for --metrics-output.
        verbosity: The verbosity level to run at.
        launcher_args: Arguments to be passed on to `sbatch`.

    Returns:
        A list of strings which can be used as an argument to `subprocess.run()`.
    """
    # Find the script to run, relative to this file.
    target_script = SCRIPT_DIR / "stitch_train.py"
    assert target_script.is_file(), f"Script file ({target_script}) not found or is not a file."
    if hardware == "v100":
        sbatch_filename = "nvtrain.sbatch"
    elif hardware == "h100":
        sbatch_filename = "hgtrain.sbatch"
    elif hardware == "h200":
        sbatch_filename = "h2train.sbatch"
    else:
        raise RuntimeError(f"Unrecognized hardware: {hardware}")
    sbatch_script = SCRIPT_DIR.parent / sbatch_filename
    assert sbatch_script.is_file(), f"SBATCH file ({sbatch_script}) not found or is not a file."

    # NOTE: We allow launching multiple different seeds from the same config, so supply these on the command line.
    train_cmd = [target_script, "--config", config_path, "--seed", seed, "--metrics-output", result_file]
    if verbosity:
        train_cmd.append("-" + ("v" * verbosity))

    # Add launcher wrapper.
    launch_cmd = ["sbatch"] + launcher_args + [sbatch_script, conda_env] + train_cmd
    launch_cmd = as_strings(launch_cmd)

    return launch_cmd


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
    datasets.add_dataset_arg(parser, dflt_data_dir=Path(__file__).parent.parent / "data")

    # Output/checkpoint args.
    parser.add_argument("--print-freq", default=10, type=int, metavar="N", help="Print frequency.")
    parser.add_argument("--save-checkpoints", action="store_true",
                        help="Save the model weights at the end of each epoch.")
    parser.add_argument("--checkpoint-initial-model", action="store_true",
                        help="Evaluate the randomly-initialized model before any updates.")
    parser.add_argument("--no-eval-checkpoints", dest="eval_checkpoints", action="store_false",
                        help="Do not evaluate each checkpoint on the entire train/test set. This can speed up training"
                             " but the downside is that you will be relying on training batches only for tracking the"
                             " progress of training.")
    parser.add_argument("-o", "--output", "--dest", metavar="FOLDER", dest="save_dir", type=Path,
                        default=Path(".").resolve(),
                        help="Location to save the model checkpoints. By default, they will be saved in the current "
                             "directory.")
    parser.add_argument("--metrics-output", "--metrics-dest", metavar="PATH", type=Path,
                        help="Location to save a dataframe of recorded metrics. (default: result.pkl in --output dir)")
    parser.add_argument("--start-epoch", metavar="N", default=0, type=int, help="Start epoch.")
    parser.add_argument("--resume-from", "--resume", metavar="FILE", type=argutils.existing_path,
                        help="Path of checkpoint to resume from. Mutually exclusive with --load-from.")
    parser.add_argument("--load-from", "--weights", metavar="FILE", type=argutils.existing_path,
                        help="Path of checkpoint to load pretrained weights from. Training will NOT be resumed from "
                        "the checkpoint, but rather will begin at --start-epoch. Mutually exclusive with "
                        "--resume-from.")
    parser.add_argument("--test-only", dest="test_only", action="store_true", help="Only test the model.")

    # Distributed/hardware args.
    parser.add_argument("-j", "--workers", default=NUM_CORES, type=int, metavar="N",
                        help="Number of data loading workers. Defaults to the number of cores detected on the current "
                             "node.")
    parser.add_argument("-e", "--epochs", default=10, type=int, metavar="N", help="Number of epochs to train.")
    parser.add_argument("-b", "--batch-size", default=32, type=int, metavar="N",
                        help="Mini-batch size. When distributed, the total batch size is (num GPUs * batch size).")
    parser.add_argument("-m", "--max-batches", type=int, metavar="N",
                        help="Maximum number of batches. Useful for quick testing/debugging.")
    argutils.add_device_arg(parser)
    parser.add_argument("--deterministic", action="store_true", help="Use only deterministic algorithms.")

    # Other args.
    argutils.add_seed_arg(parser, default_seed=1)
    argutils.add_wandb_args(parser, allow_id=allow_id)
    argutils.add_verbose_arg(parser)
    parser.add_argument("--st", "--smoke-test", dest="smoke_test", action="store_true",
                        help="Conduct a quick, full test of the training pipeline. If enabled, then a number of"
                             " arguments will be overridden to make the training run as short as possible and print in"
                             " verbose/debug mode.")
    return parser


def get_result_file(config):
    # Use the save_dir if metrics_output is not specified. Ultimately fall back to CWD.
    resfile = Path(config.get("metrics_output", config.get("save_dir", ".")))
    if not str(resfile).endswith(".pkl"):
        # Assume this is intended as a directory name, even if it doesn't exist.
        resfile = resfile / "result.pkl"
    resfile = resfile.expanduser().resolve()
    return resfile


def validate_config(config):
    """
    Prints and validates the given training config. Throws an exception in the case of invalid or missing required
    values. Non-required missing values are filled in with their defaults; note that this modifies the config in-place.

    Args:
        config: A config dict which describes the hyperparams, dataset, etc. for training an architecture.
    Returns:
        The config after validation (the same instance as was passed in).
    """
    # Output config for reference. Do it before checking config to assist debugging.
    logging.info("\n------- Config -------\n" + yaml.dump(make_pretty(config)) + "----------------------")

    # First check values for constructing the model.
    if "assembly" in config and "model" in config:
        raise RuntimeError('You may only specify one of "assembly" or "model".')
    elif "assembly" in config:
        # Slightly deprecated; this parses an older config format.
        ensure_config_param(config, "assembly", _and(of_type(list), validate_part_list))
        ensure_config_param(config, "head", _and(of_type(dict), validate_part), required=False)
        ensure_config_param(config, "reformat_options", of_type(dict), required=False)
    else:
        # "model" is the new config item, preferred over "assembly".
        ensure_config_param(config, "model", of_type(dict))

    # Now check values related to training the model.
    datasets.check_data_config(config)
    training.check_train_config(config)

    return config


def prep_config(parser, args):
    """ Process command line arguments to produce a full training config. May also edit the arguments. """
    # If we're doing a smoke test, then we need to modify the verbosity before configuring the logger.
    if args.smoke_test and args.verbose < 2:
        args.verbose = 2

    argutils.configure_logging(args, level=logging.INFO)

    # This list governs which _top-level_ args can be overridden from the command line.
    config = argutils.load_config_from_args(parser, args, ["data_path", "print_freq", "save_checkpoints",
                                                           "eval_checkpoints", "checkpoint_initial_model", "load_from",
                                                           "resume_from", "start_epoch", "test_only", "save_dir",
                                                           "metrics_output", "id", "project", "entity", "group",
                                                           "device", "workers", "deterministic", "verbose"])
    if not config.get("train_config"):
        # Exits the program with a usage error.
        parser.error(f'The given config does not have a "train_config" sub-config: {args.config}')
    # This list governs which _training_ args can be overridden from the command line.
    config["train_config"] = argutils.override_from_command_line(config["train_config"], parser, args,
                                                                 ["dataset", "seed", "epochs", "batch_size",
                                                                  "max_batches"])
    # Legacy support for older configs: move "data_augmentation" to be a sub-field of "data_preprocessing":
    if "data_augmentation" in config["train_config"]:
        config["train_config"].setdefault("data_preprocessing", {})
        if "data_augmentation" in config["train_config"]["data_preprocessing"]:
            parser.error(f'Conflict: there are two "data_augmentation" fields found in the config: {args.config}')
        config["train_config"]["data_preprocessing"]["data_augmentation"] = config["train_config"]["data_augmentation"]
        del config["train_config"]["data_augmentation"]
    # Finally, we have a list of which _data augmentation_ args can be overridden from the command line.
    config["train_config"]["data_preprocessing"] = argutils.override_from_command_line(
        config["train_config"].get("data_preprocessing", {}), parser, args, ["image_size", "data_augmentation"])

    # Conduct a quick test.
    if args.smoke_test:
        config["save_checkpoints"] = True
        config["eval_checkpoints"] = True
        config["checkpoint_initial_model"] = False
        config["train_config"]["max_batches"] = 3 * int(os.environ.get("WORLD_SIZE", 1))
        config["train_config"]["epochs"] = 1

    config = validate_config(config)

    resfile = get_result_file(config)
    if resfile.exists():
        logging.warning(f"WARNING: Will overwrite existing result file: {resfile}")

    return config


def setup_and_train(parser, config):
    """ Setup distributed processing, setup W&B, load data, load model, and commence training. """
    dist.init_distributed_mode(config)

    device = argutils.get_device(parser, config)
    argutils.set_seed(config["train_config"]["seed"])
    if dist.is_main_process():
        argutils.prepare_wandb(config)

    logging.info("Loading dataset.")
    train_data, test_data, input_shape, num_classes = datasets.load_dataset_from_config(config)

    if config["distributed"]:
        train_sampler = DistributedSampler(train_data)
        test_sampler = DistributedSampler(test_data, shuffle=False)
    else:
        # noinspection PyTypeChecker
        train_sampler = RandomSampler(train_data)
        # noinspection PyTypeChecker
        test_sampler = SequentialSampler(test_data)

    logging.info(f"Using {config['workers']} workers for data loading.")
    train_loader = DataLoader(train_data, batch_size=config["train_config"]["batch_size"], sampler=train_sampler,
                              num_workers=config["workers"], pin_memory=True, persistent_workers=config["workers"] > 1)
    test_loader = DataLoader(test_data, batch_size=config["train_config"]["batch_size"], sampler=test_sampler,
                             num_workers=config["workers"], pin_memory=True, persistent_workers=config["workers"] > 1)

    logging.info("Constructing model.")
    model = model_from_config(config, input_shape)
    model.to(device)
    logging.info(f"Model has {num_params(model):.3e} total and {num_trainable_params(model):.3e} trainable params.")

    raw_metrics = training.train(config, model, train_loader, {"Test": test_loader}, train_sampler, device)

    pe_metrics = training.per_epoch_metrics(raw_metrics)
    save_results(pe_metrics, config)
    return 0


def save_results(per_epoch_metrics, config):
    resfile = get_result_file(config)
    resfile.parent.mkdir(parents=True, exist_ok=True)
    logging.info(f"Saving results to: {str(resfile)}")
    per_epoch_metrics.to_pickle(resfile)


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
