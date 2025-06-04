"""
A script to assemble a number of experiment configs and launch Slurm jobs to execute them.

To test this script, try:
    python src/stitch_train.py -c across-scales/resnet-50.yml -n
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
from assembly import Assembly, validate_part, validate_part_list
from utils import as_strings, ensure_config_param, make_pretty, _and, num_params, num_trainable_params, of_type
from utils.slurm import call_sbatch


ADAPTERS = [
    # TODO
]

# Get the resolved path of this script, before we switch directories.
SCRIPT_DIR = Path(__file__).parent.resolve()


def build_command(cluster, conda_env, config_path, verbosity, launcher_args):
    # Find the script to run, relative to this file.
    target_script = SCRIPT_DIR / "stitch_train.py"
    assert target_script.is_file(), f"Script file ({target_script}) not found or is not a file."
    sbatch_script = SCRIPT_DIR.parent / ("hgtrain.sbatch" if cluster == "hgnodes" else "train.sbatch")
    assert sbatch_script.is_file(), f"Script file ({sbatch_script}) not found or is not a file."

    train_cmd = [target_script, "--config", config_path]
    if verbosity:
        train_cmd.append("-" + ("v" * verbosity))
    # TODO: Add metric output file to command.

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
                        help="Config for all scale experiments to run.")
    datasets.add_dataset_arg(parser, dflt_data_dir=Path(__file__).parent.parent / "data")

    # Output/checkpoint args.
    parser.add_argument("--print-freq", default=10, type=int, metavar="N", help="Print frequency.")
    parser.add_argument("--save-checkpoints", action="store_true",
                        help="Save the model weights at the end of each epoch.")
    parser.add_argument("--no-eval-checkpoints", dest="eval_checkpoints", action="store_false",
                        help="Do not evaluate each checkpoint on the entire train/test set. This can speed up training"
                             " but the downside is that you will be relying on training batches only for tracking the"
                             " progress of training.")
    parser.add_argument("-o", "--output", "--dest", metavar="FOLDER", dest="save_dir", type=Path,
                        default=Path("experiments/across-scales").resolve(), help="Root location for all experiments.")

    # Distributed/hardware args.
    parser.add_argument("-b", "--batch-size", default=256, type=int, metavar="N",
                        help="Mini-batch size. When distributed, the total batch size is (num GPUs * batch size).")
    parser.add_argument("-m", "--max-batches", type=int, metavar="N",
                        help="Maximum number of batches. Useful for quick testing/debugging.")
    argutils.add_device_arg(parser)
    parser.add_argument("--deterministic", action="store_true", help="Use only deterministic algorithms.")

    # Other/Launcher Arguments
    parser.add_argument("--cluster", metavar="NAME", default="nvgpu", choices=["nvgpu", "hgnodes"],
                        help="The Slurm partition on which to launch eval jobs.")
    parser.add_argument("--conda-env", "--conda", "--env", metavar="NAME", default="stitch",
                        help="The Conda environment to activate before running the job.")
    parser.add_argument("-n", "--dry-run", action="store_true",
                        help="Do not actually launch jobs, but only print out the equivalent commands that would be"
                             " launched.")
    parser.add_argument("--lv", "--launch-verbose", dest="launch_verbose", action="store_true",
                        help="Be verbose when launching the job (output all the launcher print statements).")

    # Other args.
    argutils.add_seed_arg(parser, default_seed=1)
    argutils.add_wandb_args(parser, allow_id=allow_id)
    argutils.add_verbose_arg(parser)
    return parser


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
    ensure_config_param(config, "assembly", _and(of_type(list), validate_part_list))
    ensure_config_param(config, "head", _and(of_type(dict), validate_part), required=False)

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
    config = argutils.load_config_from_args(parser, args, ["config", "data_path", "print_freq", "save_checkpoints",
                                                           "eval_checkpoints", "save_dir", "id", "project",
                                                           "entity", "group", "device", "workers", "deterministic",
                                                           "verbose"])
    if not config.get("train_config"):
        # Exits the program with a usage error.
        parser.error(f'The given config does not have a "train_config" sub-config: {args.config}')
    # This list governs which _training_ args can be overridden from the command line.
    config["train_config"] = argutils.override_from_command_line(config["train_config"], parser, args,
                                                                 ["dataset", "seed", "batch_size", "max_batches",
                                                                  "data_augmentation"])
    return validate_config(config)


def setup_jobs(config, args, launcher_args):
    """ Write configs for each job and launch them. """
    # create folder based on config name.
    expname = config["config"].stem
    rootdir = Path(config["save_dir"]).resolve() / expname
    result = 0
    for gap in config["gaps"]:
        blocks_to_drop = gap["blocks_to_drop"]
        num_downsamples = gap["num_downsamples"]
        for adapter in ADAPTERS:
            # Make a folder for the job.
            jobname = training.filesafe_str(["train", expname, "gap", blocks_to_drop, "adapter", adapter.__name__])
            outdir = rootdir / jobname
            if not args.dry_run:
                outdir.mkdir(parents=True, exist_ok=True)
                os.chdir(outdir)
                print(f"Run folder: {outdir}")
            else:
                print(f"Would create output folder: {outdir}. Subsequent actions would be relative to this folder"
                      f" instead of {os.getcwd()}.")

            # Do not overwrite anything if results already exist here.
            result_path = outdir / "results.pkl"
            if result_path.exists():
                print(f"Results already exist for {expname}/{jobname}. Skipping.")
                continue

            # Write a config (if doesn't exist).
            cfgfile = outdir / "config.yml"
            if not cfgfile.is_file():
                jobcfg = config.copy()
                jobcfg.pop("stages", None)
                jobcfg.pop("src_stages", None)
                jobcfg.pop("dest_stages", None)
                jobcfg.pop("gaps", None)
                # TODO write assembly parts
                jobcfg = make_pretty(jobcfg)
                if not args.dry_run:
                    cfgfile = cfgfile.resolve()
                    with open(cfgfile, "w") as f:
                        yaml.dump(jobcfg, f)
                else:
                    print(f"Would write training config to file: {str(outdir / cfgfile.name)}")
                    if args.launch_verbose:
                        print(f"\nConfig to be written:\n{jobcfg}\n\n")

            # Write the metadata (if doesn't exist).
            metafile = outdir / "metadata.yml"
            if not metafile.is_file():
                metacfg = {"arch": expname, "first_deleted_block": blocks_to_drop[0],
                           "last_deleted_block": blocks_to_drop[1], "num_downsamples": num_downsamples,
                           "adapter": adapter.__name__}
                metacfg = make_pretty(metacfg)
                if not args.dry_run:
                    metafile = metafile.resolve()
                    with open(metafile, "w") as f:
                        yaml.dump(metacfg, f)
                else:
                    print(f"Would write metadata to file: {str(outdir / metafile.name)}")
                    if args.launch_verbose:
                        print(f"\nMetadata to be written:\n{metacfg}\n\n")

            # Get the launch command.
            command = build_command(args.cluster, args.conda_env, cfgfile, args.verbose, launcher_args)

            # Launch the job.
            result += call_sbatch(command, args.launch_verbose, args.dry_run)

    return result


def main(argv=None):
    parser = create_arg_parser(__doc__)
    args, launcher_args = parser.parse_known_args(argv)

    config = prep_config(parser, args)
    return setup_jobs(config, args, launcher_args)


if __name__ == "__main__":
    sys.exit(main())
