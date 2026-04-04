"""
A script to launch fine-tuning Slurm jobs on multiple different datasets.

To test this script, try:
    python src/launch_multidataset_training.py -c moe-stitch/baseline-resnet-cifar-resisc.yml --seed 12345 -n -vv

To test one of the generated jobs locally, run:
    python src/launch_multidataset_training.py -c moe-stitch/baseline-resnet-cifar-resisc.yml --lv --do-not-launch
    cd experiments/moe/baseline-resnet-cifar-resisc/cars
    python ../../../../src/stitch_train.py -c config.yml --st --device cpu -vv
"""
import logging
import os
import sys
from copy import deepcopy
from pathlib import Path

import yaml

import utils.argparsing as argutils
from stitch_train import build_command, validate_config
from utils import make_pretty, transform_config
from utils.slurm import call_sbatch


def result_rootdir(config) -> Path:
    expname = config.get("exp_name", config["config"].stem)
    return Path(config["save_dir"]).resolve() / config["project"] / expname / config["run_name"]


def result_filename(config):
    fname = "result"
    seed = config.get("train_config", {}).get("seed")
    if seed:
        fname += f"-{seed}"
    return fname + ".pkl"


def create_arg_parser(desc, allow_abbrev=True, allow_id=True):
    """
    Creates the argument parser for this program.

    Args:
        desc (str | None): The human-readable description for the arg parser.
        allow_abbrev (bool): The `allow_abbrev` argument to `argparse.ArgumentParser()`.
        allow_id (bool): The `allow_id` argument to the `argutils.add_wandb_args()` function.

    Returns:
        argutils.ArgParser: The parser.
    """
    parser = argutils.create_parser(desc, allow_abbrev=allow_abbrev)
    parser.add_argument("-c", "--config", metavar="FILE", type=argutils.existing_path, required=True,
                        help="Config for all scale experiments to run.")
    parser.add_argument("--data-path", "--data-dir", metavar="PATH", type=argutils.resolved_path,
                        default=Path(__file__).parent.parent / "data",
                        help="The root path in which to look for the dataset.")
    parser.add_argument("--data-augmentation", action="store_true", help="Whether to enable default data augmentation.")
    parser.add_argument("--image-size", "--resize", type=argutils.image_size,
                        help="Desired image size, if different from the dataset's native size.")

    # Output/checkpoint args.
    parser.add_argument("--print-freq", default=10, type=int, metavar="N", help="Print frequency.")
    parser.add_argument("--save-checkpoints", action="store_true",
                        help="Save the model weights at the end of each epoch.")
    parser.add_argument("--no-eval-checkpoints", dest="eval_checkpoints", action="store_false",
                        help="Do not evaluate each checkpoint on the entire train/test set. This can speed up training"
                             " but the downside is that you will be relying on training batches only for tracking the"
                             " progress of training.")
    parser.add_argument("-o", "--output", "--dest", metavar="FOLDER", dest="save_dir", type=Path,
                        default=Path("experiments").resolve(), help="Root location for all experiments.")

    # Distributed/hardware args.
    parser.add_argument("-b", "--batch-size", default=256, type=int, metavar="N",
                        help="Mini-batch size. When distributed, the total batch size is (num GPUs * batch size).")
    parser.add_argument("-m", "--max-batches", type=int, metavar="N",
                        help="Maximum number of batches. Useful for quick testing/debugging.")
    argutils.add_device_arg(parser)
    parser.add_argument("--deterministic", action="store_true", help="Use only deterministic algorithms.")

    # Other/Launcher Arguments
    parser.add_argument("--hardware", metavar="NAME", default="v100", choices=["v100", "h100", "h200", "preempt"],
                        help="The type of hardware to launch on (actually this just maps to the pre-baked sbatch "
                             "scripts in the same directory as this script, and is specifically based on UVM's Slurm "
                             "cluster).")
    parser.add_argument("--conda-env", "--conda", "--env", metavar="NAME", default="stitch",
                        help="The Conda environment to activate before running the job.")
    parser.add_argument("-f", "--force", action="store_true",
                        help="Run even if results already exist. Overwrite config files if they exist.")
    parser.add_argument("-n", "--dry-run", action="store_true",
                        help="Do not actually launch jobs, but only print out the equivalent commands that would be"
                             " launched.")
    parser.add_argument("--do-not-launch", action="store_true",
                        help="Do not actually launch jobs, but DO generate all the necessary configs.")
    parser.add_argument("--lv", "--launch-verbose", dest="launch_verbose", action="store_true",
                        help="Be verbose when launching the job (output all the launcher print statements).")

    # Other args.
    argutils.add_seed_arg(parser, default_seed=1)
    argutils.add_wandb_args(parser, dflt_project="moe", allow_id=allow_id)
    parser.add_argument("--run-name", "--name", metavar="NAME",
                        help="The name of the run, used to name the result subfolder.")
    argutils.add_verbose_arg(parser)
    return parser


def validate_dataset_config(datasets):
    if not isinstance(datasets, (list, tuple)) or not all([isinstance(d, str) for d in datasets]):
        raise RuntimeError(f"'datasets' should be a list of dataset names.")
    # NOTE: Not currently possible to validate ahead of time, but may be worth adding?
    # for name in datasets:
    #     if name not in DATASETS:
    #         raise RuntimeError(f"Unrecognized dataset name: {name}")
    return True


def prep_config(parser, args, print_config=True):
    """ Process command line arguments to produce a full training config. May also edit the arguments. """
    argutils.configure_logging(args, level=logging.INFO)

    # This list governs which _top-level_ args can be overridden from the command line.
    config = argutils.load_config_from_args(parser, args, ["config", "data_path", "print_freq", "save_checkpoints",
                                                           "eval_checkpoints", "save_dir", "run_name", "id", "project",
                                                           "entity", "group", "device", "workers", "deterministic",
                                                           "verbose"])
    if not config.get("train_config"):
        # Exits the program with a usage error.
        parser.error(f'The given config does not have a "train_config" sub-config: {args.config}')
    # This list governs which _training_ args can be overridden from the command line.
    config["train_config"] = argutils.override_from_command_line(config["train_config"], parser, args,
                                                                 ["seed", "batch_size", "max_batches"])
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

    # A bit hacky: resolve all ckp_path variables to be fully resolved paths. Allows us to write the config anywhere.
    orig_dir = os.getcwd()
    os.chdir(Path(args.config).parent)
    config = transform_config(config, lambda k, v: Path(v).expanduser().resolve() if k == "ckp_path" else v)
    os.chdir(orig_dir)

    return validate_config(config, print_config=print_config, dataset_required=False)


def build_job_config(config, save_dir, dataset):
    jobcfg = deepcopy(config)
    jobcfg.pop("config", None)
    jobcfg.pop("datasets", None)
    jobcfg["train_config"].pop("seed", None)

    jobcfg["train_config"]["dataset"] = dataset
    jobcfg["save_dir"] = save_dir

    return make_pretty(jobcfg)


def setup_and_launch_jobs(config, args, launcher_args):
    """ Write configs for each job and launch them. """
    # create folder based on config name.
    rootdir = result_rootdir(config)
    result = []
    for dataset in config["datasets"]:
        print(f"\n---- LAUNCHING {dataset} ----\n")
        # Make a folder for the job.
        outdir = rootdir / dataset
        if not args.dry_run:
            outdir.mkdir(parents=True, exist_ok=True)
            os.chdir(outdir)
            print(f"Run folder: {outdir}")
        else:
            print(f"Would create output folder: {outdir}. Subsequent actions would be relative to this folder"
                  f" instead of {os.getcwd()}.")

        # Do not overwrite anything if results already exist here.
        res_fname = result_filename(config)
        result_path = outdir / res_fname
        if result_path.exists() and not args.force:
            print(f"Results already exist at {result_path}. Skipping.")
            continue

        # Write a config (overwrite if exists).
        cfgfile = outdir / "config.yml"
        jobcfg = build_job_config(config, outdir, dataset)
        if not args.dry_run:
            cfgfile = cfgfile.resolve()
            with open(cfgfile, "w") as f:
                yaml.dump(jobcfg, f)
        else:
            print(f"Would write training config to file: {str(outdir / cfgfile.name)}")
            if args.launch_verbose:
                print(f"\nConfig to be written:\n{yaml.dump(jobcfg)}\n\n")

        # Write the metadata (overwrite if exists).
        metafile = outdir / "metadata.yml"
        metacfg = {"arch": config["config"].stem, "dataset": dataset}
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
        command = build_command(args.hardware, args.conda_env, cfgfile, config["train_config"].get("seed"),
                                res_fname, args.verbose, launcher_args)

        # Launch the job.
        # NOTE: The MKL_THREADING_LAYER variable is a workaround for an issue I was experiencing on the VACC while
        #       using torchrun.
        # TODO: Need to fix this whole mess of return codes vs. returning job IDs.
        ret = call_sbatch(command, args.launch_verbose, args.dry_run or args.do_not_launch,
                          env={"MKL_THREADING_LAYER": "GNU"}, return_job_id=getattr(args, "return_job_ids", False))
        result.append(ret)
        # It seems we need to manually flush to see printouts on the cluster.
        sys.stdout.flush()
        sys.stderr.flush()

    return result


def main(argv=None):
    parser = create_arg_parser(__doc__)
    args, launcher_args = parser.parse_known_args(argv)

    config = prep_config(parser, args)
    return setup_and_launch_jobs(config, args, launcher_args)


if __name__ == "__main__":
    sys.exit(main())
