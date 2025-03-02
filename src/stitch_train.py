"""
A script to train an architecture and store the training trajectory in the corresponding cache.

To test this script, try:
    python stitch_train.py -c tests/stitch-resnet-18.yml --st
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
from assembly import Assembly
from utils import ensure_config_param, make_pretty

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
    parser.add_argument("-c", "--config", metavar="PATH", type=argutils.existing_path, required=True,
                        help="Training config file.")
    datasets.add_dataset_arg(parser, dflt_data_dir=Path(__file__).parent / "data")
    parser.add_argument("--save-checkpoints", action="store_true",
                        help="Save the model weights at the end of each epoch.")
    parser.add_argument("--no-eval-checkpoints", dest="eval_checkpoints", action="store_false",
                        help="Do not evaluate each checkpoint on the entire train/test set. This can speed up training"
                             " but the downside is that you will be relying on training batches only for tracking the"
                             " progress of training.")
    parser.add_argument("-j", "--workers", default=NUM_CORES, type=int, metavar="N",
                        help="Number of data loading workers. Defaults to the number of cores detected on the current "
                             "node.")
    parser.add_argument("-b", "--batch-size", default=256, type=int, metavar="N",
                        help="Mini-batch size. When distributed, this is the total batch size of all GPUs on the "
                             "current node.")
    parser.add_argument("-m", "--max-batches", type=int, metavar="N",
                        help="Maximum number of batches. Useful for quick testing/debugging.")
    parser.add_argument("-o", "--output", "--dest", type=Path, metavar="FOLDER",
                        help="Location to save the model checkpoints. By default, they will be saved in the current "
                             "directory.")
    parser.add_argument("--print-freq", default=10, type=int, metavar="N", help="Print frequency.")
    parser.add_argument("--world-size", default=-1, type=int, metavar="N",
                        help="Number of nodes for distributed processing.")
    parser.add_argument("--rank", default=-1, type=int, metavar="N", help="Node rank for distributed processing.")
    parser.add_argument("--dist-url", default="tcp://224.66.41.62:23456", type=str, metavar="URL",
                        help="URL used to set up distributed processing.")
    parser.add_argument("--cpu", action="store_true", help="Allow the use of CPU (if no GPU can be found).")
    parser.add_argument("--gpu", default=None, type=int, help="GPU ID to use.")
    parser.add_argument("--deterministic", action="store_true", help="Use only deterministic algorithms.")
    argutils.add_device_arg(parser)
    argutils.add_seed_arg(parser, default_seed=1)
    argutils.add_wandb_args(parser, allow_id=allow_id)
    argutils.add_verbose_arg(parser)
    parser.add_argument("--st", "--smoke-test", dest="smoke_test", action="store_true",
                        help="Conduct a quick, full test of the training pipeline. If enabled, then a number of"
                             " arguments will be overridden to make the training run as short as possible and print in"
                             " verbose/debug mode.")
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

    def of_type(types):
        def test_fn(val):
            return isinstance(val, types)

        return test_fn

    # First check values for constructing the model.
    ensure_config_param(config, "blocks", of_type(list))
    ensure_config_param(config, "adapters", of_type(list))
    ensure_config_param(config, "use_base", of_type(bool), dflt=False)
    ensure_config_param(config, "use_head", of_type(bool), dflt=False)
    ensure_config_param(config, "frozen", of_type(bool), dflt=True)
    ensure_config_param(config, "base_adapter", of_type(dict), required=False)

    # Now check values related to training the model.
    training.check_train_config(config)

    return config


def prep_config(parser, args):
    """ Process command line arguments to produce a full training config. May also edit the arguments. """
    # If we're doing a smoke test, then we need to modify the verbosity before configuring the logger.
    if args.smoke_test and args.verbose < 2:
        args.verbose = 2

    argutils.configure_logging(args, level=logging.INFO)

    # This list governs which _top-level_ args can be overridden from the command line.
    config = argutils.load_config_from_args(parser, args, ["data_path", "save_checkpoints", "eval_checkpoints",
                                                           "device", "id", "project", "entity", "group", "verbose",
                                                           "dist_url", "world_size", "rank", "gpu"])
    if not config.get("train_config"):
        # Exits the program with a usage error.
        parser.error(f'The given config does not have a "train_config" sub-config: {args.config}')
    # This list governs which _training_ args can be overridden from the command line.
    config["train_config"] = argutils.override_from_command_line(config["train_config"], parser, args,
                                                                 ["dataset", "seed", "batch_size", "max_batches"])

    # Conduct a quick test.
    if args.smoke_test:
        config["save_checkpoints"] = True
        config["eval_checkpoints"] = True
        config["checkpoint_initial_model"] = False
        config["train_config"]["max_steps"] = 1
        config["train_config"]["epochs"] = 1

    return validate_config(config)


def setup_and_train(parser, config):
    """ Setup distributed processing, setup W&B, load data, load model, and commence training. """
    dist.init_distributed_mode(config)

    device = argutils.get_device(parser, config)
    argutils.set_seed(config["train_config"]["seed"])
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
    train_loader = DataLoader(train_data, batch_size=config["batch_size"], sampler=train_sampler,
                              num_workers=config["workers"], pin_memory=True, persistent_workers=config["workers"] > 1)
    test_loader = DataLoader(test_data, batch_size=config["batch_size"], shuffle=test_sampler is None,
                             sampler=test_sampler, num_workers=config["workers"], pin_memory=True,
                             persistent_workers=config["workers"] > 1)

    logging.info("Constructing model.")
    model = Assembly(config["blocks"], config["adapters"], use_head=config.get("use_head"),
                     use_base=config.get("use_base"), base_adapter=config.get("base_adapter"),
                     all_fixed=config.get("frozen", True), input_shape=input_shape, num_classes=num_classes)
    model.to(device)

    training.train(config, model, train_loader, test_loader, device)


def main(argv=None):
    parser = create_arg_parser(__doc__)
    args = parser.parse_args(argv)

    config = prep_config(parser, args)

    setup_and_train(parser, config)


if __name__ == "__main__":
    sys.exit(main())
