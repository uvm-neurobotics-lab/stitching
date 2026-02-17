"""
A script to test heads from different fine-tunings on encoders from different fine-tunings. This also tests the
original, pre-trained encoder, before fine-tuning. Note that, in each case, this base encoder is the one specified by
the fine-tuning which produced the head.

To test this script, try:
    WANDB_MODE=disabled python src/ft_cross_test.py -c tests/ft-cross-test.yml --st
To run distributed, use torchrun and specify number of workers. For example, on a single node with 8 GPUs and 48 CPUs:
    WANDB_MODE=disabled torchrun --nproc-per-node=8 src/ft_cross_test.py -c tests/ft-cross-test.yml -j 6
"""
import argparse
import logging
import os
import sys
from pathlib import Path, PosixPath

import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler

import stitch_train
import utils.argparsing as argutils
import utils.datasets as datasets
import utils.distributed as dist
from assembly import model_from_config
from utils import _and, ensure_config_param, gt_zero, gte_zero, make_pretty, num_params, num_trainable_params, of_type
from utils.logging import StandardLog
from utils.optimization import check_metrics_config, metric_fns_from_config

NUM_CORES = os.cpu_count()
if hasattr(os, "sched_getaffinity"):
    # This function is only available on certain platforms. When running with Slurm, it can tell us the true
    # number of cores we have access to.
    NUM_CORES = len(os.sched_getaffinity(0))


def create_arg_parser(desc, allow_abbrev=True):
    """
    Creates the argument parser for this program.

    Args:
        desc (str): The human-readable description for the arg parser.
        allow_abbrev (bool): The `allow_abbrev` argument to `argparse.ArgumentParser()`.

    Returns:
        argutils.ArgParser: The parser.
    """
    parser = argutils.create_parser(desc, allow_abbrev=allow_abbrev)
    parser.add_argument("-c", "--config", metavar="FILE", type=argutils.existing_path, required=True,
                        help="Cross-test config file.")
    parser.add_argument("--data-path", "--data-dir", metavar="PATH", type=argutils.existing_path,
                        default=Path("./data").resolve(), help="The root path in which to look for the dataset.")
    parser.add_argument("-o", "--output", "--dest", metavar="PATH", type=argutils.resolved_path,
                        default=Path("./cross-test-result.pkl").resolve(),
                        help="Location to save a dataframe of recorded metrics.")

    # Distributed/hardware args.
    parser.add_argument("-j", "--workers", default=NUM_CORES, type=int, metavar="N",
                        help="Number of data loading workers. Defaults to the number of cores detected on the current "
                             "node.")
    parser.add_argument("-m", "--max-batches", type=int, metavar="N",
                        help="Maximum number of batches for each dataset. Useful for quick testing/debugging.")
    argutils.add_device_arg(parser)
    parser.add_argument("--deterministic", action="store_true", help="Use only deterministic algorithms.")

    # Other args.
    # TODO: add option to evaluate full train set.
    argutils.add_seed_arg(parser, default_seed=1)
    argutils.add_verbose_arg(parser)
    parser.add_argument("--st", "--smoke-test", dest="smoke_test", action="store_true",
                        help="Conduct a quick, full test of the training pipeline. If enabled, then a number of"
                             " arguments will be overridden to make the training run as short as possible and print in"
                             " verbose/debug mode.")
    return parser


def validate_file_list(file_list):
    if not file_list:
        raise ValueError(f"No finetune configs specified. You must specify at least one model to cross-compare.")
    if not isinstance(file_list, list):
        raise ValueError(f"Should be a list of config files.")
    for f in file_list:
        if not Path(f).is_file():
            raise ValueError(f"Not a file or not found: {f}")
    return True


def validate_config(config):
    """
    Prints and validates the given cross-test config. Throws an exception in the case of invalid or missing required
    values. Non-required missing values are filled in with their defaults; note that this modifies the config in-place.

    Args:
        config: A config dict with a list of fine-tuned models that should be compared, and some settings for how to
                run on GPU.
    Returns:
        The config after validation (the same instance as was passed in).
    """
    # Output the config for reference. Do it before checking config to assist debugging.
    logging.info("\n------- Config -------\n" + yaml.dump(make_pretty(config)) + "----------------------")

    ensure_config_param(config, "ft_configs", _and(of_type(list), validate_file_list))

    ensure_config_param(config, "data_path", argutils.existing_path)
    config["data_path"] = Path(config["data_path"]).expanduser().resolve()  # ensure the type is Path, not string.

    ensure_config_param(config, "output", argutils.resolved_path)
    config["output"] = Path(config["output"]).expanduser().resolve()  # ensure type is Path, not string.

    ensure_config_param(config, "max_batches", _and(of_type(int), gt_zero), required=False)
    ensure_config_param(config, "verbose", _and(of_type(int), gte_zero), required=False)
    ensure_config_param(config, "seed", of_type(int))

    check_metrics_config(config)

    return config


def prep_config(parser, args):
    """ Process command line arguments to produce a full training config. May also edit the arguments. """
    # If we're doing a smoke test, then we need to modify the verbosity before configuring the logger.
    if args.smoke_test and args.verbose < 2:
        args.verbose = 2

    argutils.configure_logging(args, level=logging.INFO)

    # This list governs which args can be overridden from the command line.
    config = argutils.load_config_from_args(parser, args, ["data_path", "output", "seed", "max_batches", "device",
                                                           "workers", "deterministic", "verbose"])

    # Conduct a quick test.
    if args.smoke_test:
        config["max_batches"] = 3 * int(os.environ.get("WORLD_SIZE", 1))

    return validate_config(config)


def get_result_file(config):
    resfile = Path(config["output"])
    if not str(resfile).endswith(".pkl"):
        # Assume this is intended as a directory name, even if it doesn't exist.
        resfile = resfile / "cross-test-result.pkl"
    resfile = resfile.expanduser().resolve()
    return resfile


def get_checkpoint_from_config(config):
    # Look in save_dir if specified, else look next to the config file itself.
    return Path(config.get("save_dir", Path(config["loaded_from"]).parent)) / "checkpoint.pth"


def load_weight_subset(model, ckpt_file, prefix):
    """ From the given checkpoint, load only the parts of the model beginning with `prefix`. """
    checkpoint = torch.load(ckpt_file, map_location="cpu", weights_only=True)
    substate_dict = {k: v for k, v in checkpoint["model"].items() if k.startswith(prefix)}
    model.load_state_dict(substate_dict, strict=False)


def test_one_encoder(run_config, encoder_config, model, model_without_ddp, train_loader, valid_loaders, device):
    """ Test one head + encoder pair. """
    # TODO: WARNING: This assumes all encoders have the same architecture. In a future version, it would be good to
    #       load the model architecture specified by the encoder_config. We could do this if we only assume that all
    #       models are of type `Assembly`, so then we can keep the `model.head` while replacing the `model.parts`.
    #       We would likely still need to redo the DDP initialization, which might force a completely new model.
    if encoder_config is not None:
        encoder_ckpt = get_checkpoint_from_config(encoder_config)
        logging.info(f"Loading encoder weights from: {encoder_ckpt}")
        load_weight_subset(model_without_ddp, encoder_ckpt, "parts.")
    else:
        logging.info(f"Evaluating pretrained encoder weights.")

    # HACK: We are using the logger to handle metric computation for us. Kinda hacky.
    metric_fns = metric_fns_from_config(run_config, model)
    log = StandardLog(model=None, expected_steps=0, save_freq=0, checkpoint_initial_model=False, metric_fns=metric_fns)
    log.maybe_save_and_eval(0, 0, model, train_loader, valid_loaders, None, None, run_config, device,
                            should_eval=True, should_save=False)
    return log.recorded_metrics


def test_one_dataset(run_config, head_config, encoder_cfgs, model, model_without_ddp, train_loader, valid_loaders,
                     device):
    """ For the head that was trained on this dataset, test it on top of all other encoders. """
    records = []
    # Add None to test the base weights. This must be first, before weights are overwritten.
    for i, cfg in enumerate([None] + encoder_cfgs):
        logging.info(f"----- ENCODER {i+1}/{len(encoder_cfgs)+1} -----")
        metrics_map = test_one_encoder(run_config, cfg, model, model_without_ddp, train_loader, valid_loaders, device)
        records.extend([{"head_dataset": head_config["train_config"]["dataset"],
                         "head_loc": head_config["loaded_from"],
                         "encoder_dataset": cfg["train_config"]["dataset"] if cfg else "base",
                         "encoder_loc": cfg["loaded_from"] if cfg else "",
                         **record}
                        for step, record in metrics_map.items()])
    return records


def setup_and_test_one_dataset(device, run_config, head_config, encoder_configs):
    """ Sets up dataset and model before delegating to `test_one_dataset()`. """
    logging.info("")
    logging.info(f"------------ Evaluating head: {Path(head_config['loaded_from']).parent} ------------")

    logging.info(f"Loading dataset: {head_config['train_config']['dataset']}.")
    train_data, test_data, input_shape, num_classes = datasets.load_dataset_from_config(head_config)

    if run_config["distributed"]:
        train_sampler = DistributedSampler(train_data)
        test_sampler = DistributedSampler(test_data, shuffle=False)
    else:
        # noinspection PyTypeChecker
        train_sampler = RandomSampler(train_data)
        # noinspection PyTypeChecker
        test_sampler = SequentialSampler(test_data)

    num_workers = run_config["workers"]
    logging.info(f"Using {num_workers} workers for data loading.")
    train_loader = DataLoader(train_data, batch_size=head_config["train_config"]["batch_size"], sampler=train_sampler,
                              num_workers=num_workers, pin_memory=True, persistent_workers=num_workers > 1)
    test_loader = DataLoader(test_data, batch_size=head_config["train_config"]["batch_size"], sampler=test_sampler,
                             num_workers=num_workers, pin_memory=True, persistent_workers=num_workers > 1)

    # NOTE: this ends up loading whatever pre-trained weights the head_config specifies.
    logging.info("Constructing model.")
    model = model_from_config(head_config, input_shape, num_classes)
    model.to(device)

    # Set up distributed training and checkpointing behavior.
    model_without_ddp = model
    if run_config["distributed"]:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[device])
        model_without_ddp = model.module

    head_ckpt = get_checkpoint_from_config(head_config)
    logging.info(f"Loading head from {head_ckpt}.")
    load_weight_subset(model_without_ddp, head_ckpt, "head.")

    return test_one_dataset(run_config, head_config, encoder_configs, model, model_without_ddp, train_loader,
                            {"Test": test_loader}, device)


def setup_and_test(parser, args, run_config):
    """ Setup distributed processing, parse fine-tuned model configs, and accumulate results. """
    resfile = get_result_file(run_config)
    if resfile.exists():
        logging.warning(f"WARNING: Will overwrite existing result file: {resfile}")

    dist.init_distributed_mode(run_config)

    # We disable the cudnn benchmarking because it can noticeably affect the accuracy.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    if run_config.get("deterministic"):
        # Not currently sure if there's a difference b/w this and above .deterministic. Haven't checked.
        torch.use_deterministic_algorithms(True)

    device = argutils.get_device(parser, run_config)
    argutils.set_seed(run_config["seed"])

    # NOTE: some checkpoints were accidentally saved with stray Path types, so we need to mark them as safe.
    torch.serialization.add_safe_globals([PosixPath])

    # Parse configs corresponding to different fine-tuned models.
    # HACK: Set all args to "user-invoked" to cause them to override the config settings. The fields in the Namespace
    #       below are what will be overridden.
    for a in parser._actions:
        a.user_invoked = True
    encoder_configs = []
    for cfg_file in run_config["ft_configs"]:
        ecfg = stitch_train.prep_config(parser, argparse.Namespace(config=cfg_file, save_dir=Path(cfg_file).parent,
                                                                   data_path=run_config["data_path"],
                                                                   max_batches=run_config["max_batches"], verbose=0,
                                                                   seed=run_config["seed"], smoke_test=args.smoke_test))
        ecfg["loaded_from"] = cfg_file
        encoder_configs.append(ecfg)

    # For each model, evaluate head on top of all other models.
    records = []
    for head_config in encoder_configs:
        records.extend(setup_and_test_one_dataset(device, run_config, head_config, encoder_configs))

    save_results(pd.DataFrame.from_records(records), resfile)
    return 0


def save_results(result_df, resfile):
    resfile.parent.mkdir(parents=True, exist_ok=True)
    logging.info(f"Saving results to: {str(resfile)}")
    result_df.to_pickle(resfile)


def main(argv=None):
    parser = create_arg_parser(__doc__)
    args = parser.parse_args(argv)

    config = prep_config(parser, args)

    try:
        return setup_and_test(parser, args, config)
    finally:
        dist.tear_down_distributed_mode()


if __name__ == "__main__":
    sys.exit(main())
