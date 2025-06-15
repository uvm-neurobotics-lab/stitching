"""
A script to assemble a number of experiment configs and launch Slurm jobs to execute them.

To test this script, try:
    python src/launch_scaling_experiments.py -c across-scales/resnet-50.yml --seed 12345 -n -vv
"""
import functools
import logging
import os
import sys
from copy import deepcopy
from pathlib import Path

import yaml

import utils.argparsing as argutils
import utils.datasets as datasets
import utils.training as training
from assembly import validate_part_list
from stitch_train import build_command
from utils import ensure_config_param, make_pretty, _and, of_type
from utils.slurm import call_sbatch


def result_filename(config):
    fname = "result"
    seed = config.get("train_config", {}).get("seed")
    if seed:
        fname += f"-{seed}"
    return fname + ".pkl"


def set_frozen(parts, frozen):
    if isinstance(parts, (list, tuple)):
        # List of parts.
        for p in parts:
            set_frozen(p, frozen)
    else:
        # Single part: a dict containing a single value.
        next(iter(parts.values()))["frozen"] = frozen


def get_tensor_shape_sequence(src_format, dest_format, num_downsamples):
    # Get the appropriate sequence of tensor shapes for N downsampling steps.
    in_channels = src_format[1][0]
    out_channels = dest_format[1][0]
    in_size = src_format[1][1]
    out_size = dest_format[1][1]
    if out_channels >= in_channels:
        channels = [min(in_channels * (2 ** i), out_channels) for i in range(max(num_downsamples + 1, 2))]
    else:
        channels = [max(in_channels // (2 ** i), out_channels) for i in range(max(num_downsamples + 1, 2))]
    sizes = [max(in_size // (2 ** i), out_size) for i in range(max(num_downsamples + 1, 2))]
    # Ensure the final size is `dest_format`, in case the `num_downsamples` wasn't enough to get to the right size.
    channels[-1] = out_channels
    sizes[-1] = out_size
    return channels, sizes


def stitch(src_stages, dest_stages, gap, adapter_fn):
    # Everything before the first dropped block.
    src_blocks = src_stages[:gap["blocks_to_drop"][0]]
    # Everything after the last dropped block.
    dest_blocks = dest_stages[gap["blocks_to_drop"][1] + 1:]
    num_downsamples = gap["num_downsamples"]

    # Pull them together with the specified adapter.
    src_format = next(iter(src_blocks[-1].values()))["out_format"]
    dest_format = next(iter(dest_blocks[0].values()))["in_format"]
    # `adapter_fn` provides a part list, could be multiple part configs.
    adapter_cfg = adapter_fn(src_format, dest_format, num_downsamples)
    set_frozen(src_blocks, True)
    set_frozen(adapter_cfg, False)
    set_frozen(dest_blocks, True)
    return src_blocks + adapter_cfg + dest_blocks


def stitch_no_downsample(src_stages, dest_stages, gap, adapter_fn):
    # Everything before the first dropped block.
    src_blocks = src_stages[:gap["blocks_to_drop"][0]]
    # Everything after the last dropped block.
    dest_blocks = dest_stages[gap["blocks_to_drop"][1] + 1:]
    num_downsamples = gap["num_downsamples"]

    # Pull them together with the specified adapter.
    # Modify block sizes so that no downsampling is performed.
    src_format = next(iter(src_blocks[-1].values()))["out_format"]
    dest_format = next(iter(dest_blocks[0].values()))["in_format"]
    # Change dest spatial size to keep the same as src size.
    dest_format[1][1:] = src_format[1][1:]
    # Remove all subsequent size indicators so the Assembly doesn't do any resizing.
    for b in dest_blocks:
        bargs = next(iter(b.values()))
        bargs["in_format"] = bargs["in_format"][0]
        bargs["out_format"] = bargs["out_format"][0]  # Just keep the format indicator ("img", "token", etc.).
    # `adapter_fn` provides a part list, could be multiple part configs.
    adapter_cfg = adapter_fn(src_format, dest_format, num_downsamples)
    set_frozen(src_blocks, True)
    set_frozen(adapter_cfg, False)
    set_frozen(dest_blocks, True)
    return src_blocks + adapter_cfg + dest_blocks


def finetune_stitch(src_stages, dest_stages, gap):
    """
    No adapter, and unfreeze everything!
    If we must perform downsamples, then keep the downsampling blocks.
    """
    # Everything before the first dropped block.
    first_dropped = gap["blocks_to_drop"][0]
    src_blocks = src_stages[:first_dropped]
    # Everything after the last dropped block.
    last_dropped = gap["blocks_to_drop"][1]
    dest_blocks = dest_stages[last_dropped + 1:]

    # Determine which downsamples exist in the gap and need to be kept.
    # FIXME: We pretend to know which blocks are "downsample" blocks that we should keep. But this info should be in the
    #        config, not hard-coded here.
    known_downsamples = list(filter(lambda x: x > first_dropped, [2, 4, 6]))
    down_blocks = [dest_stages[i] for i in known_downsamples if first_dropped <= i <= last_dropped]
    assert len(down_blocks) == gap["num_downsamples"], (f"num saved blocks ({len(down_blocks)}) != expected downsamples"
                                                        f" ({gap['num_downsamples']})")

    # Ensure the sizes are matching, otherwise abort the fine-tuning by returning None.
    # NOTE: These size comparisons assume we don't have a mixed-type comparison, like tuple and list.
    src_out = next(iter(src_blocks[-1].values()))["out_format"][1]
    dest_in = next(iter(dest_blocks[0].values()))["in_format"][1]
    if down_blocks:
        mid_in = next(iter(down_blocks[0].values()))["in_format"][1]
        mid_out = next(iter(down_blocks[-1].values()))["out_format"][1]
        if src_out != mid_in:
            return None
        if dest_in != mid_out:
            return None
    else:
        if src_out != dest_in:
            return None

    assembly = src_blocks + down_blocks + dest_blocks
    set_frozen(assembly, False)
    return assembly


def linear(src_format, dest_format, num_downsamples):
    channels, sizes = get_tensor_shape_sequence(src_format, dest_format, num_downsamples)
    # Use 1x1 convs if either input or output is in image-like format. Otherwise, use fully-connected layers.
    conv_or_fc = "num_conv" if (src_format[0] in ("img", "bhwc") or dest_format[0] in ("img", "bhwc")) else "num_fc"

    parts = []
    for ich, och, isz, osz in zip(channels, channels[1:], sizes, sizes[1:]):
        parts.append({
            "SimpleAdapter": {
                conv_or_fc: 1,
                "kernel_size": 1,
                "nonlinearity": False,
                "in_channels": ich,
                "out_channels": och,
                "in_format": ["img", [ich, osz, osz]],  # Use osz, so we downsample before the adapter.
            }
        })
    return parts


def linear_post_downsample(src_format, dest_format, num_downsamples):
    channels, sizes = get_tensor_shape_sequence(src_format, dest_format, num_downsamples)
    # Use 1x1 convs if either input or output is in image-like format. Otherwise, use fully-connected layers.
    conv_or_fc = "num_conv" if (src_format[0] in ("img", "bhwc") or dest_format[0] in ("img", "bhwc")) else "num_fc"

    parts = []
    for ich, och, isz, osz in zip(channels, channels[1:], sizes, sizes[1:]):
        parts.append({
            "SimpleAdapter": {
                conv_or_fc: 1,
                "kernel_size": 1,
                "nonlinearity": False,
                "in_channels": ich,
                "out_channels": och,
                "in_format": ["img", [ich, isz, isz]],  # Use isz, so we downsample after the adapter.
            }
        })
    return parts


def conv3x3(src_format, dest_format, num_downsamples):
    channels, sizes = get_tensor_shape_sequence(src_format, dest_format, num_downsamples)
    # Use 1x1 convs if either input or output is in image-like format. Otherwise, use fully-connected layers.
    conv_or_fc = "num_conv" if (src_format[0] in ("img", "bhwc") or dest_format[0] in ("img", "bhwc")) else "num_fc"

    parts = []
    for ich, och, isz, osz in zip(channels, channels[1:], sizes, sizes[1:]):
        parts.append({
            "SimpleAdapter": {
                conv_or_fc: 1,
                "kernel_size": 3,
                "in_channels": ich,
                "out_channels": och,
                "in_format": ["img", [ich, osz, osz]],  # Use osz, so we downsample before the adapter.
            }
        })
    return parts


def block(src_format, dest_format, num_downsamples, kind="ResNetBasicBlock"):
    channels, sizes = get_tensor_shape_sequence(src_format, dest_format, num_downsamples)

    parts = []
    for ich, och, isz, osz in zip(channels, channels[1:], sizes, sizes[1:]):
        parts.append({
            kind: {
                "in_channels": ich,
                "out_channels": och,
                "in_format": ["img", [ich, osz, osz]],  # Use osz, so we downsample before the adapter.
            }
        })
    return parts


def bottleneck(src_format, dest_format, num_downsamples):
    return block(src_format, dest_format, num_downsamples, "ResNetBottleneck")


def conv3x3_with_downsample(src_format, dest_format, num_downsamples):
    channels, sizes = get_tensor_shape_sequence(src_format, dest_format, num_downsamples)
    # Use 1x1 convs if either input or output is in image-like format. Otherwise, use fully-connected layers.
    conv_or_fc = "num_conv" if (src_format[0] in ("img", "bhwc") or dest_format[0] in ("img", "bhwc")) else "num_fc"

    parts = []
    for ich, och, isz, osz in zip(channels, channels[1:], sizes, sizes[1:]):
        stride = isz // osz
        parts.append({
            "SimpleAdapter": {
                conv_or_fc: 1,
                "kernel_size": 3,
                "stride": stride,
                "in_channels": ich,
                "out_channels": och,
            }
        })
    return parts


def block_with_downsample(src_format, dest_format, num_downsamples, kind="ResNetBasicBlock"):
    channels, sizes = get_tensor_shape_sequence(src_format, dest_format, num_downsamples)

    parts = []
    for ich, och, isz, osz in zip(channels, channels[1:], sizes, sizes[1:]):
        stride = isz // osz
        parts.append({
            kind: {
                "in_channels": ich,
                "out_channels": och,
                "stride": stride,
            }
        })
    return parts


def bottleneck_with_downsample(src_format, dest_format, num_downsamples):
    return block_with_downsample(src_format, dest_format, num_downsamples, "ResNetBottleneck")


# TODO: Remove the conv options for a ViT?
STITCHERS = {
    "finetune": finetune_stitch,
    "block_no_downsample": functools.partial(stitch_no_downsample, adapter_fn=block),
    "bottleneck_no_downsample": functools.partial(stitch_no_downsample, adapter_fn=bottleneck),
    "linear_no_downsample": functools.partial(stitch_no_downsample, adapter_fn=linear),
    "downsample_then_linear": functools.partial(stitch, adapter_fn=linear),
    "downsample_then_3x3conv": functools.partial(stitch, adapter_fn=conv3x3),
    "downsample_then_block": functools.partial(stitch, adapter_fn=block),
    "downsample_then_bottleneck": functools.partial(stitch, adapter_fn=bottleneck),
    "conv3x3_with_downsample": functools.partial(stitch, adapter_fn=conv3x3_with_downsample),
    "block_with_downsample": functools.partial(stitch, adapter_fn=block_with_downsample),
    "bottleneck_with_downsample": functools.partial(stitch, adapter_fn=bottleneck_with_downsample),
}


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
                        default=Path("experiments").resolve(), help="Root location for all experiments.")

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
    parser.add_argument("-f", "--force", action="store_true",
                        help="Run even if results already exist. Overwrite config files if they exist.")
    parser.add_argument("-n", "--dry-run", action="store_true",
                        help="Do not actually launch jobs, but only print out the equivalent commands that would be"
                             " launched.")
    parser.add_argument("--lv", "--launch-verbose", dest="launch_verbose", action="store_true",
                        help="Be verbose when launching the job (output all the launcher print statements).")

    # Other args.
    argutils.add_seed_arg(parser, default_seed=1)
    argutils.add_wandb_args(parser, dflt_project="across-scales", allow_id=allow_id)
    argutils.add_verbose_arg(parser)
    return parser


def validate_gap_config(gaps):
    if not isinstance(gaps, (list, tuple)):
        raise RuntimeError(f"'gaps' should be a list of gap configs.")
    for i, gap in enumerate(gaps):
        if not isinstance(gap, dict) or "blocks_to_drop" not in gap or "num_downsamples" not in gap:
            raise RuntimeError(f"Gap #{i} invalid: a gap config should be a dict containing 'blocks_to_drop' and"
                               " 'num_downsamples'.")
        if (not isinstance(gap["blocks_to_drop"], (list, tuple))
                or len(gap["blocks_to_drop"]) != 2
                or not (isinstance(gap["blocks_to_drop"][0], int) and isinstance(gap["blocks_to_drop"][1], int))
                or gap["blocks_to_drop"][0] > gap["blocks_to_drop"][1]):
            raise RuntimeError(f"Gap #{i} invalid: 'blocks_to_drop' should be a pair of integers [first, last].")
        if not isinstance(gap["num_downsamples"], int):
            raise RuntimeError(f"Gap #{i} invalid: 'num_downsamples' should be an integer.")
    return True


def validate_stitcher_config(stitchers):
    if not isinstance(stitchers, (list, tuple)):
        raise RuntimeError(f"'stitchers' should be a list of stitcher names.")
    for name in stitchers:
        if name not in STITCHERS:
            raise RuntimeError(f"Unrecognized stitcher name: {name}")
    return True


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
    # We require either one set of stages or two sets of "bottom" and "top" networks.
    if "stages" in config:
        if "src_stages" in config or "dest_stages" in config:
            raise RuntimeError(f"Either supply a single 'stages' variable, or supply 'src_stages' and 'dest_stages'"
                               " together.")
        ensure_config_param(config, "stages", _and(of_type(list), validate_part_list))
    elif not ("src_stages" in config and "dest_stages" in config):
        raise RuntimeError(f"Either supply a single 'stages' variable, or supply 'src_stages' and 'dest_stages'"
                           " together.")
    else:
        ensure_config_param(config, "src_stages", _and(of_type(list), validate_part_list))
        ensure_config_param(config, "dest_stages", _and(of_type(list), validate_part_list))
    ensure_config_param(config, "gaps", validate_gap_config)
    ensure_config_param(config, "stitchers", validate_stitcher_config)
    ensure_config_param(config, "project", of_type(str))

    # Now check values related to training the model.
    datasets.check_data_config(config)
    training.check_train_config(config)

    return config


def prep_config(parser, args):
    """ Process command line arguments to produce a full training config. May also edit the arguments. """
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


def build_assembly(config, gap, stitch_fn):
    src_stages = deepcopy(config.get("src_stages", config.get("stages")))
    dest_stages = deepcopy(config.get("dest_stages", config.get("stages")))
    return stitch_fn(src_stages, dest_stages, gap)


def build_job_config(config, save_dir, assembly):
    jobcfg = deepcopy(config)
    jobcfg.pop("config", None)
    jobcfg.pop("stages", None)
    jobcfg.pop("src_stages", None)
    jobcfg.pop("dest_stages", None)
    jobcfg.pop("gaps", None)
    jobcfg["train_config"].pop("seed", None)

    jobcfg["assembly"] = assembly
    jobcfg["save_dir"] = save_dir

    return make_pretty(jobcfg)


def setup_and_launch_jobs(config, args, launcher_args):
    """ Write configs for each job and launch them. """
    # create folder based on config name.
    expname = config["config"].stem
    rootdir = Path(config["save_dir"]).resolve() / config["project"] / expname
    stitchers_to_run = {k: STITCHERS[k] for k in config.get("stitchers", STITCHERS.keys())}
    result = 0
    for gap in config["gaps"]:
        blocks_to_drop = gap["blocks_to_drop"]
        num_downsamples = gap["num_downsamples"]
        for adapter_name, adapter in stitchers_to_run.items():
            print(f"\n---- LAUNCHING gap={blocks_to_drop}, {adapter_name} ----\n")
            # Make a folder for the job.
            jobname = "-".join(["train", expname,
                                "gap", training.filesafe_str(blocks_to_drop),
                                "adapter", adapter_name])
            outdir = rootdir / jobname
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
                print(f"Results already exist for {expname}/{jobname}/{res_fname}. Skipping.")
                continue

            # Generate the assembly for training. Skip if not applicable.
            assembly = build_assembly(config, gap, adapter)
            if not assembly:
                print(f"Adapter {adapter_name} is not applicable to this gap. Skipping.")
                continue

            # Write a config (overwrite if exists).
            cfgfile = outdir / "config.yml"
            jobcfg = build_job_config(config, outdir, assembly)
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
            metacfg = {"arch": expname, "first_deleted_block": blocks_to_drop[0],
                       "last_deleted_block": blocks_to_drop[1], "num_downsamples": num_downsamples,
                       "adapter": adapter_name}
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
            command = build_command(args.cluster, args.conda_env, cfgfile, config["train_config"].get("seed"),
                                    res_fname, args.verbose, launcher_args)

            # Launch the job.
            # NOTE: The MKL_THREADING_LAYER variable is a workaround for an issue I was experiencing on the VACC while
            #       using torchrun.
            result += call_sbatch(command, args.launch_verbose, args.dry_run, env={"MKL_THREADING_LAYER": "GNU"})

    return result


def main(argv=None):
    parser = create_arg_parser(__doc__)
    args, launcher_args = parser.parse_known_args(argv)

    config = prep_config(parser, args)
    return setup_and_launch_jobs(config, args, launcher_args)


if __name__ == "__main__":
    sys.exit(main())
