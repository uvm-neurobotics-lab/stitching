"""
Module for loading of datasets specified on command line.
"""
import argparse
from pathlib import Path

from naslib_api import TrainableNB201
from utils.argparsing import resolved_path


BENCHMARKS = {
    "nb201": TrainableNB201,
}


def benchmark_str(benchname):
    """
    This function can be used as an argument type to validate the `--benchmark` entry:
        `parser.add_argument(..., type=argutils.benchmark_str, ...)`
    An exception will be raised if an unrecognized string is given.

    Args:
        benchname: The user-supplied benchmark name.

    Returns:
        str: A key for the `BENCHMARKS` mapping.
    """
    lname = benchname.lower()
    if lname not in BENCHMARKS:
        raise argparse.ArgumentTypeError(f"Unrecognized search space: {benchname}.")
    return lname


def add_dataset_arg(parser, dflt_data_dir=Path("./data").resolve(), add_benchmark_arg=True):
    """
    Add an argument for the user to specify which search space and dataset they wish to use.
    """
    if add_benchmark_arg:
        parser.add_argument("--benchmark", "--search-space", choices=list(BENCHMARKS.keys()), type=benchmark_str,
                            help="The search space to use.")
    parser.add_argument("--dataset", type=str.lower, help="The dataset to use.")
    parser.add_argument("--data-path", "--data-dir", metavar="PATH", type=resolved_path, default=dflt_data_dir,
                        help="The root path in which to look for the dataset.")
    return parser


def load_dataset(config, device):
    BenchClass = BENCHMARKS[config["train_config"]["benchmark"]]
    return BenchClass(config, data_root=Path(config["data_path"]), device=device)
