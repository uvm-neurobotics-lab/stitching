"""
If you want to use one of the WILDS dataset, first download it with this script.

Example:
    python src/download_wilds_datasets.py --root-dir data --datasets fmow poverty
"""
import argparse
from pathlib import Path

import wilds

from utils.argparsing import resolved_path


def main(argv=None):
    """
    Downloads the latest versions of all specified datasets, if they do not already exist.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", "--data-dir", metavar="PATH", type=resolved_path,
                        default=Path("./data").resolve(),
                        help="The root path where the dataset should be downloaded to, if it doesn't exist.")
    parser.add_argument("--datasets", nargs="*", default=None,
                        help=f"Specify a space-separated list of dataset names to download. If left unspecified, the"
                             f" script will download all of the official benchmark datasets. Available choices are"
                             f" {wilds.supported_datasets}.")
    parser.add_argument("--unlabeled", default=False, type=bool,
                        help="If this flag is set, the unlabeled dataset will also be downloaded.")
    config = parser.parse_args(argv)

    if config.datasets is None:
        config.datasets = wilds.benchmark_datasets

    for dataset in config.datasets:
        if dataset not in wilds.supported_datasets:
            raise ValueError(f'{dataset} not recognized; must be one of {wilds.supported_datasets}.')

    print(f'Downloading the following datasets: {config.datasets}')
    for dataset in config.datasets:
        print(f'=== {dataset} ===')
        wilds.get_dataset(
            dataset=dataset,
            root_dir=config.root_dir,
            unlabeled=config.unlabeled,
            download=True)


if __name__ == '__main__':
    main()
