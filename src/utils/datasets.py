"""
Module for loading of datasets specified on command line.
"""
from os import PathLike
from pathlib import Path
from typing import Tuple, Union

import numpy as np
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import Dataset, Subset

from utils.argparsing import resolved_path


def make_cifar(name, split, data_root, imagenet_size=True):
    if name == "cifar10":
        cls = datasets.CIFAR10
        normalize = transforms.Normalize(mean=[x / 255 for x in [125.3, 123.0, 113.9]],
                                         std=[x / 255 for x in [63.0, 62.1, 66.7]])
    elif name == "cifar100":
        cls = datasets.CIFAR100
        normalize = transforms.Normalize(mean=[x / 255 for x in [129.3, 124.1, 112.4]],
                                         std=[x / 255 for x in [68.2, 65.4, 70.4]])
    else:
        raise ValueError(f"Unrecognized dataset: {name}")

    if split == "train":
        is_train = True
    elif split.startswith("val") or split == "test":
        is_train = False
    else:
        raise ValueError(f"Unrecognized dataset: {split}")

    if imagenet_size:
        transform = transforms.Compose([transforms.Resize(224), transforms.ToTensor(), normalize])
    else:
        transform = transforms.Compose([transforms.ToTensor(), normalize])

    return cls(data_root, is_train, transform, download=True)


def make_imagenet(split, data_root):
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return datasets.ImageFolder(data_root / "imagenet" / split, transform)


def _make_datasets(name, data_root):
    """ NOTE: Does not perform data augmentation! """
    name = name.lower()
    if name == "cifar10" or name == "cifar100":
        num_classes = 100 if name == "cifar100" else 10
        return make_cifar(name, "train", data_root), make_cifar(name, "val", data_root), num_classes
    elif name == "imagenet" or name == "imagenet-1k":
        return make_imagenet("train", data_root), make_imagenet("val", data_root), 1000
    else:
        raise ValueError(f"Unrecognized dataset: {name}")


def make_datasets(name: str, data_root: Union[str, PathLike], batch_size: int = None,
                  max_batches: int = None) -> Tuple[Dataset, Dataset, int]:
    """
    Construct specified classification dataset objects. Optionally select a random subset of each dataset of size
    (batch_size * max_batches).

    :param name: Name of the dataset to load, e.g. "imagenet" or "cifar100".
    :param data_root: Path where the dataset can be found.
    :param batch_size: The batch size, only needed if `max_batches` is being used.
    :param max_batches: Limit the dataset to this number of batches at most (for quick debugging using small data).
    :return:
        train_data (torch.utils.data.Dataset): The training set.
        test_data (torch.utils.data.Dataset): The test set.
        num_classes (int): The number of classes for classification.
    """
    train_data, test_data, num_classes = _make_datasets(name, data_root)
    if max_batches:
        if not batch_size:
            raise RuntimeError(f"Requested max {max_batches} batches, but batch size was missing ({batch_size}).")
        train_indices = np.random.choice(len(train_data), size=batch_size * max_batches, replace=False)
        reduced_train_data = Subset(train_data, train_indices)
        test_indices = np.random.choice(len(test_data), size=batch_size * max_batches, replace=False)
        reduced_test_data = Subset(test_data, test_indices)
        return reduced_train_data, reduced_test_data, num_classes
    else:
        return train_data, test_data, num_classes


def add_dataset_arg(parser, dflt_data_dir=Path("./data").resolve()):
    """
    Add an argument for the user to specify which search space and dataset they wish to use.
    TODO: Add option for data augmentation.
    """
    parser.add_argument("--dataset", type=str.lower, help="The dataset to use.")
    parser.add_argument("--data-path", "--data-dir", metavar="PATH", type=resolved_path, default=dflt_data_dir,
                        help="The root path in which to look for the dataset.")
    return parser


def load_dataset_from_config(config) -> Tuple[Dataset, Dataset, int]:
    """
    Load datasets from config. Delegates to `make_datasets()`.

    :param config: The dataset configuration.
    :return:
        train_data (torch.utils.data.Dataset): The training set.
        test_data (torch.utils.data.Dataset): The test set.
        num_classes (int): The number of classes for classification.
    """
    return make_datasets(config["train_config"]["dataset"], config["data_path"],
                         config["train_config"].get("batch_size"), config["train_config"].get("max_batches"))
