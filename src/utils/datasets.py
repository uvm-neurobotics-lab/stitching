"""
Module for loading of datasets specified on command line.
"""
from os import PathLike
from pathlib import Path
from typing import Dict, Tuple, Union

import numpy as np
import torch
import torchvision.transforms as transforms

if hasattr(transforms, "v2"):
    # Note that we must remain backward compatible with v1 to use the VACC BlackDiamond cluster. This means we don't
    # implement all the pipeline improvements recommended under v2. See here:
    # https://pytorch.org/vision/main/transforms.html#performance-considerations
    transforms = transforms.v2
import torchvision.datasets as datasets
from torch.utils.data import Dataset, Subset

from utils import _and, ensure_config_param, gt_zero, of_type
from utils.argparsing import existing_path, resolved_path


def validate_transform_list(transform_list):
    if isinstance(transform_list, list):
        return all(validate_transform_config(c) for c in transform_list)
    elif not isinstance(transform_list, bool):
        raise ValueError(f"Data augmentation config should be a single boolean or a list of transforms.")
    return True


def validate_transform_config(tform_cfg):
    if not isinstance(tform_cfg, dict):
        raise ValueError(f"Unrecognized transform config format: {tform_cfg}")
    if len(tform_cfg) > 1:
        raise ValueError(f"Each transform in the augmentation list should consist of a single class with args.")
    cls_name, args = next(iter(tform_cfg.items()))
    if not isinstance(cls_name, str) or not hasattr(transforms, cls_name):
        raise ValueError(f"Should be the name of a transform class: '{cls_name}'")
    if args and not isinstance(args, dict):
        raise ValueError(f"Expected a dictionary of arguments for {cls_name}, but instead got: {args}")
    return True


def transform_from_config(tform_cfg):
    if not validate_transform_config(tform_cfg):
        raise ValueError(f"Invalid transform config:\n{tform_cfg}")
    cls_name, args = next(iter(tform_cfg.items()))
    args = args.copy() if args is not None else {}
    if "interpolation" in args:
        args["interpolation"] = transforms.InterpolationMode(args["interpolation"])  # transform string into enum
    if "policy" in args:
        args["policy"] = transforms.AutoAugmentPolicy(args["policy"])  # transform string into enum
    if "transforms" in args:  # Allows for nested transforms, e.g. Compose or RandomApply.
        args["transforms"] = [transform_from_config(c) for c in args["transforms"]]
    TClass = getattr(transforms, cls_name)
    return TClass(**args)


def augmenter_from_config(transform_list):
    return transforms.Compose([transform_from_config(t) for t in transform_list])


def make_cifar(name, split, data_root, imagenet_size=True, augment_config=tuple()):
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

    transform = []
    if imagenet_size:
        transform.append(transforms.Resize(224))
    if is_train:
        transform.append(augmenter_from_config(augment_config))
    transform.append([transforms.ToTensor(), normalize])
    transform = transforms.Compose(transform)

    return cls(data_root, is_train, transform, download=True)


def make_imagenet(split, data_root, augment_config=tuple()):
    # NOTE: Transforms here are meant to be similar to those used in the default Torchvision example here:
    #       https://github.com/pytorch/vision/blob/main/references/classification/presets.py
    #       In that code, they always use RandomResizedCrop and RandomHorizontalFlip, even when using AutoAugment.
    #       I would have thought AutoAugment would subsume these but I am just following that code.
    transform = []
    if split == "train":
        transform.append(transforms.RandomResizedCrop(224))
        transform.append(augmenter_from_config(augment_config))
    else:
        transform.append(transforms.Resize(256))
        transform.append(transforms.CenterCrop(224))
    transform.append(transforms.ToTensor())
    transform.append(transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))
    transform = transforms.Compose(transform)
    return datasets.ImageFolder(data_root / "imagenet" / split, transform)


def _make_datasets(name, data_root, augment_config=None):
    name = name.lower()
    if augment_config and isinstance(augment_config, bool):
        # If data_augmentation = True, supply a default augmenter.
        policy = "cifar10" if name.startswith("cifar") else "imagenet"
        augment_config = [
            {"RandomHorizontalFlip": {"p": 0.5}},
            {"AutoAugment": {"policy": policy}},
        ]
    elif augment_config is None:
        augment_config = []

    if name == "cifar10" or name == "cifar100":
        num_classes = 100 if name == "cifar100" else 10
        return (make_cifar(name, "train", data_root, augment_config=augment_config),
                make_cifar(name, "val", data_root, augment_config=augment_config),
                (3, 224, 224), num_classes)
    elif name == "imagenet" or name == "imagenet-1k":
        return (make_imagenet("train", data_root, augment_config=augment_config),
                make_imagenet("val", data_root, augment_config=augment_config),
                (3, 224, 224), 1000)
    else:
        raise ValueError(f"Unrecognized dataset: {name}")


def make_datasets(name: str, data_root: Union[str, PathLike], batch_size: int = None, max_batches: int = None,
                  augment_config: Union[Dict, bool, None] = None) -> Tuple[Dataset, Dataset, Union[tuple, torch.Size], int]:
    """
    Construct specified classification dataset objects. Optionally select a random subset of each dataset of size
    (batch_size * max_batches).

    :param name: Name of the dataset to load, e.g. "imagenet" or "cifar100".
    :param data_root: Path where the dataset can be found.
    :param batch_size: The batch size, only needed if `max_batches` is being used.
    :param max_batches: Limit the dataset to this number of batches at most (for quick debugging using small data).
    :param augment_config: A config object describing extra transforms to apply to the training set.
    :return:
        train_data (torch.utils.data.Dataset): The training set.
        test_data (torch.utils.data.Dataset): The test set.
        input_shape (torch.Size | tuple): The size of the images [C, H, W].
        num_classes (int): The number of classes for classification.
    """
    train_data, test_data, input_shape, num_classes = _make_datasets(name, data_root, augment_config)
    if max_batches:
        if not batch_size:
            raise RuntimeError(f"Requested max {max_batches} batches, but batch size was missing ({batch_size}).")
        train_indices = np.random.choice(len(train_data), size=batch_size * max_batches, replace=False)
        reduced_train_data = Subset(train_data, train_indices)
        test_indices = np.random.choice(len(test_data), size=batch_size * max_batches, replace=False)
        reduced_test_data = Subset(test_data, test_indices)
        return reduced_train_data, reduced_test_data, input_shape, num_classes
    else:
        return train_data, test_data, input_shape, num_classes


def check_data_config(config: dict):
    ensure_config_param(config, "data_path", existing_path)
    config["data_path"] = Path(config["data_path"]).expanduser().resolve()  # ensure the type is Path, not string.
    ensure_config_param(config, ["train_config", "dataset"], of_type(str))
    ensure_config_param(config, ["train_config", "batch_size"], _and(of_type(int), gt_zero), required=False)
    ensure_config_param(config, ["train_config", "max_batches"], _and(of_type(int), gt_zero), required=False)
    ensure_config_param(config, ["train_config", "data_augmentation"], validate_transform_list, required=False)


def load_dataset_from_config(config: Dict) -> Tuple[Dataset, Dataset, Union[tuple, torch.Size], int]:
    """
    Load datasets from config. Delegates to `make_datasets()`.

    :param config: The dataset configuration.
    :return:
        train_data (torch.utils.data.Dataset): The training set.
        test_data (torch.utils.data.Dataset): The test set.
        input_shape (torch.Size | tuple): The size of the images [C, H, W].
        num_classes (int): The number of classes for classification.
    """
    return make_datasets(config["train_config"]["dataset"], config["data_path"],
                         config["train_config"].get("batch_size"), config["train_config"].get("max_batches"),
                         config["train_config"].get("data_augmentation", []))


def add_dataset_arg(parser, dflt_data_dir=Path("./data").resolve()):
    """
    Add an argument for the user to specify which search space and dataset they wish to use.
    """
    parser.add_argument("--dataset", type=str.lower, help="The dataset to use.")
    parser.add_argument("--data-path", "--data-dir", metavar="PATH", type=resolved_path, default=dflt_data_dir,
                        help="The root path in which to look for the dataset.")
    parser.add_argument("--data-augmentation", action="store_true", help="Whether to enable default data augmentation.")
    return parser
