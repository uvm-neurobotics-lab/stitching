"""
Module for loading of datasets specified on command line.
"""
import inspect
import numbers
import random
from os import PathLike
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torchvision.transforms.v2 as transforms
from torchvision import datasets as datasets
from torchvision.transforms.v2 import (AutoAugmentPolicy, CenterCrop, Compose, Identity, InterpolationMode, Normalize,
                                       RandomResizedCrop, Resize, RGB, ToDtype, ToImage)
from torch.utils.data import Dataset, Subset

from utils import _and, ensure_config_param, find_matching_class, gt_zero, of_type
from utils.argparsing import existing_path, image_size, resolved_path


###############################################################################
# IMAGE TRANSFORMATIONS
###############################################################################


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
CIFAR10_MEAN = tuple(x / 255 for x in (125.3, 123.0, 113.9))
CIFAR10_STD = tuple(x / 255 for x in (63.0, 62.1, 66.7))
CIFAR100_MEAN = tuple(x / 255 for x in (129.3, 124.1, 112.4))
CIFAR100_STD = tuple(x / 255 for x in (68.2, 65.4, 70.4))


class ResizeKeepRatio:
    """ Resize and Keep Ratio

    Copy & paste from `timm`
    """

    def __init__(
            self,
            size,
            longest=0.,
            interpolation=InterpolationMode.BICUBIC,
            random_scale_prob=0.,
            random_scale_range=(0.85, 1.05),
            random_aspect_prob=0.,
            random_aspect_range=(0.9, 1.11)
    ):
        if isinstance(size, (list, tuple)):
            self.size = tuple(size)
        else:
            self.size = (size, size)
        self.interpolation = interpolation
        self.longest = float(longest)  # [0, 1] where 0 == shortest edge, 1 == longest
        self.random_scale_prob = random_scale_prob
        self.random_scale_range = random_scale_range
        self.random_aspect_prob = random_aspect_prob
        self.random_aspect_range = random_aspect_range

    @staticmethod
    def get_params(
            img,
            target_size,
            longest,
            random_scale_prob=0.,
            random_scale_range=(0.85, 1.05),
            random_aspect_prob=0.,
            random_aspect_range=(0.9, 1.11)
    ):
        source_size = img.size[::-1]  # h, w
        h, w = source_size
        target_h, target_w = target_size
        ratio_h = h / target_h
        ratio_w = w / target_w
        ratio = max(ratio_h, ratio_w) * longest + min(ratio_h, ratio_w) * (1. - longest)
        if random_scale_prob > 0 and random.random() < random_scale_prob:
            ratio_factor = random.uniform(random_scale_range[0], random_scale_range[1])
            ratio_factor = (ratio_factor, ratio_factor)
        else:
            ratio_factor = (1., 1.)
        if random_aspect_prob > 0 and random.random() < random_aspect_prob:
            aspect_factor = random.uniform(random_aspect_range[0], random_aspect_range[1])
            ratio_factor = (ratio_factor[0] / aspect_factor, ratio_factor[1] * aspect_factor)
        size = [round(x * f / ratio) for x, f in zip(source_size, ratio_factor)]
        return size

    def __call__(self, img):
        """
        Args:
            img (PIL Image): Image to be cropped and resized.

        Returns:
            PIL Image: Resized, padded to at least target size, possibly cropped to exactly target size
        """
        size = self.get_params(
            img, self.size, self.longest,
            self.random_scale_prob, self.random_scale_range,
            self.random_aspect_prob, self.random_aspect_range
        )
        img = transforms.functional.resize(img, size, self.interpolation)
        return img

    def __repr__(self):
        format_string = self.__class__.__name__ + '(size={0}'.format(self.size)
        format_string += f', interpolation={self.interpolation})'
        format_string += f', longest={self.longest:.3f})'
        return format_string


def center_crop_or_pad(img: torch.Tensor, output_size: List[int], fill=0) -> torch.Tensor:
    """Center crops and/or pads the given image.
    If the image is torch Tensor, it is expected
    to have [..., H, W] shape, where ... means an arbitrary number of leading dimensions.
    If image size is smaller than output size along any edge, image is padded with 0 and then center cropped.

    Args:
        img (PIL Image or Tensor): Image to be cropped.
        output_size (sequence or int): (height, width) of the crop box. If int or sequence with single int,
            it is used for both directions.
        fill (int, Tuple[int]): Padding color

    Returns:
        PIL Image or Tensor: Cropped image.
    """
    if isinstance(output_size, numbers.Number):
        # noinspection PyTypeChecker
        output_size = (int(output_size), int(output_size))
    elif isinstance(output_size, (tuple, list)) and len(output_size) == 1:
        output_size = (output_size[0], output_size[0])

    _, image_height, image_width = transforms.functional.get_dimensions(img)
    crop_height, crop_width = output_size

    if crop_width > image_width or crop_height > image_height:
        padding_ltrb = [
            (crop_width - image_width) // 2 if crop_width > image_width else 0,
            (crop_height - image_height) // 2 if crop_height > image_height else 0,
            (crop_width - image_width + 1) // 2 if crop_width > image_width else 0,
            (crop_height - image_height + 1) // 2 if crop_height > image_height else 0,
        ]
        img = transforms.functional.pad(img, padding_ltrb, fill=fill)
        _, image_height, image_width = transforms.functional.get_dimensions(img)
        if crop_width == image_width and crop_height == image_height:
            return img

    crop_top = int(round((image_height - crop_height) / 2.0))
    crop_left = int(round((image_width - crop_width) / 2.0))
    return transforms.functional.crop(img, crop_top, crop_left, crop_height, crop_width)


def _setup_size(size, error_msg):
    if isinstance(size, numbers.Number):
        # noinspection PyTypeChecker
        return int(size), int(size)

    if isinstance(size, Sequence) and len(size) == 1:
        return size[0], size[0]

    if len(size) != 2:
        raise ValueError(error_msg)

    return size


class CenterCropOrPad(torch.nn.Module):
    """Crops the given image at the center.
    If the image is torch Tensor, it is expected
    to have [..., H, W] shape, where ... means an arbitrary number of leading dimensions.
    If image size is smaller than output size along any edge, image is padded with 0 and then center cropped.

    Args:
        size (sequence or int): Desired output size of the crop. If size is an
            int instead of sequence like (h, w), a square crop (size, size) is
            made. If provided a sequence of length 1, it will be interpreted as (size[0], size[0]).
    """

    def __init__(self, size, fill=0):
        super().__init__()
        self.size = _setup_size(size, error_msg="Please provide two dimensions (h, w) for size.")
        self.fill = fill

    def forward(self, img):
        """
        Args:
            img (PIL Image or Tensor): Image to be cropped.

        Returns:
            PIL Image or Tensor: Cropped image.
        """
        return center_crop_or_pad(img, self.size, fill=self.fill)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(size={self.size})"


class MaybeToTensor(ToImage):
    """Convert a PIL Image or ndarray to tensor if it's not already one, plus ensure uint8 dtype."""

    def __init__(self) -> None:
        super().__init__()

    def __call__(self, pic) -> torch.Tensor:
        """
        Args:
            pic (PIL Image or numpy.ndarray): Image to be converted to tensor.

        Returns:
            Tensor: Converted image.
        """
        if not isinstance(pic, torch.Tensor):
            pic = transforms.functional.to_image(pic)
        return transforms.functional.to_dtype(pic, torch.uint8, scale=True)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


def validate_transform_list(transform_list):
    # Data augmentation can be a bool, string, null, or list of transforms.
    if isinstance(transform_list, list):
        return all(validate_transform_config(c) for c in transform_list)
    elif (transform_list != "default") and not isinstance(transform_list, bool) and (transform_list is not None):
        raise ValueError("Data augmentation config should be a list of transforms or the 'default' string."
                         f"Instead, we got:\n{transform_list}")
    return True


def validate_preprocess_config(data_preprocessing):
    if not isinstance(data_preprocessing, dict):
        raise ValueError(f"Unrecognized data_preprocessing format: {data_preprocessing}")

    # The config is used as kwargs for the `build_image_transform()` function, so check that the args exist.
    argnames = list(inspect.signature(build_image_transform).parameters.keys())
    for name in data_preprocessing:
        if name not in argnames:
            raise ValueError(f"Argument '{name}' not found in 'build_image_transform' function.")
    return validate_transform_list(data_preprocessing.get("data_augmentation", []))


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
        args["interpolation"] = InterpolationMode(args["interpolation"])  # transform string into enum
    if "policy" in args:
        args["policy"] = AutoAugmentPolicy(args["policy"])  # transform string into enum
    if "transforms" in args:  # Allows for nested transforms, e.g. Compose or RandomApply.
        args["transforms"] = [transform_from_config(c) for c in args["transforms"]]
    TClass = getattr(transforms, cls_name)
    return TClass(**args)


def augmenter_from_config(transform_list):
    """Parse the given list of configs into a single transform containing all configured data augmentation."""
    if not transform_list:
        return Identity()
    return Compose([transform_from_config(t) for t in transform_list])


def build_image_transform(
    is_train: bool,
    image_size: Union[int, Tuple[int, int]],
    to_rgb: bool = True,
    mean: Optional[Tuple[float, ...]] = IMAGENET_MEAN,
    std: Optional[Tuple[float, ...]] = IMAGENET_STD,
    resize_mode: Optional[str] = "shortest",
    interpolation: Optional[str] = "bicubic",
    random_resize_scale: Tuple[float, float] = (0.9, 1.0),
    data_augmentation: Optional[Sequence[Dict[str, Dict]]] = None,
):
    """
    Create image data transforms specified in the given config. The code for resizing images is largely based on code
    from `open_clip` ([see here](https://github.com/mlfoundations/open_clip/blob/d3cdb734a2710feeb4c6307df037afa5f786a3e1/src/open_clip/transform.py#L323C1-L441C1)).

    Args:
        is_train: If True, then data augmentation will be applied (RandomResizedCrop plus any augmentations defined by
                  `data_augmentation` rules).
        image_size: Requested size of the image. For training, this will result in a RandomResizedCrop.
        to_rgb: If True, convert greyscale images to RGB.
        mean: Mean to use for normalizing the augmented image.
        std: Std dev to use for normalizing the augmented image.
        resize_mode: Valid choices are 'shortest', 'longest', or 'squash'. This only applies to test/validation image
                     resizing. 'shortest' and 'longest' preserve the aspect ratio. 'shortest' crops the image, while
                     'longest' pads the image with black fill color. 'squash' simply resizes the image to the specified
                     size, without any regard to the aspect ratio.
        interpolation: The interpolation to use for resizing, 'bilinear' or 'bicubic'.
        random_resize_scale: The `scale` parameter for `RandomResizedCrop`.
        data_augmentation: The config specifying any desired data augmentations for training.

    Returns:
        A composition of transforms which can be passed to a dataset.
    """
    mean = mean or IMAGENET_MEAN
    if not isinstance(mean, (list, tuple)):
        if isinstance(mean, int):
            mean = (mean,) * 3
        else:
            raise ValueError(f"Invalid mean value: {mean}")

    std = std or IMAGENET_STD
    if not isinstance(std, (list, tuple)):
        if isinstance(std, int):
            std = (std,) * 3
        else:
            raise ValueError(f"Invalid std value: {std}")

    interpolation = interpolation or 'bicubic'
    if interpolation not in ('bicubic', 'bilinear'):
        raise ValueError(f"Invalid interpolation value: '{interpolation}'")
    interpolation_mode = InterpolationMode.BILINEAR if interpolation == 'bilinear' else InterpolationMode.BICUBIC

    resize_mode = resize_mode or 'shortest'
    if resize_mode not in ('shortest', 'longest', 'squash'):
        raise ValueError(f"Invalid resize_mode value: '{resize_mode}'")

    if not isinstance(random_resize_scale, (list, tuple)) or len(random_resize_scale) != 2:
        raise ValueError(f"Invalid scale value: {random_resize_scale}")

    transforms = [
        MaybeToTensor(),
    ]
    if to_rgb:
        transforms.append(RGB())

    if is_train:
        transforms.extend([
            RandomResizedCrop(image_size, scale=random_resize_scale, interpolation=interpolation_mode),
            augmenter_from_config(data_augmentation),
        ])
    else:
        if resize_mode == 'longest':
            transforms.extend([
                ResizeKeepRatio(image_size, interpolation=interpolation_mode, longest=1),
                CenterCropOrPad(image_size, fill=0),
            ])
        elif resize_mode == 'squash':
            if isinstance(image_size, int):
                image_size = (image_size, image_size)
            transforms.append(Resize(image_size, interpolation=interpolation_mode))
        else:
            assert resize_mode == 'shortest'
            if not isinstance(image_size, (tuple, list)):
                image_size = (image_size, image_size)
            if image_size[0] == image_size[1]:
                # simple case, use torchvision built-in Resize w/ shortest edge mode (scalar size arg)
                transforms.append(Resize(image_size[0], interpolation=interpolation_mode))
            else:
                # resize shortest edge to matching target dim for non-square target
                transforms.append(ResizeKeepRatio(image_size))
            transforms.append(CenterCrop(image_size))

    transforms.extend([ToDtype(torch.float32, scale=True), Normalize(mean=mean, std=std)])
    return Compose(transforms)


def _default_transform_list(dataset_name):
    policy = "imagenet"
    if dataset_name.startswith("cifar"):
        policy = "cifar10"
    elif dataset_name == "svhn":
        policy = "svhn"
    tlist = [
        {"RandomHorizontalFlip": {"p": 0.5}},
        {"AutoAugment": {"policy": policy}},
    ]
    return tlist


# noinspection PyTypeChecker
def _ensure_default_preprocess_config(
        dataset_name: str, preprocess_config: Optional[Union[Dict, bool, Sequence]] = None) -> Dict[str, Any]:
    """
    Return a new version of `preprocess_config` which reformats the config and fills in any missing details with defaults.

    Args:
        dataset_name: The name of the dataset, used for choosing a default data augment policy.
        preprocess_config: The existing config, if any.

    Returns:
        An augment config in the correct format. Can be used as keyword args for `build_image_transform()`.
    """
    dataset_name = dataset_name.lower()

    if preprocess_config is None:
        preprocess_config = {}
    elif isinstance(preprocess_config, dict):
        tlist = preprocess_config.get("data_augmentation")
        if (isinstance(tlist, bool) and tlist) or (tlist == "default"):
            # If data_augmentation = True or "default", supply a default augmenter.
            preprocess_config["data_augmentation"] = _default_transform_list(dataset_name)
        elif (tlist is not None) and (not isinstance(tlist, list)):
            del preprocess_config["data_augmentation"]  # Remove "data_augmentation: false"
    else:
        raise ValueError(f"Unrecognized augment config type: \n{preprocess_config}")

    # Default to ImageNet standard size.
    preprocess_config.setdefault("image_size", 224)

    if dataset_name == "cifar10":
        preprocess_config.setdefault("mean", CIFAR10_MEAN)
        preprocess_config.setdefault("std", CIFAR10_STD)
    elif dataset_name == "cifar100":
        preprocess_config.setdefault("mean", CIFAR100_MEAN)
        preprocess_config.setdefault("std", CIFAR100_STD)
    elif dataset_name.startswith("imagenet"):
        preprocess_config.setdefault("mean", IMAGENET_MEAN)
        preprocess_config.setdefault("std", IMAGENET_STD)

    return preprocess_config


###############################################################################
# DATASETS
###############################################################################


class DTD(datasets.DTD):
    """ A version of DTD that combines the "train" and "val" splits into a single training split. """
    def __init__(
        self,
        root: Union[str, Path],
        train: bool = True,
        partition: int = 1,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = False,
    ) -> None:
        self._split = "train" if train else "test"
        self._splits = ["train", "val"] if train else ["test"]
        if not isinstance(partition, int) and not (1 <= partition <= 10):
            raise ValueError(
                f"Parameter 'partition' should be an integer with `1 <= partition <= 10`, "
                f"but got {partition} instead"
            )
        self._partition = partition
        self.loader = datasets.folder.default_loader

        # NOTE: Do not call the immediate superclass. Skip over it and call the "grandparent".
        super(datasets.DTD, self).__init__(root, transform=transform, target_transform=target_transform)
        self._base_folder = Path(self.root) / type(self).__name__.lower()
        self._data_folder = self._base_folder / "dtd"
        self._meta_folder = self._data_folder / "labels"
        self._images_folder = self._data_folder / "images"

        if download:
            self._download()

        if not self._check_exists():
            raise RuntimeError("Dataset not found. You can use download=True to download it")

        self._image_files = []
        classes = []
        for split in self._splits:
            with open(self._meta_folder / f"{split}{self._partition}.txt") as file:
                for line in file:
                    cls, name = line.strip().split("/")
                    self._image_files.append(self._images_folder.joinpath(cls, name))
                    classes.append(cls)

        self.classes = sorted(set(classes))
        self.class_to_idx = dict(zip(self.classes, range(len(self.classes))))
        self._labels = [self.class_to_idx[cls] for cls in classes]


class EuroSAT(datasets.ImageFolder):
    def __init__(self, data_root: Path, is_train: bool, preprocess_config: dict):
        split = "train" if is_train else "test"
        transform = build_image_transform(is_train, **preprocess_config)
        super().__init__(data_root / "eurosat" / "splits" / split, transform=transform)

        # Edit the class names.
        idx_to_class = dict((v, k) for k, v in self.class_to_idx.items())
        self.classes = [idx_to_class[i].replace("_", " ") for i in range(len(idx_to_class))]
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}


class RESISC45(datasets.ImageFolder):
    def __init__(self, data_root: Path, is_train: bool, preprocess_config: dict):
        split = "train" if is_train else "test"
        transform = build_image_transform(is_train, **preprocess_config)
        super().__init__(data_root / "resisc45" / split, transform=transform)

        # Edit the class names.
        idx_to_class = dict((v, k) for k, v in self.class_to_idx.items())
        self.classes = [idx_to_class[i].replace("_", " ") for i in range(len(idx_to_class))]
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}


class FER2013(datasets.ImageFolder):
    def __init__(self, data_root: Path, is_train: bool, preprocess_config: dict):
        split = "train" if is_train else "test"
        transform = build_image_transform(is_train, **preprocess_config)
        super().__init__(data_root / "fer2013" / split, transform=transform)

        # Edit the class names.
        idx_to_class = dict((v, k) for k, v in self.class_to_idx.items())
        self.classes = [idx_to_class[i].replace("_", " ") for i in range(len(idx_to_class))]
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}


class SUN397(datasets.ImageFolder):
    def __init__(self, data_root: Path, is_train: bool, preprocess_config: dict):
        split = "train" if is_train else "val"
        transform = build_image_transform(is_train, **preprocess_config)
        super().__init__(data_root / "sun397" / split, transform=transform)

        # Edit the class names.
        idx_to_class = dict((v, k) for k, v in self.class_to_idx.items())
        self.classes = [idx_to_class[i][2:].replace("_", " ") for i in range(len(idx_to_class))]
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}


def make_torchvision_dataset(name: str, data_root: Path, is_train: bool, preprocess_config: dict):
    # Find the torchvision dataset with a matching name.
    DatasetClass = find_matching_class(datasets, name, raise_error=False)
    if DatasetClass is None:
        raise ValueError(f"Unrecognized dataset: {name}")

    kwargs = {}
    if name.startswith("cifar") or name.find("mnist") >= 0:
        kwargs["train"] = is_train
    else:
        kwargs["split"] = "train" if is_train else "test"
    if not (name == "stanfordcars" or name == "fer2013"):
        kwargs["download"] = True
    if name == "emnist":
        kwargs["split"] = "digits"
    transform = build_image_transform(is_train, **preprocess_config)
    return DatasetClass(data_root, transform=transform, **kwargs)


def make_dtd(data_root: Path, is_train: bool, preprocess_config: dict):
    transform = build_image_transform(is_train, **preprocess_config)
    return DTD(data_root, is_train, transform=transform, download=True)


def make_imagenet(split: str, data_root: Path, preprocess_config: dict):
    transform = build_image_transform(split == "train", **preprocess_config)
    return datasets.ImageFolder(data_root / "imagenet" / split, transform)


def get_data_dims(dataset):
    return next(iter(dataset))[0].shape


def _make_datasets(name, data_root, preprocess_config=None):
    name = name.lower()
    preprocess_config = _ensure_default_preprocess_config(name, preprocess_config)

    if name == "imagenet" or name == "imagenet-1k":
        trainset = make_imagenet("train", data_root, preprocess_config=preprocess_config)
        testset = make_imagenet("val", data_root, preprocess_config=preprocess_config)
        return trainset, testset, get_data_dims(trainset), 1000
    elif name == "dtd":
        trainset = make_dtd(data_root, True, preprocess_config=preprocess_config)
        testset = make_dtd(data_root, False, preprocess_config=preprocess_config)
        return trainset, testset, get_data_dims(trainset), len(trainset.classes)
    elif name == "eurosat":
        trainset = EuroSAT(data_root, True, preprocess_config=preprocess_config)
        testset = EuroSAT(data_root, False, preprocess_config=preprocess_config)
        return trainset, testset, get_data_dims(trainset), len(trainset.classes)
    elif name == "fer2013":
        trainset = FER2013(data_root, True, preprocess_config=preprocess_config)
        testset = FER2013(data_root, False, preprocess_config=preprocess_config)
        return trainset, testset, get_data_dims(trainset), len(trainset.classes)
    elif name == "resisc45":
        trainset = RESISC45(data_root, True, preprocess_config=preprocess_config)
        testset = RESISC45(data_root, False, preprocess_config=preprocess_config)
        return trainset, testset, get_data_dims(trainset), len(trainset.classes)
    elif name == "sun397":
        trainset = SUN397(data_root, True, preprocess_config=preprocess_config)
        testset = SUN397(data_root, False, preprocess_config=preprocess_config)
        return trainset, testset, get_data_dims(trainset), len(trainset.classes)
    else:
        # This final clause will raise a readable error if the dataset is unrecognized.
        trainset = make_torchvision_dataset(name, data_root, True, preprocess_config=preprocess_config)
        testset = make_torchvision_dataset(name, data_root, False, preprocess_config=preprocess_config)
        return trainset, testset, get_data_dims(trainset), len(trainset.classes)


def make_datasets(
        name: str,
        data_root: Union[str, PathLike],
        batch_size: int = None,
        max_batches: int = None,
        augment_config: Union[Dict, bool, Sequence, None] = None
) -> Tuple[Dataset, Dataset, Union[tuple, torch.Size], int]:
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


###############################################################################
# CONFIG AND ARG PARSING
###############################################################################


def check_data_config(config: dict):
    ensure_config_param(config, "data_path", existing_path)
    config["data_path"] = Path(config["data_path"]).expanduser().resolve()  # ensure the type is Path, not string.
    ensure_config_param(config, ["train_config", "dataset"], of_type(str))
    ensure_config_param(config, ["train_config", "batch_size"], _and(of_type(int), gt_zero), required=False)
    ensure_config_param(config, ["train_config", "max_batches"], _and(of_type(int), gt_zero), required=False)
    ensure_config_param(config, ["train_config", "data_preprocessing"], validate_preprocess_config, required=False)


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
                         config["train_config"].get("data_preprocessing"))


def add_dataset_arg(parser, dflt_data_dir=Path("./data").resolve()):
    """
    Add an argument for the user to specify which search space and dataset they wish to use.
    """
    parser.add_argument("--dataset", type=str.lower, help="The dataset to use.")
    parser.add_argument("--data-path", "--data-dir", metavar="PATH", type=resolved_path, default=dflt_data_dir,
                        help="The root path in which to look for the dataset.")
    parser.add_argument("--data-augmentation", action="store_true", help="Whether to enable default data augmentation.")
    parser.add_argument("--image-size", "--resize", type=image_size,
                        help="Desired image size, if different from the dataset's native size.")
    return parser
