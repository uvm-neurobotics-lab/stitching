"""
A class for assembling blocks and adapters into a single module.
"""
import numpy as np
import torch.nn as nn

import adapters
import utils
from utils.models import MODEL_ZOO, load_subnet


class Subnet(nn.Module):

    def __init__(self, model_name, output_type=None, **kwargs):
        super().__init__()
        self.output_type = output_type
        self.net_type = MODEL_ZOO[model_name.split(".")[0]]["type"]
        self.net = load_subnet(model_name=model_name, **kwargs)

    def forward(self, x):
        return self.net(x)


class BaseBlock(nn.Module):
    """Used as the base of assemblies in the DeRy paper."""

    def __init__(self, input_shape, channels):
        super().__init__()
        self.conv = nn.Conv2d(input_shape[0], channels, kernel_size=7, stride=2, padding=3, bias=False)
        self.norm = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.net_type = "cnn"

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.relu(x)
        x = self.pool(x)
        return x


class ClassifierHead(nn.Module):

    def __init__(self, input_shape, num_classes):
        super().__init__()
        if len(input_shape) != 1:
            raise RuntimeError(f"Cannot stack {type(self).__name__} on top of output of shape: {input_shape}.")
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.linear = nn.Linear(input_shape[0], num_classes)
        self.net_type = "cnn"
        self.output_type = "vector"

    def forward(self, x):
        x = self.pool(x)
        x = x.view(x.shape[0], -1)
        return self.linear(x)


def validate_part_list(part_list):
    if not part_list:
        raise ValueError(f"No assembly parts specified. You must specify at least one part.")
    if not isinstance(part_list, list):
        raise ValueError(f"Assembly config should be a list of parts.")
    return all(validate_part(c) for c in part_list)


def validate_part(part_cfg):
    if not isinstance(part_cfg, dict):
        raise ValueError(f"Unrecognized part config format: {part_cfg}")
    if len(part_cfg) > 1:
        raise ValueError(f"Each part in the part list should consist of a single class with args.")
    cls_name, args = next(iter(part_cfg.items()))
    part_class_from_name(cls_name)
    if args and not isinstance(args, dict):
        raise ValueError(f"Expected a dictionary of arguments for {cls_name}, but instead got: {args}")
    return True


def part_class_from_name(cls_name):
    if hasattr(adapters, cls_name):
        return getattr(adapters, cls_name)
    elif cls_name in globals():
        return globals()[cls_name]
    else:
        raise RuntimeError(f"Assembly part class not found: '{cls_name}'")


def part_from_config(part_cfg):
    if not validate_part(part_cfg):
        raise ValueError(f"Invalid part config:\n{part_cfg}")
    cls_name, args = next(iter(part_cfg.items()))
    PartClass = part_class_from_name(cls_name)
    args = args.copy() if args is not None else {}
    if "frozen" in args:
        del args["frozen"]
    return PartClass(**args)


def freeze(part):
    for param in part.parameters():
        param.requires_grad = False


class Assembly(nn.Module):
    """
    A class which takes a list of blocks and a list of adapters and assembles them all in a sequence.

    The user can choose to leave out the adapters if they only want to use blocks (this assumes all blocks'
    inputs/outputs are compatible with each other). Also provides options for adding a prescribed block at the beginning
    of the model.

    This is based in part on Deep Model Reassembly:
    https://github.com/Adamdad/DeRy/blob/main/mmcls_addon/models/backbones/dery.py
    """

    def __init__(
            self,
            config,
            head_cfg=None,
            input_shape=None,
    ):
        super().__init__()
        if not validate_part_list(config):
            raise ValueError(f"Invalid part list config:\n{config}")

        part_list = []
        for c in config:
            part = part_from_config(c)
            part_list.append(part)
            part_args = next(iter(c.values()))
            if part_args is not None and part_args.get("frozen"):
                freeze(part)
        self.parts = nn.ModuleList(part_list)

        if head_cfg:
            if not input_shape:
                raise ValueError("To use a classifier head, you must specify the expected input shape.")
            if not isinstance(head_cfg, dict) or len(head_cfg) > 1:
                raise ValueError(f"Unrecognized head config format: {head_cfg}."
                                 " Should consist of a single class with args.")
            output_shape = utils.calculate_output_shape(self.trunk_forward, input_shape)
            head_args = next(iter(head_cfg.values()))
            head_args["input_shape"] = output_shape
            self.head = part_from_config(head_cfg)
            if head_args.get("frozen"):
                freeze(self.head)
        else:
            self.head = None

    def trunk_forward(self, x):
        # TODO: This out_shape and net_type stuff is a complete mess, inherited from DeRy. Needs refactoring.
        #       Ultimately we just need to know how to transform b/w token-based shapes and image-based shapes.
        out_shape = (x.shape[2], x.shape[3])
        for i, part in enumerate(self.parts):
            if not hasattr(part, "net_type"):
                # If no net type, we assume it already handles shape as input and output.
                x, out_shape = part(x, out_shape)
                if isinstance(x, dict):
                    x = list(x.values())[0]
            elif part.net_type == "swin":
                x = part(x, out_shape)
                if isinstance(x, dict):
                    x = list(x.values())[0]
                if getattr(part, "output_type", None) == "vector":
                    out_shape = [1, 1]
                else:
                    x, out_shape = x
            elif part.net_type == "cnn":
                x = part(x)
                if isinstance(x, dict):
                    x = list(x.values())[0]
                if getattr(part, "output_type", None) == "vector":
                    out_shape = [1, 1]
                else:
                    out_shape = (x.shape[2], x.shape[3])
            elif part.net_type == "vit":
                x = part(x)
                if isinstance(x, dict):
                    x = list(x.values())[0]
                if getattr(part, "output_type", None) == "vector":
                    out_shape = [1, 1]
                else:
                    sz = int(np.sqrt(x.shape[1]))
                    out_shape = [sz, sz]

        return x

    def forward(self, x):
        x = self.trunk_forward(x)
        if self.head:
            x = self.head(x)
        return x
