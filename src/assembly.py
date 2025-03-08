"""
A class for assembling blocks and adapters into a single module.
"""
from collections import OrderedDict

import torch.nn as nn

import utils
from adapters import DeRyAdapter
from utils.models import MODEL_ZOO, load_subnet


class ClassifierHead(nn.Module):

    def __init__(self, input_shape, num_classes):
        super().__init__()
        if len(input_shape) != 1:
            raise RuntimeError(f"Cannot stack {type(self).__name__} on top of output of shape: {input_shape}.")
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.linear = nn.Linear(input_shape[0], num_classes)

    def forward(self, x):
        x = self.pool(x)
        x = x.view(x.shape[0], -1)
        return self.linear(x)


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
            block_list,
            adapter_list=None,
            use_head=False,
            use_base=False,
            base_adapter=None,
            block_fixed=True,
            all_fixed=False,
            base_channels=64,
            input_shape=None,
            num_classes=None,
    ):
        super().__init__()
        if not block_list:
            raise ValueError(f"No blocks specified. You must specify at least one block.")
        if (adapter_list is not None) and (len(adapter_list) != len(block_list) - 1):
            raise ValueError("Number of adapters must be one fewer than the number of blocks.")

        if use_base:
            if not input_shape:
                raise ValueError("To use a new base block, you must specify the expected input_shape.")
            self.base = nn.Sequential(OrderedDict([
                ("conv0", nn.Conv2d(input_shape[0], base_channels, kernel_size=7, stride=2, padding=3, bias=False)),
                ("norm0", nn.BatchNorm2d(base_channels)),
                ("relu0", nn.ReLU(inplace=True)),
                ("pool0", nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
            ]))
        else:
            self.base = None

        blocks = []
        block_types = []
        for block_cfg in block_list:
            base_model_name = block_cfg["model_name"]
            block_types.append(MODEL_ZOO[base_model_name.split(".")[0]]["type"])
            blocks.append(load_subnet(**block_cfg))
        self.blocks = nn.ModuleList(blocks)
        self.block_types = block_types

        if adapter_list:
            adapters = []
            for adapter_cfg in adapter_list:
                adapters.append(DeRyAdapter(**adapter_cfg))
            self.adapters = nn.ModuleList(adapters)
        else:
            self.adapters = None

        if base_adapter:
            self.base_adapter = DeRyAdapter(**base_adapter)
        else:
            self.base_adapter = None

        if use_head:
            if not (input_shape and num_classes):
                raise ValueError("To use a classifier head, you must specify the expected input and output shapes.")
            output_shape = utils.calculate_output_shape(self.trunk_forward, input_shape)
            self.head = ClassifierHead(output_shape, num_classes)
        else:
            self.base = None

        if block_fixed:
            for param in self.blocks.parameters():
                param.requires_grad = False

        if all_fixed:
            for param in self.parameters():
                param.requires_grad = False

    def trunk_forward(self, x):
        if self.base:
            x = self.base(x)
        out_shape = (x.shape[2], x.shape[3])
        if self.base_adapter is not None:
            x, out_shape = self.base_adapter(x, out_shape)

        for i, block in enumerate(self.blocks):
            if i > 0 and self.adapters is not None:
                x, out_shape = self.adapters[i - 1](x, out_shape)

            if self.block_types[i] == "swin":
                x, out_shape = block(x, out_shape)
            elif self.block_types[i] == "cnn":
                x = block(x)
                if isinstance(x, dict):
                    x = list(x.values())[0]
                out_shape = (x.shape[2], x.shape[3])
            elif self.block_types[i] == "vit":
                x = block(x)

        return x

    def forward(self, x):
        x = self.trunk_forward(x)
        if self.head:
            x = self.head(x)
        return x
