"""
A class for assembling blocks and adapters into a single module.
"""
from collections import OrderedDict

import torch.nn as nn

from adapters import DeRyAdapter
from utils.models import MODEL_ZOO, load_subnet


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
            use_base=False,
            base_adapter=None,
            block_fixed=True,
            all_fixed=False,
            base_channels=64,
            in_channels=3,
    ):
        super().__init__()
        if not block_list:
            raise ValueError(f"No blocks specified. You must specify at least one block.")
        if (adapter_list is not None) and (len(adapter_list) != len(block_list) - 1):
            raise ValueError("Number of adapters must be one fewer than the number of blocks.")

        if use_base:
            self.base = nn.Sequential(OrderedDict([
                ("conv0", nn.Conv2d(in_channels, base_channels, kernel_size=7, stride=2, padding=3, bias=False)),
                ("norm0", nn.BatchNorm2d(base_channels)),
                ("relu0", nn.ReLU(inplace=True)),
                ("pool0", nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
            ]))
        else:
            self.base = None

        blocks = []
        block_types = []
        for block_cfg in block_list:
            model_name = block_cfg["model_name"]
            block_types.append(MODEL_ZOO[model_name]["type"])
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

        if block_fixed:
            for param in self.blocks.parameters():
                param.requires_grad = False

        if all_fixed:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, x):
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
