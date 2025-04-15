"""
Library of adapter modules for stitching.
"""
from functools import partial
from math import sqrt

import torch.nn as nn
import torchvision


NORM_MAPPING = {}
NORM_MAPPING["bn"] = nn.BatchNorm2d
NORM_MAPPING["batch"] = NORM_MAPPING["bn"]
NORM_MAPPING["batchnorm"] = NORM_MAPPING["bn"]
NORM_MAPPING["gn"] = partial(nn.GroupNorm, num_groups=1)
NORM_MAPPING["group"] = NORM_MAPPING["gn"]
NORM_MAPPING["groupnorm"] = NORM_MAPPING["gn"]
NORM_MAPPING["ln"] = nn.LayerNorm
NORM_MAPPING["layer"] = NORM_MAPPING["ln"]
NORM_MAPPING["layernorm"] = NORM_MAPPING["ln"]
NORM_MAPPING["in"] = partial(nn.InstanceNorm2d, affine=True)
NORM_MAPPING["instance"] = NORM_MAPPING["in"]
NORM_MAPPING["instancenorm"] = NORM_MAPPING["in"]


def img2token(x):
    return x.flatten(2).transpose(1, 2)


def token2img(x):
    # Drop the extra tokens if there are any. We expect the number of tokens to be a perfect square, so we can map it
    # onto a square image format. But ViT, for instance, has an extra token.
    # FIXME: The reason for the extra token is that ViT follows a BERT method, where the first token is a "class" token,
    #        and this token is reserved as the output of the classification. So what we are doing here is dropping the
    #        last image patch, and a linear combination of the first feature map pixels will become the output token.
    #        This is pretty weird and should be changed.
    img_sz = int(sqrt(x.shape[1]))
    token_limit = img_sz ** 2
    x = x[:, :token_limit, :]
    return x.view(-1, img_sz, img_sz, x.shape[2]).permute(0, 3, 1, 2)


def conv3x3(in_channels, out_channels, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_channels, out_channels, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False)


class ResNetBasicBlock(nn.Module):
    expansion = 1
    __constants__ = ['downsample']

    def __init__(self, in_channels, out_channels, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_type="bn"):
        super(ResNetBasicBlock, self).__init__()
        if norm_type not in NORM_MAPPING:
            raise ValueError(f"Norm type not recognized: '{norm_type}'.")
        norm_layer = NORM_MAPPING[norm_type]
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        self.in_fmt = "img"
        self.out_fmt = "img"

        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(in_channels, out_channels, stride)
        self.bn1 = norm_layer(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(out_channels, out_channels)
        self.bn2 = norm_layer(out_channels)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNetBottleneck(nn.Module):
    expansion = 4
    __constants__ = ['downsample']

    def __init__(self, in_channels, out_channels, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_type="bn"):
        super(ResNetBottleneck, self).__init__()
        if norm_type not in NORM_MAPPING:
            raise ValueError(f"Norm type not recognized: '{norm_type}'.")
        norm_layer = NORM_MAPPING[norm_type]
        self.in_fmt = "img"
        self.out_fmt = "img"

        width = int((out_channels / self.expansion) * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(in_channels, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, out_channels)
        self.bn3 = norm_layer(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class VisionTransformerBlock(torchvision.models.vision_transformer.EncoderBlock):
    """
    Equivalent to an encoder block from ViT. Default values designed to match ViT-Small.
      - For ViT-Tiny:  num_heads=3,  embed_dim=192
      - For ViT-Small: num_heads=6,  embed_dim=384
      - For ViT-Base:  num_heads=12, embed_dim=768
      - For ViT-Large: num_heads=16, embed_dim=1024
    """
    def __init__(self, num_heads: int = 6, embed_dim: int = 384, mlp_ratio: float = 4., dropout: float = 0.,
                 attention_dropout: float = 0.):
        super().__init__(
            num_heads=num_heads,
            hidden_dim=embed_dim,
            mlp_dim=int(embed_dim * mlp_ratio),
            dropout=dropout,
            attention_dropout=attention_dropout,
        )
        self.in_fmt = "bert"
        self.out_fmt = "bert"


class SimpleAdapter(nn.Module):
    """
    An adapter which can form either a simple layer or slightly more complex blocks.
    """
    def __init__(self, in_channels, out_channels, hid_channels=None, num_fc=0, num_conv=0, kernel_size=3, stride=1,
                 padding=1, leading_norm=True, nonlinearity=True):
        super().__init__()
        if num_fc < 0 or num_conv < 0:
            raise ValueError("num_fc and num_conv must be non-negative.")
        if (num_fc == 0 and num_conv == 0) or (num_fc > 0 and num_conv > 0):
            raise ValueError("Must supply only num_fc or num_conv.")
        self.is_linear = num_fc > 0
        if not hid_channels:
            hid_channels = out_channels

        layers = []
        if self.is_linear:
            self.in_fmt = None
            self.out_fmt = None
            if leading_norm:
                layers.append(nn.LayerNorm(in_channels))
            ichans = in_channels
            ochans = hid_channels
            for i in range(num_fc):
                if i == (num_fc - 1):
                    ochans = out_channels
                layers.append(nn.Linear(ichans, ochans))
                layers.append(nn.LayerNorm(ochans))
                # TODO: Inherited this particular type of ReLU from DeRy. No reason to necessarily prefer it.
                if nonlinearity:
                    layers.append(nn.LeakyReLU(0.1, inplace=True))
                if i == 0:
                    ichans = ochans

        else:
            self.in_fmt = "img"
            self.out_fmt = "img"
            if leading_norm:
                layers.append(nn.BatchNorm2d(in_channels))
            ichans = in_channels
            ochans = hid_channels
            cur_stride = stride
            for i in range(num_conv):
                if i == (num_fc - 1):
                    ochans = out_channels
                layers.append(nn.Conv2d(ichans, ochans, kernel_size=kernel_size, stride=cur_stride, padding=padding))
                layers.append(nn.BatchNorm2d(ochans))
                if nonlinearity:
                    layers.append(nn.LeakyReLU(0.1, inplace=True))
                if i == 0:
                    ichans = ochans
                    cur_stride = 1

        self.adapter = nn.Sequential(*layers)

    def forward(self, x):
        return self.adapter(x)


class DeRyAdapter(nn.Module):
    """
    This is the adapter as it appeared in Deep Model Reassembly (DeRy) (roughly, with modifications).
    From: https://github.com/Adamdad/DeRy/blob/main/mmcls_addon/models/backbones/dery.py

    This does not have `in_fmt`, `out_fmt` fields, so this is now deprecated legacy code.
    """
    def __init__(self, in_channels, out_channels, mode='cnn2cnn', num_fc=0, num_conv=0, stride=1, leading_norm=True,
                 trailing_norm=False, nonlinearity=True):
        super().__init__()
        if num_fc < 0 or num_conv < 0:
            raise ValueError("num_fc and num_conv must be non-negative.")
        if (num_fc == 0 and num_conv == 0) or (num_fc > 0 and num_conv > 0):
            raise ValueError("Must supply only num_fc or num_conv.")
        if mode not in ("cnn2cnn", "cnn2vit", "vit2cnn", "vit2vit"):
            raise ValueError(f"Unrecognized mode: {mode}")
        self.mode = mode

        layers = []
        if num_fc > 0:
            if leading_norm:
                layers.append(nn.LayerNorm(in_channels))
            for i in range(num_fc):
                if i == 0:
                    layers.append(nn.Linear(in_channels, out_channels, bias=False))
                else:
                    layers.append(nn.Linear(out_channels, out_channels, bias=False))
            if trailing_norm:
                layers.append(nn.LayerNorm(out_channels))
            if nonlinearity:
                layers.append(nn.LeakyReLU(0.1, inplace=True))

        elif num_conv > 0:
            if leading_norm:
                layers.append(nn.BatchNorm2d(in_channels))
            for i in range(num_conv):
                if i == 0:
                    layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=stride, stride=stride, padding=0,
                                            bias=False))
                else:
                    layers.append(nn.Conv2d(out_channels, out_channels, kernel_size=1, stride=1, padding=0,
                                            bias=False))
            if trailing_norm:
                layers.append(nn.BatchNorm2d(out_channels))
            if nonlinearity:
                layers.append(nn.LeakyReLU(0.1, inplace=True))

        self.adapter = nn.Sequential(*layers)

    def forward(self, x, input_shape=None):
        if self.mode == 'cnn2vit':
            # CNN 2 Vision Transformer (VIT)
            x = self.adapter(x)
            # TODO: This shape tracking doesn't actually matter and will be removed in the near future.
            imshape = (x.shape[2], x.shape[3])
            x = img2token(x)
            return x, imshape

        elif self.mode == 'cnn2cnn':
            # CNN 2 CNN
            x = self.adapter(x)
            return x, (x.shape[2], x.shape[3])

        elif self.mode == 'vit2cnn':
            # VIT 2 CNN
            x = token2img(x)
            x = self.adapter(x)
            return x, (x.shape[2], x.shape[3])

        elif self.mode in 'vit2vit':
            # VIT/Swin 2 VIT/Swin
            return self.adapter(x), input_shape
