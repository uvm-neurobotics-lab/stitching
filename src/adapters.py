"""
Library of adapter modules for stitching.
"""
import torch
import torch.nn as nn


class DeRyAdapter(nn.Module):
    """
    This is the adapter as it appeared in Deep Model Reassembly (DeRy).
    From: https://github.com/Adamdad/DeRy/blob/main/mmcls_addon/models/backbones/dery.py
    """
    def __init__(self, input_channel, output_channel, num_fc=0, num_conv=1, mode='cnn2cnn', stride=1):
        super().__init__()
        if (num_fc == 0 and num_conv == 0) or (num_fc > 0 and num_conv > 0):
            raise ValueError('Must supply only num_fc or num_conv.')
        assert mode in ['cnn2cnn', 'cnn2vit', 'vit2cnn', 'vit2vit'], 'mode is not recognized'

        layers = []
        self.mode = mode
        if num_fc > 0:
            layers.append(nn.LayerNorm(input_channel))
            for i in range(num_fc):
                if i == 0:
                    layers.append(nn.Linear(input_channel, output_channel, bias=False))
                else:
                    layers.append(nn.Linear(output_channel, output_channel, bias=False))
            layers.append(nn.LeakyReLU(0.1, inplace=True))

        elif num_conv > 0:
            layers.append(nn.BatchNorm2d(input_channel))
            for i in range(num_conv):
                if i == 0:
                    layers.append(nn.Conv2d(input_channel, output_channel, kernel_size=stride, stride=stride, padding=0,
                                            bias=False))
                else:
                    layers.append(nn.Conv2d(output_channel, output_channel, kernel_size=1, stride=1, padding=0,
                                            bias=False))
            layers.append(nn.LeakyReLU(0.1, inplace=True))

        self.adapter = nn.Sequential(*layers)

    def forward(self, x, input_shape=None):
        if self.mode == 'cnn2vit':
            # CNN 2 Vsion Transformer(VIT)
            x = self.adapter(x)
            return x.flatten(2).transpose(1, 2), (x.shape[2], x.shape[3])

        elif self.mode == 'cnn2cnn':
            # CNN 2 CNN
            x = self.adapter(x)
            return x, (x.shape[2], x.shape[3])

        elif self.mode == 'vit2cnn':
            # VIT 2 CNN
            out_channels = x.shape[2]
            token_num = x.shape[1]
            w, h = input_shape
            # noinspection PyProtectedMember
            torch._assert(w * h == token_num, 'When VIT to CNN, w x h == token_num')

            x = x.view(-1, w, h, out_channels).permute(0, 3, 1, 2)
            x = self.adapter(x)
            return x, (x.shape[2], x.shape[3])

        elif self.mode in 'vit2vit':
            # VIT/Swin 2 VIT/Swin
            return self.adapter(x), input_shape
