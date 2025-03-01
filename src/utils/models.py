"""
Provides functions to load a (possibly pre-trained) segment of a model.
"""
import os

import timm
import torch
import torchvision

from .subgraphs import create_sub_network

# WORKING_MODELS = ['resnet50', 'resnet101', 'resnet18', 'resnet50mocov2', 'resnet50byol', 'resnet50simclr',
#                   'swsl_resnext50_32x4d', 'mobilenetv3_large_100',
#                   'vit_small_patch16_224mocov3', 'vit_small_patch16_224', 'vit_base_patch16_224mae', 'vit_base_patch16_224',  'vit_tiny_patch16_224', 'vit_large_patch16_224',
#                   'swin_tiny_patch4_window7_224', 'swin_small_patch4_window7_224', 'swin_base_patch4_window7_224', 'swin_large_patch4_window7_224',
#                   'regnet_y_16gf', 'regnet_y_3_2gf', 'regnet_y_1_6gf', 'regnet_y_8gf', 'regnet_y_32gf', 'regnet_y_800mf']

# Below are the only models that work in the current version of the code.
WORKING_MODELS = ['resnet50', 'resnet101', 'resnet18',
                  'swsl_resnext50_32x4d', 'mobilenetv3_large_100',
                  # 'mobilenetv3_large_075', 'mobilenetv3_small_100',  # Why are these missing from the above list?
                  'vit_small_patch16_224', 'vit_base_patch16_224', 'vit_tiny_patch16_224',  # 'vit_large_patch16_224',
                  'swin_tiny_patch4_window7_224', 'swin_small_patch4_window7_224', 'swin_base_patch4_window7_224',
                  'swin_large_patch4_window7_224']

# TODO: These block splits often skip over some downsamples and maxpools, which could be useful to have separate.
MODEL_ZOO = {
    'resnet18': {"backend": "pytorch",
                 "type": "cnn",
                 "blocks": ['layer1.0', 'layer1.1', 'layer2.0', 'layer2.1',
                            'layer3.0', 'layer3.1', 'layer4.0', 'layer4.1']},
    'resnet50': {"backend": "pytorch",
                 "type": "cnn",
                 "blocks": ['layer1.0', 'layer1.1', 'layer1.2', 'layer2.0',
                            'layer2.1', 'layer2.2', 'layer2.3', 'layer3.0',
                            'layer3.1', 'layer3.2', 'layer3.3', 'layer3.4',
                            'layer3.5', 'layer4.0', 'layer4.1', 'layer4.2']},
    'resnet101': {"backend": "pytorch",
                  "type": "cnn",
                  "blocks": ['layer1.0', 'layer1.1', 'layer1.2', 'layer2.0', 'layer2.1', 'layer2.2', 'layer2.3',
                             'layer3.0', 'layer3.1', 'layer3.2', 'layer3.3', 'layer3.4', 'layer3.5', 'layer3.6',
                             'layer3.7', 'layer3.8', 'layer3.9', 'layer3.10', 'layer3.11', 'layer3.12', 'layer3.13',
                             'layer3.14', 'layer3.15', 'layer3.16', 'layer3.17', 'layer3.18', 'layer3.19', 'layer3.20',
                             'layer3.21', 'layer3.22', 'layer4.0', 'layer4.1', 'layer4.2']},
    'resnet152': {"backend": "pytorch",
                  "type": "cnn",
                  "blocks": ['layer1.0', 'layer1.1', 'layer1.2', 'layer2.0', 'layer2.1', 'layer2.2', 'layer2.3',
                             'layer2.4', 'layer2.5', 'layer2.6', 'layer2.7', 'layer3.0', 'layer3.1', 'layer3.2',
                             'layer3.3', 'layer3.4', 'layer3.5', 'layer3.6', 'layer3.7', 'layer3.8', 'layer3.9',
                             'layer3.10', 'layer3.11', 'layer3.12', 'layer3.13', 'layer3.14', 'layer3.15', 'layer3.16',
                             'layer3.17', 'layer3.18', 'layer3.19', 'layer3.20', 'layer3.21', 'layer3.22', 'layer3.23',
                             'layer3.24', 'layer3.25', 'layer3.26', 'layer3.27', 'layer3.28', 'layer3.29', 'layer3.30',
                             'layer3.31', 'layer3.32', 'layer3.33', 'layer3.34', 'layer3.35', 'layer4.0', 'layer4.1',
                             'layer4.2']},
    'efficientnet_b7': {"backend": "pytorch",
                        "type": "cnn",
                        "blocks": ['features.1.0.block', 'features.1.1.block', 'features.1.2.block',
                                   'features.1.3.block', 'features.2.0.block', 'features.2.1.block',
                                   'features.2.2.block', 'features.2.3.block', 'features.2.4.block',
                                   'features.2.5.block', 'features.2.6.block', 'features.3.0.block',
                                   'features.3.1.block', 'features.3.2.block', 'features.3.3.block',
                                   'features.3.4.block', 'features.3.5.block', 'features.3.6.block',
                                   'features.4.0.block', 'features.4.1.block', 'features.4.2.block',
                                   'features.4.3.block', 'features.4.4.block', 'features.4.5.block',
                                   'features.4.6.block', 'features.4.7.block', 'features.4.8.block',
                                   'features.4.9.block', 'features.5.0.block', 'features.5.1.block',
                                   'features.5.2.block', 'features.5.3.block', 'features.5.4.block',
                                   'features.5.5.block', 'features.5.6.block', 'features.5.7.block',
                                   'features.5.8.block', 'features.5.9.block', 'features.6.0.block',
                                   'features.6.1.block', 'features.6.2.block', 'features.6.3.block',
                                   'features.6.4.block', 'features.6.5.block', 'features.6.6.block',
                                   'features.6.7.block', 'features.6.8.block', 'features.6.9.block',
                                   'features.6.10.block', 'features.6.11.block', 'features.6.12.block',
                                   'features.7.0.block', 'features.7.1.block', 'features.7.2.block',
                                   'features.7.3.block']},
    'efficientnet_b6': {"backend": "pytorch",
                        "type": "cnn",
                        "blocks": ['features.1.0.block', 'features.1.1.block', 'features.1.2.block',
                                   'features.2.0.block', 'features.2.1.block', 'features.2.2.block',
                                   'features.2.3.block', 'features.2.4.block', 'features.2.5.block',
                                   'features.3.0.block', 'features.3.1.block', 'features.3.2.block',
                                   'features.3.3.block', 'features.3.4.block', 'features.3.5.block',
                                   'features.4.0.block', 'features.4.1.block', 'features.4.2.block',
                                   'features.4.3.block', 'features.4.4.block', 'features.4.5.block',
                                   'features.4.6.block', 'features.4.7.block', 'features.5.0.block',
                                   'features.5.1.block', 'features.5.2.block', 'features.5.3.block',
                                   'features.5.4.block', 'features.5.5.block', 'features.5.6.block',
                                   'features.5.7.block', 'features.6.0.block', 'features.6.1.block',
                                   'features.6.2.block', 'features.6.3.block', 'features.6.4.block',
                                   'features.6.5.block', 'features.6.6.block', 'features.6.7.block',
                                   'features.6.8.block', 'features.6.9.block', 'features.6.10.block',
                                   'features.7.0.block', 'features.7.1.block', 'features.7.2.block']},
    'efficientnet_b5': {"backend": "pytorch",
                        "type": "cnn",
                        "blocks": ['features.1.0.block', 'features.1.1.block', 'features.1.2.block',
                                   'features.2.0.block', 'features.2.1.block', 'features.2.2.block',
                                   'features.2.3.block', 'features.2.4.block', 'features.3.0.block',
                                   'features.3.1.block', 'features.3.2.block', 'features.3.3.block',
                                   'features.3.4.block', 'features.4.0.block', 'features.4.1.block',
                                   'features.4.2.block', 'features.4.3.block', 'features.4.4.block',
                                   'features.4.5.block', 'features.4.6.block', 'features.5.0.block',
                                   'features.5.1.block', 'features.5.2.block', 'features.5.3.block',
                                   'features.5.4.block', 'features.5.5.block', 'features.5.6.block',
                                   'features.6.0.block', 'features.6.1.block', 'features.6.2.block',
                                   'features.6.3.block', 'features.6.4.block', 'features.6.5.block',
                                   'features.6.6.block', 'features.6.7.block', 'features.6.8.block',
                                   'features.7.0.block', 'features.7.1.block', 'features.7.2.block']},
    'efficientnet_b4': {"backend": "pytorch",
                        "type": "cnn",
                        "blocks": ['features.1.0.block', 'features.1.1.block', 'features.2.0.block',
                                   'features.2.1.block', 'features.2.2.block', 'features.2.3.block',
                                   'features.3.0.block', 'features.3.1.block', 'features.3.2.block',
                                   'features.3.3.block', 'features.4.0.block', 'features.4.1.block',
                                   'features.4.2.block', 'features.4.3.block', 'features.4.4.block',
                                   'features.4.5.block', 'features.5.0.block', 'features.5.1.block',
                                   'features.5.2.block', 'features.5.3.block', 'features.5.4.block',
                                   'features.5.5.block', 'features.6.0.block', 'features.6.1.block',
                                   'features.6.2.block', 'features.6.3.block', 'features.6.4.block',
                                   'features.6.5.block', 'features.6.6.block', 'features.6.7.block',
                                   'features.7.0.block', 'features.7.1.block']},
    'efficientnet_b3': {"backend": "pytorch",
                        "type": "cnn",
                        "blocks": ['features.1.0.block', 'features.1.1.block', 'features.2.0.block',
                                   'features.2.1.block', 'features.2.2.block', 'features.3.0.block',
                                   'features.3.1.block', 'features.3.2.block', 'features.4.0.block',
                                   'features.4.1.block', 'features.4.2.block', 'features.4.3.block',
                                   'features.4.4.block', 'features.5.0.block', 'features.5.1.block',
                                   'features.5.2.block', 'features.5.3.block', 'features.5.4.block',
                                   'features.6.0.block', 'features.6.1.block', 'features.6.2.block',
                                   'features.6.3.block', 'features.6.4.block', 'features.6.5.block',
                                   'features.7.0.block', 'features.7.1.block']},
    'efficientnet_b2': {"backend": "pytorch",
                        "type": "cnn",
                        "blocks": ['features.1.0.block', 'features.1.1.block', 'features.2.0.block',
                                   'features.2.1.block', 'features.2.2.block', 'features.3.0.block',
                                   'features.3.1.block', 'features.3.2.block', 'features.4.0.block',
                                   'features.4.1.block', 'features.4.2.block', 'features.4.3.block',
                                   'features.5.0.block', 'features.5.1.block', 'features.5.2.block',
                                   'features.5.3.block', 'features.6.0.block', 'features.6.1.block',
                                   'features.6.2.block', 'features.6.3.block', 'features.6.4.block',
                                   'features.7.0.block', 'features.7.1.block']},
    'efficientnet_b1': {"backend": "pytorch",
                        "type": "cnn",
                        "blocks": ['features.1.0.block', 'features.1.1.block', 'features.2.0.block',
                                   'features.2.1.block', 'features.2.2.block', 'features.3.0.block',
                                   'features.3.1.block', 'features.3.2.block', 'features.4.0.block',
                                   'features.4.1.block', 'features.4.2.block', 'features.4.3.block',
                                   'features.5.0.block', 'features.5.1.block', 'features.5.2.block',
                                   'features.5.3.block', 'features.6.0.block', 'features.6.1.block',
                                   'features.6.2.block', 'features.6.3.block', 'features.6.4.block',
                                   'features.7.0.block', 'features.7.1.block']},
    'efficientnet_b0': {"backend": "pytorch",
                        "type": "cnn",
                        "blocks": ['features.1.0.block', 'features.2.0.block', 'features.2.1.block',
                                   'features.3.0.block', 'features.3.1.block', 'features.4.0.block',
                                   'features.4.1.block', 'features.4.2.block', 'features.5.0.block',
                                   'features.5.1.block', 'features.5.2.block', 'features.6.0.block',
                                   'features.6.1.block', 'features.6.2.block', 'features.6.3.block',
                                   'features.7.0.block']},
    'mobilenetv3_large_100': {"backend": "timm",
                              "type": "cnn",
                              "blocks": ['blocks.0.0', 'blocks.1.0', 'blocks.1.1',
                                         'blocks.2.0', 'blocks.2.1', 'blocks.2.2',
                                         'blocks.3.0', 'blocks.3.1', 'blocks.3.2',
                                         'blocks.3.3', 'blocks.4.0', 'blocks.4.1',
                                         'blocks.5.0', 'blocks.5.1', 'blocks.5.2', 'blocks.6.0']},
    'mobilenetv3_large_075': {"backend": "timm",
                              "type": "cnn",
                              "blocks": ['blocks.0.0', 'blocks.1.0', 'blocks.1.1',
                                         'blocks.2.0', 'blocks.2.1', 'blocks.2.2',
                                         'blocks.3.0', 'blocks.3.1', 'blocks.3.2',
                                         'blocks.3.3', 'blocks.4.0', 'blocks.4.1',
                                         'blocks.5.0', 'blocks.5.1', 'blocks.5.2', 'blocks.6.0']},
    'mobilenetv3_small_100': {"backend": "timm",
                              "type": "cnn",
                              "blocks": ['blocks.0.0', 'blocks.1.0', 'blocks.1.1',
                                         'blocks.2.0', 'blocks.2.1', 'blocks.2.2',
                                         'blocks.3.0', 'blocks.3.1', 'blocks.4.0',
                                         'blocks.4.1', 'blocks.4.2', 'blocks.5.0']},
    'mobilenetv3_small_075': {"backend": "timm",
                              "type": "cnn",
                              "blocks": ['blocks.0.0', 'blocks.1.0', 'blocks.1.1',
                                         'blocks.2.0', 'blocks.2.1', 'blocks.2.2',
                                         'blocks.3.0', 'blocks.3.1', 'blocks.4.0',
                                         'blocks.4.1', 'blocks.4.2', 'blocks.5.0']},
    'mobilenetv2_140': {"backend": "timm",
                        "type": "cnn",
                        "blocks": ['blocks.0.0', 'blocks.1.0', 'blocks.1.1', 'blocks.2.0',
                                   'blocks.2.1', 'blocks.2.2', 'blocks.3.0', 'blocks.3.1',
                                   'blocks.3.2', 'blocks.3.3', 'blocks.4.0', 'blocks.4.1',
                                   'blocks.4.2', 'blocks.5.0', 'blocks.5.1', 'blocks.5.2', 'blocks.6.0']},
    'mobilenetv2_120d': {"backend": "timm",
                         "type": "cnn",
                         "blocks": ['blocks.0.0', 'blocks.1.0', 'blocks.1.1', 'blocks.1.2',
                                    'blocks.2.0', 'blocks.2.1', 'blocks.2.2', 'blocks.2.3',
                                    'blocks.2.4', 'blocks.3.0', 'blocks.3.1', 'blocks.3.2',
                                    'blocks.3.3', 'blocks.3.4', 'blocks.3.5', 'blocks.4.0',
                                    'blocks.4.1', 'blocks.4.2', 'blocks.4.3', 'blocks.4.4',
                                    'blocks.5.0', 'blocks.5.1', 'blocks.5.2', 'blocks.5.3',
                                    'blocks.5.4', 'blocks.6.0']},
    'mobilenetv2_110d': {"backend": "timm",
                         "type": "cnn",
                         "blocks": ['blocks.0.0', 'blocks.1.0', 'blocks.1.1', 'blocks.1.2',
                                    'blocks.2.0', 'blocks.2.1', 'blocks.2.2', 'blocks.2.3',
                                    'blocks.3.0', 'blocks.3.1', 'blocks.3.2', 'blocks.3.3',
                                    'blocks.3.4', 'blocks.4.0', 'blocks.4.1', 'blocks.4.2',
                                    'blocks.4.3', 'blocks.5.0', 'blocks.5.1', 'blocks.5.2',
                                    'blocks.5.3', 'blocks.6.0']},
    'mobilenetv2_100': {"backend": "timm",
                        "type": "cnn",
                        "blocks": ['blocks.0.0', 'blocks.1.0', 'blocks.1.1', 'blocks.2.0',
                                   'blocks.2.1', 'blocks.2.2', 'blocks.3.0', 'blocks.3.1',
                                   'blocks.3.2', 'blocks.3.3', 'blocks.4.0', 'blocks.4.1',
                                   'blocks.4.2', 'blocks.5.0', 'blocks.5.1', 'blocks.5.2',
                                   'blocks.6.0']},
    'regnet_y_32gf': {"backend": "pytorch",
                      "type": "cnn",
                      "blocks": ['trunk_output.block1.block1-0', 'trunk_output.block1.block1-1',
                                 'trunk_output.block2.block2-0', 'trunk_output.block2.block2-1',
                                 'trunk_output.block2.block2-2', 'trunk_output.block2.block2-3',
                                 'trunk_output.block2.block2-4', 'trunk_output.block3.block3-0',
                                 'trunk_output.block3.block3-1', 'trunk_output.block3.block3-2',
                                 'trunk_output.block3.block3-3', 'trunk_output.block3.block3-4',
                                 'trunk_output.block3.block3-5', 'trunk_output.block3.block3-6',
                                 'trunk_output.block3.block3-7', 'trunk_output.block3.block3-8',
                                 'trunk_output.block3.block3-9', 'trunk_output.block3.block3-10',
                                 'trunk_output.block3.block3-11', 'trunk_output.block4.block4-0']},
    'regnet_y_16gf': {"backend": "pytorch",
                      "type": "cnn",
                      "blocks": ['trunk_output.block1.block1-0', 'trunk_output.block1.block1-1',
                                 'trunk_output.block2.block2-0', 'trunk_output.block2.block2-1',
                                 'trunk_output.block2.block2-2', 'trunk_output.block2.block2-3',
                                 'trunk_output.block3.block3-0', 'trunk_output.block3.block3-1',
                                 'trunk_output.block3.block3-10', 'trunk_output.block3.block3-2',
                                 'trunk_output.block3.block3-3', 'trunk_output.block3.block3-4',
                                 'trunk_output.block3.block3-5', 'trunk_output.block3.block3-6',
                                 'trunk_output.block3.block3-7', 'trunk_output.block3.block3-8',
                                 'trunk_output.block3.block3-9', 'trunk_output.block4.block4-0']},
    'regnet_y_8gf': {"backend": "pytorch",
                     "type": "cnn",
                     "blocks": ['trunk_output.block1.block1-0', 'trunk_output.block1.block1-1',
                                'trunk_output.block2.block2-0', 'trunk_output.block2.block2-1',
                                'trunk_output.block2.block2-2', 'trunk_output.block2.block2-3',
                                'trunk_output.block3.block3-0', 'trunk_output.block3.block3-1',
                                'trunk_output.block3.block3-2', 'trunk_output.block3.block3-3',
                                'trunk_output.block3.block3-4', 'trunk_output.block3.block3-5',
                                'trunk_output.block3.block3-6', 'trunk_output.block3.block3-7',
                                'trunk_output.block3.block3-8', 'trunk_output.block3.block3-9',
                                'trunk_output.block4.block4-0']},
    'regnet_y_3_2gf': {"backend": "pytorch",
                       "type": "cnn",
                       "blocks": ['trunk_output.block1.block1-0', 'trunk_output.block1.block1-1',
                                  'trunk_output.block2.block2-0', 'trunk_output.block2.block2-1',
                                  'trunk_output.block2.block2-2', 'trunk_output.block2.block2-3',
                                  'trunk_output.block2.block2-4', 'trunk_output.block3.block3-0',
                                  'trunk_output.block3.block3-1', 'trunk_output.block3.block3-2',
                                  'trunk_output.block3.block3-3', 'trunk_output.block3.block3-4',
                                  'trunk_output.block3.block3-5', 'trunk_output.block3.block3-6',
                                  'trunk_output.block3.block3-7', 'trunk_output.block3.block3-8',
                                  'trunk_output.block3.block3-9', 'trunk_output.block3.block3-10',
                                  'trunk_output.block3.block3-11', 'trunk_output.block3.block3-12',
                                  'trunk_output.block4.block4-0']},
    'regnet_y_1_6gf': {"backend": "pytorch",
                       "type": "cnn",
                       "blocks": ['trunk_output.block1.block1-0', 'trunk_output.block1.block1-1',
                                  'trunk_output.block2.block2-0', 'trunk_output.block2.block2-1',
                                  'trunk_output.block2.block2-2', 'trunk_output.block2.block2-3',
                                  'trunk_output.block2.block2-4', 'trunk_output.block2.block2-5',
                                  'trunk_output.block3.block3-0', 'trunk_output.block3.block3-1',
                                  'trunk_output.block3.block3-2', 'trunk_output.block3.block3-3',
                                  'trunk_output.block3.block3-4', 'trunk_output.block3.block3-5',
                                  'trunk_output.block3.block3-6', 'trunk_output.block3.block3-7',
                                  'trunk_output.block3.block3-8', 'trunk_output.block3.block3-9',
                                  'trunk_output.block3.block3-10', 'trunk_output.block3.block3-11',
                                  'trunk_output.block3.block3-12', 'trunk_output.block3.block3-13',
                                  'trunk_output.block3.block3-14', 'trunk_output.block3.block3-15',
                                  'trunk_output.block3.block3-16', 'trunk_output.block4.block4-0',
                                  'trunk_output.block4.block4-1']},
    'regnet_y_800mf': {"backend": "pytorch",
                       "type": "cnn",
                       "blocks": ['trunk_output.block1.block1-0', 'trunk_output.block2.block2-0',
                                  'trunk_output.block2.block2-1', 'trunk_output.block2.block2-2',
                                  'trunk_output.block3.block3-0', 'trunk_output.block3.block3-1',
                                  'trunk_output.block3.block3-2', 'trunk_output.block3.block3-3',
                                  'trunk_output.block3.block3-4', 'trunk_output.block3.block3-5',
                                  'trunk_output.block3.block3-6', 'trunk_output.block3.block3-7',
                                  'trunk_output.block4.block4-0', 'trunk_output.block4.block4-1']},
    'swsl_resnext101_32x8d': {"backend": "timm",
                              "type": "cnn",
                              "blocks": ['layer1.0', 'layer1.1', 'layer1.2', 'layer2.0', 'layer2.1', 'layer2.2',
                                         'layer2.3', 'layer3.0', 'layer3.1', 'layer3.2', 'layer3.3', 'layer3.4',
                                         'layer3.5', 'layer3.6', 'layer3.7', 'layer3.8', 'layer3.9', 'layer3.10',
                                         'layer3.11', 'layer3.12', 'layer3.13', 'layer3.14', 'layer3.15', 'layer3.16',
                                         'layer3.17', 'layer3.18', 'layer3.19', 'layer3.20', 'layer3.21', 'layer3.22',
                                         'layer4.0', 'layer4.1', 'layer4.2']},
    'swsl_resnext50_32x4d': {"backend": "timm",
                             "type": "cnn",
                             "blocks": ['layer1.0', 'layer1.1', 'layer1.2', 'layer2.0',
                                        'layer2.1', 'layer2.2', 'layer2.3', 'layer3.0',
                                        'layer3.1', 'layer3.2', 'layer3.3', 'layer3.4',
                                        'layer3.5', 'layer4.0', 'layer4.1', 'layer4.2']},
    'resnext50_32x4d': {"backend": "pytorch",
                        "type": "cnn",
                        "blocks": ['layer1.0', 'layer1.1', 'layer1.2', 'layer2.0',
                                   'layer2.1', 'layer2.2', 'layer2.3', 'layer3.0',
                                   'layer3.1', 'layer3.2', 'layer3.3', 'layer3.4',
                                   'layer3.5', 'layer4.0', 'layer4.1', 'layer4.2']},
    'vit_tiny_patch16_224': {"backend": "timm",
                             "type": "vit",
                             "blocks": [f'blocks.{i}' for i in range(12)]},
    'vit_small_patch16_224': {"backend": "timm",
                              "type": "vit",
                              "blocks": [f'blocks.{i}' for i in range(12)]},
    'vit_base_patch16_224': {"backend": "timm",
                             "type": "vit",
                             "blocks": [f'blocks.{i}' for i in range(12)]},
    'vit_large_patch16_224': {"backend": "timm",
                              "type": "vit",
                              "blocks": [f'blocks.{i}' for i in range(24)]},
    'swin_t': {"backend": "pytorch",
               "type": "swin",
               "blocks": ['features.0.0', 'features.0.1', 'features.0.2',
                          *(f'features.1.{i}' for i in range(2)), 'features.2',
                          *(f'features.3.{i}' for i in range(2)), 'features.4',
                          *(f'features.5.{i}' for i in range(6)), 'features.6',
                          *(f'features.7.{i}' for i in range(2))]},
    'swin_s': {"backend": "pytorch",
               "type": "swin",
               "blocks": ['features.0.0', 'features.0.1', 'features.0.2',
                          *(f'features.1.{i}' for i in range(2)), 'features.2',
                          *(f'features.3.{i}' for i in range(2)), 'features.4',
                          *(f'features.5.{i}' for i in range(18)), 'features.6',
                          *(f'features.7.{i}' for i in range(2))]},
    'swin_b': {"backend": "pytorch",
               "type": "swin",
               "blocks": ['features.0.0', 'features.0.1', 'features.0.2',
                          *(f'features.1.{i}' for i in range(2)), 'features.2',
                          *(f'features.3.{i}' for i in range(2)), 'features.4',
                          *(f'features.5.{i}' for i in range(18)), 'features.6',
                          *(f'features.7.{i}' for i in range(2))]},
    'swin_v2_t': {"backend": "pytorch",
                  "type": "swin",
                  "blocks": ['features.0.0', 'features.0.1', 'features.0.2',
                             *(f'features.1.{i}' for i in range(2)), 'features.2',
                             *(f'features.3.{i}' for i in range(2)), 'features.4',
                             *(f'features.5.{i}' for i in range(6)), 'features.6',
                             *(f'features.7.{i}' for i in range(2))]},
    'swin_v2_s': {"backend": "pytorch",
                  "type": "swin",
                  "blocks": ['features.0.0', 'features.0.1', 'features.0.2',
                             *(f'features.1.{i}' for i in range(2)), 'features.2',
                             *(f'features.3.{i}' for i in range(2)), 'features.4',
                             *(f'features.5.{i}' for i in range(18)), 'features.6',
                             *(f'features.7.{i}' for i in range(2))]},
    'swin_v2_b': {"backend": "pytorch",
                  "type": "swin",
                  "blocks": ['features.0.0', 'features.0.1', 'features.0.2',
                             *(f'features.1.{i}' for i in range(2)), 'features.2',
                             *(f'features.3.{i}' for i in range(2)), 'features.4',
                             *(f'features.5.{i}' for i in range(18)), 'features.6',
                             *(f'features.7.{i}' for i in range(2))]},
    'inception_v3': {"backend": "pytorch",
                     "type": "cnn",
                     "blocks": ['Conv2d_1a_3x3', 'Conv2d_2a_3x3', 'Conv2d_2b_3x3', 'maxpool1',
                                'Conv2d_3b_1x1', 'Conv2d_4a_3x3', 'maxpool2',
                                'Mixed_5b', 'Mixed_5c', 'Mixed_5d', 'Mixed_6a',
                                'Mixed_6b', 'Mixed_6c', 'Mixed_6d', 'Mixed_6e',
                                'Mixed_7a', 'Mixed_7b', 'Mixed_7c']},
    'inception_resnet_v2': {"backend": "timm",
                            "type": "cnn",
                            "blocks": ['conv2d_1a', 'conv2d_2a', 'conv2d_2b', 'maxpool_3a',
                                       'conv2d_3b', 'conv2d_4a', 'maxpool_5a',
                                       'mixed_5b', 'repeat', 'repeat_1',
                                       'mixed_7a', 'repeat_2', 'block8']},
    'convnext_tiny': {"backend": "timm",
                      "type": "cnn",
                      "blocks": ['stages.0.blocks.0', 'stages.0.blocks.1', 'stages.0.blocks.2',
                                 'stages.1.blocks.0', 'stages.1.blocks.1', 'stages.1.blocks.2',
                                 'stages.2.blocks.0', 'stages.2.blocks.1', 'stages.2.blocks.2',
                                 'stages.2.blocks.3', 'stages.2.blocks.4', 'stages.2.blocks.5',
                                 'stages.2.blocks.6', 'stages.2.blocks.7', 'stages.2.blocks.8',
                                 'stages.3.blocks.0', 'stages.3.blocks.1', 'stages.3.blocks.2']},
    'convnext_small': {"backend": "timm",
                       "type": "cnn",
                       "blocks": ['stages.0.blocks.0', 'stages.0.blocks.1', 'stages.0.blocks.2',
                                  'stages.1.blocks.0', 'stages.1.blocks.1', 'stages.1.blocks.2',
                                  'stages.2.blocks.0', 'stages.2.blocks.1', 'stages.2.blocks.2',
                                  'stages.2.blocks.3', 'stages.2.blocks.4', 'stages.2.blocks.5',
                                  'stages.2.blocks.6', 'stages.2.blocks.7', 'stages.2.blocks.8',
                                  'stages.2.blocks.9', 'stages.2.blocks.10', 'stages.3.blocks.0',
                                  'stages.2.blocks.11', 'stages.2.blocks.12', 'stages.2.blocks.13',
                                  'stages.2.blocks.14', 'stages.2.blocks.15', 'stages.2.blocks.16',
                                  'stages.2.blocks.17', 'stages.2.blocks.18', 'stages.2.blocks.19',
                                  'stages.2.blocks.20', 'stages.2.blocks.21', 'stages.2.blocks.22',
                                  'stages.2.blocks.23', 'stages.2.blocks.24', 'stages.2.blocks.25',
                                  'stages.2.blocks.26', 'stages.3.blocks.1', 'stages.3.blocks.2']},
    'convnext_base': {"backend": "timm",
                      "type": "cnn",
                      "blocks": ['stages.0.blocks.0', 'stages.0.blocks.1', 'stages.0.blocks.2',
                                 'stages.1.blocks.0', 'stages.1.blocks.1', 'stages.1.blocks.2',
                                 'stages.2.blocks.0', 'stages.2.blocks.1', 'stages.2.blocks.2',
                                 'stages.2.blocks.3', 'stages.2.blocks.4', 'stages.2.blocks.5',
                                 'stages.2.blocks.6', 'stages.2.blocks.7', 'stages.2.blocks.8',
                                 'stages.2.blocks.9', 'stages.2.blocks.10', 'stages.2.blocks.11',
                                 'stages.2.blocks.12', 'stages.2.blocks.13', 'stages.2.blocks.14',
                                 'stages.2.blocks.15', 'stages.2.blocks.16', 'stages.2.blocks.17',
                                 'stages.2.blocks.18', 'stages.2.blocks.19', 'stages.2.blocks.20',
                                 'stages.2.blocks.21', 'stages.2.blocks.22', 'stages.2.blocks.23',
                                 'stages.2.blocks.24', 'stages.2.blocks.25', 'stages.2.blocks.26',
                                 'stages.3.blocks.0', 'stages.3.blocks.1', 'stages.3.blocks.2']},
    'convnext_large': {"backend": "timm",
                       "type": "cnn",
                       "blocks": ['stages.0.blocks.0', 'stages.0.blocks.1', 'stages.0.blocks.2',
                                  'stages.1.blocks.0', 'stages.1.blocks.1', 'stages.1.blocks.2',
                                  'stages.2.blocks.0', 'stages.2.blocks.1', 'stages.2.blocks.2',
                                  'stages.2.blocks.3', 'stages.2.blocks.4', 'stages.2.blocks.5',
                                  'stages.2.blocks.6', 'stages.2.blocks.7', 'stages.2.blocks.8',
                                  'stages.2.blocks.9', 'stages.2.blocks.10', 'stages.2.blocks.11',
                                  'stages.2.blocks.12', 'stages.2.blocks.13', 'stages.2.blocks.14',
                                  'stages.2.blocks.15', 'stages.2.blocks.16', 'stages.2.blocks.17',
                                  'stages.2.blocks.18', 'stages.2.blocks.19', 'stages.2.blocks.20',
                                  'stages.2.blocks.21', 'stages.2.blocks.22', 'stages.2.blocks.23',
                                  'stages.2.blocks.24', 'stages.2.blocks.25', 'stages.2.blocks.26',
                                  'stages.3.blocks.0', 'stages.3.blocks.1', 'stages.3.blocks.2']},
    'repvgg_b3': {"backend": "timm",
                  "type": "cnn",
                  "blocks": ['stages.0.0', 'stages.0.1', 'stages.0.2', 'stages.0.3',
                             'stages.1.0', 'stages.1.1', 'stages.1.2', 'stages.1.3',
                             'stages.1.4', 'stages.1.5', 'stages.2.0', 'stages.2.1',
                             'stages.2.2', 'stages.2.3', 'stages.2.4', 'stages.2.5',
                             'stages.2.6', 'stages.2.7', 'stages.2.8', 'stages.2.9',
                             'stages.2.10', 'stages.2.11', 'stages.2.12', 'stages.2.13',
                             'stages.2.14', 'stages.2.15', 'stages.3.0']},
    'repvgg_b2': {"backend": "timm",
                  "type": "cnn",
                  "blocks": ['stages.0.0', 'stages.0.1', 'stages.0.2', 'stages.0.3',
                             'stages.1.0', 'stages.1.1', 'stages.1.2', 'stages.1.3',
                             'stages.1.4', 'stages.1.5', 'stages.2.0', 'stages.2.1',
                             'stages.2.2', 'stages.2.3', 'stages.2.4', 'stages.2.5',
                             'stages.2.6', 'stages.2.7', 'stages.2.8', 'stages.2.9',
                             'stages.2.10', 'stages.2.11', 'stages.2.12', 'stages.2.13',
                             'stages.2.14', 'stages.2.15', 'stages.3.0']},
    'repvgg_b1': {"backend": "timm",
                  "type": "cnn",
                  "blocks": ['stages.0.0', 'stages.0.1', 'stages.0.2', 'stages.0.3',
                             'stages.1.0', 'stages.1.1', 'stages.1.2', 'stages.1.3',
                             'stages.1.4', 'stages.1.5', 'stages.2.0', 'stages.2.1',
                             'stages.2.2', 'stages.2.3', 'stages.2.4', 'stages.2.5',
                             'stages.2.6', 'stages.2.7', 'stages.2.8', 'stages.2.9',
                             'stages.2.10', 'stages.2.11', 'stages.2.12', 'stages.2.13',
                             'stages.2.14', 'stages.2.15', 'stages.3.0']},
    'repvgg_b0': {"backend": "timm",
                  "type": "cnn",
                  "blocks": ['stages.0.0', 'stages.0.1', 'stages.0.2', 'stages.0.3',
                             'stages.1.0', 'stages.1.1', 'stages.1.2', 'stages.1.3',
                             'stages.1.4', 'stages.1.5', 'stages.2.0', 'stages.2.1',
                             'stages.2.2', 'stages.2.3', 'stages.2.4', 'stages.2.5',
                             'stages.2.6', 'stages.2.7', 'stages.2.8', 'stages.2.9',
                             'stages.2.10', 'stages.2.11', 'stages.2.12', 'stages.2.13',
                             'stages.2.14', 'stages.2.15', 'stages.3.0']},
}


def listify(block_spec):
    if isinstance(block_spec, str):
        block_spec = [block_spec]
    elif isinstance(block_spec, tuple):
        block_spec = list(block_spec)
    elif isinstance(block_spec, list):
        block_spec = block_spec
    else:
        raise TypeError('Block spec should be a string or tuple or list.')
    return block_spec


def load_model(model_name, backend, pretrained, ckp_path, verbose):
    if backend == 'timm':
        if ckp_path is not None:
            backbone = timm.create_model(model_name, pretrained=False, scriptable=True)
            if os.path.isfile(ckp_path):
                if verbose:
                    print(f'Loading checkpoint from: {ckp_path}')
                state_dict = torch.load(ckp_path, map_location='cpu')
                missing_keys = backbone.load_state_dict(state_dict, strict=False)
                if verbose:
                    print(missing_keys)
            else:
                raise FileNotFoundError(f'Checkpoint path does not exist: {ckp_path}')
        else:
            backbone = timm.create_model(model_name, pretrained=pretrained, scriptable=True)

    elif backend == 'pytorch':
        if ckp_path is not None:
            backbone = torchvision.models.get_model(model_name, pretrained=False)
            if os.path.isfile(ckp_path):
                if verbose:
                    print(f'Loading checkpoint from: {ckp_path}')
                state_dict = torch.load(ckp_path, map_location='cpu')
                missing_keys = backbone.load_state_dict(state_dict, strict=False)
                if verbose:
                    print(missing_keys)
            else:
                raise FileNotFoundError(f'Checkpoint path does not exist: {ckp_path}')
        else:
            backbone = torchvision.models.get_model(model_name, pretrained=pretrained)

    else:
        raise ValueError(f"Unrecognized backend: '{backend}'")

    return backbone


def load_subnet(model_name, block_input, block_output, backend="pytorch", pretrained=True, ckp_path=None,
                verbose=False):
    block_input = listify(block_input)
    block_output = listify(block_output)
    backbone = load_model(model_name, backend, pretrained, ckp_path, verbose)
    return create_sub_network(backbone, block_input, block_output)
