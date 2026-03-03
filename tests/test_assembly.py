from copy import deepcopy

import torch

from assembly import Assembly


assembly_config = [
    {"SimpleAdapter": {
        "in_channels": 3,
        "out_channels": 512,
        "num_conv": 1,
        "kernel_size": 1,
        "padding": 0,
        # "in_format": "img",
    }},
    {"ParallelPart": {
        "agg": "avg",
        "out_format": ["img", [256, 14, 14]],
        "parts": [
            {"SimpleAdapter": {
                "in_channels": 512,
                "out_channels": 256,
                "num_conv": 1,
                "kernel_size": 1,
                "padding": 0,
            }},
            {"SimpleAdapter": {
                "in_channels": 512,
                "out_channels": 256,
                "num_conv": 1,
                "kernel_size": 1,
                "padding": 0,
            }}
        ],
    }},
    {"SimpleAdapter": {
        "in_channels": 256,
        "out_channels": 128,
        "num_conv": 1,
        "kernel_size": 1,
        "padding": 0,
    }},
]


def newcfg():
    return deepcopy(assembly_config)


def test_parallel_avg():
    input_shape = [3, 14, 14]
    model = Assembly(newcfg(), input_shape=input_shape)
    out = model(torch.rand(1, *input_shape))  # batch of size 1
    assert out.shape == (1, 128, 14, 14)


def test_parallel_concat():
    cfg = newcfg()
    cfg[1]["ParallelPart"]["agg"] = "concat"
    cfg[1]["ParallelPart"]["out_format"] = ["img", [512, 14, 14]]
    cfg[2]["SimpleAdapter"]["in_channels"] = 512
    input_shape = [3, 14, 14]
    model = Assembly(cfg, input_shape=input_shape)
    out = model(torch.rand(1, *input_shape))  # batch of size 1
    assert out.shape == (1, 128, 14, 14)


def test_parallel_concat_with_seqence_format():
    cfg = newcfg()
    cfg[1]["ParallelPart"]["agg"] = "concat"
    cfg[1]["ParallelPart"]["in_format"] = ["token", [512, 196]]
    cfg[1]["ParallelPart"]["out_format"] = ["token", [512, 196]]
    cfg[1]["ParallelPart"]["parts"] = [{"SimpleAdapter": {"in_channels": 512,
                                                          "out_channels": 256,
                                                          "num_fc": 1}}] * 2
    cfg[2]["SimpleAdapter"]["in_channels"] = 512
    input_shape = [3, 14, 14]
    model = Assembly(cfg, input_shape=input_shape)
    out = model(torch.rand(1, *input_shape))  # batch of size 1
    assert out.shape == (1, 128, 14, 14)
