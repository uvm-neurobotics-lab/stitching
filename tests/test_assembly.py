from copy import deepcopy

import torch

from assembly import Assembly, unfreeze
from utils.logging import eval_mode


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
    cfg[1]["ParallelPart"]["agg"] = "concat_channels"
    cfg[1]["ParallelPart"]["in_format"] = ["token", [512, 196]]
    cfg[1]["ParallelPart"]["out_format"] = ["token", [256 * 2, 196]]
    cfg[1]["ParallelPart"]["parts"] = [{"SimpleAdapter": {"in_channels": 512,
                                                          "out_channels": 256,
                                                          "num_fc": 1}}] * 2
    cfg[2]["SimpleAdapter"]["in_channels"] = 512
    input_shape = [3, 14, 14]
    model = Assembly(cfg, input_shape=input_shape)
    out = model(torch.rand(1, *input_shape))  # batch of size 1
    assert out.shape == (1, 128, 14, 14)


def test_parallel_concat_sequence_with_seqence_format():
    cfg = newcfg()
    cfg[1]["ParallelPart"]["agg"] = "concat_sequence"
    cfg[1]["ParallelPart"]["in_format"] = ["token", [512, 14**2]]
    cfg[1]["ParallelPart"]["out_format"] = ["token", [256, 14**2 * 2]]
    cfg[1]["ParallelPart"]["parts"] = [{"SimpleAdapter": {"in_channels": 512,
                                                          "out_channels": 256,
                                                          "num_fc": 1}}] * 2
    cfg[2]["SimpleAdapter"]["num_fc"] = 1
    cfg[2]["SimpleAdapter"]["num_conv"] = 0
    input_shape = [3, 14, 14]
    model = Assembly(cfg, input_shape=input_shape)
    out = model(torch.rand(1, *input_shape))  # batch of size 1
    assert out.shape == (1, 14**2 * 2, 128)


def frozen_cfg():
    """ The standard config, but with the first adapter frozen. """
    cfg = newcfg()
    cfg[0]["SimpleAdapter"]["frozen"] = True
    return cfg


def test_frozen_part_stays_in_eval_mode_after_train():
    # `nn.Module.train()` recurses into all children, so without special handling it would undo freezing.
    input_shape = [3, 14, 14]
    model = Assembly(frozen_cfg(), input_shape=input_shape)
    model.train(True)
    assert model.training
    assert not model.parts[0].training, "Frozen part should have been held in eval mode."
    assert model.parts[1].training, "Unfrozen parts should follow the model's mode."
    assert model.parts[2].training


def test_frozen_part_does_not_update_running_stats():
    input_shape = [3, 14, 14]
    model = Assembly(frozen_cfg(), input_shape=input_shape)
    model.train(True)
    frozen_norm = model.parts[0].adapter[0]  # The leading BatchNorm2d of the frozen adapter.
    live_norm = model.parts[2].adapter[0]  # The same layer in an unfrozen adapter.
    frozen_before = frozen_norm.running_mean.clone()
    live_before = live_norm.running_mean.clone()

    model(torch.rand(4, *input_shape))

    assert torch.equal(frozen_norm.running_mean, frozen_before), "Frozen part's running stats drifted."
    assert not torch.equal(live_norm.running_mean, live_before), "Unfrozen part's running stats should update."


def test_frozen_part_inside_parallel_part_stays_in_eval_mode():
    cfg = newcfg()
    cfg[1]["ParallelPart"]["parts"][0]["SimpleAdapter"]["frozen"] = True
    model = Assembly(cfg, input_shape=[3, 14, 14])
    model.train(True)
    parallel = model.parts[1]
    assert parallel.training
    assert not parallel.parts[0].training, "Frozen sub-part should have been held in eval mode."
    assert parallel.parts[1].training


def test_unfreeze_restores_training_mode():
    input_shape = [3, 14, 14]
    model = Assembly(frozen_cfg(), input_shape=input_shape)
    unfreeze(model)
    model.train(True)
    assert all(p.requires_grad for p in model.parameters())
    assert model.parts[0].training, "After unfreezing, all parts should follow the model's mode."

    norm = model.parts[0].adapter[0]
    before = norm.running_mean.clone()
    model(torch.rand(4, *input_shape))
    assert not torch.equal(norm.running_mean, before), "Unfrozen part's running stats should update again."


def test_head_construction_does_not_disturb_the_trunk():
    # Building a head runs a dry-run forward pass to discover the trunk's output shape. In training mode that pushed
    # a batch of zeros through the trunk's normalization layers, pulling their running statistics 10% toward the
    # statistics of that all-zero batch. On a freshly initialized BatchNorm the mean happens to survive (it is
    # already zero), so assert on the variance and the batch count, which move either way -- and note that on a
    # *pretrained* trunk the mean is damaged too.
    with_head = Assembly(newcfg(), head={"ClassifierHead": {"num_classes": 10}}, input_shape=[3, 14, 14])

    norm = with_head.parts[0].adapter[0]
    assert norm.num_batches_tracked.item() == 0, "Dry run was counted as a training batch."
    assert torch.equal(norm.running_var, torch.ones_like(norm.running_var)), \
        "Dry run perturbed the trunk's running statistics."
    assert with_head.training, "Dry run should have restored the model's original training mode."


def test_dry_run_guard_protects_nonzero_stats():
    # The same invariant stated the way it actually bites on a pretrained trunk: statistics that are not already at
    # their initial values. This exercises the exact guard `Assembly.__init__` wraps its dry run in. Without it, one
    # pass of zeros pulls a mean of 2.0 down to 1.8 and a variance of 3.0 down to 2.7.
    input_shape = [3, 14, 14]
    model = Assembly(newcfg(), input_shape=input_shape)
    norm = model.parts[0].adapter[0]
    norm.running_mean.fill_(2.0)
    norm.running_var.fill_(3.0)

    with eval_mode(model):
        model.trunk_forward(torch.zeros((3,) + tuple(input_shape)))

    assert torch.equal(norm.running_mean, torch.full_like(norm.running_mean, 2.0))
    assert torch.equal(norm.running_var, torch.full_like(norm.running_var, 3.0))


def test_eval_mode_still_reaches_unfrozen_parts():
    model = Assembly(frozen_cfg(), input_shape=[3, 14, 14])
    model.eval()
    assert not any(p.training for p in model.parts), "eval() should put every part in eval mode."


def test_parallel_concat_sequence_plus_transformer_block():
    cfg = newcfg()
    cfg[1]["ParallelPart"]["agg"] = "concat_sequence"
    cfg[1]["ParallelPart"]["in_format"] = ["token", [512, 14**2]]
    cfg[1]["ParallelPart"]["out_format"] = ["token", [256, 14**2 * 2]]
    cfg[1]["ParallelPart"]["parts"] = [{"SimpleAdapter": {"in_channels": 512,
                                                          "out_channels": 256,
                                                          "num_fc": 1}}] * 2
    cfg[2] = {"VisionTransformerBlock": {"num_heads": 8,
                                         "in_format": ["token", [256, 14**2 * 2]]}}
    # TODO: Maybe also test this with an AttentionAdapter once that exists?
    input_shape = [3, 14, 14]
    model = Assembly(cfg, input_shape=input_shape)
    out = model(torch.rand(1, *input_shape))  # batch of size 1
    assert out.shape == (1, 14**2 * 2, 256)
