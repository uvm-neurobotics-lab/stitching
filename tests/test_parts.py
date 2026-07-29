"""
Tests for the individual model parts which are addressable from a config: see `assembly.part_class_from_name()`.
"""
import pytest
import torch

from assembly import MLP, VectorHead, part_class_from_name, part_from_config


# The reference feature extractor for MiniGrid observations (three 2x2 convs, no padding, no pooling), expressed
# using this repo's own ConvNet. On a 7x7 input the spatial size shrinks 7 -> 6 -> 5 -> 4.
MINIGRID_TRUNK = {
    "Assembly": {
        "parts": [{"Net": {"model_name": "convnet", "pretrained": False, "x_dim": 3, "num_blocks": 3,
                           "num_filters": [16, 32, 64], "kernel_size": 2, "stride": 1, "padding": 0,
                           "pool_size": None, "norm_type": None, "in_format": "img", "out_format": "img"}}],
        "head": {"VectorHead": {"pooled_size": [4, 4], "activation": "relu"}},
    }
}


def test_new_parts_are_addressable_by_name():
    # Parts are looked up by name from a config, so they must be resolvable without being registered anywhere.
    assert part_class_from_name("VectorHead") is VectorHead
    assert part_class_from_name("MLP") is MLP


def test_convnet_reproduces_the_reference_minigrid_extractor():
    model = part_from_config(MINIGRID_TRUNK, input_shape=(3, 7, 7), num_classes=64)
    features = model(torch.zeros(2, 3, 7, 7))
    assert features.shape == (2, 64)
    # The conv stack itself should end at [64, 4, 4], which makes the head's 4x4 pooling an identity.
    assert model.parts[0](torch.zeros(2, 3, 7, 7)).shape == (2, 64, 4, 4)


def test_vector_head_applies_trailing_activation():
    model = part_from_config(MINIGRID_TRUNK, input_shape=(3, 7, 7), num_classes=64)
    # Use a non-zero input, since an all-zero input would give non-negative features regardless.
    features = model(torch.randn(4, 3, 7, 7))
    assert (features >= 0).all(), "A relu-activated VectorHead should never emit negative features."


def test_vector_head_without_activation_is_unbounded():
    cfg = {"Assembly": {"parts": MINIGRID_TRUNK["Assembly"]["parts"],
                        "head": {"VectorHead": {"pooled_size": [4, 4]}}}}
    model = part_from_config(cfg, input_shape=(3, 7, 7), num_classes=64)
    assert (model(torch.randn(64, 3, 7, 7)) < 0).any(), "Without an activation, features should not be clamped."


def test_vector_head_accepts_num_classes():
    head = VectorHead(num_classes=12, in_format=["img", [8, 4, 4]])
    assert head(torch.zeros(2, 8, 4, 4)).shape == (2, 12)
    head = VectorHead(num_classes=5, in_format=["img", [8, 4, 4]])
    assert head(torch.zeros(2, 8, 4, 4)).shape == (2, 5)


def test_mlp_as_top_level_part():
    # A flat observation, e.g. from MiniGrid's FlatObsWrapper. As the top-level part it is handed input_shape and
    # num_classes directly, so its dimensions need not appear in the config.
    model = part_from_config({"MLP": {"in_format": None, "hidden": [64], "activation": "tanh"}},
                             input_shape=(2835,), num_classes=64)
    assert model(torch.zeros(2, 2835)).shape == (2, 64)


def test_mlp_inside_an_assembly_needs_explicit_dimensions():
    # Assembly does not forward input_shape/num_classes to its parts, so they must be spelled out.
    cfg = {"Assembly": {"parts": [{"MLP": {"in_format": None, "in_features": 30, "out_features": 8}}]}}
    model = part_from_config(cfg, input_shape=(30,), num_classes=8)
    assert model(torch.zeros(2, 30)).shape == (2, 8)


def test_mlp_with_no_hidden_layers_is_a_single_linear():
    model = MLP(in_features=10, out_features=4)
    assert model(torch.zeros(2, 10)).shape == (2, 4)
    assert sum(p.numel() for p in model.parameters()) == 10 * 4 + 4


def test_mlp_flattens_multidimensional_input():
    model = MLP(input_shape=(3, 4, 5), out_features=7)
    assert model(torch.zeros(2, 3, 4, 5)).shape == (2, 7)


def test_mlp_builds_requested_layers():
    model = MLP(in_features=10, out_features=4, hidden=[8, 6], norm="layer", activation="tanh",
                final_activation="tanh")
    kinds = [type(m).__name__ for m in model.mlp]
    assert kinds == ["Flatten", "Linear", "LayerNorm", "Tanh", "Linear", "LayerNorm", "Tanh", "Linear", "Tanh"]


def test_mlp_rejects_unknown_names_even_without_hidden_layers():
    # With no hidden layers the activation is never applied, but a typo should still be reported rather than ignored.
    with pytest.raises(ValueError, match="Unrecognized activation"):
        MLP(in_features=10, out_features=4, activation="bogus")
    with pytest.raises(ValueError, match="Unrecognized normalization"):
        MLP(in_features=10, out_features=4, norm="bogus")


def test_mlp_requires_a_width_and_an_input_size():
    with pytest.raises(RuntimeError, match="out_features"):
        MLP(in_features=10)
    with pytest.raises(RuntimeError, match="in_features"):
        MLP(out_features=4)
