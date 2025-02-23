import operator

import pytest
import timm
import torch
import torchvision

from utils.subgraphs import create_feature_extractor, create_sub_network


def deref_node(a, nodes):
    if isinstance(a, str) and a[0] == "%":
        a = a[1:]
        for n in nodes:
            if n.name == a:
                return n
        raise RuntimeError(f"Unable to find node named '{a}'.")
    else:
        return a


def deref_nodes(args, nodes):
    newargs = []
    for a in args:
        if isinstance(a, dict):
            # Special case: deref the values but not the keys, b/c this is how the args to the output nodes are
            # typically formatted.
            newargs.append({k: deref_node(v, nodes) for k, v in a.items()})
        else:
            newargs.append(deref_node(a, nodes))
    return tuple(newargs)


def leaf_function(x):
    # This would raise a TypeError if traced through
    return int(x)


class LeafModule(torch.nn.Module):

    # noinspection PyMethodMayBeStatic
    def forward(self, x, y, z):
        # This would raise a TypeError if traced through
        int(x.shape[0])
        return torch.nn.functional.relu(x + y + z + 4)


class MyModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 1, 3)
        self.leaf_module = LeafModule()

    def forward(self, x):
        y = leaf_function(x.shape[0])
        z = x
        x = self.conv(x)
        x = torch.relu(x + z*4)
        return self.leaf_module(x, y, z)


def test_tracer_leaf_function_causes_error():
    """We expect failing to specify leaf modules will cause a TypeError."""
    with pytest.raises(TypeError):
        create_feature_extractor(MyModule(), return_nodes=["leaf_module"])
    with pytest.raises(TypeError):
        create_feature_extractor(MyModule(), return_nodes=["leaf_module"],
                                 tracer_kwargs={"leaf_modules": [LeafModule]})
    with pytest.raises(TypeError):
        create_feature_extractor(MyModule(), return_nodes=["leaf_module"],
                                 tracer_kwargs={"autowrap_functions": [leaf_function]})


def test_simple_tracing():
    model = create_feature_extractor(MyModule(), return_nodes=["leaf_module"],
                                     tracer_kwargs={"leaf_modules": [LeafModule],
                                                    "autowrap_functions": [leaf_function]})
    expected_nodes = [
        {"name": "x", "op": "placeholder", "target": "x", "args": tuple(), "kwargs": {}},
        {"name": "getattr_1", "op": "call_function", "target": getattr, "args": ("%x", "shape"), "kwargs": {}},
        {"name": "getitem", "op": "call_function", "target": operator.getitem, "args": ("%getattr_1", 0), "kwargs": {}},
        {"name": "leaf_function", "op": "call_function", "target": leaf_function, "args": ("%getitem",), "kwargs": {}},
        {"name": "conv", "op": "call_module", "target": "conv", "args": ("%x",), "kwargs": {}},
        {"name": "mul", "op": "call_function", "target": operator.mul, "args": ("%x", 4), "kwargs": {}},
        {"name": "add", "op": "call_function", "target": operator.add, "args": ("%conv", "%mul"), "kwargs": {}},
        {"name": "relu", "op": "call_function", "target": torch.relu, "args": ("%add",), "kwargs": {}},
        {"name": "leaf_module", "op": "call_module", "target": "leaf_module", "args": ("%relu", "%leaf_function", "%x"), "kwargs": {}},
        {"name": "output_1", "op": "output", "target": "output", "args": ({"leaf_module": "%leaf_module"},), "kwargs": {}},
    ]
    nodes = list(model.graph.nodes)
    for expected, actual in zip(expected_nodes, nodes):
        for name, val in expected.items():
            if name == "args":
                val = deref_nodes(val, nodes)
            assert getattr(actual, name) == val, f"'{name}' are not equal for node {actual.name}"


def test_resnet18_tracing():
    model = torchvision.models.resnet18()
    model = create_feature_extractor(model, return_nodes={'layer1': 'feat1', 'layer3': 'feat2'})
    # Lazy shortcut: instead of checking all this in code, just use the string format.
    # Downside: if the print format ever changed, this would fail unnecessarily.
    assert str(model.graph) == """graph():
    %x : torch.Tensor [num_users=1] = placeholder[target=x]
    %conv1 : [num_users=1] = call_module[target=conv1](args = (%x,), kwargs = {})
    %bn1 : [num_users=1] = call_module[target=bn1](args = (%conv1,), kwargs = {})
    %relu : [num_users=1] = call_module[target=relu](args = (%bn1,), kwargs = {})
    %maxpool : [num_users=2] = call_module[target=maxpool](args = (%relu,), kwargs = {})
    %layer1_0_conv1 : [num_users=1] = call_module[target=layer1.0.conv1](args = (%maxpool,), kwargs = {})
    %layer1_0_bn1 : [num_users=1] = call_module[target=layer1.0.bn1](args = (%layer1_0_conv1,), kwargs = {})
    %layer1_0_relu : [num_users=1] = call_module[target=layer1.0.relu](args = (%layer1_0_bn1,), kwargs = {})
    %layer1_0_conv2 : [num_users=1] = call_module[target=layer1.0.conv2](args = (%layer1_0_relu,), kwargs = {})
    %layer1_0_bn2 : [num_users=1] = call_module[target=layer1.0.bn2](args = (%layer1_0_conv2,), kwargs = {})
    %add : [num_users=1] = call_function[target=operator.add](args = (%layer1_0_bn2, %maxpool), kwargs = {})
    %layer1_0_relu_1 : [num_users=2] = call_module[target=layer1.0.relu](args = (%add,), kwargs = {})
    %layer1_1_conv1 : [num_users=1] = call_module[target=layer1.1.conv1](args = (%layer1_0_relu_1,), kwargs = {})
    %layer1_1_bn1 : [num_users=1] = call_module[target=layer1.1.bn1](args = (%layer1_1_conv1,), kwargs = {})
    %layer1_1_relu : [num_users=1] = call_module[target=layer1.1.relu](args = (%layer1_1_bn1,), kwargs = {})
    %layer1_1_conv2 : [num_users=1] = call_module[target=layer1.1.conv2](args = (%layer1_1_relu,), kwargs = {})
    %layer1_1_bn2 : [num_users=1] = call_module[target=layer1.1.bn2](args = (%layer1_1_conv2,), kwargs = {})
    %add_1 : [num_users=1] = call_function[target=operator.add](args = (%layer1_1_bn2, %layer1_0_relu_1), kwargs = {})
    %layer1_1_relu_1 : [num_users=3] = call_module[target=layer1.1.relu](args = (%add_1,), kwargs = {})
    %layer2_0_conv1 : [num_users=1] = call_module[target=layer2.0.conv1](args = (%layer1_1_relu_1,), kwargs = {})
    %layer2_0_bn1 : [num_users=1] = call_module[target=layer2.0.bn1](args = (%layer2_0_conv1,), kwargs = {})
    %layer2_0_relu : [num_users=1] = call_module[target=layer2.0.relu](args = (%layer2_0_bn1,), kwargs = {})
    %layer2_0_conv2 : [num_users=1] = call_module[target=layer2.0.conv2](args = (%layer2_0_relu,), kwargs = {})
    %layer2_0_bn2 : [num_users=1] = call_module[target=layer2.0.bn2](args = (%layer2_0_conv2,), kwargs = {})
    %layer2_0_downsample_0 : [num_users=1] = call_module[target=layer2.0.downsample.0](args = (%layer1_1_relu_1,), kwargs = {})
    %layer2_0_downsample_1 : [num_users=1] = call_module[target=layer2.0.downsample.1](args = (%layer2_0_downsample_0,), kwargs = {})
    %add_2 : [num_users=1] = call_function[target=operator.add](args = (%layer2_0_bn2, %layer2_0_downsample_1), kwargs = {})
    %layer2_0_relu_1 : [num_users=2] = call_module[target=layer2.0.relu](args = (%add_2,), kwargs = {})
    %layer2_1_conv1 : [num_users=1] = call_module[target=layer2.1.conv1](args = (%layer2_0_relu_1,), kwargs = {})
    %layer2_1_bn1 : [num_users=1] = call_module[target=layer2.1.bn1](args = (%layer2_1_conv1,), kwargs = {})
    %layer2_1_relu : [num_users=1] = call_module[target=layer2.1.relu](args = (%layer2_1_bn1,), kwargs = {})
    %layer2_1_conv2 : [num_users=1] = call_module[target=layer2.1.conv2](args = (%layer2_1_relu,), kwargs = {})
    %layer2_1_bn2 : [num_users=1] = call_module[target=layer2.1.bn2](args = (%layer2_1_conv2,), kwargs = {})
    %add_3 : [num_users=1] = call_function[target=operator.add](args = (%layer2_1_bn2, %layer2_0_relu_1), kwargs = {})
    %layer2_1_relu_1 : [num_users=2] = call_module[target=layer2.1.relu](args = (%add_3,), kwargs = {})
    %layer3_0_conv1 : [num_users=1] = call_module[target=layer3.0.conv1](args = (%layer2_1_relu_1,), kwargs = {})
    %layer3_0_bn1 : [num_users=1] = call_module[target=layer3.0.bn1](args = (%layer3_0_conv1,), kwargs = {})
    %layer3_0_relu : [num_users=1] = call_module[target=layer3.0.relu](args = (%layer3_0_bn1,), kwargs = {})
    %layer3_0_conv2 : [num_users=1] = call_module[target=layer3.0.conv2](args = (%layer3_0_relu,), kwargs = {})
    %layer3_0_bn2 : [num_users=1] = call_module[target=layer3.0.bn2](args = (%layer3_0_conv2,), kwargs = {})
    %layer3_0_downsample_0 : [num_users=1] = call_module[target=layer3.0.downsample.0](args = (%layer2_1_relu_1,), kwargs = {})
    %layer3_0_downsample_1 : [num_users=1] = call_module[target=layer3.0.downsample.1](args = (%layer3_0_downsample_0,), kwargs = {})
    %add_4 : [num_users=1] = call_function[target=operator.add](args = (%layer3_0_bn2, %layer3_0_downsample_1), kwargs = {})
    %layer3_0_relu_1 : [num_users=2] = call_module[target=layer3.0.relu](args = (%add_4,), kwargs = {})
    %layer3_1_conv1 : [num_users=1] = call_module[target=layer3.1.conv1](args = (%layer3_0_relu_1,), kwargs = {})
    %layer3_1_bn1 : [num_users=1] = call_module[target=layer3.1.bn1](args = (%layer3_1_conv1,), kwargs = {})
    %layer3_1_relu : [num_users=1] = call_module[target=layer3.1.relu](args = (%layer3_1_bn1,), kwargs = {})
    %layer3_1_conv2 : [num_users=1] = call_module[target=layer3.1.conv2](args = (%layer3_1_relu,), kwargs = {})
    %layer3_1_bn2 : [num_users=1] = call_module[target=layer3.1.bn2](args = (%layer3_1_conv2,), kwargs = {})
    %add_5 : [num_users=1] = call_function[target=operator.add](args = (%layer3_1_bn2, %layer3_0_relu_1), kwargs = {})
    %layer3_1_relu_1 : [num_users=1] = call_module[target=layer3.1.relu](args = (%add_5,), kwargs = {})
    return {'feat1': layer1_1_relu_1, 'feat2': layer3_1_relu_1}"""


def test_resnet18_feature_extraction():
    model = torchvision.models.resnet18()
    model = create_feature_extractor(model, return_nodes={'layer1': 'feat1', 'layer3': 'feat2'})
    out = model(torch.rand(1, 3, 224, 224))
    assert list(out.keys()) == ["feat1", "feat2"]
    assert out["feat1"].shape == torch.Size(torch.Size([1, 64, 56, 56]))
    assert out["feat2"].shape == torch.Size(torch.Size([1, 256, 14, 14]))


def test_simple_subnet():
    model = create_sub_network(MyModule(), input_nodes=["conv"], return_nodes=["relu"],
                               tracer_kwargs={"leaf_modules": [LeafModule], "autowrap_functions": [leaf_function]})
    expected_nodes = [
        {"name": "subnet_input_1", "op": "placeholder", "target": "subnet_input_1", "args": tuple(), "kwargs": {}},
        {"name": "conv", "op": "call_module", "target": "conv", "args": ("%subnet_input_1",), "kwargs": {}},
        {"name": "mul", "op": "call_function", "target": operator.mul, "args": ("%subnet_input_1", 4), "kwargs": {}},
        {"name": "add", "op": "call_function", "target": operator.add, "args": ("%conv", "%mul"), "kwargs": {}},
        {"name": "relu", "op": "call_function", "target": torch.relu, "args": ("%add",), "kwargs": {}},
        {"name": "output_1", "op": "output", "target": "output", "args": ({"relu": "%relu"},), "kwargs": {}},
    ]
    nodes = list(model.graph.nodes)
    for expected, actual in zip(expected_nodes, nodes):
        for name, val in expected.items():
            if name == "args":
                val = deref_nodes(val, nodes)
            assert getattr(actual, name) == val, f"'{name}' are not equal for node {actual.name}"


def test_subnet_with_two_cuts_fails():
    with pytest.raises(RuntimeError):
        create_sub_network(MyModule(), input_nodes=["conv"], return_nodes=["leaf_module"],
                           tracer_kwargs={"leaf_modules": [LeafModule], "autowrap_functions": [leaf_function]})


def test_resnet18_subnet():
    model = torchvision.models.resnet18()
    model = create_sub_network(model, input_nodes=["layer2.0"], return_nodes=["layer3.0"],)
    # Lazy shortcut: instead of checking all this in code, just use the string format.
    # Downside: if the print format ever changed, this would fail unnecessarily.
    assert str(model.graph) == """graph():
    %subnet_input_1 : [num_users=2] = placeholder[target=subnet_input_1]
    %layer2_0_conv1 : [num_users=1] = call_module[target=layer2.0.conv1](args = (%subnet_input_1,), kwargs = {})
    %layer2_0_bn1 : [num_users=1] = call_module[target=layer2.0.bn1](args = (%layer2_0_conv1,), kwargs = {})
    %layer2_0_relu : [num_users=1] = call_module[target=layer2.0.relu](args = (%layer2_0_bn1,), kwargs = {})
    %layer2_0_conv2 : [num_users=1] = call_module[target=layer2.0.conv2](args = (%layer2_0_relu,), kwargs = {})
    %layer2_0_bn2 : [num_users=1] = call_module[target=layer2.0.bn2](args = (%layer2_0_conv2,), kwargs = {})
    %layer2_0_downsample_0 : [num_users=1] = call_module[target=layer2.0.downsample.0](args = (%subnet_input_1,), kwargs = {})
    %layer2_0_downsample_1 : [num_users=1] = call_module[target=layer2.0.downsample.1](args = (%layer2_0_downsample_0,), kwargs = {})
    %add_2 : [num_users=1] = call_function[target=operator.add](args = (%layer2_0_bn2, %layer2_0_downsample_1), kwargs = {})
    %layer2_0_relu_1 : [num_users=2] = call_module[target=layer2.0.relu](args = (%add_2,), kwargs = {})
    %layer2_1_conv1 : [num_users=1] = call_module[target=layer2.1.conv1](args = (%layer2_0_relu_1,), kwargs = {})
    %layer2_1_bn1 : [num_users=1] = call_module[target=layer2.1.bn1](args = (%layer2_1_conv1,), kwargs = {})
    %layer2_1_relu : [num_users=1] = call_module[target=layer2.1.relu](args = (%layer2_1_bn1,), kwargs = {})
    %layer2_1_conv2 : [num_users=1] = call_module[target=layer2.1.conv2](args = (%layer2_1_relu,), kwargs = {})
    %layer2_1_bn2 : [num_users=1] = call_module[target=layer2.1.bn2](args = (%layer2_1_conv2,), kwargs = {})
    %add_3 : [num_users=1] = call_function[target=operator.add](args = (%layer2_1_bn2, %layer2_0_relu_1), kwargs = {})
    %layer2_1_relu_1 : [num_users=2] = call_module[target=layer2.1.relu](args = (%add_3,), kwargs = {})
    %layer3_0_conv1 : [num_users=1] = call_module[target=layer3.0.conv1](args = (%layer2_1_relu_1,), kwargs = {})
    %layer3_0_bn1 : [num_users=1] = call_module[target=layer3.0.bn1](args = (%layer3_0_conv1,), kwargs = {})
    %layer3_0_relu : [num_users=1] = call_module[target=layer3.0.relu](args = (%layer3_0_bn1,), kwargs = {})
    %layer3_0_conv2 : [num_users=1] = call_module[target=layer3.0.conv2](args = (%layer3_0_relu,), kwargs = {})
    %layer3_0_bn2 : [num_users=1] = call_module[target=layer3.0.bn2](args = (%layer3_0_conv2,), kwargs = {})
    %layer3_0_downsample_0 : [num_users=1] = call_module[target=layer3.0.downsample.0](args = (%layer2_1_relu_1,), kwargs = {})
    %layer3_0_downsample_1 : [num_users=1] = call_module[target=layer3.0.downsample.1](args = (%layer3_0_downsample_0,), kwargs = {})
    %add_4 : [num_users=1] = call_function[target=operator.add](args = (%layer3_0_bn2, %layer3_0_downsample_1), kwargs = {})
    %layer3_0_relu_1 : [num_users=1] = call_module[target=layer3.0.relu](args = (%add_4,), kwargs = {})
    return {'layer3.0': layer3_0_relu_1}"""


def test_resnet18_one_block_subnet():
    model = torchvision.models.resnet18()
    model = create_sub_network(model, input_nodes=["layer2.0"], return_nodes=["layer2.0"],)
    # Lazy shortcut: instead of checking all this in code, just use the string format.
    # Downside: if the print format ever changed, this would fail unnecessarily.
    assert str(model.graph) == """graph():
    %subnet_input_1 : [num_users=2] = placeholder[target=subnet_input_1]
    %layer2_0_conv1 : [num_users=1] = call_module[target=layer2.0.conv1](args = (%subnet_input_1,), kwargs = {})
    %layer2_0_bn1 : [num_users=1] = call_module[target=layer2.0.bn1](args = (%layer2_0_conv1,), kwargs = {})
    %layer2_0_relu : [num_users=1] = call_module[target=layer2.0.relu](args = (%layer2_0_bn1,), kwargs = {})
    %layer2_0_conv2 : [num_users=1] = call_module[target=layer2.0.conv2](args = (%layer2_0_relu,), kwargs = {})
    %layer2_0_bn2 : [num_users=1] = call_module[target=layer2.0.bn2](args = (%layer2_0_conv2,), kwargs = {})
    %layer2_0_downsample_0 : [num_users=1] = call_module[target=layer2.0.downsample.0](args = (%subnet_input_1,), kwargs = {})
    %layer2_0_downsample_1 : [num_users=1] = call_module[target=layer2.0.downsample.1](args = (%layer2_0_downsample_0,), kwargs = {})
    %add_2 : [num_users=1] = call_function[target=operator.add](args = (%layer2_0_bn2, %layer2_0_downsample_1), kwargs = {})
    %layer2_0_relu_1 : [num_users=1] = call_module[target=layer2.0.relu](args = (%add_2,), kwargs = {})
    return {'layer2.0': layer2_0_relu_1}"""


def test_resnet18_sub_block_subnet():
    model = torchvision.models.resnet18()
    model = create_sub_network(model, input_nodes=["layer2.0.bn1"], return_nodes=["layer2.0.conv2"],)
    # Lazy shortcut: instead of checking all this in code, just use the string format.
    # Downside: if the print format ever changed, this would fail unnecessarily.
    assert str(model.graph) == """graph():
    %subnet_input_1 : [num_users=1] = placeholder[target=subnet_input_1]
    %layer2_0_bn1 : [num_users=1] = call_module[target=layer2.0.bn1](args = (%subnet_input_1,), kwargs = {})
    %layer2_0_relu : [num_users=1] = call_module[target=layer2.0.relu](args = (%layer2_0_bn1,), kwargs = {})
    %layer2_0_conv2 : [num_users=1] = call_module[target=layer2.0.conv2](args = (%layer2_0_relu,), kwargs = {})
    return {'layer2.0.conv2': layer2_0_conv2}"""


def test_resnet18_subnet_cutting_across_skip_connection_fails():
    with pytest.raises(RuntimeError):
        create_sub_network(torchvision.models.resnet18(), input_nodes=["layer2.0.bn1"],
                           return_nodes=["layer2.0.relu_1"])


def test_resnet18_subnet_cutting_across_skip_connection_with_two_inputs():
    model = torchvision.models.resnet18()
    model = create_sub_network(model, input_nodes=["layer2.0.bn1", "layer2.0.downsample.1"],
                               return_nodes=["layer2.0.relu_1"])
    # Lazy shortcut: instead of checking all this in code, just use the string format.
    # Downside: if the print format ever changed, this would fail unnecessarily.
    assert str(model.graph) == """graph():
    %subnet_input_1 : [num_users=1] = placeholder[target=subnet_input_1]
    %layer2_0_bn1 : [num_users=1] = call_module[target=layer2.0.bn1](args = (%subnet_input_1,), kwargs = {})
    %layer2_0_relu : [num_users=1] = call_module[target=layer2.0.relu](args = (%layer2_0_bn1,), kwargs = {})
    %layer2_0_conv2 : [num_users=1] = call_module[target=layer2.0.conv2](args = (%layer2_0_relu,), kwargs = {})
    %layer2_0_bn2 : [num_users=1] = call_module[target=layer2.0.bn2](args = (%layer2_0_conv2,), kwargs = {})
    %subnet_input_2 : [num_users=1] = placeholder[target=subnet_input_2]
    %layer2_0_downsample_1 : [num_users=1] = call_module[target=layer2.0.downsample.1](args = (%subnet_input_2,), kwargs = {})
    %add_2 : [num_users=1] = call_function[target=operator.add](args = (%layer2_0_bn2, %layer2_0_downsample_1), kwargs = {})
    %layer2_0_relu_1 : [num_users=1] = call_module[target=layer2.0.relu](args = (%add_2,), kwargs = {})
    return {'layer2.0.relu_1': layer2_0_relu_1}"""


def test_resnet18_subnet_input_node_with_two_inputs_fails():
    with pytest.raises(RuntimeError):
        create_sub_network(torchvision.models.resnet18(), input_nodes=["layer2.0.add", "layer2.0.downsample.1"],
                           return_nodes=["layer2.0.relu_1"])
    # If the function supported this concept, this is what the subnet would look like. But not currently supported.
    # assert str(model.graph) == """graph():
    # %subnet_input_2 : [num_users=1] = placeholder[target=subnet_input_2]
    # %layer2_0_downsample_1 : [num_users=1] = call_module[target=layer2.0.downsample.1](args = (%subnet_input_2,), kwargs = {})
    # %subnet_input_1 : [num_users=1] = placeholder[target=subnet_input_1]
    # %add_2 : [num_users=1] = call_function[target=operator.add](args = (%subnet_input_1, %layer2_0_downsample_1), kwargs = {})
    # %layer2_0_relu_1 : [num_users=1] = call_module[target=layer2.0.relu](args = (%add_2,), kwargs = {})
    # return {'layer2.0.relu_1': layer2_0_relu_1}"""


def test_vit_two_block_subnet():
    model = timm.create_model("vit_small_patch16_224")
    model = create_sub_network(model, input_nodes=["blocks.2"], return_nodes=["blocks.3"])
    # Lazy shortcut: instead of checking all this in code, just use the string format.
    # Downside: if the print format ever changed, this would fail unnecessarily.
    assert str(model.graph) == """graph():
    %subnet_input_1 : [num_users=2] = placeholder[target=subnet_input_1]
    %blocks_2_norm1 : [num_users=2] = call_module[target=blocks.2.norm1](args = (%subnet_input_1,), kwargs = {})
    %getattr_5 : [num_users=3] = call_function[target=builtins.getattr](args = (%blocks_2_norm1, shape), kwargs = {})
    %getitem_17 : [num_users=2] = call_function[target=operator.getitem](args = (%getattr_5, 0), kwargs = {})
    %getitem_18 : [num_users=2] = call_function[target=operator.getitem](args = (%getattr_5, 1), kwargs = {})
    %getitem_19 : [num_users=1] = call_function[target=operator.getitem](args = (%getattr_5, 2), kwargs = {})
    %blocks_2_attn_qkv : [num_users=1] = call_module[target=blocks.2.attn.qkv](args = (%blocks_2_norm1,), kwargs = {})
    %reshape_4 : [num_users=1] = call_method[target=reshape](args = (%blocks_2_attn_qkv, %getitem_17, %getitem_18, 3, 6, 64), kwargs = {})
    %permute_2 : [num_users=1] = call_method[target=permute](args = (%reshape_4, 2, 0, 3, 1, 4), kwargs = {})
    %unbind_2 : [num_users=3] = call_method[target=unbind](args = (%permute_2, 0), kwargs = {})
    %getitem_20 : [num_users=1] = call_function[target=operator.getitem](args = (%unbind_2, 0), kwargs = {})
    %getitem_21 : [num_users=1] = call_function[target=operator.getitem](args = (%unbind_2, 1), kwargs = {})
    %getitem_22 : [num_users=1] = call_function[target=operator.getitem](args = (%unbind_2, 2), kwargs = {})
    %blocks_2_attn_q_norm : [num_users=1] = call_module[target=blocks.2.attn.q_norm](args = (%getitem_20,), kwargs = {})
    %blocks_2_attn_k_norm : [num_users=1] = call_module[target=blocks.2.attn.k_norm](args = (%getitem_21,), kwargs = {})
    %scaled_dot_product_attention_2 : [num_users=1] = call_function[target=torch._C._nn.scaled_dot_product_attention](args = (%blocks_2_attn_q_norm, %blocks_2_attn_k_norm, %getitem_22), kwargs = {dropout_p: 0.0})
    %transpose_3 : [num_users=1] = call_method[target=transpose](args = (%scaled_dot_product_attention_2, 1, 2), kwargs = {})
    %reshape_5 : [num_users=1] = call_method[target=reshape](args = (%transpose_3, %getitem_17, %getitem_18, %getitem_19), kwargs = {})
    %blocks_2_attn_proj : [num_users=1] = call_module[target=blocks.2.attn.proj](args = (%reshape_5,), kwargs = {})
    %blocks_2_attn_proj_drop : [num_users=1] = call_module[target=blocks.2.attn.proj_drop](args = (%blocks_2_attn_proj,), kwargs = {})
    %blocks_2_ls1 : [num_users=1] = call_module[target=blocks.2.ls1](args = (%blocks_2_attn_proj_drop,), kwargs = {})
    %blocks_2_drop_path1 : [num_users=1] = call_module[target=blocks.2.drop_path1](args = (%blocks_2_ls1,), kwargs = {})
    %add_5 : [num_users=2] = call_function[target=operator.add](args = (%subnet_input_1, %blocks_2_drop_path1), kwargs = {})
    %blocks_2_norm2 : [num_users=1] = call_module[target=blocks.2.norm2](args = (%add_5,), kwargs = {})
    %blocks_2_mlp_fc1 : [num_users=1] = call_module[target=blocks.2.mlp.fc1](args = (%blocks_2_norm2,), kwargs = {})
    %blocks_2_mlp_act : [num_users=1] = call_module[target=blocks.2.mlp.act](args = (%blocks_2_mlp_fc1,), kwargs = {})
    %blocks_2_mlp_drop1 : [num_users=1] = call_module[target=blocks.2.mlp.drop1](args = (%blocks_2_mlp_act,), kwargs = {})
    %blocks_2_mlp_norm : [num_users=1] = call_module[target=blocks.2.mlp.norm](args = (%blocks_2_mlp_drop1,), kwargs = {})
    %blocks_2_mlp_fc2 : [num_users=1] = call_module[target=blocks.2.mlp.fc2](args = (%blocks_2_mlp_norm,), kwargs = {})
    %blocks_2_mlp_drop2 : [num_users=1] = call_module[target=blocks.2.mlp.drop2](args = (%blocks_2_mlp_fc2,), kwargs = {})
    %blocks_2_ls2 : [num_users=1] = call_module[target=blocks.2.ls2](args = (%blocks_2_mlp_drop2,), kwargs = {})
    %blocks_2_drop_path2 : [num_users=1] = call_module[target=blocks.2.drop_path2](args = (%blocks_2_ls2,), kwargs = {})
    %add_6 : [num_users=2] = call_function[target=operator.add](args = (%add_5, %blocks_2_drop_path2), kwargs = {})
    %blocks_3_norm1 : [num_users=2] = call_module[target=blocks.3.norm1](args = (%add_6,), kwargs = {})
    %getattr_6 : [num_users=3] = call_function[target=builtins.getattr](args = (%blocks_3_norm1, shape), kwargs = {})
    %getitem_23 : [num_users=2] = call_function[target=operator.getitem](args = (%getattr_6, 0), kwargs = {})
    %getitem_24 : [num_users=2] = call_function[target=operator.getitem](args = (%getattr_6, 1), kwargs = {})
    %getitem_25 : [num_users=1] = call_function[target=operator.getitem](args = (%getattr_6, 2), kwargs = {})
    %blocks_3_attn_qkv : [num_users=1] = call_module[target=blocks.3.attn.qkv](args = (%blocks_3_norm1,), kwargs = {})
    %reshape_6 : [num_users=1] = call_method[target=reshape](args = (%blocks_3_attn_qkv, %getitem_23, %getitem_24, 3, 6, 64), kwargs = {})
    %permute_3 : [num_users=1] = call_method[target=permute](args = (%reshape_6, 2, 0, 3, 1, 4), kwargs = {})
    %unbind_3 : [num_users=3] = call_method[target=unbind](args = (%permute_3, 0), kwargs = {})
    %getitem_26 : [num_users=1] = call_function[target=operator.getitem](args = (%unbind_3, 0), kwargs = {})
    %getitem_27 : [num_users=1] = call_function[target=operator.getitem](args = (%unbind_3, 1), kwargs = {})
    %getitem_28 : [num_users=1] = call_function[target=operator.getitem](args = (%unbind_3, 2), kwargs = {})
    %blocks_3_attn_q_norm : [num_users=1] = call_module[target=blocks.3.attn.q_norm](args = (%getitem_26,), kwargs = {})
    %blocks_3_attn_k_norm : [num_users=1] = call_module[target=blocks.3.attn.k_norm](args = (%getitem_27,), kwargs = {})
    %scaled_dot_product_attention_3 : [num_users=1] = call_function[target=torch._C._nn.scaled_dot_product_attention](args = (%blocks_3_attn_q_norm, %blocks_3_attn_k_norm, %getitem_28), kwargs = {dropout_p: 0.0})
    %transpose_4 : [num_users=1] = call_method[target=transpose](args = (%scaled_dot_product_attention_3, 1, 2), kwargs = {})
    %reshape_7 : [num_users=1] = call_method[target=reshape](args = (%transpose_4, %getitem_23, %getitem_24, %getitem_25), kwargs = {})
    %blocks_3_attn_proj : [num_users=1] = call_module[target=blocks.3.attn.proj](args = (%reshape_7,), kwargs = {})
    %blocks_3_attn_proj_drop : [num_users=1] = call_module[target=blocks.3.attn.proj_drop](args = (%blocks_3_attn_proj,), kwargs = {})
    %blocks_3_ls1 : [num_users=1] = call_module[target=blocks.3.ls1](args = (%blocks_3_attn_proj_drop,), kwargs = {})
    %blocks_3_drop_path1 : [num_users=1] = call_module[target=blocks.3.drop_path1](args = (%blocks_3_ls1,), kwargs = {})
    %add_7 : [num_users=2] = call_function[target=operator.add](args = (%add_6, %blocks_3_drop_path1), kwargs = {})
    %blocks_3_norm2 : [num_users=1] = call_module[target=blocks.3.norm2](args = (%add_7,), kwargs = {})
    %blocks_3_mlp_fc1 : [num_users=1] = call_module[target=blocks.3.mlp.fc1](args = (%blocks_3_norm2,), kwargs = {})
    %blocks_3_mlp_act : [num_users=1] = call_module[target=blocks.3.mlp.act](args = (%blocks_3_mlp_fc1,), kwargs = {})
    %blocks_3_mlp_drop1 : [num_users=1] = call_module[target=blocks.3.mlp.drop1](args = (%blocks_3_mlp_act,), kwargs = {})
    %blocks_3_mlp_norm : [num_users=1] = call_module[target=blocks.3.mlp.norm](args = (%blocks_3_mlp_drop1,), kwargs = {})
    %blocks_3_mlp_fc2 : [num_users=1] = call_module[target=blocks.3.mlp.fc2](args = (%blocks_3_mlp_norm,), kwargs = {})
    %blocks_3_mlp_drop2 : [num_users=1] = call_module[target=blocks.3.mlp.drop2](args = (%blocks_3_mlp_fc2,), kwargs = {})
    %blocks_3_ls2 : [num_users=1] = call_module[target=blocks.3.ls2](args = (%blocks_3_mlp_drop2,), kwargs = {})
    %blocks_3_drop_path2 : [num_users=1] = call_module[target=blocks.3.drop_path2](args = (%blocks_3_ls2,), kwargs = {})
    %add_8 : [num_users=1] = call_function[target=operator.add](args = (%add_7, %blocks_3_drop_path2), kwargs = {})
    return {'blocks.3': add_8}"""


def test_swin_beginning_chunk_subnet():
    model = torchvision.models.swin_v2_s()
    model = create_sub_network(model, input_nodes=["features.0.0"], return_nodes=["features.1.0"])
    # Lazy shortcut: instead of checking all this in code, just use the string format.
    # Downside: if the print format ever changed, this would fail unnecessarily.
    assert str(model.graph) == """graph():
    %subnet_input_1 : [num_users=1] = placeholder[target=subnet_input_1]
    %features_0_0 : [num_users=1] = call_module[target=features.0.0](args = (%subnet_input_1,), kwargs = {})
    %features_0_1 : [num_users=1] = call_module[target=features.0.1](args = (%features_0_0,), kwargs = {})
    %features_0_2 : [num_users=2] = call_module[target=features.0.2](args = (%features_0_1,), kwargs = {})
    %features_1_0_attn_relative_coords_table : [num_users=1] = get_attr[target=features.1.0.attn.relative_coords_table]
    %features_1_0_attn_cpb_mlp_0 : [num_users=1] = call_module[target=features.1.0.attn.cpb_mlp.0](args = (%features_1_0_attn_relative_coords_table,), kwargs = {})
    %features_1_0_attn_cpb_mlp_1 : [num_users=1] = call_module[target=features.1.0.attn.cpb_mlp.1](args = (%features_1_0_attn_cpb_mlp_0,), kwargs = {})
    %features_1_0_attn_cpb_mlp_2 : [num_users=1] = call_module[target=features.1.0.attn.cpb_mlp.2](args = (%features_1_0_attn_cpb_mlp_1,), kwargs = {})
    %view : [num_users=1] = call_method[target=view](args = (%features_1_0_attn_cpb_mlp_2, -1, 3), kwargs = {})
    %features_1_0_attn_relative_position_index : [num_users=1] = get_attr[target=features.1.0.attn.relative_position_index]
    %_get_relative_position_bias : [num_users=1] = call_function[target=torchvision.models.swin_transformer._get_relative_position_bias](args = (%view, %features_1_0_attn_relative_position_index, [8, 8]), kwargs = {})
    %sigmoid : [num_users=1] = call_function[target=torch.sigmoid](args = (%_get_relative_position_bias,), kwargs = {})
    %mul : [num_users=1] = call_function[target=operator.mul](args = (16, %sigmoid), kwargs = {})
    %features_1_0_attn_qkv_weight : [num_users=1] = get_attr[target=features.1.0.attn.qkv.weight]
    %features_1_0_attn_proj_weight : [num_users=1] = get_attr[target=features.1.0.attn.proj.weight]
    %features_1_0_attn_qkv_bias : [num_users=1] = get_attr[target=features.1.0.attn.qkv.bias]
    %features_1_0_attn_proj_bias : [num_users=1] = get_attr[target=features.1.0.attn.proj.bias]
    %features_1_0_attn_logit_scale : [num_users=1] = get_attr[target=features.1.0.attn.logit_scale]
    %shifted_window_attention : [num_users=1] = call_function[target=torchvision.models.swin_transformer.shifted_window_attention](args = (%features_0_2, %features_1_0_attn_qkv_weight, %features_1_0_attn_proj_weight, %mul, [8, 8], 3), kwargs = {shift_size: [0, 0], attention_dropout: 0.0, dropout: 0.0, qkv_bias: %features_1_0_attn_qkv_bias, proj_bias: %features_1_0_attn_proj_bias, logit_scale: %features_1_0_attn_logit_scale, training: True})
    %features_1_0_norm1 : [num_users=1] = call_module[target=features.1.0.norm1](args = (%shifted_window_attention,), kwargs = {})
    %features_1_0_stochastic_depth : [num_users=1] = call_module[target=features.1.0.stochastic_depth](args = (%features_1_0_norm1,), kwargs = {})
    %add : [num_users=2] = call_function[target=operator.add](args = (%features_0_2, %features_1_0_stochastic_depth), kwargs = {})
    %features_1_0_mlp : [num_users=1] = call_module[target=features.1.0.mlp](args = (%add,), kwargs = {})
    %features_1_0_norm2 : [num_users=1] = call_module[target=features.1.0.norm2](args = (%features_1_0_mlp,), kwargs = {})
    %features_1_0_stochastic_depth_1 : [num_users=1] = call_module[target=features.1.0.stochastic_depth](args = (%features_1_0_norm2,), kwargs = {})
    %add_1 : [num_users=1] = call_function[target=operator.add](args = (%add, %features_1_0_stochastic_depth_1), kwargs = {})
    return {'features.1.0': add_1}"""


def test_swin_end_chunk_subnet():
    model = torchvision.models.swin_v2_s()
    model = create_sub_network(model, input_nodes=["features.6.norm"], return_nodes=["head"])
    # Lazy shortcut: instead of checking all this in code, just use the string format.
    # Downside: if the print format ever changed, this would fail unnecessarily.
    assert str(model.graph) == """graph():
    %subnet_input_1 : [num_users=1] = placeholder[target=subnet_input_1]
    %features_6_norm : [num_users=2] = call_module[target=features.6.norm](args = (%subnet_input_1,), kwargs = {})
    %features_7_0_attn_relative_coords_table : [num_users=1] = get_attr[target=features.7.0.attn.relative_coords_table]
    %features_7_0_attn_cpb_mlp_0 : [num_users=1] = call_module[target=features.7.0.attn.cpb_mlp.0](args = (%features_7_0_attn_relative_coords_table,), kwargs = {})
    %features_7_0_attn_cpb_mlp_1 : [num_users=1] = call_module[target=features.7.0.attn.cpb_mlp.1](args = (%features_7_0_attn_cpb_mlp_0,), kwargs = {})
    %features_7_0_attn_cpb_mlp_2 : [num_users=1] = call_module[target=features.7.0.attn.cpb_mlp.2](args = (%features_7_0_attn_cpb_mlp_1,), kwargs = {})
    %view_22 : [num_users=1] = call_method[target=view](args = (%features_7_0_attn_cpb_mlp_2, -1, 24), kwargs = {})
    %features_7_0_attn_relative_position_index : [num_users=1] = get_attr[target=features.7.0.attn.relative_position_index]
    %_get_relative_position_bias_22 : [num_users=1] = call_function[target=torchvision.models.swin_transformer._get_relative_position_bias](args = (%view_22, %features_7_0_attn_relative_position_index, [8, 8]), kwargs = {})
    %sigmoid_22 : [num_users=1] = call_function[target=torch.sigmoid](args = (%_get_relative_position_bias_22,), kwargs = {})
    %mul_22 : [num_users=1] = call_function[target=operator.mul](args = (16, %sigmoid_22), kwargs = {})
    %features_7_0_attn_qkv_weight : [num_users=1] = get_attr[target=features.7.0.attn.qkv.weight]
    %features_7_0_attn_proj_weight : [num_users=1] = get_attr[target=features.7.0.attn.proj.weight]
    %features_7_0_attn_qkv_bias : [num_users=1] = get_attr[target=features.7.0.attn.qkv.bias]
    %features_7_0_attn_proj_bias : [num_users=1] = get_attr[target=features.7.0.attn.proj.bias]
    %features_7_0_attn_logit_scale : [num_users=1] = get_attr[target=features.7.0.attn.logit_scale]
    %shifted_window_attention_22 : [num_users=1] = call_function[target=torchvision.models.swin_transformer.shifted_window_attention](args = (%features_6_norm, %features_7_0_attn_qkv_weight, %features_7_0_attn_proj_weight, %mul_22, [8, 8], 24), kwargs = {shift_size: [0, 0], attention_dropout: 0.0, dropout: 0.0, qkv_bias: %features_7_0_attn_qkv_bias, proj_bias: %features_7_0_attn_proj_bias, logit_scale: %features_7_0_attn_logit_scale, training: True})
    %features_7_0_norm1 : [num_users=1] = call_module[target=features.7.0.norm1](args = (%shifted_window_attention_22,), kwargs = {})
    %features_7_0_stochastic_depth : [num_users=1] = call_module[target=features.7.0.stochastic_depth](args = (%features_7_0_norm1,), kwargs = {})
    %add_44 : [num_users=2] = call_function[target=operator.add](args = (%features_6_norm, %features_7_0_stochastic_depth), kwargs = {})
    %features_7_0_mlp : [num_users=1] = call_module[target=features.7.0.mlp](args = (%add_44,), kwargs = {})
    %features_7_0_norm2 : [num_users=1] = call_module[target=features.7.0.norm2](args = (%features_7_0_mlp,), kwargs = {})
    %features_7_0_stochastic_depth_1 : [num_users=1] = call_module[target=features.7.0.stochastic_depth](args = (%features_7_0_norm2,), kwargs = {})
    %add_45 : [num_users=2] = call_function[target=operator.add](args = (%add_44, %features_7_0_stochastic_depth_1), kwargs = {})
    %features_7_1_attn_relative_coords_table : [num_users=1] = get_attr[target=features.7.1.attn.relative_coords_table]
    %features_7_1_attn_cpb_mlp_0 : [num_users=1] = call_module[target=features.7.1.attn.cpb_mlp.0](args = (%features_7_1_attn_relative_coords_table,), kwargs = {})
    %features_7_1_attn_cpb_mlp_1 : [num_users=1] = call_module[target=features.7.1.attn.cpb_mlp.1](args = (%features_7_1_attn_cpb_mlp_0,), kwargs = {})
    %features_7_1_attn_cpb_mlp_2 : [num_users=1] = call_module[target=features.7.1.attn.cpb_mlp.2](args = (%features_7_1_attn_cpb_mlp_1,), kwargs = {})
    %view_23 : [num_users=1] = call_method[target=view](args = (%features_7_1_attn_cpb_mlp_2, -1, 24), kwargs = {})
    %features_7_1_attn_relative_position_index : [num_users=1] = get_attr[target=features.7.1.attn.relative_position_index]
    %_get_relative_position_bias_23 : [num_users=1] = call_function[target=torchvision.models.swin_transformer._get_relative_position_bias](args = (%view_23, %features_7_1_attn_relative_position_index, [8, 8]), kwargs = {})
    %sigmoid_23 : [num_users=1] = call_function[target=torch.sigmoid](args = (%_get_relative_position_bias_23,), kwargs = {})
    %mul_23 : [num_users=1] = call_function[target=operator.mul](args = (16, %sigmoid_23), kwargs = {})
    %features_7_1_attn_qkv_weight : [num_users=1] = get_attr[target=features.7.1.attn.qkv.weight]
    %features_7_1_attn_proj_weight : [num_users=1] = get_attr[target=features.7.1.attn.proj.weight]
    %features_7_1_attn_qkv_bias : [num_users=1] = get_attr[target=features.7.1.attn.qkv.bias]
    %features_7_1_attn_proj_bias : [num_users=1] = get_attr[target=features.7.1.attn.proj.bias]
    %features_7_1_attn_logit_scale : [num_users=1] = get_attr[target=features.7.1.attn.logit_scale]
    %shifted_window_attention_23 : [num_users=1] = call_function[target=torchvision.models.swin_transformer.shifted_window_attention](args = (%add_45, %features_7_1_attn_qkv_weight, %features_7_1_attn_proj_weight, %mul_23, [8, 8], 24), kwargs = {shift_size: [4, 4], attention_dropout: 0.0, dropout: 0.0, qkv_bias: %features_7_1_attn_qkv_bias, proj_bias: %features_7_1_attn_proj_bias, logit_scale: %features_7_1_attn_logit_scale, training: True})
    %features_7_1_norm1 : [num_users=1] = call_module[target=features.7.1.norm1](args = (%shifted_window_attention_23,), kwargs = {})
    %features_7_1_stochastic_depth : [num_users=1] = call_module[target=features.7.1.stochastic_depth](args = (%features_7_1_norm1,), kwargs = {})
    %add_46 : [num_users=2] = call_function[target=operator.add](args = (%add_45, %features_7_1_stochastic_depth), kwargs = {})
    %features_7_1_mlp : [num_users=1] = call_module[target=features.7.1.mlp](args = (%add_46,), kwargs = {})
    %features_7_1_norm2 : [num_users=1] = call_module[target=features.7.1.norm2](args = (%features_7_1_mlp,), kwargs = {})
    %features_7_1_stochastic_depth_1 : [num_users=1] = call_module[target=features.7.1.stochastic_depth](args = (%features_7_1_norm2,), kwargs = {})
    %add_47 : [num_users=1] = call_function[target=operator.add](args = (%add_46, %features_7_1_stochastic_depth_1), kwargs = {})
    %norm : [num_users=1] = call_module[target=norm](args = (%add_47,), kwargs = {})
    %permute : [num_users=1] = call_module[target=permute](args = (%norm,), kwargs = {})
    %avgpool : [num_users=1] = call_module[target=avgpool](args = (%permute,), kwargs = {})
    %flatten : [num_users=1] = call_module[target=flatten](args = (%avgpool,), kwargs = {})
    %head : [num_users=1] = call_module[target=head](args = (%flatten,), kwargs = {})
    return {'head': head}"""

# TODO: Add torch Swin to block list.
# TODO: Is there anything we can do to prevent an input from replacing a getattr call?
#    Example: create_sub_network(torchvision.models.swin_v2_s(), input_nodes=["features.7"], return_nodes=["head"])


def test_tracing():
    import math
    from utils.subgraphs import _get_leaf_modules_for_ops, NodePathTracer
    model = torchvision.models.swin_v2_s()
    # model = timm.create_model("vit_small_patch16_224")
    tracer = NodePathTracer(autowrap_modules=(math, torchvision.ops), leaf_modules=_get_leaf_modules_for_ops())
    graph = tracer.trace(model)
    print()
    print(graph)
    # graph.print_tabular()
    print("\n".join([str(x) for x in tracer.node_to_qualname.items()]))
