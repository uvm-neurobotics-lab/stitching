"""
Code for cutting arbitrary computation graphs into blocks/subsets.
 - Allows blocks to be reused elsewhere.
 - Allows extracting the output at a particular point within a network.

This code was adapted with many thanks from Deep Model Reassembly:
    https://github.com/Adamdad/DeRy
Primarily from this module:
    https://github.com/Adamdad/DeRy/blob/main/mmcls_addon/models/utils/feature_extraction.py
"""
import inspect
import math
import re
import warnings
from collections import OrderedDict
from copy import deepcopy
from itertools import chain
from typing import Dict, Callable, List, Union, Optional, Tuple, Any

import torch
import torchvision
from torch import fx
from torch import nn
from torch.fx.graph_module import _copy_attr


class LeafModuleAwareTracer(fx.Tracer):
    """
    An fx.Tracer that allows the user to specify a set of leaf modules, ie.
    modules that are not to be traced through. The resulting graph ends up
    having single nodes referencing calls to the leaf modules' forward methods.
    """

    def __init__(self, *args, **kwargs):
        self.leaf_modules = {}
        if "leaf_modules" in kwargs:
            leaf_modules = kwargs.pop("leaf_modules")
            self.leaf_modules = leaf_modules
        super().__init__(*args, **kwargs)

    def is_leaf_module(self, m: nn.Module, module_qualname: str) -> bool:
        if isinstance(m, tuple(self.leaf_modules)):
            return True
        return super().is_leaf_module(m, module_qualname)


class NodePathTracer(LeafModuleAwareTracer):
    """
    NodePathTracer is an FX tracer that, for each operation, also records the
    name of the Node from which the operation originated. A node name here is
    a `.` separated path walking the hierarchy from top level module down to
    leaf operation or leaf module. The name of the top level module is not
    included as part of the node name. For example, if we trace a module whose
    forward method applies a ReLU module, the name for that node will simply
    be 'relu'.

    Some notes on the specifics:
        - Nodes are recorded to `self.node_to_qualname` which is a dictionary
          mapping a given Node object to its node name.
        - Nodes are recorded in the order which they are executed during
          tracing.
        - When a duplicate node name is encountered, a suffix of the form
          _{int} is added. The counter starts from 1.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Track the qualified name of the Node being traced
        self.current_module_qualname = ""
        # A map from FX Node to the qualified name\#
        # NOTE: This is loosely like the "qualified name" mentioned in the
        # torch.fx docs https://pytorch.org/docs/stable/fx.html but adapted
        # for the purposes of the torchvision feature extractor
        self.node_to_qualname = OrderedDict()

    def call_module(self, m: torch.nn.Module, forward: Callable, args, kwargs):
        """
        Override of `fx.Tracer.call_module`
        This override:
        1) Stores away the qualified name of the caller for restoration later
        2) Adds the qualified name of the caller to
           `current_module_qualname` for retrieval by `create_proxy`
        3) Once a leaf module is reached, calls `create_proxy`
        4) Restores the caller's qualified name into current_module_qualname
        """
        old_qualname = self.current_module_qualname
        try:
            module_qualname = self.path_of_module(m)
            self.current_module_qualname = module_qualname
            if not self.is_leaf_module(m, module_qualname):
                out = forward(*args, **kwargs)
                return out
            return self.create_proxy("call_module", module_qualname, args, kwargs)
        finally:
            self.current_module_qualname = old_qualname

    def create_proxy(
            self, kind: str, target: fx.node.Target, args, kwargs, name=None, type_expr=None, *_
    ) -> fx.proxy.Proxy:
        """
        Override of `Tracer.create_proxy`. This override intercepts the recording
        of every operation and stores away the current traced module's qualified
        name in `node_to_qualname`
        """
        proxy = super().create_proxy(kind, target, args, kwargs, name, type_expr)
        self.node_to_qualname[proxy.node] = self._get_node_qualname(self.current_module_qualname, proxy.node)
        return proxy

    def _get_node_qualname(self, module_qualname: str, node: fx.node.Node) -> str:
        node_qualname = module_qualname

        if node.op != "call_module":
            # In this case module_qualname from torch.fx doesn't go all the
            # way to the leaf function/op so we need to append it
            if len(node_qualname) > 0:
                # Only append '.' if we are deeper than the top level module
                node_qualname += "."
            node_qualname += str(node)

        # Now we need to add an _{index} postfix on any repeated node names
        # For modules we do this from scratch
        # But for anything else, torch.fx already has a globally scoped
        # _{index} postfix. But we want it locally (relative to direct parent)
        # scoped. So first we need to undo the torch.fx postfix
        if re.match(r".+_[0-9]+$", node_qualname) is not None:
            node_qualname = node_qualname.rsplit("_", 1)[0]

        # ... and now we add on our own postfix
        for existing_qualname in reversed(self.node_to_qualname.values()):
            # Check to see if existing_qualname is of the form
            # {node_qualname} or {node_qualname}_{int}
            if re.match(rf"{node_qualname}(_[0-9]+)?$", existing_qualname) is not None:
                postfix = existing_qualname.replace(node_qualname, "")
                if len(postfix):
                    # existing_qualname is of the form {node_qualname}_{int}
                    next_index = int(postfix[1:]) + 1
                else:
                    # existing_qualname is of the form {node_qualname}
                    next_index = 1
                node_qualname += f"_{next_index}"
                break

        return node_qualname


class DualGraphModule(fx.GraphModule):
    """
    A derivative of `fx.GraphModule`. Differs in the following ways:
    - Requires a train and eval version of the underlying graph
    - Copies submodules according to the nodes of both train and eval graphs.
    - Calling train(mode) switches between train graph and eval graph.
    """

    def __init__(
            self, root: torch.nn.Module, train_graph: fx.Graph, eval_graph: fx.Graph, class_name: str = "GraphModule"
    ):
        """
        Args:
            root (nn.Module): module from which the copied module hierarchy is
                built
            train_graph (fx.Graph): the graph that should be used in train mode
            eval_graph (fx.Graph): the graph that should be used in eval mode
        """
        # NOTE: The DeRy code bypasses the superclass initiator! But that seems supremely stupid.
        # super(fx.GraphModule, self).__init__()
        super().__init__(root, train_graph)

        self.__class__.__name__ = class_name

        self.train_graph = train_graph
        self.eval_graph = eval_graph

        # Copy all get_attr and call_module ops (indicated by BOTH train and
        # eval graphs)
        for node in chain(iter(train_graph.nodes), iter(eval_graph.nodes)):
            if node.op in ["get_attr", "call_module"]:
                assert isinstance(node.target, str)
                _copy_attr(root, self, node.target)

        # train mode by default
        self.train()

        # NOTE: In the DeRy code, this is copied from fx.GraphModule. But we shouldn't need it now that we are properly
        # calling the super constructor.
        # assert (
        #         self.eval_graph._tracer_cls == self.train_graph._tracer_cls
        # ), "Train mode and eval mode should use the same tracer class"
        # self._tracer_cls = None
        # if self.graph._tracer_cls and "<locals>" not in self.graph._tracer_cls.__qualname__:
        #     self._tracer_cls = self.graph._tracer_cls

    def train(self, mode=True):
        """
        Swap out the graph depending on the selected training mode.
        NOTE this should be safe when calling model.eval() because that just
        calls this with mode == False.
        """
        # NOTE: Only set self.graph if the current graph is not the desired
        # one. This saves us from recompiling the graph where not necessary.
        if mode and not self.training:
            self.graph = self.train_graph
        elif not mode and self.training:
            self.graph = self.eval_graph
        return super().train(mode=mode)


def to_strdict(n: Union[str, List[str], Dict[str, str]]) -> Dict[str, str]:
    """
    Put a node list argument into Dict[str, str] format, if it is not already.
    """
    if isinstance(n, str):
        return {n: n}
    if isinstance(n, list):
        return {str(i): str(i) for i in n}
    return {str(k): str(v) for k, v in n.items()}


def _is_subseq(x, y):
    """Check if y is a subseqence of x
    https://stackoverflow.com/a/24017747/4391249
    """
    iter_x = iter(x)
    return all(any(x_item == y_item for x_item in iter_x) for y_item in y)


def _warn_graph_differences(train_tracer: NodePathTracer, eval_tracer: NodePathTracer):
    """
    Utility function for warning the user if there are differences between
    the train graph nodes and the eval graph nodes.
    """
    train_nodes = list(train_tracer.node_to_qualname.values())
    eval_nodes = list(eval_tracer.node_to_qualname.values())

    if len(train_nodes) == len(eval_nodes) and all(t == e for t, e in zip(train_nodes, eval_nodes)):
        return

    suggestion_msg = (
        "When choosing nodes for feature extraction, you may need to specify "
        "output nodes for train and eval mode separately."
    )

    if _is_subseq(train_nodes, eval_nodes):
        msg = (
            "NOTE: The nodes obtained by tracing the model in eval mode "
            "are a subsequence of those obtained in train mode. "
        )
    elif _is_subseq(eval_nodes, train_nodes):
        msg = (
            "NOTE: The nodes obtained by tracing the model in train mode "
            "are a subsequence of those obtained in eval mode. "
        )
    else:
        msg = "The nodes obtained by tracing the model in train mode are different to those obtained in eval mode. "
    warnings.warn(msg + suggestion_msg)


def _get_leaf_modules_for_ops() -> List[type]:
    members = inspect.getmembers(torchvision.ops)
    result = []
    for _, obj in members:
        if inspect.isclass(obj) and issubclass(obj, torch.nn.Module):
            result.append(obj)
    return result


def get_graph_node_names(
        model: nn.Module,
        tracer_kwargs: Optional[Dict[str, Any]] = None,
        suppress_diff_warning: bool = False,
) -> Tuple[List[str], List[str]]:
    """
    Dev utility to return node names in order of execution. See note on node
    names under :func:`create_feature_extractor`. Useful for seeing which node
    names are available for feature extraction. There are two reasons that
    node names can't easily be read directly from the code for a model:

        1. Not all submodules are traced through. Modules from ``torch.nn`` all
           fall within this category.
        2. Nodes representing the repeated application of the same operation
           or leaf module get a ``_{counter}`` postfix.

    The model is traced twice: once in train mode, and once in eval mode. Both
    sets of node names are returned.

    For more details on the node naming conventions used here, please see the
    :ref:`relevant subheading <about-node-names>` in the
    `documentation <https://pytorch.org/vision/stable/feature_extraction.html>`_.

    Args:
        model (nn.Module): model for which we'd like to print node names
        tracer_kwargs (dict, optional): a dictionary of keywork arguments for
            ``NodePathTracer`` (they are eventually passed onto
            `torch.fx.Tracer <https://pytorch.org/docs/stable/fx.html#torch.fx.Tracer>`_).
            By default it will be set to wrap and make leaf nodes all torchvision ops.
        suppress_diff_warning (bool, optional): whether to suppress a warning
            when there are discrepancies between the train and eval version of
            the graph. Defaults to False.

    Returns:
        tuple(list, list): a list of node names from tracing the model in
        train mode, and another from tracing the model in eval mode.

    Examples::

        >>> model = torchvision.models.resnet18()
        >>> train_nodes, eval_nodes = get_graph_node_names(model)
    """
    if tracer_kwargs is None:
        tracer_kwargs = {
            "autowrap_modules": (
                math,
                torchvision.ops,
            ),
            "leaf_modules": _get_leaf_modules_for_ops(),
        }
    is_training = model.training
    train_tracer = NodePathTracer(**tracer_kwargs)
    train_tracer.trace(model.train())
    eval_tracer = NodePathTracer(**tracer_kwargs)
    eval_tracer.trace(model.eval())
    train_nodes = list(train_tracer.node_to_qualname.values())
    eval_nodes = list(eval_tracer.node_to_qualname.values())
    if not suppress_diff_warning:
        _warn_graph_differences(train_tracer, eval_tracer)
    # Restore training state
    model.train(is_training)
    return train_nodes, eval_nodes


def print_model(model: nn.Module, print_module_info: bool = False, print_graph: bool = True,
                print_fn: Callable = print):
    """
    A function to print out the detailed structure of the model for logging or debugging purposes.

    Args:
        model: model on which we will extract the features
        print_module_info: Whether to print the modules (standard nn.Module string output).
        print_graph: Whether to trace and print the full computation graph.
        print_fn: The function to use for printing (e.g., `print`, `logging.info`).
    """
    def print_w_newline(msg):
        print_fn("\n" + str(msg))

    if print_module_info:
        print_w_newline(model)

    if print_graph:
        if isinstance(model, torch.fx.GraphModule):
            print_w_newline(model.graph)
        else:
            tracer = NodePathTracer(autowrap_modules=(math, torchvision.ops), leaf_modules=_get_leaf_modules_for_ops())
            graph = tracer.trace(model)
            print_w_newline(graph)


def create_feature_extractor(
        model: nn.Module,
        return_nodes: Optional[Union[List[str], Dict[str, str]]] = None,
        train_return_nodes: Optional[Union[List[str], Dict[str, str]]] = None,
        eval_return_nodes: Optional[Union[List[str], Dict[str, str]]] = None,
        tracer_kwargs: Optional[Dict[str, Any]] = None,
        suppress_diff_warning: bool = False,
) -> fx.GraphModule:
    """
    Creates a new graph module that returns intermediate nodes from a given
    model as dictionary with user specified keys as strings, and the requested
    outputs as values. This is achieved by re-writing the computation graph of
    the model via FX to return the desired nodes as outputs. All unused nodes
    are removed, together with their corresponding parameters.

    Desired output nodes must be specified as a ``.`` separated
    path walking the module hierarchy from top level module down to leaf
    operation or leaf module. For more details on the node naming conventions
    used here, please see the :ref:`relevant subheading <about-node-names>`
    in the `documentation <https://pytorch.org/vision/stable/feature_extraction.html>`_.

    Not all models will be FX traceable, although with some massaging they can
    be made to cooperate. Here's a (not exhaustive) list of tips:

        - If you don't need to trace through a particular, problematic
          sub-module, turn it into a "leaf module" by passing a list of
          ``leaf_modules`` as one of the ``tracer_kwargs`` (see example below).
          It will not be traced through, but rather, the resulting graph will
          hold a reference to that module's forward method.
        - Likewise, you may turn functions into leaf functions by passing a
          list of ``autowrap_functions`` as one of the ``tracer_kwargs`` (see
          example below).
        - Some inbuilt Python functions can be problematic. For instance,
          ``int`` will raise an error during tracing. You may wrap them in your
          own function and then pass that in ``autowrap_functions`` as one of
          the ``tracer_kwargs``.

    For further information on FX see the
    `torch.fx documentation <https://pytorch.org/docs/stable/fx.html>`_.

    Args:
        model (nn.Module): model on which we will extract the features
        return_nodes (list or dict, optional): either a ``List`` or a ``Dict``
            containing the names (or partial names - see note above)
            of the nodes for which the activations will be returned. If it is
            a ``Dict``, the keys are the node names, and the values
            are the user-specified keys for the graph module's returned
            dictionary. If it is a ``List``, it is treated as a ``Dict`` mapping
            node specification strings directly to output names. In the case
            that ``train_return_nodes`` and ``eval_return_nodes`` are specified,
            this should not be specified.
        train_return_nodes (list or dict, optional): similar to
            ``return_nodes``. This can be used if the return nodes
            for train mode are different than those from eval mode.
            If this is specified, ``eval_return_nodes`` must also be specified,
            and ``return_nodes`` should not be specified.
        eval_return_nodes (list or dict, optional): similar to
            ``return_nodes``. This can be used if the return nodes
            for train mode are different than those from eval mode.
            If this is specified, ``train_return_nodes`` must also be specified,
            and `return_nodes` should not be specified.
        tracer_kwargs (dict, optional): a dictionary of keywork arguments for
            ``NodePathTracer`` (which passes them onto it's parent class
            `torch.fx.Tracer <https://pytorch.org/docs/stable/fx.html#torch.fx.Tracer>`_).
            By default it will be set to wrap and make leaf nodes all torchvision ops.
        suppress_diff_warning (bool, optional): whether to suppress a warning
            when there are discrepancies between the train and eval version of
            the graph. Defaults to False.

    Examples::

        >>> # Feature extraction with resnet
        >>> model = torchvision.models.resnet18()
        >>> # extract layer1 and layer3, giving as names `feat1` and feat2`
        >>> model = create_feature_extractor(
        >>>     model, {'layer1': 'feat1', 'layer3': 'feat2'})
        >>> out = model(torch.rand(1, 3, 224, 224))
        >>> print([(k, v.shape) for k, v in out.items()])
        >>>     [('feat1', torch.Size([1, 64, 56, 56])),
        >>>      ('feat2', torch.Size([1, 256, 14, 14]))]

        >>> # Specifying leaf modules and leaf functions
        >>> def leaf_function(x):
        >>>     # This would raise a TypeError if traced through
        >>>     return int(x)
        >>>
        >>> class LeafModule(torch.nn.Module):
        >>>     def forward(self, x):
        >>>         # This would raise a TypeError if traced through
        >>>         int(x.shape[0])
        >>>         return torch.nn.functional.relu(x + 4)
        >>>
        >>> class MyModule(torch.nn.Module):
        >>>     def __init__(self):
        >>>         super().__init__()
        >>>         self.conv = torch.nn.Conv2d(3, 1, 3)
        >>>         self.leaf_module = LeafModule()
        >>>
        >>>     def forward(self, x):
        >>>         leaf_function(x.shape[0])
        >>>         x = self.conv(x)
        >>>         return self.leaf_module(x)
        >>>
        >>> model = create_feature_extractor(
        >>>     MyModule(), return_nodes=['leaf_module'],
        >>>     tracer_kwargs={'leaf_modules': [LeafModule],
        >>>                    'autowrap_functions': [leaf_function]})

    """
    if tracer_kwargs is None:
        tracer_kwargs = {
            "autowrap_modules": (
                math,
                torchvision.ops,
            ),
            "leaf_modules": _get_leaf_modules_for_ops(),
        }
    is_training = model.training

    assert any(
        arg is not None for arg in [return_nodes, train_return_nodes, eval_return_nodes]
    ), "Either `return_nodes` or `train_return_nodes` and `eval_return_nodes` together, should be specified"

    assert not (
            (train_return_nodes is None) ^ (eval_return_nodes is None)
    ), "If any of `train_return_nodes` and `eval_return_nodes` are specified, then both should be specified"

    assert (return_nodes is None) ^ (
            train_return_nodes is None
    ), "If `train_return_nodes` and `eval_return_nodes` are specified, then both should be specified"

    # Put *_return_nodes into Dict[str, str] format
    def to_strdict(n) -> Dict[str, str]:
        if isinstance(n, list):
            return {str(i): str(i) for i in n}
        return {str(k): str(v) for k, v in n.items()}

    if train_return_nodes is None:
        return_nodes = to_strdict(return_nodes)
        train_return_nodes = deepcopy(return_nodes)
        eval_return_nodes = deepcopy(return_nodes)
    else:
        train_return_nodes = to_strdict(train_return_nodes)
        eval_return_nodes = to_strdict(eval_return_nodes)

    # Repeat the tracing and graph rewriting for train and eval mode
    tracers = {}
    graphs = {}
    mode_return_nodes: Dict[str, Dict[str, str]] = {"train": train_return_nodes, "eval": eval_return_nodes}
    for mode in ["train", "eval"]:
        if mode == "train":
            model.train()
        elif mode == "eval":
            model.eval()

        # Instantiate our NodePathTracer and use that to trace the model
        tracer = NodePathTracer(**tracer_kwargs)
        graph = tracer.trace(model)

        name = model.__class__.__name__ if isinstance(model, nn.Module) else model.__name__
        graph_module = fx.GraphModule(tracer.root, graph, name)

        available_nodes = list(tracer.node_to_qualname.values())
        # NOTE: We don't know if we should expect this to happen
        assert len(set(available_nodes)) == len(
            available_nodes
        ), "There are duplicate nodes! Please raise an issue https://github.com/pytorch/vision/issues"
        # Check that all outputs in return_nodes are present in the model
        for query in mode_return_nodes[mode].keys():
            # To check if a query is available we need to check that at least
            # one of the available names starts with it up to a .
            if not any([re.match(rf"^{query}(\.|$)", n) is not None for n in available_nodes]):
                raise ValueError(
                    f"node: '{query}' is not present in model. Hint: use "
                    "`get_graph_node_names` to make sure the "
                    "`return_nodes` you specified are present. It may even "
                    "be that you need to specify `train_return_nodes` and "
                    "`eval_return_nodes` separately."
                )

        # Remove existing output nodes.
        orig_output_nodes = []
        for n in reversed(graph_module.graph.nodes):
            if n.op == "output":
                orig_output_nodes.append(n)
        assert len(orig_output_nodes)
        for n in orig_output_nodes:
            graph_module.graph.erase_node(n)

        # Find nodes corresponding to return_nodes and make them into new_output_nodes
        nodes = [n for n in graph_module.graph.nodes]
        new_output_nodes = OrderedDict()
        for n in reversed(nodes):
            module_qualname = tracer.node_to_qualname.get(n)
            if module_qualname is None:
                # NOTE - Know cases where this happens:
                # - Node representing creation of a tensor constant - probably
                #   not interesting as a return node
                # - When packing outputs into a named tuple like in InceptionV3
                continue
            for query in mode_return_nodes[mode]:
                depth = query.count(".")
                if ".".join(module_qualname.split(".")[: depth + 1]) == query:
                    new_output_nodes[mode_return_nodes[mode][query]] = n
                    mode_return_nodes[mode].pop(query)
                    break
        new_output_nodes = OrderedDict(reversed(list(new_output_nodes.items())))

        # And add them in the end of the graph
        with graph_module.graph.inserting_after(nodes[-1]):
            graph_module.graph.output(new_output_nodes)

        # Remove unused modules / parameters
        graph_module.graph.eliminate_dead_code()
        graph_module.recompile()

        # Keep track of the tracer and graph so we can choose the main one
        tracers[mode] = tracer
        graphs[mode] = graph

    # Warn user if there are any discrepancies between the graphs of the
    # train and eval modes
    if not suppress_diff_warning:
        _warn_graph_differences(tracers["train"], tracers["eval"])

    # Build the final graph module
    graph_module = DualGraphModule(model, graphs["train"], graphs["eval"], class_name=name)

    # Restore original training mode
    model.train(is_training)
    graph_module.train(is_training)

    return graph_module


def get_placeholder_ancestor_of(node: torch.fx.Node):
    """
    Trace a node back to its root ancestor node, which could be itself if this is a root. Just returns the first one
    found via depth-first search.
    """
    if len(node.all_input_nodes) == 0 and node.op == "placeholder":
        return node  # node is a root
    for a in node.all_input_nodes:
        root = get_placeholder_ancestor_of(a)
        if root:
            return root
    return None  # no placeholder root; root must be an attribute or a constant


def create_sub_network(
        model: nn.Module,
        input_nodes: Optional[Union[List[str], Dict[str, str]]] = None,
        return_nodes: Optional[Union[List[str], Dict[str, str]]] = None,
        train_input_nodes: Optional[Union[List[str], Dict[str, str]]] = None,
        eval_input_nodes: Optional[Union[List[str], Dict[str, str]]] = None,
        train_return_nodes: Optional[Union[List[str], Dict[str, str]]] = None,
        eval_return_nodes: Optional[Union[List[str], Dict[str, str]]] = None,
        tracer_kwargs: Optional[Dict[str, Any]] = None,
        suppress_diff_warning: bool = False,
) -> fx.GraphModule:
    """
    Given a module, return the subnet that has `input_nodes` as inputs and `return_nodes` as outputs.
    TODO: Finish docs

    IMPORTANT: This method can only currently support cases where each input node corresponds to one path through the
    graph. The graph cut must account for all edges with a single edge per node; i.e., one node may not take two inputs.
    Additionally, we use some heuristics to decide which input needs to be replaced for a given input node, and these
    may not always be correct. If something goes wrong, you may get an unintuitive error like:

        "RuntimeError: Tried to erase Node x but it still had 1 users in the graph: {conv1: None}!"

    This means that something went wrong with the graph cut, but it's hard to say exactly what. That will be up to you
    to find out.

    :param model:
    :param input_nodes:
    :param return_nodes:
    :param train_input_nodes:
    :param eval_input_nodes:
    :param train_return_nodes:
    :param eval_return_nodes:
    :param tracer_kwargs:
    :param suppress_diff_warning:
    :return:
    """
    if tracer_kwargs is None:
        tracer_kwargs = {
            "autowrap_modules": (
                math,
                torchvision.ops,
            ),
            "leaf_modules": _get_leaf_modules_for_ops(),
        }
    is_training = model.training

    assert any(
        arg is not None for arg in [return_nodes, train_return_nodes, eval_return_nodes]
    ), "Either `return_nodes` or `train_return_nodes` and `eval_return_nodes` together, should be specified"

    assert not (
            (train_return_nodes is None) ^ (eval_return_nodes is None)
    ), "If any of `train_return_nodes` and `eval_return_nodes` are specified, then both should be specified"

    assert (return_nodes is None) ^ (
            train_return_nodes is None
    ), "If `train_return_nodes` and `eval_return_nodes` are specified, then both should be specified"

    if train_return_nodes is None:
        input_nodes = to_strdict(input_nodes)
        train_input_nodes = deepcopy(input_nodes)
        eval_input_nodes = deepcopy(input_nodes)

        return_nodes = to_strdict(return_nodes)
        train_return_nodes = deepcopy(return_nodes)
        eval_return_nodes = deepcopy(return_nodes)
    else:
        train_input_nodes = to_strdict(train_input_nodes)
        eval_input_nodes = to_strdict(eval_input_nodes)

        train_return_nodes = to_strdict(train_return_nodes)
        eval_return_nodes = to_strdict(eval_return_nodes)

    # Repeat the tracing and graph rewriting for train and eval mode
    tracers = {}
    graphs = {}
    mode_return_nodes: Dict[str, Dict[str, str]] = {"train": train_return_nodes, "eval": eval_return_nodes}
    mode_input_nodes: Dict[str, Dict[str, str]] = {"train": train_input_nodes, "eval": eval_input_nodes}
    for mode in ["train", "eval"]:
        if mode == "train":
            model.train()
        elif mode == "eval":
            model.eval()

        # Instantiate our NodePathTracer and use that to trace the model
        tracer = NodePathTracer(**tracer_kwargs)
        graph = tracer.trace(model)

        name = model.__class__.__name__ if isinstance(model, nn.Module) else model.__name__
        graph_module = fx.GraphModule(tracer.root, graph, name)

        available_nodes = list(tracer.node_to_qualname.values())
        # NOTE: We don't know if we should expect this to happen
        assert len(set(available_nodes)) == len(
            available_nodes
        ), "There are duplicate nodes! Please raise an issue https://github.com/pytorch/vision/issues"
        # Check that all outputs in return_nodes are present in the model
        for query in mode_return_nodes[mode].keys():
            # To check if a query is available we need to check that at least
            # one of the available names starts with it up to a .
            if not any([re.match(rf"^{query}(\.|$)", n) is not None for n in available_nodes]):
                raise ValueError(
                    f"node: '{query}' is not present in model. Hint: use "
                    "`get_graph_node_names` to make sure the "
                    "`return_nodes` you specified are present. It may even "
                    "be that you need to specify `train_return_nodes` and "
                    "`eval_return_nodes` separately."
                )

        for query in mode_input_nodes[mode].keys():
            # To check if a query is available we need to check that at least
            # one of the available names starts with it up to a .
            if not any([re.match(rf"^{query}(\.|$)", n) is not None for n in available_nodes]):
                raise ValueError(
                    f"node: '{query}' is not present in model. Hint: use "
                    "`get_graph_node_names` to make sure the "
                    "`return_nodes` you specified are present. It may even "
                    "be that you need to specify `train_return_nodes` and "
                    "`eval_return_nodes` separately."
                )

        # Remove existing output nodes.
        orig_output_nodes = []
        for n in reversed(graph_module.graph.nodes):
            if n.op == "output":
                orig_output_nodes.append(n)
        assert len(orig_output_nodes)
        for n in orig_output_nodes:
            graph_module.graph.erase_node(n)

        # Find existing input nodes, but before we remove them we need to rewire the graph.
        orig_input_nodes = []
        for n in reversed(graph_module.graph.nodes):
            if n.op == "placeholder":
                orig_input_nodes.append(n)
        assert len(orig_input_nodes)

        # Find nodes corresponding to return_nodes and make them into new_output_nodes
        nodes = [n for n in graph_module.graph.nodes]
        new_output_nodes = OrderedDict()
        for n in reversed(nodes):
            module_qualname = tracer.node_to_qualname.get(n)
            if module_qualname is None:
                # NOTE - Know cases where this happens:
                # - Node representing creation of a tensor constant - probably
                #   not interesting as a return node
                # - When packing outputs into a named tuple like in InceptionV3
                continue
            for query in mode_return_nodes[mode]:
                depth = query.count(".")
                if ".".join(module_qualname.split(".")[: depth + 1]) == query:
                    new_output_nodes[mode_return_nodes[mode][query]] = n
                    mode_return_nodes[mode].pop(query)
                    break

        new_output_nodes = OrderedDict(reversed(list(new_output_nodes.items())))

        # And add them in the end of the graph
        with graph_module.graph.inserting_after(nodes[-1]):
            new_output = graph_module.graph.output(new_output_nodes)

        # Find nodes corresponding to input_nodes and mark them as new_input_nodes.
        nodes = [n for n in graph_module.graph.nodes]
        new_input_nodes = []  # a list of (location to replace, node to replace)
        for n in nodes:
            module_qualname = tracer.node_to_qualname.get(n)
            if module_qualname is None:
                # NOTE - Know cases where this happens:
                # - Node representing creation of a tensor constant - probably
                #   not interesting as a return node
                # - When packing outputs into a named tuple like in InceptionV3
                continue
            for query in mode_input_nodes[mode]:
                depth = query.count(".")
                if ".".join(module_qualname.split(".")[: depth + 1]) == query:
                    # This is the query node, OR A SUBNODE of the query. We don't necessarily want the first such
                    # subnode. We want the first one that can be traced back to a placeholder node, RATHER than an
                    # attribute or a constant. In other words, we want a descendant of the original graph input,
                    # because we seek to sever the graph from its original input.
                    # Furthermore, we require that only ONE of the inputs maps back to the beginning of the graph.
                    if get_placeholder_ancestor_of(n):
                        old_args = n.all_input_nodes
                        if len(old_args) == 0:
                            if n.op == "placeholder":
                                # Replace this node itself.
                                new_input_nodes.append((n, n))
                            else:
                                # get_placeholder_ancestor_of should never return a non-placeholder root.
                                raise RuntimeError(f"Programming error. This should not happen. Node: '{n}'.")
                        elif len(old_args) == 1:
                            # Replace the only input.
                            new_input_nodes.append((n, old_args[0]))
                        else:
                            # Multiple graph paths run into this node. We need to figure out which one to replace.
                            # The one to replace is the one that can be traced back to a placeholder.
                            reqd_inputs = [a for a in old_args if get_placeholder_ancestor_of(a) is not None]
                            if len(reqd_inputs) == 1:
                                new_input_nodes.append((n, reqd_inputs[0]))
                            else:
                                raise RuntimeError("Cannot use a node as input that requires more than one ancestor "
                                                   f"input: Node '{n}' matching query '{query}' requires inputs: "
                                                   f"{reqd_inputs}\n{graph_module.graph}")
                        mode_input_nodes[mode].pop(query)
                        break

        # For each new input node, replace it with a new "placeholder", which will become the new start of the graph.
        node_order = {n: i for i, n in enumerate(graph_module.graph.nodes)}
        new_placeholders = []
        for i, (loc, to_replace) in enumerate(new_input_nodes):
            def is_after_loc(user_node):
                return node_order[loc] <= node_order[user_node]

            with graph_module.graph.inserting_before(loc):
                # WARNING: We need unique placeholder names. This is a hack in the hopes that similar names will be
                # unlikely, rather than actually doing the work to unsure uniqueness.
                new_placeholders.append(graph_module.graph.placeholder(f'subnet_input_{i + 1}'))
                to_replace.replace_all_uses_with(new_placeholders[-1], is_after_loc)

        # Remove unused modules / parameters first.
        node_order = {n: i for i, n in enumerate(graph_module.graph.nodes)}
        earliest_nodedex = min([node_order[n] for n in new_placeholders])
        latest_nodedex = node_order[new_output]

        # We don't want to accidentally remove side-effectful nodes (nodes which have zero users yet may still have an
        # effect on execution). However, we don't care about them if they aren't actually part of our subnet. So only
        # call them "impure" if they are topologically within the subnet boundaries.
        def is_side_effectful(node):
            return node.is_impure() and (earliest_nodedex <= node_order[node] <= latest_nodedex)

        graph_module.graph.eliminate_dead_code(is_side_effectful)
        graph_module.recompile()  # This call may be unnecessary?

        # Remove old input node (its users should now be eliminated).
        for n in orig_input_nodes:
            graph_module.graph.erase_node(n)

        # Final recompile.
        graph_module.graph.eliminate_dead_code()
        graph_module.recompile()

        # Keep track of the tracer and graph so we can choose the main one
        tracers[mode] = tracer
        graphs[mode] = graph

    # Warn user if there are any discrepancies between the graphs of the
    # train and eval modes
    if not suppress_diff_warning:
        _warn_graph_differences(tracers["train"], tracers["eval"])

    # Build the final graph module
    graph_module = DualGraphModule(model, graphs["train"], graphs["eval"], class_name=name)

    # Restore original training mode
    model.train(is_training)
    graph_module.train(is_training)

    return graph_module
