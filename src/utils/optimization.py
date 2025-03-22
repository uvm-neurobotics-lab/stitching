"""
Utilities for optimization.
"""
import torch
from torch.optim.lr_scheduler import _LRScheduler

from utils import get_matching_module, function_from_name


class DummyScheduler(_LRScheduler):

    def __init__(self, optimizer, last_epoch=-1, verbose=False):
        super(DummyScheduler, self).__init__(optimizer, last_epoch, verbose)

    def get_lr(self):
        # Never change learning rate, just use the existing one.
        return [group['lr'] for group in self.optimizer.param_groups]


def optimizer_from_config(config, params):
    """
    Create an optimizer for the given model, using the given config.

    Args:
        config (dict): The YAML config containing the desired optimization parameters.
        params (nn.Module): The set of parameters to optimize.

    Returns:
        torch.optim.Optimizer: The new optimizer.
    """
    cls = getattr(torch.optim, config["optimizer"])
    return cls(params, **config.get("optimizer_args", {}))


def scheduler_from_config(config, opt):
    """
    Create a learning rate scheduler for the given optimizer, using the given config.

    Args:
        config (dict): The YAML config containing the desired learning rate parameters.
        opt (torch.optim.Optimizer): The optimizer to put on schedule.

    Returns:
        torch.optim.lr_scheduler._LRScheduler: The new scheduler.
        Any: A variable describing when the scheduler's `step()` function should be called. Defaults to "epochs",
             indicating that it should be called once per epoch, which is the common PyTorch convention.
    """
    sched_name = config.get("lr_scheduler")
    if not sched_name or sched_name == "DummyScheduler":
        return DummyScheduler(opt)
    sched_args = config.get("lr_scheduler_args", {})
    cls = getattr(torch.optim.lr_scheduler, sched_name)
    sched_args = sched_args.copy()
    cadence = sched_args.pop("cadence", "epochs")
    return cls(opt, **sched_args), cadence


class AuxLossFn:
    """
    A functor that will compute its loss w.r.t. the output of the given module. Adds a forward hook to the module in
    order to record its output. Subsequently, the user may call it with labels in order to compute the loss.
    """

    def __init__(self, output_module, loss_fn):
        self.output = None
        self.loss_fn = loss_fn
        output_module.register_forward_hook(self.hook)

    def __call__(self, output, labels):
        # Use the saved output rather than the given main output.
        return self.loss_fn(self.output, labels)

    def hook(self, module, args, output):
        # Every time the output module runs forward, we save the output.
        self.output = output


def create_aux_loss_fn(model, output_module_name, loss_fn):
    """
    Create an auxiliary loss function which will compute a loss with respect to the given output head (rather than
    computing loss with respect to the main module output).

    Args:
        model (nn.Module): The model whose loss will be computed.
        output_module_name (str): The fully-qualified name of the module whose output we should monitor.
        loss_fn (Callable): The loss function to call with signature (output, labels) -> value.

    Returns:
        Callable: A functor which can be called with pairs of (output batch, label batch).
    """
    mod = get_matching_module(model, output_module_name)
    return AuxLossFn(mod, loss_fn)


def loss_fns_from_config(config, model):
    """
    Returns a dictionary of all named loss functions, as specified by the given config. Multi-loss training can be
    performed with code equivalent to: `sum(loss_fn(output, labels) for loss_fn in losses.values()).backward()`.

    Args:
        config (dict): The YAML config containing the desired loss function specifications.
        model (nn.Module): The model whose loss will be computed.

    Returns:
        dict: A dict of name -> function.
    """
    loss_fn = function_from_name(config["loss_fn"])
    loss_fns = {"Main Loss": (loss_fn, 1.0)}
    for i, loss_cfg in enumerate(config.get("aux_losses", [])):
        loss_fns[f"Aux Loss {i + 1}"] = (create_aux_loss_fn(model, loss_cfg["output"],
                                                            function_from_name(loss_cfg["loss_fn"])),
                                         loss_cfg.get("weight", 1.0))
    return loss_fns


class MetricFn:
    """
    A functor that will compute its a metric w.r.t. the output of the given module. Adds a forward hook to the module in
    order to record its output. Subsequently, the user may call it with labels in order to compute the metric. If no
    `module_name` is specified, then this will use the user-supplied output.
    """

    def __init__(self, loss_fn, model=None, module_name=None):
        self.loss_fn = loss_fn
        self.module_name = module_name
        self.output = None
        if module_name:
            output_module = get_matching_module(model, module_name)
            output_module.register_forward_hook(self.hook)

    def __call__(self, output, labels):
        if self.module_name:
            # Use the saved output rather than the given main output.
            res = self.loss_fn(self.output, labels)
            # Add a more specific name if we are operating on a non-default head.
            return {f"{self.module_name} {k}": v for k, v in res.items()}
        else:
            return self.loss_fn(output, labels)

    def hook(self, module, args, output):
        # Every time the output module runs forward, we save the output.
        self.output = output


def metric_fns_from_config(config, model):
    """
    Returns a list of all named functions, as specified by the given config. These operate similar to a loss function,
    but have the signature:
        (output: Tensor, labels: Tensor) -> dict[metric_name: str, metric_value: float]

    Args:
        config (dict): The YAML config containing the desired metric specifications.
        model (nn.Module): The model on which metrics will be computed.

    Returns:
        dict: A list of metric functions.
    """
    metric_fns = []
    for metric_cfg in config.get("metrics", []):
        mfn = function_from_name(metric_cfg["metric_fn"], **metric_cfg.get("metric_fn_args", {}))
        metric_fns.append(MetricFn(mfn, model, metric_cfg.get("output")))
    return metric_fns


def limit_model_optimization(model, param_names_to_optimize):
    """
    Modifies the model parameters so that only the given params will be optimized. Uses the `requires_grad` property to
    do so.

    Args:
        model (torch.nn.Module): The model whose parameters we wish to modify.
        param_names_to_optimize (list[str] or set[str]): The list of exact names of the parameters which *should* be
            optimized, as given by `model.named_parameters()`.

    Returns:
        dict: A dict of string -> bool which stores the previous state of the `requires_grad` property of each param.
            This can be used to restore the previous state by calling `restore_grad_state()`.
    """
    saved_opt_state = {}
    # Select which layers will recieve updates during optimization, by setting the requires_grad property.
    for name, p in model.named_parameters():
        saved_opt_state[name] = p.requires_grad
        p.requires_grad_(name in param_names_to_optimize)
    return saved_opt_state
