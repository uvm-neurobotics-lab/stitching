"""
Functions to support training networks in ways that can form the basis of a NAS benchmark.
"""
import hashlib
import logging
import time
from collections.abc import Mapping, Sequence, Set
from enum import Enum
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from filelock import FileLock
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import TensorDataset, DataLoader
from tqdm.notebook import tqdm

from utils import ensure_config_param, get_leaf_modules, load_yaml, make_pretty, restore_grad_state, sizeof
from utils.logging import eval_mode, overall_metrics, StandardLog
from utils.optimization import (limit_model_optimization, loss_fns_from_config, metric_fns_from_config,
                                optimizer_from_config, scheduler_from_config)


class Metric(Enum):
    TRAIN_ACC = "Train Accuracy"
    TRAIN_LOSS = "Train Loss"
    TRAIN_TIME = "Train Time"
    VAL_ACC = "Validation Accuracy"
    TEST_ACC = "Test Accuracy"
    TEST_TIME = "Test Time"
    PARAMS = "Parameters"
    FLOPS = "FLOPS"
    EPOCHS = "Epochs"
    LR = "LR"

    def get_name(self):
        """Returns a human-readable name for the metric."""
        return self.value

    def get_filename(self):
        """Returns a machine-compatible name for the metric, good for use as part of a filename or table column."""
        return self.value.lower().replace(" ", "-")


LOG2METRIC = {
    "Loss": Metric.TRAIN_LOSS,
    "Cumulative Train Time": Metric.TRAIN_TIME,
    "Overall/Test Eval Time": Metric.TEST_TIME,
    "Overall/Train Top-1 Accuracy": Metric.TRAIN_ACC,
    "Overall/Valid Top-1 Accuracy": Metric.VAL_ACC,
    "Overall/Test Top-1 Accuracy": Metric.TEST_ACC,
    "Overall/Train Accuracy": Metric.TRAIN_ACC,
    "Overall/Valid Accuracy": Metric.VAL_ACC,
    "Overall/Test Accuracy": Metric.TEST_ACC,
}

NO_NAN_COLS = [Metric.TRAIN_ACC, Metric.VAL_ACC, Metric.TEST_ACC, Metric.TRAIN_TIME, Metric.TEST_TIME]


class RollingAverage:
    def __init__(self, window_size=10):
        self.window_sz = window_size
        self.window = []

    def update(self, val):
        self.window.append(val)
        if len(self.window) > self.window_sz:
            del self.window[0]

    def latest(self):
        if len(self.window) == 0:
            return np.nan
        return self.window[-1]

    def mean(self):
        if len(self.window) == 0:
            return np.nan
        return np.mean(self.window)


def check_train_config(config: dict):
    """
    Validates the given config dict. Since we will use this config to create a unique hash to describe training, we
    require all variables to be present, even if they are optional. So this function will fill in missing variables.

    Args:
        config: The config describing how training will be done.
    Raises:
        RuntimeError: if the config is invalid.
    """

    def gt_zero(x):
        return x > 0

    def gte_zero(x):
        return x >= 0

    def _and(*args):
        def test_fn(val):
            return np.all([cond(val) for cond in args])

        return test_fn

    def of_type(types):
        def test_fn(val):
            return isinstance(val, types)

        return test_fn

    def one_of(options):
        def test_fn(val):
            return val in options

        return test_fn

    ensure_config_param(config, "verbose", _and(of_type(int), gte_zero), required=False)
    ensure_config_param(config, "save_checkpoints", of_type(bool), required=False)
    ensure_config_param(config, "eval_checkpoints", of_type(bool), required=False)
    ensure_config_param(config, "checkpoint_initial_model", of_type(bool), required=False)
    if config.get("save_checkpoints"):
        ensure_config_param(config, "save_dir", of_type((str, Path)))

    ensure_config_param(config, "train_config", of_type(dict))
    ensure_config_param(config, ["train_config", "benchmark"], of_type(str))
    ensure_config_param(config, ["train_config", "dataset"], of_type(str))
    ensure_config_param(config, ["train_config", "batch_size"], _and(of_type(int), gt_zero))
    ensure_config_param(config, ["train_config", "seed"], of_type(int))
    ensure_config_param(config, ["train_config", "train_method"], one_of(("lstsq", "sgd")), dflt="sgd")

    if config["train_config"]["train_method"] == "sgd":
        ensure_config_param(config, ["train_config", "epochs"], _and(of_type(int), gte_zero))
        ensure_config_param(config, ["train_config", "loss_fn"], of_type(str), "cross_entropy")
        ensure_config_param(config, ["train_config", "optimizer"], of_type(str), "Adam")
        ensure_config_param(config, ["train_config", "optimizer_args"], of_type(dict), {"lr": 0.1})
        ensure_config_param(config, ["train_config", "lr_scheduler"], of_type(str), required=False)
        ensure_config_param(config, ["train_config", "lr_scheduler_args"], of_type(dict), required=False)
        ensure_config_param(config, ["train_config", "max_grad_norm"], gte_zero, 0)

        ensure_config_param(config, ["train_config", "aux_losses"], of_type(list), required=False)
        for i, elem in enumerate(config["train_config"].get("aux_losses", [])):
            if not isinstance(elem, dict):
                raise RuntimeError('Each item in "aux_losses" should be a config with two elements.'
                                   f" Instead, we found a {type(elem)} in position {i}.")
            try:
                ensure_config_param(elem, "output", of_type(str))
                ensure_config_param(elem, "loss_fn", of_type(str))
                ensure_config_param(elem, "weight", gte_zero, required=False)
            except RuntimeError as e:
                raise RuntimeError(f'In position {i} of "aux_losses": {str(e)}')

    ensure_config_param(config, "metrics", of_type(list), [{"metric_fn": "accuracy"}])
    for i, elem in enumerate(config.get("metrics", [])):
        if not isinstance(elem, dict):
            raise RuntimeError('Each item in "metrics" should be a sub-config (a dict).'
                               f" Instead, we found a {type(elem)} in position {i}.")
        try:
            ensure_config_param(elem, "metric_fn", of_type(str))
            ensure_config_param(elem, "metric_fn_args", of_type(dict), required=False)
            ensure_config_param(elem, "output", of_type(str), required=False)
        except RuntimeError as e:
            raise RuntimeError(f'In position {i} of "metrics": {str(e)}')


def get_cache_lock_object(save_dir):
    return FileLock(save_dir.parent / (save_dir.name + ".lock"))


def validate_config(config):
    """
    Prints and validates the given training config. Throws an exception in the case of invalid or missing required
    values. Non-required missing values are filled in with their defaults; note that this modifies the config in-place.

    Args:
        config: A config dict which describes the hyperparams, dataset, etc. for training an architecture.
    Returns:
        The config after validation (the same instance as was passed in).
    """
    # Output config for reference. Do it before checking config to assist debugging.
    logging.info("\n---- Train Config ----\n" + yaml.dump(make_pretty(config)) + "----------------------")
    check_train_config(config)
    return config


def ensure_configs_match(config, config_dest):
    other_config = load_yaml(config_dest)
    if config != other_config:
        raise RuntimeError("Config does not match the one on disk! Is this a hash collision?\n"
                           f"Current config:\n{config}\n"
                           f"Config on disk:\n{other_config}\n"
                           f"Loaded from file: {config_dest}")


def prep_cache_folder(config: dict, root_path: Path):
    """
    Uses the given training config to derive a unique storage location for this config and its corresponding artifacts,
    unless the user has already provided one through the "save_dir" config param. If a new directory is derived, then
    the config will be modified with a new "save_dir" param. Saves the config in the resulting folder.

    Args:
        config: A config dict which describes the hyperparams, dataset, etc. for training an architecture.
        root_path: The directory under which all results should be stored. A subdirectory under this one will be
            determined by a unique hash of the config.
    Returns:
        `FileLock`: A lock object to synchronize reads/writes to the cache, as given by `get_cache_lock_object()`.
    """
    # Get a unique hash for this configuration. Use that to identify the cache file.
    hashed_config = config["train_config"].copy()
    # Remove variables which should not contribute to the uniqueness.
    hashed_config.pop("epochs", None)
    hashed_config.pop("max_steps", None)
    hashed_config = make_pretty(hashed_config, sort=True)

    # Derive a unique directory, unless the user provided their own.
    if config.get("save_dir"):
        # Convert existing directory to resolved path.
        save_dir = Path(config.get("save_dir")).resolve()
        logging.debug("Using save directory:\n" + str(save_dir) + "----------------------")
    else:
        save_dir = root_path / str(hashlib.shake_128(yaml.dump(hashed_config).encode()).hexdigest(8))
        logging.debug("Save dir derived from config:\n" + yaml.dump(hashed_config) + "----------------------")
    config["save_dir"] = save_dir

    # Print the final directory and ensure it exists.
    logging.info(f"Saving results at: {save_dir}")
    save_dir.mkdir(parents=True, exist_ok=True)

    # Write the config to the resulting folder. Do it in a multiprocess-safe way with a file lock.
    config_dest = save_dir / "train-config.yml"
    lock = get_cache_lock_object(save_dir)
    if not config_dest.exists():
        with lock:
            if not config_dest.exists():  # Do another check to protect against race conditions.
                with open(config_dest, "w") as f:
                    yaml.dump(make_pretty(hashed_config), f)
            else:
                ensure_configs_match(make_pretty(hashed_config), config_dest)
    else:
        ensure_configs_match(make_pretty(hashed_config), config_dest)
    return lock


def per_epoch_metrics(metrics_map):
    """ Transforms the output of `train()` into a mapping of (Metric enum) -> (trajectory over training epochs). """
    # Reformat: list of records --> DF per step
    records = []
    for step, record in metrics_map.items():
        records.append({"Step": step, **record})
    df = pd.DataFrame.from_records(records)

    # We allow for a missing 0th epoch (meaning "checkpoint_initial_model" was false). If missing, prepend a blank row
    # for the 0th step of the 0th epoch.
    if not (df["Epoch"] == 0).any():
        df = pd.concat([pd.DataFrame([[0, 0]], columns=["Epoch", "Step"]), df], ignore_index=True)

    # Reformat: one row per step --> one row per epoch
    steps = df.groupby("Epoch").agg({"Step": "max"})["Step"]
    per_epoch_metrics = df.iloc[df["Step"].isin(steps).tolist()]

    # Sanity check: make sure all epochs are present with no gaps.
    epoch_check = np.full(int(per_epoch_metrics["Epoch"].max() + 1), False, dtype=np.bool_)
    for e in per_epoch_metrics["Epoch"]:
        epoch_check[e] = True
    if not np.all(epoch_check):
        raise RuntimeError(f"Missing epochs: {list(np.argwhere(epoch_check == False).flatten())}")

    # Reformat: DF --> list per column. Final format should be `result[metric][epoch]`.
    result = {}
    for src, dst in LOG2METRIC.items():
        if src in per_epoch_metrics.columns:
            if dst in NO_NAN_COLS:
                # Sanity check: ensure no missing values in these columns, except potentially the 0th step.
                metric_nans = np.isnan(per_epoch_metrics.loc[1:, src])
                if np.any(metric_nans):
                    raise RuntimeError(f"Missing values for {dst.get_name()} at epochs: "
                                       f"{list(np.argwhere(metric_nans).flatten())}")
            result[dst] = per_epoch_metrics[src].tolist()
    return result


def filesafe_str(obj):
    if isinstance(obj, Mapping):
        return "-" + (",".join([filesafe_str(k) + ";" + filesafe_str(obj[k]) for k in obj])) + "-"
    elif isinstance(obj, (Sequence, Set)):
        return "-" + (",".join([filesafe_str(e) for e in obj])) + "-"
    else:
        return str(obj)


def filesafe_model_name(model):
    if not hasattr(model, "get_hash"):
        return ""
    return filesafe_str(model.get_hash()).strip("-")


def eval_model(model: torch.nn.Module, train_loader, valid_loaders, config, device="cuda", print_fn=None):
    """
    Evaluate the given model on the given data loaders, using metrics specified by the given config.

    Returns:
        dict: A mapping of resulting metrics averaged over each data loader.
    """
    # NOTE: The model will automatically be placed into eval mode by the config.
    model.to(device)
    metric_fns = metric_fns_from_config(config, model)
    loaders = {"Train": train_loader, **{k.capitalize(): v for k, v in valid_loaders.items()}}

    metrics = {}
    for key, loader in loaders.items():
        metrics[key] = overall_metrics(model, loader, metric_fns, device, print_fn=print_fn)
    return metrics


def as_training_metrics(metrics):
    """ Converts the metrics dict given by `eval_model` into a format similar to the one returned by `train`. """
    newmetrics = {}
    for split, splitrics in metrics.items():
        for mk, mv in splitrics.items():
            newmetrics[f"Overall/{split} {mk}"] = mv
    return newmetrics


def print_memory_stats(rank=None):
    import psutil
    import subprocess

    GB = 1024**3

    logging.info(f"Memory info for proc {rank if rank is not None else 'Main'}:")
    
    # System memory stats.
    meminfo = psutil.virtual_memory()
    logging.info(f"Total: {meminfo.total / GB:.2f} GB")
    logging.info(f"Avail: {meminfo.available / GB:.2f} GB")
    if hasattr(meminfo, "shared"):
        logging.info(f"Shared: {meminfo.shared / GB:.2f} GB")
    logging.info(f"Used: {meminfo.percent}%")
    logging.info("")
    
    # Process memory stats.
    process = psutil.Process()
    procinfo = process.memory_info()
    logging.info(f"Proc Resident Usage: {procinfo.rss / GB:.2f} GB")
    logging.info(f"Proc Virtual: {procinfo.vms / GB:.2f} GB")
    if hasattr(meminfo, "shared"):
        logging.info(f"Proc Shared: {procinfo.shared / GB:.2f} GB")
    logging.info("")


def cache_probe_inputs(pm, probes, loaders, device, show_progress=False, print_freq=0):
    """
    Run data from all the given `loaders` through the given `ProbedModule`, and cache all inputs as they arrive at each
    probe, along with the corresponding labels. These then can serve as a dataset for training each probe.

    Args:
        pm: The model to run; usually a `ProbedModule`, or a module which wraps a `ProbedModule`.
        probes: The dict of probes for the module, e.g. as returned by `ProbedModule.probes`.
        loaders: A dict from split name to `DataLoader`.
        device: The device to run inference on (cached inputs will be copied to CPU).
        show_progress: Whether to show a progress bar with `tqdm`.
        print_freq: If set, this will print progress every `print_freq` batches. If 0 (default), progress messages are
            not printed during the data loading loop.

    Returns:
        cached_inputs (dict): Cached input batches, dict[probe name -> dict[split -> batch list]]
        cached_labels (dict): Cached label batches, dict[split -> batch list]
    """
    import gc
    maybe_tqdm = tqdm if show_progress else (lambda x: x)
    cached_inputs = {pname: {split: [] for split in loaders} for pname in probes}
    cached_labels = {split: [] for split in loaders}

    with eval_mode(pm):
        for split, loader in loaders.items():
            logging.info(f"Running through {len(loader)} batches of {split} set...")
            data_time = RollingAverage(window_size=max(10, 10 * print_freq))
            batch_time = RollingAverage(window_size=max(10, 10 * print_freq))
            end = time.time()
            num_batches = len(loader)
            num_digits = int(np.log10(num_batches)) + 1
            for i, (ims, labels) in enumerate(maybe_tqdm(loader)):
                data_time.update(time.time() - end)

                cached_labels[split].append(labels.detach().cpu())
                _ = pm(ims.to(device))
                for pname, p in probes.items():
                    cached_inputs[pname][split].append(p.curr_input)

                batch_time.update(time.time() - end)
                end = time.time()
                gc.collect()
                torch.cuda.empty_cache()
                if print_freq > 0 and (i % print_freq == 0 or i == (len(loader) - 1)):
                    logging.info(f"({i+1: >{num_digits}}/{num_batches}) "
                                 f"Batch Time: {batch_time.latest():.3f}s ({batch_time.mean():.3f}s avg) | "
                                 f"Data Time: {data_time.latest():.3f}s ({data_time.mean():.3f}s avg) | ")
                    print_memory_stats(device)
                    actsize = sizeof((cached_inputs, cached_labels)) / 1024 / 1024 / 1024
                    logging.info(f"Size of cached activations: {actsize:.2f} GB")
                    # This diagnostic is the same every time so we only need to see it once.
                    if i == 0:
                        for pname, p in probes.items():
                            if hasattr(p, "curr_input") and p.curr_input is not None:
                                logging.info(f"    {pname} shape = {p.curr_input.shape}")

    actsize = sizeof((cached_inputs, cached_labels)) / 1024 / 1024 / 1024
    logging.info(f"Size of cached activations: {actsize:.2f} GB")
    return cached_inputs, cached_labels


def train_one_epoch(model, criterion, optimizer, data_loader, device, epoch, args, model_ema=None, scaler=None):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", utils.SmoothedValue(window_size=1, fmt="{value}"))
    metric_logger.add_meter("img/s", utils.SmoothedValue(window_size=10, fmt="{value}"))

    header = f"Epoch: [{epoch}]"
    for i, (image, target) in enumerate(metric_logger.log_every(data_loader, args.print_freq, header)):
        start_time = time.time()
        image, target = image.to(device), target.to(device)
        with torch.cuda.amp.autocast(enabled=scaler is not None):
            output = model(image)
            loss = criterion(output, target)

        optimizer.zero_grad()
        if scaler is not None:
            scaler.scale(loss).backward()
            if args.clip_grad_norm is not None:
                # we should unscale the gradients of optimizer's assigned params if do gradient clipping
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if args.clip_grad_norm is not None:
                nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad_norm)
            optimizer.step()

        if model_ema and i % args.model_ema_steps == 0:
            model_ema.update_parameters(model)
            if epoch < args.lr_warmup_epochs:
                # Reset ema buffer to keep copying weights during warmup period
                model_ema.n_averaged.fill_(0)

        acc1, acc5 = utils.accuracy(output, target, topk=(1, 5))
        batch_size = image.shape[0]
        metric_logger.update(loss=loss.item(), lr=optimizer.param_groups[0]["lr"])
        metric_logger.meters["acc1"].update(acc1.item(), n=batch_size)
        metric_logger.meters["acc5"].update(acc5.item(), n=batch_size)
        metric_logger.meters["img/s"].update(batch_size / (time.time() - start_time))


def evaluate(model, criterion, data_loader, device, print_freq=100, log_suffix=""):
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = f"Test: {log_suffix}"

    num_processed_samples = 0
    with torch.inference_mode():
        for image, target in metric_logger.log_every(data_loader, print_freq, header):
            image = image.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            output = model(image)
            loss = criterion(output, target)

            acc1, acc5 = utils.accuracy(output, target, topk=(1, 5))
            # FIXME need to take into account that the datasets
            # could have been padded in distributed setup
            batch_size = image.shape[0]
            metric_logger.update(loss=loss.item())
            metric_logger.meters["acc1"].update(acc1.item(), n=batch_size)
            metric_logger.meters["acc5"].update(acc5.item(), n=batch_size)
            num_processed_samples += batch_size
    # gather the stats from all processes

    num_processed_samples = utils.reduce_across_processes(num_processed_samples)
    if (
        hasattr(data_loader.dataset, "__len__")
        and len(data_loader.dataset) != num_processed_samples
        and torch.distributed.get_rank() == 0
    ):
        # See FIXME above
        warnings.warn(
            f"It looks like the dataset has {len(data_loader.dataset)} samples, but {num_processed_samples} "
            "samples were used for the validation, which might bias the results. "
            "Try adjusting the batch size and / or the world size. "
            "Setting the world size to 1 is always a safe bet."
        )

    metric_logger.synchronize_between_processes()

    print(f"{header} Acc@1 {metric_logger.acc1.global_avg:.3f} Acc@5 {metric_logger.acc5.global_avg:.3f}")
    return metric_logger.acc1.global_avg


def train(model: torch.nn.Module, train_loader, valid_loaders, config, device="cuda"):
    """
    Fully train the given model, on the given dataset, according to the given config.

    Returns:
        dict: A mapping from step # -> metrics.
    """
    logging.info(f"Model to train:\n{model}\n")
    if hasattr(model, "modules_str"):
        logging.info(f"Model shape:\n{model.modules_str()}")
    train_config = config["train_config"]
    model.to(device)

    # Set up progress/checkpoint logger. If double-verbose, print every step. Else, print a few times per epoch.
    once_per_epoch = len(train_loader)
    print_freq = 100 if config.get("verbose", 0) <= 1 else 1
    save_freq = once_per_epoch if config.get("save_checkpoints") else 0
    eval_freq = once_per_epoch if config.get("eval_checkpoints") else 0
    metric_fns = metric_fns_from_config(config, model)
    log = StandardLog(model, metric_fns, print_freq=print_freq, save_freq=save_freq, eval_freq=eval_freq,
                      save_dir=config.get("save_dir"), model_name=filesafe_model_name(model),
                      checkpoint_initial_model=config.get("checkpoint_initial_model", True))
    log.begin(model, train_loader, valid_loaders, device)

    optimizer = optimizer_from_config(train_config, model.parameters())
    scheduler = scheduler_from_config(train_config, optimizer)
    loss_fns = loss_fns_from_config(train_config, model)
    max_grad_norm = train_config["max_grad_norm"]

    # BEGIN TRAINING
    step = 1
    max_steps = train_config.get("max_steps", float("inf"))
    max_epochs = train_config["epochs"] + 1
    for epoch in range(1, max_epochs):  # Epoch/step counts will be 1-based.
        step = run_one_epoch(model, train_loader, valid_loaders, optimizer, loss_fns, log, epoch, step, max_steps,
                             max_grad_norm=max_grad_norm, device=device)
        if step > max_steps:
            break
        scheduler.step()

    return log.close(min(step - 1, max_steps), min(epoch, max_epochs), model, train_loader, valid_loaders, device,
                     bool(config.get("eval_checkpoints")), bool(config.get("save_checkpoints")))


def run_one_epoch(model, train_loader, valid_loaders, optimizer, loss_fns, log, epoch, step, max_steps=float("inf"),
                  opt_params=None, max_grad_norm=0, device=None):
    """ Run one training epoch. """
    log.begin_epoch(step, epoch, model, train_loader, valid_loaders, optimizer, device)

    # Only optimize the given layers during this epoch.
    saved_opt_state = None
    if opt_params:
        saved_opt_state = limit_model_optimization(model, opt_params)

    model.train()
    for images, labels in train_loader:
        run_one_step(images, labels, model, optimizer, loss_fns, log, epoch, step, max_grad_norm, device)
        step += 1
        if step > max_steps:
            break

    # Reset the opt state.
    if opt_params:
        restore_grad_state(model, saved_opt_state)

    log.end_epoch(step - 1, epoch, model, train_loader, valid_loaders, optimizer, device)
    return step


def run_one_step(images, labels, model, optimizer, loss_fns, log, epoch, step, max_grad_norm=0, device=None):
    # Move data to GPU once loaded.
    images, labels = images.to(device), labels.to(device)

    # Forward pass.
    out, losses, total_weight = forward_pass(model, images, labels, loss_fns)
    loss = sum(losses.values()) / total_weight  # Normalize, or else this could grow to be unstable.

    # Backpropagate.
    optimizer.zero_grad()
    loss.backward()
    if max_grad_norm > 0:
        clip_grad_norm_(model.parameters(), max_grad_norm)
    optimizer.step()

    # Record accuracy and other metrics. Do this after the backward pass in case we want to record gradients or
    # save the latest model.
    log.step(step, epoch, loss, out, labels, model, all_losses=losses)


def forward_pass(model, ims, labels, loss_fns):
    total_weight = torch.Tensor([w for _, w in loss_fns.values()]).sum()
    if len(ims) == 0:
        out = torch.tensor([])
        losses = {name: torch.nan for name, loss_fn in loss_fns.items()}
    else:
        out = model(ims)
        losses = {name: loss_fn(out, labels) * weight for name, (loss_fn, weight) in loss_fns.items()}
    return out, losses, total_weight
