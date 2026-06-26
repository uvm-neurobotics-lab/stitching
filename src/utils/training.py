"""
Functions to support training networks in ways that can form the basis of a NAS benchmark.
"""
import itertools
import logging
import warnings
from collections.abc import Mapping, Sequence, Set
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Union

import numpy as np
import pandas as pd
import torch
import yaml
from torch.nn.utils import clip_grad_norm_

try:
    import wandb
except ImportError:
    wandb = None

from utils import ensure_config_param, make_pretty, restore_grad_state, gte_zero, _and, of_type, one_of
from utils.logging import StandardLog, eval_mode
from utils.optimization import (check_aux_loss_config, check_metrics_config, limit_model_optimization,
                                loss_fns_from_config, metric_fns_from_config, optimizer_from_config,
                                scheduler_from_config)
from utils.models import TaskSlice


@dataclass
class TaskInfo:
    name: str
    model: torch.nn.Module
    loader: torch.utils.data.DataLoader
    loss_fns: dict[str, Callable]
    loss_scale: float = 1.0


def check_train_config(config: dict):
    """
    Validates the given config dict. Since we will use this config to create a unique hash to describe training, we
    require all variables to be present, even if they are optional. So this function will fill in missing variables.

    Args:
        config: The config describing how training will be done.
    Raises:
        RuntimeError: if the config is invalid.
    """

    ensure_config_param(config, "verbose", _and(of_type(int), gte_zero), required=False)
    ensure_config_param(config, "save_checkpoints", of_type(bool), required=False)
    ensure_config_param(config, "eval_checkpoints", of_type(bool), required=False)
    ensure_config_param(config, "checkpoint_initial_model", of_type(bool), required=False)
    ensure_config_param(config, "resume_from", of_type((str, Path)), required=False)
    ensure_config_param(config, "load_from", of_type((str, Path)), required=False)
    ensure_config_param(config, "strict_load", of_type(bool), required=False)
    ensure_config_param(config, "test_only", of_type(bool), required=False)
    # If we are only testing, then weights must be supplied by one of these two methods.
    if config.get("test_only") and not ("resume_from" in config or "load_from" in config):
        warnings.warn(f"We are testing the INITIAL model, but you have not provided weights via the 'load_from' or "
                      "'resume_from' arguments.")
    if "resume_from" in config and "load_from" in config:
        raise RuntimeError(f"'load_from' and 'resume_from' are mutually exclusive. Supply only one.")
    ensure_config_param(config, "start_epoch", _and(of_type(int), gte_zero), dflt=0)
    ensure_config_param(config, "save_dir", of_type((str, Path)), required=config.get("save_checkpoints"))
    if "save_dir" in config:
        config["save_dir"] = Path(config["save_dir"]).expanduser().resolve()  # ensure type is Path, not string.

    ensure_config_param(config, "train_config", of_type(dict))
    ensure_config_param(config, ["train_config", "seed"], of_type(int))

    ensure_config_param(config, ["train_config", "epochs"], _and(of_type(int), gte_zero))
    ensure_config_param(config, ["train_config", "max_steps"], _and(of_type(int), gte_zero), required=False)
    ensure_config_param(config, ["train_config", "loss_fn"], of_type(str), "cross_entropy")
    # TODO: Improve opt/sched config to look cleaner like assembly config.
    ensure_config_param(config, ["train_config", "optimizer"], of_type(str), "Adam")
    ensure_config_param(config, ["train_config", "optimizer_args"], of_type(dict), {"lr": 0.1})
    ensure_config_param(config, ["train_config", "lr_scheduler"], of_type(str), required=False)
    ensure_config_param(config, ["train_config", "lr_scheduler_args"], of_type(dict), required=False)
    ensure_config_param(config, ["train_config", "lr_scheduler_args", "cadence"], one_of(["epochs", "steps"]),
                        required=False)
    ensure_config_param(config, ["train_config", "max_grad_norm"], gte_zero, 0)

    check_aux_loss_config(config["train_config"])
    check_metrics_config(config)


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


def per_epoch_metrics(metrics_map, **metadata):
    """
    Transforms the output of `train()` into a dataframe recording all metrics at the end of each training epoch.
    Metrics are columns and epochs are rows. Kwarg inputs to the function are prepended as metadata to each row.
    """
    # Reformat: list of records --> DF per step
    records = []
    for step, record in metrics_map.items():
        records.append({**metadata, "Step": step, **record})
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

    return per_epoch_metrics


def filesafe_str(obj):
    if isinstance(obj, Mapping):
        return ",".join([filesafe_str(k) + ";" + filesafe_str(obj[k]) for k in obj])
    elif isinstance(obj, (Sequence, Set)) and not isinstance(obj, str):
        return ",".join([filesafe_str(e) for e in obj])
    else:
        return str(obj)


def filesafe_model_name(model):
    if not hasattr(model, "get_hash"):
        return ""
    return filesafe_str(model.get_hash()).strip("-")


def as_training_metrics(metrics):
    """ Converts the metrics dict given by `eval_model` into a format similar to the one returned by `train`. """
    newmetrics = {}
    for split, splitrics in metrics.items():
        for mk, mv in splitrics.items():
            newmetrics[f"Overall/{split} {mk}"] = mv
    return newmetrics


def print_memory_stats(rank=None):
    import psutil

    GB = 1024 ** 3

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


def compute_loss_scales(task_infos: list[TaskInfo], device: Union[str, int, torch.device], num_batches: int = 3):
    """Compute initial loss magnitudes for each task to use as normalization denominators."""
    if len(task_infos) == 1:
        return

    for task_info in task_infos:
        with eval_mode(task_info.model):
            total = 0.0
            count = 0
            for images, labels in itertools.islice(task_info.loader, num_batches):
                if len(images) == 0:
                    continue
                images, labels = images.to(device), labels.to(device)
                _, losses, total_weight = forward_pass(task_info.model, images, labels, task_info.loss_fns)
                total += (sum(losses.values()) / total_weight).item()
                count += 1
            task_info.loss_scale = total / count if count > 0 else 1.0


def train(config, model, train_loaders, valid_loaders, train_sampler, device):
    train_config = config["train_config"]
    model.to(device)

    # If only a single train loader was passed in, then we're doing single-task training with an empty task name.
    if not isinstance(train_loaders, dict):
        train_loaders = {"": train_loaders}

    # Setup the optimization.
    if config.get("deterministic"):
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True)
    else:
        torch.backends.cudnn.benchmark = True

    optimizer = optimizer_from_config(train_config, model.parameters())
    scheduler, sched_cadence = scheduler_from_config(train_config, optimizer)
    max_grad_norm = train_config["max_grad_norm"]

    # Set up distributed training and checkpointing behavior.
    model_without_ddp = model
    if config["distributed"]:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[device])
        model_without_ddp = model.module

    # Load model checkpoint if requested.
    if config.get("resume_from"):
        logging.info(f"Resuming checkpoint at {config['resume_from']}.")
        checkpoint = torch.load(config["resume_from"], map_location="cpu", weights_only=True)
        model_without_ddp.load_state_dict(checkpoint["model"], config.get("strict_load", True))
        if not config.get("test_only"):
            optimizer.load_state_dict(checkpoint["optimizer"])
            scheduler.load_state_dict(checkpoint["scheduler"])
        config["start_epoch"] = checkpoint["epoch"] + 1
    elif config.get("load_from"):
        logging.info(f"Resuming checkpoint at {config['load_from']}.")
        checkpoint = torch.load(config["load_from"], map_location="cpu", weights_only=True)
        model_without_ddp.load_state_dict(checkpoint["model"], config.get("strict_load", True))

    # Build per-task info structures.
    task_configs = train_config.get("tasks", [{}])
    if len(task_configs) != len(train_loaders):
        raise RuntimeError(f"Task configs (len={len(task_configs)}):\n{task_configs}\n"
                           f"do not match train loaders (len={len(train_loaders)}):\n{train_loaders}")
    # IMPORTANT: This assumes train loaders are in the same order as task configs.
    task_infos = []
    for i, ((name, loader), task_cfg) in enumerate(zip(train_loaders.items(), task_configs)):
        if task_cfg:
            # This merging of configs allows us to fall back to global loss configuration if the task doesn't have
            # its own specific configuration.
            cfg = dict(train_config)
            cfg.update(task_cfg)
            # loss_fns_from_config gets the non-DDP model so aux-loss hooks attach to the right module.
            task_loss_fns = loss_fns_from_config(cfg, TaskSlice(model_without_ddp, i))
            # If doing multi-task training, we expect model output to be a tuple w/ index i corresponding to dataset i.
            # So we wrap it in a TaskSlice for each task.
            task_model = TaskSlice(model, i)
        else:
            # We're not doing multi-task training, so don't use the task config.
            task_loss_fns = loss_fns_from_config(train_config, model_without_ddp)
            task_model = model

        task_infos.append(TaskInfo(name, task_model, loader, task_loss_fns))

    # If doing multi-task training, compute per-task loss scales from the first few batches (for normalization).
    if len(task_infos) > 1:
        compute_loss_scales(task_infos, device)
        logging.info(f"Per-task loss scales: { {t.name: f'{t.loss_scale:.4f}' for t in task_infos} }")

    # Set up progress/checkpoint logger.
    max_steps = train_config.get("max_steps", float("inf"))
    max_epochs = train_config["epochs"]
    steps_per_epoch = max(len(ldr) for ldr in train_loaders.values())
    expected_steps = min(max_steps, max_epochs * steps_per_epoch)
    metric_fns = metric_fns_from_config(config, model)
    once_per_epoch = steps_per_epoch
    print_freq = config.get("print_freq", 10) if config.get("verbose", 0) <= 1 else 1
    save_freq = once_per_epoch if config.get("save_checkpoints") else 0
    if max_epochs > 30:
        # TODO: Should make this configurable or come up with a better heuristic, but this works for now.
        save_freq *= 10
    eval_freq = once_per_epoch if config.get("eval_checkpoints") else 0
    log = StandardLog(model, expected_steps, metric_fns, task_names=[t.name for t in task_infos],  # FIXME: not sure we need to store this
                      print_freq=print_freq, save_freq=save_freq,
                      eval_freq=eval_freq, save_dir=config.get("save_dir"), model_name=filesafe_model_name(model),
                      use_wandb=wandb is not None, checkpoint_initial_model=config.get("checkpoint_initial_model",
                                                                                       config.get("save_checkpoints")))

    if config.get("test_only"):
        # We disable the cudnn benchmarking because it can noticeably affect the accuracy.
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        return log.close(0, 0, model, train_loaders, valid_loaders, optimizer, scheduler, config, device,
                         should_eval=True, should_save=False)

    # BEGIN TRAINING
    step = 1
    log.begin(model, train_loaders, valid_loaders, optimizer, scheduler, config, device)

    for epoch in range(config.get("start_epoch") + 1, max_epochs + 1):  # Epoch/step counts will be 1-based.
        if config["distributed"] and train_sampler is not None:
            train_sampler.set_epoch(epoch)
        step = run_one_epoch(model, task_infos, valid_loaders, optimizer, scheduler, sched_cadence, config, log, epoch,
                             step, max_steps, max_grad_norm=max_grad_norm, device=device)
        if step > max_steps:
            break
        if sched_cadence == "epochs":
            scheduler.step()

    return log.close(min(step - 1, max_steps), min(epoch, max_epochs), model, train_loaders, valid_loaders, optimizer,
                     scheduler, config, device, bool(config.get("eval_checkpoints")),
                     bool(config.get("save_checkpoints")))


def run_one_epoch(model, task_infos, valid_loaders, optimizer, scheduler, sched_cadence, config, log, epoch, step,
                  max_steps=float("inf"), opt_params=None, max_grad_norm=0, device=None):
    """ Run one training epoch. """
    steps_per_epoch = max(len(t.loader) for t in task_infos)
    task_loaders = {t.name: t.loader for t in task_infos}
    log.begin_epoch(step, epoch, model, task_loaders, valid_loaders, optimizer, device)

    # Only optimize the given layers during this epoch.
    saved_opt_state = None
    if opt_params:
        saved_opt_state = limit_model_optimization(model, opt_params)

    # Align loaders: cycle shorter ones so every step has a batch from every task.
    aligned_iters = [itertools.islice(itertools.cycle(t.loader), steps_per_epoch) for t in task_infos]

    model.train()
    for batches in zip(*aligned_iters):
        log.begin_step(step, epoch, optimizer)
        total_loss, losses, outs, labels = run_one_step(task_infos, batches, optimizer, max_grad_norm, device)
        if sched_cadence == "steps":
            scheduler.step()
        log.end_step(step, epoch, total_loss, outs, labels, model, all_losses=losses)
        step += 1
        if step > max_steps:
            break

    # Reset the opt state.
    if opt_params:
        restore_grad_state(model, saved_opt_state)

    log.end_epoch(step - 1, epoch, model, task_loaders, valid_loaders, optimizer, scheduler, config, device)
    return step


def run_one_step(task_infos, batches, optimizer, max_grad_norm=0, device=None):
    """Unified single/multi-task training step. Accumulates gradients across all tasks before stepping."""
    optimizer.zero_grad()
    all_losses = {}
    all_outs = []
    all_labels = []

    for task, batch in zip(task_infos, batches):
        images, labels = batch[0].to(device), batch[1].to(device)
        out, losses, total_weight = forward_pass(task.model, images, labels, task.loss_fns)
        (sum(losses.values()) / (total_weight * task.loss_scale)).backward()  # accumulates into .grad buffers
        prefix = f"{task.name}/" if task.name else ""
        all_losses.update({f"{prefix}{k}": v for k, v in losses.items()})
        all_outs.append(out.detach())
        all_labels.append(labels)

    # Only clip grads and taken an optimizer step after processing all task batches.
    if max_grad_norm > 0:
        all_params = itertools.chain.from_iterable(g["params"] for g in optimizer.param_groups)
        clip_grad_norm_(all_params, max_grad_norm)
    optimizer.step()

    total_loss = sum(all_losses.values()) / len(all_outs)
    return total_loss, all_losses, all_outs, all_labels


def forward_pass(model, ims, labels, loss_fns):
    total_weight = torch.Tensor([w for _, w in loss_fns.values()]).sum()
    if len(ims) == 0:
        out = torch.tensor([])
        losses = {name: torch.nan for name, loss_fn in loss_fns.items()}
    else:
        out = model(ims)
        losses = {name: loss_fn(out, labels) * weight for name, (loss_fn, weight) in loss_fns.items()}
    return out, losses, total_weight
