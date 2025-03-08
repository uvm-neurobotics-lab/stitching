"""
Utilities for logging progress metrics and saving checkpoints.

This API is admittedly an organically grown mess and ought to be refactored in the next version. The MetricLogger
from the Torchvision distributed training example is a good one:
https://github.com/pytorch/vision/blob/124dfa404f395db90280e6dd84a51c50c742d5fd/references/classification/utils.py#L69
Some of the ingredients below was already taken from there.
"""
import datetime
import logging
import re
import warnings
from collections import defaultdict, deque
from time import time, strftime, gmtime

import torch

try:
    import wandb
except ImportError:
    wandb = None

import utils.distributed as dist


class SmoothedValue:
    """
    Track a series of values and provide access to:
        - smoothed values over a window in the local process.
        - the global series average over all processes.
    """

    def __init__(self, window_size=20, fmt=None):
        if fmt is None:
            # Default format: <recent value> (<all-time avg>)
            fmt = "{median:.4f} ({global_avg:.4f})"
        self.fmt = fmt
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.global_total = 0.0
        self.global_count = 0

    def update(self, value, n=1):
        self.deque.append(value)
        self.count += n
        self.total += value * n
        self.global_count += n
        self.global_total += value * n

    def synchronize_between_processes(self):
        """Warning: does not synchronize the deque!"""
        t = dist.reduce_across_processes([self.count, self.total])
        t = t.tolist()
        self.global_count = int(t[0])
        self.global_total = t[1]

    @property
    def median(self):
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.mean().item()

    @property
    def global_avg(self):
        if self.global_count == 0:
            return float("nan")
        return self.global_total / self.global_count

    @property
    def max(self):
        return max(self.deque)

    @property
    def value(self):
        return self.deque[-1]

    def __str__(self):
        return self.fmt.format(
            median=self.median, avg=self.avg, global_avg=self.global_avg, max=self.max, value=self.value
        )


def log_gradient_stats(metrics, model, name=None):
    if name and not name.endswith("/"):
        name += "/"
    elif name is None:
        name = ""

    params = [p for p in model.parameters() if p.grad is not None]
    if not params:
        return

    norm_type = 2
    device = params[0].grad.device
    # Computed the same way as in torch.nn.utils.clip_grad_norm_().
    norm = torch.norm(torch.stack([torch.norm(p.grad.detach(), norm_type).to(device) for p in params]), norm_type)
    metrics["gradients/" + name + "norm"] = norm


class eval_mode(torch.inference_mode):
    """
    Context-manager that sets Torch in Inference Mode (`torch.inference_mode()`) and also puts a module in eval mode
    (`torch.nn.Module.eval()`).
    """

    def __init__(self, module):
        super().__init__()
        self.module = module
        self.prev_train = False

    def __new__(cls, module):
        return super().__new__(cls)

    def __enter__(self):
        super().__enter__()
        self.prev_train = self.module.training
        self.module.train(False)

    def __exit__(self, exc_type, exc_value, traceback):
        super().__exit__(exc_type, exc_value, traceback)
        self.module.train(self.prev_train)


def overall_metrics(model, data_loader, header, metric_fns, device, delimiter="\t", print_fn=None, print_freq=10):
    """
    Evaluate the model on each batch and return the average over all samples for all given metric functions. Metric
    functions should follow the signature:
        (output: Tensor, labels: Tensor) -> dict[metric_name: str, metric_value: float]
    This returns a dict containing all metrics. If there are no batches, this returns an empty dictionary. If all
    batches are empty, this may return a dict populated with NaNs.
    """
    # Allow the tensors to be empty.
    nbatches = len(data_loader)
    if nbatches == 0:
        return {}

    if print_fn:
        print_fn(f"Computing accuracy of {nbatches} batches from {header}...")

    metric_values = defaultdict(SmoothedValue)
    num_processed_samples = 0
    start = time()
    end = time()
    iter_time = SmoothedValue(fmt="{avg:.4f}")
    data_time = SmoothedValue(fmt="{avg:.4f}")
    with eval_mode(model):
        for i, (ims, labels) in enumerate(data_loader):
            data_time.update(time() - end)
            # ntraft: Not sure why we do non-blocking here, but that's what the PyTorch example did.
            ims, labels = ims.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            out = model(ims)

            batch_size = len(labels)
            num_processed_samples += batch_size
            for metric in metric_fns:
                md = metric(out, labels)
                for k, v in md.items():
                    metric_values[k].update(v, batch_size)

            iter_time.update(time() - end)
            if i % print_freq == 0:
                log_msg = [
                    f"{header} Eval",
                    f"[{i:{str(len(str(nbatches)))}d}/{nbatches}]",
                    f"Iter Time: {iter_time}",
                    f"Data Time: {data_time}",
                ]
                if torch.cuda.is_available():
                    log_msg.append(f"Max Mem: {torch.cuda.max_memory_allocated() / 1024.0 / 1024.0:.0f} MB")
                print_fn(delimiter.join(log_msg))

    num_processed_samples = dist.reduce_across_processes(num_processed_samples)
    if (hasattr(data_loader.dataset, "__len__")
            and len(data_loader.dataset) != num_processed_samples
            and dist.is_main_process()):
        warnings.warn(
            f"It looks like the dataset has {len(data_loader.dataset)} samples, but {num_processed_samples} samples "
            "were used for the validation, which might bias the results. Try adjusting the batch size and / or the "
            "world size. Setting the world size to 1 is always a safe bet."
        )

    # NOTE: This synchronization relies on dictionaries maintaining insertion order, Python 3.7+.
    for meter in metric_values.values():
        meter.synchronize_between_processes()
    result = {k: meter.global_avg for k, meter in metric_values.items()}

    end = time()
    result["Time/Eval Total"] = end - start

    return result


class BaseLog:

    def __init__(self, metric_fns, metrics_to_print, eval_freq, save_freq, save_dir=None, model_name="",
                 use_wandb=False, checkpoint_initial_model=True, eval_full_train_set=False, print_delimiter="\t"):
        self.metric_fns = metric_fns
        self.metrics_to_print = metrics_to_print
        if self.metrics_to_print:
            available_metrics = list(metric_fns) if metric_fns is not None else []
            for k in self.metrics_to_print:
                if k not in available_metrics:
                    raise ValueError(f'Metric "{k}" not found in available metrics: {available_metrics}')
        self.save_freq = save_freq
        self.eval_freq = eval_freq
        self.save_dir = save_dir
        if self.save_dir:
            self.save_dir.mkdir(parents=True, exist_ok=True)
        elif self.save_freq > 0:
            raise ValueError(f"Requested saving of checkpoints but no save location provided.")
        self.model_name = "model"
        if model_name:
            self.model_name += "-" + model_name
            if re.search(r'[^A-Za-z0-9+=_\-.,; ]', model_name):
                raise ValueError(f'This model name is not ideal for use as a filename: "{model_name}"')
        # Only the main process should report to W&B.
        self.use_wandb = use_wandb and dist.is_main_process()
        if use_wandb and not wandb:
            raise RuntimeError("You are trying to use Weights & Biases but it is not installed. You must either"
                               " install it or not request it.")
        self.checkpoint_initial_model = checkpoint_initial_model
        self.eval_full_train_set = eval_full_train_set
        self.delimiter = print_delimiter
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self.recorded_metrics = defaultdict(dict)
        self.recorded_counts = {}
        self.smoothed_metrics = defaultdict(SmoothedValue)
        self.last_save_step = -1
        self.last_eval_step = -1
        self.start_time = None
        self.train_time = 0

    def warning(self, msg):
        self.logger.warning(msg)

    def info(self, msg):
        self.logger.info(msg)

    def debug(self, msg):
        self.logger.debug(msg)

    def record(self, metrics, it, batch_size=None):
        self.recorded_metrics[it].update(metrics)
        if batch_size is not None:
            self.recorded_counts[it] = batch_size

        for k, v in metrics.items():
            self.smoothed_metrics[k].update(v, batch_size if batch_size else 1)

        if self.use_wandb:
            # Turn the keys into strings for Weights and Biases.
            wmetrics = {k.get_name(): v for k, v in metrics.items()}
            wandb.log(wmetrics, step=it)

    def begin(self, model, train_loader, valid_loaders, optimizer, scheduler, config, device):
        self.start_time = time()
        if self.checkpoint_initial_model and (self.save_freq > 0 or self.eval_freq > 0):
            self.maybe_save_and_eval(0, 0, model, train_loader, valid_loaders, optimizer, scheduler, config, device)

    @torch.inference_mode()
    def maybe_save_and_eval(self, it, epoch, model, train_loader, valid_loaders, optimizer, scheduler, config, device,
                            should_eval=None, should_save=None):
        # If the caller has made a particular request, honor it. Otherwise...
        if should_save is None:
            # Turn on saving if it is time to save.
            should_save = self.save_freq > 0 and it % self.save_freq == 0
        # But do not save if the model for this iteration was already saved (the same iteration can be called twice).
        should_save &= (it != self.last_save_step)
        # If the caller has made a particular request, honor it. Otherwise...
        if should_eval is None:
            # Eval if it is time to eval or save.
            should_eval = should_save or (self.eval_freq > 0 and it % self.eval_freq == 0)
        should_eval &= (it != self.last_eval_step)  # but not if we already did it.

        metrics = {"Epoch": epoch}

        # Run a test on the full dataset.
        if should_eval:
            self.last_eval_step = it
            if self.eval_full_train_set:
                loaders = {"Train": train_loader, **{k.capitalize(): v for k, v in valid_loaders.items()}}
            else:
                loaders = {k.capitalize(): v for k, v in valid_loaders.items()}
            self.info(f"Checkpoint {it} Performance:")
            for key, loader in loaders.items():
                metric_dict = overall_metrics(model, loader, key, self.metric_fns, device, self.delimiter,
                                              print_fn=self.debug)
                metric_str = ""
                for mk, mv in metric_dict.items():
                    if mk == "Time/Eval Total":
                        continue  # Special print for this one.
                    levels = mk.split("/", maxsplit=2)
                    if len(levels) == 1:
                        levels = ["Overall", mk]
                    metrics[f"{levels[0]}/{key} {levels[1]}"] = mv
                    if (not self.metrics_to_print) or (mk in self.metrics_to_print):
                        metric_str += f"{mk} = {mv:.3f} | "
                if metric_str:
                    # TODO: tab separate instead?
                    metric_str = metric_str[:-2]  # remove ending separator: "| "
                self.info(f"    {key} {metric_str}"
                          f"(Time to Eval = {strftime('%H:%M:%S', gmtime(metric_dict['Time/Eval Total']))})")

        # Save the model.
        if should_save and dist.is_main_process():
            self.last_save_step = it
            save_path = self.save_dir / f"{self.model_name}-{it}.pth"
            self.info(f"Saving model to: {save_path}")
            checkpoint = {
                # TODO: Do we need this to be the "model without ddp"?
                #     (model.module if isinstance(model, torch.nn.parallel.DistributedDataParallel) else model)
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "epoch": epoch,
                "config": config,
            }
            dist.save_on_master(checkpoint, save_path)
            dist.save_on_master(checkpoint, self.save_dir / "checkpoint.pth")

        # Keep track of total time elapsed during training.
        if self.start_time is not None:
            metrics["Time/Total"] = time() - self.start_time

        self.record(metrics, it)

    def close(self, it, epoch, model, train_loader, valid_loaders, optimizer, scheduler, config, device,
              should_eval=True, should_save=True):
        self.maybe_save_and_eval(it, epoch, model, train_loader, valid_loaders, optimizer, scheduler, config, device,
                                 should_eval, should_save)
        if self.start_time is not None:
            total_time_str = str(datetime.timedelta(seconds=int(time() - self.start_time)))
            self.info(f"Training Complete. Time: {total_time_str}")
        return self.recorded_metrics


class StandardLog(BaseLog):

    def __init__(self, model, expected_steps, metric_fns, metrics_to_print=tuple(), print_freq=50, save_freq=256,
                 eval_freq=256, save_dir=None, model_name="", use_wandb=False, checkpoint_initial_model=True,
                 eval_full_train_set=False, log_gradients=False, print_delimiter="\t"):
        super().__init__(metric_fns, metrics_to_print, eval_freq, save_freq, save_dir, model_name, use_wandb,
                         checkpoint_initial_model, eval_full_train_set, print_delimiter)
        self.expected_steps = expected_steps
        self.steps_in_epoch = 0
        self.epoch_start_step = -1
        self.epoch_start_time = -1
        self.step_start_time = -1
        self.step_end_time = -1
        self.print_freq = print_freq
        self.log_gradients = log_gradients
        if use_wandb and self.log_gradients:
            wandb.watch(model, log_freq=print_freq)  # log gradient histograms automatically

    def begin_epoch(self, it, epoch, model, train_loader, valid_loaders, optimizer, device):
        self.record({"lr": optimizer.param_groups[0]["lr"]}, it)  # NOTE: assumes only one param group for now.
        self.info(f"---- Beginning Epoch {epoch} ({len(train_loader)} batches) ----")
        self.epoch_start_step = it
        self.epoch_start_time = self.step_end_time = time()
        self.steps_in_epoch = len(train_loader)

    def end_epoch(self, it, epoch, model, train_loader, valid_loaders, optimizer, scheduler, config, device):
        self.info(f"---------------------- End Epoch {epoch} ----------------------")

        # If the eval function isn't going to recompute performance on train, then compile a summary here instead.
        if not self.eval_full_train_set:
            self.info("Summary of per-batch metrics:")
            # NOTE: This relies on dictionaries maintaining insertion order, Python 3.7+.
            summaries = defaultdict(SmoothedValue)
            for step in range(self.epoch_start_step, it + 1):
                if step in self.recorded_metrics:
                    for k, v in self.recorded_metrics[step].items():
                        if k in ("Epoch", "Train Time", "lr"):  # Don't accumulate these metrics.
                            continue
                        if k.startswith("Batch "):  # Turn "Batch" metrics into "Overall" metrics.
                            k = k[6:]
                        summaries[k].update(v, self.recorded_counts.get(step, 1))
            for mname, val in summaries.items():
                val.synchronize_between_processes()
                self.record({"Overall/Train " + mname: val.global_avg}, it)
                if (not self.metrics_to_print) or (mname in self.metrics_to_print):
                    self.info(f"    Train {mname}: {val.global_avg:.3f}")

        self.maybe_save_and_eval(it, epoch, model, train_loader, valid_loaders, optimizer, scheduler, config, device)

        self.record({"Time/Per Epoch": time() - self.epoch_start_time}, it)

    def begin_step(self, it, epoch):
        self.step_start_time = time()
        self.record({"Time/Data": self.step_start_time - self.step_end_time}, it)

    def end_step(self, it, epoch, loss, out, labels, model, all_losses=None, metric_fns=None):
        if not all_losses:
            all_losses = {}
        if not metric_fns:
            metric_fns = self.metric_fns

        # Compute metrics.
        # TODO: may want to separate out metrics which are independent of batch size. Epoch, loss, img per sec, max mem
        metrics = {"Epoch": epoch, "Loss": loss.item(), **{k: v.item() for k, v in all_losses.items()}}
        for metric in metric_fns:
            md = metric(out, labels)
            for mk, mv in md.items():
                metrics["Batch " + mk] = mv
        if self.use_wandb and self.log_gradients:
            log_gradient_stats(metrics, model)

        # Track runtime & memory performance.
        batch_size = len(labels)
        metrics["Time/Step"] = time() - self.step_end_time
        metrics["Time/Img Per Sec"] = batch_size / (time() - self.step_start_time)
        if torch.cuda.is_available():
            metrics["Max Mem"] = torch.cuda.max_memory_allocated() / 1024.0 / 1024.0

        # Record metrics
        self.record(metrics, it, batch_size)

        # Finally, print metrics periodically.
        time_to_print = (it % self.print_freq == 0 or it == 1)  # Force print on the first step.
        if time_to_print:
            time_per_step = (time() - self.start_time) / it
            eta = time_per_step * (self.expected_steps - it)
            msg = [f"Epoch {epoch}",
                   f"[{it - self.epoch_start_step + 1:{str(len(str(self.steps_in_epoch)))}d}/{self.steps_in_epoch}]",
                   f"ETA: {datetime.timedelta(seconds=int(eta))}",
                   f"Batch Loss: {loss.item():.3f}"]
            for mk, mv in self.smoothed_metrics.items():
                if (not self.metrics_to_print) or (mk in self.metrics_to_print):
                    mv.synchronize_between_processes()
                    msg.append(f"{mk}: {mv}")
            self.info(self.delimiter.join(msg))

        self.step_end_time = time()
