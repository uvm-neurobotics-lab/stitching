"""
Utilities for logging progress metrics and saving checkpoints.
"""
import datetime
import logging
import re
from collections import defaultdict, deque
from time import time, strftime, gmtime

import numpy as np
import torch

try:
    import wandb
except ImportError:
    wandb = None

from utils.distributed import reduce_across_processes


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

    def __enter__(self):
        super().__enter__()
        self.prev_train = self.module.training
        self.module.train(False)

    def __exit__(self, exc_type, exc_value, traceback):
        super().__exit__()
        self.module.train(self.prev_train)


def overall_metrics(model, data_loader, metric_fns, device, print_fn=None):
    """
    Evaluate the model on each batch and return the average over all samples for all given metric functions. Metric
    functions should follow the signature:
        (output: Tensor, labels: Tensor) -> dict[metric_name: str, metric_value: float]
    This returns a dict containing all metrics. If there are no batches, this returns an empty dictionary. If all
    batches are empty, this may return a dict populated with NaNs.
    """
    # Allow the tensors to be empty.
    if len(data_loader) == 0:
        return {}

    if print_fn:
        print_fn(f"Computing accuracy of {len(data_loader)} batches...")

    start = time()
    per_batch_metrics = defaultdict(list)
    with eval_mode(model):
        for ims, labels in data_loader:
            ims, labels = ims.to(device), labels.to(device)
            out = model(ims)
            for metric in metric_fns:
                md = metric(out, labels)
                for k, v in md.items():
                    per_batch_metrics[k].append((v, len(labels)))

    result = {}
    for k, v in per_batch_metrics.items():
        metric_per_batch = torch.Tensor(v)
        vals = metric_per_batch[:, 0]
        weights = metric_per_batch[:, 1]
        if weights.sum() == 0:
            result[k] = np.nan
        else:
            # Skip any batches where the value was NaN.
            idx = vals.isnan()
            result[k] = np.average(vals[~idx], weights=weights[~idx]).item()

    end = time()
    elapsed = end - start
    result["Eval Time"] = elapsed

    return result


class SmoothedValue:
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """

    def __init__(self, window_size=20, fmt=None):
        if fmt is None:
            fmt = "{median:.4f} ({global_avg:.4f})"
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt

    def update(self, value, n=1):
        self.deque.append(value)
        self.count += n
        self.total += value * n

    def synchronize_between_processes(self):
        """
        Warning: does not synchronize the deque!
        """
        t = reduce_across_processes([self.count, self.total])
        t = t.tolist()
        self.count = int(t[0])
        self.total = t[1]

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
        return self.total / self.count

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


class MetricLogger:
    def __init__(self, delimiter="\t"):
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v)

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{attr}'")

    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append(f"{name}: {str(meter)}")
        return self.delimiter.join(loss_str)

    def synchronize_between_processes(self):
        for meter in self.meters.values():
            meter.synchronize_between_processes()

    def add_meter(self, name, meter):
        self.meters[name] = meter

    def log_every(self, iterable, print_freq, header=None):
        i = 0
        if not header:
            header = ""
        start_time = time()
        end = time()
        iter_time = SmoothedValue(fmt="{avg:.4f}")
        data_time = SmoothedValue(fmt="{avg:.4f}")
        space_fmt = ":" + str(len(str(len(iterable)))) + "d"
        if torch.cuda.is_available():
            log_msg = self.delimiter.join(
                [
                    header,
                    "[{0" + space_fmt + "}/{1}]",
                    "eta: {eta}",
                    "{meters}",
                    "time: {time}",
                    "data: {data}",
                    "max mem: {memory:.0f}",
                ]
            )
        else:
            log_msg = self.delimiter.join(
                [header, "[{0" + space_fmt + "}/{1}]", "eta: {eta}", "{meters}", "time: {time}", "data: {data}"]
            )
        MB = 1024.0 * 1024.0
        for obj in iterable:
            data_time.update(time() - end)
            yield obj
            iter_time.update(time() - end)
            if i % print_freq == 0:
                eta_seconds = iter_time.global_avg * (len(iterable) - i)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                if torch.cuda.is_available():
                    print(
                        log_msg.format(
                            i,
                            len(iterable),
                            eta=eta_string,
                            meters=str(self),
                            time=str(iter_time),
                            data=str(data_time),
                            memory=torch.cuda.max_memory_allocated() / MB,
                        )
                    )
                else:
                    print(
                        log_msg.format(
                            i, len(iterable), eta=eta_string, meters=str(self), time=str(iter_time), data=str(data_time)
                        )
                    )
            i += 1
            end = time()
        total_time = time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print(f"{header} Total time: {total_time_str}")


class BaseLog:

    def __init__(self, metric_fns, metrics_to_print, eval_freq, save_freq, save_dir=None, model_name="",
                 use_wandb=False, checkpoint_initial_model=True, eval_full_train_set=False):
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
        self.use_wandb = use_wandb
        if use_wandb and not wandb:
            raise RuntimeError("You are trying to use Weights & Biases but it is not installed. You must either"
                               " install it or not request it.")
        self.checkpoint_initial_model = checkpoint_initial_model
        self.eval_full_train_set = eval_full_train_set
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self.recorded_metrics = {}
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

    def record(self, metrics, it):
        if it not in self.recorded_metrics:
            self.recorded_metrics[it] = metrics.copy()
        else:
            self.recorded_metrics[it].update(metrics)
        if self.use_wandb:
            # Turn the keys into strings for Weights and Biases.
            wmetrics = {k.get_name(): v for k, v in metrics.items()}
            wandb.log(wmetrics, step=it)

    def begin(self, model, train_loader, valid_loaders, device):
        self.start_time = time()
        if self.checkpoint_initial_model and (self.save_freq > 0 or self.eval_freq > 0):
            self.maybe_save_and_eval(0, 0, model, train_loader, valid_loaders, device)

    @torch.no_grad()
    def maybe_save_and_eval(self, it, epoch, model, train_loader, valid_loaders, device, should_eval=None,
                            should_save=None):
        # Turn on saving if it is time to save, or if the caller requires it.
        should_save = bool(should_save) or (self.save_freq > 0 and it % self.save_freq == 0)
        # Do not save if the model for this iteration was already saved (the same iteration can be called twice).
        should_save &= (it != self.last_save_step)
        # Eval if it is time to eval or save...
        should_eval = ((bool(should_eval) or bool(should_save) or it % self.eval_freq == 0)
                       and it != self.last_eval_step  # but not if we already did it,
                       and self.eval_freq > 0)  # and not if we don't want it at all.
        metrics = {"Epoch": epoch}

        # Run full test on training data.
        if should_eval:
            self.last_eval_step = it
            if self.eval_full_train_set:
                loaders = {"Train": train_loader, **{k.capitalize(): v for k, v in valid_loaders.items()}}
            else:
                loaders = {k.capitalize(): v for k, v in valid_loaders.items()}
            self.info(f"Checkpoint {it} Performance:")
            for key, loader in loaders.items():
                metric_dict = overall_metrics(model, loader, self.metric_fns, device, print_fn=self.debug)
                metric_str = ""
                for mk, mv in metric_dict.items():
                    metrics[f"Overall/{key} {mk}"] = mv
                    if (not self.metrics_to_print) or (mk in self.metrics_to_print):
                        metric_str += f"{mk} = {mv:.3f} | "
                if metric_str:
                    metric_str = metric_str[:-2]  # remove ending separator: "| "
                self.info(f"    {key} {metric_str}"
                          f"(Time to Eval = {strftime('%H:%M:%S', gmtime(metric_dict['Eval Time']))})")

        # Save the model.
        if should_save:
            self.last_save_step = it
            save_path = self.save_dir / f"{self.model_name}-{it}.pt"
            self.info(f"Saving model to: {save_path}")
            torch.save(model.state_dict(), save_path)

        # Keep track of total time elapsed during training.
        metrics["Cumulative Train Time"] = time() - self.start_time
        self.record(metrics, it)

    def close(self, it, epoch, model, train_loader, valid_loaders, device, should_eval=True, should_save=True):
        self.maybe_save_and_eval(it, epoch, model, train_loader, valid_loaders, device, should_eval, should_save)
        return self.recorded_metrics


class StandardLog(BaseLog):

    def __init__(self, model, metric_fns, metrics_to_print=tuple(), print_freq=50, save_freq=256, eval_freq=256,
                 save_dir=None, model_name="", use_wandb=False, checkpoint_initial_model=True,
                 eval_full_train_set=False):
        super().__init__(metric_fns, metrics_to_print, eval_freq, save_freq, save_dir, model_name, use_wandb,
                         checkpoint_initial_model, eval_full_train_set)
        self.epoch_start = -1
        self.start = -1
        self.print_freq = print_freq
        if use_wandb:
            wandb.watch(model, log_freq=print_freq)  # log gradient histograms automatically

    def begin_epoch(self, it, epoch, model, train_loader, valid_loaders, optimizer, device):
        self.record({"lr": optimizer.param_groups[0]["lr"]}, it)  # NOTE: assumes only one param group for now.
        self.info(f"---- Beginning Epoch {epoch} ({len(train_loader)} batches) ----")
        self.epoch_start = it
        self.start = time()

    def end_epoch(self, it, epoch, model, train_loader, valid_loaders, optimizer, device):
        self.info(f"---------------------- End Epoch {epoch} ----------------------")

        # If the eval function isn't going to recompute performance on train, then compile a summary here instead.
        if not self.eval_full_train_set:
            self.info("Summary of per-batch metrics:")
            # This is inefficient b/c recorded_metrics is being kept as a map, so we need to iterate through the whole
            # thing. If we ever have longer training times we may want to do it differently.
            sums = defaultdict(float)
            counts = defaultdict(int)
            for step, metrics in self.recorded_metrics.items():
                if step >= self.epoch_start:
                    for k, v in metrics.items():
                        if k in ("Epoch", "Train Time", "lr"):  # Don't accumulate these metrics.
                            continue
                        if k.startswith("Batch "):  # Turn "Batch" metrics into "Overall metrics.
                            k = k[6:]
                        sums[k] += v
                        counts[k] += 1
            for mname in sums:
                val = sums[mname] / counts[mname]
                self.record({"Overall/Train " + mname: val}, it)
                if (not self.metrics_to_print) or (mname in self.metrics_to_print):
                    self.info(f"    Train {mname} = {val:.3f}")

        self.maybe_save_and_eval(it, epoch, model, train_loader, valid_loaders, device)

    def step(self, it, epoch, loss, out, labels, model, all_losses=None, metric_fns=None):
        if not all_losses:
            all_losses = {}
        if not metric_fns:
            metric_fns = self.metric_fns
        metrics = {"Epoch": epoch, "Loss": loss.item(), **{k: v.item() for k, v in all_losses.items()}}
        metric_str = ""
        for metric in metric_fns:
            md = metric(out, labels)
            for mk, mv in md.items():
                metrics["Batch " + mk] = mv
                if (not self.metrics_to_print) or (mk in self.metrics_to_print):
                    metric_str += f"| {mk} = {mv:.3f} "
        if self.use_wandb:
            log_gradient_stats(metrics, model)

        time_to_print = (it % self.print_freq == 0 or it == 1)  # Force print on the first step.
        if time_to_print:
            # Track runtime since last printout.
            end = time()
            elapsed = end - self.start
            self.start = end
            metrics["Train Time"] = elapsed

            self.info(f"Step {it}: Batch Loss = {loss.item():.3f} {metric_str}"
                      f"({strftime('%H:%M:%S', gmtime(elapsed))})")

        self.record(metrics, it)
