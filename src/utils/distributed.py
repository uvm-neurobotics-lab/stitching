"""
Utilities for distributed training.
"""
import logging
import os

import torch
import torch.distributed as dist


def setup_distributed_printing(is_master):
    """
    This function disables printing when not in master process
    """

    # Override the print() function.
    import builtins as __builtin__

    builtin_print = __builtin__.print

    # noinspection PyShadowingBuiltins
    def print(*args, **kwargs):
        force = kwargs.pop("force", False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print

    # Override python logging.
    if not is_master:
        for h in logging.root.handlers:
            h.flush()
        logging.root.setLevel(logging.WARN)


def init_distributed_mode(config):
    if "WORLD_SIZE" in os.environ:
        # Distributed mode via torchrun.
        config["rank"] = int(os.environ["RANK"])
        config["world_size"] = int(os.environ["WORLD_SIZE"])
        config["device"] = int(os.environ["LOCAL_RANK"])
    else:
        # Non-distributed.
        logging.info("Not using distributed mode.")
        config["distributed"] = False
        return

    if not torch.cuda.is_available():
        raise RuntimeError(f"Distributed training was requested, but no GPU was found.")

    config["distributed"] = True

    torch.cuda.set_device(config["device"])
    config["dist_backend"] = "nccl"
    logging.info(f"Distributed init: rank {config['rank']}, GPU {config['device']}")
    torch.distributed.init_process_group(backend=config["dist_backend"], world_size=config["world_size"],
                                         rank=config["rank"], device_id=torch.device(config["device"]))
    torch.distributed.barrier()
    setup_distributed_printing(config["rank"] == 0)


def tear_down_distributed_mode():
    if is_dist_avail_and_initialized():
        dist.destroy_process_group()


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    return get_rank() == 0


def save_on_master(*args, **kwargs):
    if is_main_process():
        torch.save(*args, **kwargs)


def reduce_across_processes(val, op=dist.ReduceOp.SUM):
    if not is_dist_avail_and_initialized():
        # nothing to sync, but we still convert to tensor for consistency with the distributed case.
        return torch.tensor(val)

    t = torch.tensor(val, device="cuda")
    dist.barrier()
    dist.all_reduce(t, op)
    return t
