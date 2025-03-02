"""
Utilities for distributed training.
"""
import os

import torch
import torch.distributed as dist


def setup_distributed_printing(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__

    builtin_print = __builtin__.print

    # noinspection PyShadowingBuiltins
    def print(*args, **kwargs):
        force = kwargs.pop("force", False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print


def init_distributed_mode(config):
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        config["rank"] = int(os.environ["RANK"])
        config["world_size"] = int(os.environ["WORLD_SIZE"])
        config["gpu"] = int(os.environ["LOCAL_RANK"])
    elif "SLURM_PROCID" in os.environ:
        config["rank"] = int(os.environ["SLURM_PROCID"])
        config["gpu"] = config["rank"] % torch.cuda.device_count()
    elif "rank" in config:
        pass
    else:
        print("Not using distributed mode.")
        config["distributed"] = False
        return

    config["distributed"] = True

    torch.cuda.set_device(config["gpu"])
    config["dist_backend"] = "nccl"
    print(f"| distributed init (rank {config['rank']}): {config['dist_url']}", flush=True)
    torch.distributed.init_process_group(backend=config["dist_backend"], init_method=config["dist_url"],
                                         world_size=config["world_size"], rank=config["rank"])
    torch.distributed.barrier()
    setup_distributed_printing(config["rank"] == 0)


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


def reduce_across_processes(val):
    if not is_dist_avail_and_initialized():
        # nothing to sync, but we still convert to tensor for consistency with the distributed case.
        return torch.tensor(val)

    t = torch.tensor(val, device="cuda")
    dist.barrier()
    dist.all_reduce(t)
    return t
