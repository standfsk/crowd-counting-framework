from typing import Dict
import os
import random

import numpy as np
import torch
import torch.distributed as dist
from torch import Tensor


def is_dist_avail_and_initialized() -> bool:
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True

def get_world_size() -> int:
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()

def reduce_mean(tensor: Tensor, nprocs: int) -> Tensor:
    if not is_dist_avail_and_initialized():
        return tensor

    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= nprocs
    return rt

def reduce_dict(input_dict: Dict[str, Tensor], nprocs: int) -> dict:
    if nprocs < 2:
        return input_dict
    with torch.no_grad():
        names = []
        values = []
        for k in sorted(input_dict.keys()):
            names.append(k)
            values.append(input_dict[k])
        values = torch.stack(values, dim=0)
        reduce_mean(values, nprocs)
        reduced_dict = {k:v for k,v in zip(names, values)}
    return reduced_dict

def setup(local_rank: int, nprocs: int) -> None:
    if nprocs > 1:
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = "12366"
        dist.init_process_group("nccl", rank=local_rank, world_size=nprocs)
    else:
        print("Single process. No need to setup dist.")


def cleanup(ddp: bool = True) -> None:
    if ddp:
        dist.destroy_process_group()


def init_seeds(seed: int, cuda_deterministic: bool = False) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if cuda_deterministic:  # slower, but reproducible
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:  # faster, not reproducible
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True


def barrier(ddp: bool = True) -> None:
    if ddp:
        dist.barrier()