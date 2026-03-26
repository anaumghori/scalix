import datetime
import os
from functools import lru_cache
from typing import Optional
import torch
from torch import distributed as dist
from torch.distributed.distributed_c10d import ProcessGroup

from scalix.utils import find_free_port

default_pg_timeout = datetime.timedelta(minutes=20)  # Timeout for process group initialization

def is_initialized() -> bool:
    return dist.is_initialized()

def barrier(group: Optional[ProcessGroup] = None):
    return dist.barrier(group=group)

def get_backend(group: Optional[ProcessGroup] = None) -> str:
    return dist.get_backend(group)

def destroy_process_group():
    return dist.destroy_process_group()


def new_group(
    ranks=None,
    timeout=default_pg_timeout,
    backend=None,
    pg_options=None,
) -> ProcessGroup:
    if not ranks:
        raise ValueError("Cannot create a group with no ranks inside it")

    return dist.new_group(
        ranks=ranks,
        timeout=timeout,
        backend=backend,
        pg_options=pg_options,
    )


@lru_cache
def get_rank(group: Optional[ProcessGroup] = None) -> int:
    """Similar to dist.get_rank but raises if the current process is not part of the group"""
    result = dist.get_rank(group)
    if result == -1:
        raise RuntimeError("Cannot call get_rank on a group in which current process is not a part")
    return result


def initialize_torch_distributed():
    """
    Initializes torch distributed using environment variables: RANK, WORLD_SIZE, LOCAL_RANK, MASTER_PORT
    """
    rank = int(os.getenv("RANK", "0"))
    world_size = int(os.getenv("WORLD_SIZE", "1"))
    local_rank = int(os.getenv("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    backend = "nccl"
    port = os.getenv("MASTER_PORT")
    if port is None:
        port = find_free_port()
    else:
        port = int(port)

    init_method = f"env://localhost:{port}"
    dist.init_process_group(
        backend=backend,
        init_method=init_method,
        world_size=world_size,
        rank=rank,
        timeout=default_pg_timeout,
    )

    return True