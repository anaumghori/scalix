import datetime
import os
from functools import cache, lru_cache
from typing import List, Optional, Tuple
import torch
from torch import distributed as dist
from torch.distributed.distributed_c10d import ProcessGroup
from packaging import version

from scalix.utils import find_free_port

default_pg_timeout = datetime.timedelta(minutes=20)  # Timeout for process group initialization

torch_version_above_1_13 = version.parse(torch.__version__) >= version.parse("1.13.0")
Work = dist.Work if torch_version_above_1_13 else dist._Work
ReduceOp = dist.ReduceOp

def is_initialized() -> bool:
    return dist.is_initialized()

def barrier(group: Optional[ProcessGroup] = None):
    return dist.barrier(group=group)

def get_backend(group: Optional[ProcessGroup] = None) -> str:
    return dist.get_backend(group)

def destroy_process_group():
    return dist.destroy_process_group()

def send(tensor, dst, group=None, tag=0):
    return dist.send(tensor, dst=dst, group=group, tag=tag)

def recv(tensor, src=None, group=None, tag=0):
    return dist.recv(tensor, src=src, group=group, tag=tag)

def isend(tensor, dst, group=None, tag=0):
    return dist.isend(tensor, dst, group=group, tag=tag)

def irecv(tensor, src, group=None, tag=0):
    return dist.irecv(tensor, src, group=group, tag=tag)

def P2POp(op, tensor, peer, group=None, tag=0):
    return dist.P2POp(op, tensor, peer, group=group, tag=tag)

def batch_isend_irecv(p2p_op_list):
    return dist.batch_isend_irecv(p2p_op_list)

def all_reduce(tensor, op=ReduceOp.SUM, group: Optional[ProcessGroup] = None, async_op: bool = False):
    return dist.all_reduce(tensor, op=op, group=group, async_op=async_op)

def all_gather_into_tensor(output_tensor, input_tensor, group: Optional[ProcessGroup] = None, async_op: bool = False):
    return dist.all_gather_into_tensor(output_tensor, input_tensor, group=group, async_op=async_op)

def reduce_scatter_tensor(
    output,
    input,
    op=ReduceOp.SUM,
    group: Optional[ProcessGroup] = None,
    async_op: bool = False,
):
    return dist.reduce_scatter_tensor(output, input, op=op, group=group, async_op=async_op)

def all_reduce_coalesced(tensors, op=ReduceOp.SUM, group: Optional[ProcessGroup] = None, async_op: bool = False):
    return dist.all_reduce_coalesced(tensors=tensors, op=op, group=group, async_op=async_op)

def reduce_scatter_coalesced(output_tensor_list, input_tensor_lists, op=ReduceOp.SUM, group: Optional[ProcessGroup] = None):
    return dist.reduce_scatter_coalesced(
        output_tensor_list=output_tensor_list,
        input_tensor_lists=input_tensor_lists,
        op=op,
        group=group,
    )

def all_gather_object(object_list, obj, group: Optional[ProcessGroup] = None):
    return dist.all_gather_object(object_list, obj, group=group)


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


@cache
def get_global_rank(group: ProcessGroup, group_rank: int) -> int:
    if torch_version_above_1_13:
        return dist.get_global_rank(group, group_rank=group_rank)
    else:
        # Support pytorch 1.12
        return dist.distributed_c10d._get_global_rank(group=group, rank=group_rank)


def get_global_ranks(group: ProcessGroup) -> Tuple[int]:
    return tuple(sorted((get_global_rank(group, i) for i in range(group.size()))))


def initialize_torch_distributed():
    """
    Initializes torch distributed from the standard environment variables used by
    torchrun and Slurm-backed launches.

    Expected variables:
    - RANK
    - WORLD_SIZE
    - LOCAL_RANK
    - MASTER_ADDR
    - MASTER_PORT

    For single-process local runs we fill in reasonable defaults so the same
    code path still works without a launcher.
    """
    if dist.is_initialized():
        return False

    rank = int(os.getenv("RANK", "0"))
    world_size = int(os.getenv("WORLD_SIZE", "1"))
    local_rank = int(os.getenv("LOCAL_RANK", "0"))

    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        backend = "nccl"
    else:
        backend = "gloo"

    # torch.distributed reads MASTER_ADDR / MASTER_PORT from the environment when
    # init_method="env://". Launchers such as torchrun and Slurm are expected to
    # populate them already for multi-process jobs.
    if world_size == 1:
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ.setdefault("MASTER_PORT", str(find_free_port()))
    else:
        if os.getenv("MASTER_ADDR") is None:
            raise RuntimeError("MASTER_ADDR must be set for multi-process distributed initialization")
        if os.getenv("MASTER_PORT") is None:
            raise RuntimeError("MASTER_PORT must be set for multi-process distributed initialization")

    dist.init_process_group(
        backend=backend,
        init_method="env://",
        world_size=world_size,
        rank=rank,
        timeout=default_pg_timeout,
    )

    return True
