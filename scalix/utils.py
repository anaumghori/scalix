import functools
import operator
import torch
import random
import socket
from logging import Logger
from packaging import version
from typing import Optional

from scalix import distributed as dist

class Singleton(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super().__call__(*args, **kwargs)
        return cls._instances[cls]


# In training loops, certain temporary tensors (for example, all-gather buffers) are created every 
# forward pass and then discarded. Repeatedly calling torch.empty for large tensors causes: 
# allocator overhead, memory fragmentation and higher peak memory usage. This class avoids that by 
# keeping a persistent buffer and reusing it across iterations.

class MemoryBuffer(metaclass=Singleton):
    def __init__(self):
        # Keys are (name, dtype) pairs so that the same logical buffer name can be
        # independently maintained for different dtypes without collision.
        self.buffer = {}

    def get(self, name: str, shape: tuple, dtype: torch.dtype = torch.bfloat16) -> torch.Tensor:
        required_numel = functools.reduce(operator.mul, shape, 1)

        # Grow the backing allocation if it does not exist or is too small.
        # Over-allocating is intentional: the flat tensor is sliced to the exact
        # required size on return, so a larger buffer is always safe to reuse.
        if (name, dtype) not in self.buffer or self.buffer[name, dtype].numel() < required_numel:
            self.buffer[name, dtype] = torch.empty(
                required_numel, dtype=dtype, device=torch.cuda.current_device(), requires_grad=False
            )

        # Return a view of exactly the requested shape backed by the pooled allocation.
        return self.buffer[name, dtype][:required_numel].view(shape)


def get_untyped_storage(tensor: torch.Tensor) -> torch.UntypedStorage:
    if version.parse(torch.__version__) >= version.parse("2.0"):
        return tensor.untyped_storage()
    else:
        return tensor.storage().untyped()


def tensor_from_untyped_storage(untyped_storage: torch.UntypedStorage, dtype: torch.dtype):
    device = untyped_storage.device
    tensor = torch.empty([], dtype=dtype, device=device)
    tensor.set_(source=untyped_storage)
    return tensor


def find_free_port(min_port: int = 2000, max_port: int = 65000) -> int:
    while True:
        port = random.randint(min_port, max_port)
        try:
            with socket.socket() as sock:
                sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                sock.bind(("localhost", port))
                return port
        except OSError:
            continue


def log_rank(
    msg: str,
    logger: Logger,
    level: int,
    group: Optional[dist.ProcessGroup] = None,
    rank: Optional[int] = None,
    category: Optional[str] = None,
    is_separator: bool = False,
    main_rank_only: bool = False,
    **kwargs,
):
    if group is None:
        from torch import distributed as torch_dist
        group = torch_dist.distributed_c10d._get_default_group()

    if category is not None:
        kwargs["extra"] = kwargs.get("extra", {})
        kwargs["extra"]["category"] = category

    if is_separator:
        kwargs["extra"] = kwargs.get("extra", {})
        kwargs["extra"]["separator"] = True

    if main_rank_only:
        rank = 0

    if rank is None or dist.get_rank(group) == rank:
        if is_separator:
            logger.log(level, "=" * 50, **kwargs)
        logger.log(level, msg, **kwargs)
        if is_separator:
            logger.log(level, "=" * 50, **kwargs)