import functools
import operator
import torch
import random
import socket

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